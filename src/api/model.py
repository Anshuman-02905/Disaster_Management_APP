import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import jsonify
from tensorflow.keras.models import load_model  # Use tensorflow.keras for compatibility
from pymongo import MongoClient
from PIL import Image
import numpy as np
from bson import ObjectId
import gridfs
from io import BytesIO
import json
from config import Config
from datetime import datetime
import os
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
import re



class PostData:
    def __init__(self, post_id='', text='', images=None, nlp_prediction=None, image_prediction=None, date=None,upload_stamp=None):
        self.post_id = post_id
        self.text = text
        self.images = images if images is not None else []
        self.nlp_prediction = nlp_prediction
        self.image_prediction = image_prediction
        self.date = date if date is not None else datetime.utcnow()  # Set the current date by default
        self.upload_stamp=upload_stamp

    def to_dict(self):
        return {
            "post_id": self.post_id,
            "text": self.text,
            "images": self.images,
            "nlp_prediction": self.nlp_prediction,
            "image_prediction": self.image_prediction,
            "date": self.date,
            "upload_stamp":self.upload_stamp
        }

class Predictor:
    def __init__(self):
        # Load the NLP model and vectorizer
        self.nlp_model = self.load_nlp_model(Config.NLP_MODEL_PATH)
        self.nlp_vectorizer = self.load_vectorizer(Config.NLP_VECTORIZER_PATH)
        self.classes = Config.CLASSES
        
        # Load the image classification model
        self.image_model = self.load_image_model(Config.IMAGE_MODEL_PATH)
        
        # MongoDB setup
        self.client = MongoClient(Config.MONGO_URI)
        self.db = self.client[Config.DB_NAME]
        self.collection = self.db[Config.COLLECTION_NAME]
        self.fs = gridfs.GridFS(self.db)   # GridFS for image retrieval
        self.prediction = self.db[Config.PREDICTIONS_NAME]

        self.lemmatizer = WordNetLemmatizer()



    def load_nlp_model(self, model_path):
        with open(model_path, 'rb') as model_file:
            return pickle.load(model_file)

    def load_vectorizer(self, vectorizer_path):
        with open(vectorizer_path, 'rb') as vec_file:
            return pickle.load(vec_file)

    def load_image_model(self, model_path):
        return load_model(model_path)

    def retrieve_posts_from_mongo(self):
        # Retrieve all posts from MongoDB
        latest_stamp = self.collection.find_one(sort=[("upload_stamp", -1)])["upload_stamp"]
        print(latest_stamp)
        posts_with_latest_stamp = self.collection.find({"upload_stamp": latest_stamp})
        return [PostData(post_id=str(post['post_id']), text=post['text'], images=post.get('images', []),upload_stamp=post.get('upload_stamp')) for post in posts_with_latest_stamp]

    def fetch_image_by_id(self, image_id):
        try:
            # Fetch the image binary data from GridFS
            grid_out = self.fs.get(image_id)
            image_data = grid_out.read()
            
            # Convert binary to an image (PIL)
            img = Image.open(BytesIO(image_data))
            return img
        except Exception as e:
            print(f"Error fetching image with ID {image_id}: {e}")
            return None

    def preprocess_image(self, img):
        try:
            img = img.resize((150, 150)) #Adjust the image size
            img = np.array(img) / 255.0  # Normalize the image
            if img.shape[-1] == 4:  # If image has alpha channel (RGBA), convert to 3 channels (RGB)
                img = img[..., :3]
            if img.shape != (150, 150, 3):
                print(f"Skipping image due to incorrect shape: {img.shape}")
                return None
            return np.expand_dims(img, axis=0)  # Add batch dimension
        except Exception as e:
            print(f"Error processing image: {e}")
            return None

    def predict_image(self, img):
        try:
            image_prediction = {}
            if img is not None:
                preprocessed_img = self.preprocess_image(img)
                if preprocessed_img is not None:
                    # Get the prediction as a NumPy array
                    pred = self.image_model.predict(preprocessed_img).tolist()[0]  # Convert to list
                    
                    # Get the predicted class (index of the highest probability)
                    predicted_class_index = int(np.argmax(pred))  # Convert to int
                    
                    # Get the class label from the classes dictionary
                    predicted_class_label = self.classes.get(predicted_class_index, "UNKNOWN")

                    probabilities_dict = {Config.CLASSES[i]: float(prob) for i, prob in enumerate(pred)}# Convert probabilities to float

                    
                    image_prediction={
                       "probabilities": probabilities_dict,  
                        "predicted_class_index": predicted_class_index,
                        "predicted_class_label": predicted_class_label  # Add the label
                    }
                else:
                    image_prediction={
                        "probabilities": {'UNKNOWN':-1.0},  # Use float for consistency
                        "predicted_class_index": -1,
                        "predicted_class_label": "NONE"  # Assign label for no image case
                    }
            else:
                image_prediction={
                    "probabilities": [{'UNKNOWN':-1.0}],  # Use float for consistency
                    "predicted_class_index": -1,
                    "predicted_class_label": "NONE"  # Assign label for missing image case
                }
            return image_prediction
        except Exception as e:
            print(str(e))
            return []

    def preprocess_text(self,text):
        #Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
        #Remove special characters, punctuation, and numbers
        text = re.sub(r'[^A-Za-z\s]', '', text)
        #Convert text to lowercase
        text = text.lower()
        #Remove stopwords
        stop_words = set(stopwords.words('english')) 
        pronouns_to_keep={'he', 'she', 'they','our', 'it', 'i' 'we', 'you', 'him', 'her', 'them', 'us', 'me'}
        #Remove pronouns from stop words
        stop_words -= pronouns_to_keep
        text_tokens = text.split()  # Tokenize text
        text_tokens = [word for word in text_tokens if word not in stop_words]
        #Lemmatization (convert words to base form)
        text_tokens = [self.lemmatizer.lemmatize(word) for word in text_tokens]
        #Join tokens back into a single string
        cleaned_text = ' '.join(text_tokens)
        return cleaned_text

    def predict_nlp(self, new_texts):
        new_texts=[self.preprocess_text(new_texts[0])]
        new_text_tfidf = self.nlp_vectorizer.transform(new_texts)
        predictions = list(self.nlp_model.predict(new_text_tfidf))
        labeled_predictions = [Config.NLP_DICT[pred] for pred in predictions]

        return labeled_predictions

    def run_predictions_on_scraped_data(self):
        # Retrieve posts from MongoDB
        posts = self.retrieve_posts_from_mongo()

        # Process each post individually
        for post in posts:
            # Make NLP prediction
            nlp_prediction = self.predict_nlp([post.text])  # Assuming the model outputs an array

            # Make image prediction (if there are images)
            image_predictions = []
            if post.images:  # Assuming images contains ObjectId(s) of the images in MongoDB
                image_ids = [image_id for image_id in post.images]
                for ids in image_ids:
                    img = self.fetch_image_by_id(ids)
                    image_predictions.append(self.predict_image(img))

            # Update post with predictions
            post.nlp_prediction = nlp_prediction
            post.image_prediction = image_predictions if image_predictions else "No Image"

        return [post.to_dict() for post in posts]
    
    def upload_to_DB(self,predictions):

        for prediction in predictions:
            prediction['post_id'] = prediction['post_id']
            
            # Convert each image id to ObjectId
            prediction['images'] = [ObjectId(image_id) for image_id in prediction['images']]
            
            # Add current date if not present
            if 'date' not in prediction:
                prediction['date'] = datetime.utcnow()

            # Insert the prediction into MongoDB
            for key, val in prediction.items():
                print(f"Key: {key}, Value: {val}, Val Type: {type(val)}")
                
            self.prediction.insert_one(prediction)

class Downloaded:
    def __init__(self):
        # MongoDB setup
        self.client = MongoClient(Config.MONGO_URI)
        self.db = self.client[Config.DB_NAME]
        self.collection = self.db[Config.PREDICTIONS_NAME]
        self.fs = gridfs.GridFS(self.db)  # GridFS for image retrieval

    def fetch_posts_with_nlp_prediction(self, prediction_value=1):

        try:
            # Querying the collection for posts where 'nlp_prediction' equals the given value
            latest_stamp = self.collection.find_one(sort=[("upload_stamp", -1)])["upload_stamp"]

            query = {"upload_stamp":latest_stamp,
                     "nlp_prediction": ['Disaster']}
            
            posts = self.collection.find(filter=query)
            
            # Convert the posts to a list of dictionaries
            result = [post for post in posts]
            result=self.convert_toJasonify(result)
            return result
    

        except Exception as e:
            print(f"Error fetching posts with nlp_prediction = {prediction_value}: {str(e)}")
            return []
        
    def convert_toJasonify(self, posts):
        for post in posts:
            # Convert '_id' to string
            if isinstance(post.get('_id'), ObjectId):
                post['_id'] = str(post['_id'])

            # Convert 'post_id' to string if it's an ObjectId
            if isinstance(post.get('post_id'), ObjectId):
                post['post_id'] = str(post['post_id'])

            # Convert 'images' list from ObjectId to string
            if 'images' in post:
                post['images'] = [str(image_id) if isinstance(image_id, ObjectId) else image_id for image_id in post['images']]
                
            if isinstance(post.get('date'), datetime):
                post['date'] = post['date'].strftime('%Y-%m-%d %H:%M:%S')
                
            
        
        return posts

        

def main():
    try:
        Downloader = Downloaded()
        predictions = Downloader.fetch_posts_with_nlp_prediction()
        json_file_path = os.path.join('E:\\Projects\\DisasterClassification\\src', 'predictions.json')  # Adjust path as needed
        with open(json_file_path, 'w') as json_file:
            json.dump(predictions, json_file, indent=4)
        
    except Exception as e:
        print(str(e))

if __name__ == "__main__":
    main()


    