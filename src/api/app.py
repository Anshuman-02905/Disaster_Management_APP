from flask import Flask, jsonify, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import io
import traceback
import os
from flask_cors import CORS
from utils import TwitterScraper, TwitterPostCleaner, PostUploader
from model import Predictor, Downloaded
import traceback
from config import Config

# Initialize Flask app
app = Flask(__name__, template_folder='E:\\Projects\\DisasterClassification\\template')
CORS(app)

# About Page Route
@app.route('/')
def home():
    return render_template('about.html')

# Image Classification Page Route
@app.route('/image_classification')
def image_classification_page():
    return render_template('image_classification.html')

# Text Classification Page Route
@app.route('/text_classification')
def text_classification_page():
    return render_template('text_classification.html')

# Live Scraper Page Route
@app.route('/live_scraper')
def live_scraper_page():
    return render_template('live_scraper.html')

# Route to start the scraper
@app.route('/start_scraper', methods=['POST'])
def start_scraper():
    try:
        # Start scraper with a default hashtag and post limit
        search_query = request.json.get('search_query', "%23disaster")  # Default to #disaster
        max_posts = request.json.get('max_posts', 50)

        # Scraping process
        scraper = TwitterScraper(search_query, max_posts=max_posts, scroll_pause_time=2)
        scraper.get_data()

        # Clean scraped posts
        cleaner = TwitterPostCleaner()
        cleaner.clean_posts()

        # Upload cleaned posts to MongoDB
        uploader = PostUploader()
        upload_stamp=uploader.upload_posts()

        return jsonify({"message": f"Scraping, cleaning, and uploading {max_posts} posts with query {search_query} successful with upload_stamp{upload_stamp}"}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Route to show predictions
@app.route('/show_prediction', methods=['POST'])
def show_predictions():
    try:
        predictor = Predictor()

        # Run predictions on the scraped data
        predictions = predictor.run_predictions_on_scraped_data()

        # Upload predictions to MongoDB
        predictor.upload_to_DB(predictions)

        Downloader = Downloaded()
        predictions = Downloader.fetch_posts_with_nlp_prediction()

        # Convert ObjectIds and date to strings (as needed)
        predictions = Downloader.convert_toJasonify(predictions)

        return jsonify({"message": "Prediction and upload successful.", "predictions": predictions}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

# Route for image classification (empty for now)
@app.route('/image_classification', methods=['POST'])
def image_classification():
    try:
        # Check if a file is part of the request
        if 'image' not in request.files:
            return jsonify({"error": "No image file part in the request."}), 400

        file = request.files['image']

        # If no file is selected
        if file.filename == '':
            return jsonify({"error": "No selected file."}), 400

        # Validate the file
        if file and allowed_file(file.filename):
            # Secure the filename
            filename = secure_filename(file.filename)
            # Optional: Save the file to the server
           
            # Alternatively: Open the image without saving, using in-memory handling
            img = Image.open(file.stream)

            predictor = Predictor()
            prediction=predictor.predict_image(img)
          

            return jsonify(prediction), 200

        return jsonify({"error": "Invalid file format."}), 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Route for text classification (empty for now)
@app.route('/text_classification', methods=['POST'])
def text_classification():
    # Placeholder for text classification logic
    try:
        # Get the text input from the request
        text = request.json.get('text', '')

        if not text:
            return jsonify({"error": "No text provided."}), 400

        # Perform text classification (replace with your model logic)
        predictor = Predictor()
        prediction=predictor.predict_nlp([str(text)])

        return jsonify({"prediction": prediction}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
