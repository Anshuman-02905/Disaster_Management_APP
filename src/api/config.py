
import os
import datetime

class Config:
    # MongoDB configuration
    MONGO_URI = "mongodb://localhost:27017/"
    DB_NAME = "Twitter"
    COLLECTION_NAME = "Posts"

    # Scraper configuration
    SCROLL_PAUSE_TIME = 8
    OUTPUT_PATH = "E:\\Projects\\DisasterClassification\\src\\temp"
    ALL_POSTS_XPATH = '/html/body/div[1]/div/div/div[2]/main/div/div/div/div[1]/div/div[3]/section/div/div'


    # Chrome WebDriver configuration
    CHROME_USER_DATA_DIR = r"user-data-dir=C:\Users\Asus\AppData\Local\Google\Chrome\User Data"
    CHROME_PROFILE_DIRECTORY = "Default"

      # Model paths
    NLP_MODEL_PATH = r'E:\Projects\DisasterClassification\src\models\TextClassification.pkl'
    NLP_VECTORIZER_PATH = r'E:\Projects\DisasterClassification\src\models\TextClasssification_vectorizer.pkl'
    IMAGE_MODEL_PATH = r'E:\Projects\DisasterClassification\src\models\ImageClassification.keras'

    # MongoDB configurations
    MONGO_URI = 'mongodb://localhost:27017/'
    DB_NAME = 'Twitter'
    PREDICTIONS_NAME='Predictions'

    # Image classification classes
    CLASSES = {
        -1: "NONE",
        0: 'Damaged_Infrastructure',
        1: 'Fire_Disaster',
        2: 'Human_Damage',
        3: 'Land_Disaster',
        4: 'Non_Damage',
        5: 'Water_Disaster'
    }

