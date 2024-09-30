
import os

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