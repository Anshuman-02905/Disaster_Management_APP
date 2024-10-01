from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from PIL import Image
import time
from pymongo import MongoClient
import gridfs
import os
import requests
from config import Config
from datetime import datetime


class TwitterScraper:
    def __init__(self, search_query, max_posts=50, scroll_pause_time=2):
        self.search_query = search_query
        self.max_posts = max_posts
        self.scroll_pause_time = Config.SCROLL_PAUSE_TIME
        self.collected_posts = dict()
        self.output_path =  Config.OUTPUT_PATH

        # Set up Chrome options
        self.chrome_options = Options()
        #self.chrome_options.add_argument(Config.CHROME_USER_DATA_DIR)
        self.chrome_options.add_argument( r"user-data-dir=C:\Users\Asus\AppData\Local\Google\Chrome\User Data")
        #self.chrome_options.add_argument(f"profile-directory={Config.CHROME_PROFILE_DIRECTORY}")
        self.chrome_options.add_argument(f"profile-directory=Default")

        # Set up the Chrome WebDriver using the WebDriver Manager
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=self.chrome_options)
        self.driver.get(f"https://x.com/search?q={self.search_query}&src=typed_query&f=live")

        # Allow the page to load
        time.sleep(5)

    def download_image(self, image_url, save_path):
        """Download image from a URL and save it."""
        try:
            img_data = requests.get(image_url).content
            with open(save_path, 'wb') as img_file:
                img_file.write(img_data)
            print(f"Image saved at: {save_path}")
        except Exception as e:
            print(f"Failed to download image from {image_url}. Error: {e}")

    def save_text(self, text, save_path):
        """Save the text content to a file."""
        try:
            with open(save_path, 'w', encoding='utf-8') as text_file:
                text_file.write(text)
            print(f"Text saved at: {save_path}")
        except Exception as e:
            print(f"Failed to save text. Error: {e}")

    def explore_div(self, element, post_num):
        """Recursively explore a div, its children, and check for images."""
        folder_path = os.path.join(self.output_path, f"Post_{post_num}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Extract and save the text content
        text_content = element.text.strip()
        if text_content:
            print(f"Text found for Post {post_num}: {text_content}")
            text_save_path = os.path.join(folder_path, "post_text.txt")
            self.save_text(text_content, text_save_path)

        # Find and download all images in the post
        images = element.find_elements(By.TAG_NAME, 'img')
        for idx, img in enumerate(images):
            src = img.get_attribute('src')
            if src:
                print(f"Image found for Post {post_num}: {src}")
                image_save_path = os.path.join(folder_path, f"image_{idx + 1}.jpg")
                self.download_image(src, image_save_path)

    def extract_posts(self):
        """Extract posts and images from the current page."""
        all_posts_xpath = Config.ALL_POSTS_XPATH
        posts = self.driver.find_elements(By.XPATH, all_posts_xpath + '/div')
        post_num = len(self.collected_posts.keys())

        for post in posts:
            if post.id not in self.collected_posts.keys():
                self.collected_posts[post.id] = post
                self.explore_div(post, post_num)
                post_num += 1

        print(f"Found {len(posts)} posts.")

    def get_data(self):
        """Scroll and extract posts until the target number of posts is reached."""
        try:
            while len(self.collected_posts.keys()) < self.max_posts:
                print(f"{len(self.collected_posts.keys())} / {self.max_posts} posts collected.")
                self.extract_posts()

                # Scroll down
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

                # Wait for new posts to load
                time.sleep(self.scroll_pause_time)
                print("SCROLLED")

                # Break the loop if no new posts are loaded
                if len(self.driver.find_elements(By.XPATH, Config.ALL_POSTS_XPATH)) == 0:
                    print("No new posts found, stopping scrolling.")
                    break

            print(f"Collected {len(self.collected_posts)} posts in total.")
            # Close the browser when done
            self.driver.quit()
        except Exception as e:
            print(e)
            return 

class TwitterPostCleaner:
    def __init__(self):
        self.base_path = Config.OUTPUT_PATH
        # List of valid image extensions
        self.valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']

    def is_valid_image(self, file_path):
        """Check if the image is valid and larger than 100x100 pixels."""
        try:
            with Image.open(file_path) as img:
                # Verify the image to check if it's complete and not corrupted
                img.verify()  # If this fails, it raises an exception

            # Reopen the image to check its size (verify() closes the file)
            with Image.open(file_path) as img_reopen:
                if img_reopen.width < 100 or img_reopen.height < 100:
                    print(f"Deleting {file_path} (Size: {img_reopen.size}) - Too small")
                    os.remove(file_path)  # Delete the file if it's too small
                    return False
                else:
                    print(f"Image {file_path} is valid (Size: {img_reopen.size})")
                    return True
        except (Image.UnidentifiedImageError, OSError) as er:
            print(f"Invalid image {file_path}. Error: {er}")
            os.remove(file_path)  # Delete the file if it's invalid or unreadable
            return False

    def clean_posts(self):
        """Clean all posts by checking images in each folder."""
        # Loop through the Post folders (e.g., Post_1, Post_2)
        for folder in os.listdir(self.base_path):
            folder_path = os.path.join(self.base_path, folder)

            # Only proceed if it is a folder and the folder name starts with 'Post'
            if os.path.isdir(folder_path) and folder.startswith('Post'):
                print(f"Processing folder: {folder}")
                # Loop through each file in the folder
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)

                    # Get the file extension and check if it's a valid image file
                    file_ext = os.path.splitext(file_name)[1].lower()
                    if file_ext not in self.valid_extensions:
                        print(f"Ignoring non-image file: {file_name}")
                        continue  # Skip non-image files like .txt

                    # Check if the image is valid and large enough
                    self.is_valid_image(file_path)
class PostData:
    def __init__(self, post_id='', text='', images=None, date=None):
        self.post_id = post_id  # ID extracted from the second line
        self.text = text
        self.images = images if images is not None else []
        self.date = date if date is not None else datetime.utcnow()  # Set to current date by default (UTC)

    def to_dict(self):
        return {
            "post_id": self.post_id,
            "text": self.text,
            "images": self.images,
            "date": self.date  # Include the date in the dictionary representation
        }

class PostUploader:
    def __init__(self):
        self.root_folder = Config.OUTPUT_PATH  # e.g., 'E:/Projects/DisasterClassification/src/temp'
        self.client = MongoClient(Config.MONGO_URI)
        self.db = self.client[Config.DB_NAME]
        self.collection = self.db[Config.COLLECTION_NAME]
        self.fs = gridfs.GridFS(self.db)  # To store images in MongoDB using GridFS if needed

    def create_post_object(self, post_folder):
        # Initialize a PostData object
        post_data = PostData()

        # Traverse through the post folder for text and images
        for file in os.listdir(post_folder):
            file_path = os.path.join(post_folder, file)
            if file.endswith('.txt'):
                # Read the text file and extract the second line as ID
                with open(file_path, 'r', encoding='utf-8') as txt_file:
                    lines = txt_file.readlines()
                    if len(lines) >= 2:
                        post_data.post_id = lines[1].strip()  # Extract the second line
                    post_data.text = "".join(lines)  # Optionally save all text content

            elif file.endswith(('.png', '.jpg', '.jpeg')):
                # Store image path or upload to GridFS
                with open(file_path, 'rb') as img_file:
                    image_id = self.fs.put(img_file)  # Upload to GridFS and get image_id
                    post_data.images.append(image_id)  # Append the image ID to the list

        return post_data

    def process_posts(self):
        posts = []

        # Iterate through each post folder (Post1, Post2, etc.)
        for post_folder in os.listdir(self.root_folder):
            full_path = os.path.join(self.root_folder, post_folder)
            if os.path.isdir(full_path):
                post_data = self.create_post_object(full_path)
                posts.append(post_data)

        return posts

    def upload_posts(self):
        # Get the list of posts
        posts = self.process_posts()

        # Upload each post to MongoDB
        for post in posts:
            self.collection.insert_one(post.to_dict())
            print(f"Uploaded post ID: {post.post_id} - Text: {post.text[:30]}... with {len(post.images)} images.")

    

