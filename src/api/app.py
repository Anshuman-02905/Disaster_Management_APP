from utils import TwitterScraper,TwitterPostCleaner
from pymongo import MongoClient
from utils import PostUploader 
from model import Predictor
def main():
    try:
        search_query = "%23disaster"  # Hashtag to search for
        scraper = TwitterScraper(search_query, max_posts=50, scroll_pause_time=2)
        scraper.get_data()
        cleaner = TwitterPostCleaner()  # Create an instance of the class

        cleaner.clean_posts()  # Start cleaning the posts
        uploader = PostUploader()
        uploader.upload_posts()# uploaded data to Mongo DB
        predictor = Predictor()
        predictions = predictor.run_predictions_on_scraped_data()
        predictor.upload_to_DB(predictions)
    except Exception as e:
        print(str(e))

if __name__ == "__main__":
    main()

