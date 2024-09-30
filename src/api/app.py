from utils import TwitterScraper,TwitterPostCleaner
from pymongo import MongoClient
from utils import PostUploader 
def main():
    try:
        search_query = "%23disaster"  # Hashtag to search for
        scraper = TwitterScraper(search_query, max_posts=50, scroll_pause_time=2)
        scraper.get_data()
        cleaner = TwitterPostCleaner()  # Create an instance of the class
        cleaner.clean_posts()  # Start cleaning the posts
        uploader = PostUploader()
        uploader.upload_posts()

    except Exception as e:
        print(str(e))

if __name__ == "__main__":
    main()

