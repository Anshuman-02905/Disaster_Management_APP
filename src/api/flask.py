from flask import Flask, jsonify, request
from utils import TwitterScraper, TwitterPostCleaner, PostUploader
from model import Predictor

app = Flask(__name__)

@app.route('/start_scraper', methods=['POST'])
def start_scraper():
    try:
        search_query = request.json.get('search_query', '%23disaster')  # Default to #disaster
        scraper = TwitterScraper(search_query, max_posts=50, scroll_pause_time=2)
        scraper.get_data()
        
        cleaner = TwitterPostCleaner()
        cleaner.clean_posts()
        
        uploader = PostUploader()
        uploader.upload_posts()  # Upload data to MongoDB
        
        predictor = Predictor()
        predictions = predictor.run_predictions_on_scraped_data()
        
        return jsonify({"status": "success", "predictions": predictions}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/')
def index():
    return "Welcome to the Disaster Management App!"

if __name__ == "__main__":
    app.run(debug=True)
