from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from utils import TwitterScraper, TwitterPostCleaner, PostUploader
from model import Predictor,Downloaded
import traceback

# Initialize Flask app
app = Flask(__name__, template_folder='E:\\Projects\\DisasterClassification\\template')
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

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
        uploader.upload_posts()

        return jsonify({"message": f"Scraping, cleaning, and uploading {max_posts} posts with query {search_query} successful."}), 200
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
        predictions=Downloader.convert_toJasonify(predictions)

        return jsonify({"message": "Prediction and upload successful.", "predictions": predictions}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
