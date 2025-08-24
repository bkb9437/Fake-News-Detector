from flask import Flask, render_template, request, jsonify, flash
import os
import logging
from predictor import predict_news

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    error_message = None

    if request.method == "POST":
        try:
            user_input = request.form.get("news", "").strip()

            # Input validation
            if not user_input:
                error_message = "Please enter some news text to analyze."
            elif len(user_input) < 10:
                error_message = "Please enter at least 10 characters for meaningful analysis."
            elif len(user_input) > 10000:
                error_message = "Text is too long. Please limit to 10,000 characters."
            else:
                # Make prediction
                prediction, confidence = predict_news(user_input)
                logger.info(f"Prediction made: {prediction} with {confidence}% confidence")

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            error_message = "An error occurred while analyzing the text. Please try again."

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           error_message=error_message)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for programmatic access"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400

        prediction, confidence = predict_news(text)

        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'status': 'success'
        })

    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({'error': 'Analysis failed'}), 500


@app.errorhandler(404)
def not_found(error):
    return render_template("error.html", error="Page not found"), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template("error.html", error="Internal server error"), 500


if __name__ == "__main__":
    # Check if model files exist
    if not os.path.exists("model/news_model.pkl") or not os.path.exists("model/tfidf_vectorizer.pkl"):
        logger.warning("Model files not found. Please run train_model.py first.")

    app.run(debug=True, host='0.0.0.0', port=5000)