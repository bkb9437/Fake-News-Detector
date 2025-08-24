import joblib
import os
import logging
import re

logger = logging.getLogger(__name__)

class NewsPredictor:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self._load_models()

    def _load_models(self):
        """Load model and vectorizer"""
        try:
            if not os.path.exists("model/news_model.pkl"):
                raise FileNotFoundError("Model file not found")
            if not os.path.exists("model/tfidf_vectorizer.pkl"):
                raise FileNotFoundError("Vectorizer file not found")

            self.model = joblib.load("model/news_model.pkl")
            self.vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
            logger.info("Models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def _simple_preprocess(self, text):
        """Simple preprocessing that matches training"""
        if not text or not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = str(text).lower()
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def predict(self, text):
        """Make prediction"""
        if not text or not isinstance(text, str):
            raise ValueError("Invalid input")

        try:
            # Preprocess
            cleaned = self._simple_preprocess(text)
            if not cleaned or len(cleaned.strip()) < 3:
                return "Uncertain", 50.0

            # Vectorize
            vector = self.vectorizer.transform([cleaned])

            # Predict
            probabilities = self.model.predict_proba(vector)[0]

            # probabilities[0] = fake, probabilities[1] = real
            fake_prob = probabilities[0]
            real_prob = probabilities[1]

            prediction = "Real" if real_prob > fake_prob else "Fake"
            confidence = round(max(real_prob, fake_prob) * 100, 2)

            return prediction, confidence

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "Error", 50.0

# Global predictor instance
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = NewsPredictor()
    return _predictor

def predict_news(text):
    """Public interface for prediction"""
    predictor = get_predictor()
    return predictor.predict(text)
