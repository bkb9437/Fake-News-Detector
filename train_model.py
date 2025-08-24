import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
import nltk
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# NLTK resources
def download_nltk_resources():
    """Download required NLTK resources"""
    resources = ['stopwords', 'wordnet', 'omw-1.4']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            logger.info(f"Downloaded {resource}")
        except Exception as e:
            logger.warning(f"Could not download {resource}: {e}")


download_nltk_resources()

from load_data import load_news_data
from preprocess import preprocess_text


def validate_data_files(fake_path, real_path):
    """Validate that data files exist and are readable"""
    if not os.path.exists(fake_path):
        raise FileNotFoundError(f"Fake news dataset not found: {fake_path}")
    if not os.path.exists(real_path):
        raise FileNotFoundError(f"Real news dataset not found: {real_path}")

    logger.info(f"Data files validated: {fake_path}, {real_path}")


def safe_preprocess(text):
    """Safe preprocessing wrapper with error handling"""
    try:
        if pd.isna(text) or text is None:
            return ""
        return preprocess_text(str(text))
    except Exception as e:
        logger.warning(f"Error in preprocessing: {e}")
        return ""


def train_news_classifier():
    """Train the fake news classification model"""
    try:
        # Paths to data files
        fake_path = "data/Fake.csv"
        real_path = "data/True.csv"

        # Validate data files
        validate_data_files(fake_path, real_path)

        # Load dataset
        logger.info("📊 Loading dataset...")
        data = load_news_data(fake_path, real_path)
        logger.info(f"Loaded {len(data)} samples")
        logger.info(f"Class distribution:\n{data['label'].value_counts()}")

        # Handle missing/invalid content values
        initial_count = len(data)
        data['content'] = data['content'].fillna("").astype(str)

        # Remove completely empty content
        data = data[data['content'].str.strip() != ""]
        logger.info(f"Removed {initial_count - len(data)} empty entries")

        if len(data) == 0:
            raise ValueError("No valid data found after cleaning")

        # Apply preprocessing
        logger.info("🔄 Preprocessing text...")
        data['clean_text'] = data['content'].apply(safe_preprocess)

        # Remove entries where preprocessing resulted in empty strings
        before_cleaning = len(data)
        data = data[data['clean_text'].str.strip() != ""]
        after_cleaning = len(data)

        logger.info(
            f"Preprocessing complete. Removed {before_cleaning - after_cleaning} entries with insufficient content")
        logger.info(f"Final dataset size: {len(data)} samples")

        if len(data) < 100:
            raise ValueError("Insufficient data for training (need at least 100 samples)")

        # Features & labels
        X = data['clean_text']
        y = data['label']

        # Check class balance
        class_counts = y.value_counts()
        logger.info(f"Class distribution after cleaning:\n{class_counts}")

        if min(class_counts) < 50:
            logger.warning("Imbalanced dataset detected. Consider data augmentation.")

        # TF-IDF Vectorization
        logger.info("🔢 Vectorizing text...")
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95,  # Ignore terms that appear in more than 95% of documents
            strip_accents='unicode',
            lowercase=True
        )

        X_vectorized = vectorizer.fit_transform(X)
        logger.info(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
        logger.info(f"Feature matrix shape: {X_vectorized.shape}")

        # Train-Test Split with stratify
        logger.info("📊 Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")

        # Train Model with optimized parameters
        logger.info("🧠 Training XGBoost Classifier...")
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        # Cross-validation
        logger.info("🔄 Performing cross-validation...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Evaluate on test set
        logger.info("📈 Evaluating model...")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        logger.info(f"Test accuracy: {accuracy:.4f}")
        logger.info("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

        logger.info("\n🎯 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        # Feature importance
        feature_names = vectorizer.get_feature_names_out()
        feature_importance = model.feature_importances_
        top_features_idx = np.argsort(feature_importance)[-10:]

        logger.info("\n🔝 Top 10 Most Important Features:")
        for idx in reversed(top_features_idx):
            logger.info(f"{feature_names[idx]}: {feature_importance[idx]:.4f}")

        # Save Model + Vectorizer
        logger.info("💾 Saving model and vectorizer...")
        os.makedirs("model", exist_ok=True)

        model_path = "model/news_model.pkl"
        vectorizer_path = "model/tfidf_vectorizer.pkl"

        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)

        # Save training metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'dataset_size': len(data),
            'test_accuracy': float(accuracy),
            'cv_mean_accuracy': float(cv_scores.mean()),
            'cv_std_accuracy': float(cv_scores.std()),
            'vocabulary_size': len(vectorizer.vocabulary_),
            'model_params': model.get_params()
        }

        import json
        with open("model/training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("✅ Training complete!")
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Vectorizer saved to: {vectorizer_path}")
        logger.info(f"Metadata saved to: model/training_metadata.json")

        return model, vectorizer, accuracy

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


def test_model_predictions():
    """Test the trained model with sample predictions"""
    try:
        logger.info("🧪 Testing model with sample predictions...")

        # Load the trained model
        model = joblib.load("model/news_model.pkl")
        vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

        # Test samples
        test_samples = [
            "Scientists discover new planet in distant solar system with potential for life",
            "BREAKING: Aliens land in New York City, demand to speak with world leaders immediately",
            "Stock market closes higher as investors remain optimistic about economic recovery",
            "Miracle cure discovered: Drink this one weird ingredient to lose 50 pounds overnight"
        ]

        for i, sample in enumerate(test_samples, 1):
            cleaned = preprocess_text(sample)
            vector = vectorizer.transform([cleaned])
            prob = model.predict_proba(vector)[0][1]
            prediction = "Real" if prob > 0.5 else "Fake"
            confidence = round(max(prob, 1 - prob) * 100, 2)

            logger.info(f"Test {i}: {prediction} ({confidence}% confidence)")
            logger.info(f"Text: {sample[:80]}...")
            logger.info("-" * 50)

    except Exception as e:
        logger.error(f"Model testing failed: {str(e)}")


if __name__ == "__main__":
    try:
        # Train the model
        model, vectorizer, accuracy = train_news_classifier()

        # Test with sample predictions
        test_model_predictions()

        logger.info(f"🎉 Training completed successfully with {accuracy:.2%} accuracy!")

    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        exit(1)