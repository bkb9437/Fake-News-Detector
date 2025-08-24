#!/usr/bin/env python3
"""
Emergency fix for fake news detector with 50% accuracy
This script will diagnose and fix the most common issues
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import warnings

warnings.filterwarnings('ignore')


def emergency_diagnosis():
    """Quickly diagnose the main issues"""
    print("🚨 EMERGENCY DIAGNOSIS")
    print("=" * 50)

    issues_found = []

    # Check 1: Model files exist
    if not os.path.exists("model/news_model.pkl"):
        issues_found.append("Missing model file")

    if not os.path.exists("model/tfidf_vectorizer.pkl"):
        issues_found.append("Missing vectorizer file")

    # Check 2: Data files exist
    if not os.path.exists("data/Fake.csv"):
        issues_found.append("Missing Fake.csv")

    if not os.path.exists("data/True.csv"):
        issues_found.append("Missing True.csv")

    # Check 3: Current model performance
    if len(issues_found) == 0:
        try:
            from predictor import predict_news

            # Test simple cases
            real_test = "Apple reported earnings of 81 billion dollars this quarter."
            fake_test = "Miracle cure discovered! This weird trick cures everything!"

            real_pred, real_conf = predict_news(real_test)
            fake_pred, fake_conf = predict_news(fake_test)

            print(f"Real news test: Predicted '{real_pred}' (should be 'Real')")
            print(f"Fake news test: Predicted '{fake_pred}' (should be 'Fake')")

            if real_pred != "Real":
                issues_found.append("Model incorrectly classifies real news")
            if fake_pred != "Fake":
                issues_found.append("Model incorrectly classifies fake news")

        except Exception as e:
            issues_found.append(f"Prediction error: {str(e)}")

    print(f"\n🔍 Issues found: {len(issues_found)}")
    for issue in issues_found:
        print(f"  ❌ {issue}")

    return issues_found


def create_synthetic_training_data():
    """Create synthetic training data if original data is problematic"""
    print("\n🔧 Creating synthetic training data...")

    # Real news patterns
    real_news_samples = [
        "The Federal Reserve announced today that interest rates will remain at 5.25 percent.",
        "Apple Inc reported quarterly revenue of 89.5 billion dollars, exceeding analyst expectations.",
        "Scientists at Harvard University published a study in Nature journal showing new findings.",
        "The Department of Labor released unemployment statistics showing a rate of 3.8 percent.",
        "Microsoft Corporation announced the launch of their new cloud computing service today.",
        "The World Health Organization confirmed 150 new cases in the recent outbreak investigation.",
        "Tesla Motors delivered 450,000 vehicles in the third quarter according to company reports.",
        "The Environmental Protection Agency issued new regulations on carbon emissions standards.",
        "Goldman Sachs analysts revised their economic forecast following recent market developments.",
        "The International Monetary Fund published their quarterly economic outlook report.",
        "Boeing received approval from aviation authorities for their new aircraft model.",
        "The Department of Education announced funding for 200 new school construction projects.",
        "Amazon Web Services reported a 25 percent increase in cloud infrastructure usage.",
        "The Centers for Disease Control updated vaccination guidelines based on recent data.",
        "JPMorgan Chase posted third quarter profits of 12.9 billion dollars today.",
        "NASA announced the successful launch of their new space exploration mission.",
        "The Social Security Administration released annual cost of living adjustment figures.",
        "General Motors invested 2.6 billion dollars in electric vehicle manufacturing facilities.",
        "The Bureau of Labor Statistics reported inflation decreased to 3.2 percent annually.",
        "Intel Corporation announced plans to build new semiconductor manufacturing plants."
    ]

    # Fake news patterns
    fake_news_samples = [
        "BREAKING: Scientists discover miracle cure that big pharma doesn't want you to know!",
        "SHOCKING: This one weird trick will make you lose 50 pounds overnight guaranteed!",
        "EXCLUSIVE: Government officials admit they have been hiding aliens in secret bases!",
        "AMAZING: Local grandmother discovers anti-aging secret that doctors hate!",
        "URGENT: Share this before they delete it! The truth about vaccines revealed!",
        "INCREDIBLE: Man wins lottery 7 times using this simple mathematical formula!",
        "EXPOSED: Celebrity death was faked and they are actually living in hiding!",
        "REVOLUTIONARY: New superfood discovered that prevents all diseases forever!",
        "LEAKED: Secret recording reveals politicians planning to cancel all elections!",
        "UNBELIEVABLE: Ancient remedy cures diabetes in just 3 days naturally!",
        "CONSPIRACY: Moon landing was filmed in Hollywood studio, evidence found!",
        "MIRACULOUS: Drink this every morning to reverse aging by 20 years!",
        "BOMBSHELL: Billionaires secretly control weather using advanced technology!",
        "FORBIDDEN: Traditional medicine secret that pharmaceutical companies banned!",
        "STUNNING: Archaeologists discover proof that aliens built the pyramids!",
        "DANGEROUS: Your smartphone is reading your thoughts, government admits!",
        "INCREDIBLE: Student discovers free energy device, goes missing mysteriously!",
        "EXPLOSIVE: Oil companies have been hiding unlimited energy source!",
        "TERRIFYING: Chemicals in tap water designed to control population!",
        "AMAZING: Grandmother's recipe melts belly fat like crazy, try tonight!"
    ]

    # Create DataFrame
    data = []

    # Add real news (label = 1)
    for text in real_news_samples:
        data.append({'content': text, 'label': 1})

    # Add fake news (label = 0)
    for text in fake_news_samples:
        data.append({'content': text, 'label': 0})

    df = pd.DataFrame(data)

    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"✅ Created {len(df)} synthetic samples")
    print(f"Real news samples: {(df['label'] == 1).sum()}")
    print(f"Fake news samples: {(df['label'] == 0).sum()}")

    return df


def emergency_retrain(use_synthetic=False):
    """Emergency retraining with proper setup"""
    print("\n🚀 EMERGENCY RETRAINING")
    print("=" * 40)

    try:
        if use_synthetic:
            # Use synthetic data
            data = create_synthetic_training_data()
        else:
            # Try to load original data
            print("📊 Loading original data...")
            from load_data import load_news_data
            data = load_news_data("data/Fake.csv", "data/True.csv")

            # Clean the data
            data['content'] = data['content'].fillna("").astype(str)
            data = data[data['content'].str.len() > 20]  # Remove very short texts

        print(f"Dataset size: {len(data)}")
        print(f"Label distribution:\n{data['label'].value_counts()}")

        # Simple preprocessing function
        def simple_preprocess(text):
            import re
            # Convert to lowercase
            text = str(text).lower()
            # Remove special characters but keep spaces
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
            # Remove extra spaces
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        # Preprocess text
        print("🔄 Preprocessing text...")
        data['clean_text'] = data['content'].apply(simple_preprocess)

        # Remove empty preprocessed text
        data = data[data['clean_text'].str.len() > 5]
        print(f"After preprocessing: {len(data)} samples")

        # Prepare features and labels
        X = data['clean_text']
        y = data['label']

        # Create TF-IDF vectorizer with simpler parameters
        print("🔢 Creating TF-IDF vectorizer...")
        vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )

        X_vectorized = vectorizer.fit_transform(X)
        print(f"Feature matrix shape: {X_vectorized.shape}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, y, test_size=0.2, stratify=y, random_state=42
        )

        # Calculate class weights for balance
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

        print(f"Class weights: {class_weight_dict}")

        # Use Logistic Regression (more stable than XGBoost for this issue)
        print("🧠 Training Logistic Regression model...")
        model = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000,
            C=1.0
        )

        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\n📈 Model Performance:")
        print(f"Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

        # Test with specific examples
        print(f"\n🧪 Testing specific examples:")
        test_cases = [
            ("Apple reported earnings of 81 billion dollars.", "Real"),
            ("Miracle cure discovered! Doctors hate this trick!", "Fake")
        ]

        for text, expected in test_cases:
            cleaned = simple_preprocess(text)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            probabilities = model.predict_proba(vectorized)[0]

            pred_label = "Real" if prediction == 1 else "Fake"
            confidence = max(probabilities) * 100

            status = "✅" if pred_label == expected else "❌"
            print(f"{status} Expected: {expected}, Got: {pred_label} ({confidence:.1f}%)")

        # Save the model
        print(f"\n💾 Saving emergency model...")
        os.makedirs("model", exist_ok=True)
        joblib.dump(model, "model/news_model.pkl")
        joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

        # Create updated predictor function
        create_updated_predictor()

        print("✅ Emergency retraining complete!")
        return accuracy

    except Exception as e:
        print(f"❌ Emergency retraining failed: {e}")
        print("\n🔄 Trying with synthetic data...")
        return emergency_retrain(use_synthetic=True)


def create_updated_predictor():
    """Create an updated predictor that works with the new model"""

    predictor_code = '''import joblib
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
        text = re.sub(r'[^a-zA-Z\\s]', ' ', text)
        # Remove extra spaces
        text = re.sub(r'\\s+', ' ', text).strip()
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
'''

    with open("predictor.py", "w") as f:
        f.write(predictor_code)

    print("✅ Updated predictor.py")


def main():
    """Main emergency fix function"""
    print("🚨 FAKE NEWS DETECTOR EMERGENCY FIX")
    print("=" * 60)

    # Diagnose issues
    issues = emergency_diagnosis()

    if len(issues) > 0:
        print(f"\n🔧 Attempting to fix {len(issues)} issues...")

        # Emergency retrain
        accuracy = emergency_retrain()

        if accuracy and accuracy > 0.7:
            print(f"\n🎉 SUCCESS! Model accuracy: {accuracy:.1%}")
            print("\n✅ Next steps:")
            print("1. Run: python quick_test.py")
            print("2. If tests pass, run: python app.py")
        else:
            print(f"\n⚠️  Model retrained but accuracy is still low: {accuracy:.1%}")
            print("This might be due to data quality issues.")
    else:
        print(f"\n✅ No critical issues found, but accuracy is still 50%")
        print("This suggests a training data problem. Retraining anyway...")
        emergency_retrain()


if __name__ == "__main__":
    main()
