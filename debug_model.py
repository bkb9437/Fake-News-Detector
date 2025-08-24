import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import json
import os
from preprocess import preprocess_text


def debug_model():
    """Debug the trained model to identify issues"""

    print("🔍 DEBUGGING FAKE NEWS DETECTOR MODEL")
    print("=" * 50)

    # Check if model files exist
    model_path = "model/news_model.pkl"
    vectorizer_path = "model/tfidf_vectorizer.pkl"
    metadata_path = "model/training_metadata.json"

    if not os.path.exists(model_path):
        print("❌ Model file not found. Please run train_model.py first.")
        return

    if not os.path.exists(vectorizer_path):
        print("❌ Vectorizer file not found. Please run train_model.py first.")
        return

    # Load model and vectorizer
    print("📁 Loading model and vectorizer...")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # Check training metadata
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"📊 Training Accuracy: {metadata.get('test_accuracy', 'Unknown'):.2%}")
        print(f"📈 CV Mean Accuracy: {metadata.get('cv_mean_accuracy', 'Unknown'):.2%}")
        print(f"📚 Dataset Size: {metadata.get('dataset_size', 'Unknown')}")
        print(f"🔤 Vocabulary Size: {metadata.get('vocabulary_size', 'Unknown')}")

    # Test with simple samples
    print("\n🧪 TESTING WITH SAMPLE TEXTS")
    print("-" * 30)

    test_samples = [
        ("Real news sample",
         "The Federal Reserve announced today that it will keep interest rates unchanged at 5.25%. Chairman Powell cited ongoing inflation concerns in the decision."),
        ("Fake news sample",
         "BREAKING: Scientists discover aliens living in underground cities. Government has been hiding this for decades. Click here to learn the shocking truth!"),
        ("Simple real", "Apple reported quarterly earnings of $89 billion, beating analyst expectations."),
        ("Simple fake", "Miracle cure discovered! This one weird trick cures all diseases overnight!")
    ]

    for label, text in test_samples:
        try:
            # Preprocess
            cleaned = preprocess_text(text)
            print(f"\n📝 {label}:")
            print(f"Original: {text[:100]}...")
            print(f"Cleaned: {cleaned[:100]}...")

            if not cleaned or len(cleaned.strip()) == 0:
                print("⚠️  Preprocessing resulted in empty text!")
                continue

            # Vectorize
            vector = vectorizer.transform([cleaned])
            print(f"Vector shape: {vector.shape}")
            print(f"Vector nnz: {vector.nnz} (non-zero elements)")

            if vector.nnz == 0:
                print("⚠️  Vectorization resulted in zero vector!")
                continue

            # Predict
            probabilities = model.predict_proba(vector)[0]
            fake_prob = probabilities[0]
            real_prob = probabilities[1]

            print(f"Fake probability: {fake_prob:.4f}")
            print(f"Real probability: {real_prob:.4f}")

            prediction = "Real" if real_prob > 0.5 else "Fake"
            confidence = max(real_prob, fake_prob) * 100

            print(f"🎯 Prediction: {prediction} ({confidence:.1f}% confidence)")

        except Exception as e:
            print(f"❌ Error processing sample: {e}")

    # Check model's class distribution understanding
    print(f"\n🏷️  MODEL CLASSES")
    print("-" * 20)
    print(f"Classes: {model.classes_}")
    print(f"Class 0 (should be Fake): {model.classes_[0]}")
    print(f"Class 1 (should be Real): {model.classes_[1]}")

    # Feature analysis
    print(f"\n🔤 FEATURE ANALYSIS")
    print("-" * 20)
    feature_names = vectorizer.get_feature_names_out()
    print(f"Total features: {len(feature_names)}")
    print(f"Sample features: {feature_names[:10]}")

    # Check if model is biased
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        top_indices = np.argsort(importance)[-10:]
        print(f"\n🔝 Top 10 Important Features:")
        for idx in reversed(top_indices):
            print(f"  {feature_names[idx]}: {importance[idx]:.4f}")

    # Test with original training data if available
    print(f"\n📊 CHECKING TRAINING DATA LABELS")
    print("-" * 30)

    try:
        from load_data import load_news_data
        data = load_news_data("data/Fake.csv", "data/True.csv")

        print(f"Original data shape: {data.shape}")
        print(f"Label distribution:")
        print(data['label'].value_counts())
        print(f"Label 0 count: {(data['label'] == 0).sum()} (should be fake)")
        print(f"Label 1 count: {(data['label'] == 1).sum()} (should be real)")

        # Check a few samples from each class
        fake_sample = data[data['label'] == 0].iloc[0]
        real_sample = data[data['label'] == 1].iloc[0]

        print(f"\n📄 Sample fake news title: {fake_sample.get('title', 'N/A')}")
        print(f"📄 Sample real news title: {real_sample.get('title', 'N/A')}")

    except Exception as e:
        print(f"⚠️  Could not load training data: {e}")

    return model, vectorizer


def fix_model_bias():
    """Attempt to fix model bias by retraining with balanced data"""

    print(f"\n🔧 ATTEMPTING TO FIX MODEL BIAS")
    print("=" * 40)

    try:
        from load_data import load_news_data
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer
        from xgboost import XGBClassifier
        from sklearn.utils import class_weight

        # Load data
        print("📊 Loading training data...")
        data = load_news_data("data/Fake.csv", "data/True.csv")

        # Clean and preprocess
        data['content'] = data['content'].fillna("").astype(str)
        data['clean_text'] = data['content'].apply(lambda x: preprocess_text(str(x)))

        # Remove empty texts
        data = data[data['clean_text'].str.strip() != ""]

        print(f"Dataset size after cleaning: {len(data)}")
        print(f"Class distribution:\n{data['label'].value_counts()}")

        if len(data) < 100:
            print("❌ Insufficient data for retraining")
            return

        # Prepare features
        X = data['clean_text']
        y = data['label']

        # Use class weights to handle imbalance
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(y),
            y=y
        )
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

        print(f"Class weights: {class_weight_dict}")

        # Create new vectorizer with different parameters
        vectorizer = TfidfVectorizer(
            max_features=5000,  # Reduced to prevent overfitting
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            strip_accents='unicode',
            lowercase=True,
            sublinear_tf=True  # Apply sublinear tf scaling
        )

        X_vectorized = vectorizer.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, y, test_size=0.2, stratify=y, random_state=42
        )

        # Train new model with class weights
        print("🧠 Training new balanced model...")
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            n_estimators=50,  # Reduced to prevent overfitting
            max_depth=4,  # Reduced depth
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=class_weights[1] / class_weights[0],  # Handle imbalance
            random_state=42
        )

        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        print("\n📈 New model performance:")
        print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

        # Test on our samples
        print("\n🧪 Testing new model:")
        test_samples = [
            ("Real", "Apple reported quarterly earnings of $89 billion today."),
            ("Fake", "Miracle cure discovered! This weird trick cures everything!")
        ]

        for expected, text in test_samples:
            cleaned = preprocess_text(text)
            if cleaned:
                vector = vectorizer.transform([cleaned])
                probs = model.predict_proba(vector)[0]
                pred = "Real" if probs[1] > 0.5 else "Fake"
                conf = max(probs) * 100

                status = "✅" if pred == expected else "❌"
                print(f"{status} Expected: {expected}, Got: {pred} ({conf:.1f}%)")

        # Save the new model
        print("\n💾 Saving improved model...")
        os.makedirs("model", exist_ok=True)
        joblib.dump(model, "model/news_model.pkl")
        joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

        print("✅ Model retrained and saved!")

    except Exception as e:
        print(f"❌ Error during model retraining: {e}")


if __name__ == "__main__":
    # Debug the current model
    try:
        model, vectorizer = debug_model()

        # Ask user if they want to attempt a fix
        print(f"\n" + "=" * 60)
        response = input("🤔 Would you like to attempt retraining the model? (y/n): ")

        if response.lower() in ['y', 'yes']:
            fix_model_bias()
        else:
            print("💡 Manual fixes you can try:")
            print("1. Check that your data files have the correct format")
            print("2. Ensure fake news is labeled as 0 and real news as 1")
            print("3. Verify your dataset has balanced classes")
            print("4. Run train_model.py again with the improved version")

    except Exception as e:
        print(f"❌ Debug script failed: {e}")
        print("Please ensure you have run train_model.py first.")