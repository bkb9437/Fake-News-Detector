#!/usr/bin/env python3
"""
Quick test script to verify the fake news detector is working properly
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_predictor():
    """Test the predictor with known samples"""

    print("🧪 QUICK TEST: Fake News Detector")
    print("=" * 50)

    try:
        from predictor import predict_news

        # Test samples with expected results
        test_cases = [
            {
                'text': "Apple Inc. reported quarterly revenue of $81.8 billion for Q3 2023, beating analyst expectations. The company's iPhone sales grew 15% compared to the same period last year, driven by strong demand in emerging markets.",
                'expected': 'Real',
                'description': 'Corporate earnings (should be Real)'
            },
            {
                'text': "BREAKING: Scientists discover that drinking lemon water every morning can cure diabetes completely within 30 days! Big Pharma doesn't want you to know this simple trick that doctors hate.",
                'expected': 'Fake',
                'description': 'Health misinformation (should be Fake)'
            },
            {
                'text': "The Federal Reserve announced today it will maintain the federal funds rate at 5.25-5.5% for the third consecutive meeting. Fed Chair Jerome Powell cited ongoing inflation concerns in the decision.",
                'expected': 'Real',
                'description': 'Economic policy news (should be Real)'
            },
            {
                'text': "SHOCKING: Aliens have been living among us for decades and the government has been covering it up! Leaked documents reveal secret alien bases in Nevada. Share this before they delete it!",
                'expected': 'Fake',
                'description': 'Conspiracy theory (should be Fake)'
            }
        ]

        correct_predictions = 0
        total_tests = len(test_cases)

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🧪 Test {i}: {test_case['description']}")
            print("-" * 60)

            try:
                prediction, confidence = predict_news(test_case['text'])
                expected = test_case['expected']

                # Check if prediction matches expectation
                is_correct = prediction == expected
                if is_correct:
                    correct_predictions += 1
                    status = "✅ CORRECT"
                else:
                    status = "❌ INCORRECT"

                print(f"Text: {test_case['text'][:100]}...")
                print(f"Expected: {expected}")
                print(f"Predicted: {prediction} ({confidence}% confidence)")
                print(f"Result: {status}")

            except Exception as e:
                print(f"❌ Error during prediction: {e}")

        # Summary
        print(f"\n" + "=" * 60)
        print(f"📊 TEST SUMMARY")
        print(f"Correct predictions: {correct_predictions}/{total_tests}")
        print(f"Accuracy: {correct_predictions / total_tests * 100:.1f}%")

        if correct_predictions == total_tests:
            print("🎉 All tests passed! Your model is working correctly.")
        elif correct_predictions >= total_tests * 0.75:
            print("⚠️  Most tests passed, but there might be room for improvement.")
        else:
            print("🚨 Many tests failed. Your model needs attention.")
            print("\n💡 Possible issues:")
            print("- Model training data imbalance")
            print("- Incorrect label assignment (0=Fake, 1=Real)")
            print("- Overfitting to training data")
            print("- Text preprocessing issues")

        return correct_predictions / total_tests

    except ImportError as e:
        print(f"❌ Could not import predictor: {e}")
        print("Make sure you have trained the model first using train_model.py")
        return 0
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 0


def test_preprocessing():
    """Test the text preprocessing function"""

    print(f"\n🔧 TESTING TEXT PREPROCESSING")
    print("-" * 40)

    try:
        from preprocess import preprocess_text

        test_texts = [
            "This is a NORMAL sentence with punctuation!",
            "Visit https://example.com for more info",
            "Contact us at test@email.com",
            "   Extra    spaces   and   tabs   ",
            "123 Numbers and $pecial ch@racters!!!"
        ]

        for text in test_texts:
            processed = preprocess_text(text)
            print(f"Original:  '{text}'")
            print(f"Processed: '{processed}'")
            print()

    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")


def check_model_files():
    """Check if required model files exist"""

    print(f"\n📁 CHECKING MODEL FILES")
    print("-" * 30)

    required_files = [
        "model/news_model.pkl",
        "model/tfidf_vectorizer.pkl"
    ]

    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({file_size:,} bytes)")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            all_exist = False

    if not all_exist:
        print(f"\n⚠️  Missing model files. Run 'python train_model.py' first.")

    return all_exist


if __name__ == "__main__":
    print("🚀 Starting comprehensive test...")

    # Check model files
    files_exist = check_model_files()

    if not files_exist:
        print("❌ Cannot proceed without model files.")
        sys.exit(1)

    # Test preprocessing
    test_preprocessing()

    # Test predictions
    accuracy = test_predictor()

    print(f"\n🎯 FINAL RESULT")
    print("=" * 20)
    if accuracy >= 0.75:
        print("✅ Your fake news detector is working well!")
    else:
        print("🔧 Your model needs improvement. Consider:")
        print("1. Running the debug script: python debug_model.py")
        print("2. Retraining with balanced data")
        print("3. Checking your dataset format")

    print(f"\n💡 Next steps:")
    print("- Run 'python app.py' to start the web interface")
    print("- Test with your own news samples")
    print("- Check the training metadata in model/training_metadata.json")