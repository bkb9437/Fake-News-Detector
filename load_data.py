import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_news_data(fake_path, real_path):
    """
    Load fake and real news data with proper validation and labeling
    """
    try:
        # Load datasets
        print(f"📁 Loading fake news from: {fake_path}")
        fake_df = pd.read_csv(fake_path)
        print(f"📁 Loading real news from: {real_path}")
        real_df = pd.read_csv(real_path)

        # Display dataset info
        print(f"Fake news dataset shape: {fake_df.shape}")
        print(f"Real news dataset shape: {real_df.shape}")
        print(f"Fake news columns: {list(fake_df.columns)}")
        print(f"Real news columns: {list(real_df.columns)}")

        # Check for required columns
        required_columns = ['title', 'text']
        for col in required_columns:
            if col not in fake_df.columns:
                print(f"⚠️  Warning: Column '{col}' not found in fake news dataset")
            if col not in real_df.columns:
                print(f"⚠️  Warning: Column '{col}' not found in real news dataset")

        # Handle missing columns by creating them if necessary
        if 'title' not in fake_df.columns:
            fake_df['title'] = ""
        if 'text' not in fake_df.columns:
            fake_df['text'] = fake_df.iloc[:, 0] if len(fake_df.columns) > 0 else ""

        if 'title' not in real_df.columns:
            real_df['title'] = ""
        if 'text' not in real_df.columns:
            real_df['text'] = real_df.iloc[:, 0] if len(real_df.columns) > 0 else ""

        # Clean the data
        fake_df['title'] = fake_df['title'].fillna("").astype(str)
        fake_df['text'] = fake_df['text'].fillna("").astype(str)
        real_df['title'] = real_df['title'].fillna("").astype(str)
        real_df['text'] = real_df['text'].fillna("").astype(str)

        # CRITICAL: Assign labels correctly
        # 0 = Fake, 1 = Real (this is standard binary classification)
        fake_df['label'] = 0  # Fake news
        real_df['label'] = 1  # Real news

        print(f"✅ Assigned labels: Fake=0 ({len(fake_df)} samples), Real=1 ({len(real_df)} samples)")

        # Combine datasets
        df = pd.concat([fake_df, real_df], ignore_index=True)

        # Shuffle the dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Create content column (combine title and text)
        df['content'] = df['title'].astype(str) + " " + df['text'].astype(str)

        # Remove entries with insufficient content
        df = df[df['content'].str.len() > 10]  # At least 10 characters

        # Final validation
        print(f"📊 Final dataset shape: {df.shape}")
        print(f"📊 Final label distribution:")
        print(df['label'].value_counts().sort_index())

        # Verify we have both classes
        unique_labels = df['label'].unique()
        if len(unique_labels) != 2:
            raise ValueError(f"Expected 2 classes (0,1), found: {unique_labels}")

        if 0 not in unique_labels or 1 not in unique_labels:
            raise ValueError(f"Missing class labels. Found: {unique_labels}")

        # Show sample from each class
        print(f"\n📄 Sample fake news (label=0):")
        fake_sample = df[df['label'] == 0].iloc[0]
        print(f"Title: {fake_sample['title'][:100]}...")

        print(f"\n📄 Sample real news (label=1):")
        real_sample = df[df['label'] == 1].iloc[0]
        print(f"Title: {real_sample['title'][:100]}...")

        return df

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def validate_dataset_format(df):
    """Validate the loaded dataset has correct format"""

    required_columns = ['content', 'label']
    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check label values
    unique_labels = sorted(df['label'].unique())
    if unique_labels != [0, 1]:
        raise ValueError(f"Labels should be [0, 1], found: {unique_labels}")

    # Check for empty content
    empty_content = df['content'].str.strip().eq('').sum()
    if empty_content > 0:
        print(f"⚠️  Warning: {empty_content} entries have empty content")

    print("✅ Dataset format validation passed")
    return True