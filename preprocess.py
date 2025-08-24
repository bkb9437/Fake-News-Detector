import re
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configure logging
logger = logging.getLogger(__name__)


# Download required NLTK data (with error handling)
def download_nltk_data():
    """Download required NLTK data with error handling"""
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)  # Additional data for lemmatizer
        except Exception as e:
            logger.warning(f"Could not download NLTK data: {e}")


# Initialize NLTK data
download_nltk_data()

# Initialize components with error handling
try:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except Exception as e:
    logger.warning(f"Error initializing NLTK components: {e}")
    # Fallback to basic stop words
    stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
    lemmatizer = None


def preprocess_text(text):
    """
    Preprocess text with comprehensive error handling and validation
    """
    try:
        # Input validation
        if not text or not isinstance(text, str):
            return ""

        # Basic cleaning
        text = str(text).strip()
        if not text:
            return ""

        # Remove HTML tags if any
        text = re.sub(r'<[^>]+>', ' ', text)

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' ', text)

        # Remove excessive whitespace and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        # Convert to lowercase and tokenize
        tokens = re.findall(r'\b\w+\b', text.lower())

        if not tokens:
            return ""

        # Filter tokens
        cleaned_tokens = []
        for token in tokens:
            # Skip very short tokens, numbers, and stop words
            if (len(token) > 2 and
                    not token.isdigit() and
                    token not in stop_words and
                    token.isalpha()):  # Only alphabetic tokens

                # Apply lemmatization if available
                if lemmatizer:
                    try:
                        token = lemmatizer.lemmatize(token)
                    except Exception as e:
                        logger.debug(f"Lemmatization failed for token '{token}': {e}")
                        # Continue without lemmatization

                cleaned_tokens.append(token)

        # Join tokens
        result = " ".join(cleaned_tokens)

        # Final validation
        if len(result.strip()) == 0:
            logger.debug("Preprocessing resulted in empty string")
            return ""

        return result

    except Exception as e:
        logger.error(f"Error in preprocess_text: {e}")
        # Return empty string on any error
        return ""


def validate_processed_text(text):
    """Validate that processed text is suitable for analysis"""
    if not text or len(text.strip()) < 5:
        return False, "Processed text too short"

    words = text.split()
    if len(words) < 3:
        return False, "Not enough meaningful words"

    # Check if text has reasonable diversity (not just repeated words)
    unique_words = set(words)
    if len(unique_words) < len(words) * 0.3:  # Less than 30% unique words
        return False, "Text lacks diversity"

    return True, "Text is valid"