import re, string
from bs4 import BeautifulSoup
import contractions
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Make sure necessary NLTK data is available
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# -----------------------------
# Negation Handling Function
# -----------------------------
def handle_negation(tokens):
    result = []
    i = 0
    
    while i < len(tokens):
        if tokens[i] == "not" and i + 1 < len(tokens):
            # Combine: "not good" → "not_good"
            result.append("not_" + tokens[i+1])
            i += 2
        else:
            result.append(tokens[i])
            i += 1

    return result


# -----------------------------
# Main Preprocessing Function
# -----------------------------
def preprocess_review(review):
    # Lowercase
    review = review.lower()

    # Remove HTML tags
    review = BeautifulSoup(review, "html.parser").get_text()

    # Handle contractions: "don't" → "do not"
    review = contractions.fix(review)

    # Convert emojis
    review = emoji.demojize(review)

    # Remove punctuation
    review = re.sub(f"[{string.punctuation}]", " ", review)

    # Tokenize
    tokens = nltk.word_tokenize(review)

    # Remove tokens that are not alphanumeric
    tokens = [token for token in tokens if token.replace("_", "").isalnum()]

    # Stopword removal
    stop_words = set(stopwords.words("english"))
    stop_words.discard("not")
    tokens = [word for word in tokens if word not in stop_words]

    # Apply negation handling
    tokens = handle_negation(tokens)

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word, pos="v") for word in tokens]

    # Rejoin
    cleaned_text = " ".join(tokens)

    return cleaned_text