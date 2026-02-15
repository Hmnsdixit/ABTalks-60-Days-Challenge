# Day 13: Spam Detection Preprocessing Pipeline (NLP)
# Focus: Preprocessing ONLY (No Model Training)

import re
import string

# Basic Stopwords List
STOPWORDS = {
    "is", "in", "the", "and", "a", "an", "of", "to", "for", "on",
    "with", "as", "by", "at", "from", "this", "that", "are", "we",
    "you", "have", "has", "be", "now", "just", "here"
}

def load_dataset(file_path):
    """Load text messages from dataset file"""
    with open(file_path, "r", encoding="utf-8") as file:
        messages = file.readlines()
    return [msg.strip() for msg in messages if msg.strip()]


def lowercase_text(text):
    """Convert text to lowercase"""
    return text.lower()


def remove_punctuation(text):
    """Remove punctuation marks"""
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_numbers_special(text):
    """Remove numbers and special characters"""
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text


def tokenize(text):
    """Split text into tokens"""
    return text.split()


def remove_stopwords(tokens):
    """Remove stopwords from tokens"""
    return [word for word in tokens if word not in STOPWORDS]


def simple_stemming(tokens):
    """Basic stemming (rule-based)"""
    stemmed = []
    for word in tokens:
        if word.endswith("ing"):
            word = word[:-3]
        elif word.endswith("ed"):
            word = word[:-2]
        elif word.endswith("s") and len(word) > 3:
            word = word[:-1]
        stemmed.append(word)
    return stemmed


def spam_preprocessing_pipeline(text):
    """
    Reusable Spam Preprocessing Pipeline
    Steps:
    1. Lowercasing
    2. Remove punctuation
    3. Remove numbers & special characters
    4. Tokenization
    5. Stopword removal
    6. Stemming
    """
    original_text = text

    # Step 1: Lowercase
    text = lowercase_text(text)

    # Step 2: Remove punctuation
    text = remove_punctuation(text)

    # Step 3: Remove numbers & special characters
    text = remove_numbers_special(text)

    # Step 4: Tokenization
    tokens = tokenize(text)

    # Step 5: Remove stopwords
    tokens = remove_stopwords(tokens)

    # Step 6: Stemming
    tokens = simple_stemming(tokens)

    return original_text, tokens


# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    dataset_path = "spam_dataset.txt"

    print("=== Day 13: Spam Detection Preprocessing Pipeline ===\n")

    # Load dataset
    messages = load_dataset(dataset_path)

    print(f"Total Messages Loaded: {len(messages)}\n")

    # Process each message
    for i, message in enumerate(messages, 1):
        original, processed_tokens = spam_preprocessing_pipeline(message)

        print(f"Message {i} (Original):")
        print(original)

        print("Processed Tokens:")
        print(processed_tokens)

        print("-" * 60)
