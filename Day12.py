# Day 12: Build a Reusable NLP Pipeline
# Community NLP Challenge

import re
import string
from collections import Counter

# Basic stopwords list (no external library needed)
STOPWORDS = {
    "is", "in", "the", "and", "a", "an", "of", "to", "for", "on",
    "with", "as", "by", "at", "from", "that", "this", "it", "be",
    "are", "was", "were", "or"
}

def clean_text(text):
    """Remove punctuation and special characters"""
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)  # remove numbers
    return text

def lowercase_text(text):
    """Convert text to lowercase"""
    return text.lower()

def tokenize(text):
    """Split text into tokens"""
    return text.split()

def remove_stopwords(tokens):
    """Remove common stopwords"""
    return [word for word in tokens if word not in STOPWORDS]

def stemming(tokens):
    """Simple rule-based stemming (basic)"""
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

def nlp_pipeline(text, use_stemming=True):
    """
    Reusable NLP Pipeline Function
    Steps:
    1. Lowercasing
    2. Cleaning
    3. Tokenization
    4. Stopword Removal
    5. Stemming (Optional)
    """
    print("Original Text:", text)

    # Step 1: Lowercase
    text = lowercase_text(text)

    # Step 2: Clean text
    text = clean_text(text)

    # Step 3: Tokenize
    tokens = tokenize(text)

    # Step 4: Remove Stopwords
    tokens = remove_stopwords(tokens)

    # Step 5: Optional Stemming
    if use_stemming:
        tokens = stemming(tokens)

    return tokens


# ===== Sample Input (Testing Pipeline) =====
sample_text = """
NLP Pipeline is very powerful and useful for processing text data in Machine Learning and AI systems.
"""

# Run Pipeline
processed_tokens = nlp_pipeline(sample_text)

print("\nProcessed Tokens:")
print(processed_tokens)

# Word Frequency Analysis
freq = Counter(processed_tokens)

print("\nWord Frequency:")
print(freq)
