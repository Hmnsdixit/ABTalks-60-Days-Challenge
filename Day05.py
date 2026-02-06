import re

# Step 1: Normalize text
def normalize_text(text):
    return text.lower()

# Step 2: Remove URLs, numbers, symbols
def remove_noise(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

# Step 3: Tokenization
def tokenize(text):
    return text.split()

# Step 4: Remove stopwords
def remove_stopwords(tokens):
    stopwords = {"is", "the", "and", "to", "in", "of", "for", "on", "with", "a", "an"}
    return [word for word in tokens if word not in stopwords]

# Step 5: Simple stemming
def stem_words(tokens):
    suffixes = ["ing", "ed", "s"]
    stemmed = []
    for word in tokens:
        for suf in suffixes:
            if word.endswith(suf):
                word = word.replace(suf, "")
        stemmed.append(word)
    return stemmed

# Step 6: Full preprocessing pipeline
def preprocess_article(text):
    text = normalize_text(text)
    text = remove_noise(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stem_words(tokens)
    return " ".join(tokens)

# Example dataset (News Articles)
news_articles = [
    "Breaking News: Python is used in AI & ML systems!",
    "Sports Update: India won the match by 6 runs.",
    "Tech News: New AI models are transforming data science."
]

print("Cleaned News Articles:\n")

for article in news_articles:
    print(preprocess_article(article))
