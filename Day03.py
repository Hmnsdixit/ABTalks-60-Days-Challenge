# Day 03 – Text preprocessing
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Sample product reviews
reviews = [
    "I loved the quality of this product",
    "The product is running smoothly",
    "Customers are liking the new design"
]

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

print("Original Reviews:")
for review in reviews:
    print(review)

print("\nAfter Stemming:")
for review in reviews:
    words = word_tokenize(review.lower())
    stemmed_words = [stemmer.stem(word) for word in words]
    print(stemmed_words)

print("\nAfter Lemmatization:")
for review in reviews:
    words = word_tokenize(review.lower())
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    print(lemmatized_words)
