# Day 01 – Python Basics
def tokenize_text(text):
    text = text.lower()
    return text.split()

def remove_stopwords(words):
    stopwords = ['is', 'to', 'the', 'a', 'an', 'and', 'in', 'on', 'of']
    return [word for word in words if word not in stopwords]

def word_frequency(words):
    freq = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1
    return freq

paragraph = "Python is easy to learn and Python is powerful"

tokens = tokenize_text(paragraph)
filtered_tokens = remove_stopwords(tokens)
frequency = word_frequency(filtered_tokens)

print("Tokens:", tokens)
print("After Stopwords Removal:", filtered_tokens)
print("Word Frequency:", frequency)
