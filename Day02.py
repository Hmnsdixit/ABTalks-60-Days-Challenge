# Day 02 – NLP Basics
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

paragraph = "Python is easy to learn. It is widely used in data science. NLP is very interesting."

sentences = sent_tokenize(paragraph)

print("Sentence Tokenization:")
for sentence in sentences:
    print(sentence)

print("\nWord Tokenization:")
for sentence in sentences:
    words = word_tokenize(sentence)
    print(words)
