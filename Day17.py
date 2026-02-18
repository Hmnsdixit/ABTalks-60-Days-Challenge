# 🚀 Day 17: NLP + Probability using SciPy

import re
from collections import Counter
import numpy as np
from scipy.stats import poisson

# -------------------------------
# Step 1: Sample Text Corpus
# -------------------------------
text = """
AI is transforming the world. 
AI and Data Science are powerful fields.
AI models learn from data and make predictions.
"""

print("Original Text:\n", text)

# -------------------------------
# Step 2: Text Preprocessing
# -------------------------------
# Lowercase
text = text.lower()

# Remove punctuation
text = re.sub(r'[^a-z\s]', '', text)

# Tokenization (split into words)
tokens = text.split()

print("\nTokens:")
print(tokens)

# -------------------------------
# Step 3: Word Frequency Count
# -------------------------------
word_counts = Counter(tokens)
total_words = len(tokens)

print("\nWord Frequency:")
for word, count in word_counts.items():
    print(f"{word}: {count}")

print("\nTotal Words:", total_words)

# -------------------------------
# Step 4: Probability of Each Word
# -------------------------------
print("\nWord Probabilities (P(word)):")
word_probabilities = {}

for word, count in word_counts.items():
    prob = count / total_words
    word_probabilities[word] = prob
    print(f"P({word}) = {prob:.4f}")

# -------------------------------
# Step 5: Convert frequencies to array (for statistical modeling)
# -------------------------------
frequencies = np.array(list(word_counts.values()))
mean_frequency = np.mean(frequencies)

print("\nMean Word Frequency:", mean_frequency)

# -------------------------------
# Step 6: Poisson Distribution Modeling
# -------------------------------
print("\nPoisson Distribution Probabilities:")
print("(Modeling rare vs frequent word occurrences)")

for word, count in word_counts.items():
    poisson_prob = poisson.pmf(count, mean_frequency)
    print(f"Poisson P(X={count}) for '{word}' = {poisson_prob:.4f}")

# -------------------------------
# Step 7: Interpretation
# -------------------------------
print("\n📊 Interpretation:")
print("- Common words have higher probability.")
print("- Rare words have lower probability.")
print("- Text word frequencies often follow skewed distribution.")
print("- Poisson helps model occurrence of words in corpus.")
