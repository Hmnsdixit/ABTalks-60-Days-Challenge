# Day 06: Word Frequency Distribution (NLP Statistics)

import string
from collections import Counter
import matplotlib.pyplot as plt

# Sample text dataset
text = """
Python is amazing. Python is widely used in AI, ML, and Data Science.
Data Science uses Python for analysis.
AI and ML depend on clean data.
"""

# Step 1: Convert text to lowercase
text = text.lower()

# Step 2: Remove punctuation
text = text.translate(str.maketrans("", "", string.punctuation))

# Step 3: Tokenization
words = text.split()

# Step 4: Remove stopwords
stopwords = ["is", "in", "and", "for", "on", "the", "to", "of"]
cleaned_words = [word for word in words if word not in stopwords]

# Step 5: Word frequency calculation
word_freq = Counter(cleaned_words)

# Step 6: Get top 20 words
top_words = word_freq.most_common(20)

# Step 7: Print word frequencies
print("Word Frequency Distribution:\n")
for word, count in top_words:
    print(word, ":", count)

# Step 8: Visualization
words_list = [item[0] for item in top_words]
counts_list = [item[1] for item in top_words]

plt.figure(figsize=(10, 5))
plt.bar(words_list, counts_list)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Top 20 Most Frequent Words")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
