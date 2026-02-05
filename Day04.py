import string

# Sample input text
text = "Python is amazing! It is used in AI, ML, and Data Science 2024."

# Step 1: Convert to lowercase
text = text.lower()

# Step 2: Remove punctuation
text = text.translate(str.maketrans('', '', string.punctuation))

# Step 3: Remove numbers
text = ''.join([char for char in text if not char.isdigit()])

# Step 4: Tokenize text
words = text.split()

# Step 5: Remove stopwords
stopwords = ['is', 'in', 'it', 'and', 'to', 'the']
cleaned_words = [word for word in words if word not in stopwords]

# Step 6: Join words back to sentence
cleaned_text = ' '.join(cleaned_words)

print("Original Text:")
print(text)

print("\nCleaned Text:")
print(cleaned_text)
