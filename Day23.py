# Day 23: TF-IDF + Logistic Regression Text Classification

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# Step 1: Dataset
# -------------------------------
texts = [
    "Win money now",
    "Limited offer just for you",
    "Call now and claim prize",
    "Hello how are you",
    "Let's meet tomorrow",
    "Are you coming today",
]

labels = [1, 1, 1, 0, 0, 0]  # 1 = Spam, 0 = Not Spam

# -------------------------------
# Step 2: TF-IDF Feature Extraction
# -------------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
y = labels

# -------------------------------
# Step 3: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Step 4: Train Model
# -------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------------------
# Step 5: Prediction
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Step 6: Evaluation
# -------------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# Step 7: Custom Input Test
# -------------------------------
test_text = ["Congratulations! You have won a lottery"]
test_vector = vectorizer.transform(test_text)
prediction = model.predict(test_vector)

print("\nCustom Input:", test_text[0])
print("Prediction:", "Spam" if prediction[0] == 1 else "Not Spam")