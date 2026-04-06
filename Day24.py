# Day 24: Model Evaluation (Real Implementation)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# -------------------------------
# Step 1: Dataset (Balanced)
# -------------------------------
texts = [
    "Win money now",
    "Limited offer just for you",
    "Call now and claim prize",
    "Free lottery entry",
    "Congratulations you won prize",
    "Exclusive deal for you",

    "Hello how are you",
    "Let's meet tomorrow",
    "Are you coming today",
    "Good morning",
    "See you soon",
    "Let's have lunch"
]

labels = [1,1,1,1,1,1, 0,0,0,0,0,0]

# -------------------------------
# Step 2: TF-IDF
# -------------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
y = labels

# -------------------------------
# Step 3: Split (IMPORTANT FIX)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y   # 🔥 MUST
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
# Step 6: Metrics
# -------------------------------
print("\n📊 Evaluation Metrics\n")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# -------------------------------
# Step 7: Classification Report
# -------------------------------
print("\n📄 Classification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# Step 8: Confusion Matrix
# -------------------------------
print("\n📊 Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# -------------------------------
# Step 9: Interpretation
# -------------------------------
print("\n📌 Interpretation:")
print("- High Precision = Few false spam predictions")
print("- High Recall = Most spam messages detected")
print("- F1 Score balances both Precision & Recall")