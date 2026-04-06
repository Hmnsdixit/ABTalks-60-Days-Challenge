# Day 28: Weekly ML Assessment (Review + Pipeline + Evaluation)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# Step 1: Dataset
# -------------------------------
texts = [
    "I love this product",
    "Amazing experience",
    "Very happy with the service",
    "Best purchase ever",
    "Highly recommended",

    "Worst product",
    "Very bad experience",
    "I hate this",
    "Not satisfied",
    "Terrible service"
]

labels = [
    1,1,1,1,1,   # Positive
    0,0,0,0,0    # Negative
]

# -------------------------------
# Step 2: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels,
    test_size=0.3,
    random_state=42,
    stratify=labels
)

# -------------------------------
# Step 3: ML Pipeline
# -------------------------------
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('model', LogisticRegression())
])

# -------------------------------
# Step 4: Train Model
# -------------------------------
pipeline.fit(X_train, y_train)

# -------------------------------
# Step 5: Prediction
# -------------------------------
y_pred = pipeline.predict(X_test)

# -------------------------------
# Step 6: Evaluation
# -------------------------------
print("\n📊 Model Evaluation:\n")

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# Step 7: Manual Testing
# -------------------------------
test_samples = [
    "I really love this",
    "This is very bad",
    "Excellent product",
    "Not good"
]

predictions = pipeline.predict(test_samples)

print("\n🧪 Sample Predictions:\n")
for text, pred in zip(test_samples, predictions):
    print(f"{text} --> {'Positive 😊' if pred == 1 else 'Negative 😡'}")

# -------------------------------
# Step 8: Final Learnings Print
# -------------------------------
print("\n📌 Key Learnings:")
print("- Text preprocessing is important")
print("- TF-IDF converts text to numbers")
print("- Logistic Regression is good for classification")
print("- Evaluation metrics help measure performance")