# Day 27: Sentiment Analysis (Mini Project)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# -------------------------------
# Step 1: Dataset (Sentiment Data)
# -------------------------------
texts = [
    "I love this product",
    "This is amazing",
    "Very happy with the service",
    "I am satisfied",
    "Excellent experience",

    "I hate this",
    "Very bad service",
    "I am disappointed",
    "Worst experience ever",
    "Not good at all"
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
# Step 3: Pipeline
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
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# Step 7: Custom Testing
# -------------------------------
test_reviews = [
    "I really love this service",
    "This is the worst product",
    "Amazing quality and experience",
    "I am not happy"
]

predictions = pipeline.predict(test_reviews)

print("\n🧪 Sentiment Predictions:\n")
for text, pred in zip(test_reviews, predictions):
    print(f"{text} --> {'Positive 😊' if pred == 1 else 'Negative 😡'}")