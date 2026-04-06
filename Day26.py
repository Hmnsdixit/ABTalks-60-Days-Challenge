# Day 26: Complete ML Pipeline (Text Classification)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# -------------------------------
# Step 1: Dataset
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
# Step 2: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels,
    test_size=0.3,
    random_state=42,
    stratify=labels
)

# -------------------------------
# Step 3: Build Pipeline
# -------------------------------
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('model', LogisticRegression())
])

# -------------------------------
# Step 4: Train Pipeline
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
# Step 7: Test on New Data
# -------------------------------
test_texts = [
    "You won a free lottery",
    "Let's go for dinner tonight"
]

predictions = pipeline.predict(test_texts)

print("\n🧪 Custom Predictions:")
for text, pred in zip(test_texts, predictions):
    print(f"{text} --> {'Spam' if pred == 1 else 'Not Spam'}")