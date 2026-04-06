# Day 25: Hyperparameter Tuning using GridSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
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
# Step 3: Pipeline (TF-IDF + Model)
# -------------------------------
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('model', LogisticRegression())
])

# -------------------------------
# Step 4: Hyperparameter Grid
# -------------------------------
param_grid = {
    'tfidf__ngram_range': [(1,1), (1,2)],
    'tfidf__max_df': [0.75, 1.0],
    'model__C': [0.1, 1, 10],
    'model__solver': ['liblinear']
}

# -------------------------------
# Step 5: GridSearchCV
# -------------------------------
grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring='f1',
    verbose=1
)

grid.fit(X_train, y_train)

# -------------------------------
# Step 6: Best Parameters
# -------------------------------
print("\n🔥 Best Parameters:")
print(grid.best_params_)

# -------------------------------
# Step 7: Prediction
# -------------------------------
y_pred = grid.predict(X_test)

# -------------------------------
# Step 8: Evaluation
# -------------------------------
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# Step 9: Custom Test
# -------------------------------
test_text = ["You have won free money"]
prediction = grid.predict(test_text)

print("\n🧪 Test Input:", test_text[0])
print("Prediction:", "Spam" if prediction[0] == 1 else "Not Spam")