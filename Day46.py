from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(title="NLP Model API 🚀")

# Request schema
class TextInput(BaseModel):
    text: str

# Dummy NLP Model (basic logic)
def analyze_text(text):
    text_lower = text.lower()

    if "good" in text_lower or "great" in text_lower or "awesome" in text_lower:
        sentiment = "Positive 😊"
    elif "bad" in text_lower or "worst" in text_lower or "hate" in text_lower:
        sentiment = "Negative 😡"
    else:
        sentiment = "Neutral 😐"

    length = len(text.split())

    return {
        "sentiment": sentiment,
        "word_count": length
    }

# Home route
@app.get("/")
def home():
    return {"message": "NLP Model API is running 🚀"}

# NLP prediction route
@app.post("/analyze")
def analyze(data: TextInput):
    result = analyze_text(data.text)

    return {
        "input_text": data.text,
        "analysis": result
    }