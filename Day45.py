from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# Initialize app
app = FastAPI(title="AI Prediction API 🚀")

# Dummy ML model (simple logic)
def predict_model(features):
    # Example logic (sum of inputs)
    result = sum(features)
    
    if result > 10:
        return "High"
    elif result > 5:
        return "Medium"
    else:
        return "Low"

# Request schema
class InputData(BaseModel):
    values: list[float]

# Home route
@app.get("/")
def home():
    return {"message": "AI Prediction API is running 🚀"}

# Prediction route
@app.post("/predict")
def predict(data: InputData):
    prediction = predict_model(data.values)
    
    return {
        "input": data.values,
        "prediction": prediction
    }