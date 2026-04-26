from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="🚀 AI Prediction API")

# Input schema
class InputData(BaseModel):
    values: list[int]

# Home route
@app.get("/")
def home():
    return {"message": "🚀 FastAPI app running with Uvicorn!"}

# Prediction route
@app.post("/predict")
def predict(data: InputData):
    values = data.values
    avg = sum(values) / len(values)

    if avg > 5:
        result = "High"
    elif avg > 2:
        result = "Medium"
    else:
        result = "Low"

    return {
        "input": values,
        "prediction": result
    }