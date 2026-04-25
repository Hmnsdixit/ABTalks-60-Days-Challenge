from flask import Flask, request, jsonify

app = Flask(__name__)

# Home Route
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "🚀 Flask AI API is running!"
    })


# Prediction Route (Same logic as Day 45 FastAPI)
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "values" not in data:
        return jsonify({"error": "Please provide 'values' in JSON"}), 400

    values = data["values"]

    # Simple logic (same as before)
    avg = sum(values) / len(values)

    if avg > 5:
        result = "High"
    elif avg > 2:
        result = "Medium"
    else:
        result = "Low"

    return jsonify({
        "input": values,
        "prediction": result
    })


if __name__ == "__main__":
    app.run(debug=True)