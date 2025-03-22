from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load the trained model
MODEL_PATH = "prognosis_model.joblib"  # Ensure this file exists
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ Error: Model file not found!")

# Define symptoms list (must match the training dataset)
SYMPTOM_LIST = [
    "Fatigue", "Headache", "High Fever", "Nausea", "Loss of Appetite",
    "Cough", "Shortness of Breath", "Chest Pain", "Dizziness", "Sore Throat"
]

@app.route("/", methods=["GET"])
def home():
    return "Disease Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        name = data.get("name", "Unknown")
        age = data.get("age", 0)
        selected_symptoms = data.get("symptoms", [])

        if not selected_symptoms:
            return jsonify({"error": "No symptoms provided"}), 400

        # Convert symptoms to binary format (1 if selected, 0 if not)
        input_vector = [1 if symptom in selected_symptoms else 0 for symptom in SYMPTOM_LIST]
        input_vector = np.array(input_vector).reshape(1, -1)

        # Predict disease
        prediction = model.predict(input_vector)[0]

        return jsonify({
            "name": name,
            "age": int(age),
            "predictions": prediction
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

                        

        

    


                       
        
   
