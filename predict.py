import numpy as np
import pickle

# Load the trained model
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Define the symptoms in the same order as used during training
SYMPTOMS = ["fatigue", "headache","high_fever", "nausea", "loss_of_appetite"]

def predict_disease(input_symptoms):
    """
    Predict disease based on input symptoms.
    
    :param input_symptoms: Dictionary with symptom names as keys and 0 or 1 as values.
    :return: Predicted disease as a string.
    """
    # Convert input symptoms into a NumPy array
    input_vector = np.array([1 if symptom in input_symptoms else 0 for symptom in SYMPTOMS]).reshape(1, -1)




    
    # Make prediction
    prediction = model.predict(input_vector)
    
    return prediction[0]

