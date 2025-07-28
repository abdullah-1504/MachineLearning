from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os

# Load model and scaler from same folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "best_knn_model.joblib")
scaler_path = os.path.join(BASE_DIR, "knn_scaler.joblib")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Initialize FastAPI
app = FastAPI()

# Define the input structure
class CancerInput(BaseModel):
    radius_worst: float
    perimeter_mean: float
    concave_points_mean: float
    concave_points_worst: float
    radius_mean: float
    area_mean: float
    area_worst: float
    concavity_mean: float
    concavity_worst: float
    perimeter_worst: float

@app.get("/")
def root():
    return {
        "message": "🧬 Breast Cancer Detection API is active.",
        "usage": "POST to /predict/ with 10 selected features."
    }

@app.post("/predict/")
def predict(data: CancerInput):
    X = np.array([[getattr(data, field) for field in data.__fields__]])
    X_scaled = scaler.transform(X)
    prediction = int(model.predict(X_scaled)[0])
    label = "Malignant" if prediction == 1 else "Benign"
    return {
        "prediction": prediction,
        "diagnosis": label
    }
