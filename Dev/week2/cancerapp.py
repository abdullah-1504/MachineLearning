from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained KNN model and scaler
model = joblib.load(r"e:\OneDrive\Documents\coding\MLInternship\week2\KNNcancermodel.joblib")
scaler = joblib.load(r"e:\OneDrive\Documents\coding\MLInternship\week2\knn_scaler.joblib")

app = FastAPI(title="Breast Cancer Detection API")

# Actual selected features from the notebook (7 features, order as printed)
class CancerFeatures(BaseModel):
    area_worst: float
    concavity_mean: float
    concavity_worst: float
    concave_points_worst: float
    concave_points_mean: float
    radius_worst: float
    area_mean: float

@app.post("/predict")
def predict_cancer(features: CancerFeatures):
    try:
        # Convert input to numpy array and scale
        X = np.array([[getattr(features, f) for f in features.__fields__]])
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0].tolist()
        return {
            "prediction": int(prediction),
            "probability": proba,
            "classes": ["Benign (0)", "Malignant (1)"]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# To run: uvicorn cancerapp:app --reload
