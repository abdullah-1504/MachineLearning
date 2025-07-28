from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained KNN wine model and scaler
model = joblib.load('winemodel.joblib')
scaler = joblib.load('winescaler.joblib')

# Use the actual selected features from the model
selected_features = [
    'free sulfur dioxide',
    'fixed acidity',
    'volatile acidity',
    'total sulfur dioxide',
    'residual sugar'
]

app = FastAPI(title="Wine Type Detection API")

class WineFeatures(BaseModel):
    free_sulfur_dioxide: float
    fixed_acidity: float
    volatile_acidity: float
    total_sulfur_dioxide: float
    residual_sugar: float

@app.post("/predict")
def predict_wine_type(features: WineFeatures):
    try:
        X = np.array([[getattr(features, f) for f in features.__fields__]])
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0].tolist()
        return {
            "prediction": int(prediction),
            "probability": proba,
            "classes": ["White (0)", "Red (1)"]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# To run: uvicorn wine_fastapi_app:app --reload
