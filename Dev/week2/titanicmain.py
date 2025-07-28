from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os

# Load model, scaler, and encoders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "titanic_best_knn_model.joblib"))
scaler = joblib.load(os.path.join(BASE_DIR, "titanic_knn_scaler.joblib"))
sex_encoder = joblib.load(os.path.join(BASE_DIR, "titanic_sex_encoder.joblib"))
cabin_encoder = joblib.load(os.path.join(BASE_DIR, "titanic_cabin_encoder.joblib"))
embarked_encoder = joblib.load(os.path.join(BASE_DIR, "titanic_embarked_encoder.joblib"))

app = FastAPI()

# Update the fields below to match the features used for training, in the correct order!
class TitanicInput(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Cabin: str
    Embarked: str
    # Add/remove fields as per your final model's features

@app.get("/")
def root():
    return {"message": "🚢 Titanic Survival Prediction API. POST to /predict/ with passenger features."}

@app.post("/predict/")
def predict(data: TitanicInput):
    # Prepare input in the correct order
    input_dict = data.dict()
    # Encode categorical features
    input_dict["Sex"] = sex_encoder.transform([input_dict["Sex"]])[0]
    input_dict["Cabin"] = cabin_encoder.transform([input_dict["Cabin"]])[0]
    input_dict["Embarked"] = embarked_encoder.transform([input_dict["Embarked"]])[0]
    # Arrange features in the order used for training
    X = np.array([[input_dict["Pclass"], input_dict["Sex"], input_dict["Age"], input_dict["SibSp"],
                   input_dict["Parch"], input_dict["Fare"], input_dict["Cabin"], input_dict["Embarked"]]])
    X_scaled = scaler.transform(X)
    prediction = int(model.predict(X_scaled)[0])
    survived = "Survived" if prediction == 1 else "Not Survived"
    return {"prediction": prediction, "survived": survived}