from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
model = joblib.load("titanic_model.pkl")

# Define the input data structure
class Passenger(BaseModel):
    Pclass: int
    Sex: int
    Fare: float
    Cabin: int
    cabindeck: int
    hascabin: int
    Embarked: int

app = FastAPI()

@app.post("/predict")
def predict(passenger: Passenger):
    data = np.array([[passenger.Pclass, passenger.Sex, passenger.Fare, passenger.Cabin, passenger.cabindeck, passenger.hascabin, passenger.Embarked]])
    prediction = model.predict(data)
    return {"survived": int(prediction[0])}

import requests

url = "http://127.0.0.1:8000/predict"
data = {
    "Pclass": 3,
    "Sex": 1,
    "Fare": 7.25,
    "Cabin": 0,
    "cabindeck": 0,
    "hascabin": 0,
    "Embarked": 1
}
response = requests.post(url, json=data)
print(response.json())