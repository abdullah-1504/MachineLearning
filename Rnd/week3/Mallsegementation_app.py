from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load models
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
classifier = joblib.load("random_forest_model.pkl")

app = FastAPI()

# Input data structure
class UserInput(BaseModel):
    Gender: int     # 1 = Male, 0 = Female
    Age: float
    Annual_Income_k: float
    Spending_Score: float

@app.post("/predict-cluster/")
def predict_cluster(data: UserInput):
    X = np.array([[data.Gender, data.Age, data.Annual_Income_k, data.Spending_Score]])
    X_scaled = scaler.transform(X)
    cluster = int(kmeans.predict(X_scaled)[0])
    return {"cluster": cluster}

@app.post("/predict-segment/")
def predict_segment(data: UserInput):
    X = np.array([[data.Gender, data.Age, data.Annual_Income_k, data.Spending_Score]])
    X_scaled = scaler.transform(X)
    segment = int(classifier.predict(X_scaled)[0])
    return {"segment": segment}
@app.get("/")
def read_root():
    return { "Welcome to the Customer Segmentation API! Use /predict-cluster/ or /predict-segment/ to get predictions."}


#inference scriot cli tool

#cli tool to call this api 

