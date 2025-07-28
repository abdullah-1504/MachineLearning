import streamlit as st
import joblib
import numpy as np

# Load model, scaler, and selected features
model = joblib.load('linear_regression_model.pkl')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('selected_features.pkl')

st.title('Boston Housing Price Prediction')
st.write('Enter the values for the features below:')

# Create input fields for each selected feature
def user_input_features():
    inputs = {}
    for feature in selected_features:
        value = st.number_input(f'{feature}', value=0.0, format='%f')
        inputs[feature] = value
    return np.array([list(inputs.values())])

X_input = user_input_features()

if st.button('Predict'):
    X_scaled = scaler.transform(X_input)
    prediction = model.predict(X_scaled)[0]
    st.success(f'Predicted House Price (in $1000s): {prediction:.2f}')
