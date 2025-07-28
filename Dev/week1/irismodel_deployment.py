import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib

def prepare_model():
    # Load and prepare data
    df = pd.read_csv('E:/Downloads/iris.data.csv')
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'species']
    
    # Select features (based on your VIF analysis)
    selected_features = ['sepal width', 'petal width']  # Update with your selected features
    
    # Prepare features and target
    X = df[selected_features]
    le = LabelEncoder()
    y = le.fit_transform(df['species'])
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = LogisticRegression(random_state=42, multi_class='multinomial')
    model.fit(X_scaled, y)
    
    # Save model, scaler, and label encoder
    joblib.dump(model, 'iris_model.pkl')
    joblib.dump(scaler, 'iris_scaler.pkl')
    joblib.dump(le, 'iris_label_encoder.pkl')
    joblib.dump(selected_features, 'iris_features.pkl')
    
    print("Model and preprocessing objects saved successfully!")
    return selected_features

def predict_species(features_dict):
    """
    Make predictions on new data
    features_dict should contain values for all selected features
    """
    try:
        # Load saved objects
        model = joblib.load('iris_model.pkl')
        scaler = joblib.load('iris_scaler.pkl')
        le = joblib.load('iris_label_encoder.pkl')
        selected_features = joblib.load('iris_features.pkl')
        
        # Validate input features
        if not all(feature in features_dict for feature in selected_features):
            raise ValueError(f"Input must contain all features: {selected_features}")
        
        # Prepare input data
        X = pd.DataFrame([features_dict])[selected_features]
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = model.predict(X_scaled)
        species = le.inverse_transform(prediction)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        
        # Prepare response
        class_probabilities = {
            le.inverse_transform([i])[0]: round(prob * 100, 2)
            for i, prob in enumerate(probabilities)
        }
        
        return {
            "predicted_species": species,
            "confidence_scores": class_probabilities
        }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Train and save model
    selected_features = prepare_model()
    
    # Example prediction
    test_features = {
        "sepal length": 5.1,
        "sepal width": 3.5,
        "petal length": 1.4,
        "petal width": 0.2
    }
    
    result = predict_species(test_features)
    print("\nTest Prediction:")
    print("-" * 50)
    print(f"Input features: {test_features}")
    print(f"Prediction result: {result}")