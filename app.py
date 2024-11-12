from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('house_prediction.pkl', 'rb'))
scaler = StandardScaler()

# Define input features based on features selected after VIF
input_features = ['MedInc', 'HouseAge',  'Population','AveOccup' ]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get values from form
    input_values = [float(request.form[feature]) for feature in input_features]
    
    # Convert input to DataFrame for scaling
    input_data = pd.DataFrame([input_values], columns=input_features)
    scaled_data = scaler.fit_transform(input_data)  # Ensure to fit scaler as per training set if using pre-scaler

    # Make prediction
    prediction = model.predict(scaled_data)
    
    # Format the result
    result = f"Predicted House Price: ${prediction[0]:,.2f}"
    
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True) 