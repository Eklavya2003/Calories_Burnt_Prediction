from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__, template_folder="templates")

# Load or train the model
MODEL_PATH = "calories_model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except (FileNotFoundError, pickle.UnpicklingError):
    print("Model not found. Training a new model...")
    
    # Load dataset
    calories = pd.read_csv('calories.csv')
    exercise_data = pd.read_csv('exercise.csv')
    
    # Merge datasets
    data = pd.merge(exercise_data, calories, on='User_ID')
    
    # Ensure all necessary columns are present
    feature_columns = ['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
    missing_cols = [col for col in feature_columns if col not in data.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in dataset: {missing_cols}")
    
    # Convert Gender to numeric
    data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
    
    # Convert data to numeric and handle missing values
    X = data[feature_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
    Y = pd.to_numeric(data['Calories']).fillna(0)
    
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Train model
    model = XGBRegressor()
    model.fit(X_train, Y_train)
    
    # Save the model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print("Model trained and saved successfully.")

@app.route('/')
def home():
    return render_template('Calories_Burnt_Prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data safely
        input_data = {
            'Gender': request.form.get('Gender', type=float),
            'Age': request.form.get('Age', type=float),
            'Height': request.form.get('Height', type=float),
            'Weight': request.form.get('Weight', type=float),
            'Duration': request.form.get('Duration', type=float),
            'HeartRate': request.form.get('HeartRate', type=float),
            'BodyTemperature': request.form.get('BodyTemperature', type=float)
        }
        
        # Ensure all values are numeric
        if None in input_data.values():
            return render_template('Calories_Burnt_Prediction.html', pred="Error: Invalid input. Please enter valid numbers.")

        # Convert input to NumPy array and reshape
        input_array = np.array(list(input_data.values())).reshape(1, -1)

        # Predict
        prediction = model.predict(input_array)[0]

        return render_template('Calories_Burnt_Prediction.html', pred=f'Predicted Calories Burnt: {prediction:.2f}')
    
    except Exception as e:
        return render_template('Calories_Burnt_Prediction.html', pred=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)


