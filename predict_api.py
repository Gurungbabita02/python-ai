import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import joblib
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI
app = FastAPI()

# Initialize model variable
model = None

def load_model():
    """Function to load the pre-trained model"""
    global model
    try:
        model = joblib.load('crash_predictor1.pkl')  # Load the pre-trained model
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        model = None

# Load the model when the app starts
load_model()

# Enable CORS so frontend can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API endpoint for prediction
@app.post("/predict")
async def predict(request: Request):
    try:
        # Check if the model is loaded
        if model is None:
            return {"error": "Model not loaded properly, please try again later."}

        # Get the data sent by the user
        data = await request.json()
        last5 = data.get("last5")  # Get the last 5 crash points from the user

        # Validate user input
        if not last5 or len(last5) != 5:
            return {"error": "Need exactly 5 crash points"}

        # Predict the next crash point using the model
        predicted_crash = model.predict([last5])[0]
        predicted_crash_rounded = round(float(predicted_crash), 2)

        # Optionally, send the user data to retrain the model
        retrain_model_with_user_data(last5)  # Call the function to retrain the model

        return {"predicted_crash": predicted_crash_rounded}

    except Exception as e:
        return {"error": f"Server error: {str(e)}"}

# Function to retrain the model with new user data
def retrain_model_with_user_data(user_data):
    try:
        # Load the previous dataset (or create a new one if it doesn't exist)
        try:
            df = pd.read_csv("crash_data.csv")
        except FileNotFoundError:
            df = pd.DataFrame(columns=[f'cp{i+1}' for i in range(5)] + ['target'])

        # Convert user data into features (previous crash points)
        user_features = np.array(user_data)  # Convert the input data into a numpy array

        # Generate the target value (set to NaN for now since we don't know the next crash point)
        user_target = np.nan

        # Create a new row for the user data
        new_data = pd.DataFrame([list(user_features) + [user_target]], columns=df.columns)
        
        # Append the new data to the dataframe
        df = df.append(new_data, ignore_index=True)

        # Save the updated dataset to the CSV file
        df.to_csv("crash_data.csv", index=False)
        print(f"User data added to crash_data.csv: {user_data}")

        # Retrain the model using the updated dataset
        retrain_model(df)

    except Exception as e:
        print(f"Error in retraining model with user data: {e}")

# Function to retrain the model using the updated data
def retrain_model(updated_df):
    try:
        # Features and target (using previous 5 crash points as features and the next crash point as target)
        X = updated_df[[f'cp{i+1}' for i in range(5)]]
        y = updated_df['target']

        # Split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model using the updated dataset
        model = XGBRegressor(objective='reg:squarederror', random_state=42)
        model.fit(X_train, y_train)

        # Save the retrained model
        joblib.dump(model, 'crash_predictor1.pkl')
        print("✅ Model retrained and saved as crash_predictor1.pkl")

    except Exception as e:
        print(f"Error retraining model: {e}")

# Initial function to train the model with simulated crash data
def train_initial_model():
    # Simulate crash data (replace with real historical crash data if available)
    np.random.seed(42)
    crash_data = np.random.uniform(1.0, 10.0, 1000)  # This is for simulation

    # Generate dataset: 5 previous crash points as features, next as target
    data = []
    for i in range(len(crash_data) - 5):
        features = crash_data[i:i+5]
        target = crash_data[i+5]
        data.append(list(features) + [target])

    # Create DataFrame
    columns = [f'cp{i+1}' for i in range(5)] + ['target']
    df = pd.DataFrame(data, columns=columns)

    # Features and target
    X = df[[f'cp{i+1}' for i in range(5)]]
    y = df['target']

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model using XGBoost
    model = XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model as 'crash_predictor1.pkl'
    joblib.dump(model, 'crash_predictor1.pkl')
    print("✅ Initial model trained and saved as crash_predictor1.pkl")

# Uncomment this to train the initial model only once
# train_initial_model()
