from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from xgboost import XGBRegressor
import pandas as pd

app = FastAPI()

# Try loading the trained model at the application startup
model = None  # Initialize model as None initially

def load_model():
    global model
    try:
        # Try loading the model
        model = joblib.load('crash_predictor1.pkl')
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        model = None  # Model loading failed, set model to None

# Load the model on startup
load_model()

# Enable CORS so frontend can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(request: Request):
    try:
        # Check if the model is loaded
        if model is None:
            return {"error": "Model not loaded properly, please try again later."}

        # Get the data sent by the user
        data = await request.json()
        last5 = data.get("last5")

        if not last5 or len(last5) != 5:
            return {"error": "Need exactly 5 crash points"}

        # Predict the next crash point
        predicted_crash = model.predict([last5])[0]
        predicted_crash_rounded = round(float(predicted_crash), 2)

        return {"predicted_crash": predicted_crash_rounded}

    except Exception as e:
        return {"error": f"Server error: {str(e)}"}

