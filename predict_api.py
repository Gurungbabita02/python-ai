from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from typing import List

app = FastAPI()
# Load the trained model
model = joblib.load('crash_predictor.pkl')

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
        data = await request.json()
        last5 = data.get("last5")

        if not last5 or len(last5) != 5:
            return { "error": "Need exactly 5 crash points" }

        predicted_crash = model.predict([last5])[0]
        return { "predicted_crash": float(round(predicted_crash, 2)) }

    except Exception as e:
        return { "error": f"Server error: {str(e)}" }
