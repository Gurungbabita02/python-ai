import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Simulate realistic crash data
np.random.seed(42)
crash_data = np.random.uniform(1.0, 10.0, 1000)  # Simulate 1000 crash points

# Create a DataFrame
data = pd.DataFrame({ 'crash': crash_data })

# Create the feature: rolling average of last 5 crashes (as input for prediction)
data['avg_last5'] = data['crash'].rolling(window=5).mean().shift(1)

# Drop rows with NaN (first 5 rows due to rolling + shift)
data = data.dropna()

# Define features (X) and target (y)
X = data[['avg_last5']]
y = data['crash']

# Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'crash_predictor.pkl')
print("✅ Model trained and saved as crash_predictor.pkl")
