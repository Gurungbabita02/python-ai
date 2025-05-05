import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Simulate crash data (replace with actual data)
np.random.seed(42)
data = pd.DataFrame({
    'crash': np.random.uniform(1.0, 10.0, 1000)  # Crash points between 1 and 10
})

# Generate features (previous 5 rounds' crash points)
data['avg_last5'] = data['crash'].rolling(5).mean().shift(1)
data = data.dropna()

# Define features (X) and target (y)
X = data[['avg_last5']]
y = data['crash']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'crash_predictor.pkl')
print("✅ Model trained and saved as crash_predictor.pkl")
