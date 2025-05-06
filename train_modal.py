# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# import joblib

# # Simulate crash data (replace this with real historical crash data if available)
# np.random.seed(42)
# crash_data = np.random.uniform(1.0, 10.0, 1000)

# # Generate dataset: 5 previous crash points as features, next as target
# data = []
# for i in range(len(crash_data) - 5):
#     features = crash_data[i:i+5]
#     target = crash_data[i+5]
#     data.append(list(features) + [target])

# # Create DataFrame
# columns = [f'cp{i+1}' for i in range(5)] + ['target']
# df = pd.DataFrame(data, columns=columns)

# # Features and target
# X = df[[f'cp{i+1}' for i in range(5)]]
# y = df['target']

# # Split into train/test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Save model
# joblib.dump(model, 'crash_predictor.pkl')
# print("✅ Model trained and saved as crash_predictor.pkl")
