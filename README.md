# weather-based-prediction-of-wind-turbine-energy-output-a-next-generation-approach-
# weather_based_wind_energy_prediction.py

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# Load and Preprocess Data
# -----------------------------
def load_and_preprocess_data(csv_path):
    data = pd.read_csv(csv_path)

    X = data[['wind_speed', 'temperature', 'pressure', 'humidity']]
    y = data['power_output']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

# -----------------------------
# Train Model
# -----------------------------
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/trained_model.pkl")

    return model, X_test, y_test

# -----------------------------
# Evaluate Model
# -----------------------------
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("Model Performance")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")

# -----------------------------
# Predict Power Output
# -----------------------------
def predict_power(weather_input, scaler):
    model = joblib.load("models/trained_model.pkl")
    weather_scaled = scaler.transform(
        np.array(weather_input).reshape(1, -1)
    )
    return model.predict(weather_scaled)[0]

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    # Sample dataset creation (only if not exists)
    os.makedirs("data", exist_ok=True)
    csv_path = "data/sample_weather_data.csv"

    if not os.path.exists(csv_path):
        sample_data = pd.DataFrame({
            "wind_speed": [5.2, 6.5, 7.8, 9.0, 10.5],
            "temperature": [25, 24, 23, 22, 21],
            "pressure": [1012, 1010, 1008, 1005, 1003],
            "humidity": [60, 55, 50, 45, 40],
            "power_output": [120, 180, 260, 340, 430]
        })
        sample_data.to_csv(csv_path, index=False)

    # Pipeline
    X, y, scaler = load_and_preprocess_data(csv_path)
    model, X_test, y_test = train_model(X, y)
    evaluate_model(model, X_test, y_test)

    # Prediction Example
    new_weather = [8.5, 23, 1007, 48]  # wind_speed, temp, pressure, humidity
    predicted_power = predict_power(new_weather, scaler)

    print(f"Predicted Wind Turbine Power Output: {predicted_power:.2f} kW")
