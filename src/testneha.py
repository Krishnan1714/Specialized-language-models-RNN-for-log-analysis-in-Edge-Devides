import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------------
# 1. Load and Preprocess Dataset
# -------------------------
df = pd.read_csv("assets/cleaned_data_1.csv")  # Update file path if needed
df.columns = df.columns.str.strip()  # Clean column names

# Define features used during training (order matters)
features = ["Air temperature", "Process temperature", "Rotational speed",
            "Torque", "Tool wear", "Target", "Type", "Failure Type"]

# Scale the features
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[features])

# -------------------------
# 2. Create Sequences for One-Step Forecasting
# -------------------------
def create_sequences(data, seq_length=10, forecast_horizon=1):
    """
    Creates sequences for one-step forecasting.
    Each input is a window of 'seq_length' rows,
    and each target is the next row (forecast_horizon=1).
    """
    X, y = [], []
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length : i + seq_length + forecast_horizon])
    return np.array(X), np.array(y)

sequence_length = 10
forecast_horizon = 1  # one-step forecast
X_seq, y_seq = create_sequences(df_scaled, seq_length=sequence_length, forecast_horizon=forecast_horizon)

# Since forecast_horizon=1, y_seq has shape (n_samples, 1, n_features); squeeze it:
y_seq = y_seq.squeeze(axis=1)

print("X_seq shape:", X_seq.shape)  # Expected: (n_samples, 10, 8)
print("y_seq shape:", y_seq.shape)  # Expected: (n_samples, 8)

# -------------------------
# 3. Load the Trained Model and Make Predictions
# -------------------------
model = load_model("models/lstm_model.h5")  # Update model path if needed
predictions = model.predict(X_seq)
print("Predictions shape:", predictions.shape)  # Expected: (n_samples, 8)

# -------------------------
# 4. Inverse Transform Predictions and Targets to Original Scale
# -------------------------
predictions_original = scaler.inverse_transform(predictions)
y_seq_original = scaler.inverse_transform(y_seq)

# -------------------------
# 5. Compute Regression Metrics
# -------------------------
mse = mean_squared_error(y_seq_original, predictions_original)
mae = mean_absolute_error(y_seq_original, predictions_original)
r2 = r2_score(y_seq_original, predictions_original)
epsilon = 1e-10  # To avoid division by zero
mape = np.mean(np.abs((y_seq_original - predictions_original) / (y_seq_original + epsilon))) * 100

print("📈 Evaluation Metrics:")
print(f"✅ Mean Squared Error (MSE): {mse:.4f}")
print(f"✅ Mean Absolute Error (MAE): {mae:.4f}")
print(f"✅ R² Score: {r2:.4f}")
print(f"✅ Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# -------------------------
# 6. Visualization: Plot Actual vs. Predicted for a Selected Feature
# -------------------------
# For example, plot "Failure Type"
feature_to_plot = "Failure Type"
feature_index = features.index(feature_to_plot)

plt.figure(figsize=(12, 6))
plt.plot(y_seq_original[:, feature_index], label="Actual", linestyle="dashed", color="blue")
plt.plot(predictions_original[:, feature_index], label="Predicted", color="red")
plt.xlabel("Time Steps")
plt.ylabel(feature_to_plot)
plt.title(f"Actual vs. Predicted for {feature_to_plot}")
plt.legend()
plt.show()
