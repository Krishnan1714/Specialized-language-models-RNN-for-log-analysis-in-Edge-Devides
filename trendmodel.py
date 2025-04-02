import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
import os

# Define device (use GPU if available, otherwise fallback to CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the cleaned and scaled dataset
df = pd.read_csv("cleaned_data_scaled.csv")

# Define features and target columns
features = ["Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear", "Target", "Type", "Failure Type"]
target_column = "Target"

# Separate features based on their scaling methods
standard_scaled = ["Air temperature", "Process temperature", "Torque"]  # Features scaled using StandardScaler
robust_scaled = ["Rotational speed", "Tool wear", "Target", "Type", "Failure Type"]  # Features scaled using RobustScaler

# Create sequences for LSTM model training
def create_sequences(data, seq_length=20):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])  # Sequence of features
        y.append(data[i + seq_length])      # Corresponding target sequence
    return np.array(X), np.array(y)

# Prepare training and testing data
X, y = create_sequences(df[features].values, seq_length=20)

# Use TimeSeriesSplit for cross-validation
ts = TimeSeriesSplit(n_splits=5)
train_idx, test_idx = list(ts.split(X))[-1]  # Select the last split for training and testing
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Convert data to PyTorch tensors and move to the selected device
X_train, X_test = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(X_test, dtype=torch.float32).to(device)
y_train, y_test = torch.tensor(y_train, dtype=torch.float32).to(device), torch.tensor(y_test, dtype=torch.float32).to(device)

# Create DataLoader for batch processing during training
batch_size = 32
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

# Define the LSTM model class
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)  # LSTM layer
        self.fc = nn.Linear(hidden_size, len(features))  # Fully connected layer for output

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # Pass through LSTM layer
        return self.fc(lstm_out[:, -1, :])  # Return output from the last time step


model_path = "trained_model.pth"
# Instantiate the model and move to device
model = LSTMModel(input_size=len(features)).to(device)

# Load pre-trained model if available
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print("Model loaded from disk.")
else:
    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop for the model
    for epoch in range(20):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()          # Reset gradients
            output = model(X_batch)        # Forward pass
            loss = criterion(output, y_batch)  # Compute loss
            loss.backward()               # Backpropagation
            optimizer.step()              # Update weights
    torch.save(model.state_dict(),model_path)
    print("Model trained and saved.")

# Inverse scaling of predictions
def inverse_transform(predictions):
    pred_df = pd.DataFrame(predictions, columns=features)
    for col in standard_scaled:
        scaler = StandardScaler()
        scaler.fit(df[[col]])
        pred_df[col] = scaler.inverse_transform(pred_df[[col]])  # Inverse transform for standard scaled features
    for col in robust_scaled:
        scaler = RobustScaler()
        scaler.fit(df[[col]])
        pred_df[col] = scaler.inverse_transform(pred_df[[col]])  # Inverse transform for robust scaled features
    return pred_df

# Model evaluation and prediction
model.eval()
print("MODEL EVAL \n",model.eval())
predictions = model(X_test).cpu().detach().numpy()  # Generate predictions on the test set
predictions_original = inverse_transform(predictions)  # Inverse scale the predictions
print(predictions_original.head())  # Display first few predicted rows

# Trend analysis for predicted and actual values
def trend_analysis(actual, predicted):
    actual_df = pd.DataFrame(actual, columns=features)
    predicted_df = pd.DataFrame(predicted, columns=features)
    trends = {}
    trend_text = ""
    for col in features:
        actual_trend = np.polyfit(range(len(actual_df[col])), actual_df[col], 1)[0]
        predicted_trend = np.polyfit(range(len(predicted_df[col])), predicted_df[col], 1)[0]
        trends[col] = (actual_trend, predicted_trend)
        trend_text += f"{col}: Actual Trend: {'Increasing' if actual_trend > 0 else 'Decreasing' if actual_trend < 0 else 'Stable'}, Predicted Trend: {'Increasing' if predicted_trend > 0 else 'Decreasing' if predicted_trend < 0 else 'Stable'}\n"
    return trends, trend_text

# Perform trend analysis
trends, trend_text = trend_analysis(y_test.cpu().numpy(), predictions)
print("Trend Analysis (Actual vs Predicted):", trends)
print(trend_text)

# Generate predictions
with torch.no_grad():
    predictions = model(X_test).cpu().numpy()  # Convert to NumPy for evaluation
    actual = y_test.cpu().numpy()  # Convert ground truth to NumPy

# Compute evaluation metrics
mae = mean_absolute_error(actual, predictions)
mse = mean_squared_error(actual, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(actual, predictions)

# Print evaluation results
print(f"Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# Save predictions to CSV
def save_predictions_to_csv(predictions, filename="predicted_data.csv"):
    predictions_original = inverse_transform(predictions)
    if os.path.isfile(filename):
        predictions_original.to_csv(filename, mode='a', index=False, header=False)
    else:
        predictions_original.to_csv(filename, mode='w', index=False)
    print(f"Predictions appended to {filename}")


# save

save_predictions_to_csv(predictions)
