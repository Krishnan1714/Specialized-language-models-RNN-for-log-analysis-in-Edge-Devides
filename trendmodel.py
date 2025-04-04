import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
import os
from sklearn.preprocessing import OrdinalEncoder


# Define device (use GPU if available, otherwise fallback to CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the cleaned and scaled dataset
df = pd.read_csv("assets/cleaned_data_scaled.csv")

# One-Hot Encode Failure Type
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[["Failure Type"]])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(["Failure Type"]))
df = pd.concat([df.drop("Failure Type", axis=1), encoded_df], axis=1)

# Define features and target columns
features = ["Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear", "Target"]+ list(encoder.get_feature_names_out(["Failure Type"]))

# Separate features based on their scaling methods
standard_scaled = ["Air temperature", "Process temperature", "Torque"]  # Features scaled using StandardScaler
robust_scaled = ["Rotational speed", "Tool wear"]  # Features scaled using RobustScaler

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


model_path = "trained_lstm_model.pth"
# Instantiate the model and move to device
model = LSTMModel(input_size=len(features)).to(device)

# Load pre-trained model if available
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print("Model loaded from disk.")
else:
     # Define loss function and optimizer
    criterion_regression = nn.MSELoss()
    criterion_binary = nn.BCEWithLogitsLoss()
    criterion_multiclass = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)

            # --- Slice the outputs ---
            output_reg = output[:, :5]                  # Regression outputs
            output_target = output[:, 5]                # Binary classification (logits)
            output_failure = output[:, 6:]              # Multiclass classification (logits)

            # --- Slice the labels ---
            y_reg = y_batch[:, :5]
            y_target = y_batch[:, 5]
            y_failure = torch.argmax(y_batch[:, 6:], dim=1)  # Convert one-hot to class index

            # --- Compute losses ---
            loss_reg = criterion_regression(output_reg, y_reg)
            loss_target = criterion_binary(output_target, y_target)
            loss_failure = criterion_multiclass(output_failure, y_failure)

            # --- Total loss ---
            loss = loss_reg + loss_target + loss_failure
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/20 - Loss: {total_loss:.4f}")              # Update weights
    torch.save(model.state_dict(),model_path)
    print("Model trained and saved.")

# Inverse scaling of predictions
# def inverse_transform(predictions):
#     # Define feature groups
#     standard_features = ["Torque", "Process temperature", "Air temperature"]
#     robust_features = ["Rotational speed", "Tool wear"]

#     # Load dataset (to get scaling parameters)
#     df = pd.read_csv('assets/predictive_maintenance_updated.csv')
#     # Drop unnecessary columns
#     df.drop(['Timestamp'], axis=1, inplace=True,errors='ignore')

#     # Rename columns for consistency
#     df = df.rename(columns={
#         'Air temperature [K]': 'Air temperature',
#         'Process temperature [K]': 'Process temperature',
#         'Rotational speed [rpm]': 'Rotational speed',
#         'Torque [Nm]': 'Torque', 
#         'Tool wear [min]': 'Tool wear'
#     })

#     # Drop misclassified instances
#     df = df[~((df['Target'] == 1) & (df['Failure Type'] == 'No Failure'))]
#     df = df[~((df['Target'] == 0) & (df['Failure Type'] == 'Random Failures'))]
#     df.reset_index(drop=True, inplace=True)

#     # Drop duplicates and handle missing values
#     df = df.drop_duplicates().dropna()



#     # Fit scalers on original data
#     scaler_standard = StandardScaler()
#     scaler_robust = RobustScaler()
#     scaler_standard.fit(df[standard_features])
#     scaler_robust.fit(df[robust_features])
#     encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
#     encoded = encoder.fit_transform(df[["Failure Type"]])
#     encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(["Failure Type"]))
#     df = pd.concat([df.drop("Failure Type", axis=1), encoded_df], axis=1)

#     # Convert predictions to DataFrame
#     pred_df = pd.DataFrame(predictions, columns=standard_features + robust_features + ["Target"] + encoder.get_feature_names_out(["Failure Type"]).tolist())
#     # Apply inverse transformations
#     pred_df[standard_features] = scaler_standard.inverse_transform(pred_df[standard_features])
#     pred_df[robust_features] = scaler_robust.inverse_transform(pred_df[robust_features])

#     failure_type_cols = encoder.get_feature_names_out(["Failure Type"])
#     pred_df["Failure Type"] = encoder.inverse_transform(pred_df[failure_type_cols].values).ravel()

#     pred_df = pred_df[features]
#     # Reorder standard features: [Air temperature, Process temperature, Torque]
#     return pred_df



def inverse_transform(predictions):
    # Define feature groups
    standard_features = ["Torque", "Process temperature", "Air temperature"]
    robust_features = ["Rotational speed", "Tool wear"]

    # Load dataset (to get scaler params)
    df = pd.read_csv('assets/predictive_maintenance_updated.csv')
    df.drop(['Timestamp'], axis=1, inplace=True, errors='ignore')

    # Rename columns for consistency
    df = df.rename(columns={
        'Air temperature [K]': 'Air temperature',
        'Process temperature [K]': 'Process temperature',
        'Rotational speed [rpm]': 'Rotational speed',
        'Torque [Nm]': 'Torque',
        'Tool wear [min]': 'Tool wear'
    })

    # Clean and prepare data
    df = df[~((df['Target'] == 1) & (df['Failure Type'] == 'No Failure'))]
    df = df[~((df['Target'] == 0) & (df['Failure Type'] == 'Random Failures'))]
    df = df.drop_duplicates().dropna().reset_index(drop=True)

    # Fit scalers
    scaler_standard = StandardScaler()
    scaler_robust = RobustScaler()
    scaler_standard.fit(df[standard_features])
    scaler_robust.fit(df[robust_features])

    # Fit encoder
    encoder.fit(df[["Failure Type"]])
    encoded = encoder.transform(df[["Failure Type"]])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(["Failure Type"]))
    df = pd.concat([df.drop("Failure Type", axis=1), encoded_df], axis=1)

    # Prepare prediction DataFrame
    failure_type_cols = encoder.get_feature_names_out(["Failure Type"])
    column_names = standard_features + robust_features + ["Target"] + list(failure_type_cols)
    pred_df = pd.DataFrame(predictions, columns=column_names)

    # Inverse scale
    pred_df[standard_features] = scaler_standard.inverse_transform(pred_df[standard_features])
    pred_df[robust_features] = scaler_robust.inverse_transform(pred_df[robust_features])

    # Inverse one-hot encode
    pred_df["Failure Type"] = encoder.inverse_transform(pred_df[failure_type_cols].values).ravel()

    # Final column ordering
    final_columns = standard_features + robust_features + ["Target", "Failure Type"]
    pred_df = pred_df[final_columns]

    return pred_df
# Model evaluation and prediction
model.eval()
print("MODEL EVAL \n",model.eval())
predictions = model(X_test).cpu().detach().numpy()  # Generate predictions on the test set
predictions_original = inverse_transform(predictions)  # Inverse scale the predictions
# Inverse transform actual values
actual_original = inverse_transform(y_test.cpu().numpy())

# Print first few rows of actual values
print("First five actual values:\n", actual_original.head())

print("First five predictions:\n",predictions_original.head())  # Display first few predicted rows

# Trend analysis for predicted and actual values
def trend_analysis(actual, predicted):
    actual_df = pd.DataFrame(actual, columns=features)
    predicted_df = pd.DataFrame(predicted, columns=features)
    trends = {}
    trend_text = ""
    for col in features:
    if col not in ['Target']:
        actual_trend = float(np.polyfit(range(len(actual_df[col])), actual_df[col], 1)[0])
        predicted_trend = float(np.polyfit(range(len(predicted_df[col])), predicted_df[col], 1)[0])
        trends[col] = (actual_trend, predicted_trend)
        trend_text += f"{col}:\n Actual Trend:{'Increasing' if actual_trend > 0 else 'Decreasing' if actual_trend < 0 else 'Stable'}, Predicted Trend: {'Increasing' if predicted_trend > 0 else 'Decreasing' if predicted_trend < 0 else 'Stable'}\n"
    return trends, trend_text


def describe_trend(slope):
    if slope > 0.0001:
        return f"Increasing 📈 (Rate: +{slope:.6f})"
    elif slope < -0.0001:
        return f"Decreasing 📉 (Rate: {slope:.6f})"
    else:
        return f"Stable ➖ (Rate: {slope:.6f})"

def get_readable_label(col, label_encoder):
    if "Failure Type" in col and "_" in col:
        try:
            index = int(col.split("_")[-1])
            return f"Failure Type: {label_encoder.categories_[0][index]}"
        except (IndexError, ValueError):
            return col
    return col

def trend_analysis_only_predicted(actual, predicted, features, label_encoder=None):
    predicted_df = pd.DataFrame(predicted, columns=features)
    trend_descriptions = "\n=== 🔮 Predicted Trend Summary ===\n"

    for col in features:
        slope = float(np.polyfit(range(len(predicted_df[col])), predicted_df[col], 1)[0])
        label = get_readable_label(col, label_encoder) if label_encoder else col
        trend_descriptions += f"{label} → {describe_trend(slope)}\n"

    return trend_descriptions

print(trend_analysis_only_predicted(y_test.cpu().numpy(),predictions,features))
# Perform trend analysis
trends, trend_text = trend_analysis(y_test.cpu().numpy(), predictions)
# print("\nTrend Analysis (Actual vs Predicted slope):\n")
# for feature, (actual_trend, predicted_trend) in trends.items():
#     print(f"{feature}:\t Actual Trend = {actual_trend:.6f} \t Predicted Trend = {predicted_trend:.6f}")



# Generate predictions
# with torch.no_grad():
#     predictions = model(X_test).cpu().numpy()  # Convert to NumPy for evaluation
#     actual = y_test.cpu().numpy()  # Convert ground truth to NumPy

# # Compute evaluation metrics
# mae = mean_absolute_error(actual, predictions)
# mse = mean_squared_error(actual, predictions)
# rmse = np.sqrt(mse)
# r2 = r2_score(actual, predictions)

# # Print evaluation results
# print(f"Evaluation Metrics:")
# print(f"Mean Absolute Error (MAE): {mae:.4f}")
# print(f"Mean Squared Error (MSE): {mse:.4f}")
# print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
# print(f"R-squared (R²): {r2:.4f}")

# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

# with torch.no_grad():
#     model.eval()
#     predictions = model(X_test)

#     # ----- Ground truth -----
#     y_target_true = y_test[:, 5].cpu().numpy()  # Binary target
#     y_failure_true = torch.argmax(y_test[:, 6:], dim=1).cpu().numpy()  # Multiclass target
#     y_reg_true = y_test[:, :5].cpu().numpy()  # Regression ground truth

#     # ----- Predicted outputs -----
#     pred_reg = predictions[:, :5]  # Regression
#     pred_target = predictions[:, 5]  # Binary (logits)
#     pred_failure = predictions[:, 6:]  # Multiclass (logits)

#     # --- Binary Classification: Target ---
#     pred_target_prob = torch.sigmoid(pred_target)
#     pred_target_label = (pred_target_prob > 0.5).float().cpu().numpy()

#     acc_target = accuracy_score(y_target_true, pred_target_label)
#     f1_target = f1_score(y_target_true, pred_target_label)
#     precision_target = precision_score(y_target_true, pred_target_label)
#     recall_target = recall_score(y_target_true, pred_target_label)

#     print("=== Binary Classification (Target) ===")
#     print(f"Accuracy: {acc_target:.4f}")
#     print(f"F1 Score: {f1_target:.4f}")
#     print(f"Precision: {precision_target:.4f}")
#     print(f"Recall: {recall_target:.4f}\n")

#     # --- Multiclass Classification: Failure Type ---
#     pred_failure_labels = torch.argmax(pred_failure, dim=1).cpu().numpy()
#     print("=== Multiclass Classification (Failure Type) ===")
#     print(classification_report(y_failure_true, pred_failure_labels, target_names=encoder.categories_[0]))

#     # --- Regression Metrics ---
#     pred_reg_np = pred_reg.cpu().numpy()

#     mae = mean_absolute_error(y_reg_true, pred_reg_np)
#     mse = mean_squared_error(y_reg_true, pred_reg_np)
#     rmse = np.sqrt(mse)
#     r2 = r2_score(y_reg_true, pred_reg_np)

#     print("=== Regression (Sensor Features) ===")
#     print(f"MAE: {mae:.4f}")
#     print(f"MSE: {mse:.4f}")
#     print(f"RMSE: {rmse:.4f}")
#     print(f"R²: {r2:.4f}")



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
