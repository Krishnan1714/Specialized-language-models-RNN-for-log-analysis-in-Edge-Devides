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
import torch.onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("assets/cleaned_data_scaled.csv")

encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[["Failure Type"]])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(["Failure Type"]))
df = pd.concat([df.drop("Failure Type", axis=1), encoded_df], axis=1)

features = ["Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear", "Target"] + list(encoder.get_feature_names_out(["Failure Type"]))

standard_scaled = ["Air temperature", "Process temperature", "Torque"]
robust_scaled = ["Rotational speed", "Tool wear"]

def create_sequences(data, seq_length=20):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(df[features].values, seq_length=20)

ts = TimeSeriesSplit(n_splits=5)
train_idx, test_idx = list(ts.split(X))[-1]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

X_train, X_test = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(X_test, dtype=torch.float32).to(device)
y_train, y_test = torch.tensor(y_train, dtype=torch.float32).to(device), torch.tensor(y_test, dtype=torch.float32).to(device)

batch_size = 32
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, len(features))

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


model_path = "trained_lstm_model.pth"
model = LSTMModel(input_size=len(features)).to(device)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print("Model loaded from disk.")
else:
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

            output_reg = output[:, :5]
            output_target = output[:, 5]
            output_failure = output[:, 6:]

            y_reg = y_batch[:, :5]
            y_target = y_batch[:, 5]
            y_failure = torch.argmax(y_batch[:, 6:], dim=1)

            loss_reg = criterion_regression(output_reg, y_reg)
            loss_target = criterion_binary(output_target, y_target)
            loss_failure = criterion_multiclass(output_failure, y_failure)

            loss = loss_reg + loss_target + loss_failure
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/20 - Loss: {total_loss:.4f}")
    torch.save(model.state_dict(),model_path)
    print("Model trained and saved.")
    print("Trained model saved as 'trained_lstm_model.pth'")
    # Quantize the trained model and save it
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.LSTM}, dtype=torch.qint8
    )
    torch.save(quantized_model.state_dict(), "quantized_trendmodel.pth")
    print("Quantized model saved as 'quantized_trendmodel.pth'")


def inverse_transform(predictions):
    standard_features = ["Torque", "Process temperature", "Air temperature"]
    robust_features = ["Rotational speed", "Tool wear"]

    df = pd.read_csv('assets/predictive_maintenance_updated.csv')
    df.drop(['Timestamp'], axis=1, inplace=True, errors='ignore')

    df = df.rename(columns={
        'Air temperature [K]': 'Air temperature',
        'Process temperature [K]': 'Process temperature',
        'Rotational speed [rpm]': 'Rotational speed',
        'Torque [Nm]': 'Torque',
        'Tool wear [min]': 'Tool wear'
    })


    df = df[~((df['Target'] == 1) & (df['Failure Type'] == 'No Failure'))]
    df = df[~((df['Target'] == 0) & (df['Failure Type'] == 'Random Failures'))]
    df = df.drop_duplicates().dropna().reset_index(drop=True)

    scaler_standard = StandardScaler()
    scaler_robust = RobustScaler()
    scaler_standard.fit(df[standard_features])
    scaler_robust.fit(df[robust_features])

    encoder.fit(df[["Failure Type"]])
    encoded = encoder.transform(df[["Failure Type"]])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(["Failure Type"]))
    df = pd.concat([df.drop("Failure Type", axis=1), encoded_df], axis=1)

    failure_type_cols = encoder.get_feature_names_out(["Failure Type"])
    column_names = standard_features + robust_features + ["Target"] + list(failure_type_cols)

    if predictions.shape[1] < len(column_names):
        diff = len(column_names) - predictions.shape[1]
        predictions = np.hstack([predictions, np.zeros((predictions.shape[0], diff))])
    elif predictions.shape[1] > len(column_names):
        predictions = predictions[:, :len(column_names)]

    pred_df = pd.DataFrame(predictions, columns=column_names)
    pred_df[standard_features] = scaler_standard.inverse_transform(pred_df[standard_features])
    pred_df[robust_features] = scaler_robust.inverse_transform(pred_df[robust_features])
    pred_df["Failure Type"] = encoder.inverse_transform(pred_df[failure_type_cols].values).ravel()
    pred_df["Target"] = (pred_df["Target"] > 0.5).astype(int)
    final_columns = standard_features + robust_features + ["Target", "Failure Type"]
    pred_df = pred_df[final_columns]

    return pred_df

model.eval()
print("MODEL EVAL \n",model.eval())
predictions = model(X_test).cpu().detach().numpy()
predictions_original = inverse_transform(predictions)
actual_original = inverse_transform(y_test.cpu().numpy())

print("First five actual values:\n", actual_original.head())
print("First five predictions:\n",predictions_original.head())

def trend_analysis(actual, predicted):
    actual_df = pd.DataFrame(actual, columns=features)
    predicted_df = pd.DataFrame(predicted, columns=features)
    trends = {}
    trend_text = ""
    for col in features:        
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
    failure_labels = label_encoder.categories_[0] if label_encoder else []
    if "Failure Type" in col and "_" in col:
        try:
            index = int(col.split("_")[-1])
            label = failure_labels[index] if index < len(failure_labels) else "Unknown"
            return f"{col} → Failure Type: {label}"
            
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

print(trend_analysis_only_predicted(y_test.cpu().numpy(), predictions, features, label_encoder=encoder))
trends, trend_text = trend_analysis(y_test.cpu().numpy(), predictions)

def save_predictions_to_csv(predictions, filename="predicted_data.csv"):
    predictions_original = inverse_transform(predictions)
    if os.path.isfile(filename):
        predictions_original.to_csv(filename, mode='a', index=False, header=False)
    else:
        predictions_original.to_csv(filename, mode='w', index=False)
    print(f"Predictions appended to {filename}")

save_predictions_to_csv(predictions)



# Dummy input for ONNX export
dummy_input = torch.randn(1, 20, len(features)).to(device)

# Export original model
onnx_file_normal = "trendmodel.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_file_normal,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=13
)
print(f"✅ Exported original model to ONNX: {onnx_file_normal}")

# Quantize again (since your quantized_model is only inside else block)
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear, nn.LSTM}, dtype=torch.qint8
)

# Export quantized model


quantize_dynamic(
    model_input="trendmodel.onnx",
    model_output="quantized_trendmodel.onnx",
    weight_type=QuantType.QInt8
)

print("✅ Quantized ONNX model saved as 'quantized_trendmodel.onnx'")

