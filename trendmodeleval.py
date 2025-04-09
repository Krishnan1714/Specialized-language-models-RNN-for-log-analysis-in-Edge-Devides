import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, classification_report
from trendmodel import model, X_test, y_test, encoder

model.eval()
with torch.no_grad():
    predictions = model(X_test)

    y_target_true = y_test[:, 5].cpu().numpy()
    y_failure_true = torch.argmax(y_test[:, 6:], dim=1).cpu().numpy()
    y_reg_true = y_test[:, :5].cpu().numpy()

    pred_reg = predictions[:, :5]
    pred_target = predictions[:, 5]
    pred_failure = predictions[:, 6:]

    pred_target_prob = torch.sigmoid(pred_target)
    pred_target_label = (pred_target_prob > 0.5).float().cpu().numpy()

    acc_target = accuracy_score(y_target_true, pred_target_label)
    f1_target = f1_score(y_target_true, pred_target_label)
    precision_target = precision_score(y_target_true, pred_target_label)
    recall_target = recall_score(y_target_true, pred_target_label)

    print("=== Binary Classification (Target) ===")
    print(f"Accuracy: {acc_target:.4f}")
    print(f"F1 Score: {f1_target:.4f}")
    print(f"Precision: {precision_target:.4f}")
    print(f"Recall: {recall_target:.4f}\n")

    pred_failure_labels = torch.argmax(pred_failure, dim=1).cpu().numpy()
    print("=== Multiclass Classification (Failure Type) ===")
    labels_present = np.unique(np.concatenate([y_failure_true, pred_failure_labels]))
print(classification_report(
    y_failure_true, 
    pred_failure_labels, 
    labels=labels_present, 
    target_names=[encoder.categories_[0][i] for i in labels_present],
    zero_division=0
))

pred_reg_np = pred_reg.cpu().numpy()
mae = mean_absolute_error(y_reg_true, pred_reg_np)
mse = mean_squared_error(y_reg_true, pred_reg_np)
rmse = np.sqrt(mse)
r2 = r2_score(y_reg_true, pred_reg_np)

print("=== Regression (Sensor Features) ===")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
