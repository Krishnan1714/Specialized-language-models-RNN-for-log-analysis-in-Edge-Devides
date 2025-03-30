import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import torch
import trendmodel as tmd

# Load the original and predicted data
original = pd.read_csv("cleaned_data_scaled.csv")
predicted = pd.read_csv("predicted_data.csv")

# Inverse transform the predicted data to get original scale
predicted_original = tmd. inverse_transform(predicted.values)

# Calculate Mean Squared Error (MSE)
def calculate_mse(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return mse

# Calculate Root Mean Squared Error (RMSE)
def calculate_rmse(mse):
    return np.sqrt(mse)

# Calculate Dynamic Time Warping (DTW) distance
def calculate_dtw(actual, predicted):
    distance, _ = fastdtw(actual, predicted, dist=euclidean)
    return distance

print(f"Shape of actual: {tmd.y_test.cpu().numpy().shape}")
print(f"Shape of predicted: {tmd.predictions.shape}")

# Evaluate the model
mse = calculate_mse(original[tmd.features].values, predicted_original.values)
rmse = calculate_rmse(mse)
dtw_distance = calculate_dtw(original[tmd.features].values, predicted_original.values)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Dynamic Time Warping (DTW) Distance: {dtw_distance}")
