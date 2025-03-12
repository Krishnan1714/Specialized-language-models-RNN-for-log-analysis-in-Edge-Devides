import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch

# ------------------------------
# 1. Check GPU Availability
# ------------------------------
 

# ------------------------------
# 2. Load Dataset from CSV File
# ------------------------------
# Even though the file extension is ".csv.xls", we're reading it as CSV.
file_path = "assets/Synthetic_Power_Management_IoT_Dataset.csv"
df = pd.read_csv(file_path, encoding='utf-8-sig')

# ------------------------------
# 3. Display Column Names
# ------------------------------
print("Column names:", df.columns.tolist())
# Expected columns:
# ['Timestamp', 'Battery_Voltage (V)', 'Battery_Current (A)', 'Charge_Cycles',
#  'Discharge_Cycles', 'Solar_Panel_Voltage (V)', 'Solar_Panel_Current (A)',
#  'Power_Distribution_Unit_Load (W)', 'Power_Efficiency (%)', 'Temperature (Â°C)', 'Fault_Flag']

# ------------------------------
# 4. Prepare Data for Fault Prediction
# ------------------------------
# Define features (excluding Timestamp) and the target column.
features = [
    "Battery_Voltage (V)", "Battery_Current (A)", "Charge_Cycles", "Discharge_Cycles",
    "Solar_Panel_Voltage (V)", "Solar_Panel_Current (A)",
    "Power_Distribution_Unit_Load (W)", "Power_Efficiency (%)",
    "Temperature (°C)"
]
target = "Fault_Flag"

if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found. Found columns: {df.columns.tolist()}")

# ------------------------------
# 5. Split the Data into Training and Testing Sets
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df[target], test_size=0.2, random_state=42
)

# ------------------------------
# 6. Train the Fault Prediction Model using XGBoost
# ------------------------------
# Using the new recommended parameters:
# - tree_method="hist" with device set to "cuda" for GPU training.
# - predictor set to "gpu_predictor" if GPU is available.
fault_model = xgb.XGBClassifier(
    n_estimators=50,         # Lightweight: fewer trees
    max_depth=3,             # Shallow trees for fast inference
    learning_rate=0.1,
    tree_method="hist",      # Use CPU-based hist method but set device for GPU training
    device="cuda" if use_gpu else "cpu",  # Use 'cuda' if GPU is available
    predictor="gpu_predictor" if use_gpu else "auto",  # Use GPU predictor if available
    verbosity=1
)

fault_model.fit(X_train, y_train)

# ------------------------------
# 7. Evaluate the Model
# ------------------------------
y_pred = fault_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# ------------------------------
# 8. Save the Model for IoT Deployment
# ------------------------------
fault_model.save_model("fault_prediction_xgboost.json")
print("Model saved as fault_prediction_xgboost.json")
