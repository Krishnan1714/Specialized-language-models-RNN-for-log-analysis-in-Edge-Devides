import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, RobustScaler

# Load Data
df = pd.read_csv('assets/predictive_maintenance.csv')

# Drop unnecessary columns
df.drop(['UDI', 'Product ID'], axis=1, inplace=True)

# Rename columns
df = df.rename(columns={
    'Air temperature [K]': 'Air temperature',
    'Process temperature [K]': 'Process temperature',
    'Rotational speed [rpm]': 'Rotational speed',
    'Torque [Nm]': 'Torque', 
    'Tool wear [min]': 'Tool wear'
})

# Fix inconsistent target labels
df_failure = df[df['Target'] == 1]
df_no_failure = df[df['Target'] == 0]

# Remove misclassified instances
df = df[~((df['Target'] == 1) & (df['Failure Type'] == 'No Failure'))]
df = df[~((df['Target'] == 0) & (df['Failure Type'] == 'Random Failures'))]

# Reset index
df.reset_index(inplace=True, drop=True)

# Ordinal Encoding for categorical variables
ord_enc = OrdinalEncoder(categories=[['L', 'M', 'H'], df['Failure Type'].unique().tolist()])
encoded_features = ord_enc.fit_transform(df[['Type', 'Failure Type']])

df.drop(['Type', 'Failure Type'], axis=1, inplace=True)
df[['Type', 'Failure Type']] = pd.DataFrame(encoded_features, index=df.index)

# Scaling Data
scaler_robust = RobustScaler()
df[['Rotational speed', 'Torque']] = scaler_robust.fit_transform(df[['Rotational speed', 'Torque']])

scaler_minmax = MinMaxScaler()
df[['Air temperature', 'Process temperature', 'Tool wear']] = scaler_minmax.fit_transform(df[['Air temperature', 'Process temperature', 'Tool wear']])

# Save cleaned dataset
df.to_csv('cleaned_data.csv', index=False)
