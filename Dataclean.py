
# Scaling Data
scaler_robust = RobustScaler()
df[['Rotational speed', 'Torque']] = scaler_robust.fit_transform(df[['Rotational speed', 'Torque']])

scaler_minmax = MinMaxScaler()
df[['Air temperature', 'Process temperature', 'Tool wear']] = scaler_minmax.fit_transform(df[['Air temperature', 'Process temperature', 'Tool wear']])

# Save cleaned dataset
df.to_csv('cleaned_data.csv', index=False)
