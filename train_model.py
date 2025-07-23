import os
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Ensure directories exist
model_dir = "/Users/harshithyvs/Desktop/VIT/MarghaDharshi/models"
os.makedirs(model_dir, exist_ok=True)

# Load dataset
data_path = "/Users/harshithyvs/Desktop/crop_rotation_dataset.csv"
df = pd.read_csv(data_path)

# Clean column names (strip spaces)
df.columns = df.columns.str.strip()

# Feature Selection
expected_features = ["Soil_Type", "Rainfall_Requirement", "Temperature_Range", "Previous_Crop", "Month_Planted", "Region"]
target = "Next_Crop"

# Check if all required columns exist
missing_features = [col for col in expected_features + [target] if col not in df.columns]
if missing_features:
    raise KeyError(f"Missing columns in dataset: {missing_features}")

# Function to convert range values to numeric (taking average)
def convert_range_to_numeric(value):
    if isinstance(value, str):
        value = value.replace("mm", "").replace("°C", "").strip()
        if "-" in value:
            low, high = map(float, value.split("-"))
            return (low + high) / 2  # Take the average
        return float(value)
    return value  # If already numeric, return as is

# Convert Rainfall & Temperature columns
df["Rainfall_Requirement"] = df["Rainfall_Requirement"].apply(convert_range_to_numeric)
df["Temperature_Range"] = df["Temperature_Range"].apply(convert_range_to_numeric)

# Encode categorical variables
categorical_columns = ["Soil_Type", "Previous_Crop", "Month_Planted", "Region", target]
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Convert categorical to numerical
    label_encoders[col] = le

# Save label encoders
joblib.dump(label_encoders, os.path.join(model_dir, "label_encoders.pkl"))

# Splitting dataset
X = df[expected_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save trained model
joblib.dump(model, os.path.join(model_dir, "crop_recommendation_model.pkl"))

# Save feature names for inference
joblib.dump(X.columns.tolist(), os.path.join(model_dir, "model_features.pkl"))

