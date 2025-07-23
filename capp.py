import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_model, load_data

# Load trained model
model = load_model("/Users/harshithyvs/Desktop/Projects/MarghaDharshi/models/crop_recommendation_model.pkl")

# Load label encoders
label_encoders = joblib.load("/Users/harshithyvs/Desktop/Projects/MarghaDharshi/models/label_encoders.pkl")

# Load dataset for dropdown options
data_path = "/Users/harshithyvs/Desktop/Projects/MarghaDharshi/data/crop_rotation_dataset.csv"
df = load_data(data_path)

# Extract unique values for dropdowns
soil_types = df["Soil_Type"].unique().tolist()
months = df["Month_Planted"].unique().tolist()
regions = df["Region"].unique().tolist()
crops = df["Previous_Crop"].unique().tolist()

# Streamlit UI
st.title("🌱 MarghaDharshi - Crop Recommendation")

st.sidebar.header("User Input")
soil_type = st.sidebar.selectbox("Select Soil Type", soil_types)
month = st.sidebar.selectbox("Current Month", months)
region = st.sidebar.selectbox("Region", regions)
last_crop = st.sidebar.selectbox("Last Crop Harvested", crops)

# Convert input to model format
input_data = pd.DataFrame([[soil_type, month, region, last_crop]], 
                          columns=["Soil_Type", "Month_Planted", "Region", "Previous_Crop"])

# Apply label encoding using saved encoders
for col in ["Soil_Type", "Month_Planted", "Region", "Previous_Crop"]:
    if col in label_encoders:
        input_data[col] = label_encoders[col].transform(input_data[col])

# Load model feature names to ensure correct input
model_features = joblib.load("/Users/harshithyvs/Desktop/Projects/MarghaDharshi/models/model_features.pkl")

# Align features
missing_cols = set(model_features) - set(input_data.columns)
for col in missing_cols:
    input_data[col] = 0  # Add missing features with default value

input_data = input_data.reindex(columns=model_features, fill_value=0)

# Predict Crop
if st.sidebar.button("Predict Crop"):
    prediction = model.predict(input_data)
    recommended_crop = label_encoders['Next_Crop'].inverse_transform([prediction[0]])[0]
    
    # Extract temperature and rainfall requirements
    crop_info = df[df["Next_Crop"] == recommended_crop][["Temperature_Range", "Rainfall_Requirement"]].dropna().iloc[0]
    temperature = crop_info["Temperature_Range"]
    rainfall = crop_info["Rainfall_Requirement"]
    
    st.success(f"🌾 Recommended Crop: {recommended_crop}\n\n🌡️ Required Temperature: {temperature}°C\n\n🌧️ Required Rainfall: {rainfall} mm")