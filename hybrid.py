import streamlit as st
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image # type: ignore
from utils import load_model, load_data

# Load Models
soil_model = tf.keras.models.load_model("/Users/harshithyvs/Desktop/Projects/MarghaDharshi/Soil_Module/models/soil_classifier_model.h5")
crop_model = load_model("/Users/harshithyvs/Desktop/Projects/MarghaDharshi/models/crop_recommendation_model.pkl")

# Load Label Encoders & Dataset
data_path = "/Users/harshithyvs/Desktop/crop_rotation_dataset.csv"
df = load_data(data_path)
label_encoders = joblib.load("/Users/harshithyvs/Desktop/Projects/MarghaDharshi/models/label_encoders.pkl")
model_features = joblib.load("/Users/harshithyvs/Desktop/Projects/MarghaDharshi/models/model_features.pkl")

# Define Soil Classes
SOIL_CLASSES = ['Alluvial', 'Black', 'Laterite', 'Red']
IMG_SIZE = (128, 128)

# UI Layout
st.set_page_config(layout="wide")
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: bold;
            color: #2E7D32;
        }
        .sub-title {
            text-align: center;
            font-size: 1.2rem;
            color: #555;
        }
        .uploaded-image {
            display: flex;
            justify-content: center;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🌱 MarghaDharshi - Smart Crop Advisor</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>A Hybrid ML Model for Automated Soil Classification and Crop Planning</div>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("Upload Soil Image")
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility='hidden')
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        img = image.load_img(uploaded_file, target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        soil_prediction = soil_model.predict(img_array)
        soil_type = SOIL_CLASSES[np.argmax(soil_prediction)]
        confidence = np.max(soil_prediction) * 100
        
        st.success(f"**Soil Type:** {soil_type} ({confidence:.2f}%)")

with col2:
    st.subheader("Crop Recommendation")
    
    if uploaded_file:
        month = st.selectbox("Current Month", df["Month_Planted"].unique().tolist())
        region = st.selectbox("Region", df["Region"].unique().tolist())
        last_crop = st.selectbox("Last Crop Harvested", df["Previous_Crop"].unique().tolist())
        
        if st.button("Recommend Crop", use_container_width=True):
            input_data = pd.DataFrame([[soil_type, month, region, last_crop]],
                                      columns=["Soil_Type", "Month_Planted", "Region", "Previous_Crop"])
            
            for col in input_data.columns:
                if col in label_encoders:
                    input_data[col] = label_encoders[col].transform(input_data[col])
            
            missing_cols = set(model_features) - set(input_data.columns)
            for col in missing_cols:
                input_data[col] = 0
            input_data = input_data.reindex(columns=model_features, fill_value=0)
            
            prediction = crop_model.predict(input_data)
            recommended_crop = label_encoders['Next_Crop'].inverse_transform([prediction[0]])[0]
            
            crop_info = df[df["Next_Crop"] == recommended_crop][["Temperature_Range", "Rainfall_Requirement"]].dropna().iloc[0]
            
            st.success(f"🌾 Recommended Crop: {recommended_crop}")
            st.success(f"🌡️ **Temperature:** {crop_info['Temperature_Range']}°C")
            st.success(f"🌧️ **Rainfall:** {crop_info['Rainfall_Requirement']} mm")
