# Crop Recommendation System with Soil Classification

## 🌱 Overview

This is an AI-powered decision-support system designed to help farmers optimize agricultural practices by integrating **CNN-based soil classification** and **traditional crop rotation datasets**. This hybrid model provides accurate soil type classification and recommends the best crops based on past cropping patterns, current season, and soil characteristics.

## 🚀 Features

- **Automated Soil Classification**: Uses a CNN model to classify soil types from uploaded images.
- **Crop Recommendation System**: Suggests the best crops based on soil type, month, region, and previous crops.
- **Seamless UI**: Built with Streamlit, offering an intuitive and user-friendly experience.
- **Data-Driven Insights**: Incorporates historical crop rotation data to optimize yield and sustainability.

## 🛠️ Technologies Used

- **Machine Learning**: TensorFlow (CNN for soil classification), Scikit-learn (Crop recommendation)
- **Python Libraries**: Pandas, NumPy, Joblib, Matplotlib, Seaborn
- **Web Framework**: Streamlit

## 🖼️ Usage

1. **Upload a Soil Image**: The CNN model classifies the soil type.
2. **Select Crop Parameters**: Choose the current month, region, and last harvested crop.
3. **Get Recommendations**: The system predicts the best crop for optimal yield.

## 📊 Dataset Details

The crop rotation dataset contains:

- **Soil Types**: Alluvial, Black, Laterite, Red
- **Month-wise Crop Data**
- **Region-specific Recommendations**
- **Temperature & Rainfall Requirements**

## 🎯 Future Enhancements

- Integrating real-time weather data for dynamic recommendations.
- Adding support for **fertilizer suggestions** based on soil quality.
- Extending crop recommendation with **pest control measures**.

