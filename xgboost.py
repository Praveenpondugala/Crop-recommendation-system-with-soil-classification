import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import re

# Step 2: Preprocessing
# Load the dataset
df = pd.read_csv('crop_recommendation_dataset.csv')
print(df.head())

# Parse ranges
def parse_range(value, unit):
    numbers = re.findall(r'\d+', value)
    return int(numbers[0]), int(numbers[1])

df[['Rainfall_Min', 'Rainfall_Max']] = df['Rainfall_Requirement'].apply(lambda x: pd.Series(parse_range(x, 'mm')))
df[['Temp_Min', 'Temp_Max']] = df['Temperature_Range'].apply(lambda x: pd.Series(parse_range(x, '°C')))
df['Growth_Duration'] = df['Growth_Duration'].str.replace(' days', '').astype(int)

# New derived features
df['Rainfall_Range'] = df['Rainfall_Max'] - df['Rainfall_Min']
df['Temp_Range'] = df['Temp_Max'] - df['Temp_Min']
df['Temp_Avg'] = (df['Temp_Min'] + df['Temp_Max']) / 2

# Seasonal encoding
month_to_season = {
    'March': 'Spring', 'April': 'Spring', 'May': 'Spring',
    'June': 'Monsoon', 'July': 'Monsoon', 'August': 'Monsoon',
    'September': 'Autumn', 'October': 'Autumn', 'November': 'Autumn',
    'December': 'Winter', 'January': 'Winter', 'February': 'Winter'
}
df['Season_Planted'] = df['Month_Planted'].apply(lambda x: month_to_season[x.split('-')[0]])
df['Season_Harvested'] = df['Month_Harvested'].apply(lambda x: month_to_season[x.split('-')[0]])

# Crop rotation feature
df['Rotation_Pair'] = df['Previous_Crop'] + '_' + df['Next_Crop']

# Encode categorical variables
categorical_cols = ['Soil_Type', 'Previous_Crop', 'Next_Crop', 'Season_Planted', 'Season_Harvested', 'Region', 'Rotation_Pair']
df_encoded = pd.get_dummies(df[categorical_cols], drop_first=True)
df = pd.concat([df.drop(categorical_cols + ['Rainfall_Requirement', 'Temperature_Range', 'Month_Planted', 'Month_Harvested'], axis=1), df_encoded], axis=1)

# Step 3: Define Features and Target
X = df.drop(columns=['Crop_Name'])
y = df['Crop_Name']

# Encode target
le_y = LabelEncoder()
y_encoded = le_y.fit_transform(y)

# Step 4: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Step 5: Train XGBoost Model
xgb_model = XGBClassifier(
    objective='multi:softmax',
    num_class=len(y.unique()),
    learning_rate=0.1,
    n_estimators=200,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss'
)
xgb_model.fit(X_train, y_train)

# Step 6: Make Predictions
y_pred = xgb_model.predict(X_test)

# Step 7: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le_y.classes_))

# Step 8: Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importance (Top 10):")
print(feature_importance.head(10))

# Step 9: Example Prediction
sample_input = X_test.iloc[0].values.reshape(1, -1)
predicted_crop = le_y.inverse_transform(xgb_model.predict(sample_input))[0]
print(f"\nPredicted Crop for sample input: {predicted_crop}")