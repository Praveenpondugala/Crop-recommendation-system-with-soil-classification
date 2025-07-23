import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import re

# Step 1: Load the Generated Dataset
df = pd.read_csv('crop_recommendation_dataset.csv')
print("Dataset loaded:")
print(df.head())

# Step 2: Preprocessing
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

# Step 5: Scale Features (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train Logistic Regression Model
lr_model = LogisticRegression(
    multi_class='multinomial',  # For multi-class classification
    solver='lbfgs',             # Suitable for multinomial logistic regression
    max_iter=1000,              # Ensure convergence
    random_state=42
)
lr_model.fit(X_train_scaled, y_train)

# Step 7: Make Predictions
y_pred = lr_model.predict(X_test_scaled)

# Step 8: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le_y.classes_))

# Step 9: Feature Coefficients (analogous to feature importance)
# For multi-class, coefficients are per class; we'll average their absolute values
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient (Avg Abs)': np.mean(np.abs(lr_model.coef_), axis=0)
}).sort_values(by='Coefficient (Avg Abs)', ascending=False)
print("\nTop 10 Feature Coefficients (Average Absolute Value):")
print(coef_df.head(10))

# Step 10: Example Prediction
sample_input = X_test_scaled[0].reshape(1, -1)
predicted_crop = le_y.inverse_transform(lr_model.predict(sample_input))[0]
print(f"\nPredicted Crop for sample input: {predicted_crop}")