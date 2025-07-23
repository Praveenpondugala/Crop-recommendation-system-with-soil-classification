import pandas as pd
import joblib

def load_data(file_path):
    """Load dataset from a CSV file."""
    return pd.read_csv("/Users/harshithyvs/Desktop/crop_rotation_dataset.csv")

def save_model(model, filename):
    """Save trained model using joblib."""
    joblib.dump(model, filename)

def load_model(filename):
    """Load trained model using joblib."""
    return joblib.load(filename)
