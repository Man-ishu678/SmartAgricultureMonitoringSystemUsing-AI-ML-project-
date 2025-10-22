# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("/Users/manishkumar/Documents/Agrisense/Crop_recommendation.csv")
  # Make sure this file is in the same folder
print("Dataset Loaded Successfully!\n")

# Display first few rows
print(data.head())

# Check for missing values
print("\nMissing Values in Dataset:")
print(data.isnull().sum())

# Split data into features (X) and label (y)
X = data.drop('label', axis=1)  # Features: N, P, K, temperature, humidity, ph, rainfall
y = data['label']              # Target: crop label

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Function to predict crop based on new sensor readings
def predict_crop(sensor_readings):
    sensor_readings = np.array(sensor_readings).reshape(1, -1)
    sensor_readings = scaler.transform(sensor_readings)
    prediction = model.predict(sensor_readings)
    return prediction[0]

# Example usage:
# Format: N, P, K, temperature, humidity, ph, rainfall
new_sensor_readings = [90, 42, 43, 20.87, 82.02, 6.5, 202.93]
predicted_crop = predict_crop(new_sensor_readings)
print(f"\nPredicted Crop: {predicted_crop}")
