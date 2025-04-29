# ================================================================
# 1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
# ================================================================

# 2. Synthetic Dataset Generation (Correct Data)

# Random data generation - 1000 users (500 authorized, 500 unauthorized)
np.random.seed(42)

authorized = pd.DataFrame({
    'typing_speed': np.random.normal(45, 5, 500),
    'key_press_duration': np.random.normal(180, 10, 500),
    'typing_rhythm_variance': np.random.normal(8, 1, 500),
    'mouse_movement_speed': np.random.normal(320, 20, 500),
    'mouse_click_interval': np.random.normal(500, 30, 500),
    'swipe_pressure': np.random.normal(0.5, 0.05, 500),
    'user_id': np.ones(500)  # 1 - Authorized
})

unauthorized = pd.DataFrame({
    'typing_speed': np.random.normal(30, 5, 500),
    'key_press_duration': np.random.normal(130, 10, 500),
    'typing_rhythm_variance': np.random.normal(12, 2, 500),
    'mouse_movement_speed': np.random.normal(270, 30, 500),
    'mouse_click_interval': np.random.normal(450, 30, 500),
    'swipe_pressure': np.random.normal(0.4, 0.05, 500),
    'user_id': np.zeros(500)  # 0 - Unauthorized
})

# Combine datasets
df = pd.concat([authorized, unauthorized], axis=0).reset_index(drop=True)

# Save the dataset to a CSV file safely
csv_filename = "behavioral_data.csv"

# Only save if file does not exist
if not os.path.exists(csv_filename):
    df.to_csv(csv_filename, index=False)
    print(f"✅ Synthetic dataset created and saved as '{csv_filename}'!")
else:
    print(f"ℹ️ File '{csv_filename}' already exists, using the existing one.")

# ================================================================

# 3. Load Dataset from CSV
try:
    df = pd.read_csv(csv_filename)
    print("✅ Dataset loaded successfully!")
    print(df.head())
except Exception as e:
    print(f"⚠️ Error loading dataset: {e}")

# ================================================================

# 4. Feature and Target Split
X = df.drop('user_id', axis=1)
y = df['user_id']

# 5. Data Standardization (Very Important for Behavioral Biometrics)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================================================================

# 6. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("✅ Data split successfully!")

# ================================================================

# 7. Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# ================================================================

# 8. Predictions and Evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ================================================================

# 9. Save the Model and Scaler
model_filename = "behavioral_authentication_model1.pkl"
scaler_filename = "scaler.pkl"

joblib.dump(model, model_filename)
joblib.dump(scaler, scaler_filename)

print(f"✅ Model saved as '{model_filename}' and scaler saved as '{scaler_filename}'!")

# ================================================================

# 10. Real-Time Prediction System
print("\n🔐 Real-Time User Authentication System 🔐")
print("👇 கீழே உங்க behavior values-ஐ உள்ளிடுங்கள்:")

# Load the model and scaler again
model = joblib.load(model_filename)
scaler = joblib.load(scaler_filename)

try:
    typing_speed = float(input("Typing Speed (WPM): "))
    key_press_duration = float(input("Key Press Duration (ms): "))
    typing_rhythm_variance = float(input("Typing Rhythm Variance: "))
    mouse_movement_speed = float(input("Mouse Movement Speed (px/sec): "))
    mouse_click_interval = float(input("Mouse Click Interval (ms): "))
    swipe_pressure = float(input("Swipe Pressure (0.0 - 1.0): "))

    # Combine input and scale
    user_input = np.array([[typing_speed, key_press_duration, typing_rhythm_variance,
                            mouse_movement_speed, mouse_click_interval, swipe_pressure]])

    user_input_scaled = scaler.transform(user_input)

    # Prediction
    prediction = model.predict(user_input_scaled)

    # Result
    if prediction[0] == 1:
        print("\n✅ Prediction: Authorized User ✅")
    else:
        print("\n❌ Prediction: Unauthorized User ❌")

except Exception as e:
    print(f"⚠️ Error during prediction: {e}")

# ================================================================
