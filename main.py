# ================================================================
# 1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
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

print("✅ Synthetic dataset created successfully!")
print(df.head())

# ================================================================

# 3. Feature and Target Split
X = df.drop('user_id', axis=1)
y = df['user_id']

# 4. Data Standardization (Very Important for Behavioral Biometrics)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================================================================

# 5. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("✅ Data split successfully!")

# ================================================================

# 6. Random Forest Classifier
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print("✅ Model trained successfully!")

# ================================================================

# 7. Predictions and Evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2f}")

print(classification_report(y_test, y_pred))

# ================================================================

# 8. Save the Model (Optional)
import joblib
joblib.dump(model, 'behavioral_authentication_model.pkl')
print("✅ Model saved as 'behavioral_authentication_model.pkl'")
