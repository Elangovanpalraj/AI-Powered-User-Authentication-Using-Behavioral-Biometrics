import pandas as pd

# Dataset Load
df = pd.read_csv('behavioral_biometrics_dataset.csv')

# First 5 rows பார்க்கலாம்
print(df.head())

# ==================================================================================

# Feature columns
X = df.drop('user_id', axis=1)

# Target column
y = df['user_id']

# ========================================================================================

from sklearn.model_selection import train_test_split

# Data split: 80% Training, 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("✅ Data split ஆனது!")

#  ===============================================================================================

from sklearn.ensemble import RandomForestClassifier

# Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Model Train பண்ணல்
model.fit(X_train, y_train)

print("✅ Model Successfully Train ஆனது!")

# =============================================================================================
from sklearn.metrics import accuracy_score, classification_report

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2f}")

# Classification Report
print(classification_report(y_test, y_pred))
# ================================================================================================


