# app.py
from flask import Flask, render_template, request
import numpy as np
import joblib

# Flask App
app = Flask(__name__)

# Load model and scaler
model = joblib.load('behavioral_authentication_model1.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        typing_speed = float(request.form['typing_speed'])
        key_press_duration = float(request.form['key_press_duration'])
        typing_rhythm_variance = float(request.form['typing_rhythm_variance'])
        mouse_movement_speed = float(request.form['mouse_movement_speed'])
        mouse_click_interval = float(request.form['mouse_click_interval'])
        swipe_pressure = float(request.form['swipe_pressure'])

        # Prepare input
        user_input = np.array([[typing_speed, key_press_duration, typing_rhythm_variance,
                                mouse_movement_speed, mouse_click_interval, swipe_pressure]])
        user_input_scaled = scaler.transform(user_input)

        # Predict
        prediction = model.predict(user_input_scaled)

        if prediction[0] == 1:
            result = "✅ Authorized User"
        else:
            result = "❌ Unauthorized User"

        return render_template('result.html', prediction=result)
    except Exception as e:
        return f"⚠️ Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
