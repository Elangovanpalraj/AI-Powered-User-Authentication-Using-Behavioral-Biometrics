import numpy as np
import pandas as pd

# பயனரின் typing மற்றும் mouse data மாதிரி உருவாக்கல்

# எத்தனை மாதிரி data (rows) தேவை என்று குறிப்பிடுங்கள்
num_samples = 1000

# Data உருவாக்குதல்
data = {
    'typing_speed': np.random.normal(40, 5, num_samples),  # Words per minute
    'key_press_duration': np.random.normal(150, 20, num_samples),  # milliseconds
    'typing_rhythm_variance': np.random.normal(10, 2, num_samples),
    'mouse_movement_speed': np.random.normal(300, 50, num_samples),  # pixels per second
    'mouse_click_interval': np.random.normal(500, 100, num_samples),  # milliseconds
    'swipe_pressure': np.random.normal(0.5, 0.1, num_samples),  # normalized 0 to 1
    'user_id': np.random.choice([0, 1], num_samples)  # 0 = Unauthorized User, 1 = Authorized User
}

# DataFrame உருவாக்குதல்
df = pd.DataFrame(data)

# Dataset save செய்யலாம்
df.to_csv('behavioral_biometrics_dataset.csv', index=False)

print("✅ Dataset உருவாக்கப்பட்டது! ('behavioral_biometrics_dataset.csv' ஆக சேமிக்கப்பட்டது)")
df.head()
