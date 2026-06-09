# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

# ML & audio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import librosa

# For display in notebook
from IPython.display import Audio, display



# If a dataset exists in /kaggle/input, try to load it automatically
dataset_path = None
for root, dirs, files in os.walk('/kaggle/input'):
    for f in files:
        if f.lower().endswith('.csv'):
            dataset_path = os.path.join(root, f)
            break
    if dataset_path:
        break

if dataset_path:
    print("Found dataset:", dataset_path)
    crime_df = pd.read_csv(dataset_path)
else:
    # Create a synthetic mock dataset
    np.random.seed(42)
    crime_df = pd.DataFrame({
        "crime_severity": np.random.randint(1, 6, 1000),   # 1-5
        "poor_lighting": np.random.randint(0, 2, 1000),    # 0/1
        "crowd_density": np.random.randint(0, 4, 1000),    # 0-3
        "timestamp": pd.date_range(start="2024-01-01", periods=1000, freq="H")
    })
    print("Using synthetic mock dataset (no CSV found).")

crime_df.head()



crime_df['hour'] = pd.to_datetime(crime_df['timestamp']).dt.hour
crime_df['risk_score'] = (
    crime_df['crime_severity'] * 0.5 +
    crime_df['poor_lighting'] * 0.3 +
    crime_df['crowd_density'] * 0.2
)
crime_df[['crime_severity','poor_lighting','crowd_density','hour','risk_score']].head()



X = crime_df[['crime_severity','poor_lighting','crowd_density','hour']]
y = (crime_df['risk_score'] > crime_df['risk_score'].median()).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
acc = model.score(X_test, y_test)
print(f"Risk Model test accuracy: {acc:.3f}")



sample = np.array([[4,1,0,22]])
pred = model.predict(sample)[0]
print("Predicted label (1=High risk, 0=Low risk):", pred)
prob = model.predict_proba(sample)[0]
print("Probability (Low risk, High risk):", prob)



def detect_distress_placeholder(audio_file_path):
    """
    Placeholder detection using MFCC mean energy threshold.
    Returns "DISTRESS" or "NORMAL" or error message.
    """
    try:
        y, sr = librosa.load(audio_file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        # simplistic rule: high energy in first mfcc coefficient may indicate shout/distress
        if mfcc_mean[0] > 120:
            return "DISTRESS"
        else:
            return "NORMAL"
    except Exception as e:
        return f"Audio load error: {e}"

# If you uploaded an audio file to /kaggle/input, put its path here:
example_audio = None
for root, dirs, files in os.walk('/kaggle/input'):
    for f in files:
        if f.lower().endswith(('.wav', '.mp3', '.flac')):
            example_audio = os.path.join(root, f)
            break
    if example_audio:
        break

if example_audio:
    print("Using audio:", example_audio)
    print("Detection result:", detect_distress_placeholder(example_audio))
else:
    print("No audio file found in /kaggle/input. To test, upload an audio sample (help.wav) and re-run.")



import time
import random

class GPSAgent:
    def __init__(self):
        pass
    def track(self):
        # simulate a location label
        return random.choice(["Safe", "Unsafe"])

class RiskAgent:
    def __init__(self, model):
        self.model = model
    def predict(self, features):
        # For demo: features is a dict with keys matching training columns
        arr = np.array([[features['crime_severity'], features['poor_lighting'],
                         features['crowd_density'], features['hour']]])
        prob = self.model.predict_proba(arr)[0][1]
        return prob  # probability of high risk

class AudioAgent:
    def __init__(self, audio_path=None):
        self.audio_path = audio_path
    def listen(self):
        # If we have an audio sample, use it sometimes to simulate "help"
        if self.audio_path and random.random() < 0.2:  # 20% chance to use the sample
            return detect_distress_placeholder(self.audio_path)
        # else random simulation
        return random.choice(["NORMAL", "DISTRESS" if random.random() < 0.05 else "NORMAL"])

class AlertAgent:
    def notify(self, message):
        # In real system, this would send SMS / call API / push notification
        print("ðŸš¨ ALERT:", message)

# Create agents
gps_agent = GPSAgent()
risk_agent = RiskAgent(model)
audio_agent = AudioAgent(example_audio)
alert_agent = AlertAgent()

# Simulate monitoring loop
print("Starting monitoring loop (10 steps)...\n")
for step in range(10):
    gps_status = gps_agent.track()
    # simulate features for current location
    features = {
        'crime_severity': random.randint(1,5),
        'poor_lighting': random.randint(0,1),
        'crowd_density': random.randint(0,3),
        'hour': random.randint(0,23)
    }
    risk_prob = risk_agent.predict(features)
    audio_status = audio_agent.listen()
    
    print(f"Step {step+1}: GPS={gps_status}, RiskProb={risk_prob:.2f}, Audio={audio_status}")
    
    # Decision logic: trigger alert if unsafe signals found
    if gps_status == "Unsafe" or risk_prob > 0.7 or audio_status == "DISTRESS":
        alert_msg = f"User in danger. GPS={gps_status}, risk_prob={risk_prob:.2f}, audio={audio_status}"
        alert_agent.notify(alert_msg)
    
    time.sleep(0.5)



plt.figure(figsize=(8,4))
plt.hist(crime_df['risk_score'], bins=30, color='#D9534F', alpha=0.8)
plt.title("Risk Score Distribution")
plt.xlabel("Risk Score")
plt.ylabel("Count")
plt.grid(alpha=0.2)
plt.show()


