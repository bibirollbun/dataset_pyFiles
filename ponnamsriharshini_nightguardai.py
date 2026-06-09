# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install google-generativeai gradio tensorflow opencv-python mediapipe speechrecognition pydub
!pip install tensorflow gradio google-generativeai geocoder gTTS pydub SpeechRecognition


import os
from kaggle_secrets import UserSecretsClient
import google.generativeai as genai

# Fetch API key stored in Kaggle Secrets
user_secrets = UserSecretsClient()
API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")

# Configure API
genai.configure(api_key=API_KEY)

print("API Key Loaded Successfully!")


import tensorflow as tf

model = tf.keras.applications.MobileNetV2(weights="imagenet")

def detect_threat(image):
    img = tf.image.resize(image, (224, 224))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    preds = model.predict(tf.expand_dims(img, 0))
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0][0]
    label = decoded[1]
    confidence = float(decoded[2])

    if label in ["person", "knife", "gun", "weapon"]:
        return "Threat", confidence
    return "Safe", confidence

    import cv2
import numpy as np

def enhance_night_mode(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return enhanced_rgb, gray

def detect_threat(image):
    enhanced_rgb, gray = enhance_night_mode(image)

    brightness = np.mean(gray)        # â­� Adds brightness sensor
    low_light = brightness < 70       # ~night mode threshold

    img = tf.image.resize(enhanced_rgb, (224, 224))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    preds = model.predict(tf.expand_dims(img, 0), verbose=0)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0][0]

    label = decoded[1]
    confidence = float(decoded[2])

    threat = label.lower() in ["person", "knife", "gun", "weapon"]

    if threat:
        return "Threat", confidence, brightness, low_light
    return "Safe", confidence, brightness, low_light



import speech_recognition as sr

recognizer = sr.Recognizer()

def listen_for_help():
    with sr.Microphone() as source:
        print("ğŸ�¤ Listening for HELP...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio).lower()
        if "help me" in text:
            return True
        return False
    except:
        return False


danger_agent_flash = genai.GenerativeModel("gemini-2.0-flash")
danger_agent_pro = genai.GenerativeModel("gemini-2.0-pro")

def analyze_danger(prompt):
    response = danger_agent_flash.generate_content(prompt)
    text = response.text.lower()

    danger_level = "low"
    if "attack" in text or "weapon" in text or "danger" in text:
        danger_level = "high"
        response = danger_agent_pro.generate_content(prompt)

    return response.text, danger_level


import geocoder

def get_location():
    g = geocoder.ip('me')
    if g.ok:
        lat, lng = g.latlng
        return f"https://www.google.com/maps/search/?api=1&query={lat},{lng}"
    return "Location unavailable"


def add_log(status, confidence, brightness, low_light):
    logs.append({
        "status": status,
        "confidence": round(confidence,2),
        "brightness": round(brightness,2),   # new
        "night_mode": low_light,             # new
        "location": get_location()
    })
    return logs


from gtts import gTTS
from IPython.display import Audio

def speak_alert(text_en):
    multilingual_text = (
        f"English: {text_en}. "
        f"Hindi: à¤–à¤¤à¤°à¤¾! à¤®à¤¦à¤¦ à¤•à¤°à¥‹! "
        f"Telugu: à°ªà±�à°°à°®à°¾à°¦à°‚! à°¸à°¹à°¾à°¯à°‚ à°šà±‡à°¯à°‚à°¡à°¿!"
    )
    tts = gTTS(multilingual_text, lang="en")
    tts.save("alert.mp3")
    return Audio("alert.mp3", autoplay=True), multilingual_text


import gradio as gr
import numpy as np

def process_frame(frame):
    image_np = np.array(frame)
    status, conf, brightness, low_light = detect_threat(image_np)

    night_state = "ON ğŸŒ™" if low_light else "OFF ğŸ”†"

    alert_voice, alert_text = None, None

    if status == "Threat" or listen_for_help():
        msg, danger = analyze_danger("Person is in danger at night")
        alert_voice, alert_text = speak_alert(msg)
        add_log(status, conf, brightness, low_light)

    return (
        f"Status: {status} ({conf:.2f}) | Night Mode: {night_state} | Light: {brightness:.1f}",
        alert_text,
        get_location(),
        logs,
        alert_voice
    )

with gr.Blocks(title="Agentic AI Safety App") as demo:
    gr.Markdown("## ğŸ›¡ï¸� AI Personal Safety Agent â€” Multilingual + Multi-Agent")
    with gr.Row():
        camera = gr.Image(sources=["webcam"], streaming=True)
        out = gr.Textbox(label="Detection")
    voice = gr.Audio(label="Voice Alert")
    loc = gr.Textbox(label="Location Link")
    history = gr.JSON(label="Emergency Logs")

    camera.change(
        fn=process_frame,
        inputs=camera,
        outputs=[out, voice, loc, history, voice]
    )

demo.launch()




