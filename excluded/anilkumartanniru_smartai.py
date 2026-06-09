#
#  === FULLY UPDATED Anil AI Assistant with Voice/Text Input Switch ===

import speech_recognition as sr
import pyttsx3
import datetime
import webbrowser
import os
import json
import time
import smtplib
import requests
import pywhatkit
import threading
import re
from email.message import EmailMessage
from cryptography.fernet import Fernet

from googletrans import Translator
import screen_brightness_control as sbc
import ctypes
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import openai

# === CONFIG ===
USE_VOICE = False  # True = microphone input, False = type input

sender_email = "anilkumartanniru250@gmail.com"
receiver_email = "anilkumarthanniru250@gmail.com"
password = "gqln ikqy qxaz rhke"
openai.api_key = "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"
weather_api_key = "12345abcdefweatherREALKEY"

key = Fernet.generate_key()
cipher_suite = Fernet(key)
translator = Translator()
engine = pyttsx3.init()
engine.setProperty('rate', 170)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

memory_file = "memory.json"
recognizer = sr.Recognizer()

def speak(text):
    print("Anil AI:", text)
    engine.say(text)
    engine.runAndWait()

def load_memory():
    try:
        with open(memory_file, "r") as f:
            return json.load(f)
    except:
        return {}

def save_memory(data):
    with open(memory_file, "w") as f:
        json.dump(data, f, indent=2)

memory = load_memory()

def listen_command():
    if USE_VOICE:
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            try:
                query = recognizer.recognize_google(audio)
                print("You said:", query)
                return query.lower()
            except sr.UnknownValueError:
                return ""
            except sr.RequestError:
                speak("Sorry, I am offline.")
                return ""
    else:
        return input("Type command: ").lower()

def convert_currency_from_text(text):
    match = re.search(r'(\d+)', text)
    if not match:
        speak("Please specify the amount to convert.")
        return
    amount = float(match.group(1))
    try:
        url = "https://api.exchangerate-api.com/v4/latest/INR"
        data = requests.get(url).json()
        rate = data["rates"].get("USD")
        if rate:
            result = round(amount * rate, 2)
            speak(f"{amount} rupees is approximately {result} dollars.")
        else:
            speak("Currency conversion failed.")
    except:
        speak("Error fetching currency rates.")

def tell_time():
    now = datetime.datetime.now()
    speak(f"The time is {now.strftime('%I:%M %p')}")

def set_reminder(text):
    memory["reminder"] = text
    save_memory(memory)
    speak("Reminder set.")

def get_reminder():
    if "reminder" in memory:
        speak(f"Reminder: {memory['reminder']}")
    else:
        speak("No reminders found.")

def fetch_weather():
    try:
        city = "Delhi"
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={weather_api_key}&units=metric"
        res = requests.get(url).json()
        if res.get("main"):
            temp = res["main"]["temp"]
            desc = res["weather"][0]["description"]
            speak(f"{city} has {temp}°C and {desc}.")
        else:
            speak("Failed to get weather.")
    except Exception as e:
        speak("Weather API error.")
        print("Weather Error:", e)

def fetch_news():
    speak("Opening news.")
    webbrowser.open("https://news.google.com")

def open_app_or_website(command):
    apps = {"notepad": "notepad.exe", "calculator": "calc.exe"}
    websites = {"google": "https://www.google.com", "youtube": "https://www.youtube.com",
                "facebook": "https://www.facebook.com", "twitter": "https://www.twitter.com",
                "gmail": "https://mail.google.com"}
    name = command.lower().replace("open", "").strip()
    if name in apps:
        os.system(apps[name])
    elif name in websites:
        webbrowser.open(websites[name])
    else:
        webbrowser.open(f"https://{name}.com")
    speak(f"Opening {name}.")

def play_music_on_youtube(query):
    speak(f"Playing {query} on YouTube.")
    pywhatkit.playonyt(query)

def dictation_to_text():
    return listen_command()

def encrypt_text(text):
    return cipher_suite.encrypt(text.encode()).decode()

def send_email(body, subject="Anil AI Email", attachment_path=None):
    encrypted_body = encrypt_text(body)
    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.set_content(f"Encrypted Message:\n{encrypted_body}")

    if attachment_path:
        with open(attachment_path, "rb") as f:
            msg.add_attachment(f.read(), maintype="application", subtype="octet-stream", filename=os.path.basename(f.name))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, password)
        server.send_message(msg)
        server.quit()
        speak("Email sent.")
    except Exception as e:
        speak(f"Failed to send email: {e}")

def schedule_email(delay, body, subject="Anil AI Email", attachment_path=None):
    timer = threading.Timer(delay, send_email, args=(body, subject, attachment_path))
    timer.start()

def translate_text(text, src="en", dest="te"):
    translated = translator.translate(text, src=src, dest=dest)
    speak(translated.text)

def control_system(query):
    if "brightness" in query:
        sbc.set_brightness(50)
    elif "shutdown" in query:
        os.system("shutdown /s /t 1")
    elif "lock" in query:
        ctypes.windll.user32.LockWorkStation()
    elif "volume" in query:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume.iid, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        volume.SetMasterVolumeLevelScalar(0.3, None)
    speak("System updated.")

def chatgpt_reply(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        reply = response["choices"][0]["message"]["content"]
        speak(reply)
    except Exception as e:
        speak("Failed to connect to ChatGPT API.")
        print("ChatGPT Error:", e)

def tell_joke():
    res = requests.get("https://v2.jokeapi.dev/joke/Any?type=single").json()
    speak(res.get("joke", "Couldn't fetch joke."))

def fun_fact():
    res = requests.get("https://uselessfacts.jsph.pl/random.json?language=en").json()
    speak(res.get("text", "No fun fact today."))

def send_whatsapp_message():
    speak("Say the number with country code.")
    number = listen_command().replace(" ", "")
    speak("What is your message?")
    message = listen_command()
    t = datetime.datetime.now()
    pywhatkit.sendwhatmsg(f"+{number}", message, t.hour, t.minute + 2)
    speak("Message scheduled.")

def set_alarm_from_text(text):
    match = re.search(r'(\d{1,2}):?(\d{0,2})\s*(am|pm)?', text.lower())
    if not match:
        speak("Sorry, I didn’t understand the time.")
        return
    hour = int(match.group(1))
    minute = int(match.group(2)) if match.group(2) else 0
    period = match.group(3)

    if period == 'pm' and hour < 12:
        hour += 12
    elif period == 'am' and hour == 12:
        hour = 0

    now = datetime.datetime.now()
    alarm_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if alarm_time < now:
        alarm_time += datetime.timedelta(days=1)

    delay = (alarm_time - now).total_seconds()
    speak(f"Alarm set for {alarm_time.strftime('%I:%M %p')}")
    threading.Timer(delay, lambda: speak("⏰ Wake up! This is your alarm.")).start()

def tell_story():
    prompt = "Tell me a short, original story for kids."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        story = response["choices"][0]["message"]["content"]
        speak("Here is a story for you.")
        speak(story)
    except Exception as e:
        speak("Sorry, I couldn’t get a story right now.")
        print("Story Error:", e)

def daily_summary():
    speak("Here is your daily summary.")
    tell_time()
    get_reminder()
    fetch_weather()
    fetch_news()

# === MAIN LOOP ===
def main_loop():
    speak("Hi, I am Anil, your assistant.")
    while True:
        query = listen_command()
        if "stop" in query:
            speak("Bye!")
            break
        elif "time" in query:
            tell_time()
        elif "remind me" in query:
            set_reminder(query.replace("remind me", "").strip())
        elif "what is my reminder" in query:
            get_reminder()
        elif "weather" in query:
            fetch_weather()
        elif "convert" in query:
            convert_currency_from_text(query)
        elif "news" in query:
            fetch_news()
        elif "open" in query:
            open_app_or_website(query)
        elif "play" in query:
            play_music_on_youtube(query.replace("play", "").strip())
        elif "send email" in query:
            speak("Dictate message.")
            body = dictation_to_text()
            schedule_email(30, body)
        elif "remember" in query:
            memory["note"] = query.replace("remember", "").strip()
            save_memory(memory)
            speak("Got it.")
        elif "what did i say" in query:
            speak(memory.get("note", "Nothing stored."))
        elif "change voice" in query:
            current = engine.getProperty('voice')
            engine.setProperty('voice', voices[0].id if current == voices[1].id else voices[1].id)
            speak("Voice changed.")
        elif "translate" in query:
            speak("What to translate?")
            translate_text(listen_command())
        elif any(x in query for x in ["brightness", "shutdown", "lock", "volume"]):
            control_system(query)
        elif "chat with me" in query:
            speak("Sure. Talk to me.")
            while True:
                q = listen_command()
                if "stop chat" in q:
                    speak("Chat ended.")
                    break
                chatgpt_reply(q)
        elif "joke" in query:
            tell_joke()
        elif "fun fact" in query:
            fun_fact()
        elif "send whatsapp" in query:
            send_whatsapp_message()
        elif "set alarm" in query:
            set_alarm_from_text(query)
        elif "tell me a story" in query:
            tell_story()
        elif "daily summary" in query or "give me my summary" in query:
            daily_summary()
        else:
            speak("Please repeat.")

main_loop()


