import requests
import numpy as np
import pandas as pd
import seaborn as sns
import folium
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
import requests
from io import BytesIO
from IPython.display import display
import pytz
import random
import warnings
warnings.filterwarnings("ignore")


API_KEY = ""
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"


def weather(city):
    url = f"{BASE_URL}?q={city}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()

        if response.status_code != 200:
            raise Exception(f"Error: {data.get('message', 'Unknown error')}")

        return {
            'city': data.get('name', 'Unknown'),
            'current_temp': round(data['main'].get('temp', 0)),
            'feels_like': round(data['main'].get('feels_like', 0)),
            'temp_min': round(data['main'].get('temp_min', 0)),
            'temp_max': round(data['main'].get('temp_max', 0)),
            'humidity': round(data['main'].get('humidity', 0)),
            'description': data['weather'][0]['description'] if 'weather' in data else 'N/A',
            'country': data['sys'].get('country', 'N/A'),
            'WindGustSpeed': data['wind'].get('speed', 0),
            'WindGustDir': data['wind'].get('deg', 0),
            'Pressure': data['main'].get('pressure', 0),
            'lat': data['coord'].get('lat', 0),
            'lon': data['coord'].get('lon', 0)
        }
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return {}



def load_data(filename):
    try:
        df = pd.read_csv(filename, sep=',', encoding='utf-8')
        df = df.dropna()
        df = df.drop_duplicates()
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()


def prepare(data):
    le = LabelEncoder()
    if 'WindGustDir' in data.columns:
        data['WindGustDir'] = le.fit_transform(data['WindGustDir'].astype(str))
    if 'RainTomorrow' in data.columns:
        data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'].astype(str))
    required_columns = ['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure']
    available_columns = [col for col in required_columns if col in data.columns]
    if len(available_columns) < len(required_columns):
        print(f"Warning: Missing columns {set(required_columns) - set(available_columns)}")
    X = data[available_columns]
    y = data['RainTomorrow'] if 'RainTomorrow' in data.columns else None
    return X, y, le


def train_rain_model(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    return model


def prepare_regression_data(data, feature):
    x, y = [], []
    for i in range(len(data) - 1):
        x.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i + 1])
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
    return x, y


def train_regression_model(x, y):
    model = XGBRegressor(random_state=42)
    model.fit(x, y)
    return model


def predict_future(model, value):
    prediction = [value]
    for _ in range(5):
        next_value = model.predict(np.array([prediction[-1]]).reshape(-1, 1))
        prediction.append(next_value[0])
    return prediction[1:]


def map (lat, lon, city, country):
    city_map = folium.Map(location=[lat, lon], zoom_start=10)
    folium.Marker(
        location=[lat, lon],
        popup=f"{city}, {country}",
        tooltip=f"Click for info about {city}",
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(city_map)
    display(city_map)
    return city_map


def weather_view():
    city = input("Enter city name: ")
    current_weather = weather(city)

    if not current_weather:
        print("Failed to retrieve weather data.")
        return

    historical_data = load_data('weather.csv')
    x, y, le = prepare(historical_data)
    rain_model = train_rain_model(x, y)

    wind_deg = current_weather['WindGustDir'] % 360
    compass_points = [
        ("N", 0, 22.5), ("NE", 22.5, 67.5), ("E", 67.5, 112.5), ("SE", 112.5, 157.5),
        ("S", 157.5, 202.5), ("SW", 202.5, 247.5), ("W", 247.5, 292.5), ("NW", 292.5, 337.5), ("N", 337.5, 360)
    ]
    compass_direction = next((point for point, start, end in compass_points if start <= wind_deg < end), "Unknown")
    compass_direction_encoded = le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1

    current_data = {
        'MinTemp': current_weather['temp_min'],
        'MaxTemp': current_weather['temp_max'],
        'WindGustDir': compass_direction_encoded,
        'WindGustSpeed': current_weather['WindGustSpeed'],
        'Humidity': current_weather['humidity'],
        'Pressure': current_weather['Pressure'],
    }

    current_df = pd.DataFrame([current_data])
    rain_prediction = rain_model.predict(current_df)[0]
    rain_probability = "Yes" if rain_prediction == 1 else "No"

    x_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
    x_hum, y_hum = prepare_regression_data(historical_data, 'Humidity')
    x_wind, y_wind = prepare_regression_data(historical_data, 'WindGustSpeed')

    temp_model = train_regression_model(x_temp, y_temp)
    hum_model = train_regression_model(x_hum, y_hum)
    wind_model = train_regression_model(x_wind, y_wind)

    timezone = pytz.timezone("Africa/Cairo")
    now = datetime.now(timezone)
    future_times = [(now + timedelta(minutes=30 * i)).strftime("%H:%M:%S") for i in range(10)]

    future_temps = [predict_future(temp_model, current_weather['temp_min'])[0] + random.uniform(-0.5, 0.5) for _ in range(10)]
    future_hums = [predict_future(hum_model, current_weather['humidity'])[0] + random.uniform(-0.5, 0.5) for _ in range(10)]
    future_winds = [predict_future(wind_model, current_weather['WindGustSpeed'])[0] + random.uniform(-0.5, 0.5) for _ in range(10)]
    future_rain = [rain_probability] * 10

    print(f"\nCity: {current_weather['city']}, {current_weather['country']}")
    print(f"Current Temperature: {current_weather['current_temp']}°C")
    print(f"Feels Like: {current_weather['feels_like']}°C")
    print(f"Min Temperature: {current_weather['temp_min']}°C, Max Temperature: {current_weather['temp_max']}°C")
    print(f"Humidity: {current_weather['humidity']}%")
    print(f"Description: {current_weather.get('description', 'Unknown')}")
    print(f"Pressure: {current_weather['Pressure']} hPa")
    print(f"Wind: {current_weather['WindGustSpeed']} km/h, Direction: {compass_direction}")
    print(f"Rain Prediction: {rain_probability}\n")

    print("Future Weather Predictions:")
    print(f"{'Time'.ljust(12)}{'Temp (°C)'.ljust(12)}{'Humidity (%)'.ljust(14)}{'Wind (km/h)'.ljust(14)}{'Rain'.ljust(6)}")
    print("-" * 55)

    for time, temp, hum, wind, rain in zip(future_times, future_temps, future_hums, future_winds, future_rain):
        print(f"{time.ljust(12)}{str(round(temp,1)).ljust(12)}{str(round(hum,1)).ljust(14)}{str(round(wind,1)).ljust(14)}{rain.ljust(6)}")

    city_map = map(current_weather['lat'], current_weather['lon'], current_weather['city'], current_weather['country'])
    city_map.save("weather_map.html")
    print("\nMap saved as 'weather_map.html'.")


weather_view()

