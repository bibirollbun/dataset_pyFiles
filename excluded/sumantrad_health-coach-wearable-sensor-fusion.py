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


# Install dependencies as needed:
!pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter


# Set the path to the file you'd like to load
file_path = "heartrate.csv"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "rjodlowski/health-vitals-dataset",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:", df.head())


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np

X = df.drop(columns=['label'])
Y = df['label']

le = LabelEncoder()
##df['label_'] = le.fit_transform(Y)

encoder = OneHotEncoder(sparse_output=False)
Y = encoder.fit_transform(Y.values.reshape(-1, 1))

df['label_'] = [1 if df.label[i]=='okay' else 0 for i in range(len(df))]

df['min'] = X.min(axis=1)
df['min']
df['max'] = X.max(axis=1)
df['mean'] = X.mean(axis=1)
df['std'] = X.std(axis=1)
df['range'] = df['max'] - df['min']
df['mode'] = X.mode(axis=1)[0]
df = df[['min', 'max', 'mean', 'std', 'range', 'mode', 'label_']]

df.head()


#correlation
df.corr()


from sklearn.model_selection import train_test_split
X = df.drop(columns=['label_'])
Y = df['label_']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)



from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Classification Report:\n", classification_report(Y_test, Y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))


from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=300, random_state=42)
rf_model.fit(X_train, Y_train)
rf_Y_pred = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(Y_test, rf_Y_pred))
print("Random Forest Classification Report:\n", classification_report(Y_test, rf_Y_pred))
print("Random Forest Confusion Matrix:\n", confusion_matrix(Y_test, rf_Y_pred))
print("Random Forest Feature Importances:\n", rf_model.feature_importances_)


#save model
!mkdir models
import joblib
joblib.dump(rf_model, 'models/heartrate_model.pkl')


# Set the path to the file you'd like to load
file_path = "skin_temperature.csv"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "rjodlowski/health-vitals-dataset",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:", df.head())


from sklearn.preprocessing import LabelEncoder
X = df.drop(columns=['label'])
Y = df['label']

le = LabelEncoder()
df['label_'] = le.fit_transform(Y)

df['label_2'] = [1 if df.label[i]=='okay' else 0 for i in range(len(df))]

df['min'] = X.min(axis=1)
df['min']
df['max'] = X.max(axis=1)
df['mean'] = X.mean(axis=1)
df['std'] = X.std(axis=1)
df['range'] = df['max'] - df['min']
df['mode'] = X.mode(axis=1)[0]
df = df[['min', 'max', 'mean', 'std', 'range', 'mode', 'label_']]

df.head()



df.corr()


from sklearn.model_selection import train_test_split
X = df.drop(columns=['label_'])
Y = df['label_']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)



from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Classification Report:\n", classification_report(Y_test, Y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))


from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)
rf_Y_pred = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(Y_test, rf_Y_pred))
print("Random Forest Classification Report:\n", classification_report(Y_test, rf_Y_pred))
print("Random Forest Confusion Matrix:\n", confusion_matrix(Y_test, rf_Y_pred))
print("Random Forest Feature Importances:\n", rf_model.feature_importances_)


#save model
import joblib
joblib.dump(rf_model, 'models/skin_temperature_model.pkl')


# Set the path to the file you'd like to load
file_path = "spo2.csv"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "rjodlowski/health-vitals-dataset",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:", df.head())


from sklearn.preprocessing import LabelEncoder
X = df.drop(columns=['label'])
Y = df['label']

le = LabelEncoder()
df['label_'] = le.fit_transform(Y)

df['label_2'] = [1 if df.label[i]=='okay' else 0 for i in range(len(df))]

df['min'] = X.min(axis=1)
df['min']
df['max'] = X.max(axis=1)
df['mean'] = X.mean(axis=1)
df['std'] = X.std(axis=1)
df['range'] = df['max'] - df['min']
df['mode'] = X.mode(axis=1)[0]
df = df[['min', 'max', 'mean', 'std', 'range', 'mode', 'label_']]

df.head()


df.corr()


from sklearn.model_selection import train_test_split
X = df.drop(columns=['label_'])
Y = df['label_']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Classification Report:\n", classification_report(Y_test, Y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))


from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)
rf_Y_pred = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(Y_test, rf_Y_pred))
print("Random Forest Classification Report:\n", classification_report(Y_test, rf_Y_pred))
print("Random Forest Confusion Matrix:\n", confusion_matrix(Y_test, rf_Y_pred))
print("Random Forest Feature Importances:\n", rf_model.feature_importances_)


#save model
import joblib
joblib.dump(rf_model, 'models/spo2_model.pkl')


import logging
import re
import json


#define enums
from enum import Enum, auto

class SensorType(Enum):
    HEART_RATE = "heart_rate"
    TEMPERATURE = "temperature"
    SPO2 = "spo2"

class HealthCondition(Enum):
    HEALTHY = "healthy"
    NOT_HEALTHY = "not healthy"


# testing function to simulate sensor data retrieval
from typing import Union
def get_sensor_data_0(sensor_type) -> str:
    logging.info("Called sensor type - %s", sensor_type)
    if sensor_type == "heart_rate":
        return "72" # route to fetch_process_detect(heart_rate_sensor)
    elif sensor_type == "temperature":
        return "98.6" # route to fetch_process_detect(temperature_sensor)
    elif sensor_type == "spo2":
        return "95" # route to fetch_process_detect(spo2_sensor)
    else:
        return "-1"


get_sensor_data_0("heart_rate")  # Example call to the function


#wearable sensor data generation stub for testing
#This simulates data from sensors on a health monitoring device like fitbit or apple watch
import numpy as np

def read_data(sensor_type):
    # This function would contain the logic to read data for a specific sensor type
    logging.info("Reading data for sensor type: %s", sensor_type)
    if sensor_type == "heart_rate":
        return np.random.randint(60, 100, size=30)
    elif sensor_type == "temperature":
        return np.random.normal(loc=98.6, scale=0.5, size=30)
    elif sensor_type == "spo2":
        return np.random.randint(90, 100, size=30)
    return None


#basic features used for time series data based prediction
class Features:
    def __init__(self):
        self.min = 0
        self.max = 0
        self.mean = 0
        self.std = 0
        self.range = 0
        self.mode = 0
    def __str__(self):
        return f"Features(mean={self.mean}, std={self.std}, min={self.min}, max={self.max}, range={self.range}, mode={self.mode})"


#pretrained sensor models, stored as dictionary for flexibility in trying different models

MODELS = {
    "heart_rate": {
        "model":"models/heartrate_model.pkl",
        "feature_importance":{"min":0.1001169, "max":0.10261616, "mean":0.0826924, "std":0.38662158, "range":0.3018172, "mode":0.02613575}
    },
    "temperature": {
        "model":"models/skin_temperature_model.pkl",
        "feature_importance":{"min":0.10955198, "max":0.15537635, "mean":0.0810221, "std":0.38798217, "range":0.21992766, "mode":0.04613975}
    },
    "spo2": {
        "model":"models/spo2_model.pkl",
        "feature_importance":{"min":0.23091379, "max":0.01044805, "mean":0.13358294, "std":0.43384486, "range":0.17252367, "mode":0.01868669}
    }
}


#sensor class to handle fetching, processing, and predicting health status based on sensor data

import numpy as np
import joblib
import pandas as pd

class Sensor:
    def __init__(self,sensor_type):
        self.sensor_type = sensor_type

    def fetch_data(self):
        # This function would contain the logic to fetch the sensor data
        logging.info("Fetching data for sensor type: %s", self.sensor_type)
        data = read_data(self.sensor_type)
        if data.size > 0:
            logging.info("Data fetched successfully for sensor type: %s", self.sensor_type)
            return data
        else:
            logging.error("Failed to fetch data for sensor type: %s", self.sensor_type)
            return None

    def process_data(self, sensor_data) -> Features:
        logging.info("Featurizing data: %s for sensor type: %s", sensor_data, self.sensor_type)
        features = Features()
        # Extract relevant features from the sensor data
        features.min = np.min(sensor_data)
        features.max = np.max(sensor_data)
        features.mean = np.mean(sensor_data)
        features.std = np.std(sensor_data)
        features.mode = np.argmax(np.bincount(sensor_data.astype(int)))
        features.range = features.max - features.min
        logging.info("Extracted features: %s", features)
        return features

    def detect_health(self, features: Features):
        logging.info("Detecting health status for %s sensor", self.sensor_type)
        model = joblib.load(MODELS[self.sensor_type]["model"])
        input_data = np.array([[features.min, features.max, features.mean, features.std, features.range, features.mode]])
        input_data = pd.DataFrame(input_data, columns=['min', 'max', 'mean', 'std', 'range', 'mode'])
        logging.info("Input data for prediction: %s", input_data)
        prediction = model.predict(input_data)
        return HealthCondition.HEALTHY if prediction[0] == 1 else HealthCondition.NOT_HEALTHY

    def detection_pipeline(self):
        logging.info("Running detection pipeline for sensor type: %s", self.sensor_type)
        sensor_data = self.fetch_data()
        if sensor_data is None:
            logging.error("Invalid sensor type: %s", self.sensor_type)
            return None
        #sensor_data = np.array([float(x) for x in sensor_data.split(",")])
        features = self.process_data(sensor_data)
        prediction = self.detect_health(features)
        logging.info("Detection result: %s", prediction)
        return prediction
    
    def data_pipeline(self):
        logging.info("Running data pipeline for sensor type: %s", self.sensor_type)
        sensor_data = self.fetch_data()
        if sensor_data is None:
            logging.error("Invalid sensor type: %s", self.sensor_type)
            return None
        #sensor_data = np.array([float(x) for x in sensor_data.split(",")])
        logging.info("data extracted: %s", sensor_data)
        return sensor_data[-1]  # Return the last value as an example
    
    def featurization_pipeline(self):
        logging.info("Running featurization pipeline for sensor type: %s", self.sensor_type)
        sensor_data = self.fetch_data()
        if sensor_data is None:
            logging.error("Invalid sensor type: %s", self.sensor_type)
            return None
        #sensor_data = np.array([float(x) for x in sensor_data.split(",")])
        features = self.process_data(sensor_data)
        logging.info("Featurized data: %s", features)
        return features



#testing the Sensor class
s = Sensor("spo2")


#testing fetch
data = s.fetch_data()
print(data)
#data = [175]*30  # Simulating data for testing


#testing featurization
features = s.process_data(data)
print(features)


#testing detection
detection = s.detect_health(features)
print(detection)


#testing detection pipeline
detection_result = s.detection_pipeline()
print(detection_result)


def get_sensor_detection(sensor_type):
    logging.info("Called sensor type - %s for detection", sensor_type)
    sensor = Sensor(sensor_type=sensor_type)
    return sensor.detection_pipeline()


# Example usage
sensor_type = "spo2"
detection = get_sensor_detection(sensor_type)
print(f"Detection for {sensor_type} sensor: {detection}")


#testing data pipeline
s = Sensor("spo2")
data_result = s.data_pipeline()
print(f"Data for {s.sensor_type} sensor: {data_result}")


def get_sensor_data(sensor_type):
    logging.info("Called sensor type - %s for data", sensor_type)
    sensor = Sensor(sensor_type=sensor_type)
    return sensor.data_pipeline()


# Example usage
sensor_type = "spo2"
data = get_sensor_data(sensor_type)
print(f"Data for {sensor_type} sensor: {data}")


#testing featurization pipeline
s = Sensor("spo2")
features_result = s.featurization_pipeline()
print(f"Features for {s.sensor_type} sensor: {features_result}")


def get_sensor_features(sensor_type):
    logging.info("Called sensor type - %s for features", sensor_type)
    sensor = Sensor(sensor_type=sensor_type)
    return sensor.featurization_pipeline()


#sample usage
sensor_type = "spo2"
features = get_sensor_features(sensor_type)
print(f"Features for {sensor_type} sensor: {features}")


import os
os.chdir('/kaggle/working/')

#if not os.path.exists('/kaggle/tmp'):
#    os.mkdir('/kaggle/tmp')
#os.chdir('/kaggle/tmp/')

print(os.getcwd())

import subprocess
import os

def run(commands):
    for command in commands:
        with subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, bufsize = 1) as sp:
            for line in sp.stdout:
                line = line.decode("utf-8", errors = "replace")
                if "undefined reference" in line:
                    raise RuntimeError("Failed Processing.")
                print(line, flush = True, end = "")
        pass
    pass
pass


commands = [
        "curl -fsSL https://ollama.com/install.sh | sh",
]
run(commands)

import os
os.system("/usr/local/bin/ollama serve &")
os.system("echo 'ollama test'")



!pip install ollama
!ollama pull gemma3n:e4b
from ollama import chat
from ollama import ChatResponse

# 12b is better so try and use it first
model = 'gemma3n:e4b'


# Note, the argument model_prompt is specific here
def model_call(model_prompt):
    
    response: ChatResponse = chat(model=model, messages=[
      {
        'role': 'user',
        'content': model_prompt,
      },
    ])
    return response['message']['content']

user_prompt = "Say hello to the class"

# Note, the argument user_prompt is specific here
model_call(user_prompt)


def augmented_model_call(system_prompt, user_prompt, print_prompt = False):
    combined_prompt = f"{system_prompt}\n{user_prompt}"

    if print_prompt:
        print(combined_prompt)
    
    return model_call(combined_prompt)


pattern = f"```json\n(.*?)\n``"

def parse_response(model_response):
    if tool_call := re.search(pattern, model_response):
        #import pdb; pdb.set_trace()
        return json.loads(tool_call.groups(0)[0])
    return None

model_response = '```json\n[{"sensor_type": "heart_rate"}]\n```\n'
parse_response(model_response)


system_prompt = '''
You have the following functions available
 def get_sensor_detection(sensor_type: str)
   """Given a sensor type returns the data for that sensor as a dict of format {"sensor_type": sensor_type, "value": value}"""

SensorType is an enum with the following key value pairs:
- SensorType.HEART_RATE: "heart_rate"
- SensorType.TEMPERATURE: "temperature"
- SensorType.SPO2: "spo2"

HealthCondition is an enum with the following key value pairs:
- HealthCondition.HEALTHY: "healthy"
- HealthCondition.NOT_HEALTHY: "not healthy"

 If you are asked for heartrate, sensor_type list will be [SensorType.HEART_RATE]
 If you are asked for temperature or skin temperature, sensor_type list will be [SensorType.TEMPERATURE]
 If you are asked for spo2, sensor_type list will be [SensorType.SPO2]
 If you are asked for heartrate and spo2, sensor_type list will be [SensorType.HEART_RATE, SensorType.SPO2]
 If you are asked for heartrate, temperature, and spo2, sensor_type list will be [SensorType.HEART_RATE, SensorType.TEMPERATURE, SensorType.SPO2]
 If you are asked for health vitals, sensor_type list will be [SensorType.HEART_RATE, SensorType.TEMPERATURE, SensorType.SPO2]
 If you are asked for heartrate and temperature, sensor_type list will be [SensorType.HEART_RATE, SensorType.TEMPERATURE]
 If you are asked for temperature and spo2, sensor_type list will be [SensorType.TEMPERATURE, SensorType.SPO2]

 If you have a list of sensors, you need to call this function for each sensor type in the list
 If you need to call this function, return the list in json [{"sensor_type": sensor_type}] and return nothing else
 otherwise respond normally
'''


user_prompt = "What are my health vitals?"
augmented_model_call(system_prompt, user_prompt)


#monitoring

import sqlite3
import datetime
import pathlib

DB_PATH = pathlib.Path("chat_log.db")

def setup_database():
    """Create a simple SQLite table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT,
            response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

if not DB_PATH.exists():
    print(f"Database {DB_PATH} does not exist. Setting up the database.")
    setup_database()


# use datasette
# datasette chat_log.db --cors --port 8001


def chat_interaction_c(user_prompt):
    system_prompt = '''
    You have the following functions available
        def get_sensor_detection(sensor_type)
        """Given a sensor type returns the data for that sensor as a dict of format {"sensor_type": sensor_type, "value": value}"""

        SensorType is an enum with the following key value pairs:
        - SensorType.HEART_RATE: "heart_rate"
        - SensorType.TEMPERATURE: "temperature"
        - SensorType.SPO2: "spo2"

        HealthCondition is an enum with the following key value pairs:
        - HealthCondition.HEALTHY: "healthy"
        - HealthCondition.NOT_HEALTHY: "not healthy"

        If you are asked for heartrate, sensor_type list will be [SensorType.HEART_RATE]
        If you are asked for temperature or skin temperature, sensor_type list will be [SensorType.TEMPERATURE]
        If you are asked for spo2, sensor_type list will be [SensorType.SPO2]
        If you are asked for heartrate and spo2, sensor_type list will be [SensorType.HEART_RATE, SensorType.SPO2]
        If you are asked for heartrate, temperature, and spo2, sensor_type list will be [SensorType.HEART_RATE, SensorType.TEMPERATURE, SensorType.SPO2]
        If you are asked for health vitals, sensor_type list will be [SensorType.HEART_RATE, SensorType.TEMPERATURE, SensorType.SPO2]
        If you are asked for heartrate and temperature, sensor_type list will be [SensorType.HEART_RATE, SensorType.TEMPERATURE]
        If you are asked for temperature and spo2, sensor_type list will be [SensorType.TEMPERATURE, SensorType.SPO2]

        If you have a list of sensors, you need to call this function for each sensor type in the list
        If you need to call this function, return the list in json [{"sensor_type": sensor_type}] and return nothing else
        otherwise respond normally
        '''

    # Get a model response. Right now we don't know if it's a function call or chat response
    model_response = augmented_model_call(system_prompt, user_prompt)

    #print(f"Model response: {model_response}")

    # Regex to see if we have the json which indicates a function call
    function_call_json = parse_response(model_response)

    # If it's not a function call, return the response
    if not function_call_json:
        return model_response


    print(f"Function call detected: {function_call_json}")

    # Since we detect a function call
    all_data = []
    features = []
    # Get the sensor data for the requested sensor types
    for item in function_call_json:
        #logging.info("Processing sensor type", item[0])
        print(f"Processing sensor type - {item['sensor_type']}")
        sensor_data = get_sensor_detection(item["sensor_type"])
        all_data.append({'sensor_type': item["sensor_type"], 'value': sensor_data})
        # Get the features for the sensor data
        sensor_features = get_sensor_features(item["sensor_type"])
        features.append({'sensor_type': item["sensor_type"], 'features': sensor_features})

    # We have a choice here
    # We could return the sensor data directly to the user
    # But for a nicer experience let's reinject it into the LLM for a better final response
    sensor_data = ", ".join([f"{data['sensor_type']} is {data['value']}" for data in all_data])
    # We can also add the features to the response
    features_data = ", ".join([f"{feature['sensor_type']} features are {feature['features']}" for feature in features])
    feature_importance = ", ".join([f"{feature['sensor_type']} feature importance is {MODELS[feature['sensor_type']]['feature_importance']}" for feature in features])
    # Combine sensor data and features data for the final response
    function_response_prompt = f"Tell me the {sensor_data} as if you were a health coach. \
        Use the feature data {features_data} to explain your response knowing how important each feature is from {feature_importance}. \
        Use std, mean, min and max to explain your response. \
    "

    # We already checked for sensor data so we don't need to go again
    model_response = model_call(function_response_prompt)

    #add monitoring data to the database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_history (prompt, response) VALUES (?, ?)", (user_prompt, model_response))
    conn.commit()
    conn.close()

    return model_response

chat_interaction_c("Can you say hi to Hugo?")


chat_interaction_c("What are my heart rate and spo2 readings?")


chat_interaction_c("What are my health vitals?")


import gradio as gr


# def chat_with_model(prompt):
#     response = ollama.chat(model=model, messages=[{'role': 'user', 'content': prompt}])
#     return response['message']['content']

iface = gr.Interface(
    fn=chat_interaction_c,
    inputs=gr.Textbox(lines=2, placeholder="Type your message here..."),
    outputs="text",
    title="Chat with Gemma",
    description="Enter a message and get a response from the Gemma 3n model.",
)

iface.launch()

