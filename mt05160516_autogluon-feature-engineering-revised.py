import os
import sys
import string
import re
import pandas as pd

!pip install autogluon
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.core.metrics import make_scorer

import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar

import warnings
warnings.filterwarnings('ignore')


from supplemental_english import *  # REGION_CODES, GOVERNMENT_CODES

# Ensure the logs directory exists
os.makedirs("./logs", exist_ok=True)

# Define log file path
log_file_path = "./logs/training_log.txt"

# Function to log messages to both console and file
def printt(message):
    print(message)
    with open(log_file_path, "a") as log_file:
        log_file.write(message + "\n")

# SMAPE
def smape(y_true, y_pred):
    y_pred = np.exp(y_pred)  
    y_true = np.exp(y_true) 
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100

def find_importance_values_for_plate(plate: str, gov_codes: dict) -> tuple:
    letters = plate[0] + plate[4:6]  # Extracts letters
    numbers = int(plate[1:4])  # Extracts numbers
    region_code = plate[6:]  # Extracts region code

    # print(plate, "---", letters, numbers, region_code)
    
    for (code_letters, num_range, region), details in gov_codes.items():
        if letters == code_letters and region_code == region:
            if num_range[0] <= numbers <= num_range[1]:  # Checks if within range
                return (details[2], details[3])  # Importance values
    
    return (0, 0)  # Ordinary plate, no government affiliation


def add_advantage_on_road_and_significance(data: pd.DataFrame) -> pd.DataFrame:
    def apply_helper(row):
        advantage_on_road, significance = find_importance_values_for_plate(row["plate"], GOVERNMENT_CODES)
        return pd.Series({
            "advantage_on_road": advantage_on_road,
            "significance": significance,
        })

    data[["advantage_on_road", "significance"]] = data.apply(apply_helper, axis=1)
    return data

def encode_plate(plate: str) -> list[int]:
    encoded = []
    for char in plate:
        if char in char2idx:
            encoded.append(char2idx[char])
        else:
            encoded.append(0)
    return encoded

# Define constants
PLATE_POSSIBLE_LETTERS = "ABEKMHOPCTYX"  # 12 total
ALL_CHARS = PLATE_POSSIBLE_LETTERS + string.digits  # 12 + 10 = 22 total
RANDOM_STATE = 37
char2idx = {c: i for i, c in enumerate(ALL_CHARS)}  # char to identifier map


# preprocess data
def get_region_code(plate):
    region_code = str(int(plate[6:]))
    for region, codes in REGION_CODES.items():
        if region_code in codes:
            return region
    return "Unknown"
#adding features
def is_symmetric_plate(plate):
    match = re.match(r'([A-Z])\d{3}([A-Z]{2})\d{2,3}', plate)
    if match:
        prefix = match.group(1)#1st digit
        suffix = match.group(2)#2nd two-digits
        return suffix.startswith(prefix)#if 2nd digits starts from 1st digit
    return False

def is_001_pattern(num_str):
    return num_str == '001'
def is_xxx_pattern(str):
    return str and len(set(str)) == 1
    

def is_repeated_digits(num_str):
    #all int same number
    return num_str and len(set(num_str)) == 1

def is_round_number(num_str):
    try:
        num = int(num_str)
        return num % 100 == 0  # 100 * x num
    except:
        return False

def process_data(csv_link, region_price_dict):
    
    # Read CSV file
    df = pd.read_csv(
        csv_link,
        dtype={
            "id": int,
            "plate": str,
        },
        parse_dates=["date"],
    )


    # Ensure 'date' is in datetime format
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    # Extracting date features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.weekday
    # Removing unnecessary columns
    df = df.drop(columns=["date"])
    df = df.drop(columns=["id"])
    
    # Adding features (advantage on road (bool), significance (int))
    df = add_advantage_on_road_and_significance(df)
    
    # Standardizing plate format (ensuring 9-character plates)
    df["plate"] = df["plate"].apply(lambda plate: plate if len(plate) == 9 else f"{plate[:6]}0{plate[6:]}")

    # add region code
    df["region_name"] = df["plate"].apply(get_region_code).astype(str)

    # add series
    df["plate_number"] = df["plate"].apply(lambda plate: plate[1:4]).astype(str)

    # add number
    df["plate_series"] = df["plate"].apply(lambda plate: plate[0]+plate[4:6]).astype(str)

    # add region number
    df["plate_region"] = df["plate"].apply(lambda plate: plate[6:]).astype(str)

    # extract 777 999 continuous flag
    df['continuous_num'] = df['plate'].apply(is_symmetric_plate)

    #xxx yyy
    df['continuous_char'] = df['plate_series'].apply(is_xxx_pattern)

    #is_001
    df['is_001'] = df['plate_number'].apply(is_001_pattern)

    #is_repeated_numbers
    df['is_rpt_num'] = df['plate_number'].apply(is_repeated_digits)

    #is_round
    df['is_round_num'] = df['plate_number'].apply(is_round_number)
    
    df = df.drop(columns=["plate"], errors="ignore")

    # map region average to each records
    df["region_avg_price"] = df["region_name"].map(region_price_dict)

    # Apply logarithm transformation
    df['price'] = np.log1p(df['price'])
    df['region_avg_price'] = np.log1p(df['region_avg_price'])
        
    return df

# get dict of average region price
train_link = "/kaggle/input/russian-car-plates-prices-prediction/train.csv"
df = pd.read_csv(
    train_link,
    dtype={
        "id": int,
        "plate": str,
    },
    parse_dates=["date"],
)

df["region_code"] = df["plate"].apply(get_region_code)
df['region_avg_price'] = df.groupby("region_code")["price"].transform("mean") 
region_avg_price_dict = df.groupby("region_code")["region_avg_price"].first().to_dict()

# read data
dataset_link = "/kaggle/input/russian-car-plates-prices-prediction/train.csv"
train_df = process_data(dataset_link, region_avg_price_dict)
train_df.head(2)


# Define smape as metrics for AutoGluon
smape_scorer = make_scorer(name='smape1', score_func=smape, greater_is_better=False)

# Train AutoGluon
predictor = TabularPredictor(label='price', eval_metric=smape_scorer).fit(
    train_df,
    time_limit=3600,
)


# Load test dataset
test_data = pd.read_csv(
    "/kaggle/input/russian-car-plates-prices-prediction/test.csv",
    dtype={"id": int, "plate": str},
    parse_dates=["date"],
)

test_ids = test_data["id"].copy()

xgb_df_test = process_data("/kaggle/input/russian-car-plates-prices-prediction/test.csv", region_avg_price_dict)

# Make prediction
test_data = TabularDataset(xgb_df_test)
test_pred = predictor.predict(test_data)
test_pred = np.round(np.expm1(test_pred))
submission = pd.DataFrame()
submission['id'] = test_ids
submission['price'] = test_pred
submission.to_csv('submission.csv',index=False)
print('Done producing submission.csv')

