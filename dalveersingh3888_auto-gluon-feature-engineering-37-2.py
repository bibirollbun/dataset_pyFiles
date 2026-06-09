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


!pip install --quiet autogluon


!pip install holidays


import os
import sys
import string
import pandas as pd
from autogluon.tabular import TabularDataset,TabularPredictor
from autogluon.core.metrics import make_scorer
import xgboost as xgb
import torch
import matplotlib.pyplot as plt
import holidays


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
                return (details[1],details[2], details[3])  # Importance values
    
    return (0,0, 0)  # Ordinary plate, no government affiliation


def add_advantage_on_road_and_significance(data: pd.DataFrame) -> pd.DataFrame:
    def apply_helper(row):
        forbidden_to_buy,advantage_on_road, significance = find_importance_values_for_plate(row["plate"], GOVERNMENT_CODES)
        return pd.Series({
            "forbidden_to_buy": forbidden_to_buy,
            "advantage_on_road": advantage_on_road,
            "significance": significance,
        })

    data[["forbidden_to_buy","advantage_on_road", "significance"]] = data.apply(apply_helper, axis=1)
    return data

def encode_plate(plate: str) -> list[int]:
    encoded = []
    for char in plate:
        if char in char2idx:
            encoded.append(char2idx[char])
        else:
            encoded.append(0)
    return encoded


PLATE_POSSIBLE_LETTERS = "ABEKMHOPCTYX"  # 12 total
ALL_CHARS = PLATE_POSSIBLE_LETTERS + string.digits  # 12 + 10 = 22 total
RANDOM_STATE = 37
char2idx = {c: i for i, c in enumerate(ALL_CHARS)} 


prestigious_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 111, 222, 333, 444, 555, 666, 777, 888, 999,
                       100, 200, 300, 400, 500, 600, 700, 800, 900, 7]  # 007 as 7 for simplicity
prestigious_letter_series = ["AAA", "MMM", "EEE", "KKK", "OOO", "PPP", "CCC", "TTT", "XXX"]

# Feature engineering functions
def has_repeated_letters(letters):
    return len(set(letters)) < len(letters)  # True if any letter repeats

def has_repeated_numbers(num):
    n_str = str(num)
    return len(set(n_str)) < len(n_str)  # True if any digit repeats

def has_sequential_numbers(num):
    n_str = str(num).zfill(3)  # Pad to at least 3 digits for consistency
    seq_up = ["123", "234", "345", "456", "567", "678", "789"]
    seq_down = ["987", "876", "765", "654", "543", "432", "321"]
    return any(seq in n_str for seq in seq_up) or any(seq in n_str for seq in seq_down)

def has_mirror_numbers(num):
    n_str = str(num)
    return n_str == n_str[::-1]


def get_region_code(plate):
    region_code = str(int(plate[6:]))
    for region, codes in REGION_CODES.items():
        if region_code in codes:
            return region
    return "Unknown"

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

    ru_holidays = holidays.Russia(years=[2021,2022, 2023, 2024, 2025])
    df['holiday'] = df['date'].apply(lambda x:ru_holidays.get(x))
    
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
    
    df = df.drop(columns=["plate"], errors="ignore")

    # map region average to each records
    df["region_avg_price"] = df["region_name"].map(region_price_dict)

    # Apply logarithm transformation
    df['price'] = np.log1p(df['price'])
    df['region_avg_price'] = np.log1p(df['region_avg_price'])

    df["is_prestigious_number"] = df["plate_number"].apply(lambda x: 1 if x in prestigious_numbers else 0)
    df["is_prestigious_letter"] = df["plate_series"].apply(lambda x: 1 if x in prestigious_letter_series else 0)
    df["has_repeated_letters"] = df["plate_number"].apply(has_repeated_letters)
    df["has_repeated_numbers"] = df["plate_number"].apply(has_repeated_numbers)
    df["has_sequential_numbers"] = df["plate_number"].apply(has_sequential_numbers)
    df["has_mirror_numbers"] = df["plate_number"].apply(has_mirror_numbers)
    
        
    return df


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


counts = train_df["holiday"].value_counts().sort_index()
plt.figure(figsize=(8, 6))
plt.bar(counts.index, counts.values, color='blue', alpha=0.6)
plt.title("Count of Is Prestigious Number")
plt.xlabel("Is Prestigious Number (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.xticks([0, 1])
plt.grid(True)
plt.show()


smape_scorer = make_scorer(name='smape1', score_func=smape,greater_is_better=False)


predictor = TabularPredictor(label='price', eval_metric=smape_scorer).fit(
    train_df,
    time_limit=9600,
    num_gpus=1 if torch.cuda.is_available() else 0  # Explicitly request 1 GPU
)



predictor.fit_summary()


## Load test dataset
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




