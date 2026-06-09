!pip install autogluon


import pandas as pd
import numpy as np
import supplemental_english
import re 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models import FastText

from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.core.metrics import make_scorer


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


train = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/train.csv", index_col=0)
test = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/test.csv", index_col=0)
region_codes = supplemental_english.REGION_CODES


train.head()


train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])


train["year"] = train["date"].dt.year
train["month"] = train["date"].dt.month
train["day"] = train["date"].dt.day
train["weekday"] = train["date"].dt.day_name()

test["year"] = test["date"].dt.year
test["month"] = test["date"].dt.month
test["day"] = test["date"].dt.day
test["weekday"] = test["date"].dt.day_name()


def parse_plate(plate):
    match = re.match(r"([A-Z��-携])(\d{3})([A-Z��-携]{2})(\d+)$", plate)
    if match:
        return match.groups() 
    return None, None, None, None

train[["letter_1", "digits", "letters_2", "region"]] = train["plate"].apply(lambda x: pd.Series(parse_plate(x)))
test[["letter_1", "digits", "letters_2", "region"]] = test["plate"].apply(lambda x: pd.Series(parse_plate(x)))


train['all_letters'] = train['letter_1'] + train['letters_2']
test['all_letters'] = test['letter_1'] + test['letters_2']


def get_region_name(region_code):
    for region, codes in region_codes.items():
        if region_code in codes:
            return region
    return "Unknown"

train["region_name"] = train["region"].apply(get_region_name)
test["region_name"] = test["region"].apply(get_region_name)


train.isna().sum()


train.head()


latin_to_cyrillic_map = {
    'A': '��', 'B': '��', 'C': '小', 'E': '��', 'H': '��', 'K': '��', 'M': '��', 
    'O': '��', 'P': '��', 'T': '孝', 'Y': '校', 'X': '啸', 'C': '小'
}


def latin_to_cyrillic(letters):
    return ''.join([latin_to_cyrillic_map.get(letter, letter) for letter in letters])

train['all_letters_rus'] = train['all_letters'].apply(latin_to_cyrillic)
test['all_letters_rus'] = test['all_letters'].apply(latin_to_cyrillic)
train.head()


train['number_rus'] = train['all_letters_rus'] + train['digits']
train['number_lat'] = train['all_letters'] + train['digits']


test['number_rus'] = test['all_letters_rus'] + test['digits']
test['number_lat'] = test['all_letters'] + test['digits']


train.head()


def split_to_characters(text):
    return list(text)

model_lat = FastText(sentences=train['number_lat'].apply(split_to_characters).tolist(), vector_size=10, window=3, min_count=1, workers=4)
model_rus = FastText(sentences=train['number_rus'].apply(split_to_characters).tolist(), vector_size=10, window=3, min_count=1, workers=4)

def get_fasttext_embedding(model, text):
    return sum([model.wv[char] for char in text if char in model.wv])

train['number_embeddings_lat'] = train['number_lat'].apply(lambda x: get_fasttext_embedding(model_lat, x))
train['number_embeddings_rus'] = train['number_rus'].apply(lambda x: get_fasttext_embedding(model_rus, x))

test['number_embeddings_lat'] = test['number_lat'].apply(lambda x: get_fasttext_embedding(model_lat, x))
test['number_embeddings_rus'] = test['number_rus'].apply(lambda x: get_fasttext_embedding(model_rus, x))

train.head()


train['embedding_mean'] = train['number_embeddings_lat'].apply(np.mean)
train['embedding_std'] = train['number_embeddings_lat'].apply(np.std)

test['embedding_mean'] = test['number_embeddings_lat'].apply(np.mean)
test['embedding_std'] = test['number_embeddings_lat'].apply(np.std)

train['embedding_mean_rus'] = train['number_embeddings_rus'].apply(np.mean)
train['embedding_std_rus'] = train['number_embeddings_rus'].apply(np.std)

test['embedding_mean_rus'] = test['number_embeddings_rus'].apply(np.mean)
test['embedding_std_rus'] = test['number_embeddings_rus'].apply(np.std)


def split_digits_and_letters(digits, letters, letters_rus):
    digit_columns = [digits[i] if i < len(digits) else '' for i in range(3)] 
    letter_columns = [letters[i] if i < len(letters) else '' for i in range(3)] 
    letter_rus_columns = [letters_rus[i] if i < len(letters_rus) else '' for i in range(3)]  
    return digit_columns + letter_columns + letter_rus_columns  

train[['digit_1', 'digit_2', 'digit_3', 'letter_1', 'letter_2', 'letter_3', 'letter_rus_1', 'letter_rus_2', 'letter_rus_3']] = train.apply(
    lambda row: pd.Series(split_digits_and_letters(row['digits'], row['all_letters'], row['all_letters_rus'])),
    axis=1
)


test[['digit_1', 'digit_2', 'digit_3', 'letter_1', 'letter_2', 'letter_3', 'letter_rus_1', 'letter_rus_2', 'letter_rus_3']] = test.apply(
    lambda row: pd.Series(split_digits_and_letters(row['digits'], row['all_letters'], row['all_letters_rus'])),
    axis=1
)


train.head()


plt.figure(figsize=(12, 6))
sns.lineplot(data=train.sort_values("date"), x="date", y="price", marker="o", label="Price", color='#807490')
plt.title("Changing prices during time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid()
plt.show()


plt.figure(figsize=(10, 5))
sns.histplot(train["price"], bins=30, kde=True, color='#807490')
plt.title("Price distribution")
plt.xlabel("Price")
plt.ylabel("Amount")
plt.grid()
plt.show()


plt.figure(figsize=(15, 6))
sns.barplot(data=train, x="region_name", y="price", ci=None, color='#807490')
plt.xticks(rotation=90, fontsize=9)
plt.title("Mean price per region")
plt.xlabel("Region")
plt.ylabel("Price")
plt.show()


plt.figure(figsize=(10, 5))
sns.boxplot(data=train, x="letter_1", y="price", color='#807490')
plt.title("First letter - price")
plt.xlabel("First letter")
plt.ylabel("Price")
plt.grid()
plt.show()


plt.figure(figsize=(10, 5))
sns.boxplot(data=train, x="weekday", y="price", order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], color='#807490')
plt.title("Weekday - price")
plt.xlabel("Weekday")
plt.ylabel("Price")
plt.grid()
plt.show()


train.columns


categorical = ['year', 'month', 'weekday',
       'digits', 'region', 'all_letters', 'region_name',
       'all_letters_rus', 'digit_1', 'digit_2',
       'digit_3', 'letter_1', 'letter_2', 'letter_3', 'letter_rus_1', 'letter_rus_2',
       'letter_rus_3']


numerical = ['embedding_mean', 'embedding_std',
       'embedding_mean_rus', 'embedding_std_rus']


X = categorical + numerical


y = ['price']


for col in categorical:
    train[col] = train[col].astype(str)
    test[col] = test[col].astype(str)
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")


def smape(y_true, y_pred):
    y_pred = np.exp(y_pred)  
    y_true = np.exp(y_true) 
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100


label = "price" 


smape_scorer = make_scorer(name='smape', score_func=smape, greater_is_better=False)

predictor = TabularPredictor(label=label, eval_metric=smape_scorer).fit(
    train[X + y], 
    time_limit=3600, 
    presets="best_quality" 
)


y_pred = predictor.predict(test)


y_pred


submission = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/sample_submission.csv', index_col=0)


submission['price'] = y_pred


submission.to_csv('submission.csv')

