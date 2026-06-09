# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import matplotlib.pyplot as plt
import seaborn as sns
import supplemental_english  
from sklearn.model_selection import train_test_split
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# Load datasets
train = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/train.csv", index_col=0)
test_df = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/test.csv", index_col=0)
train


from supplemental_english import GOVERNMENT_CODES, REGION_CODES

class FeatureEng:
    def __init__(self, train):
        """Initialize with the training dataset."""
        self.data = train 
        self.gov_dict = GOVERNMENT_CODES  
        self.reg_dict= REGION_CODES

    def getYear(self):
        """Convert 'date' to datetime and extract the year relative to 2021."""
        self.data["date"] = pd.to_datetime(self.data["date"])
        self.data["Year"] = self.data["date"].dt.year - 2021  

    def get_region(self, plate):
        if not isinstance(plate, str) or len(plate) < 7:
            return 0, 0
        """Extracts the region code from the plate."""
        region_code = plate[6:]
        numbers=plate[1:4]
        for region_name, codes in self.reg_dict.items():
            if region_code in codes:
                return numbers, region_code, region_name
        return 0, "unknown region"

    def encode_plates(self, plate):
        russian_letters = "Ğ�Ğ’Ğ•ĞšĞœĞ�Ğ�Ğ Ğ¡Ğ¢Ğ£Ğ¥"
        latin_to_cyrillic = str.maketrans("ABEKMHOPCTYX", "Ğ�Ğ’Ğ•ĞšĞœĞ�Ğ�Ğ Ğ¡Ğ¢Ğ£Ğ¥")
        plate = plate.translate(latin_to_cyrillic)
        letter_to_index = {char: index for index, char in enumerate(russian_letters, start=1)}
        letters=plate[0]+plate[4:6]
        encoded_plate = [letter_to_index.get(char, 0) for char in letters]
        return encoded_plate

    def extract_features(self, plate):
        """Extracts important features from plate using the GOVERNMENT_CODES dictionary."""
        if not isinstance(plate, str) or len(plate) < 7:
            return 0, 0, 0

        letters = plate[0] + plate[4:6]  # Extracts first letter + last two letters
        numbers = int(plate[1:4])  # Extracts numeric part
        region_code = plate[6:]  # Extracts the region code

        for (code_letters, num_range, region), values in self.gov_dict.items():
            if letters == code_letters and region_code == region:
                if num_range[0] <= numbers <= num_range[1]:  
                    return values[1], values[2], values[3]  # forbidden_to_buy, advantage_on_road, significance

        return 0, 0, 0 

    def add_features_to_data(self):
       
        def apply_helper(row):
            """Helper function to apply on each row."""
            encoded_plate = self.encode_plates(row["plate"])
            numbers, region_code, region_name = self.get_region(row["plate"])
            forbidden_to_buy, advantage_on_road, significance = self.extract_features(row["plate"])
            plate_split = {f'char{i+1}': encoded_plate[i] if i < len(encoded_plate) else 0 for i in range(3)}
            
            return pd.Series({
                "region_code": region_code,
                "region_name": region_name,
                "numbers" : numbers,
                "forbidden_to_buy": forbidden_to_buy,
                "advantage_on_road": advantage_on_road,
                "significance": significance,
                **plate_split
            })

        self.data[["region_code", "region_name", "numbers", "forbidden_to_buy", "advantage_on_road", "significance", 
                   "char1", "char2", "char3"]] = self.data.apply(apply_helper, axis=1)

        return self.data  # Return the modified DataFrame


x=FeatureEng(train)
x.getYear()
x.add_features_to_data()
train.tail()


# Group by region and calculate mean price
price_by_region = train.groupby("region_name", as_index=False)["price"].mean()
plt.figure(figsize=(12, 6))
plt.bar(price_by_region["region_name"], price_by_region["price"], color='skyblue')
plt.xlabel("Region")
plt.ylabel("Average Price")
plt.title("ğŸ“Š Price by Region")
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



fig, ax = plt.subplots(figsize=(12, 6))
warnings.filterwarnings("ignore", category=FutureWarning)
# Group data
forbidden_groups = [group["price"].values for _, group in train.groupby("advantage_on_road")]
box = ax.boxplot(forbidden_groups, labels=["No Advantage", "Advantage"], 
                 showfliers=False, patch_artist=True, boxprops=dict(facecolor="lightblue"),
                 medianprops=dict(color="red", linewidth=2))
ax.set_xlabel("Advantage on road")
ax.set_ylabel("Prices")
ax.set_title("Car Prices by Advantaged cars")
ax.grid(True, linestyle="--", alpha=0.6)
warning='ignore'
plt.show()



fig, ax = plt.subplots(figsize=(12, 6))
warnings.filterwarnings("ignore", category=FutureWarning)
# Group data
forbidden_groups = [group["price"].values for _, group in train.groupby("forbidden_to_buy")]
box = ax.boxplot(forbidden_groups, labels=["Not Forbidden", "Forbidden"], 
                 showfliers=False, patch_artist=True, boxprops=dict(facecolor="lightblue"),
                 medianprops=dict(color="red", linewidth=2))
ax.set_xlabel("Forbidden to Buy")
ax.set_ylabel("Prices")
ax.set_title("Car Prices by Forbidden to Buy")
ax.grid(True, linestyle="--", alpha=0.6)
warning='ignore'
plt.show()



# Extract year from date
train["year"] = train["date"].dt.year
warnings.filterwarnings("ignore", category=FutureWarning)
# Group by year and calculate mean price
mean_price_by_year = train.groupby("year", as_index=False)["price"].mean()
plt.figure(figsize=(10, 5))
plt.plot(mean_price_by_year["year"], mean_price_by_year["price"], marker='o', linestyle='-', color='b')
plt.xlabel("Year")
plt.ylabel("Average Price")
plt.title("Average Car Price by Year")
plt.grid(True)
plt.show()



train.columns


# Define scaling function
def scale_features(df):
    sc = StandardScaler()
    scaled_array = sc.fit_transform(df)
    return sc.fit_transform(df)

# Drop non-numeric and target column before scaling
train_ = train.drop(["plate", "date", "region_name", "price", "year"], axis=1)
train_scaled = scale_features(train_)
target = train["price"].values 
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))

# Define search space (Fixed: Removed duplicate 'n_estimators')
space = {
    'n_estimators': hp.quniform("n_estimators", 100, 300, 10),
    'max_depth': hp.quniform("max_depth", 3, 18, 1),
    'gamma': hp.uniform("gamma", 1, 9),
    'reg_alpha': hp.quniform("reg_alpha", 40, 180, 1),
    'reg_lambda': hp.uniform("reg_lambda", 0, 1),
    'colsample_bytree': hp.uniform("colsample_bytree", 0.5, 1),
    'min_child_weight': hp.quniform("min_child_weight", 0, 10, 1),
    'seed': 0
}

# Split data
x_train, x_test, y_train, y_test = train_test_split(train_scaled, target, test_size=0.2, random_state=42)

# Objective function for optimization
def objective(space):
    clf = xgb.XGBRegressor(
        n_estimators=int(space['n_estimators']),
        max_depth=int(space['max_depth']),
        gamma=space['gamma'],
        reg_alpha=int(space['reg_alpha']),
        reg_lambda=space['reg_lambda'],
        min_child_weight=int(space['min_child_weight']),
        colsample_bytree=space['colsample_bytree'],
        objective="reg:squarederror",
        eval_metric="mae",
        seed=0
    )

    # Train model
    clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], early_stopping_rounds=10, verbose=False)

    # Predictions
    pred = clf.predict(x_test)

    # Calculate SMAPE
    smape_score = smape(y_test, pred)
    print("SMAPE:", smape_score)

    return {'loss': smape_score, 'status': STATUS_OK}  # Minimize SMAPE

# Run Hyperparameter Optimization
trials = Trials()
best_hyperparams = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=150, trials=trials)

print("Best Hyperparameters:", best_hyperparams)



# Training final model with best hyperparameters
final_model = xgb.XGBRegressor(
    n_estimators=150,
    max_depth=int(best_hyperparams['max_depth']),
    gamma=best_hyperparams['gamma'],
    reg_alpha=int(best_hyperparams['reg_alpha']),
    reg_lambda=best_hyperparams['reg_lambda'],
    min_child_weight=int(best_hyperparams['min_child_weight']),
    colsample_bytree=best_hyperparams['colsample_bytree'],
    objective="reg:squarederror",
    eval_metric="mae",
    seed=0
)

final_model.fit(x_train, y_train)
predictions = final_model.predict(x_test)

#final smape
final_smape = smape(y_test, predictions)
print("Final SMAPE:", final_smape)



final_model.get_booster().feature_names = train_.columns.tolist()  
fig, ax = plt.subplots(figsize=(12, 6))  
# Plot importance with actual feature names
xgb.plot_importance(final_model, ax=ax)
plt.show()


y=FeatureEng(test_df)
y.getYear()
y.add_features_to_data()
test_df.tail()



test = test_df.drop(["plate", "date", "region_name", "price"], axis=1)
test_scaled = scale_features(test)


preds = final_model.predict(test_scaled)
preds


submission = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/sample_submission.csv')
submission['price'] = preds
submission


#submitting results
submission.to_csv('submission.csv', index=False)

