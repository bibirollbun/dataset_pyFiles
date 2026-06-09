!pip install autoviz


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from autoviz import data_cleaning_suggestions
from autoviz import AutoViz_Class
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")


data_cleaning_suggestions(train)


data_cleaning_suggestions(test)


plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(train['Weight Capacity (kg)'], kde=True, bins=30, color='blue', ax=axes[0])
axes[0].set_title('Distribution of Weight Capacity (kg)')
axes[0].set_xlabel('Weight Capacity (kg)')
axes[0].set_ylabel('Frequency')

sns.histplot(train['Price'], kde=True, bins=30, color='green', ax=axes[1])
axes[1].set_title('Distribution of Price')
axes[1].set_xlabel('Price')
axes[1].set_ylabel('Frequency')

plt.show()


plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.countplot(data=train, x='Brand', palette='viridis', ax=axes[0])
axes[0].set_title('Count of Backpacks by Brand')
axes[0].set_xlabel('Brand')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)

sns.countplot(data=train, x='Material', palette='magma', ax=axes[1])
axes[1].set_title('Count of Backpacks by Material')
axes[1].set_xlabel('Material')
axes[1].set_ylabel('Count')

plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(data=train, x='Waterproof', y='Price', palette='pastel')
plt.title('Price Distribution by Waterproof Feature')
plt.xlabel('Waterproof')
plt.ylabel('Price')
plt.show()


plt.figure(figsize=(10, 6))
sns.violinplot(data=train, x='Size', y='Price', palette='cool')
plt.title('Price Distribution by Size')
plt.xlabel('Size')
plt.ylabel('Price')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(data=train, x='Laptop Compartment', y='Price', palette='Set3')
plt.title('Price Distribution by Laptop Compartment')
plt.xlabel('Laptop Compartment')
plt.ylabel('Price')
plt.show()


plt.figure(figsize=(12, 6))
sns.countplot(data=train, x='Color', palette='rainbow')
plt.title('Count of Backpacks by Color')
plt.xlabel('Color')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


class DataProcessing:
    def __init__(self, train, test):
        """Initialize the class with train and test dataframes."""
        self.train_df = train.copy(deep=True)
        self.test_df = test.copy(deep=True)

        self.train_df.drop(columns=['id'], inplace=True, errors='ignore')
        self.test_df.drop(columns=['id'], inplace=True, errors='ignore')

    def numerical_imputer(self, df):
        """Impute missing values in numerical columns using KNNImputer."""
        numerical_columns = ['Weight Capacity (kg)']
        imputer = KNNImputer(n_neighbors=5)
        df[numerical_columns] = imputer.fit_transform(df[numerical_columns])

    def categorical_imputer(self, df):
        """Impute missing values in categorical columns using mode (most frequent value)."""
        categorical_columns = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
        imputer = SimpleImputer(strategy="most_frequent")
        df[categorical_columns] = imputer.fit_transform(df[categorical_columns])

    def encode_categorical(self, df):
        """Encode categorical features using Label Encoding."""
        categorical_columns = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
        label_encoders = {}

        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col]) 
            label_encoders[col] = le
        
        return label_encoders  

    def process_data(self):
        """Apply imputations and encoding to both train and test datasets."""
        self.numerical_imputer(self.train_df)
        self.numerical_imputer(self.test_df)
        self.categorical_imputer(self.train_df)
        self.categorical_imputer(self.test_df)

        train_label_encoders = self.encode_categorical(self.train_df)
        test_label_encoders = self.encode_categorical(self.test_df)

        return self.train_df, self.test_df, train_label_encoders, test_label_encoders


data_processor = DataProcessing(train, test)
train_processed, test_processed, train_encoders, test_encoders = data_processor.process_data()


X = train_processed.drop(columns=['Price'])
y = train_processed['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


space = {
    'max_depth': hp.quniform("max_depth", 3, 18, 1),
    'gamma': hp.uniform('gamma', 1, 9),
    'reg_alpha': hp.quniform('reg_alpha', 40, 180, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
    'n_estimators': 180,
    'seed': 0
}

def objective(space):
    clf = xgb.XGBRegressor(
        n_estimators=int(space['n_estimators']),
        max_depth=int(space['max_depth']),
        gamma=space['gamma'],
        reg_alpha=int(space['reg_alpha']),
        reg_lambda=space['reg_lambda'],
        min_child_weight=int(space['min_child_weight']),
        colsample_bytree=space['colsample_bytree'],
        objective='reg:squarederror',
        enable_categorical=True,
        eval_metric="rmse",
        early_stopping_rounds=10
    )

    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    y_pred = clf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"RMSE: {rmse:.4f}")
    return {'loss': rmse, 'status': STATUS_OK}

trials = Trials()
best_hyperparams = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)


print("\nBest Hyperparameters:", best_hyperparams)


best_params = {
    'n_estimators': 180,
    'max_depth': int(4),
    'gamma': 4.516162159169303,
    'reg_alpha': int(157),
    'reg_lambda': 0.8489669502335644,
    'min_child_weight': int(6),
    'colsample_bytree': 0.5352306987669925,
    'objective': 'reg:squarederror',
    'enable_categorical': True,
    'random_state': 42
}

model = xgb.XGBRegressor(**best_params)
model.fit(X_train, y_train)
y_pred = model.predict(test_processed)

sub['Price'] = y_pred
sub.to_csv("submission.csv", index=False)


test_processed










