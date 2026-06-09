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


# Import libraries
import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
import tensorflow as tf
from tensorflow.keras.layers import Normalization, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import warnings
warnings.filterwarnings("ignore")


# Import data
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
origin = pd.read_csv("/kaggle/input/calories-burnt-prediction/calories.csv")


# Rename columns to match train data
origin.rename(columns={"User_ID": "id", "Gender": "Sex"}, inplace=True)


# Drop "id"
train.drop("id", axis=1, inplace=True)
test.drop("id", axis=1, inplace=True)
origin.drop("id", axis=1, inplace=True)


# Concat train and original data
train = pd.concat([train, origin], axis=0, ignore_index=True)
train.info()


# Numeric and Categorical features
categorical_features = [col for col in train.columns if train[col].dtype in ["object", "category"]]
numeric_features = [col for col in train.columns if train[col].dtype in ["int64", "float64"]]

print(f"Categorical Features:\t{categorical_features}")
print(f"Numeric Features:\t{numeric_features}")


# Target feature
target = "Calories"

# Remove "Calories" from numeric_features
numeric_features.remove(target)


# Target feature
plt.figure(figsize=(14, 5))
sns.histplot(train["Calories"], kde=True, bins=30)
plt.title("Distribution of Calories")
plt.show()

# Summary statistics
print(f"Statistics:")
print(train["Calories"].describe())

# Skewness and Kurtosis
print(f"Skewness:\t{train['Calories'].skew():.3f}")
print(f"Kurtosis:\t{train['Calories'].kurtosis():.3f}")


# Calorie burn comparison between sex
# Statistics
print(train.groupby("Sex")["Calories"].describe())

# Visulaization
plt.figure(figsize=(14, 5))
sns.boxplot(data=train, x="Sex", y="Calories")
plt.title("Calories by Sex")
plt.show()

# t-test for group differences
male_cals = train[train['Sex'] == 'male']['Calories']
female_cals = train[train['Sex'] == 'female']['Calories']
t_stat, p_val = stats.ttest_ind(male_cals, female_cals)
print(f"T-test p-value:\t{p_val:.4f}")


# Create new features
def create_new_features(df, features):
    new_df = df.copy()
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            f1, f2 = features[i], features[j]
            new_df[f"{f1}x{f2}"] = df[f1] * df[f2]

    return new_df

train = create_new_features(train, numeric_features)
test = create_new_features(test, numeric_features)


# Physiological metrics
# def add_bmi(df):
#     df_copy = df.copy()
#     df_copy['BMI'] = df_copy['Weight'] / (df_copy['Height']/100) ** 2
#     return df_copy

# train = add_bmi(train)
# test = add_bmi(test)


# Prepare X, y for fold
# Encode categorical feature
le = LabelEncoder()
train["Sex"] = le.fit_transform(train["Sex"])
test["Sex"] = le.fit_transform(test["Sex"])

X = train.drop(target, axis=1)
y = np.log1p(train[target])


# Custom Root mean squared log error
def rmsle(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(tf.math.log1p(y_true) - tf.math.log1p(y_pred))))


# Callbacks
def make_callbacks():
    lr_callback = ReduceLROnPlateau(
        monitor='val_rmsle',     
        factor=0.5,              
        patience=3,              
        verbose=1,               
        min_lr=1e-6              
    )
    early_stop_cb = EarlyStopping(
        monitor="val_rmsle", 
        patience=15,            
        restore_best_weights=True,
        mode="min", 
        verbose=1
    )
    return [lr_callback, early_stop_cb]


# KFold setup
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
models, rmsle_scores = [], [] # Store model and scores
oof = np.zeros(len(train))
preds = np.zeros(len(test))

# KFold loop
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n{'-'*15}Fold {fold+1} / {n_splits}{'-'*15}")

    # Split the data
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    X_test = test.copy()

    # Normalization to adapt data mean and variance
    normalizer = Normalization()
    normalizer.adapt(X_train.values)

    # Model architecture
    model = Sequential([
        normalizer,
        Dense(256, activation='swish', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(128, activation='swish'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='swish'),
        Dropout(0.2),
        
        Dense(32, activation='swish'),
        
        Dense(1, activation='relu')  # Ensures non-negative predictions
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=rmsle,
        metrics=[rmsle]
    )

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=128,
        callbacks=make_callbacks(),
        verbose=1
    )

    models.append(model)
    oof[val_idx] = model.predict(X_val, batch_size=256).flatten()
    preds += model.predict(X_test, batch_size=512).flatten()
    score = np.sqrt(mean_squared_log_error(oof[val_idx], y_val))
    rmsle_scores.append(score)
    print(f"\n{fold+1} RMSLE Score:\t{score}")


preds /= n_splits
print("\nCross-Validation Results:")
print(f"Mean RMSLE: {np.mean(rmsle_scores):.4f} ± {np.std(rmsle_scores):.4f}")


# Submission
submission["Calories"] = np.expm1(preds)
submission.to_csv("submission.csv", index=False)
submission.head(10)




