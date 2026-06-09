# !nvidia-smi


# !pip install cupy-cuda12x --extra-index-url=https://pypi.nvidia.com


# !nvcc --version


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import skew, kurtosis
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from cuml.preprocessing import TargetEncoder

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

from xgboost import XGBRegressor
import xgboost as xgb


pd.set_option('display.max_columns',None)


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
extra_train = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")

train.head()


extra_train.head()


train = pd.concat([train, extra_train], ignore_index=True)
train


train.info()


train.describe()


print(train.isnull().sum())
test.isnull().sum()


# missing_values = train_df.isnull().sum()
# missing_values[missing_values > 0]


plt.figure(figsize=(12, 6))
sns.heatmap(train.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values in Training Data')
plt.show()


NUMS = ["Weight Capacity (kg)"]
CATS = train.select_dtypes(include=["object"]).columns.tolist()


TARGET = "Price"


# categorical_cols = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
# for col in categorical_cols:
#     train_df[col].fillna(train_df[col].mode()[0], inplace=True)
#     test_df[col].fillna(test_df[col].mode()[0], inplace=True)

# numerical_cols = ["Compartments", "Weight Capacity (kg)"]
# for col in numerical_cols:
#     train_df[col].fillna(train_df[col].median(), inplace=True)
#     test_df[col].fillna(test_df[col].median(), inplace=True)


train[CATS] = train[CATS].fillna("Missing")
test[CATS] = test[CATS].fillna("Missing")


for col in NUMS:
    train[col + "_missing"] = train[col].isna().astype(int)
    test[col + "_missing"] = test[col].isna().astype(int)
    CATS.append(f"{col}_missing")

    median_value = train[col].median()
    train[col] = train[col].fillna(median_value)
    test[col] = test[col].fillna(median_value)

    train[col + "_int"] = train[col].astype(int)
    test[col + "_int"] = test[col].astype(int)
    CATS.append(f"{col}_int")

    train[col + "_frac"] = ((train[col] - train[col].astype(int)) * 1000).round().astype(int)
    test[col + "_frac"] = ((test[col] - test[col].astype(int)) * 1000).round().astype(int)
    CATS.append(f"{col}_frac")


for col in CATS:
    le = LabelEncoder()
    all_data = pd.concat([train[col], test[col]], axis=0)
    le.fit(all_data)
    train[col] = le.transform(train[col]).astype('int32')
    test[col] = le.transform(test[col]).astype('int32')


numeric_features = NUMS
cat_unique_counts = {col: train[col].nunique() for col in CATS}


missing_values = train.isnull().sum()
missing_values[missing_values > 0]


missing_values = test.isnull().sum()
missing_values[missing_values > 0]


train.describe()


# Checking skewness
print(f"Skewness of Price: {skew(train['Price'])}")
print(f"Kurtosis of Price: {kurtosis(train['Price'])}")


plt.figure(figsize=(10, 5))
sns.histplot(train["Price"], bins=50, kde=True, color="blue")
plt.title("Distribution of Backpack Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(x=train["Price"], color="red")
plt.title("Boxplot of Price")
plt.show()


plt.figure(figsize=(12, 6))
sns.countplot(y=train["Brand"], order=train["Brand"].value_counts().index, palette="coolwarm")
plt.title("Distribution of Brands")
plt.show()


plt.figure(figsize=(10, 5))
sns.countplot(x=train["Material"], order=train["Material"].value_counts().index, palette="viridis")
plt.xticks(rotation=45)
plt.title("Distribution of Materials Used in Backpacks")
plt.show()


plt.figure(figsize=(10, 5))
sns.countplot(x=train["Style"], order=train["Style"].value_counts().index, palette="magma")
plt.title("Distribution of Backpack Styles")
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(train[["Compartments", "Weight Capacity (kg)","Price"]].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


for col in CATS:
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=train, x=col, y='Price', palette='Set2')
    plt.title(f'Price vs {col}')
    plt.xticks(rotation=45)
    plt.show()


plt.figure(figsize=(8, 5))
sns.scatterplot(x=train["Weight Capacity (kg)"], y=train["Price"], alpha=0.5)
plt.title("Price vs Weight Capacity")
plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(x=train["Compartments"], y=train["Price"], palette="Blues")
plt.title("Price vs Number of Compartments")
plt.show()


for col in NUMS:
    plt.figure(figsize=(10, 6))
    sns.histplot(train[col], kde=True, bins=30, color='green')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()


for col in CATS:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=train[col], color='purple')
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
    plt.show()


TE_FEATURES = train.drop(columns=["id", "Price"]).columns.tolist()


n_splits = 10
kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)

test_preds_all = np.zeros((len(test),))
val_rmse_list = []


n_splits = 10
kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)

test_preds_all = np.zeros((len(test),))
val_rmse_list = []

fold = 1
for train_idx, val_idx in kf.split(train):
    print(f"\n----- Fold {fold} -----")

    train_df = train.iloc[train_idx].copy()
    val_df   = train.iloc[val_idx].copy()
    test_df = test.copy()

    y_train = train_df[TARGET].values
    y_val   = val_df[TARGET].values

    TE = TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='mean')
    for col in TE_FEATURES:
        TE.fit(train_df[col], y_train)
        train_df[f"TE_{col}"] = TE.transform(train_df[col])
        val_df[f"TE_{col}"] = TE.transform(val_df[col])
        test_df[f"TE_{col}"] = TE.transform(test_df[col])

    TE_feature_names = [f"TE_{col}" for col in TE_FEATURES]

    X_train_num = train_df[numeric_features].values.astype(np.float32)
    X_val_num   = val_df[numeric_features].values.astype(np.float32)
    X_test_num  = test_df[numeric_features].values.astype(np.float32)

    TE_train = train_df[TE_feature_names].values.astype(np.float32)
    TE_val   = val_df[TE_feature_names].values.astype(np.float32)
    TE_test  = test_df[TE_feature_names].values.astype(np.float32)

    input_dim = X_train_num.shape[1]
    input_layer = Input(shape=(input_dim,), name="input")

    encoded = layers.Dense(64, activation='relu')(input_layer)
    latent = layers.Dense(32, activation='relu', name='latent')(encoded)

    decoded = layers.Dense(64, activation='relu')(latent)
    reconstructed = layers.Dense(input_dim, activation='linear', name='reconstructed')(decoded)

    supervised_branch = layers.Dense(16, activation='relu')(latent)
    supervised_output = layers.Dense(1, activation='linear', name='supervised')(supervised_branch)

    sae = Model(inputs=input_layer, outputs=[reconstructed, supervised_output], name="supervised_autoencoder")
    sae.compile(optimizer='adam',
                loss={'reconstructed': 'mse', 'supervised': 'mse'},
                loss_weights={'reconstructed': 1.0, 'supervised': 1.0})
    sae.summary()

    sae_es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    sae.fit(
        X_train_num, {'reconstructed': X_train_num, 'supervised': y_train},
        validation_data=(X_val_num, {'reconstructed': X_val_num, 'supervised': y_val}),
        epochs=50,
        batch_size=512,
        callbacks=[sae_es],
        verbose=1
    )

    encoder = Model(inputs=input_layer, outputs=latent, name="encoder")
    latent_train = encoder.predict(X_train_num)
    latent_val   = encoder.predict(X_val_num)
    latent_test  = encoder.predict(X_test_num)

    X_train_final = np.hstack([latent_train, TE_train])
    X_val_final   = np.hstack([latent_val, TE_val])
    X_test_final  = np.hstack([latent_test, TE_test])

    print("Fold", fold, "Final feature shape:", X_train_final.shape)

    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        device="cuda",
        early_stopping_rounds=10
    )

    xgb_model.fit(
        X_train_final, y_train,
        eval_set=[(X_val_final, y_val)],
        verbose=500
    )

    val_pred = xgb_model.predict(X_val_final)
    rmse_val = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"Fold {fold} Validation RMSE:", rmse_val)
    val_rmse_list.append(rmse_val)

    test_pred = xgb_model.predict(X_test_final)
    test_preds_all += test_pred

    fold += 1

test_preds_final = test_preds_all / n_splits


print("\nAverage Validation RMSE over folds:", np.mean(val_rmse_list))
print("Final Test Predictions:")
print(test_preds_final)


# label_encoders = {}

# for col in categorical_cols:
#     le = LabelEncoder()
#     train_df[col] = le.fit_transform(train_df[col])
#     test_df[col] = le.transform(test_df[col])
#     label_encoders[col] = le


# correlation_matrix = train_df.corr()
# correlation_matrix


# print("Correlation with Price:")
# display(correlation_matrix['Price'].sort_values(ascending=False))


# X = train_df.drop(["id", "Price"], axis=1)
# y = train_df["Price"]

# X_test = test_df.drop("id", axis=1)

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)

# xgb_model.fit(X_train, y_train)


# y_val_pred = xgb_model.predict(X_val)

# mse = mean_squared_error(y_val, y_val_pred)
# rmse = np.sqrt(mse)
# print(f"Validation RMSE: {rmse:.4f}")


submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
submission.Price = test_preds_final

submission.to_csv("submission.csv", index=False)
submission.head(50)




