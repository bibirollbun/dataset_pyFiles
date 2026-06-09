import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


print(train.shape, test.shape)
train.sample(10)



train.info()


# Check for missing values
train.isnull().sum()
test.isnull().sum()



train.describe()



# Distribution of the target variable
plt.figure(figsize=(10, 4))
sns.histplot(train['Calories'], kde=True, bins=50)
plt.title('Calories Distribution')
plt.show()




# Log distribution (helpful since RMSLE is used)
plt.figure(figsize=(10, 4))
sns.histplot(np.log1p(train['Calories']), kde=True, bins=50, color='orange')
plt.title('Log(1 + Calories) Distribution')
plt.show()


sns.countplot(data=train, x='Sex')
plt.title('Sex Distribution')
plt.show()

sns.boxplot(data=train, x='Sex', y='Calories')
plt.title('Calories by Sex')
plt.show()



numerical_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
correlation = train[numerical_cols + ['Calories']].corr()
print(correlation['Calories'].sort_values(ascending=False))



plt.figure(figsize=(10, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()



sns.scatterplot(data=train, x='Duration', y='Calories', hue='Sex')
plt.title('Duration vs Calories')
plt.show()



for col in numerical_cols:
    sns.histplot(train[col], kde=True)
    plt.title(f'{col} Distribution')
    plt.show()



import numpy as np

# Get common numeric columns between train and test
train_numeric = train.select_dtypes(include=[np.number])
test_numeric = test.select_dtypes(include=[np.number])
common_cols = list(set(train_numeric.columns) & set(test_numeric.columns))

# Remove id and target if present
for col_to_remove in ['id', 'Calories']:
    if col_to_remove in common_cols:
        common_cols.remove(col_to_remove)

# KDE Plot
for col in sorted(common_cols):
    plt.figure(figsize=(6, 4))
    sns.kdeplot(train[col], label='Train', fill=True)
    sns.kdeplot(test[col], label='Test', fill=True)
    plt.title(f'Distribution of {col} in Train vs Test')
    plt.legend()
    plt.show()



train['AgeGroup'] = pd.cut(train['Age'], bins=[0,20,30,40,50,60,70,80], labels=False)
train.groupby('AgeGroup')['Calories'].mean().plot(kind='bar')
plt.title('Average Calories by Age Group')
plt.ylabel('Calories')
plt.show()



# Constant columns
const_cols = [col for col in train.columns if train[col].nunique() == 1]
print("Constant columns:", const_cols)

# Duplicate rows
print("Duplicate rows:", train.duplicated().sum())



import pandas as pd 
import numpy as np 
import os 
import time
import logging 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from category_encoders import TargetEncoder

from tqdm.auto import tqdm
from itertools import combinations
import warnings
warnings.simplefilter('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


numerical_features = ["Age","Height","Weight","Duration","Heart_Rate","Body_Temp"]

def add_feature_cross_terms(df, numerical_features):
    df_new = df.copy()
    for i in range(len(numerical_features)):
        for j in range(i + 1, len(numerical_features)):  
            feature1 = numerical_features[i]
            feature2 = numerical_features[j]
            cross_term_name = f"{feature1}_x_{feature2}"
            df_new[cross_term_name] = df_new[feature1] * df_new[feature2]
    
    return df_new

train = add_feature_cross_terms(train, numerical_features)
test = add_feature_cross_terms(test, numerical_features)


le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])

train["Sex"] = train["Sex"].astype("category")
test["Sex"] = test["Sex"].astype("category")

X = train.drop(columns=["id", "Calories"])
y = np.log1p(train["Calories"])
X_test = test.drop(columns=["id"])


FOLDS = 33
FEATURES = X.columns.tolist()

# KFold setup
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Arrays to store predictions
oof = np.zeros(len(train))
pred = np.zeros(len(test))

# Start CV loop
for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"\n{'#'*10} Fold {i+1} {'#'*10}")
    
    x_train = X.iloc[train_idx].copy()
    y_train = y.iloc[train_idx]
    x_valid = X.iloc[valid_idx].copy()
    y_valid = y.iloc[valid_idx]
    x_test = X_test.copy()

    # No categorical target encoding in this dataset, but you can add if needed
    
    start = time.time()

    # Train model
    model = XGBRegressor(
        device="cuda" if XGBRegressor().get_params().get("device") == "cuda" else "cpu",
        max_depth=10,
        colsample_bytree=0.7,
        subsample=0.9,
        n_estimators=2000,
        learning_rate=0.02,
        gamma=0.01, 
        max_delta_step=2,
        early_stopping_rounds=100,
        eval_metric="rmse",
        enable_categorical=True
    )

    model.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        verbose=100
    )

    # Predict OOF and test
    oof[valid_idx] = model.predict(x_valid)
    pred += model.predict(x_test)

    rmse = np.sqrt(mean_squared_error(y_valid, oof[valid_idx]))
    print(f"Fold {i+1} RMSE: {rmse:.4f}")
    print(f"Feature engineering & training time: {time.time() - start:.1f} sec")

# Average test predictions
pred /= FOLDS

# Final RMSE
full_rmse = np.sqrt(mean_squared_error(y, oof))
print(f"\nFinal CV RMSE: {full_rmse:.4f}")


y_preds = np.expm1(pred)
print('predict mean :',y_preds.mean())
print('predict median :',np.median(y_preds))

y_preds = np.clip(y_preds,1,314)
print('predict mean after clip:',y_preds.mean())
print('predict median after clip:',np.median(y_preds))

submission["Calories"] = y_preds
submission.to_csv("submission.csv", index=False)
submission.head()




