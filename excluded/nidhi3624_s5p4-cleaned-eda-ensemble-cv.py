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

import warnings
warnings.filterwarnings("ignore", category=Warning)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")


# Train inspection

print(train.head())
print(train.info())
print(train.describe(include='all'))
print(f"Shape: {train.shape}")
print(f"Columns: {train.columns.tolist()}")


# Test inspection

print(test.head())
print(test.info())
print(test.describe(include='all'))
print(f"Shape: {test.shape}")
print(f"Columns: {test.columns.tolist()}")


# Sample submission format
print(sample.head())
print(f"Shape: {sample.shape}")
print(f"Columns: {sample.columns.tolist()}")


print("in train dataset")
train.isnull().mean() * 100


print("in test dataset")
test.isnull().mean() * 100


def improved_imputation(df):
    # Add missing indicators
    df['Episode_Length_missing'] = df['Episode_Length_minutes'].isnull().astype(int)
    df['Guest_Popularity_missing'] = df['Guest_Popularity_percentage'].isnull().astype(int)
    
    # Group-aware imputation
    df['Episode_Length_minutes'] = df.groupby('Genre')['Episode_Length_minutes'].transform(
        lambda x: x.fillna(x.median()))
    
    # For guest popularity, assume missing means no guest (0%)
    df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].fillna(0)
    
    # For the single missing ad value
    df['Number_of_Ads'] = df['Number_of_Ads'].fillna(df['Number_of_Ads'].median())
    
    return df

train = improved_imputation(train)
test = improved_imputation(test)


train.isnull().mean() * 100


test.isnull().mean() * 100


print(f"Duplicate rows: {train.duplicated().sum()}")
print(f"Duplicate rows: {test.duplicated().sum()}")


train.info()


numerical_features = ["Episode_Length_minutes", "Host_Popularity_percentage", "Guest_Popularity_percentage", "Number_of_Ads"]
categorical_features = ["Podcast_Name", "Episode_Title", "Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]


print(train["Genre"].value_counts())  
print(" ")
print(train["Publication_Day"].value_counts())
print(" ")
print(train["Publication_Time"].value_counts())
print(" ")
print(train["Episode_Sentiment"].value_counts())
print(" ")


numeric_features = ["Episode_Length_minutes", "Host_Popularity_percentage", "Guest_Popularity_percentage", "Number_of_Ads", "Listening_Time_minutes"]

# Assuming numerical_features is your list of column names
corr_matrix = train[numeric_features].corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix of Selected Numerical Features")
plt.show()


for feature in numerical_features:
    plt.figure(figsize=(12, 2))

    plt.subplot(1, 2, 1)
    plt.title(f'Histogram of {feature} in Train')
    sns.histplot(train[feature], kde=True, bins=100)
    plt.xlabel(feature)
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.title(f'Histogram of {feature} in Test')
    sns.histplot(test[feature], kde=True, bins=100)
    plt.xlabel(feature)
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


target = train["Listening_Time_minutes"]
features = train.drop(["id","Listening_Time_minutes"], axis=1)
test_features = test.drop(["id"], axis=1)


print(features.shape, test_features.shape)


#categorical features from train n test
cat_features = features.select_dtypes(include=['object'])
cat_test_features = test_features.select_dtypes(include=['object'])
print(cat_features.shape, cat_test_features.shape)


#one hot encoding for categorical features
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False, sparse=False, handle_unknown='ignore')
encoder.fit(cat_features)

enc_features = pd.DataFrame(encoder.transform(cat_features), columns=encoder.get_feature_names_out(cat_features.columns),index=cat_features.index)
enc_test_features = pd.DataFrame(encoder.transform(cat_test_features), columns=encoder.get_feature_names_out(cat_test_features.columns), index=cat_test_features.index)

print(enc_features.shape, enc_test_features.shape)


#numeric features in train n test
num_features = features.select_dtypes(include=['float','int'])
num_test_features = test_features.select_dtypes(include=['float','int'])
print(num_features.shape, num_test_features.shape)


#standardising numeric features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(num_features)

sc_features = pd.DataFrame(scaler.transform(num_features), columns = scaler.get_feature_names_out(num_features.columns), index=num_features.index)
sc_test_features = pd.DataFrame(scaler.transform(num_test_features), columns = scaler.get_feature_names_out(num_test_features.columns), index=num_test_features.index)

print(sc_features.shape, sc_test_features.shape)


#concatenating features
final_features = pd.concat([enc_features, sc_features], axis=1)
final_test_features = pd.concat([enc_test_features, sc_test_features], axis=1)
print(final_features.shape, final_test_features.shape)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(final_features, target, test_size=0.2, random_state=42, stratify=target)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape )


from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

xgbr_model = XGBRegressor(
    n_estimators=1000,         
    learning_rate=0.05,        
    max_depth=8,               
    subsample=0.8,             
    colsample_bytree=0.8,      
    reg_alpha=0.1,             
    reg_lambda=1.0,            
    n_jobs=-1,
    random_state=42,
    verbose=0
)

xgbr_model.fit(X_train, y_train)
xgbr_y_pred = xgbr_model.predict(X_test)

xgbr_mse = mean_squared_error(y_test, xgbr_y_pred)
xgbr_rmse = np.sqrt(xgbr_mse)

print("XGBoost Regressor:", xgbr_rmse)


from lightgbm import LGBMRegressor

lgb_model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=128,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)]
)

lgb_y_pred = lgb_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, lgb_y_pred))
print(f"LightGBM RMSE: {rmse:.4f}")


from catboost import CatBoostRegressor

cat_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=0
)

cat_model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
cat_y_pred = cat_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, cat_y_pred))
print(f"CatBoost RMSE: {rmse:.4f}")


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(final_features))
test_preds = np.zeros(len(final_test_features))
rmse_list = []

for fold, (train_idx, val_idx) in enumerate(kf.split(final_features)):
    print(f"Fold {fold + 1}")
    X_train_cv, X_val_cv = final_features.iloc[train_idx], final_features.iloc[val_idx]
    y_train_cv, y_val_cv = target.iloc[train_idx], target.iloc[val_idx]

    model = StackingRegressor(
    estimators=[
        ('xgb', xgbr_model),
        ('lgb', lgb_model),
        ('cat', cat_model)
    ],
    final_estimator=RidgeCV(),  
    n_jobs=-1
)
    
    model.fit(X_train_cv, y_train_cv)
    
    val_preds = model.predict(X_val_cv)
    oof_preds[val_idx] = val_preds
    rmse = mean_squared_error(y_val_cv, val_preds, squared=False)
    rmse_list.append(rmse)
    print(f"Fold {fold + 1} RMSE: {rmse}")

    test_preds += model.predict(final_test_features) / kf.n_splits

print(f"\nAverage CV RMSE: {np.mean(rmse_list)}")



from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV

stack_model = StackingRegressor(
    estimators=[
        ('xgb', xgbr_model),
        ('lgb', lgb_model),
        ('cat', cat_model)
    ],
    final_estimator=RidgeCV(),  
    n_jobs=-1
)

stack_model.fit(X_train, y_train)
stack_pred = stack_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, stack_pred))
print(f"Stacked Model RMSE: {rmse:.4f}")


test_pred = stack_model.predict(final_test_features)
print(test_pred)




submissions = pd.DataFrame({"id":test.id, "Listening_Time_minutes":test_pred})
print(submissions.shape)
submissions.head()


submissions.to_csv("submission.csv", index=False)



sub = pd.read_csv("/kaggle/working/submission.csv")
sub.head()




