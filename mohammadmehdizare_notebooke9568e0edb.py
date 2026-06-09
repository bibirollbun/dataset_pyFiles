! pip install miceforest


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import numpy as np, pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from miceforest import ImputationKernel

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from lightgbm.callback import early_stopping

from scipy.stats import rankdata

# Import pandas profiling library
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)



data_dictionary = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv")
data_dictionary


test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
print("Test shape:", test.shape )

train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
print("Train shape:",train.shape)
train.head()


train.info()


train.describe()


df = pd.DataFrame(columns=['Column', 'Data Type' , 'Missing Count', 'Percentage'])

# Get columns with missing values
missing_columns = train.columns[train.isnull().any()].tolist()
for col in missing_columns:
    data_type = data_dictionary[data_dictionary['variable']==col]['type']
    missin_count = train[col].isnull().sum()
    total_count = len(train[col])
    pencentage = missin_count*100/total_count
    df.loc[len(df)] = [col,data_type,missin_count,pencentage]

df.sort_values(by='Percentage',ascending=False)


train = train.drop(['tce_match','mrd_hct'],axis=1)
test = test.drop(['tce_match','mrd_hct'],axis=1)


df = df[df['Column'] != 'tce_match']
df = df[df['Column'] != 'mrd_hct']

missing_columns.remove('tce_match')
missing_columns.remove('mrd_hct')
fig = plt.figure(figsize=(20,5))
sns.barplot(data=df,x='Column',y='Missing Count')
plt.xticks(rotation=45)
plt.show()


plt.hist(train.loc[train['efs']==1,"efs_time"],bins=100,label="efs=1, Did Not Survive")
plt.hist(train.loc[train['efs']==0,"efs_time"],bins=100,label="efs=0, Maybe Survived")
plt.xlabel("Time of Observation, efs_time")
plt.ylabel("Density")
plt.title("Times of Observation. Either time to death, or time observed alive.")
plt.legend()
plt.show()


def make_new_hla(train):
  train['hla_nmdp_6'] = (train['hla_match_a_low'].fillna(0) + train['hla_match_b_low'].fillna(0) + train['hla_match_drb1_high'].fillna(0))

  train['hla_low_res_6'] = (train['hla_match_a_low'].fillna(0) + train['hla_match_b_low'].fillna(0) + train['hla_match_drb1_low'].fillna(0))

  train['hla_high_res_6'] = (train['hla_match_a_high'].fillna(0) + train['hla_match_b_high'].fillna(0) + train['hla_match_drb1_high'].fillna(0))

  train['hla_low_res_8'] = (train['hla_match_a_low'].fillna(0) + train['hla_match_b_low'].fillna(0) + train['hla_match_c_low'].fillna(0)
                              + train['hla_match_drb1_low'].fillna(0))

  train['hla_high_res_8'] = (train['hla_match_a_high'].fillna(0) + train['hla_match_b_high'].fillna(0) + train['hla_match_c_high'].fillna(0)
                              + train['hla_match_drb1_high'].fillna(0))

  train['hla_low_res_10'] = (train['hla_match_a_low'].fillna(0) + train['hla_match_b_low'].fillna(0) + train['hla_match_c_low'].fillna(0)
                              + train['hla_match_drb1_low'].fillna(0) + train['hla_match_dqb1_low'].fillna(0) )

  train['hla_high_res_10'] = (train['hla_match_a_high'].fillna(0) + train['hla_match_b_high'].fillna(0) + train['hla_match_c_high'].fillna(0)
                              + train['hla_match_drb1_high'].fillna(0) + train['hla_match_dqb1_high'].fillna(0))

  train = train.drop(['hla_match_a_low','hla_match_b_low','hla_match_c_low','hla_match_a_high','hla_match_b_high','hla_match_c_high',
                        'hla_match_drb1_high','hla_match_drb1_low','hla_match_dqb1_high','hla_match_dqb1_low'],axis=1)

  return train



train= make_new_hla(train)
test=make_new_hla(test)


train.shape


# Distribution of target variable
plt.figure(figsize=(8, 6))
sns.countplot(x='efs', data=train)
plt.title('Distribution of efs (Target Variable)')
plt.show()

# Boxplots for numerical features
numerical_features = train.select_dtypes(include=np.number).columns
for col in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='efs', y=col, data=train)
    plt.title(f'Boxplot of {col} vs. efs')
    plt.show()



categorical_features = train.select_dtypes(exclude=np.number).columns
for col in categorical_features:
  print(f"Value counts for {col}:")
  print(train[col].value_counts())
  plt.figure(figsize=(8, 6))
  sns.countplot(x=col, hue='efs', data=train)
  plt.title(f'Distribution of {col} vs efs')
  plt.xticks(rotation=45, ha='right')
  plt.show()



# Combine train and test for consistent encoding
combined_df = pd.concat([train, test], axis=0)
combined_encoded = pd.get_dummies(combined_df ,dummy_na=True)


import re
combined_encoded.columns = [re.sub(r'[!@#$%^&*(){}\[\];:,./<>?\\|`~\=_\']', '_', col) for col in combined_encoded.columns]
mice_kernel = ImputationKernel(
data = combined_encoded.reset_index(drop=True),
random_state = 42
)
mice_kernel.mice(2)
combined_encoded = mice_kernel.complete_data()
combined_encoded.describe()


# Split back into train and test
train_encoded = combined_encoded.iloc[:len(train)]
test_encoded = combined_encoded.iloc[len(train):]



test_encoded.head()


sns.boxplot(data= train_encoded[train_encoded['efs']==1], x='efs_time')


sns.boxplot(data= train_encoded[train_encoded['efs']==0], x='efs_time')



upper_limit = train_encoded['efs_time'].mean() + 3*train_encoded['efs_time'].std()
lower_limit = train_encoded['efs_time'].mean() - 3*train_encoded['efs_time'].std()

df_train_clean = train_encoded.loc[(train_encoded['efs_time'] <= upper_limit) & (train_encoded['efs_time'] >= lower_limit)]

df_train_original = train_encoded
df_train = df_train_clean
df_train.reset_index(drop=True, inplace=True)


sns.boxplot(data= df_train_clean[df_train_clean['efs']==0], x='efs_time')



print(f'Training data shape : {df_train.shape}')
print(f'Test data shape : {test_encoded.shape}')


T=df_train['efs_time']
E=df_train['efs']
X=df_train.drop(['efs','efs_time','ID'],axis=1)
X_test=test_encoded.drop(['efs','efs_time',"ID"],axis=1)


# Robust scaling
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Prediction storage
test_predictions = np.zeros(len(X_test))

# Advanced LightGBM parameters
params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'dart',
    'num_leaves': 127,
    'learning_rate': 0.01,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': 10,
    'min_data_in_leaf': 20,
    'lambda_l1': 0.5,
    'lambda_l2': 0.5,
    'verbosity': -1
}

# Cross-validation
for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, E), 1):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    T_train, T_val = T.iloc[train_idx], T.iloc[val_idx]
    E_train, E_val = E.iloc[train_idx], E.iloc[val_idx]

    train_data = lgb.Dataset(
        X_train,
        label=-T_train,
        weight=E_train
    )

    val_data = lgb.Dataset(
        X_val,
        label=-T_val,
        weight=E_val
    )

    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
    )
    #print metrics
    print(f'Fold {fold} - MAE: {model.best_score["valid_1"]["l1"]}')
    # Predict test data
    test_predictions += model.predict(X_test_scaled) / 5


predictions= test_predictions


# Create submission with normalization
submission = pd.DataFrame({
    'ID': test_encoded["ID"],
    'prediction': (predictions - predictions.min()) / (predictions.max() - predictions.min())
})

# Save submission
submission.to_csv('submission_lgb.csv', index=False)


print("Submission Preview:")
print(submission.head())
print("\nPrediction Statistics:")
print(submission['prediction'].describe())


from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

FOLDS = 5
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(X))
pred_efs = np.zeros(len(test_encoded))

for i, (train_index, test_index) in enumerate(kf.split(X, E)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)

    x_train = X.iloc[train_index]  # Changed line: Using .iloc to select rows by index
    y_train = E.iloc[train_index] # Changed line: Using .iloc to select rows by index "efs"]
    x_valid = X.iloc[test_index]  # Changed line: Using .iloc to select rows by index
    y_valid = E.iloc[test_index] # Changed line: Using .iloc to select rows by index "efs"]
    x_test = test_encoded.drop(['efs','efs_time',"ID"],axis=1)

    model_xgb = XGBClassifier(
        device="cuda",
        max_depth=3,
        colsample_bytree=0.7129400756425178,
        subsample=0.8185881823156917,
        n_estimators=20_000,
        learning_rate=0.04425768131771064,
        eval_metric="auc",
        early_stopping_rounds=50,
        objective='binary:logistic',
        scale_pos_weight=1.5379160847615545,
        min_child_weight=4,
        enable_categorical=True,
        gamma=3.1330719334577584
    )
    model_xgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        verbose=100
    )

    # INFER OOF (Probabilities -> Binary)
    oof_xgb[test_index] = (model_xgb.predict_proba(x_valid)[:, 1] > 0.5).astype(int)
    # INFER TEST (Probabilities -> Average Probs)
    pred_efs += model_xgb.predict_proba(x_test)[:, 1]

# COMPUTE AVERAGE TEST PREDS
pred_efs = (pred_efs / FOLDS > 0.5).astype(int)



# EVALUATE PERFORMANCE
accuracy = accuracy_score(E, oof_xgb)
f1 = f1_score(E, oof_xgb)
roc_auc = roc_auc_score(E, oof_xgb)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")


train=df_train
train["y"] = train.efs_time.values
mx = train.loc[train.efs==1,"efs_time"].max()
mn = train.loc[train.efs==0,"efs_time"].min()
train.loc[train.efs==0,"y"] = train.loc[train.efs==0,"y"] + mx - mn
train.y = train.y.rank()
train.loc[train.efs==0,"y"] += len(train)//2
train.y = train.y / train.y.max()

plt.hist(train.loc[train.efs==1,"y"],bins=100,label="efs=1, Did Not Survive")
plt.hist(train.loc[train.efs==0,"y"],bins=100,label="efs=0, Maybe Survived")
plt.xlabel("Transformed Target y")
plt.ylabel("Density")
plt.title("Transformed Target y using both efs and efs_time.")
plt.legend()
plt.show()


y=train.y
X=train.drop(['efs','efs_time','ID','y'],axis=1)
X_test=test_encoded.drop(['efs','efs_time',"ID"],axis=1)


%%time
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)
oof_xgb = np.zeros(len(train))
pred_xgb = np.zeros(len(test_encoded))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)

    x_train = X[train_index]
    y_train = y[train_index]
    x_valid = X[test_index]
    y_valid = y[test_index]
    x_test = test_encoded.drop(['efs','efs_time',"ID"],axis=1)

    model_xgb = XGBRegressor(
        device="cpu",
        max_depth=5,
        colsample_bytree=0.4309907360736148,
        subsample=0.6727848987288046,
        n_estimators=10_000,
        learning_rate=0.03509792076095853,
        eval_metric="mae",
        early_stopping_rounds=25,
        objective='reg:logistic',
        enable_categorical=True,
        min_child_weight=10,
        reg_alpha= 2.950200470036872,
        reg_lambda= 1.484334590329492,
        gamma = 0.008314053362236895
    )
    model_xgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        verbose=100
    )

    # INFER OOF
    oof_xgb[test_index] = model_xgb.predict(x_valid)
    # INFER TEST
    pred_xgb += model_xgb.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_xgb /= FOLDS


feature_importance = model_xgb.feature_importances_
cols=train.columns
cols=cols.drop(['efs','efs_time','ID','y'])
importance_df = pd.DataFrame({
    "Feature": cols,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)[:10]
plt.figure(figsize=(10, 15))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance")
plt.gca().invert_yaxis()  # Flip features for better readability
plt.show()


pred_xgb


submission = pd.DataFrame({
    'ID': test_encoded["ID"],
    'prediction':pred_xgb})
submission.to_csv('submission.csv', index=False)


submission




