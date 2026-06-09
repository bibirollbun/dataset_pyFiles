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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

from sklearn.metrics import roc_auc_score, accuracy_score
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

rand_seed = 7


train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
train_df = train_df.drop("id", axis=1)

test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
test_df = test_df.drop("id", axis=1)



train_df.head()



test_df.head()





print("Train Dataset \n")
print(train_df.info())
print("~" * 75)
print("\n Test Dataset \n")
print(test_df.info())



train_df.describe()


test_df.describe()


print("NULL VALUES")

print("\n Train Dataset \n")
print(train_df.isnull().sum())
print("~" * 75)
print("\n Test Dataset \n")
print(test_df.isnull().sum())


cat_cols = train_df.select_dtypes(include="object").columns.to_list()
num_cols = train_df.select_dtypes(include="number").columns.to_list()


for col_name in cat_cols:
    print(
        f"{col_name} \n ***Train_df***-> {train_df[col_name].value_counts()} \n\n ***Test_df*** -> {test_df[col_name].value_counts()} \n\n"
    )


import math

strata_sample = train_df.groupby('diagnosed_diabetes', sort = False).apply(
    lambda x: x.sample(frac=0.15)
).droplevel(0).sample(frac=1, random_state=rand_seed)
 
strata_sample


plt.figure(figsize=(8, 6))
ax = sns.countplot(
    y="diagnosed_diabetes", data = train_df, hue="diagnosed_diabetes", palette="Reds"
)
plt.title("Distribution of Target Variable")
plt.ylabel("diagnosed_diabetes")
plt.xlabel("Count")
plt.legend().remove()
plt.tight_layout()
plt.show()


plt.figure(figsize = (16, 10))

corr_matrix = strata_sample[num_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(data=strata_sample[num_cols].corr(), mask=mask, annot = True, fmt='.2f', cmap = "coolwarm" )
plt.show()


train_df['gender'].astype('category').cat.codes


train_df.groupby(train_df['gender'].astype('category').cat.codes)['diagnosed_diabetes'].mean()


overall_mean = train_df['diagnosed_diabetes'].mean()
np.full(len(train_df), overall_mean, dtype=np.float32).shape


def target_encoding_optimized(train, predict, target, cols, n_splits=10):

    # creating copy of the train and test
    train_result = train.copy()
    predict_result = predict.copy()

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    overall_mean = train[target].mean()

    for col in cols:
        # converting categorical code into numerical to save moemory while processing
        train_codes = train[col].astype('category').cat.codes
        predict_codes  = predict[col].astype('category').cat.codes

        # it calclulate mean for each unique category in 'col' using codes we have calculated above
        code_to_global_mean = train.groupby(train_codes)[target].mean()

        # K-Fold encoding
        mean_encoded = np.full(len(train), overall_mean, dtype=np.float32)

        for tr_idx, val_idx in kfold.split(train):
            tr_codes = train_codes.iloc[tr_idx]
            val_codes = train_codes.iloc[val_idx]
            tr_target = train[target].iloc[tr_idx]
            
            # Calculate fold means
            fold_means = tr_target.groupby(tr_codes).mean()
            
            # Map validation codes to fold means
            val_means = val_codes.map(fold_means)
            
            # Fill missing with global means
            nan_mask = val_means.isna()
            if nan_mask.any():
                val_means[nan_mask] = val_codes.map(code_to_global_mean)[nan_mask]
            
            mean_encoded[val_idx] = val_means.fillna(overall_mean).values
        
        train_result[f'mean_{col}'] = mean_encoded
        
        # Test data encoding
        test_encoded = predict_codes.map(code_to_global_mean).fillna(overall_mean)
        predict_result[f'mean_{col}'] = test_encoded.astype(np.float32).values
    
    return train_result, predict_result



train_encoded, test_encoded = target_encoding_optimized(
    train_df, 
    test_df, 
    target='diagnosed_diabetes', 
    cols=cat_cols,
    n_splits=5
)


train_encoded.head()


test_encoded.head()


train_encoded['employment_status']


def convert_dtype(df):
    df = df.copy()
    df = df.astype({col: "category" for col in df.select_dtypes("object").columns})
    
    return df

train_df_preproc = convert_dtype(train_encoded)
test_df_preproc = convert_dtype(test_encoded)

num_cols_preproc = test_df_preproc.select_dtypes(include="number").columns



num_cols_preproc


train_df_preproc['employment_status']


def encode_categorical(tr_df, ts_df, cols):
    for col in cols:
        le = LabelEncoder()
        tr_df[col] = le.fit_transform(tr_df[col].astype(str))
        ts_df[col] = le.transform(
            ts_df[col]
            .astype(str)
            .map(lambda x: x if x in le.classes_ else le.classes_[0])
        )
    return tr_df, ts_df


# Apply encoding
train, test = encode_categorical(train_df_preproc, test_df_preproc, cat_cols)


scaler = StandardScaler()
train[num_cols_preproc] = scaler.fit_transform(train[num_cols_preproc])
test[num_cols_preproc] = scaler.transform(test[num_cols_preproc])



X = train.drop(columns=["diagnosed_diabetes"])
y = train["diagnosed_diabetes"].astype("int16")
X_test = test


test.info()


X.info()


train_df.info()


train_df.describe()


X.describe()


X.head()


# ## Hyperparameter Tuning (Optuna)

# def objective_lgbm(trial):

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.25, random_state=rand_seed
#     )

#     lgbm_params = {
#         "device": (
#             "gpu" if LGBMClassifier().get_params().get("device") == "gpu" else "cpu"
#         ),
#         "metric": "auc",
#         "n_estimators": trial.suggest_int("n_estimators", 800, 1800),
#         "num_leaves": trial.suggest_int("num_leaves", 70, 110),
#         "max_depth": trial.suggest_int("max_depth", 4, 15),
#         "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.09),
#         "subsample": trial.suggest_uniform("subsample", 0.8, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.9),
#         "reg_alpha": trial.suggest_float("reg_alpha", 6.1, 7.2),
#         "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 0.04),
#     }

#     # Fit the model
#     model_lgbm = LGBMClassifier(**lgbm_params, 
#                                 random_state=rand_seed, 
#                                 verbose=-1)

#     model_lgbm.fit(X_train, y_train)

#     # Predict and calculate accuracy score
#     y_pred = model_lgbm.predict(X_test)

#     return  accuracy_score(y_test, y_pred)


# study_lgbm = optuna.create_study(study_name="LGBM_diabetes", direction="maximize")
# optuna.logging.set_verbosity(optuna.logging.WARNING)
# study_lgbm.optimize(objective_lgbm, n_trials=200, show_progress_bar=True)


# print("Best trial:", study_lgbm.best_trial)


# print("Best parameters:", study_lgbm.best_params)


best_parameters = {'n_estimators': 1419,
                  'num_leaves': 99, 
                  'max_depth': 4, 
                  'learning_rate': 0.0811563529212108, 
                  'subsample': 0.8013276891512052, 
                  'colsample_bytree': 0.880771415168706, 
                  'reg_alpha': 6.519661725929783, 
                  'reg_lambda': 0.03006704024250326,
                  'path_smooth': 17
                  }

model = LGBMClassifier(**best_parameters, random_state=rand_seed, verbose = -1)

model.fit(X, y)


pred_lgb = model.predict_proba(X_test)[:,1]

pred_lgb


submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
submission["diagnosed_diabetes"] = pred_lgb


submission.to_csv("submission.csv", index=False)
submission.head()




