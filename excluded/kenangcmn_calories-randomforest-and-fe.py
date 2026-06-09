import numpy as np 
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


train_df.head()


test_df.head()


train_df.info()


train_df.describe()


for col in train_df.select_dtypes(include='float64').columns:
    train_df[col] = train_df[col].astype('float32')
for col in train_df.select_dtypes(include='int64').columns:
    train_df[col] = train_df[col].astype('int32')


print("Missing Values in train data:")
print(train_df.isnull().sum())

print("\nMissing Values in test data:")
print(test_df.isnull().sum())


train_df["Sex"] = train_df["Sex"].astype("category")
test_df["Sex"] = test_df["Sex"].astype("category")


rmv = ["Calories"]
features = [c for c in train_df.columns if c not in rmv]
cats = [c for c in features if train_df[c].dtype == "category"]
nums = [c for c in features if c not in cats]

print(f"Features: {len(features)}\nCategorical: {len(cats)}\nNumerical: {len(nums)}")


train_df["BMI"] = train_df["Weight"] / np.square(train_df["Height"]/100).astype("float32")
test_df["BMI"] = test_df["Weight"] / np.square(test_df["Height"]/100).astype("float32")


def bmi_to_weighttype(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi <= 24.9:
        return "NormalWeight"
    elif bmi <= 29.9:
        return "Overweight"
    else:
        return "Obesity"

train_df["WeightType"] = train_df["BMI"].apply(bmi_to_weighttype).astype("category")
test_df["WeightType"] = test_df["BMI"].apply(bmi_to_weighttype).astype("category")


def age_to_group(age):
    if age <= 18:
        return "Child"
    elif age <= 30:
        return "Young Adult"
    elif age <= 50:
        return "Adult"
    else:
        return "Senior"

train_df["AgeGroup"] = train_df["Age"].apply(age_to_group).astype("category")
test_df["AgeGroup"] = test_df["Age"].apply(age_to_group).astype("category")


rmv = ["Calories"]
features = [c for c in train_df.columns if c not in rmv]
cats = [c for c in features if train_df[c].dtype == "category"]
nums = [c for c in features if c not in cats]

print(f"Features: {len(features)}\nCategorical: {len(cats)}\nNumerical: {len(nums)}")


from sklearn.preprocessing import LabelEncoder

label_encoders = {col: LabelEncoder() for col in cats}

for col in cats:
    train_df[col] = label_encoders[col].fit_transform(train_df[col])
    test_df[col] = label_encoders[col].transform(test_df[col])


print(cats)


def add_feature_crosses(df):
    df = df.copy()
    df["HeartLoad"] = df["Heart_Rate"] * df["Duration"]
    df["TempHeartInteraction"] = df["Body_Temp"] * df["Heart_Rate"]
    df["BMI_Duration"] = df["BMI"] * df["Duration"]
    return df


def add_feature_ratios(df):
    df = df.copy()
    df["Weight_per_Height"] = df["Weight"] / df["Height"]
    df["HeartRate_per_Age"] = df["Heart_Rate"] / df["Age"]
    df['Intensity'] = df['Heart_Rate'] / df['Duration']
    return df


train_df = add_feature_crosses(train_df)
train_df = add_feature_ratios(train_df)

test_df = add_feature_crosses(test_df)
test_df = add_feature_ratios(test_df)


from itertools import combinations
from tqdm.auto import tqdm

encoded_columns = []
encode_columns = ['Sex', 'WeightType', 'AgeGroup']
pair_size = [2,3]

for r in pair_size:
    for cols in tqdm(list(combinations(encode_columns, r))):
        new_col_name = '_'.join(cols)
        
        train_df[new_col_name] = train_df[list(cols)].astype(str).agg('_'.join, axis=1)
        train_df[new_col_name] = train_df[new_col_name].astype('category')
        
        test_df[new_col_name] = test_df[list(cols)].astype(str).agg('_'.join, axis=1)
        test_df[new_col_name] = test_df[new_col_name].astype('category')

        encoded_columns.append(new_col_name)


train_df = train_df.drop(columns=["id"])
test_df = test_df.drop(columns=["id"])
train_df["Calories"] = np.log1p(train_df["Calories"])


rmv = ["Calories"]
features = [c for c in train_df.columns if c not in rmv]
cats = [c for c in features if train_df[c].dtype == "category"]


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold, train_test_split
from colorama import Fore, Style, Back
import gc
import time
from tqdm.auto import tqdm


"""
params = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'n_jobs': [-1]
"""


"""
grid_search = RandomizedSearchCV(RandomForestRegressor(),
                                 param_distributions=params,  
                                 verbose=1,
                                 n_iter=50,       
                                 cv=5,            
                                 random_state=42,
                                 n_jobs=-1)      

grid_search.fit(train_df.iloc[:20000][features], train_df.iloc[:20000][rmv].values.ravel())

best_params = grid_search.best_params_
print(best_params)
"""


def target_encode(train_df, test_df, col, target, stats='mean', prefix='TE'):
    col_name = f"{prefix}_{col}"
    
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

    agg = train_df.groupby(col)[target].agg(stats)

    if isinstance(agg, pd.DataFrame):
        agg = agg.iloc[:, 0]

    test_df[col_name] = test_df[col].map(agg)

    test_df[col_name] = test_df[col_name].fillna(agg.mean())

    train_df[col] = train_df[col].astype('object')
    test_df[col] = test_df[col].astype('object')

    return test_df


print("Garbage is collecting...")
gc.collect()
print("Garbage was collected succesfully!\n")

FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_preds_rf = np.zeros(len(train_df))
test_preds_rf = np.zeros(len(test_df))

rf_params = {'n_jobs': -1,
             'n_estimators': 150,
             'min_samples_split': 2,
             'min_samples_leaf': 2,
             'max_features': 'log2',
             'max_depth': 20}

for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df)):
    print(f"### Fold {fold+1} is processing...###")

    start_time = time.time()

    X_train = train_df.loc[train_idx, features + rmv].reset_index(drop=True)
    y_train = X_train[rmv]
    X_valid, y_valid  = train_df.loc[valid_idx, features].reset_index(drop=True), train_df.loc[valid_idx, rmv].reset_index(drop=True)
    X_test = test_df[features].reset_index(drop=True)

    kf2 = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

    for fold2, (train_idx2, valid_idx2) in enumerate(kf2.split(X_train)):
        train2 = X_train.iloc[train_idx2].copy()
        valid2 = X_train.iloc[valid_idx2][features].copy()

        
        for col in tqdm(encoded_columns, total=len(encoded_columns), desc=f"Second KFold's {fold2+1} / {FOLDS} columns"):
            te_col = f'TE_{col}'
            valid2 = target_encode(train2, valid2, col, rmv, stats='mean', prefix="TE")
            X_train.loc[valid_idx2, te_col] = valid2[te_col].values

        del train2, valid2

    gc.collect()

    for col in encoded_columns:
        X_valid = target_encode(X_train, X_valid, col, rmv, stats='mean', prefix="TE")
        X_test = target_encode(X_train, X_test, col, rmv, stats='mean', prefix="TE")

    te_cols = [f'TE_{col}' for col in encoded_columns]
    X_train.drop(rmv + encoded_columns, axis=1, inplace=True)
    X_valid.drop(encoded_columns, axis=1, inplace=True)
    X_test.drop(encoded_columns, axis=1, inplace=True)

    print("Fitting...\n")
    model = RandomForestRegressor(**rf_params)
    model.fit(X_train, y_train)

    oof_preds_rf[valid_idx] = model.predict(X_valid)
    test_preds_rf += model.predict(X_test) / FOLDS

    gc.collect()
    end_time = time.time()
    time_for_each_fold = end_time - start_time
    mins = time_for_each_fold // 60
    secs = time_for_each_fold % 60
    print(f"Fold {fold+1} completed in {int (mins)} min {int (secs)} s!\n")
rmsle = np.sqrt(mean_squared_log_error(np.expm1(train_df[rmv]), np.expm1(oof_preds_rf)))
print(Fore.GREEN + f"Validation RMSE: {rmsle}"+ Style.RESET_ALL)


sub = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
sub["Calories"] = np.expm1(test_preds_rf)
sub.to_csv("submission.csv", index=False)

