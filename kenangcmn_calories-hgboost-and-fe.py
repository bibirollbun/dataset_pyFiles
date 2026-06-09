import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
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


train_df = train_df.drop(columns=["id"])
test_df = test_df.drop(columns=["id"])


rmv = ["Calories"]
features = [c for c in train_df.columns if c not in rmv]
cats = [c for c in features if train_df[c].dtype == "category"]
nums = [c for c in features if c not in cats]

print(f"Total Features: {len(features)}\nCategorical: {len(cats)}\nNumerical: {len(nums)}")


print("Categorical Columns: ", cats)
print("Numerical Columns: ", nums)


from sklearn.preprocessing import LabelEncoder

label_encoders = {col: LabelEncoder() for col in cats}

for col in cats:
    train_df[col] = label_encoders[col].fit_transform(train_df[col])
    test_df[col] = label_encoders[col].transform(test_df[col])


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
    df['log_Duration'] = np.log1p(df['Duration'])
    df['log_Intensity'] = np.log1p(df['Intensity'])
    return df


train_df = add_feature_crosses(train_df)
train_df = add_feature_ratios(train_df)

test_df = add_feature_crosses(test_df)
test_df = add_feature_ratios(test_df)


train_df["Calories"] = np.log1p(train_df["Calories"])


rmv = ["Calories"]
features = [c for c in train_df.columns if c not in rmv]
cats = [c for c in features if train_df[c].dtype == "category"]


from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error


FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_preds_hgb = np.zeros(len(train_df))
test_preds_hgb = np.zeros(len(test_df))

hgb_params = {"learning_rate": 0.025,
              "max_iter": 2500,
              "max_leaf_nodes": 198,
              "min_samples_leaf": 33,
              "l2_regularization": 0.0153, 
              "max_bins": 239,            
              "random_state": 42,
             }

for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df)):
    print(f"### Fold {fold+1} is processing...###")

    X_train, y_train = train_df.iloc[train_idx][features], train_df.iloc[train_idx][rmv]
    X_valid, y_valid = train_df.iloc[valid_idx][features], train_df.iloc[valid_idx][rmv]
    X_test = test_df[features]

    model = HistGradientBoostingRegressor(**hgb_params)
   
    model.fit(X_train, y_train)

    oof_preds_hgb[valid_idx] = model.predict(X_valid)
    test_preds_hgb += model.predict(X_test) / FOLDS
    print("✅")
rmsle = np.sqrt(mean_squared_log_error(np.expm1(train_df[rmv]), np.expm1(oof_preds_hgb)))
print(f"\nValidation RMSLE: {rmsle}")


sub = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
sub["Calories"] = np.clip(np.expm1(test_preds_hgb),1,314)
sub.to_csv("submission.csv", index=False)
print("✅ Submission file is saved succesfully!")

