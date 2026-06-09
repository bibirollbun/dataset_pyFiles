import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import gc
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Visualization Libraries
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import squarify
%matplotlib inline

colors= ['#1c76b6', '#a7dae9', '#eb6a20', '#f59d3d', '#677fa0', '#d6e4ed', '#f7e9e5']
sns.set_palette(colors)


file_path = "/kaggle/input/mlx-2-0-regression/train.csv" 
train = pd.read_csv(file_path)
train


file_path = "/kaggle/input/mlx-2-0-regression/test.csv" 
test = pd.read_csv(file_path)
test


file_path = "/kaggle/input/mlx-2-0-regression/sample_submission.csv" 
sample_submission = pd.read_csv(file_path)
sample_submission


ids = test['id']
train = train.drop(columns=['id'], axis=1)
test = test.drop(columns=['id'], axis=1)


train = train.drop_duplicates()


train


def customDescription(df: pd.DataFrame, numeric_only: bool = False):
    if numeric_only:
        df = df.select_dtypes(include=np.number)
    
    desc = pd.DataFrame(index=df.columns.to_list())
    desc['type'] = df.dtypes
    desc['count'] = df.count()
    desc['nunique'] = df.nunique()
    desc['null'] = df.isnull().sum()
    
    # Calculate mode and handle multiple modes
    modes = df.mode()
    desc['mode'] = np.nan  # Default to NaN
    for col in df.columns:
        if len(modes[col].dropna()) == 1:  # Single mode exists
            desc.loc[col, 'mode'] = modes[col].iloc[0]
        else:  # Multiple modes
            desc.loc[col, 'mode'] = np.nan

    # Calculate least frequent value
    desc['least_frequent'] = np.nan  # Default to NaN
    for col in df.columns:
        value_counts = df[col].value_counts(dropna=False)
        if not value_counts.empty:
            least_freq_count = value_counts.min()  # Find the minimum frequency
            least_freq_values = value_counts[value_counts == least_freq_count].index
            
            if len(least_freq_values) == 1:  # If exactly one least frequent value exists
                desc.loc[col, 'least_frequent'] = least_freq_values[0]
            else:  # Multiple least frequent values
                desc.loc[col, 'least_frequent'] = np.nan
    
    # Handle numeric columns
    numeric_cols = df.select_dtypes(include=np.number)
    if not numeric_cols.empty:
        numeric_desc = numeric_cols.describe().T.drop(columns=['count', 'std', '25%', '50%', '75%'], axis=1)
        for col in numeric_cols.columns:
            desc.loc[col, 'mean'] = numeric_desc.loc[col, 'mean']
            desc.loc[col, 'min'] = numeric_desc.loc[col, 'min']
            desc.loc[col, 'max'] = numeric_desc.loc[col, 'max']
    
    # Handle datetime columns
    datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]'])
    for col in datetime_cols.columns:
        desc.loc[col, 'min'] = df[col].min()
        desc.loc[col, 'max'] = df[col].max()
    
    return desc


features = train.drop(columns=['target'])


customDescription(features)


customDescription(test)


target_column = 'target'
categorical_columns = train.select_dtypes(include=['object']).columns
numerical_columns = train.select_dtypes(exclude=['object']).columns.drop(target_column)

print("Target Column:", target_column)
print("\nCategorical Columns:", categorical_columns.tolist())
print("\nNumerical Columns:", numerical_columns.tolist())


# counts = train['publication_timestamp'].value_counts().reset_index()
# counts.columns = ['publication_timestamp', 'Frequency']
# counts


label = 'composition_label_0'

train_labels = set(train[label].unique())
test_labels = set(test[label].unique())

common_labels = train_labels.intersection(test_labels)

train[label] = train[label].apply(lambda x: x if x in common_labels else 'Other')
test[label] = test[label].apply(lambda x: x if x in common_labels else 'Other')

counts = train[label].value_counts().reset_index()
counts.columns = [label, 'Frequency']

rare_labels = counts[counts['Frequency'] <= 5][label]

train[label] = train[label].apply(lambda x: 'Rare' if x in rare_labels.values else x)

counts = test[label].value_counts().reset_index()
counts.columns = [label, 'Frequency']

rare_labels = counts[counts['Frequency'] <= 5][label]

test[label] = test[label].apply(lambda x: 'Rare' if x in rare_labels.values else x)


label = 'composition_label_1'

train_labels = set(train[label].unique())
test_labels = set(test[label].unique())

common_labels = train_labels.intersection(test_labels)

train[label] = train[label].apply(lambda x: x if x in common_labels else 'Other')
test[label] = test[label].apply(lambda x: x if x in common_labels else 'Other')

counts = train[label].value_counts().reset_index()
counts.columns = [label, 'Frequency']

rare_labels = counts[counts['Frequency'] <= 5][label]

train[label] = train[label].apply(lambda x: 'Rare' if x in rare_labels.values else x)

counts = test[label].value_counts().reset_index()
counts.columns = [label, 'Frequency']

rare_labels = counts[counts['Frequency'] <= 5][label]

test[label] = test[label].apply(lambda x: 'Rare' if x in rare_labels.values else x)


label = 'composition_label_1'

train_labels = set(train[label].unique())
test_labels = set(test[label].unique())

common_labels = train_labels.intersection(test_labels)

train[label] = train[label].apply(lambda x: x if x in common_labels else 'Other')
test[label] = test[label].apply(lambda x: x if x in common_labels else 'Other')

counts = train[label].value_counts().reset_index()
counts.columns = [label, 'Frequency']

rare_labels = counts[counts['Frequency'] <= 5][label]

train[label] = train[label].apply(lambda x: 'Rare' if x in rare_labels.values else x)

counts = test[label].value_counts().reset_index()
counts.columns = [label, 'Frequency']

rare_labels = counts[counts['Frequency'] <= 5][label]

test[label] = test[label].apply(lambda x: 'Rare' if x in rare_labels.values else x)


label = 'creator_collective'

train_labels = set(train[label].unique())
test_labels = set(test[label].unique())

common_labels = train_labels.intersection(test_labels)

train[label] = train[label].apply(lambda x: x if x in common_labels else 'Other')
test[label] = test[label].apply(lambda x: x if x in common_labels else 'Other')

counts = train[label].value_counts().reset_index()
counts.columns = [label, 'Frequency']

rare_labels = counts[counts['Frequency'] <= 2][label]

train[label] = train[label].apply(lambda x: 'Rare' if x in rare_labels.values else x)

counts = test[label].value_counts().reset_index()
counts.columns = [label, 'Frequency']

rare_labels = counts[counts['Frequency'] <= 2][label]

test[label] = test[label].apply(lambda x: 'Rare' if x in rare_labels.values else x)


label = 'composition_label_2'

train_labels = set(train[label].unique())
test_labels = set(test[label].unique())

common_labels = train_labels.intersection(test_labels)

train[label] = train[label].apply(lambda x: x if x in common_labels else 'Other')
test[label] = test[label].apply(lambda x: x if x in common_labels else 'Other')

counts = train[label].value_counts().reset_index()
counts.columns = [label, 'Frequency']

rare_labels = counts[counts['Frequency'] <= 5][label]

train[label] = train[label].apply(lambda x: 'Rare' if x in rare_labels.values else x)

counts = test[label].value_counts().reset_index()
counts.columns = [label, 'Frequency']

rare_labels = counts[counts['Frequency'] <= 5][label]

test[label] = test[label].apply(lambda x: 'Rare' if x in rare_labels.values else x)


label = 'track_identifier'

train_labels = set(train[label].unique())
test_labels = set(test[label].unique())

common_labels = train_labels.intersection(test_labels)

train[label] = train[label].apply(lambda x: x if x in common_labels else 'Other')
test[label] = test[label].apply(lambda x: x if x in common_labels else 'Other')

counts = train[label].value_counts().reset_index()
counts.columns = [label, 'Frequency']

rare_labels = counts[counts['Frequency'] <= 5][label]

train[label] = train[label].apply(lambda x: 'Rare' if x in rare_labels.values else x)

counts = test[label].value_counts().reset_index()
counts.columns = [label, 'Frequency']

rare_labels = counts[counts['Frequency'] <= 5][label]

test[label] = test[label].apply(lambda x: 'Rare' if x in rare_labels.values else x)


# # Convert to datetime
# train['publication_timestamp'] = pd.to_datetime(train['publication_timestamp'], errors='coerce')

# # Get the earliest date in the column as the reference
# reference_date = train['publication_timestamp'].min()

# # Extract features
# train['pub_year'] = train['publication_timestamp'].dt.year
# train['pub_month'] = train['publication_timestamp'].dt.month
# train['pub_day'] = train['publication_timestamp'].dt.day
# train['pub_dayofweek'] = train['publication_timestamp'].dt.dayofweek  # 0=Monday
# train['pub_weekofyear'] = train['publication_timestamp'].dt.isocalendar().week
# train['pub_quarter'] = train['publication_timestamp'].dt.quarter
# train['pub_is_weekend'] = train['pub_dayofweek'].isin([5, 6]).astype(int)

# # Days since the earliest publication
# train['days_since_first_pub'] = (train['publication_timestamp'] - reference_date).dt.days


# train = train.drop(columns = ['weekday_of_release','publication_timestamp'])
train = train.drop(columns = ['publication_timestamp'])


# # Convert to datetime
# test['publication_timestamp'] = pd.to_datetime(test['publication_timestamp'], errors='coerce')

# # Extract features
# test['pub_year'] = test['publication_timestamp'].dt.year
# test['pub_month'] = test['publication_timestamp'].dt.month
# test['pub_day'] = test['publication_timestamp'].dt.day
# test['pub_dayofweek'] = test['publication_timestamp'].dt.dayofweek  # 0=Monday
# test['pub_weekofyear'] = test['publication_timestamp'].dt.isocalendar().week
# test['pub_quarter'] = test['publication_timestamp'].dt.quarter
# test['pub_is_weekend'] = test['pub_dayofweek'].isin([5, 6]).astype(int)

# # Days since the earliest publication
# test['days_since_first_pub'] = (test['publication_timestamp'] - reference_date).dt.days


# test = test.drop(columns = ['weekday_of_release','publication_timestamp'])
test = test.drop(columns = ['publication_timestamp'])


cols_to_encode = ['composition_label_0', 'composition_label_1', 'creator_collective', 'composition_label_2', 'track_identifier']

for col in cols_to_encode:
    # Compute frequencies on train only
    freq = train[col].value_counts(normalize=True)

    # Replace original column with frequency-encoded values
    train[col] = train[col].map(freq)
    test[col] = test[col].map(freq)

test[cols_to_encode] = test[cols_to_encode].fillna(0)


# from sklearn.preprocessing import LabelEncoder

# # Categorical columns to encode
# cat_cols = ['composition_label_0', 'composition_label_1',
#             'season_of_release', 'lunar_phase', 'creator_collective',
#             'composition_label_2', 'track_identifier']

# # Initialize a dictionary to store encoders for each column
# encoders = {}

# # Apply Label Encoding
# for col in cat_cols:
#     le = LabelEncoder()
    
#     # Fit on combined data to cover all categories
#     combined = pd.concat([train[col], test[col]], axis=0).astype(str)
#     le.fit(combined)
    
#     # Transform train and test
#     train[col] = le.transform(train[col].astype(str))
#     test[col] = le.transform(test[col].astype(str))
    
#     # Save the encoder if you need it later (e.g., for inverse transform)
#     encoders[col] = le


from sklearn.preprocessing import LabelEncoder

# Categorical columns to encode
cat_cols = ['season_of_release', 'lunar_phase', 'weekday_of_release']

# Initialize a dictionary to store encoders for each column
encoders = {}

# Apply Label Encoding
for col in cat_cols:
    le = LabelEncoder()
    
    # Fit on combined data to cover all categories
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    
    # Transform train and test
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    
    # Save the encoder if you need it later (e.g., for inverse transform)
    encoders[col] = le


from sklearn.feature_selection import mutual_info_regression

def make_mi_scores(X, y):
    X = X.copy()
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


train = train.fillna(-1)


test = test.fillna(-1)


# X = train.drop(columns=['target'], axis=1)  
# y = train['target']

# mi_scores = make_mi_scores(X, y)

# print(mi_scores)
# # print(mi_scores.tail(20))  

# plt.figure(dpi=100, figsize=(8, 5))
# plot_mi_scores(mi_scores)
# # plot_mi_scores(mi_scores.tail(20))  


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = train.drop(columns=['target'], axis=1)  
y = train['target']
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import numpy as np
import gc
import warnings
warnings.filterwarnings("ignore")

def get_model_predictions(X, y, df_test, model_func):
    test_preds = np.zeros(len(df_test))
    val_preds = np.zeros(len(X))
    cv = KFold(n_splits=10, shuffle=True, random_state=9)

    for fold, (train_ind, valid_ind) in enumerate(cv.split(X, y)):
        X_train, y_train = X.iloc[train_ind], y.iloc[train_ind]
        X_val, y_val = X.iloc[valid_ind], y.iloc[valid_ind]

        model = model_func()

        # Fit based on model type
        if model_func == lgb_model:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(-1)]
            )
        elif model_func == xgb_model:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=0
            )
        elif model_func == catboost_model:
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val)
            )
        else:  # For RF and ET
            model.fit(X_train, y_train)

        gc.collect()

        y_pred_val = model.predict(X_val)
        y_pred_val = y_pred_val.round()

        # Evaluate
        rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        mae = mean_absolute_error(y_val, y_pred_val)
        r2 = r2_score(y_val, y_pred_val)

        print("-" * 60)
        print(f"{model_func.__name__} Fold {fold}")
        print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
        print("-" * 60)
        
        val_preds[valid_ind] = y_pred_val
        test_preds += model.predict(df_test) / cv.n_splits
        gc.collect()

    test_preds = test_preds.round()
    return val_preds, test_preds


def lgb_model():
    return lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.1, device='gpu')

def xgb_model():
    return xgb.XGBRegressor(n_estimators=1000, learning_rate=0.1, tree_method='gpu_hist',predictor='gpu_predictor')

def catboost_model():
    return CatBoostRegressor(
        iterations=1000,
        learning_rate=0.1,
        task_type='GPU',
        eval_metric='RMSE',
        verbose = 0
    )

def rf_model():
    return RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

def et_model():
    return ExtraTreesRegressor(
        random_state=42,
        n_jobs=-1,
        verbose=0
    )


print("1. XGBRegressor")
xgb_val_preds, xgb_test_preds = get_model_predictions(X, y, test, xgb_model)


print("2. LGBMRegressor")
lgb_val_preds, lgb_test_preds = get_model_predictions(X, y, test, lgb_model)


# cat_cols = [
#     'composition_label_0', 'composition_label_1', 'creator_collective',
#     'composition_label_2', 'track_identifier',
#     'pub_year', 'pub_month', 'pub_dayofweek', 'pub_is_weekend',
#     'season_of_release', 'lunar_phase'
# ]

# for col in cat_cols:
#     X[col] = X[col].astype(str)
#     test[col] = test[col].astype(str)
    
# # 1. Convert to category dtype
# X[cat_cols] = X[cat_cols].astype('category')
# test[cat_cols] = test[cat_cols].astype('category')

# # 2. Get categorical feature indices for CatBoost
# cat_feature_indices = [X.columns.get_loc(col) for col in cat_cols]

# print("CatBoost categorical feature indices:", cat_feature_indices)


print("3. CatBoostRegressor")
cat_val_preds, cat_test_preds = get_model_predictions(X, y, test, catboost_model)


print("4. ExtraTreesRegressor")
et_val_preds, et_test_preds = get_model_predictions(X, y, test, et_model)


val_preds_df = pd.DataFrame({
    'lgb': lgb_val_preds,
    'xgb': xgb_val_preds,
    'catb': cat_val_preds,
    'et': et_val_preds
})

test_preds_df = pd.DataFrame({
    'lgb': lgb_test_preds,
    'xgb': xgb_test_preds,
    'catb': cat_test_preds,
    'et': et_test_preds
})


from sklearn.linear_model import LinearRegression

# Stage 1: Base meta-model
base_meta_model = LinearRegression()
base_meta_model.fit(val_preds_df, y)

meta_train_preds = base_meta_model.predict(val_preds_df)
meta_test_preds = base_meta_model.predict(test_preds_df)

residuals = y - meta_train_preds

rmse = np.sqrt(mean_squared_error(y, meta_train_preds))
mae = mean_absolute_error(y, meta_train_preds)
r2 = r2_score(y, meta_train_preds)

print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

#9.17976


sample_submission['target'] = meta_test_preds
sample_submission.to_csv('submission.csv', index=False)


sample_submission['target'] = lgb_test_preds
sample_submission.to_csv('lgb_submission.csv', index=False)


sample_submission['target'] = xgb_test_preds
sample_submission.to_csv('xgb_submission.csv', index=False)


sample_submission['target'] = cat_test_preds
sample_submission.to_csv('cat_submission.csv', index=False)


sample_submission['target'] = et_test_preds
sample_submission.to_csv('et_submission.csv', index=False)


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("kaggle_api")

import json
import os

# Save the API key
os.makedirs("/root/.kaggle", exist_ok=True)
with open("/root/.kaggle/kaggle.json", "w") as f:
    json.dump({"username": "akinduhiman", "key": api_key}, f)

os.chmod("/root/.kaggle/kaggle.json", 600)


!kaggle competitions submit -c mlx-2-0-regression -f submission.csv -m "Message"


!kaggle competitions submit -c mlx-2-0-regression -f lgb_submission.csv -m "Message"


!kaggle competitions submit -c mlx-2-0-regression -f xgb_submission.csv -m "Message"


!kaggle competitions submit -c mlx-2-0-regression -f cat_submission.csv -m "Message"


!kaggle competitions submit -c mlx-2-0-regression -f et_submission.csv -m "Message"




