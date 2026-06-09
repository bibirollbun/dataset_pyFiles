import pandas as pd
import numpy as np
import seaborn as sns
import shap
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool


pod_data_path = "/kaggle/input/playground-series-s5e4/train.csv"
pod_data = pd.read_csv(pod_data_path)

print("Podcast data before imputation:")
pod_data.head()



from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')

imputation_features = ["Episode_Length_minutes", "Host_Popularity_percentage", "Number_of_Ads", 
                       "Guest_Popularity_percentage"]
imp.fit(pod_data[imputation_features])
pod_data[imputation_features] = np.round(imp.transform(pod_data[imputation_features]), 2)

print("Podcast data after imputation:")
pod_data.head()


print(pod_data["Genre"].unique())


def feature_engineering(df, test = False):
    # Cutting off certain feature values to prevent outliers/nonsensical values
    df.loc[df['Number_of_Ads'] > 3, 'Number_of_Ads'] = 5
    df.loc[df['Episode_Length_minutes'] > 120.99, 'Episode_Length_minutes'] = 120.99
    df.loc[df['Guest_Popularity_percentage'] > 100, 'Guest_Popularity_percentage'] = 100
    df.loc[df['Host_Popularity_percentage'] > 100, 'Host_Popularity_percentage'] = 100

    if not test:
        df['Listening_Time_minutes'] = np.minimum(df['Listening_Time_minutes'], df['Episode_Length_minutes'])

    # Finding relationships between publication time and genre
    df["Time_of_week"] = df["Publication_Day"] + "_" + df["Publication_Time"]
    df["Spooky"] = ((df["Publication_Time"] == "Evening") | (df["Publication_Time"] == "Night")) & (df["Genre"] == "True Crime")
    df["Day_starter"] = (df["Genre"] == "News") & (df["Publication_Time"] == "Morning")
    df["Weekend"] = (df["Publication_Day"] == "Saturday") | (df["Publication_Day"] == "Sunday")

    # Episode length/number, and number of ads
    df["Hour_or_longer"] = df["Episode_Length_minutes"] >= 60
    df["Length_bin"] = np.floor(df["Episode_Length_minutes"]/10)
    df["Episode_bin"] = np.floor(df["Episode_Title"].str[-2:].astype(int)/10)
    df["Ads_per_hour"] = np.where(df["Number_of_Ads"] == 0, 0, np.round(df["Number_of_Ads"] / (df["Episode_Length_minutes"]/60),2))

    # Various transformations of host/guest popularity
    df["Guest_more_popular"] = df["Guest_Popularity_percentage"] > df["Host_Popularity_percentage"]
    df["Total_popularity"] = df["Guest_Popularity_percentage"] + df["Host_Popularity_percentage"]
    df["Guest_Popularity_percentage_bin"] = np.round(df["Guest_Popularity_percentage"]/5,0)
    df["Host_Popularity_percentage_bin"] = np.round(df["Host_Popularity_percentage"]/5,0)
    df["Guest_extreme_pop"] = 4*np.square((df["Guest_Popularity_percentage"]/100)-0.5)
    df["Host_extreme_pop"] = 4*np.square((df["Host_Popularity_percentage"]/100)-0.5)
    


feature_engineering(pod_data)

pod_data.head()


features = pod_data.columns.tolist()
remove_features = ["id", "Podcast_Name", "Episode_Title", "Listening_Time_minutes"]
features = [f for f in features if f not in remove_features]

cat_attribs = ["Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment", 
               "Time_of_week"]

X = pod_data[features].copy()
y = pod_data["Listening_Time_minutes"].copy()

X[cat_attribs] = X[cat_attribs].astype('category')


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Identify columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessor
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(drop='first'), categorical_cols)
])


rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

rf_model.fit(X, y)


# Assuming X and y are already defined
kf = KFold(n_splits=10, shuffle=True, random_state=1)
train_scores = []
val_scores = []

for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
    xgb_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=10,
    min_child_weight= 4,
    colsample_bytree=0.66,
    subsample=0.9,
    gamma=1.6,
    reg_alpha=5.5,
    reg_lambda=8,
    eval_metric="rmse",
    early_stopping_rounds=100,
    random_state=1212,
    tree_method="hist",
    enable_categorical=True,
    verbosity=0
    )
    # Splitting data into training and validation sets
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Fitting the model
    xgb_model.fit(X_train, y_train, 
              eval_set=[(X_val, y_val)], 
              verbose=False)

    # Predicting on the training and validation sets
    y_train_pred = xgb_model.predict(X_train)
    y_val_pred = xgb_model.predict(X_val)

    # RMSE scores
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

    train_scores.append(train_rmse)
    val_scores.append(val_rmse)

    print(f"Fold {fold}: Train RMSE = {train_rmse:.4f}, Validation RMSE = {val_rmse:.4f}")

# Calculating mean RMSE scores
mean_train_rmse = np.mean(train_scores)
mean_val_rmse = np.mean(val_scores)

print(f"\nMean Train RMSE: {mean_train_rmse:.4f}")
print(f"Mean Validation RMSE: {mean_val_rmse:.4f}")

xgb_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=10,
    min_child_weight= 4,
    colsample_bytree=0.66,
    subsample=0.9,
    gamma=1.6,
    reg_alpha=5.5,
    reg_lambda=8,
    eval_metric="rmse",
    random_state=1212,
    tree_method="hist",
    enable_categorical=True,
    verbosity=0
)

xgb_model.fit(X,y)


kf = KFold(10, shuffle=True, random_state=1)
kf_splits = kf.split(X)
scores1 = []
test_preds1 = []

for i, (train_idx, val_idx) in enumerate(kf_splits):
    cat_model = CatBoostRegressor(
        iterations = 5000,
        learning_rate = 0.08,
        depth = 15,
        l2_leaf_reg = 4,
        loss_function = "RMSE",
        eval_metric = "RMSE",
        random_seed = 42,
        verbose = 1000,
        task_type = "GPU"
        )
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    train_pool = Pool(X_train_fold, y_train_fold, cat_features=cat_attribs)
    val_pool = Pool(X_val_fold, y_val_fold, cat_features=cat_attribs)
    
    cat_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100)

    val_pred = cat_model.predict(X_val_fold)
    score = mean_squared_error(y_val_fold, val_pred, squared=False)
    scores1.append(score)

    print(f'CatBoost Fold {i + 1} rmse: {score}')
print(f'CatBoost rmse: {np.mean(scores1):.5f};')

cat_model = CatBoostRegressor(
        iterations = 5000,
        learning_rate = 0.08,
        depth = 15,
        l2_leaf_reg = 4,
        loss_function = "RMSE",
        eval_metric = "RMSE",
        random_seed = 42,
        verbose = 1000,
        task_type = "GPU"
        )
cat_model.fit(X, y, cat_features = tuple(categorical_cols))


test_data_path = "/kaggle/input/playground-series-s5e4/test.csv"
test_data = pd.read_csv(test_data_path)

test_data[imputation_features] = np.round(imp.transform(test_data[imputation_features]), 2)

feature_engineering(test_data, test = True)

test_X = test_data[features].copy()

test_X[cat_attribs] = test_X[cat_attribs].astype('category')

rf_preds = rf_model.predict(test_X)
xgb_preds = xgb_model.predict(test_X)
cat_preds = cat_model.predict(test_X)

y_preds = 0.10 * rf_preds + 0.50 * xgb_preds + 0.40 * cat_preds 


submission = pd.DataFrame({'id': test_data['id'], 'Listening_Time_minutes': y_preds})
submission.to_csv('submission.csv', index=False)
print(submission.head())

