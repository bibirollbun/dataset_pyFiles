import numpy as np 
import pandas as pd
pd.set_option('display.max_columns', 100)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import time

import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train = pd.read_csv("/kaggle/input/steel-quality-challenge/train.csv", sep=",")
df_test = pd.read_csv("/kaggle/input/steel-quality-challenge/test.csv", sep=",")


df_train.head()


# data is mostly numerical
df_train.dtypes


# no missing values
df_train.isna().sum()


fig, ax = plt.subplots(2,1 , figsize=(12,4))
sns.countplot(data=df_train, x="machine_id", ax=ax[0])
ax[0].axhline(np.mean(df_train['machine_id'].value_counts()), color="black", linewidth=1)
sns.countplot(data=df_train, x="operator_id", ax=ax[1], color="#30a2da")
ax[1].axhline(np.mean(df_train['operator_id'].value_counts()), color='black', linewidth=1)
plt.tight_layout()
plt.show()


num_cols = [x for x in df_train.select_dtypes("number") if x != "id"]
for f in num_cols:
    fig, ax = plt.subplots(1, 2, figsize=(16,2.5), width_ratios=[3, 1])
    sns.boxplot(data=df_train, x="machine_id", y=f, ax=ax[0])
    sns.histplot(data=df_train, x=f, ax=ax[1])
    plt.tight_layout()
plt.show()


sec_reference = 24*60*60
year_reference = 365.2425

def cyclic_encoding_time(time: str):
    """convert an hour in a day as "HH:MM:SS" into its cosine and sine features
    """
    val = time.split(":")
    time_day_sec = int(val[0])*3600+ int(val[1])*60 + int(val[2])
    return np.sin(time_day_sec * (2 * np.pi / sec_reference)), np.cos(time_day_sec * (2 * np.pi / sec_reference))


def cylic_encoding_day(day: pd._libs.tslibs.timestamps.Timestamp):
    """convert a date as "YYYY:MM"DD" into a day in a year then into its cosine and sine features
    """
    day_in_year = day.timetuple().tm_yday
    return np.sin(day_in_year * (2 * np.pi / year_reference)), np.cos(day_in_year * (2 * np.pi / year_reference))

# convert to Datetime first
df_train["production_date_dt"] = pd.to_datetime(df_train["production_date"])
df_test["production_date_dt"] = pd.to_datetime(df_test["production_date"])

# cycliv encoding
df_train[['np_sin_day', 'np_cos_day']] = df_train.apply(lambda x: cylic_encoding_day(x["production_date_dt"]), axis=1, result_type='expand')
df_train[['np_sin_time', 'np_cos_time']] = df_train.apply(lambda x: cyclic_encoding_time(x["production_time"]), axis=1, result_type='expand')
df_test[['np_sin_day', 'np_cos_day']] = df_test.apply(lambda x: cylic_encoding_day(x["production_date_dt"]), axis=1, result_type='expand')
df_test[['np_sin_time', 'np_cos_time']] = df_test.apply(lambda x: cyclic_encoding_time(x["production_time"]), axis=1, result_type='expand')



from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid, cross_validate, cross_val_score, ShuffleSplit
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, VotingRegressor
import xgboost as xgb
from sklearn.feature_selection import SelectKBest, f_classif, f_regression


num_cols = ['plate_thickness', 'plate_length', 'min_luminosity', 'defect_area', 'brightness_index', 'edge_index', 'square_index', 'total_luminosity', 'cooling_rate','processing_time','temperature']
# since the cyclic variables are already within [-1, 1], there's no need to include them in the pipeline
num_cols_untouched = ['np_sin_day','np_cos_day','np_sin_time','np_cos_time']
cat_cols = ['machine_id']
target = "quality_score"

df_test = df_test[num_cols + num_cols_untouched + cat_cols]

X = df_train[num_cols + num_cols_untouched + cat_cols]
y = df_train[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    shuffle=True,
    random_state=1234)

print(f"train: {X_train.shape}, test: {X_test.shape}")


# Create a pipeline
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), 
           ("scaler", StandardScaler())]
)

categorical_onehot_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_onehot_transformer, cat_cols),
    ],
    remainder='passthrough'
)

preprocessor


regressors = {'LinearRegression': LinearRegression(),
              'Ridge': Ridge(),
              'RandomForest': RandomForestRegressor(),
              'GradientBoostingRegressor': GradientBoostingRegressor(),
              'KnnRegressor': KNeighborsRegressor(),
              'BaggingRegressor': BaggingRegressor(),
              'SGDRegressor': SGDRegressor(),
              'XGBRegressor': xgb.XGBRegressor()
       }

random_state = 42
cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=1234)

metrics = ['neg_root_mean_squared_error', 'r2']
regressor_fitted = []
cross_validate_res = [] # list to hold CV results

for cnt, (clf_name, clf) in enumerate(regressors.items()):
    
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor), 
            (clf_name, clf)]
    )
    
    print(f"processing {clf_name}")
    res = cross_validate(pipe, X_train, y_train, cv=cv, return_train_score=True, scoring=metrics)
    pipe.fit(X_train, y_train)
    regressor_fitted.append({"name": clf_name, "test_predictions": pipe.predict(X_test) })
    # regressor_fitted[clf_name] = pipe.predict(X_test)
    res_df = pd.DataFrame(res).mean()
    res_df = pd.DataFrame(res_df).apply(pd.to_numeric).transpose()
    res_df['Regressor'] = clf_name
    cross_validate_res.append(res_df)


fig, ax = plt.subplots(3, len(regressors), figsize=(22,7))
cnt=0
for result in regressor_fitted:
    ax[0, cnt].scatter(y_test, result["test_predictions"], s=2) 
    ax[0, cnt].set_xlim([0,1])
    ax[0, cnt].set_ylim([0,1])
    ax[0, cnt].axline([0,0],[1, 1], color='red', linewidth=1, linestyle="--", label="best line fit")
    ax[1, cnt].scatter(y_test, y_test - result["test_predictions"], s=2)
    ax[1, cnt].set_ylim([-1, 1])
    ax[2, cnt].hist(y_test - result["test_predictions"])
    ax[2, cnt].set_xlim([-1, 1])
    
    ax[0, cnt].set_title(result["name"], fontsize=12)
    ax[0, cnt].set_xlabel('ground truth')
    ax[0, cnt].set_ylabel('prediction')
    ax[1, cnt].set_xlabel('ground truth')
    ax[1, cnt].set_ylabel('Residuals')
    ax[2, cnt].set_xlabel('Residuals')
    
    ax[0,cnt].set_title(result["name"], fontsize=12)

    for i in range(0,3):
        ax[i, cnt].xaxis.set_tick_params(labelsize=10)
        ax[i, cnt].yaxis.set_tick_params(labelsize=10)
        ax[i, cnt].xaxis.label.set_size(10)
        ax[i, cnt].yaxis.label.set_size(10)
    
    cnt += 1
plt.tight_layout()
plt.show()


pd.concat(cross_validate_res, ignore_index=True).style.background_gradient(cmap="viridis")


# Create a pipeline
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="median")), 
        ("scaler", StandardScaler())]
)

categorical_onehot_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_onehot_transformer, cat_cols),
    ],
    remainder='passthrough'
)

pipeline = Pipeline(steps = [
    ("preprocessor", preprocessor), 
    ('regressor', SGDRegressor(random_state=42))
 ])

param_grid = {
    'regressor__loss': ['squared_error', 'huber'],
    'regressor__penalty': ['l1', 'l2', 'elasticnet'],
    'regressor__alpha': [1e-3, 1e-2, 1e-1, 1, 10, 100],
    # 'regressor__alpha': np.logspace(-4, 4, 20),
    'regressor__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
    # 'regressor__eta0': np.logspace(-3, 0, 4),
    # 'regressor__max_iter': [1000],
    # 'regressor__tol': [1e-3]
}


grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_root_mean_squared_error', verbose=1)
start = time.time()
grid_search.fit(X_train, y_train)
end = time.time()


# Print the best hyperparameters and score
print(f"Best hyperparameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")
print(f"timing: {end-start}")

# Evaluate the best model on the train set
best_model = grid_search.best_estimator_
train_score = best_model.score(X_train, y_train)
print("Train score:", train_score)

# Evaluate the best model on the test set
test_score = best_model.score(X_test, y_test)
print("Test score:", test_score)


train_preds = best_model.predict(X_train)
test_preds = best_model.predict(X_test)

fig, ax = plt.subplots(2, 3, figsize=(8,4))
ax[0,0].scatter(y_train, train_preds, s=0.5) 
ax[0,0].set_xlabel('ground truth (train)')
ax[0,0].set_ylabel('prediction (train)')
ax[0,0].axline([0,0],[1, 1], color='red', linewidth=1, linestyle="--", label="best line fit")
ax[0,1].hist(y_train - train_preds) 
ax[0,1].set_xlabel('residuals (train)')
ax[0,2].scatter(y_train, y_train - train_preds, s=0.5) 
ax[0,2].set_xlabel('ground truth (train)')
ax[0,2].set_ylabel('residuals (train)')

ax[1,0].scatter(y_test, test_preds, s=0.5) 
ax[1,0].set_xlabel('ground truth (test)')
ax[1,0].set_ylabel('prediction (test)')
ax[1,0].axline([0,0],[1,1], color='red', linewidth=1, linestyle="--", label="best line fit")
ax[1,1].hist(y_test - test_preds) 
ax[1,1].set_xlabel('residuals (test)')
ax[1,2].scatter(y_test, y_test - test_preds, s=0.5) 
ax[1,2].set_xlabel('ground truth (test)')
ax[1,2].set_ylabel('residuals (test)')

plt.tight_layout()
plt.show()


final_preds = best_model.predict(df_test)
sample = pd.read_csv("/kaggle/input/steel-quality-challenge/sample_submission.csv", sep=",")
sample["quality_score"] = final_preds
# sample.to_csv("sample_solution_steel-quality-gradientboosting-tuned.csv", index=False)


sample.head(10)




