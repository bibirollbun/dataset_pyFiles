import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")

print("***************** Train ********************")
print(f"Number of rows : {train.shape[0]}")
print(f"Number of columns : {train.shape[1]}")

print("***************** Test *********************")
print(f"Number of rows : {test.shape[0]}")
print(f"Number of columns : {test.shape[1]}")
print("*********************************************")


print("******************************* Train ************************************")
print(train.info())
print("******************************* Test ************************************")
print(test.info())


# Null Values
train_null_count = train.isna().sum()
train_null_percent = train_null_count / train.shape[0] * 100
train_dtypes = train.dtypes

test_null_count = test.isna().sum()
test_null_percent = test_null_count / test.shape[0] * 100
test_dtypes = test.dtypes

summary_df = pd.DataFrame({
    "Train Null Count": train_null_count,
    "Train Null Percent": train_null_percent,
    "Train Data Type": train_dtypes,
    "Test Null Count": test_null_count,
    "Test Null Percent": test_null_percent,
    "Test Data Type": test_dtypes
})
summary_df


# Check for duplicate rows
train.duplicated().sum()


# Number of unique values in columns
train.nunique()


test.describe().style.background_gradient("summer")


num_cols = ["Episode_Length_minutes", "Host_Popularity_percentage", "Guest_Popularity_percentage", "Number_of_Ads", "Listening_Time_minutes"]
train[num_cols].corr().style.background_gradient("summer")


train.head()
num_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']
cat_cols = [i for i in train.columns[1:-1] if i not in num_cols ]
target_col = ['Listening_Time_minutes']


for col in num_cols:
    fig, axes = plt.subplots(1, 2, figsize = (12, 5))
    sns.histplot(train[col], kde = True, ax = axes[0])
    sns.boxplot(train[col], ax = axes[1])
    plt.show()


# cat_cols
fig, axes = plt.subplots(3, 2, figsize=(12, 14))
axes = axes.ravel()

for i, col in enumerate(cat_cols):
    sns.countplot(data=train, x=col, ax=axes[i])
    axes[i].tick_params(axis='x', rotation=90)
    axes[i].set_title(f'Count Plot of {col}')

plt.tight_layout()
plt.show()


# 3. Feature Engineering
def safe_convert_time(x):
    try:
        return pd.to_datetime(x, format='%H:%M').hour
    except:
        if isinstance(x, str):
            x = x.lower()
            if 'night' in x:
                return 22
            elif 'morning' in x:
                return 9
            elif 'afternoon' in x:
                return 15
            elif 'evening' in x:
                return 18
        return np.nan

def weekend_flag(day):
    if day in ['Saturday', 'Sunday']:
        return 1
    else:
        return 0

for df in [train, test]:
    # Time related
    df['Publication_Hour'] = df['Publication_Time'].apply(safe_convert_time)
    df['Host_Guest_Interaction'] = df['Host_Popularity_percentage'] * df['Guest_Popularity_percentage']
    
    # New feature: Weekend flag
    df['Is_Weekend'] = df['Publication_Day'].apply(weekend_flag)
    
    # New feature: Popularity per Ad
    df['Popularity_per_Ad'] = (df['Host_Popularity_percentage'] + df['Guest_Popularity_percentage']) / (df['Number_of_Ads'] + 1)

# 4. Data Preprocessing
numerical_features = [
    "Episode_Length_minutes",
    "Host_Popularity_percentage",
    "Guest_Popularity_percentage",
    "Number_of_Ads",
    "Publication_Hour",
    "Host_Guest_Interaction",
    "Is_Weekend",
    "Popularity_per_Ad"
]

categorical_features = [
    "Podcast_Name",
    "Episode_Title",
    "Genre",
    "Publication_Day",
]

# Impute missing values
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")
train[numerical_features] = num_imputer.fit_transform(train[numerical_features])
test[numerical_features] = num_imputer.transform(test[numerical_features])
train[categorical_features] = cat_imputer.fit_transform(train[categorical_features])
test[categorical_features] = cat_imputer.transform(test[categorical_features])

# 5. Set up Features and Target
y = np.log1p(train['Listening_Time_minutes'])  # log-transform here
X = train.drop(['Listening_Time_minutes', 'id', 'Publication_Time'], axis=1)


# Select categorical and numerical columns
categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]
numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

my_cols = categorical_cols + numerical_cols
X = X[my_cols].copy()



# Preprocessing Pipelines
numerical_transformer = SimpleImputer(strategy='constant')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])



from catboost import CatBoostRegressor

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('catboost', CatBoostRegressor(
    iterations=3000,
    depth=8,
    learning_rate=0.03,
    loss_function='RMSE',
    random_seed=42,
    verbose=0,
    od_type='Iter',      # ⏱️ Enables early stopping
    od_wait=20
)
)
])


import optuna
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

def objective(trial):
    params = {
        'iterations': 300,  # Reduced from 3000 for speed
        'depth': trial.suggest_int('depth', 6, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'random_seed': 42,
        'loss_function': 'RMSE',
        'verbose': 0,
        'od_type': 'Iter',
        'od_wait': 20
    }

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('catboost', CatBoostRegressor(**params))
    ])

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=kf)
    
    # Print CV RMSE per trial (optional)
    trial.set_user_attr("cv_rmse", -scores.mean())
    print(f"Trial RMSE: {-scores.mean():.4f}")

    return -scores.mean()

# Run Optuna optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Best params and CV score
best_params = study.best_params
average_cv_rmse = study.best_trial.user_attrs["cv_rmse"]
print("Best parameters:", best_params)
print(f"Average CV RMSE: {average_cv_rmse:.4f}")

# Fit final model
final_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('catboost', CatBoostRegressor(
        iterations=3000,
        random_seed=42,
        loss_function='RMSE',
        verbose=0,
        **best_params
    ))
])

final_model.fit(X, y)

# Evaluate on training set
y_pred = final_model.predict(X)
train_rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f'Overall Train RMSE: {train_rmse:.4f}')


# Predict on Test Set
X_test = test.drop(['id', 'Publication_Time'], axis=1)
X_test = X_test[my_cols].copy()
test_predictions_log = final_model.predict(X_test)
test_predictions = np.expm1(test_predictions_log)  # reverse log1p


# Re-load the 'id' column from original test file
test_ids = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')['id']

# Prepare submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    'Listening_Time_minutes': test_predictions
})

# Save to CSV for submission
submission.to_csv('submission.csv', index=False)

