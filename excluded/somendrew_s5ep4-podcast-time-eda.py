import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


train.shape,test.shape


train.head()


train.describe()


train.info()


train.isnull().sum()


test.isnull().sum()


cols = train.columns
cols


cat_cols = [col for col in cols if train[col].dtype == "object"]
cat_cols


for col in cat_cols:
    print(train[col].value_counts())
    print("_______________________________________________")


for col in cat_cols:
    plt.figure(figsize=(20, 4))  # Optional: Adjust figure size
    sns.countplot(data=train, x=col)
    plt.title(f"Countplot of {col}")
    plt.xticks(rotation=45)  # Rotate x-axis labels if needed
    plt.show()  # Display the plot properly


plt.figure(figsize=(20, 5))
sns.countplot(data = train, x = "Genre", hue = "Episode_Sentiment")


plt.figure(figsize=(20, 5))
sns.countplot(data = train, x = "Publication_Time", hue = "Episode_Sentiment")


num_cols = [col for col in train.columns if col not in cat_cols and col != "id"]
num_cols


train.isnull().sum()


num_features = len(num_cols)
fig, axes = plt.subplots(num_features, 2, figsize=(12, 5 * num_features))

# Loop through numerical columns and create plots
for i, col in enumerate(num_cols):
    # Histogram
    sns.histplot(train[col], bins=30, kde=True, ax=axes[i, 0])
    axes[i, 0].set_title(f'Distribution of {col}')
    
    # Boxplot
    sns.boxplot(x=train[col], ax=axes[i, 1])
    axes[i, 1].set_title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()


outliers = np.sum(train['Number_of_Ads'] > 3)
outliers


outliers2 = np.sum(train['Episode_Length_minutes'] > 120)
outliers2


train.isnull().sum()


train['Episode_Length_minutes'].fillna(train['Episode_Length_minutes'].mode()[0], inplace=True)
train['Guest_Popularity_percentage'].fillna(train['Guest_Popularity_percentage'].mode()[0], inplace=True)
train['Number_of_Ads'].fillna(train['Number_of_Ads'].mode()[0], inplace=True)



test['Episode_Length_minutes'].fillna(test['Episode_Length_minutes'].mode()[0], inplace=True)
test['Guest_Popularity_percentage'].fillna(test['Guest_Popularity_percentage'].mode()[0], inplace=True)
test['Number_of_Ads'].fillna(test['Number_of_Ads'].mode()[0], inplace=True)


train.isnull().sum()


test.isnull().sum()


def remove_outliers(df):
    df = df[df['Number_of_Ads'] <= 3]
    df = df[df['Episode_Length_minutes'] <= 120]
    return df

train = remove_outliers(train)

updated_outliers = np.sum(train['Number_of_Ads'] > 3)
print(updated_outliers)

updated_outliers2 = np.sum(train['Episode_Length_minutes'] > 120)
print(updated_outliers2)



train.columns


train.head()


train['Genre'].value_counts()


for col in cat_cols:
    print(train[col].value_counts())
    print("---------------------------------")


def target_encode_smooth(df, col_name, target_col, k=3):
    """
    Perform smoothed target encoding.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    col_name (str): The name of the column to encode (categorical feature).
    target_col (str): The name of the target column (numerical target).
    k (float): Smoothing factor. Higher values mean more weight to the global mean.

    Returns:
    pd.Series: A pandas Series containing the smoothed target encoding.
    """
    global_mean = df[target_col].mean()

    category_counts = df.groupby(col_name)[target_col].count()
    category_means = df.groupby(col_name)[target_col].mean()

    smooth_means = (category_counts * category_means + k * global_mean) / (category_counts + k)

    return df[col_name].map(smooth_means)



train['Episode_Title_smooth_encoded'] = target_encode_smooth(
    df=train,
    col_name='Episode_Title',
    target_col='Listening_Time_minutes',
    k=3
)
train['Podcast_Name_smooth_encoded'] = target_encode_smooth(
    df=train,
    col_name='Podcast_Name',
    target_col='Listening_Time_minutes',
    k=3
)




test['Episode_Title_smooth_encoded'] = target_encode_smooth(
    df=train,
    col_name='Episode_Title',
    target_col='Listening_Time_minutes',
    k=3
)
test['Podcast_Name_smooth_encoded'] = target_encode_smooth(
    df=train,
    col_name='Podcast_Name',
    target_col='Listening_Time_minutes',
    k=3
)



train.drop(['Podcast_Name','Episode_Title'],axis = 1, inplace = True)


test.drop(['Podcast_Name','Episode_Title'],axis = 1, inplace = True)


train.head()


def one_hot_encode(df, columns, drop_first=False):
    return pd.concat(
        [df.drop(columns=columns, axis=1)] + 
        [pd.get_dummies(df[col], prefix=col, drop_first=drop_first).astype(int) for col in columns],
        axis=1
    )

# Example usage
train = one_hot_encode(train, ['Publication_Day', 'Publication_Time', 'Episode_Sentiment', 'Genre'], drop_first=True)
test = one_hot_encode(test, ['Publication_Day', 'Publication_Time', 'Episode_Sentiment', 'Genre'], drop_first=True)



train.head()


y = train["Listening_Time_minutes"]
X = train.drop('Listening_Time_minutes',axis = 1 )



from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# model = XGBRegressor()
# model.fit(X_train,y_train)

# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# rmse
# 13.1016


# def xgb_objective(trial):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
#         'subsample': trial.suggest_loguniform('subsample', 0.5, 1),
#         'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.5, 1),
#         'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
#         'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
#         'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
#         'random_state': 42,
#         'n_jobs': -1
#     }
    
#     # Define StratifiedKFold cross-validation
#     cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
#     rmse = []  # To store accuracy for each fold
    
#     #cross validation
#     X_np = X.values
#     y_np = y.values

#     for train_idx, val_idx in cv.split(X_np):
#         X_train_fold, X_val_fold = X_np[train_idx], X_np[val_idx]
#         y_train_fold, y_val_fold = y_np[train_idx], y_np[val_idx]
        
        
#         model = XGBRegressor(**params)
#         model.fit(X_train_fold,y_train_fold)
        
#         # Predict and calculate accuracy for this fold
#         preds = model.predict(X_val_fold)
#         mse = mean_squared_error(y_val_fold, preds)
#         rmse_score = np.sqrt(mse)
#         rmse.append(rmse_score)
    
#     return sum(rmse) / len(rmse)


import optuna
from sklearn.model_selection import KFold


# # Let's get 10 best parameters (1 best parameter per 1 study)
# best_params = []

# study = optuna.create_study(direction='minimize')
# study.optimize(xgb_objective, n_trials=50)
    
# best_params.append(study.best_params)



# best_params


# best_params = [{'n_estimators': 852,
#   'max_depth': 10,
#   'learning_rate': 0.022256865497444607,
#   'subsample': 0.6249069818225592,
#   'colsample_bytree': 0.8528813925458255,
#   'gamma': 0.4918535659456626,
#   'min_child_weight': 2,
#   'reg_alpha': 0.008002879726661406,
#   'reg_lambda': 3.0682223745812507e-06}]

best_params = [{
    'n_estimators': 331,
    'max_depth': 9,
    'learning_rate': 0.08710206030353791,
    'subsample': 0.8730775308377182,
    'colsample_bytree': 0.5122938519439793,
    'gamma': 0.000544224161652546,
    'min_child_weight': 5,
    'reg_alpha': 3.0084997494843593e-05,
    'reg_lambda': 0.010485820431314883
}]



model = XGBRegressor(**best_params[0])
model.fit(X_train,y_train)
# Predict and calculate accuracy for this fold
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
rmse_score = np.sqrt(mse)
rmse_score


test_preds = model.predict(test)


sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')



sample_submission.head()


sample_submission['Listening_Time_minutes'] = test_preds
sample_submission.head(10)


sample_submission.to_csv('submission.csv', index=False)
print("File Saved!!")

