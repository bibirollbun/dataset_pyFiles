# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # visualization
import seaborn as sns

from sklearn.preprocessing import StandardScaler # scaling
from sklearn.preprocessing import LabelEncoder # encoder
from sklearn.model_selection import train_test_split # data split

from sklearn.linear_model import LinearRegression # modelling
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error # evaluation

import optuna
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.model_selection import KFold

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# read the dataset
df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df.head()


# more info
df.info()


# columns
columns = df.columns
print(f'features present in the dataset\n{columns}')
print(f'type check for columns: {type(columns)}')

# convert to list
columns = list(columns)
print(f'type check and columns display\n{type(columns)}\n{columns}')


# Gender Vs Heart_Rate
gender_hr = df.groupby(['Sex'])['Heart_Rate'].mean().reset_index()
gender_hr

# plot figure
plt.figure(figsize=(12, 8))
sns.barplot(x='Sex', y='Heart_Rate', data=gender_hr)
plt.xlabel('Gender')
plt.ylabel('Heart Rate')
plt.title('Sex Vs Hear_Rate')
plt.tight_layout
plt.show()


# Age Vs Heart_Rate
print(df['Age'].min(), df['Age'].max())
def categorize_age(age):
    if age <= 30:
        return 'Young (≤30)'
    elif 31 <= age <= 45:
        return 'Middle (31–45)'
    else:
        return 'Old (>45)'

df['Age_Group'] = df['Age'].apply(categorize_age)

# group by Age Group
age_hr = df.groupby('Age_Group')['Heart_Rate'].mean().reset_index()

# plot figure
plt.figure(figsize=(12, 8))
sns.barplot(x='Age_Group', y='Heart_Rate', data=age_hr, order=['Young (≤30)', 'Middle (31–45)', 'Old (>45)'])
plt.xlabel('Age Group')
plt.ylabel('Average Heart Rate')
plt.title('Average Heart Rate by Age Group')
plt.tight_layout()
plt.show()


# Heart_Rate distribution
heart_rate = df['Heart_Rate']

# density plot
sns.kdeplot(heart_rate, fill=True)
plt.title('Density Curve')
plt.xlabel('heart_rate distribution')
plt.ylabel('density')
plt.tight_layout()
plt.show()


# Calories distribution
calorie = df['Calories']

# density plot
sns.kdeplot(calorie, fill=True)
plt.title('Density Curve')
plt.xlabel('Calorie distribution')
plt.ylabel('density')
plt.tight_layout()
plt.show()


# Height Vs Heart_Rate
plt.figure(figsize=(24, 12))

# Height vs Heart Rate
plt.subplot(1, 2, 1)
sns.scatterplot(x='Height', y='Heart_Rate', data=df, alpha=0.5)
plt.xlabel('Height', fontsize=16)
plt.ylabel('Heart Rate', fontsize=16)
plt.title('Height vs Heart Rate', fontsize=16)

# Weight vs Heart Rate
plt.subplot(1, 2, 2)
sns.scatterplot(x='Weight', y='Heart_Rate', data=df, alpha=0.5)
plt.xlabel('Weight', fontsize=16)
plt.ylabel('Heart Rate', fontsize=16)
plt.title('Weight vs Heart Rate', fontsize=16)

plt.tight_layout()
plt.show()


plt.figure(figsize=(24, 10))

# Body Temp vs Heart Rate
plt.subplot(1, 2, 1)
sns.scatterplot(x='Body_Temp', y='Heart_Rate', data=df, alpha=0.5)
plt.xlabel('Body Temperature (°F)', fontsize=16)
plt.ylabel('Heart Rate (bpm)', fontsize=16)
plt.title('Body Temperature vs Heart Rate', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Calories vs Heart Rate
plt.subplot(1, 2, 2)
sns.scatterplot(x='Calories', y='Heart_Rate', data=df, alpha=0.5)
plt.xlabel('Calories Burned', fontsize=16)
plt.ylabel('Heart Rate (bpm)', fontsize=16)
plt.title('Calories Burned vs Heart Rate', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()


columns[:-1]


columns


df[columns[:-1]]


# label encoding
encoder = LabelEncoder()
encoder.fit(df['Sex'])

# transform data
df['Sex'] = encoder.transform(df['Sex'])
df.head()


# correlation features to calories
from yellowbrick.target import FeatureCorrelation

visualizer = FeatureCorrelation(labels=columns[:-1])
visualizer.fit(df[columns[:-1]], df['Calories'])
visualizer.show()


# scale the data
def preprocess(scale=True, df=df, columns=columns):
    scaler = StandardScaler()
    if scale:
        df_scaled = scaler.fit_transform(df[columns[:-1]])
        df_scaled = pd.DataFrame(df_scaled, columns=columns[:-1], index=df.index)
        df_scaled[columns[-1]] = df[columns[-1]]
        
    else:
        df_scaled = df

    # split the data
    X = df_scaled[columns[:-1]].drop(['id', 'Height', 'Weight'], axis=1)
    y = df_scaled[columns[-1]]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, y_train, X_val, y_val
    
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[columns[:-1]])

# preserve columns
df_scaled = pd.DataFrame(df_scaled, columns=columns[:-1], index=df.index)
df_scaled[columns[-1]] = df[columns[-1]]
df_scaled.head()


# x and y
X = df_scaled[columns[:-1]]
y = df_scaled[columns[-1]]

# split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42, 
                                                    test_size=0.2, shuffle=False)
X_train.shape, y_train.shape, X_val.shape, y_val.shape


regressor_0 = LinearRegression()
regressor_0.fit(X_train, y_train)


tr_preds = np.clip(regressor_0.predict(X_train), a_min=1, a_max=None)
mean_squared_log_error(y_train, tr_preds)


# from the correlation visualizer let's remove id during training
X_train_copy = X_train.drop(['id', 'Height', 'Weight'], axis=1)
X_val_copy = X_val.drop(['id', 'Height', 'Weight'], axis=1)

# fit the new model
regressor_1 = LinearRegression()
regressor_1.fit(X_train_copy, y_train)


# evaluation
tr_preds = np.clip(regressor_1.predict(X_train_copy), a_min=1, a_max=None)
mean_squared_log_error(y_train, tr_preds)


# get the data
X_tr, y_tr, X_v, y_v = preprocess(scale=False)

# clip negative values
def safe_msle(y_true, y_pred):
    y_pred = np.clip(y_pred, a_min=1, a_max=None)
    return mean_squared_log_error(y_true, y_pred)

msle_scorer = make_scorer(safe_msle, greater_is_better=False)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.1, 0.01, 0.2, 0.05),
        'max_depth': trial.suggest_int('max_depth', 3, 5, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 5, 10),
        'subsample': trial.suggest_float('sub_sample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'objective': 'reg:squaredlogerror',
        'verbosity': 0
    }

    model = XGBRegressor(**params)

    # kfold with msle
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    msle_scores = cross_val_sore(model, X_tr, y_tr, 
                                scoring=msle_scorer,
                                cv=kf)

    return msle_scores


# conduct study
study = optuna


# evaluation metrics
preds =  np.clip(regressor_2.predict(X_v), a_min=1, a_max=None)
mean_squared_log_error(y_v, preds)


from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_log_error, make_scorer

# metrics
msle_scorer = make_scorer(mean_squared_log_error, greater_is_better=False)

# finetune XGBoost model
xgb_model = XGBRegressor(objective='reg:squaredlogerror', verbosity=0)

param_dist = {
    'n_estimators': [100, 300, 500, 700],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [1, 5, 10, 20]
}

# instantiate the model
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=50,
    scoring=msle_scorer,
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_tr, y_tr)

best_model = random_search.best_estimator_


# best model metrics
y_pred = np.clip(best_model.predict(X_v), a_min=1, a_max=None)
msle = mean_squared_log_error(y_v, y_pred)
print("Best Parameters:", random_search.best_params_)
print("Validation MSLE:", msle)


# test data
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
test.head()


# encode Sex
encoder.fit(test['Sex'])

# transform data
test['Sex'] = encoder.transform(test['Sex'])
test.head()


# scale
# test_scaled = scaler.fit_transform(test)

# preserve columns
# test = pd.DataFrame(test_scaled, columns=columns[:-1], index=test.index)
ids = test['id']
test.drop(['id', 'Height', 'Weight'], axis=1, inplace=True)
test.head()


ids


# make predictions
submission = pd.DataFrame(columns=['id', columns[-1]])
submission['id'] = ids
submission[columns[-1]] = np.clip(best_model.predict(test), a_min=1, a_max=None)
submission


submission['Calories'].min()


# submit the file
submission.to_csv('submission_4.csv', index=False)




