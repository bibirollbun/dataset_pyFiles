import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")


sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")

print(sample_submission.shape)
sample_submission.head()


test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

print(test.shape)
test.head()


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")

print(train.shape)
train.head()


test.info()


train.info()


train_en = train.copy()
test_en = test.copy()

train_en.shape, test_en.shape


# new features
train_en['BMI'] = train_en['Weight'] / ((train_en['Height'] / 100) ** 2)
train_en['Workout_Intensity'] = train_en['Heart_Rate'] * train_en['Duration']
train_en['Workout_Intensity_Age'] = (train_en['Heart_Rate'] * train_en['Duration']) / train_en['Age']
train_en['Duration_Weight'] = train_en['Duration'] / train_en['Weight']
train_en['Heart_Rate_Weight'] = train_en['Heart_Rate'] / train_en['Weight']
train_en['Duration_Squared'] = train_en['Duration'] ** 2

test_en['BMI'] = test['Weight'] / ((test['Height'] / 100) ** 2)
test_en['Workout_Intensity'] = test_en['Heart_Rate'] * test_en['Duration']
test_en['Workout_Intensity_Age'] = (test_en['Heart_Rate'] * test_en['Duration']) / test_en['Age']
test_en['Duration_Weight'] = test_en['Duration'] / test_en['Weight']
test_en['Heart_Rate_Weight'] = test_en['Heart_Rate'] / test_en['Weight']
test_en['Duration_Squared'] = test_en['Duration'] ** 2

train_en.shape, test_en.shape


train_female = train_en[train_en['Sex'] == 'female'].drop(['Sex'], axis=1)
train_male = train_en[train_en['Sex'] == 'male'].drop(['Sex'], axis=1)

test_female = test_en[test_en['Sex'] == 'female'].drop(['Sex'], axis=1)
test_male = test_en[test_en['Sex'] == 'male'].drop(['Sex'], axis=1)

train_female.shape, train_male.shape, test_female.shape, test_male.shape


corrmat_female = (train_female.drop(['id'], axis=1)).corr()

plt.figure(figsize=(6, 6))
k=13
cols = corrmat_female.nlargest(k, 'Calories')['Calories'].index
cm = np.corrcoef(train_female[cols].values.T)
sns.set(font_scale=0.9)
hm = sns.heatmap(cm, cbar=False, annot=True, square=True, fmt='.2f', annot_kws={'size': 9}, 
                 yticklabels=cols.values, xticklabels=cols.values, cmap="Purples")
hm.xaxis.tick_top()
plt.xticks(rotation=45, ha='left')
plt.show()


corrmat_male = (train_male.drop(['id'], axis=1)).corr()

plt.figure(figsize=(6, 6))
k=13
cols = corrmat_male.nlargest(k, 'Calories')['Calories'].index
cm = np.corrcoef(train_male[cols].values.T)
sns.set(font_scale=0.9)
hm = sns.heatmap(cm, cbar=False, annot=True, square=True, fmt='.2f', annot_kws={'size': 9}, 
                 yticklabels=cols.values, xticklabels=cols.values, cmap="Purples")
hm.xaxis.tick_top()
plt.xticks(rotation=45, ha='left')
plt.show()


train_female.columns


columns = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 
           'Calories', 'BMI', 'Workout_Intensity', 'Workout_Intensity_Age', 
           'Duration_Weight', 'Heart_Rate_Weight', 'Duration_Squared']

for feature in columns:
    plt.figure(figsize=(12, 2))

    plt.subplot(1, 2, 1)
    plt.title(f'Histogram of {feature} in train_female')
    sns.histplot(train_female[feature], kde=True, bins=100)
    plt.xlabel(feature)
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.title(f'Histogram of {feature} in train_male')
    sns.histplot(train_male[feature], kde=True, bins=100)
    plt.xlabel(feature)
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer

def plot_transformation(data, gender, ax1, ax2):
    sns.histplot(data['Calories'], kde=True, bins=100, ax=ax1)
    ax1.set_title(f'[{gender}] Calories Distribution', fontsize=10)
    ax1.set_xlabel('Calories')
    ax1.set_ylabel('Frequency')
    
    original_skew = skew(data['Calories'])
    ax1.text(0.95, 0.95, f'Skewness: {original_skew:.2f}', 
             transform=ax1.transAxes, ha='right', va='top')

    pt = PowerTransformer(method='yeo-johnson')
    transformed = pt.fit_transform(data['Calories'].values.reshape(-1, 1))
    data['Calories_pt'] = transformed

    sns.histplot(data['Calories_pt'], kde=True, bins=100, ax=ax2)
    ax2.set_title(f'[{gender}] Trandformed Distribution', fontsize=10)
    ax2.set_xlabel('Calories_pt')
    ax2.set_ylabel(' ')

    transformed_skew = skew(data['Calories_pt'])
    ax2.text(0.95, 0.95, f'Skewness: {transformed_skew:.2f}', 
             transform=ax2.transAxes, ha='right', va='top')
    return pt

fig, axes = plt.subplots(2, 2, figsize=(14, 6))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

pt_female = plot_transformation(train_female, 'Female', axes[0][0], axes[0][1])
pt_male = plot_transformation(train_male, 'Male', axes[1][0], axes[1][1])

plt.show()


# target and features selection
train_female_features = train_female.drop(['id', 'Calories', 'Calories_pt'], axis=1)
train_male_features = train_male.drop(['id', 'Calories', 'Calories_pt'], axis=1)

train_female_target = train_female['Calories_pt']
train_male_target = train_male['Calories_pt']

test_female_features = test_female.drop(['id'], axis=1)
test_male_features = test_male.drop(['id'], axis=1)

print(train_female_features.shape, train_female_target.shape, train_male_features.shape, train_male_target.shape) 
test_female_features.shape, test_male_features.shape


# standardization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit_transform(train_female_features)

train_female_std = scaler.transform(train_female_features)
test_female_std = scaler.transform(test_female_features)

train_female_std = pd.DataFrame(data=train_female_std, columns=train_female_features.columns)
test_female_std = pd.DataFrame(data=test_female_std, columns=test_female_features.columns)

print(train_female_std.shape, test_female_std.shape)
train_female_std.head(2)


# standardization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit_transform(train_male_features)

train_male_std = scaler.transform(train_male_features)
test_male_std = scaler.transform(test_male_features)

train_male_std = pd.DataFrame(data=train_male_std , columns=train_male_features.columns)
test_male_std = pd.DataFrame(data=test_male_std, columns=test_male_features.columns)

print(train_male_std.shape, test_male_std.shape)
train_male_std.head(2)


# split into train and validation data
from sklearn.model_selection import (train_test_split, StratifiedKFold)
X_train_female, X_val_female, y_train_female, y_val_female = (
    train_test_split(train_female_std, train_female_target, test_size=0.2, random_state=42))

print(X_train_female.shape, X_val_female.shape, y_train_female.shape, y_val_female.shape)


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

pipeline = Pipeline(steps=[('impute', SimpleImputer(strategy='mean'))])
pipeline


def prepare_model(algorithm, X_train, y_train): 
    model = Pipeline(steps=[('preprocessing', pipeline),('algorithm', algorithm)])
    model.fit(X_train, y_train)
    return model


from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error

def rmsle(y_true, y_pred):
    """Root Mean Squared Logarithmic Error"""
    y_true_non_negative = np.maximum(y_true, 0)
    y_pred_non_negative = np.maximum(y_pred, 0)
    log_true = np.log1p(y_true_non_negative)
    log_pred = np.log1p(y_pred_non_negative)
    squared_error = np.square(log_true - log_pred)
    return np.sqrt(np.mean(squared_error))

algorithms = [RandomForestRegressor(n_jobs=-1, random_state=42), 
              #AdaBoostRegressor(random_state=42), 
              GradientBoostingRegressor(random_state=42),
              #BaggingRegressor(n_jobs=-1, random_state=42), 
              #SVR(), #take too much time
              #DecisionTreeRegressor(random_state=42), 
              #ExtraTreeRegressor(random_state=42),
              #LinearRegression(), 
              #SGDRegressor(max_iter=1000, tol=1e-3, random_state=42), 
              #KNeighborsRegressor(),  
              xgb.XGBRegressor(random_state=42), 
              lgb.LGBMRegressor(random_state=42, verbose=-1), 
              CatBoostRegressor(verbose=0, random_state=42)]


names = []
times = []
mse = []
rmse = []
rmsle_scores = [] 

for algorithm in algorithms:
    name = type(algorithm).__name__
    names.append(name)
    start_time = time.time()

    model = prepare_model(algorithm, X_train_female, y_train_female)
    pred_female = model.predict(X_val_female)
    end_time = time.time()
    times.append(end_time - start_time)
    mse.append(mean_squared_error(y_val_female, pred_female))
    rmse.append(np.sqrt(mean_squared_error(y_val_female, pred_female)))
    rmsle_scores.append(rmsle(y_val_female, pred_female)) 

print('Regression Results in Algorithms')
results_dict = {'Algorithm': names, 'MSE': mse, 'RMSE': rmse, 'RMSLE': rmsle_scores, 'Time': times} 
print(pd.DataFrame(results_dict).sort_values(by='RMSLE', ascending=1))

"""
Regression Results in Algorithms
                    Algorithm       MSE      RMSE     RMSLE       Time
11          CatBoostRegressor  0.002132  0.046173  0.019654  23.998036
10              LGBMRegressor  0.002181  0.046700  0.019859   2.277291
9                XGBRegressor  0.002224  0.047162  0.020258   1.446287
2   GradientBoostingRegressor  0.002388  0.048863  0.020386  72.037363
0       RandomForestRegressor  0.002353  0.048509  0.020685  91.529032
6            LinearRegression  0.002893  0.053790  0.021457   0.268810
7                SGDRegressor  0.002938  0.054204  0.021551   0.506614
3            BaggingRegressor  0.002539  0.050389  0.021552  13.289751
8         KNeighborsRegressor  0.004203  0.064830  0.025108  11.934512
5          ExtraTreeRegressor  0.004269  0.065339  0.027459   1.319945
4       DecisionTreeRegressor  0.004295  0.065533  0.027461   3.570830
1           AdaBoostRegressor  0.026668  0.163305  0.070540  34.761258
"""


# Ensemble Model (VotingRegressor)
from sklearn.ensemble import VotingRegressor

cat_model = CatBoostRegressor(verbose=0, random_state=42)
lgbm_model = lgb.LGBMRegressor(random_state=42, verbose=-1)
xgb_model = xgb.XGBRegressor(random_state=42)
gb_model = GradientBoostingRegressor(random_state=42)

Voting_models = [
    ('CatBoost', cat_model), 
    ('LGBM', lgbm_model), 
    ('XGBoost', xgb_model),
    ('GradientBoost', gb_model),
]

female_Voting_model = VotingRegressor(estimators=Voting_models)

start_time = time.time()
female_Voting_model.fit(X_train_female, y_train_female)

train_female_Voting_pred = female_Voting_model.predict(X_train_female)
val_female_Voting_pred = female_Voting_model.predict(X_val_female)

end_time = time.time()
Voting_time = end_time - start_time

train_female_Voting_rmsle = rmsle(y_train_female, train_female_Voting_pred)
val_female_Voting_rmsle = rmsle(y_val_female, val_female_Voting_pred)

print('Ensemble Model (FemaleVotingRegressor)')
print(f"train_RMSLE: {train_female_Voting_rmsle:.6f}")
print(f"val_RMSLE: {val_female_Voting_rmsle:.6f}")
print(f"Time: {Voting_time:.6f}")
"""
Ensemble Model (FemaleVotingRegressor)
train_RMSLE: 0.017957
val_RMSLE: 0.019537
Time: 102.323123
"""


# Ensemble Model (StackingRegressor)
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression

base_models = [
    ('CatBoost', cat_model), 
    ('LGBM', lgbm_model), 
    ('XGBoost', xgb_model), 
    ('GradientBoost', gb_model)
]

meta_model = LinearRegression()
female_stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=5)

start_time = time.time()
female_stacking_model.fit(X_train_female, y_train_female)

train_female_stacking_pred = female_stacking_model.predict(X_train_female)
val_female_stacking_pred = female_stacking_model.predict(X_val_female)

end_time = time.time()
stacking_time = end_time - start_time

train_female_stacking_rmsle = rmsle(y_train_female, train_female_stacking_pred)
val_female_stacking_rmsle = rmsle(y_val_female, val_female_stacking_pred)

print('Female Ensemble Model (StackingRegressor)')
print(f"train_RMSLE: {train_female_stacking_rmsle:.6f}")
print(f"val_RMSLE: {val_female_stacking_rmsle:.6f}")
print(f"Time: {stacking_time:.6f}")
"""
Female Ensemble Model (StackingRegressor)
train_RMSLE: 0.017736
val_RMSLE: 0.019521
Time: 486.536897
"""


# stacking 
y_val_female_df = pd.DataFrame(y_val_female)
female_stacking_pred_df = pd.DataFrame(
    val_female_stacking_pred, columns= ['female_val_stacking_pred'], index=y_val_female.index )
y_val_female_pred_df = pd.concat([y_val_female_df, female_stacking_pred_df], axis=1)

print(y_val_female_pred_df.shape)
y_val_female_pred_df.head()


# convert to original values
female_calories_pt_values = y_val_female_pred_df['Calories_pt'].values.reshape(-1, 1)
female_calories = pt_female.inverse_transform(female_calories_pt_values)

female_stacking_pred_values = y_val_female_pred_df['female_val_stacking_pred'].values.reshape(-1, 1)
pred_female_calories = pt_female.inverse_transform(female_stacking_pred_values)

y_val_female_pred_df['female_Calories'] = female_calories.flatten()
y_val_female_pred_df['pred_female_Calories'] = pred_female_calories.flatten()

print(y_val_female_pred_df.shape)
y_val_female_pred_df.head()


# evaluation for stacking
female_stacking_rmsle = rmsle(y_val_female_pred_df['female_Calories'], 
                              y_val_female_pred_df['pred_female_Calories'])
female_stacking_rmsle
# 0.04870878088500117


# split into train and validation data
from sklearn.model_selection import (train_test_split, StratifiedKFold)
X_train_male, X_val_male, y_train_male, y_val_male = (
    train_test_split(train_male_std, train_male_target, test_size=0.2, random_state=42))

print(X_train_male.shape, X_val_male.shape, y_train_male.shape, y_val_male.shape)

val_male_idx = X_val_male.index 


names = []
times = []
mse = []
rmse = []
rmsle_scores = [] 

for algorithm in algorithms:
    name = type(algorithm).__name__
    names.append(name)
    start_time = time.time()

    model = prepare_model(algorithm, X_train_male, y_train_male)
    pred_male = model.predict(X_val_male)
    end_time = time.time()
    times.append(end_time - start_time)
    mse.append(mean_squared_error(y_val_male, pred_male))
    rmse.append(np.sqrt(mean_squared_error(y_val_male, pred_male)))
    rmsle_scores.append(rmsle(y_val_male, pred_male)) 

print('Regression Results in Algorithms')
results_dict = {'Algorithm': names, 'MSE': mse, 'RMSE': rmse, 'RMSLE': rmsle_scores, 'Time': times} 
print(pd.DataFrame(results_dict).sort_values(by='RMSLE', ascending=1))

"""
Regression Results in Algorithms
                    Algorithm       MSE      RMSE     RMSLE       Time
11          CatBoostRegressor  0.003308  0.057516  0.022818  23.696088
10              LGBMRegressor  0.003550  0.059582  0.023743   2.168240
9                XGBRegressor  0.003551  0.059586  0.023799   1.459019
0       RandomForestRegressor  0.003718  0.060976  0.024134  88.114107
2   GradientBoostingRegressor  0.003835  0.061924  0.024370  70.737612
3            BaggingRegressor  0.004011  0.063330  0.025131  13.515333
6            LinearRegression  0.004901  0.070008  0.025571   0.280838
7                SGDRegressor  0.004944  0.070317  0.025631   0.521615
8         KNeighborsRegressor  0.005806  0.076198  0.028645  11.977791
5          ExtraTreeRegressor  0.006760  0.082221  0.032591   1.358001
4       DecisionTreeRegressor  0.006903  0.083087  0.032980   3.537139
1           AdaBoostRegressor  0.068686  0.262079  0.106723  34.455339
"""


# Ensemble Model (VotingRegressor)
from sklearn.ensemble import VotingRegressor

cat_model = CatBoostRegressor(verbose=0, random_state=42)
lgbm_model = lgb.LGBMRegressor(random_state=42, verbose=-1)
xgb_model = xgb.XGBRegressor(random_state=42)
rf_model = RandomForestRegressor(n_jobs=-1, random_state=42)

Voting_models = [
    ('CatBoost', cat_model), 
    ('LGBM', lgbm_model), 
    ('XGBoost', xgb_model), 
    ('RandomForest', rf_model)
]

male_Voting_model = VotingRegressor(estimators=Voting_models)

start_time = time.time()
male_Voting_model.fit(X_train_male, y_train_male)

train_male_Voting_pred = male_Voting_model.predict(X_train_male)
val_male_Voting_pred = male_Voting_model.predict(X_val_male)

end_time = time.time()
Voting_time = end_time - start_time

train_male_Voting_rmsle = rmsle(y_train_male, train_male_Voting_pred)
val_male_Voting_rmsle = rmsle(y_val_male, val_male_Voting_pred)

print('male Ensemble Model (VotingRegressor)')
print(f"train_RMSLE: {train_male_Voting_rmsle:.6f}")
print(f"val_RMSLE: {val_male_Voting_rmsle:.6f}")
print(f"Time: {Voting_time:.6f}")

"""
male Ensemble Model (VotingRegressor)
train_RMSLE: 0.018044
val_RMSLE: 0.022791
Time: 123.472583
"""


# Ensemble Model (StackingRegressor)
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression

base_models = [
    ('CatBoost', cat_model), 
    ('LGBM', lgbm_model), 
    ('XGBoost', xgb_model), 
    ('RandomForest', rf_model)
]

meta_model = LinearRegression()
male_stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=5)

start_time = time.time()
male_stacking_model.fit(X_train_male, y_train_male)

train_male_stacking_pred = male_stacking_model.predict(X_train_male)
val_male_stacking_pred = male_stacking_model.predict(X_val_male)

end_time = time.time()
stacking_time = end_time - start_time

train_male_stacking_rmsle = rmsle(y_train_male, train_male_stacking_pred)
val_male_stacking_rmsle = rmsle(y_val_male, val_male_stacking_pred)

print('male Ensemble Model (StackingRegressor)')
print(f"train_RMSLE: {train_male_stacking_rmsle:.6f}")
print(f"val_RMSLE: {val_male_stacking_rmsle:.6f}")
print(f"Time: {stacking_time:.6f}")

"""
male Ensemble Model (StackingRegressor)
train_RMSLE: 0.019160
val_RMSLE: 0.022675
Time: 588.768442
"""


# stacking 
y_val_male_df = pd.DataFrame(y_val_male)
male_stacking_pred_df = pd.DataFrame(
    val_male_stacking_pred, columns= ['male_val_stacking_pred'], index=y_val_male.index )
y_val_male_pred_df = pd.concat([y_val_male_df, male_stacking_pred_df], axis=1)

print(y_val_male_pred_df.shape)
y_val_male_pred_df.head()


# convert to original values
male_calories_pt_values = y_val_male_pred_df['Calories_pt'].values.reshape(-1, 1)
male_calories = pt_male.inverse_transform(male_calories_pt_values)

male_stacking_pred_values = y_val_male_pred_df['male_val_stacking_pred'].values.reshape(-1, 1)
pred_male_calories = pt_male.inverse_transform(male_stacking_pred_values)

y_val_male_pred_df['male_Calories'] = male_calories.flatten()
y_val_male_pred_df['pred_male_Calories'] = pred_male_calories.flatten()

print(y_val_male_pred_df.shape)
y_val_male_pred_df.head()


# evaluation for stacking
male_stacking_rmsle = rmsle(y_val_male_pred_df['male_Calories'], 
                       y_val_male_pred_df['pred_male_Calories'])
male_stacking_rmsle

# 0.06833287194675082


# val_female data
female_df = y_val_female_pred_df[['female_Calories', 'pred_female_Calories']].copy()
female_df.columns = ['val_Calories', 'val_pred_Calories']

# val_male data
male_df = y_val_male_pred_df[['male_Calories', 'pred_male_Calories']].copy()
male_df.columns = ['val_Calories', 'val_pred_Calories']

val_pred_df = pd.concat([female_df, male_df])
val_pred_df = val_pred_df.sort_index()

print(val_pred_df.shape)
val_pred_df.head()


# evaluation for stacking
val_stacking_rmsle = rmsle(val_pred_df['val_Calories'], val_pred_df['val_pred_Calories'])
val_stacking_rmsle

# 0.059319058976997484


# prediction by stacking model

female_stacking_pred = female_stacking_model.predict(test_female_std)
male_stacking_pred = male_stacking_model.predict(test_male_std)

female_stacking_pred, male_stacking_pred


# Covert to Calories
female_pred_reshaped = female_stacking_pred.reshape(-1, 1)
female_pred_inverse = pt_female.inverse_transform(female_pred_reshaped)
female_pred_Calories = female_pred_inverse.flatten()

female_pred_Calories


# Covert to Calories
male_pred_reshaped = male_stacking_pred.reshape(-1, 1)
male_pred_inverse = pt_male.inverse_transform(male_pred_reshaped)
male_pred_Calories = male_pred_inverse.flatten()

male_pred_Calories


test_female_idx = test_female.index
test_male_idx = test_male.index

female_pred_series = pd.Series(female_pred_Calories, index=test_female_idx)
male_pred_series = pd.Series(male_pred_Calories, index=test_male_idx)

test_pred_series = pd.concat([female_pred_series, male_pred_series])
test_pred_series = test_pred_series.loc[test.index]

submission = pd.DataFrame({'id': test.id, 'Calories': test_pred_series})

print(submission.shape)
submission.head()


submission.to_csv('submission.csv', index=False)


submission = pd.read_csv('/kaggle/working/submission.csv')
submission.head()




