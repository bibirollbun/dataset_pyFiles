# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, make_scorer
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import optuna
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test =  pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sub =  pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


train.head()


train.drop('id', axis= 1,inplace=True)


train.isnull().sum()


train.info()


train['num_lanes'].describe()


num_columns = [i for i in train.columns if train[i].dtype in ['int64','float64'] and i != 'accident_risk' ]


cat_columns = train.select_dtypes(include=['object']).columns.tolist()


bool_columns = train.select_dtypes(include=['bool']).columns.tolist()


print(' numerical columns   :   {} \n categoricak columns :   {}, \n bool columns        :   {}'.format(num_columns,cat_columns,bool_columns))


for i in cat_columns:
    print(train[i].unique())


# train[num_columns].describe()


titles = {
    'num_lanes': 'Average Accident Risk by Number of Lanes',
    'curvature': 'Average Accident Risk by Road Curvature',
    'speed_limit': 'Average Accident Risk by Speed Limit',
    'num_reported_accidents': 'Average Accident Risk by Reported Accidents'
}
xlabels = {
    'num_lanes': 'Number of Lanes',
    'curvature': 'Road Curvature',
    'speed_limit': 'Speed Limit (km/h)',
    'num_reported_accidents': 'Number of Reported Accidents'
}
for i in num_columns:
    plt.figure(figsize=(8,5))
    sns.barplot(x=i, y='accident_risk', data=train, estimator='mean', palette='viridis')
    
    plt.title(titles[i], fontsize=14, fontweight='bold')
    plt.xlabel(xlabels[i], fontsize=12)
    plt.ylabel('Average Accident Risk', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


corr_cols = num_columns + ['accident_risk']
correlation_matrix = train[corr_cols].corr()
plt.figure(figsize=(12, 9))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('numeric values corr map', fontsize=16)
plt.show()


n = len(cat_columns)
cols = 3 
rows = (n + cols - 1) // cols  
plt.figure(figsize=(cols * 5, rows * 5))
for idx, col in enumerate(cat_columns, 1):
    plt.subplot(rows, cols, idx)
    data = train[col].value_counts()
    plt.pie(
        data,
        labels=data.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=plt.cm.viridis(np.linspace(0,1,len(data))),
        wedgeprops={'width':0.4}
    )
    plt.title(f'{col.replace("_", " ").capitalize()} Distribution', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()



n = len(bool_columns)
cols = 3   
rows = (n + cols - 1) // cols
plt.figure(figsize=(cols * 5, rows * 4))
for idx, col in enumerate(bool_columns, 1):
    plt.subplot(rows, cols, idx)
    counts = train[col].value_counts().sort_index()
    sns.barplot(
        x=counts.index.astype(str),
        y=counts.values,
        palette='viridis'
    )
    
    plt.title(f'{col.replace("_", " ").capitalize()} Count', fontsize=12, fontweight='bold')
    plt.xlabel(col.replace('_', ' ').capitalize(), fontsize=11)
    plt.ylabel('Count', fontsize=11)
    plt.tight_layout()

plt.show()



def create_advanced_features(df):
    df_new = df.copy()
    df_new['weather_lighting_interaction'] = df_new['weather'].astype(str) + '_' + df_new['lighting'].astype(str)
    df_new['road_time_interaction'] = df_new['road_type'].astype(str) + '_' + df_new['time_of_day'].astype(str)
    df_new['school_day_traffic'] = (df_new['school_season'].astype(int) * (1 - df_new['holiday'].astype(int))).astype(int)
    df_new['danger_factor_curve_speed'] = df_new['curvature'] * df_new['speed_limit']
    df_new['accidents_per_lane'] = df_new['num_reported_accidents'] / (df_new['num_lanes'] + 1e-6)
    df_new['speed_limit_sq'] = df_new['speed_limit']**2
    df_new['poor_visibility'] = ((df_new['weather'] == 'foggy') | (df_new['lighting'].isin(['night', 'dim']))).astype(int)
    df_new['is_rush_hour'] = ((df_new['time_of_day'].isin(['morning', 'evening'])) & (df_new['holiday'] == 0)).astype(int)
    return df_new

train = create_advanced_features(train)
test = create_advanced_features(test)

road_type_stats = train.groupby('road_type')['speed_limit'].agg(['mean', 'std']).reset_index()
road_type_stats.columns = ['road_type', 'road_type_avg_speed', 'road_type_std_speed']

train = pd.merge(train, road_type_stats, on='road_type', how='left')
test = pd.merge(test, road_type_stats, on='road_type', how='left')

train['speed_anomaly_score'] = (train['speed_limit'] - train['road_type_avg_speed']) / (train['road_type_std_speed'] + 1e-6)
test['speed_anomaly_score'] = (test['speed_limit'] - test['road_type_avg_speed']) / (test['road_type_std_speed'] + 1e-6)



new_cat_columns = ['weather_lighting_interaction', 'road_time_interaction']
new_num_columns = [
    'school_day_traffic', 'danger_factor_curve_speed', 'accidents_per_lane', 
    'speed_limit_sq', 'poor_visibility', 'is_rush_hour', 
    'road_type_avg_speed', 'road_type_std_speed', 'speed_anomaly_score'
]

all_num_columns = num_columns + bool_columns + new_num_columns
all_cat_columns = cat_columns + new_cat_columns


bool_mapping = {False:0, True:1}
for i in bool_columns:
    train[i] = train[i].map(bool_mapping)
    test[i] = test[i].map(bool_mapping)


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), all_cat_columns),
        ('num', StandardScaler(), all_num_columns),
    ],
    remainder='passthrough'
)


X = train.drop('accident_risk', axis=1)
y = train['accident_risk']


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

rmse_scorer = make_scorer(rmse, greater_is_better=False)  
cv = KFold(n_splits=5, shuffle=True, random_state=41)


# def objective(trial):
#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 800, 3000),
#         "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.03, log=True),
#         "max_depth": trial.suggest_int("max_depth", 4, 10),
#         "subsample": trial.suggest_float("subsample", 0.5, 0.9),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
#         "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
#         "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 2.0, log=True),
#         "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
#         "gamma": trial.suggest_float("gamma", 0.0, 2.0),
#         "random_state": 42,
#         "n_jobs": -1,
#         "verbosity": 0,
#         "tree_method": "hist",
#     }

#     model = XGBRegressor(**params)


    
#     pipe = Pipeline([
#         ('preprocessor', preprocessor),
#         ('model', model)
#     ])

    
#     scores = cross_val_score(pipe, X, y, scoring=rmse_scorer, cv=cv, n_jobs=-1)
#     mean_neg_rmse = scores.mean()
#     return -mean_neg_rmse


# study = optuna.create_study(direction='minimize', study_name='xgb_ps')
# study.optimize(objective, n_trials=40, show_progress_bar=True)



# print("Best RMSE:", study.best_value)
# print("Best Params:", study.best_trial.params)


# best_params = study.best_trial.params.copy()
# best_params.update({
#     "random_state": 42,
#     "n_jobs": -1,
#     "verbosity": 0,
#     "tree_method": "hist",
# })


# best_params


best_params = {
    'n_estimators': 8000,
    'learning_rate': 0.013995941838349492,
    'max_depth': 9,
    'min_child_weight': 3,
    'subsample': 0.7537849361802125,
    'colsample_bytree': 0.7534776889002607,
    'reg_alpha': 0.7548072875343894,
    'reg_lambda': 0.9909635181970773,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0,
    'tree_method': 'hist',
}


final_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(**best_params))
])


oof_preds = np.zeros(len(X))
cv_scores = []
best_iterations = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    prep = clone(preprocessor).fit(X_train)
    X_train_t = prep.transform(X_train)
    X_val_t   = prep.transform(X_val)

    model = XGBRegressor(
        **best_params,
        early_stopping_rounds=100,   
        eval_metric='rmse',
    )

    model.fit(
        X_train_t, y_train,
        eval_set=[(X_val_t, y_val)],
        verbose=False
    )
    
    best_iterations.append(model.best_iteration)
    val_preds = model.predict(X_val_t)
    oof_preds[val_idx] = val_preds

    fold_rmse = mean_squared_error(y_val, val_preds, squared=False)
    cv_scores.append(fold_rmse)
    print(f"Fold {fold} RMSE: {fold_rmse:.6f}")

print(f"\nMean CV RMSE: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")


prep = clone(preprocessor).fit(X)
X_all_t   = prep.transform(X)
X_test_t  = prep.transform(test)

avg_best_iteration = int(np.mean(best_iterations))
print(f"\nMEAN best iteration    : {avg_best_iteration}..")

best_params['n_estimators'] = avg_best_iteration

final_model = XGBRegressor(**best_params)


best_params


final_model.fit(X_all_t, y)


test_preds = final_model.predict(X_test_t)
test_preds = np.clip(test_preds, 0, 1)  

sub['accident_risk'] = test_preds  
sub.to_csv('/kaggle/working/submission.csv', index=False)



ls


sub.head()

