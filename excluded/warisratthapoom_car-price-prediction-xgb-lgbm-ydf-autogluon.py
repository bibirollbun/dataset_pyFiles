!pip install autogluon


from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, VotingRegressor
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
import warnings
from xgboost import XGBRegressor
 
# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=Warning, module="LightGBM")


train = pd.read_csv('/kaggle/input/car-price-prediction-x/train.csv')
test = pd.read_csv('/kaggle/input/car-price-prediction-x/test.csv')
train['IsTrain'] = 1
test['IsTrain'] = 0
all_data = pd.concat([train, test.drop('Id', axis=1)], axis=0)
all_data.head()


all_data.describe()


all_data.info()


all_data['model'].unique()


sns.barplot(train,x='model', y='price', hue='status', palette='pastel')


sns.boxplot(train,x='model', y='price', palette='pastel')


sns.lineplot(train, x='year', y='price', hue='model', palette='pastel')


sns.barplot(train, x='type', y='price', hue='model', palette='pastel')


plt.figure(figsize=(12,8))
sns.barplot(train, x='color', y='price', palette='pastel')


sns.lineplot(train, x='motor_volume', y='price')


all_data['wheel'].value_counts()


columns = ['price', 'year']
outlier_percentage = {}

for column in columns:
    column_data = train[column]

    Q1 = np.percentile(column_data, 25)
    Q3 = np.percentile(column_data, 75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = column_data[(column_data < lower) | (column_data > upper)]
    outlier_percentage[column] = len(outliers) / len(column_data) * 100

    print(f"Count of outliers in column '{column}': {len(outliers)}")
    print(f"Percentage of outliers in column '{column}':{len(outliers) / len(column_data) * 100:.2f}%")
    print()
    print(f"Lower : {lower}")
    print(f"Upper : {upper}")
    print()
    print(f"Data Outlier': {np.array(outliers)}")


def remove_outliers(df):
    column_list = ['price', 'year']

    for col in column_list:
        column_data = df[col]

        # Detect Outliers
        Q1 = np.percentile(column_data, 25)
        Q3 = np.percentile(column_data, 75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # Remove Outlier
        df = df[(column_data >= lower) & (column_data <= upper)]

    return df


all_data['running'].apply(lambda x: float(x.split('  ')[0]) * 1.6 if x.split('  ')[1] == 'miles' else float(x.split('  ')[0]))


def convert_to_km(df):
    df['running_km'] = df['running'].apply(lambda x: float(x.split('  ')[0]) * 1.6 if x.split('  ')[1] == 'miles' else float(x.split('  ')[0]))
    return df.drop('running', axis=1)


all_data['year'].apply(lambda x: 2025 - x)


def convert_to_age(df):
    df['age'] = df['year'].apply(lambda x: 2025 - x)
    return df.drop('year', axis=1)


preprocess = all_data.copy()
preprocess.columns


columns_to_drop = ['wheel', 'price', 'IsTrain']
numerical_columns = ['age', 'running_km', 'motor_volume']
categorical_columns = ['model', 'color', 'type', 'motor_type']
ordinal_columns = ['status']


numerical_transformer = Pipeline(
    steps = [('StandardScaler', StandardScaler())]
)

categorical_transformer = Pipeline(
    steps = [('OneHotEncoder', OneHotEncoder(handle_unknown='ignore', sparse_output = False))]
)

ordinal_transformer = Pipeline(
    steps = [('OrdinalEncoder', OrdinalEncoder(categories=[['crashed', 'normal', 'good', 'excellent', 'new']]))]
)


preprocessor = ColumnTransformer(
    transformers=[
        ('column_dropper', 'drop', columns_to_drop),
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns),
        ('ordinal', ordinal_transformer, ordinal_columns)
    ]
)

pipeline = Pipeline(
    steps=[
        ('convert_to_km', FunctionTransformer(convert_to_km)),
        ('convert_to_age', FunctionTransformer(convert_to_age)),
        ('preprocessing', preprocessor),
    ]
)


# Remove Outliers
preprocess_train = remove_outliers(preprocess[preprocess['IsTrain'] == 1])
preprocess_test = preprocess[preprocess['IsTrain'] == 0]

X = pipeline.fit_transform(preprocess_train)
y = np.log(preprocess_train['price']) # Normalize the price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_submission = pipeline.transform(preprocess_test)


models = {
    'SVR': SVR(),
    'RandomForest': RandomForestRegressor(random_state=42),
    'XGBRegressor': XGBRegressor(random_state=42),
    'LGBMRegressor': LGBMRegressor(force_row_wise=True, objective='regression', max_bin=255, random_state=42),
    'CatBoostRegressor': CatBoostRegressor(random_state=42)
}

param_grids = {
    'SVR': {
        'kernel': ['linear', 'rbf'],
        'C': [1e-2, 1e-1, 1],
        'gamma': [1e-3, 1e-2, 1e-1]
    },
    'RandomForest': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'XGBRegressor': {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 10],
        'learning_rate': [0.001, 0.01, 0.1]
    },
    'LGBMRegressor': {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 10],
        'learning_rate': [0.001, 0.01, 0.1]
    },
    'CatBoostRegressor': {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 10],
        'learning_rate': [0.001, 0.01, 0.1]
    }
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

grids = {}
for model_name, model in models.items():
    grids[model_name] = GridSearchCV(estimator=model, param_grid=param_grids[model_name], cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    grids[model_name].fit(X_train, y_train)
    best_params = grids[model_name].best_params_
    best_score = grids[model_name].best_score_
    
    print(f'Best parameters for {model_name}: {best_params}')
    print(f'Best Score for {model_name}: {best_score}\n')


def evaluate(model, model_name, X_test, y_test):
    print(f"Model: {model_name}")
    
    y_pred = model.predict(X_test)
    y_pred = np.e**(y_pred)
    y_test = np.e**(y_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"R-squared (R²): {r2:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
    
for model_name, model in grids.items():
    evaluate(model, model_name, X_test, y_test)


# All models
estimators = [
    ('SVR', grids['SVR'].best_estimator_),
    ('RandomForest', grids['RandomForest'].best_estimator_),
    ('XGBRegressor', grids['XGBRegressor'].best_estimator_),
    ('LGBMRegressor', grids['LGBMRegressor'].best_estimator_),
    ('CatBoostRegressor', grids['CatBoostRegressor'].best_estimator_)
]

all_voting_regressor = VotingRegressor(estimators=estimators, n_jobs=-1)
all_voting_regressor.fit(X_train, y_train)

all_stacking_regressor = StackingRegressor(estimators=estimators, n_jobs=-1)
all_stacking_regressor.fit(X_train, y_train)


evaluate(all_voting_regressor, 'All Models Voting Regressor', X_test, y_test)
evaluate(all_stacking_regressor, 'All Models Stacking Regressor', X_test, y_test)


# Top 3 best models
estimators = [
    ('RandomForest', grids['RandomForest'].best_estimator_),
    ('LGBMRegressor', grids['LGBMRegressor'].best_estimator_),
    ('CatBoostRegressor', grids['CatBoostRegressor'].best_estimator_)
]

top3_voting_regressor = VotingRegressor(estimators=estimators, n_jobs=-1)
top3_voting_regressor.fit(X_train, y_train)

top3_stacking_regressor = StackingRegressor(estimators=estimators, n_jobs=-1)
top3_stacking_regressor.fit(X_train, y_train)



evaluate(top3_voting_regressor, 'Voting Regressor', X_test, y_test)
evaluate(top3_stacking_regressor, 'Stacking Regressor', X_test, y_test)


import ydf
print("Available Custom Models:\n" + '\n'.join([f"   •  {attr}" for attr in dir(ydf) if attr.endswith("Learner")]))


processed_train = pd.concat([pd.DataFrame(X_train, columns=pipeline.named_steps['preprocessing'].get_feature_names_out()), pd.DataFrame(y_train, columns=['price']).reset_index(drop=True)], axis=1)
processed_test = pd.concat([pd.DataFrame(X_test, columns=pipeline.named_steps['preprocessing'].get_feature_names_out()), pd.DataFrame(y_test, columns=['price']).reset_index(drop=True)], axis=1)


tuner = ydf.RandomSearchTuner(num_trials=50)
tuner.choice("num_trees", [50, 100, 200, 300, 500, 1000])
tuner.choice("max_depth", [5, 10, 15, 20, 25, 30])
tuner.choice("growing_strategy", ['LOCAL', 'BEST_FIRST_GLOBAL'])
ydf_model = ydf.RandomForestLearner(label="price", 
                                    task=ydf.Task.REGRESSION, 
                                    tuner=tuner                                    
                                    ).train(processed_train)
ydf_model.describe()
evaluate(ydf_model, 'ydf', processed_test.drop('price', axis=1), processed_test['price'])


from autogluon.tabular import TabularPredictor

autogluon_model = TabularPredictor(label='price', eval_metric='rmse')
autogluon_model.fit(processed_train,
              time_limit=600,
              presets='best_quality',
              num_bag_folds=5,
              num_bag_sets=5)


autogluon_model.leaderboard(processed_test, silent=True)


y_pred = autogluon_model.predict(processed_test, model='ExtraTreesMSE_BAG_L2')
y_pred = np.e**(y_pred)
y_test = np.e**(processed_test['price'])
    
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"R-squared (R²): {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")


sample_submission = pd.read_csv('/kaggle/input/car-price-prediction-x/sample_submission.csv')
sample_submission.head()


X_submission = pd.DataFrame(X_submission, columns=pipeline.named_steps['preprocessing'].get_feature_names_out())
X_submission['price'] = np.nan
y_submission = autogluon_model.predict(X_submission, model='ExtraTreesMSE_BAG_L2')
y_submission = np.e**y_submission
submission_df = pd.concat([sample_submission['Id'], y_submission], axis=1)

submission_df.to_csv('submission.csv', index=False)

