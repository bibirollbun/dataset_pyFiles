import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor





train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

train_ids = train_df['id']
test_ids = test_df['id']

train_df.drop('id', axis=1, inplace=True)
test_df.drop('id', axis=1, inplace=True)


train_df.info()


train_df.describe().T


train_df.isna().sum()


plt.figure(figsize=(12,4))
plt.title("Missing values - Training set")
sns.heatmap(train_df.isnull(), cbar=False)
plt.show()


num_cols = train_df.select_dtypes(include=['number']).columns.tolist()

for col in num_cols:
    fig, axes = plt.subplots(1,2, figsize=(12,4))
    
    sns.histplot(data=train_df, x=col, ax=axes[0], kde=True)
    sns.boxplot(data=train_df, x=col, ax=axes[1])
    
    axes[0].set_title(f"Histplot of {col}")
    axes[1].set_title(f"Boxplot of {col}")
    
    plt.tight_layout()
    plt.show()


to_drop=train_df.query('Episode_Length_minutes > 121') #325 min length while 2nd longest is 120min ...
train_df.drop(to_drop.index, axis=0, inplace=True)
to_drop


to_drop=train_df.query('Number_of_Ads > 12') #We have some instances with extremely high !FLOAT! values
train_df.drop(to_drop.index, axis=0, inplace=True)
to_drop


cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    print(train_df[col].value_counts())
    print('*'*30)



for col in cat_cols[2:]:
    plt.figure(figsize=(12,4))
    plt.title(f"Countplot of {col}")
    sns.countplot(data=train_df, x=col, palette='coolwarm')
    plt.show()


def extract_episode(value):
    return int(value.split()[1])

train_df['Episode_Title'] = train_df['Episode_Title'].apply(extract_episode)
train_df['Episode_Title']


corr_matrix = train_df.corr(numeric_only=True)

plt.title("Correlations")
sns.heatmap(corr_matrix, annot=True)
plt.show()


weekday_dict={
    'Monday':1,
    'Tuesday':2,
    'Wednesday':3,
    'Thursday':4,
    'Friday':5,
    'Saturday':6,
    'Sunday':7
}

train_df['Day_Numeric'] = train_df['Publication_Day'].map(weekday_dict)
train_df['Day_Cos'] = np.cos(2 * np.pi * train_df['Day_Numeric'] / 7)
train_df['Day_Sin'] = np.sin(2 * np.pi * train_df['Day_Numeric'] / 7)


target = train_df['Listening_Time_minutes']
X= train_df.drop('Listening_Time_minutes', axis=1)


imputer = SimpleImputer(strategy='median')

columns_to_impute = ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads']


X[columns_to_impute] = imputer.fit_transform(X[columns_to_impute])
for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].astype('category')

X_train, X_test, y_train, y_test = train_test_split(X, target)


num_cols = X.select_dtypes(include = ['number']).columns.tolist()
cat_cols = X.select_dtypes(include = ['category']).columns.tolist()
cat_cols.remove('Episode_Sentiment')



log_transformer = FunctionTransformer(np.log1p)

num_pipeline = make_pipeline(log_transformer, StandardScaler())
cat_pipeline = OneHotEncoder()

preprocessing = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols),
    ('ord', OrdinalEncoder(categories=[['Negative', 'Neutral', 'Positive']]), ['Episode_Sentiment'])
], remainder='passthrough')


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


decision_tree_model = DecisionTreeRegressor()
random_forest_model = RandomForestRegressor()
gradient_boosting_model = GradientBoostingRegressor()
xgb_model = XGBRegressor()
lgbm_model = LGBMRegressor()

models = {
    #"DecisionTreeRegressor": decision_tree_model,
    #"RandomForestRegressor": random_forest_model,
    #"GradientBoostingRegressor": gradient_boosting_model,
    "XGBRegressor": xgb_model,
    "LGBMRegressor": lgbm_model
}

for name, model in models.items():
    print(name)
    preds = cross_val_predict(make_pipeline(preprocessing, model), X_train, y_train, cv=3, verbose=2)
    print(f"RMSE: {rmse(y_train, preds):.3f}")
    print('*'*30) 



xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

# ------------------
# Full Pipeline
# ------------------
pipeline = Pipeline(steps=[
    ('preprocess', preprocessing),
    ('regressor', xgb)
])

# ------------------
# Grid Search Params
# ------------------
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [3, 5, 7],
    'regressor__learning_rate': [0.01, 0.1, 0.3],
    'regressor__subsample': [0.8, 1.0],
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2
)

# ------------------
# Fit the model
# ------------------
grid_search.fit(X, target)
print("Best parameters:", grid_search.best_params_)
print("Best score:", -grid_search.best_score_)


final_model=grid_search.best_estimator_



final_model.fit(X,target)


test_df['Day_Numeric'] = test_df['Publication_Day'].map(weekday_dict)
test_df['Day_Cos'] = np.cos(2 * np.pi * test_df['Day_Numeric'] / 7)
test_df['Day_Sin'] = np.sin(2 * np.pi * test_df['Day_Numeric'] / 7)

test_df['Episode_Title'] = test_df['Episode_Title'].apply(extract_episode)

# Assuming X is your DataFrame
imputer = SimpleImputer(strategy='median')

# Specify the columns to impute
columns_to_impute = ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads']

# Fit the imputer and transform only the selected columns
test_df[columns_to_impute] = imputer.fit_transform(test_df[columns_to_impute])


preds = final_model.predict(test_df)


final_df = pd.DataFrame({'id': test_ids, 'Listening_Time_minutes': preds})

final_df.to_csv('final_predictions.csv', index=False)










