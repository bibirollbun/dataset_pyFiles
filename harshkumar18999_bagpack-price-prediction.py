!pip install bayesian-optimization
!pip install lightgbm
!pip install catboost
!pip install category_encoders


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
import warnings
from bayes_opt import BayesianOptimization
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
import logging
import category_encoders as ce
warnings.filterwarnings('ignore')


df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
df_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')

df = pd.concat([df, df_extra])


test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


df.head()


df.shape


df.info()


df.isnull().sum()


df.dtypes


plt.subplots(5, 2, figsize=(15, 25))

for i, col in enumerate(df.columns[1:]):
  plt.subplot(5, 2, i+1)
  plt.title(col)

  if df[col].dtype == 'float64':
    sns.histplot(df[col], kde=True)

  else:
    sns.countplot(x=df[col])

plt.show()


df.groupby(['Brand'])['Price'].describe()


df.groupby(['Material'])['Price'].describe()


df.groupby(['Style'])['Price'].describe()


df.groupby(['Color'])['Price'].describe()


df.groupby(['Compartments'])['Price'].describe()


df.groupby(['Size'])['Price'].describe()


df.head()


df['Laptop Compartment'] = df['Laptop Compartment'].map({'Yes': 1, 'No': 0})
test_df['Laptop Compartment'] = test_df['Laptop Compartment'].map({'Yes': 1, 'No': 0})
df['Waterproof'] = df['Waterproof'].map({'Yes': 1, 'No': 0})
test_df['Waterproof'] = test_df['Waterproof'].map({'Yes': 1, 'No': 0})


TE = ce.target_encoder.TargetEncoder()

cat_cols = ['Brand', 'Material', 'Style', 'Color', 'Size']

df[cat_cols] = TE.fit_transform(df[cat_cols], df['Price'])


test_df[cat_cols] = TE.transform(test_df[cat_cols])


imputer = ColumnTransformer(
    transformers=[
        ('imputer_num', SimpleImputer(strategy='mean'), ['Weight Capacity (kg)']),
        ('imputer_cat', SimpleImputer(strategy='most_frequent'), ['Brand', 'Material', 'Style', 'Color', 'Compartments', 'Size'])
        ], remainder='passthrough'
)


X = df.drop(['id', 'Price'], axis=1)
y = df['Price']


X_eval = test_df.drop(['id'], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def xgb_evaluation(max_depth, learning_rate, n_estimators, min_child_weight,
                   gamma, subsample, colsample_bytree, reg_alpha, reg_lambda):
    params = {
        'max_depth': int(max_depth),
        'learning_rate': learning_rate,
        'n_estimators': int(n_estimators),
        'min_child_weight': min_child_weight,
        'gamma': gamma,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'early_stopping_rounds':50
    }



    model = XGBRegressor(**params, objective='reg:squarederror')

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    y_pred = model.predict(X_test)
    return -np.sqrt(mean_squared_error(y_test, y_pred))

pbounds = {
    'max_depth': (3, 15),               # Controls tree depth (higher values increase model complexity)
    'learning_rate': (0.01, 0.3),       # Step size shrinkage for boosting
    'n_estimators': (50, 1000),         # Number of boosting rounds
    'min_child_weight': (1, 10),        # Minimum sum of instance weight in a child
    'gamma': (0, 1),                    # Minimum loss reduction required for further split
    'subsample': (0.5, 1.0),            # Fraction of training data for each boosting round
    'colsample_bytree': (0.5, 1.0),     # Fraction of features to use per tree
    'reg_alpha': (0, 10),               # L1 regularization (Lasso)
    'reg_lambda': (0, 10),              # L2 regularization (Ridge)
}


optimizer = BayesianOptimization(
    f=xgb_evaluation,
    pbounds=pbounds,
    random_state=42
)

optimizer.maximize(init_points=5, n_iter=25)

print(optimizer.max)


params ={'colsample_bytree': 0.6369945961213718, 'gamma': 0.8555783115262258, 'learning_rate': 0.1265882165755032, 'max_depth': 5, 'min_child_weight': 5.667786163772342, 'n_estimators': 398, 'reg_alpha': 2.8248405702787025, 'reg_lambda': 8.485330663706238, 'subsample': 0.9257241859043901}
xgb_model = XGBRegressor(**params)


xgb_model.fit(X, y, eval_set=[(X_test, y_test)])


logging.getLogger('lightgbm').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, message=".*Found whitespace in feature_names.*")

def lgb_evaluate(num_leaves, max_depth, learning_rate, n_estimators, min_child_samples, min_child_weight, subsample, colsample_bytree, reg_alpha, reg_lambda, gamma):
    params = {
        'metric': 'rmse',
        'n_estimators': int(n_estimators),
        'num_leaves': int(num_leaves),
        'max_depth': int(max_depth),
        'learning_rate': learning_rate,
        'verbose': -1,
        'min_child_samples': int(min_child_samples),
        'min_child_weight': min_child_weight,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'gamma': gamma,
        'early_stopping_rounds':30
    }

    # Train the model
    model = lgb.LGBMRegressor(**params)

    # Predict and evaluate
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)]
    )

    y_pred = model.predict(X_test)

    return -np.sqrt(mean_squared_error(y_test, y_pred))

# Set up the bounds for each hyperparameter
pbounds = {
    'num_leaves': (20, 300),              # Controls complexity; higher values capture more interactions
    'max_depth': (3, 15),                 # Limits tree depth (use -1 for no limit)
    'learning_rate': (0.01, 0.3),         # Step size shrinkage
    'n_estimators': (50, 1000),           # Number of boosting rounds
    'min_child_samples': (5, 100),        # Minimum data needed in a leaf
    'min_child_weight': (1e-3, 10),       # Minimum sum of instance weight in a leaf
    'subsample': (0.5, 1.0),              # Fraction of data for each boosting round (0.7 to prevent overfitting)
    'colsample_bytree': (0.5, 1.0),       # Fraction of features per tree
    'reg_alpha': (0, 10),                 # L1 regularization
    'reg_lambda': (0, 10),                # L2 regularization
    'gamma': (0, 1),                      # Minimum loss reduction for further partitioning
}

# Initialize Bayesian Optimization
optimizer = BayesianOptimization(
    f=lgb_evaluate,
    pbounds=pbounds,
    random_state=42,
)

# Perform optimization
optimizer.maximize(init_points=5, n_iter=25)

# Print the best result found by Bayesian Optimization
print(optimizer.max)


params =  {'colsample_bytree': 0.6759590696964592, 'gamma': 0.8382630619268105, 'learning_rate': 0.09691279408651629, 'max_depth': 5, 'min_child_samples': 57, 'min_child_weight': 7.278412715076966, 'n_estimators': 669, 'num_leaves': 73, 'reg_alpha': 6.568270186481002, 'reg_lambda': 5.186855204728545, 'subsample': 0.6224733589320252}


lgb_model = lgb.LGBMRegressor(**params)
lgb_model.fit(X, y)


! pip install catboost


def ctb_evaluate(num_leaves, max_depth, learning_rate, n_estimators):
    params = {
        'metric': 'rmse',
        'n_estimators': int(n_estimators),
        'num_leaves': int(num_leaves),
        'max_depth': int(max_depth),
        'learning_rate': learning_rate,
        'verbose': -1,
    }

    # Train the model
    model = lgb.LGBMRegressor(**params)

    # Predict and evaluate
    accuracy = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error').mean()

    return accuracy

# Set up the bounds for each hyperparameter
pbounds = {
    'n_estimators': (20, 150),  # Example range for n_estimators
    'border'
    'max_depth': (3, 16),      # Example range for max_depth
    'learning_rate': (0.01, 0.75) # Example range for learning_rate
}

# Initialize Bayesian Optimization
optimizer = BayesianOptimization(
    f=lgb_evaluate,
    pbounds=pbounds,
    random_state=42,
)

# Perform optimization
optimizer.maximize(init_points=5, n_iter=25)

# Print the best result found by Bayesian Optimization
print(optimizer.max)


import catboost


def catboost_evaluation(depth, learning_rate, iterations, l2_leaf_reg,
                         bagging_temperature, random_strength, border_count,
                         min_data_in_leaf, one_hot_max_size):

    params = {
        'depth': int(depth),
        'learning_rate': learning_rate,
        'iterations': int(iterations),
        'l2_leaf_reg': l2_leaf_reg,
        'bagging_temperature': bagging_temperature,
        'random_strength': random_strength,
        'border_count': int(border_count),
        'min_data_in_leaf': int(min_data_in_leaf),
        'one_hot_max_size': int(one_hot_max_size),
        'loss_function': 'RMSE',  # Regression task
        'eval_metric': 'RMSE',
        'verbose': False
    }

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    cat_model = catboost.CatBoostRegressor(**params)
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)

    # Predict and evaluate
    y_pred = cat_model.predict(X_val)
    return -np.sqrt(mean_squared_error(y_val, y_pred))  # Negative RMSE for maximization


pbounds = {
    'depth': (4, 12),  # Tree depth (4 to 12)
    'learning_rate': (0.01, 0.3),  # Learning rate (0.01 to 0.3)
    'iterations': (100, 1000),  # Number of trees (100 to 1000)
    'l2_leaf_reg': (1, 10),  # L2 regularization (1 to 10)
    'bagging_temperature': (0, 1),  # Controls subsampling (0 to 1)
    'random_strength': (1, 10),  # Weight of random splits (1 to 10)
    'border_count': (32, 255),  # Number of splits for numeric features (32 to 255)
    'min_data_in_leaf': (1, 50),  # Minimum number of samples in a leaf (1 to 50)
    'one_hot_max_size': (2, 10),  # One-hot encoding for categorical features (2 to 10)
}


# Bayesian Optimization
optimizer = BayesianOptimization(
    f=catboost_evaluation,
    pbounds=pbounds,
    random_state=42
)

optimizer.maximize(init_points=5, n_iter=25)
print(optimizer.max)


cat_params = {'iterations': 500, 'depth': 6, 'l2_leaf_reg': 2.7077429266423243, 'learning_rate': 0.051879077487633055, 'verbose':100, 'loss_function': 'RMSE', 'eval_metric': 'RMSE'}


cat_model = catboost.CatBoostRegressor(**cat_params)


pd.DataFrame({'id': test_df['id'], 'Price': model.predict(X_eval)}).to_csv('submission.csv', index=False)


!kaggle competitions submit -c playground-series-s5e2 -f submission.csv -m "CatBoostRegressor(max_depth=5, learning_rate=0.033163999000637275, l2_leaf_reg= 9.182247292936465, loss_function='RMSE', eval_metric='RMSE', verbose=0)"

