
import numpy as np 
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train_data = pd.read_csv("/kaggle/input/playground-series-s4e12/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s4e12/test.csv")


train_data.info()


train_data.isnull().sum()





train_data.nunique()


train_data.drop(columns=['id', 'Policy Start Date'], inplace=True)
test_data.drop(columns=['id', 'Policy Start Date'], inplace=True)





# extract X and y from training data
X = train_data.drop(['Premium Amount'], axis=1)
y = train_data['Premium Amount']


# extract the numerical and categorical columns
numeric_cols = X.select_dtypes(include="number").columns
categorical_cols = X.select_dtypes(include="object").columns

print("Numerical Columns are:\n", numeric_cols)
print()
print("Categorical Columns are: \n", categorical_cols)


# fill the missing values
from sklearn.impute import SimpleImputer

# Numeric imputer (median)
num_imputer = SimpleImputer(strategy='median')
X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

# Categorical imputer (mode)
cat_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])



# now, checking null counts of every column
X.isna().sum()


# a informative summary
for col in categorical_cols:
    print(col)
    print("Uniques count: ", X[col].nunique())
    print("The values are : \n", pd.DataFrame(X[col].value_counts().reset_index()))
    print()


# split data into train and test parts
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)


X_train


# filling the categorical values with corresponding numerical values
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder()

X_train[categorical_cols] = oe.fit_transform(X_train[categorical_cols])
X_test[categorical_cols] = oe.transform(X_test[categorical_cols])


# scaling the numerical cols types data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train[numeric_cols] = sc.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = sc.transform(X_test[numeric_cols])


X_train





Xy_sample = X_train.join(y_train).sample(20000, random_state=42)
X_train_sample = Xy_sample.drop("Premium Amount", axis=1)
y_train_sample = Xy_sample["Premium Amount"]


# developing normal models
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
import numpy as np

models = {
    "LinearRegression": LinearRegression(),
    "Lasso": Lasso(alpha=0.001, max_iter=10000),
    "Ridge": Ridge(alpha=1.0),
    "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000)
}

results = {}

for name, model in models.items():
    model.fit(X_train_sample, y_train_sample)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = rmse

# Print results
for model, rmse in results.items():
    print(f"{model}: RMSE = {rmse:.2f}")



from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

models = {
    "DecisionTree": DecisionTreeRegressor(random_state=42, max_depth=10),
    "RandomForest": RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10),
    "GradientBoosting": GradientBoostingRegressor(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=5)
}

results = {}

for name, model in models.items():
    # model training on sample data
    model.fit(X_train_sample, y_train_sample)

    # prediction on the whole test data
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = rmse

# Print results
for model, rmse in results.items():
    print(f"{model}: RMSE = {rmse:.2f}")



# finding the best parameters of both best algorithms using GridSearchCV

from sklearn.model_selection import GridSearchCV

# Models and param grids
models = {
    "RandomForest": (RandomForestRegressor(random_state=42, n_jobs=-1),
                     {
                         'n_estimators': [100, 200],
                         'max_depth': [10, 20, None],
                         'min_samples_split': [2, 5],
                         'min_samples_leaf': [1, 2],
                         'max_features': ['sqrt', 'log2']
                     }),
    "GradientBoosting": (GradientBoostingRegressor(random_state=42),
                         {
                             'n_estimators': [100, 200],
                             'learning_rate': [0.05, 0.1],
                             'max_depth': [3, 5],
                             'min_samples_split': [2, 5],
                             'min_samples_leaf': [1, 2]
                         })
}

results = {}

for name, (model, params) in models.items():
    grid = GridSearchCV(model, params, cv=3,
                        scoring='neg_root_mean_squared_error',
                        n_jobs=-1, verbose=1)
    grid.fit(X_train_sample, y_train_sample)
    results[name] = {"best_params": grid.best_params_, "best_rmse": -grid.best_score_}

# Print results
for name, res in results.items():
    print(f"{name}: Best RMSE = {res['best_rmse']:.2f}, Params = {res['best_params']}")




# re-building the both best models with best parameters, to get true results.

# Best params from GridSearchCV
best_rf_params = {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 2, 
                  'min_samples_split': 5, 'n_estimators': 200}

best_gb_params = {'learning_rate': 0.05, 'max_depth': 5, 'min_samples_leaf': 2, 
                  'min_samples_split': 5, 'n_estimators': 100}

# Refit models on full training set
rf_best = RandomForestRegressor(**best_rf_params, random_state=42)
rf_best.fit(X_train_sample, y_train_sample)

gb_best = GradientBoostingRegressor(**best_gb_params, random_state=42)
gb_best.fit(X_train_sample, y_train_sample)

# Evaluate on test set
rf_preds = rf_best.predict(X_test)
gb_preds = gb_best.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_preds))

print(f"Final RandomForest RMSE on Test set: {rf_rmse:.2f}")
print(f"Final GradientBoosting RMSE on Test set: {gb_rmse:.2f}")








