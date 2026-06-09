import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


from sklearn.model_selection import train_test_split

# Read the data (concat to add additional training data)
# train_data = pd.concat([pd.read_csv("../input/playground-series-s5e2/train.csv", index_col="id"), pd.read_csv("../input/playground-series-s5e2/training_extra.csv", index_col="id")], axis=0)
train_data = pd.read_csv("/kaggle/input/student-bag-price-prediction-dataset/Noisy_Student_Bag_Price_Prediction_Dataset.csv")
train_data = train_data.dropna(axis=0).drop_duplicates()
X = train_data.drop(["Price"], axis=1)
y = train_data.Price
X_test = pd.read_csv("../input/playground-series-s5e2/test.csv", index_col="id")


X


from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


# Get columns with qualitative entries and their corresponing cardinality
object_cols = [col for col in X.columns 
               if X[col].dtype == "object"]
n_unique = [X[col].nunique() for col in object_cols]
print(dict(zip(object_cols, n_unique)))

# Select which columns will be encoded with one-hot encoding and which will be encoded with ordinal encoding
ordinal_cols = ["Size", "Laptop Compartment", "Waterproof"]
OH_cols = list(set(object_cols)-set(ordinal_cols))


# Define new dataframes for labeled values
labeled_X = X.drop(OH_cols, axis=1)
labeled_X_test = X_test.drop(OH_cols, axis=1)


# Ordinal Encoding
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
labeled_X[ordinal_cols] = enc.fit_transform(labeled_X[ordinal_cols])
labeled_X_test[ordinal_cols] = enc.transform(labeled_X_test[ordinal_cols])

# One-hot Encoding
enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
OH_X = pd.DataFrame(enc.fit_transform(X[OH_cols]))
OH_X_test = pd.DataFrame(enc.transform(X_test[OH_cols]))

labeled_X = pd.concat([labeled_X.reset_index(drop=True), OH_X.reset_index(drop=True)], axis=1)
labeled_X_test = pd.concat([labeled_X_test.reset_index(drop=True), OH_X_test.reset_index(drop=True)], axis=1)


# Make labels string for imputer
labeled_X.columns = labeled_X.columns.astype(str)
labeled_X_test.columns = labeled_X_test.columns.astype(str) 

# Impute numerical entries
imputer = SimpleImputer()

cols = labeled_X.columns
labeled_X = pd.DataFrame(imputer.fit_transform(labeled_X))
labeled_X_test = pd.DataFrame(imputer.transform(labeled_X_test))
labeled_X.columns = cols
labeled_X_test.columns = cols


labeled_X


from sklearn.metrics import mean_squared_error

def evaluate_model(model, X_train, X_val, y_train, y_val):
    model.fit(X_train, y_train, **{ 
        'eval_set' : [(X_val, y_val)],
        'verbose' : False}) # fit model
    preds = model.predict(X_val) # get model predictions
    return np.sqrt(mean_squared_error(y_val, preds)) # return RMSE


import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR 
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

# Initialize the models
cat_model = CatBoostRegressor(
    loss_function='RMSE',
    verbose=False
    )
xgb_model1 = xgb.XGBRegressor(
    objective='reg:squarederror', 
    max_depth=4,
    learning_rate=0.01,
    n_estimators=3000,
    colsample_bytree=0.5,
    min_child_weight=1,
    early_stopping_rounds=5,
    n_jobs=4,
    random_state=0
    )

xgb_model2 = xgb.XGBRegressor(
    objective='reg:squarederror', 
    max_depth=15,
    learning_rate=0.01,
    n_estimators=3000,
    colsample_bytree=0.5,
    min_child_weight=1,
    early_stopping_rounds=5,
    n_jobs=4,
    tree_method='hist',
    random_state=0
    )

xgb_model3 = xgb.XGBRegressor(
    objective='reg:squarederror', 
    max_depth=10,
    learning_rate=0.05,
    n_estimators=3000,
    colsample_bytree=0.5,
    min_child_weight=1,
    early_stopping_rounds=5,
    n_jobs=4,
    random_state=0
    )
elnet_model1 = ElasticNet(
    l1_ratio=0.25,
    random_state=0
)
elnet_model2 = ElasticNet(
    l1_ratio=0.5,
    random_state=0
)
elnet_model3 = ElasticNet(
    l1_ratio=0.75,
    random_state=0
)
rf_model = RandomForestRegressor(random_state=0)
svm_model = SVR()
mlp_model1 = MLPRegressor(
    hidden_layer_sizes=(50, ), 
    activation="logistic")
mlp_model2 = MLPRegressor(
    hidden_layer_sizes=(50, ), 
    activation="relu")
mlp_model3 = MLPRegressor(
    hidden_layer_sizes=(75, ), 
    activation="logistic")

# Split data
X_train, X_val, y_train, y_val = train_test_split(labeled_X, y, train_size=0.7)

models_tags = [
               # ("CatBoost", cat_model),
               # ("XGBoost 1", xgb_model1),
               ("XGBoost 2", xgb_model2),
               # ("XGBoost 3", xgb_model3),
               # ("Elastic Net 1", elnet_model1), 
               # ("Elastic Net 2", elnet_model2), 
               # ("Elastic Net 3", elnet_model3), 
               # ("Random Forests", rf_model), 
               # ("Support Vector Machines", svm_model),
               # ("Neural Network 1", mlp_model1),
               # ("Neural Network 2", mlp_model2),
               # ("Neural Network 3", mlp_model3),
              ]

# Uncomment code below to run comparison
for (tag, model) in models_tags:
    print(f"{tag} RMSE:\t {evaluate_model(model, X_train, X_val, y_train, y_val):.3f}")


xgb_model2.best_iteration


# fit chosen  model on all data
best_model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    max_depth=15,
    learning_rate=0.01,
    n_estimators=180,
    colsample_bytree=0.5,
    min_child_weight=1,
    tree_method='hist',
    random_state=0
    )
best_model.fit(labeled_X, y)


# get predictions
preds = best_model.predict(labeled_X_test)

# get submission dataframe
submission = pd.DataFrame({"id" : labeled_X_test.index + 300000, "Price" : preds})

# save submission file
submission.to_csv('../../kaggle/working/submission.csv', index=False)
print("Submission file saved!")


submission

