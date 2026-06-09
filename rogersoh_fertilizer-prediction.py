# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import matplotlib.pyplot as plt

# statistic
from scipy.stats import chi2_contingency


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
train.head()


test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


# Feature engineering:
def feature_engineer(df):
    df = df.copy()
    df["soil_crop"] = df["Soil Type"]+"_"+ df["Crop Type"]
    df["temp_humidity"] = df["Temparature"] * df["Humidity"]
    df["temp_moisture"] = df["Temparature"] * df["Moisture"]
    df["humidity_moisture"] = df["Humidity"] * df["Moisture"]
    df["Potassium_Nitrogen"] = df["Potassium"] / df["Nitrogen"]
    df["Phosphorous_Nitrogen"] = df["Phosphorous"] / df["Nitrogen"]
    # df["Phosphorous_Potassium"] = df["Phosphorous"]/(df["Potassium"]+ 1e-6)
    return df


print("Duplicates = ",train.duplicated().sum())
print("Null or NA = ", train.isna().sum().sum())


train.describe()


import seaborn as sns

cols = ["Temparature","Moisture","Soil Type",'Crop Type',]
fig = plt.figure(figsize=(18,10), layout = "constrained")
for i,col in enumerate(cols):
    plt.subplot(2,2,i+1)
    sns.countplot(train, x=col)
    plt.xticks(rotation = 90)
    plt.title(col)
plt.show()


sns.catplot(train, x = "Fertilizer Name", y= "Phosphorous",kind="box",);
plt.xticks(rotation = 90);



long = pd.melt(train, id_vars = ["Fertilizer Name"],
               value_vars = ["Nitrogen", "Potassium", "Phosphorous"])
long.head()


sns.catplot(long, x = "variable", y = "value", hue="Fertilizer Name", kind = "box")
plt.xticks(rotation =  90)
plt.show()


long = pd.melt(train, id_vars = ["Fertilizer Name"],
               value_vars = ["Temparature", "Humidity", "Moisture",
                             "Nitrogen", "Potassium", "Phosphorous"])

fig = plt.figure(figsize = (12, 8))
sns.catplot(data = long, x = "variable", y = "value", kind = "box", hue="Fertilizer Name")
plt.xticks(rotation = 90)
plt.title("Box plot of the measured category")
plt.show()


contingency = pd.crosstab(train['Soil Type'], train['Fertilizer Name'])
print("Cross Tab Soil Type and Fertilizer Name\n",contingency)
sns.heatmap(contingency);


chi2, p, dof, expected = chi2_contingency(contingency)

print(f"Chi2 statistic: {chi2:.4f}")
print(f"p-value: {np.round(p, 4)}")
print(f"degree of freedom: {dof}")


soil_contingency = pd.crosstab(train["Crop Type"], train["Fertilizer Name"])
print("Cross Tab Crop Type and Fertilizer Name\n",soil_contingency)
sns.heatmap(soil_contingency);


chi2, p, dof, expected = chi2_contingency(soil_contingency)
print(f"Chi2 statistic: {chi2:.4f}")
print(f"p-value: {np.round(p, 4)}")
print(f"degree of freedom: {dof}")


from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, MinMaxScaler, LabelEncoder
from sklearn import set_config
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, f1_score, recall_score
from sklearn.metrics import roc_curve, precision_score
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import xgboost as xgb
from xgboost import XGBClassifier

from scipy.stats import randint, uniform

import time
import joblib # for saving the model

# set_config(transform_output="pandas") 
RandomStateNumber = 13



# setup the training, validation data
train.head()
X = train.drop(["Fertilizer Name"], axis = 1)
y = train["Fertilizer Name"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size= 0.2, random_state= RandomStateNumber, stratify= y)
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_val = le.transform(y_val)


filename = "/kaggle/working/le_classes.pkl"
joblib.dump(le, filename)


# set up Pipeline
feature_transformer = FunctionTransformer(feature_engineer)

numeric_features = [
    "Temparature", "Humidity", "Moisture",
    "Nitrogen", "Potassium", "Phosphorous",
]

num_scalers_features = ['temp_humidity','temp_moisture',
     'humidity_moisture','Potassium_Nitrogen',
     'Phosphorous_Nitrogen']

categorical_features = ["Soil Type", "Crop Type", "soil_crop"]

# Column Transformer for the different transformation OneHotEncoder(sparse_output=False)
transformColumns = ColumnTransformer(
    transformers =[
        ('num', MinMaxScaler(), numeric_features),
        ('z_scaler', StandardScaler(), num_scalers_features),
         ('cat', OneHotEncoder(), categorical_features)
         ],
    remainder = "passthrough",
    # verbose_feature_names_out= False
)

preprocessing = Pipeline(
    [
        ('feature_engineering', feature_transformer),
        ('transformColumns', transformColumns)
        ]
    )

# # Create pipeline
pipe = Pipeline([
    ("preprocess", preprocessing),
    ('classifier', XGBClassifier(random_state= RandomStateNumber)) # Placeholder model
])

#Data without feature engineering
cat_features = ["Soil Type", "Crop Type"]

transformCol = ColumnTransformer(
    transformers =[
        ('num', MinMaxScaler(), numeric_features),
         ('cat', OneHotEncoder(), cat_features)
         ],
    remainder = "passthrough",
    # verbose_feature_names_out= False
)



# pipeline for reduced features
num_reduced = [ "Moisture",
    "Nitrogen", "Potassium", "Phosphorous",
]


transformCol_reduced = ColumnTransformer(
    transformers =[
        ('num', MinMaxScaler(), num_reduced),
         ('cat', OneHotEncoder(), cat_features)
         ],
    remainder = "passthrough",
    # verbose_feature_names_out= False
)


# Stratified KFold for cross valuation score

stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state= RandomStateNumber)



# custom scoring: map_at_3

def map_at_3(y: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Compute MAP@3 using NumPy arrays.    
    Args:
        y: 1D array of true labels (n_observations,)
        y_pred_proba: 2D array of predicted labels (n_observations, n_predictions)
                Must have at least 3 columns
    Returns:
        MAP@3 score (float)
    """
    # Extract top 3 predictions
    top3 = np.argsort(y_pred_proba, axis=1)[:, -3:][:, ::-1]  
    # Create boolean mask of correct predictions
    correct_mask = (top3 == y[:, None])
    # Find first correct position in each row
    first_correct_idx = np.argmax(correct_mask, axis=1)
    # Mask for rows with at least one correct prediction
    has_correct = np.any(correct_mask, axis=1)
    # Compute precision scores: 1/(position+1) for correct predictions, else 0
    precisions = np.where(has_correct, 1.0 / (first_correct_idx + 1), 0.0)
    return np.mean(precisions)



from sklearn.metrics import make_scorer
# Create scorer object
map3_scorer_obj = make_scorer(map_at_3, needs_proba=True, greater_is_better=True)


param_grid = {
    "classifier":[XGBClassifier(random_state= RandomStateNumber)],
    "classifier__n_estimators": [100, 200, 400],
    "classifier__learning_rate": uniform(0.01, 0.3),
    "classifier__max_depth": randint(3, 10),
    "classifier__min_child_weight": [3,5, 7],
    "classifier__subsample": uniform(0.5, 0.85),
    "classifier__colsample_bytree": [0.7, 0.85, 1.0],
    "classifier__reg_alpha": uniform(0.0, 2.0),
    "classifier__reg_lambda": uniform(0.0, 5.0)
}


# xgb_model = XGBClassifier(random_state= RandomStateNumber)
search = RandomizedSearchCV(
    pipe, param_grid, cv = stratified_cv, 
    scoring = map3_scorer_obj, n_iter=20, 
    n_jobs=-1, verbose = 2
)


early_stop = False # set logic for early stop


# Hyperparameter Tuning

# import time
# start = time.time()
# search.fit(X_train, y_train)
# timeTaken = time.time() - start
# Result of Hyperparameter tuning

# print(f"time taken for randomizesearchcv, {timeTaken:.0f}\n")
# best_xgb = search.best_estimator_
# best_param = search.best_params_
# print("best Parameter:", best_param)
# print("best score MAP@3", search.best_score_)

# import joblib
# filename = "/kaggle/working/best_model.pkl"
# joblib.dump(best_xgb, filename)


params_final = {"n_estimators":147,
                      "colsample_bytree": 0.7, 
                      "learning_rate" : 0.19706409256626184, 
                      "max_depth" : 7, "min_child_weight":3,
                      "reg_alpha" : 1.0693958585032026, 
                      "reg_lambda": 2.8860794492405994, 
                      "subsample" : 0.7957520995589666,
                     "random_state" : RandomStateNumber}


params = {"n_estimators":1000, "early_stopping_rounds":10, 
                      "eval_metric" : "mlogloss", 
                      "colsample_bytree": 0.7, 
                      "learning_rate" : 0.19706409256626184, 
                      "max_depth" : 7, "min_child_weight":3,
                      "reg_alpha" : 1.0693958585032026, 
                      "reg_lambda": 2.8860794492405994, 
                      "subsample" : 0.7957520995589666,
                     "random_state" : RandomStateNumber}


X_train_scaled = pipe.named_steps['preprocess'].fit_transform(X_train)
X_val_scaled = pipe.named_steps['preprocess'].transform(X_val)


if early_stop:
    model = XGBClassifier(**params)
    # Fit the model with early stopping
    model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)    
    # Report the best score on the best iteration
    print(f'Best score {model.best_score}, Best iteration {model.best_iteration}')
else:
    model = XGBClassifier(**params_final)
    model.fit(X_train_scaled, y_train, verbose=False)
    


print("Validation metrics\n")
y_pred_proba = model.predict_proba(X_val_scaled)
map_3 = map_at_3(y_val, y_pred_proba)
print("MAP@3 score:", map_3)
y_pred = model.predict(X_val_scaled)
cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=le.classes_)
disp.plot()
plt.title(f" Confusion Matrix")
plt.xticks(rotation = 90)
plt.show()

print(classification_report(y_val, y_pred, target_names=le.classes_))


y_val_label = le.classes_[y_val]
y_pred_label = le.classes_[y_pred]
top3 = np.argsort(y_pred_proba, axis=1)[:, -3:][:, ::-1]
top3_label = le.classes_[top3]


label_prediction = {"y_true":y_val_label, "y_pred": y_pred_label}
prediction_df = pd.DataFrame(label_prediction)
prediction_df["Top 3 prediction"]= [" ".join(row) for row in top3_label]

print("Validation dataset")
prediction_df.head()


X_train_2 = transformCol.fit_transform(X_train)
X_val_2 = transformCol.transform(X_val)


if early_stop:
    model_2 = XGBClassifier(**params)
    # Fit the model with early stopping
    model_2.fit(X_train_2, y_train, eval_set=[(X_val_2, y_val)], verbose=False)
    # Report the best score on the best iteration
    print(f'Best score {model.best_score}, Best iteration {model.best_iteration}')
else:
    model_2 = XGBClassifier(**params_final)
    model_2.fit(X_train_2, y_train, verbose = False)


print("Validation metrics\n")
y_pred_proba = model_2.predict_proba(X_val_2)
map_3_2 = map_at_3(y_val, y_pred_proba)
print("MAP@3 score:", map_3_2)
y_pred = model_2.predict(X_val_2)
cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=le.classes_)
disp.plot()
plt.title(f" Confusion Matrix")
plt.xticks(rotation = 90)
plt.show()

print(classification_report(y_val, y_pred, target_names=le.classes_))


X_train_3 = transformCol_reduced.fit_transform(X_train)
X_val_3 = transformCol_reduced.transform(X_val)

if early_stop:
    model_3 = XGBClassifier(**params)
    # Fit the model with early stopping
    model_3.fit(X_train_3, y_train, eval_set=[(X_val_3, y_val)], verbose=False)
    # Report the best score on the best iteration
    print(f'Best score {model.best_score}, Best iteration {model.best_iteration}')
else:
    model_3 = XGBClassifier(**params_final)
    model_3.fit(X_train_3, y_train)


print("Validation metrics\n")
y_pred_proba = model_3.predict_proba(X_val_3)
map_3_3 = map_at_3(y_val, y_pred_proba)
print("MAP@3 score:", map_3_3)
y_pred = model_3.predict(X_val_3)
cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=le.classes_)
disp.plot()
plt.title(f" Confusion Matrix")
plt.xticks(rotation = 90)
plt.show()

print(classification_report(y_val, y_pred, target_names=le.classes_))


map_3_List = [map_3, map_3_2, map_3_3]
print(map_3_List)


np.argmax(map_3_List)


match np.argmax(map_3_List):
    case 0:
        test_transform = pipe.named_steps['preprocess'].transform(test)
        y_pred_proba = model.predict_proba(test_transform)
        print("features engineering model have high MAP@3")
    case 1:
        test_transform = transformCol.transform(test)
        y_pred_proba = model_2.predict_proba(test_transform)
        print("model without features engineering have high MAP@3")
    case 2:
        test_transform = transformCol_reduced.transform(test)
        y_pred_proba = model_3.predict_proba(test_transform)
        print("reduced model have high MAP@3")

top3 = np.argsort(y_pred_proba, axis=1)[:, -3:][:, ::-1]
top3_label = le.classes_[top3]

# save submission.csv
submission["Fertilizer Name"] = [" ".join(row) for row in top3_label]

submission.to_csv("/kaggle/working/submission.csv", index=False)
submission.head()

