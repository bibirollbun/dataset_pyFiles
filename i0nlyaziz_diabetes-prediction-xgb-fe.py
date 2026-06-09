import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# import the necessary libraries


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

# load the data


test_ids = test['id']

# save the id column


train.head()


test.head()


train["activity_screen_ratio"] = train["physical_activity_minutes_per_week"] / (train["screen_time_hours_per_day"] + 1)
train["chol_ratio"] = train["cholesterol_total"] / train["hdl_cholesterol"]
train["bp_ratio"] = train["systolic_bp"] / train["diastolic_bp"]
train["bmi_age"] = train["bmi"] * train["age"]


test["activity_screen_ratio"] = test["physical_activity_minutes_per_week"] / (test["screen_time_hours_per_day"] + 1)
test["chol_ratio"] = test["cholesterol_total"] / test["hdl_cholesterol"]
test["bp_ratio"] = test["systolic_bp"] / test["diastolic_bp"]
test["bmi_age"] = test["bmi"] * test["age"]


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

columnns = ['gender','ethnicity','education_level','income_level','smoking_status','employment_status']

for i in columnns :
  train[i] = encoder.fit_transform(train[i])

# preprocess the train data


train.drop(columns=['id'],inplace=True)

# drop the unnecessary column


train.isnull().sum().sum()

# check for missing values


train.duplicated().sum()

# check for duplicates values


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

columnns = ['gender','ethnicity','education_level','income_level','smoking_status','employment_status']

for i in columnns :
  test[i] = encoder.fit_transform(test[i])

# preprocess the train data


test.drop(columns=['id'],inplace=True)

# drop the unnecessary column


test.isnull().sum().sum()

# check for missing values


test.duplicated().sum()

# check for duplicates values


x = train.drop(columns=['diagnosed_diabetes'],axis=1)
y = train['diagnosed_diabetes']


from sklearn.model_selection import train_test_split

x_train , x_valid , y_train , y_valid = train_test_split(x,y,test_size=0.3,random_state=42)

# split the data


from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

xgb_model = XGBClassifier(random_state=42, verbosity=0)

xgb_params = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 1, 5],
    "reg_alpha": [0, 0.1, 1],
    "reg_lambda": [1, 5, 10],
    "min_child_weight": [1, 3, 5]
}

xgb_random = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=xgb_params,
    n_iter=50,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1,
    random_state=42
)

xgb_random.fit(x_train, y_train)

print("Best XGB params:", xgb_random.best_params_)
print("Best XGB score:", xgb_random.best_score_)

# use RandomizedSearchCV to find the best hyperparameters


from lightgbm import LGBMClassifier

Model = XGBClassifier(
    subsample=1.0,
    reg_lambda=5,
    reg_alpha=1,
    n_estimators=200,
    min_child_weight=1,
    max_depth=5,
    learning_rate=0.2,
    gamma=0,
    colsample_bytree=0.6
)

Model.fit(x_train, y_train)

# train the model


from sklearn.metrics import roc_auc_score

y_proba = Model.predict_proba(x_valid)[:, 1]

auc_score = roc_auc_score(y_valid, y_proba)
print(f"AUC Score: {auc_score:.4f}")

# evaluate the model with AUC score


from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_valid, y_proba)


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'XGBoost (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# evaluate the model with ROC curve


from xgboost import plot_importance

plt.figure(figsize=(12, 8))
plot_importance(Model, max_num_features=20, importance_type='gain', height=0.6)
plt.title("Top 20 Feature Importances")
plt.show()

# visualize feature importance


predictions = Model.predict_proba(test)[:, 1]


submission = pd.DataFrame({
    "id": test_ids,
    "diagnosed_diabetes": predictions
})

submission.to_csv("submission.csv", index=False)


submission

