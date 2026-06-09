!pip install --force-reinstall --no-deps scikit-learn==1.3.2 imbalanced-learn==0.11.0


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model  import LogisticRegression
from sklearn.metrics import r2_score, classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
import sklearn.utils
from imblearn.over_sampling import SMOTE


data = pd.read_csv("/kaggle/input/ultimate-customer-churn-prediction-challenge/train.csv")
test_data = pd.read_csv("/kaggle/input/ultimate-customer-churn-prediction-challenge/test.csv")
data.head()


data.info()


data.describe()


le = LabelEncoder()
data["Gender"] = le.fit_transform(data["Gender"])
test_data["Gender"] = le.fit_transform(test_data["Gender"])
data["Location"] = le.fit_transform(data["Location"])
test_data["Location"] = le.fit_transform(test_data["Location"])
data["Subscription_Type"] = le.fit_transform(data["Subscription_Type"])
test_data["Subscription_Type"] = le.fit_transform(test_data["Subscription_Type"])
data["Last_Interaction_Type"] = le.fit_transform(data["Last_Interaction_Type"])
test_data["Last_Interaction_Type"] = le.fit_transform(test_data["Last_Interaction_Type"])
data.head()


print(data.shape)
data = data.drop_duplicates()
print(data.shape)


data = data.drop("Customer_ID", axis=1)
data["Monthly_Spending"] = data["Monthly_Spending"].astype(int)
test_data["Monthly_Spending"] = test_data["Monthly_Spending"].astype(int)
data.head()


corr_matrix = data.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix Heatmap", fontsize=14)
plt.show()


scaler = StandardScaler()
cols = ["Age","Gender","Location","Account_Age_Months","Monthly_Spending","Total_Usage_Hours",
                "Support_Calls","Late_Payments","Streaming_Usage","Discount_Used", "Promo_Opted_In",
                "Satisfaction_Score","Complaint_Tickets","Subscription_Type","Last_Interaction_Type"]

data[cols] = scaler.fit_transform(data[cols])
test_data[cols] = scaler.fit_transform(test_data[cols])
data.head()


X = data.drop('Churn', axis=1)
y = data['Churn']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)


xgb = XGBClassifier(
    use_label_encoder=False, 
    eval_metric="auc", 
    random_state=42
)

param_grid = {
    "n_estimators": [100, 200, 250],       # number of boosting rounds (trees)
    "max_depth": [3, 5, 7],           # tree depth
    "learning_rate": [0.05, 0.1],     # shrinkage step
    "subsample": [0.8, 1.0],          # row sampling
    "colsample_bytree": [0.8, 1.0]    # feature sampling
}


grid_xgb = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring="r2",   
    cv=5,               
    verbose=1
)

grid_xgb.fit(X_train, y_train)

print("Best Params:", grid_xgb.best_params_)
print("Best CV Score:", grid_xgb.best_score_)
best_xgb = grid_xgb.best_estimator_
y_pred = best_xgb.predict(X_val)


# print("Using XGBoost:")
# xgb_model = XGBClassifier(n_estimators=350, learning_rate=0.05, max_depth=3)
# xgb_model.fit(X_train, y_train)
# y_pred = xgb_model.predict(X_val)
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))
print("R2 Score", r2_score(y_val, y_pred))


X_test = test_data.drop('Customer_ID', axis=1)
y_pred = best_xgb.predict(X_test)
submission = pd.DataFrame({
    'Customer_ID': test_data['Customer_ID'],
    'Churn': y_pred
})
submission.to_csv('submission.csv', index=False)
submission.head()

