import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import VotingRegressor

from warnings import filterwarnings
filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


train.head()


train.shape


test.shape


train.describe()


train.isnull().sum().sum()


train.info()


numeric_list = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp", "Calories"]

melted_train = train[numeric_list].melt(var_name="Variable", value_name="Value")

plt.figure(figsize=(12, 6))
sns.boxplot(data=melted_train, x="Variable", y="Value", palette="Set2")
plt.title("Boxplot of Variables", fontsize=14)
plt.ylabel("Value Range")
plt.xlabel("")
plt.grid(axis='y')

plt.tight_layout()
plt.show()


numeric_list = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]

melted_test = test[numeric_list].melt(var_name="Variable", value_name="Value")

plt.figure(figsize=(12, 6))
sns.boxplot(data=melted_test, x="Variable", y="Value", palette="Set2")
plt.title("Boxplot of Variables", fontsize=14)
plt.ylabel("Value Range")
plt.xlabel("")
plt.grid(axis='y')

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
corr_matrix = train[numeric_list].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heat Map of Numerical Variables")
plt.tight_layout()
plt.show()


train[numeric_list].corr()


def cap_outliers_iqr(train_df, test_df, columns):
    for col in columns:
        Q1 = train_df[col].quantile(0.25)
        Q3 = train_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR

        train_df.loc[train_df[col] > upper_limit, col] = upper_limit
        train_df.loc[train_df[col] < lower_limit, col] = lower_limit

        test_df.loc[test_df[col] > upper_limit, col] = upper_limit
        test_df.loc[test_df[col] < lower_limit, col] = lower_limit

    return train_df, test_df


cols = ["Height", "Weight", "Heart_Rate", "Body_Temp"]
train, test = cap_outliers_iqr(train, test, cols)


Q3 = train["Calories"].quantile(0.75)
Q1 = train["Calories"].quantile(0.25)
IQR = Q3 - Q1

Upper_limit = Q3 + 1.5 * IQR


outlier_upper = train["Calories"] > Upper_limit


train.loc[outlier_upper, "Calories"] = Upper_limit


train["BMI"] = train["Weight"] / (train["Height"] / 100) ** 2
test["BMI"] = test["Weight"] / (test["Height"] / 100) ** 2

train["Effort_Score"] = train["Heart_Rate"] * train["Duration"]
test["Effort_Score"] = test["Heart_Rate"] * test["Duration"]

train["HR_per_min"] = train["Heart_Rate"] / train["Duration"]
test["HR_per_min"] = test["Heart_Rate"] / test["Duration"]

train["HeartRate_per_BodyTemp"] = train["Heart_Rate"] / train["Body_Temp"]
test["HeartRate_per_BodyTemp"] = test["Heart_Rate"] / test["Body_Temp"]

train["HR_age_ratio"] = train["Heart_Rate"] / train["Age"]
test["HR_age_ratio"] = test["Heart_Rate"] / test["Age"]


train["Weight_per_Height"] = train["Weight"] / train["Height"]
test["Weight_per_Height"] = test["Weight"] / test["Height"]

train["Age_BMI"] = train["Age"] * train["BMI"]
test["Age_BMI"] = test["Age"] * test["BMI"]

train["BodyTemp_Duration"] = train["Body_Temp"] * train["Duration"]
test["BodyTemp_Duration"] = test["Body_Temp"] * test["Duration"]

mean_body_temp_train = train["Body_Temp"].mean()
train["BodyTemp_Deviation"] = train["Body_Temp"] - mean_body_temp_train
mean_body_temp_test = test["Body_Temp"].mean()
test["BodyTemp_Deviation"] = test["Body_Temp"] - mean_body_temp_test

train["Max_Heart_Rate"] = 220 - train["Age"]
test["Max_Heart_Rate"] = 220 - test["Age"]
train["HeartRate_Max_HR_Ratio"] = train["Heart_Rate"] / train["Max_Heart_Rate"]
test["HeartRate_Max_HR_Ratio"] = test["Heart_Rate"] / test["Max_Heart_Rate"]

train["Effort_Score_per_Duration"] = train["Effort_Score"] / train["Duration"]
test["Effort_Score_per_Duration"] = test["Effort_Score"] / test["Duration"]

train["Weight_Age"] = train["Weight"] * train["Age"]
test["Weight_Age"] = test["Weight"] * test["Age"]

train["Weight_per_BMI"] = train["Weight"] / train["BMI"]
test["Weight_per_BMI"] = test["Weight"] / test["BMI"]

train["Exercise_Duration_Category"] = pd.cut(train["Duration"], bins=[0, 30, 60, 180], labels=["Short", "Medium", "Long"])
test["Exercise_Duration_Category"] = pd.cut(test["Duration"], bins=[0, 30, 60, 180], labels=["Short", "Medium", "Long"])

train["Age_Group"] = pd.cut(train["Age"], bins=[0, 30, 50, 100], labels=["Young", "Middle-Aged", "Old"])
test["Age_Group"] = pd.cut(test["Age"], bins=[0, 30, 50, 100], labels=["Young", "Middle-Aged", "Old"])


train["Duration_Heart_BodyTemp"] = train["Duration"] * train["Heart_Rate"] * train["Body_Temp"]
test["Duration_Heart_BodyTemp"] = test["Duration"] * test["Heart_Rate"] * test["Body_Temp"]


train["Height_Duration_BodyTemp"] = train["Height"] * train["Duration"] * train["Body_Temp"]
test["Height_Duration_BodyTemp"] = test["Height"] * test["Duration"] * test["Body_Temp"]

train["Weight_Duration_HeartRate"] = train["Weight"] * train["Duration"] * train["Heart_Rate"]
test["Weight_Duration_HeartRate"] = test["Weight"] * test["Duration"] * test["Heart_Rate"]


features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI', 'Effort_Score', 
            'HR_per_min', 'HeartRate_per_BodyTemp', 'HR_age_ratio', 'Weight_per_Height', 'Age_BMI', 
            'BodyTemp_Duration', 'BodyTemp_Deviation', 'HeartRate_Max_HR_Ratio', 'Effort_Score_per_Duration', 
            'Weight_Age', 'Weight_per_BMI','Duration_Heart_BodyTemp','Height_Duration_BodyTemp','Weight_Duration_HeartRate']


scaler = StandardScaler()

train[features] = scaler.fit_transform(train[features])
test[features] = scaler.transform(test[features])


label_cols = ["Exercise_Duration_Category", "Age_Group"]
le = LabelEncoder()

for col in label_cols:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])


train["Sex"] = train["Sex"].map({"male": 1, "female": 0})
test["Sex"] = test["Sex"].map({"male": 1, "female": 0})


train = pd.get_dummies(train, columns=["Sex", "Exercise_Duration_Category", "Age_Group"], drop_first=True)
test = pd.get_dummies(test, columns=["Sex", "Exercise_Duration_Category", "Age_Group"], drop_first=True)


bool_cols = train.select_dtypes("bool").columns

train[bool_cols] = train[bool_cols].astype(int)
test[bool_cols] = test[bool_cols].astype(int)


train.head()


X_train = train.drop(["id", "Calories"], axis=1)
y = train["Calories"]

X_test = test.drop(["id"],axis = 1)


xgb_model = XGBRegressor(subsample = 0.8252835138864835, reg_lambda = 3.501605991491236, reg_alpha= 9.019718559862135, n_estimators= 494, max_depth= 12,
learning_rate= 0.025720450946032463, gamma= 1.8903496577328514, colsample_bytree= 0.5789286073361388,min_child_weight =  7)


xgb_model.fit(X_train,y)
y_pred_xgb = xgb_model.predict(X_test)


submission = pd.DataFrame({
    "id": test["id"],
    "Calories": y_pred_xgb
})

submission.to_csv("submission_xgb_regression.csv", index=False)


hist_model = HistGradientBoostingRegressor(l2_regularization= 0.6576128923003434, learning_rate =0.05783086033354717,max_bins= 213,max_depth = 13,max_iter= 982,min_samples_leaf= 86, random_state=42)


hist_model.fit(X_train,y)
y_pred_hist = hist_model.predict(X_test)


submission = pd.DataFrame({
    "id": test["id"],
    "Calories": y_pred_hist 
})

submission.to_csv("submission_hist_regression.csv", index=False)


ridge_model = Ridge(random_state=42,alpha = 0.1 , max_iter = 200)


ridge_model.fit(X_train,y)
y_pred_ridge = ridge_model.predict(X_test)
y_pred_ridge_clipped = np.clip(y_pred_ridge, 0, None)


submission = pd.DataFrame({
    "id": test["id"],
    "Calories": y_pred_ridge_clipped
})

submission.to_csv("submission_ridge_regression.csv", index=False)


lasso_model = Lasso(random_state=42,alpha = 100 , max_iter = 200)


lasso_model.fit(X_train,y)
y_pred_lasso = lasso_model.predict(X_test)
y_pred_lasso_clipped = np.clip(y_pred_lasso, 0, None)


submission = pd.DataFrame({
    "id": test["id"],
    "Calories": y_pred_lasso_clipped
})

submission.to_csv("submission_lasso_regression.csv", index=False)


mlp_model = MLPRegressor(
    hidden_layer_sizes=(64,32),
    activation="tanh",
    alpha=0.001,
    learning_rate_init=0.0001,
    max_iter=1000,
    random_state=42,
    early_stopping=True
)

mlp_model.fit(X_train, y)

y_pred = mlp_model.predict(X_test)
y_pred_mlp_clipped = np.clip(y_pred, 0, None)



submission = pd.DataFrame({
    "id": test["id"],
    "Calories": y_pred_mlp_clipped
})

submission.to_csv("submission_mlp_regression.csv", index=False)


params = {
    'objective': 'regression',             
    'boosting_type': 'gbdt',                
    'num_leaves': 31,                       
    'max_depth': 12,                      
    'learning_rate': 0.0257,                
    'n_estimators': 494,                    
    'subsample': 0.825,              
    'colsample_bytree': 0.579,             
    'reg_lambda': 3.50,                    
    'reg_alpha': 9.02,                 
    'min_child_weight': 7,                
    'metric': 'mse',                
    'verbosity': -1,                        
    'random_state': 42                 
}

lgb_model = lgb.LGBMRegressor(**params)
lgb_model.fit(X_train, y)


y_pred_lgb = lgb_model.predict(X_test)
y_pred_lgb_clipped = np.clip(y_pred_lgb, 0, None)


submission = pd.DataFrame({
    "id": test["id"],
    "Calories": y_pred_lgb_clipped
})

submission.to_csv("submission_lgb_regression.csv", index=False)


params = {           
    'num_leaves': 1915,                       
    'max_depth': 12,                      
    'learning_rate': 0.038721797842131186,                
    'n_estimators': 553,                    
    'subsample': 0.7107759787306983,              
    'colsample_bytree': 0.7863208714953395,             
    'reg_lambda': 0.9630375961746135,                    
    'reg_alpha': 0.025075625522452082,                 
    'min_child_samples': 97,                              
    'random_state': 42                 
}

lgb_model_optuna = lgb.LGBMRegressor(**params)
lgb_model_optuna.fit(X_train, y)


y_pred_lgb_optuna = lgb_model_optuna.predict(X_test)
y_pred_lgb_optuna_clipped = np.clip(y_pred_lgb_optuna, 0, None)


submission = pd.DataFrame({
    "id": test["id"],
    "Calories": y_pred_lgb_optuna_clipped
})

submission.to_csv("submission_lgb_optuna_regression.csv", index=False)


voting_xgb_light = VotingRegressor(
    estimators = [
        ("xgb", xgb_model),
        ("light" , lgb_model_optuna )
    ]
)

voting_xgb_light.fit(X_train,y)

y_pred_xgb_light = voting_xgb_light.predict(X_test)
y_pred_xgb_light_clipped = np.clip(y_pred_xgb_light, 0, None)


submission = pd.DataFrame({
    "id": test["id"],
    "Calories": y_pred_xgb_light_clipped
})

submission.to_csv("submission_lgb_xgb_regression.csv", index=False)


voting_mlp_light = VotingRegressor(
    estimators = [
        ("mlp", mlp_model),
        ("light" , lgb_model_optuna )
    ]
)

voting_mlp_light.fit(X_train,y)

y_pred_mlp_light = voting_mlp_light.predict(X_test)
y_pred_mlp_light_clipped = np.clip(y_pred_mlp_light, 0, None)


submission = pd.DataFrame({
    "id": test["id"],
    "Calories": y_pred_mlp_light_clipped
})

submission.to_csv("submission_mlp_light_regression.csv", index=False)


voting_ann_xgb = VotingRegressor(
    estimators = [
        ("ann", mlp_model),
        ("xgb" , xgb_model)
    ]
)

voting_ann_xgb.fit(X_train,y)

y_pred_ann_xgb = voting_ann_xgb.predict(X_test)
y_pred_ann_xgb_clipped = np.clip(y_pred_ann_xgb, 0, None)


submission = pd.DataFrame({
    "id": test["id"],
    "Calories": y_pred_ann_xgb_clipped
})

submission.to_csv("submission_xgb_ann_regression.csv", index=False)


voting_lxlm = VotingRegressor(
    estimators = [
        ("lx" , voting_xgb_light),
        ("lm", voting_mlp_light)
    ]
)

voting_lxlm.fit(X_train,y)

y_pred_lxlm = voting_lxlm.predict(X_test)
y_pred_lxlm_clipped = np.clip(y_pred_lxlm, 0, None)


submission = pd.DataFrame({
    "id": test["id"],
    "Calories": y_pred_lxlm_clipped
})

submission.to_csv("submission_lxlm_regression.csv", index=False)


voting_lxmx = VotingRegressor(
    estimators = [
        ("lx" , voting_xgb_light),
        ("mx", voting_ann_xgb)
    ]
)

voting_lxmx.fit(X_train,y)

y_pred_lxmx = voting_lxmx.predict(X_test)
y_pred_lxmx_clipped = np.clip(y_pred_lxmx, 0, None)


submission = pd.DataFrame({
    "id": test["id"],
    "Calories": y_pred_lxmx_clipped
})

submission.to_csv("submission_lxmx_regression.csv", index=False)


voting_lmxm = VotingRegressor(
    estimators = [
        ("lm" , voting_mlp_light),
        ("xm", voting_ann_xgb)
    ]
)

voting_lmxm.fit(X_train,y)

y_pred_lmxm = voting_lmxm.predict(X_test)
y_pred_lmxm_clipped = np.clip(y_pred_lmxm, 0, None)


submission = pd.DataFrame({
    "id": test["id"],
    "Calories": y_pred_lmxm_clipped
})

submission.to_csv("submission_lmxm_regression.csv", index=False)




