# Metric: Root Mean Squared Error


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


X = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


X


X.info()


median = X["accident_risk"].median()
mean = X["accident_risk"].mean()
plt.axvline(median, color='red', linestyle='dashed', linewidth=2)
plt.axvline(mean, color='blue', linestyle='dashed', linewidth=2)
sns.kdeplot(X["accident_risk"])


X["lighting"].value_counts()


sns.set_style("whitegrid")
sns.kdeplot(
    data=X,
    x="accident_risk",
    hue="lighting",
    palette={"daylight": "blue", "dim": "purple", "night": "black"},
    common_norm=False,
    fill=True,
    alpha=0.4
)
plt.title("Distribution of accident risk according to lighting")


X["weather"].value_counts()


sns.kdeplot(
    data=X,
    x="accident_risk",
    hue="weather",
    palette={"foggy": "grey", "clear": "orange", "rainy": "blue"},
    fill=True,
    alpha=0.4
)
plt.title("Distribution of accident risk according to weather")


numerical_columns = X.select_dtypes(include = "number").columns


matrix = X[numerical_columns].corr()
sns.heatmap(matrix, annot = True, cmap="coolwarm", fmt = ".2f")


X["num_lanes"].hist()


sns.kdeplot(
    data=X,
    x="accident_risk",
    hue="num_lanes",
    palette={1: "grey", 2: "orange", 3: "blue", 4 :"red"},
    fill=True,
    alpha=0.4
)
plt.title("Distribution of accident risk according to lane number")


from sklearn.preprocessing import LabelEncoder, StandardScaler


X.drop(columns = ["id"],inplace = True)


X["curvature_x_speed_limit"] = X["curvature"] * X["speed_limit"] 
X["curvature**2"] = X["curvature"] ** 2
X["curvature**3"] = X["curvature"] ** 3
X["num_reported_accidents**2"] = X["num_reported_accidents"] ** 2
X["num_reported_accidents**3"] = X["num_reported_accidents"] ** 3



test_data["curvature_x_speed_limit"] = test_data["curvature"] * test_data["speed_limit"] 
test_data["curvature**2"] = test_data["curvature"] ** 2
test_data["curvature**3"] = test_data["curvature"] ** 3
test_data["num_reported_accidents**2"] = test_data["num_reported_accidents"] ** 2
test_data["num_reported_accidents**3"] = test_data["num_reported_accidents"] ** 3


X


# FEATURE ENGINEERING
def f(X):
    return \
    0.3 * X["curvature"] + \
    0.2 * (X["lighting"] == "night").astype(int) + \
    0.1 * (X["weather"] != "clear").astype(int) + \
    0.2 * (X["speed_limit"] >= 60).astype(int) + \
    0.1 * (X["num_reported_accidents"] > 2).astype(int)

def clip(f):
    def clip_f(X):
        sigma = 0.05
        mu = f(X)
        a, b = -mu/sigma, (1-mu)/sigma
        Phi_a, Phi_b = scipy.stats.norm.cdf(a), scipy.stats.norm.cdf(b)
        phi_a, phi_b = scipy.stats.norm.pdf(a), scipy.stats.norm.pdf(b)
        return mu*(Phi_b-Phi_a)+sigma*(phi_a-phi_b)+1-Phi_b
    return clip_f

train = clip(f)(X)
test = clip(f)(test_data)

X['score'] = train
test_data['score']  = test

X


categorical_columns = X.select_dtypes(exclude = "number").columns


label_encoder = {}
for col in categorical_columns: 
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoder[col] = le
X


for col in categorical_columns:
    test_data[col] = label_encoder[col].transform(test_data[col])


X_train, X_test,y_train, y_test = train_test_split(X.drop(columns = ["accident_risk"]),X[["accident_risk"]], test_size = 0.2)


base_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'verbose': -1,
    'random_state': 42,
    'force_col_wise': True  
}


 param_grid = {
    'n_estimators': [1000, 1500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [7, 10],
    'num_leaves': [63, 127],
    'min_child_samples': [50, 100],
}


lgbm = lgb.LGBMRegressor(**base_params)


grid_search = GridSearchCV(
    estimator = lgbm,
    param_grid = param_grid,
    cv = 3,
    scoring = "neg_root_mean_squared_error",
    n_jobs = 1,
    verbose = 2,
    return_train_score=True
)


grid_search.fit(X_train, y_train.values.ravel())


print(f"\nBest parameters:\n{grid_search.best_params_}")
print(f"\nBest CV RMSE: {-grid_search.best_score_:.6f}")


best_lgbmodel = grid_search.best_estimator_


y_pred_train = best_lgbmodel.predict(X_train)
y_pred_test = best_lgbmodel.predict(X_test)


train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print("\n" + "=" * 60)
print("FINAL MODEL PERFORMANCE")
print("=" * 60)
print(f"Train RMSE: {train_rmse:.6f}")
print(f"Test RMSE:  {test_rmse:.6f}")
print(f"Train R²:   {train_r2:.6f}")
print(f"Test R²:    {test_r2:.6f}")


# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': best_lgbmodel.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "=" * 60)
print("TOP 10 FEATURE IMPORTANCES")
print("=" * 60)
print(feature_importance.head(10).to_string(index=False))


test_id = test_data['id']
test_data = test_data.drop(columns=['id'])


pred = best_lgbmodel.predict(test_data)


submission = pd.DataFrame({
        'id': test_id,
        'accident_risk': pred
    })


submission.shape


submission.to_csv("Road_Accident_Risk_submission.csv", index = False)

