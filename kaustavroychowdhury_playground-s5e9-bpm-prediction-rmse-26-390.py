import pandas as pd 
import numpy as np 

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor 
from xgboost import XGBRegressor 
from lightgbm import LGBMRegressor

import matplotlib.pyplot as plt 
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")



train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')

test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


train.head()


test.head()


train.info()


train.shape


train.columns


train['BeatsPerMinute'].value_counts()


train.duplicated().sum()


plt.figure(figsize=(10, 8))
sns.heatmap(train.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


# Quick distribution
sns.histplot(train["BeatsPerMinute"], kde=True, bins=30)
plt.title("Distribution of Beats Per Minute (Target)")
plt.show()


X = train.drop(columns=['id', 'BeatsPerMinute'])

y = train['BeatsPerMinute']

X_test_final = test.drop(columns=['id'])


num_columns = X.columns.tolist()


numeric_pipe = Pipeline([
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_columns)
    ],
    remainder="drop"
)





X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)



def evaluate_model(model, X_train, y_train, X_valid, y_valid, name):
    # Train
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_valid)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    mae = mean_absolute_error(y_valid, y_pred)
    r2 = r2_score(y_valid, y_pred)

    print(f"\n--- {name} ---")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE : {mae:.3f}")
    print(f"RÂ²  : {r2:.3f}")

    return {"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2}


dt_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeRegressor(random_state=42))
])


dt_pipe.fit(X_train, y_train)


y_pred = dt_pipe.predict(X_valid)


#Metrics
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
mae = mean_absolute_error(y_valid, y_pred)
r2 = r2_score(y_valid, y_pred)


print(f"RMSE: {rmse:.3f}")
print(f"MAE : {mae:.3f}")
print(f"RÂ²  : {r2:.3f}")


xgb_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(random_state=42))
])


lgbm_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LGBMRegressor(random_state=42))
])


lr_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])


lr_pipe.fit(X_train, y_train)


y_pred = lr_pipe.predict(X_valid)


#Metrics
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
mae = mean_absolute_error(y_valid, y_pred)
r2 = r2_score(y_valid, y_pred)


print(f"RMSE: {rmse:.3f}")
print(f"MAE : {mae:.3f}")
print(f"RÂ²  : {r2:.3f}")


rf_pipe = Pipeline(steps=[
    ('pre', preprocessor),
    ('model', RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42, max_depth= 3))
])


rf_pipe.fit(X_train, y_train)


# ============================
# ðŸ“Š Collect Results
# ============================
models = [
    ("Linear Regression", lr_pipe),
    ("Decision Tree", dt_pipe),
    ("Random Forest", rf_pipe),
    ("XGBoost", xgb_pipe),
    ("LightGBM", lgbm_pipe)
]

results = []
for name, model in models:
    res = evaluate_model(model, X_train, y_train, X_valid, y_valid, name)
    results.append(res)

# Convert results to DataFrame
results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df)



# checking which one RMSE is low

sns.barplot(data=results_df, x="Model", y="RMSE")
plt.title("RMSE Comparison")
plt.show()


#best_model = lr_pipe

#cv_scores = cross_val_score(best_model, X, y, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)


#cv_scores


#best_model.fit(X, y)



#preds = best_model.predict(X_test_final)


best_model = rf_pipe

best_model.fit(X, y)

#cv_scores = cross_val_score(best_model, X, y, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)


preds = best_model.predict(X_test_final)


#submission = pd.DataFrame({
#    "id": test["id"],
#    "BeatsPerMinute": preds
#})
#
#submission.to_csv("submission.csv", index=False)
#print("âœ… submission.csv saved!")


submission = pd.DataFrame({
    "id": test["id"],
    "BeatsPerMinute": preds
})

submission.to_csv("submission.csv", index=False)
print("âœ… submission.csv saved!")




