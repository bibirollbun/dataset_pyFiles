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


df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
#print(df.info())


# --- Grouping feature type ---
num_cat_features = ["num_lanes", "speed_limit", "num_reported_accidents"] #--> numeric categori
num_con_features = ['curvature'] #--> numeric continue
cat_features = ["road_type", "lighting", "weather", "time_of_day"] #--> Categori
bool_features = ["road_signs_present", "public_road", "holiday", "school_season"] #--> boleant
target_features = ['accident_risk'] #--> contine


from sklearn.model_selection import train_test_split

X=df.drop(columns=['id','accident_risk'])
y=df['accident_risk']
id_col = df["id"]


X_train, X_test, y_train, y_test, id_train, id_test= train_test_split(
    X, y, id_col,
    test_size=0.2,      # 20% data untuk test
    random_state=42,    # supaya hasil konsisten
    stratify=y          # optional: menjaga distribusi target
)

#display(X_train.head(5))
#display(y_train.head(5))
#display(id_train.head(5))


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from xgboost import XGBRegressor



preprocessor = ColumnTransformer(
    transformers=[
        ("num_cat", MinMaxScaler(), num_cat_features),
        ("num_con", StandardScaler(), num_con_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("bool", "passthrough", bool_features)
    ]
)

# --- 4. Pipeline dengan model ---
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("xgb", XGBRegressor(
            objective="reg:logistic",
            random_state=42))
])

# --- 5. Train-test split ---
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 6. Fit model ---
model.fit(X_train, y_train)


from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
# --- 7. Evaluasi ---
y_pred = model.predict(X_test)
b_rmse = mean_squared_error(y_test, y_pred, squared=False)
b_r2 = r2_score(y_test, y_pred)

#print(f"RMSE: {b_rmse:.5f}")
#print(f"R²: {b_r2:.5f}")

b_mae = mean_absolute_error(y_test, y_pred)
b_mse = mean_squared_error(y_test, y_pred)
b_rmse = np.sqrt(b_mse)
b_r2 = r2_score(y_test, y_pred)

print(f"MAE  : {b_mae:8.5f}")
print(f"MSE  : {b_mse:8.5f}")
print(f"RMSE : {b_rmse:8.5f}")
print(f"R²   : {b_r2:8.5f}")


from sklearn.model_selection import RandomizedSearchCV


# Hyperparameter grid untuk RandomizedSearchCV
param_distributions = {
    "xgb__n_estimators": [50, 100, 200, 300],
    "xgb__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "xgb__max_depth": [2, 3, 4, 5, 6],
    "xgb__subsample": [0.6, 0.8, 1.0],
    "xgb__colsample_bytree": [0.6, 0.8, 1.0]
}

# RandomizedSearchCV
random_search = RandomizedSearchCV(
    model,
    param_distributions=param_distributions,
    n_iter=10,              # jumlah kombinasi random yang dicoba
    cv=3,                   # 3-fold cross validation
    scoring ='r2',
#    scoring="neg_root_mean_squared_error",           # metric evaluasi
    random_state=42,
    n_jobs=-1               # paralel semua core
)

# Fit RandomizedSearchCV
random_search.fit(X_train, y_train)

print("Best Param    :")#,grid_search.best_params_)
for key, value in random_search.best_params_.items():
          print(f"     {key:<25}: {value}")
print("Best CV Score    :",random_search.best_score_)


print("Best Params:", random_search.best_params_)
print("Best CV Score:", random_search.best_score_)




# Evaluasi di test set
y_opt_pred = random_search.predict(X_test)

mae = mean_absolute_error(y_test, y_opt_pred)
mse = mean_squared_error(y_test, y_opt_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_opt_pred)

print("===============================")
print("        Best Model   Base Model")
print("===============================")
print(f"MAE  : {mae:8.5f}     {b_mae:8.5f}")
print(f"MSE  : {mse:8.5f}     {b_mse:8.5f}")
print(f"RMSE : {rmse:8.5f}     {b_rmse:8.5f}")
print(f"R²   : {r2:8.5f}     {b_r2:8.5f}")
print("===============================")



#load test DF
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

#Split test DF
X_sub=df_sub.drop(columns=['id'])
id_sub = df_sub["id"]

#Pred Submision y
final_pred = random_search.predict(X_sub)
#display(final_pred)


submission = pd.DataFrame({
    "id": id_sub,
    "accident_risk": final_pred   # nama kolom target sesuai kompetisi
})
display(submission.head())

submission.to_csv('submission.csv', index=False)
print("\nSubmission file created: submission.csv")
print(f"Predictions range: [{final_pred.min():.4f}, {final_pred.max():.4f}]")
print(f"Mean prediction: {final_pred.mean():.4f}")

print("\n" + "="*60)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"Expected RMSE: ~{rmse:8.5f}")
print(f"R2 Score     : ~{r2:8.5f}")
print("Submission ready for upload!")

