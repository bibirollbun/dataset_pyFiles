import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm 

df_train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")


# Including both Numerical and Categorical Data
X_num = df_train.select_dtypes(include='number').drop(columns=['diagnosed_diabetes', 'id'])
X_cat = pd.get_dummies(df_train[['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']], 
                                 drop_first=True, dtype=int)
# Combining them 
X_final = pd.concat([X_num, X_cat], axis=1)

X = X_final
Y = df_train['diagnosed_diabetes']


# XGBoost TRAINING MODEL
from xgboost import XGBClassifier

xgb_model = XGBClassifier(n_estimators = 600, # Number of trees that will partake in decisions for the final model
                          learning_rate = 0.05, # 
                          max_depth = 6,
                          tree_method = "hist")

xgb_model.fit(X_final, Y)
print(f"XGBoost Training Score is: {xgb_model.score(X_final, Y):.4f}")


# PLOT TREE
import matplotlib.pyplot as plt
from xgboost import plot_tree

fig, ax = plt.subplots(figsize=(40, 30), dpi=500)
plot_tree(xgb_model, num_trees=0, ax=ax, rankdir='LR')

plt.show()


# FINAL SUBMISSION FOR XGBOOST #1 
df_test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
df_train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")

# New Variables, Feature Engineering  

for df in [df_test, df_train]:
    df['BMI_x_Age'] = df['bmi'] * df['age']
    df['Risk_Count'] = df['hypertension_history'] + df['cardiovascular_history']
    df['Genetic_History'] = df['family_history_diabetes'] * df['bmi']
    df['Activity Age'] = df['physical_activity_minutes_per_week'] * df['age']
    
X_num = df_train.select_dtypes(include='number').drop(columns=['diagnosed_diabetes', 'id'])
X_cat = pd.get_dummies(df_train[['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']], 
                                 drop_first=True, dtype=int)

X_final = pd.concat([X_num, X_cat], axis=1)
Y = df_train['diagnosed_diabetes']

# Test Code

X_num_test = df_test.select_dtypes(include='number').drop(columns=['id'])
X_cat_test = pd.get_dummies(df_test[['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']], 
                            drop_first=True, dtype=int)

X_test = pd.concat([X_num_test, X_cat_test], axis=1)
X_test = X_test.reindex(columns=X_final.columns, fill_value=0)

    
# Model
xgb_model = XGBClassifier(n_estimators = 3200, 
                          learning_rate = 0.01,  
                          max_depth = 5,
                          subsample=0.6,
                          colsample_bytree=0.4,
                          min_child_weight=5,
                          tree_method = "hist", 
                          random_state=22)

xgb_model.fit(X_final, Y)
#print(f"XGBoost Training Score is: {xgb_model.score(X_final, Y):.4f}")

predictions = xgb_model.predict_proba(X_test)[:, 1]
submission = pd.DataFrame({'id': df_test['id'], 'diagnosed_diabetes': predictions})

submission.to_csv('submission.csv', index=False)
print("Submission Successful!")

