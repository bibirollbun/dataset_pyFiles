import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt

%pip install --quiet git+https://github.com/iseedeep/deeprage.git@main

from deeprage.core import val_bar, val_pie, val_all_hist, compare_columns, RageReport


df_train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv', index_col='id')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv', index_col='id')

display(df_train.head())
display(df_test.head())


df_train.info()


df_test.info()


df_train.describe()


# Visualizing Categorical Distribution: Sex
val_pie(df_train, 'Sex')


compare_columns(df_train)


val_all_hist(df_train, kde=True, freq=True)


# Age buckets(young/mid/senior)
bins = [0, 30, 50, 100]
labels = ['young','mid','senior']
df_train['AgeGroup'] = pd.cut(df_train['Age'], bins=bins, labels=labels)
df_test['AgeGroup']  = pd.cut(df_test['Age'],  bins=bins, labels=labels)

# BMI
df_train['BMI'] = df_train['Weight'] / ( (df_train['Height']/100) ** 2 )
df_test['BMI']  = df_test['Weight']  / ( (df_test['Height']/100) ** 2 )

# HR per minute
df_train['HR_per_min'] = df_train['Heart_Rate'] / df_train['Duration']
df_test['HR_per_min']  = df_test['Heart_Rate'] / df_test['Duration']

# Temp Ã— duration (long hot workouts burn more)
df_train['Temp_Dur'] = df_train['Body_Temp'] * df_train['Duration']
df_test['Temp_Dur']  = df_test['Body_Temp'] * df_test['Duration']

# Sex Ã— intensity interaction
df_train["Sex"] = df_train["Sex"].map({"male": 0, "female": 1})
df_test["Sex"] = df_test["Sex"].map({"male": 0, "female": 1})
df_train['Sex_HRxDur'] = df_train['Sex'] * df_train['HR_per_min']
df_test['Sex_HRxDur']  = df_test['Sex']  * df_test['HR_per_min']


import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import  mean_squared_log_error, mean_squared_error, r2_score

X = pd.get_dummies(df_train.drop(columns='Calories'), drop_first=True)
y = df_train['Calories']
X_test = pd.get_dummies(df_test, drop_first=True).reindex(columns=X.columns, fill_value=0)


# â”€â”€â”€  Define the CV function  â”€â”€â”€â”€â”€â”€â”€â”€
def k_fold_cv(model, X, y, n_splits=5, eval_folds=2):
    """Runs the first `eval_folds` of a KFold, reports metrics, and collects preds/trues."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmsle_list, rmse_list, r2_list, best_iters = [], [], [], []
    all_preds, all_trues = [], []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        if fold >= eval_folds:
            break

        print(f"â�¡ï¸� Fold {fold}")
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model.fit(
            X_tr, np.log1p(y_tr),
            eval_set=[(X_val, np.log1p(y_val))],
            eval_metric='rmse',
            early_stopping_rounds=30,
            verbose=False
        )

        best_iters.append(model.best_iteration or model.n_estimators)

        preds = np.expm1(model.predict(X_val)).clip(0)
        all_preds.extend(preds)
        all_trues.extend(y_val)

        rmsle = np.sqrt(mean_squared_log_error(y_val, preds))
        rmse  = np.sqrt(mean_squared_error(y_val, preds))
        r2    = r2_score(y_val, preds)

        print(f"   ğŸ”¹ RMSLE: {rmsle:.4f}")
        print(f"   ğŸ”¹ RMSE : {rmse:.2f}")
        print(f"   ğŸ”¹ RÂ²   : {r2:.4f}\n")

        rmsle_list.append(rmsle)
        rmse_list.append(rmse)
        r2_list.append(r2)

    print(f"ğŸ�� Mean over {eval_folds} folds:")
    print(f"   â€¢ RMSLE = {np.mean(rmsle_list):.4f}")
    print(f"   â€¢ RMSE  = {np.mean(rmse_list):.2f}")
    print(f"   â€¢ RÂ²    = {np.mean(r2_list):.4f}\n")
    return all_preds, all_trues, best_iters



# â”€â”€â”€ Instantiate XGBRegressor â”€â”€â”€â”€â”€â”€
xgb_model = XGBRegressor(
    tree_method='hist',
    n_estimators=1500,
    max_depth=8,
    learning_rate=0.007,
    subsample=0.9,
    colsample_bytree=0.9,
    objective='reg:squarederror',
    random_state=42,
    verbosity=0
)


# â”€â”€â”€ Run 2-fold CV, get preds â”€â”€â”€â”€â”€â”€â”€â”€â”€
preds, trues, best_iters = k_fold_cv(xgb_model, X, y)

# â”€â”€â”€ Scatter plot of CV predictions vs. actuals â”€â”€â”€â”€â”€â”€â”€
plt.figure(figsize=(6,6))
plt.scatter(trues, preds, alpha=0.3)
lims = [min(trues+preds), max(trues+preds)]
plt.plot(lims, lims, 'k--')
plt.xlabel("Actual Calories")
plt.ylabel("Predicted Calories")
plt.title("CV Fold Predictions vs. Actuals")
plt.grid(False)
plt.show()


# â”€â”€â”€ Retrain final model on ALL data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
avg_iter = int(np.mean(best_iters))
print(f"ğŸ”§ Retraining on full dataset for {avg_iter} rounds\n")

final_model = XGBRegressor(
    tree_method='hist',
    n_estimators=avg_iter,
    max_depth=8,
    learning_rate=0.007,
    subsample=0.9,
    colsample_bytree=0.9,
    objective='reg:squarederror',
    random_state=42,
    verbosity=0
)
final_model.fit(X, np.log1p(y), verbose=False)


# Generating & Formatting Final Predictions
raw_preds = final_model.predict(X_test)
cal_preds = np.expm1(raw_preds).clip(0).round().astype(int)

submission = pd.DataFrame({
    'id':       X_test.index,
    'Calories': cal_preds
})
submission.to_csv('submission.csv', index=False)
print("submission.csv is ready now")

