# Standard Libraries ---
import os
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

#Visualization ---
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML

#Scikit-Learn Core & Metrics ---
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc


# Models ---
import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Load Data ---
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")



#see the correlation among numerical features:
target =  "diagnosed_diabetes"
numerical_features = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_features.remove("id")
numerical_features.remove(target)
numerical_features.remove("family_history_diabetes")
numerical_features.remove("hypertension_history")
numerical_features.remove("cardiovascular_history")


correlation_columns = numerical_features +  [target]
matrix = train[correlation_columns].corr()
correlations_with_diabetes = matrix[target].drop(target).sort_values(ascending = False)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# --- LEFT PLOT (correlation of numeric predictors against target)
# Define colors
colors = ['green' if x > 0 else 'red' for x in correlations_with_diabetes.values]
axes[0].barh(correlations_with_diabetes.index, correlations_with_diabetes.values, color=colors)
axes[0].axvline(x=0, color='black', linewidth=1.2)
axes[0].set_title('Correlation of Numeric Predictors with Diabetes')
axes[0].set_xlabel('r')
axes[0].grid(axis='x', linestyle='--', alpha=0.5)

# --- RIGHT PLOT (Entire correlation matrix)
mask = np.triu(np.ones_like(matrix, dtype=bool))
sns.heatmap(
    matrix, 
    annot=True, 
    fmt=".2f", 
    cmap='coolwarm', 
    vmin=-1, vmax=1, 
    mask = mask,
    ax=axes[1]  # <--- Changed to axes[1]
)
axes[1].set_title('Correlation Matrix')
plt.tight_layout()
plt.show()



fig, axes = plt.subplots(5,3, figsize = (10 , 15))
axes = axes.flatten()

for i, col in enumerate(numerical_features):
    sns.kdeplot(data=train, x=col, ax=axes[i], fill=True, label='Train', color='blue', alpha=0.3)
    sns.kdeplot(data=test, x=col, ax=axes[i], fill=True, label='Test', color='orange', alpha=0.3)
    axes[i].set_title(col)
    axes[i].legend()

plt.tight_layout()
plt.show()


#for categorical features lets look at the box plots:
nominal_columns = ["gender", "ethnicity", "employment_status", "smoking_status"]
ordinal_columns = ["education_level", "income_level"]
categorical_features = nominal_columns + ordinal_columns + ["family_history_diabetes" , "hypertension_history" , "cardiovascular_history"]

# 1. Setup the container for side-by-side layout
html_output = '<div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: flex-start;">'

for col in categorical_features:
    # --- Create the Data Table ---
    d1 = train[col].value_counts(normalize=True)
    d2 = test[col].value_counts(normalize=True)
    
    # Transpose so it's compact (Rows=Train/Test, Cols=Categories)
    df = pd.DataFrame({'Train': d1, 'Test': d2}).T.fillna(0)
    df.loc['Diff'] = (df.loc['Train'] - df.loc['Test']).abs()
    
    # --- Style the Table ---
    # Highlights: Percent format, Red gradient on Diff row
    styler = (df * 100).style.format("{:.1f}%")\
        .background_gradient(cmap='Reds', subset=pd.IndexSlice[['Diff'], :], axis=1)\
        .set_table_attributes('style="border-collapse: collapse; font-size: 12px;"')\
        .set_caption(f"<b>{col}</b>") # Bold title inside the table caption

    # --- Add to HTML String ---
    # Convert the styled table to HTML and wrap it in a div
    html_output += f'<div>{styler.to_html()}</div>'

# 2. Close container and Render
html_output += '</div>'
display(HTML(html_output))

#chi square test & cramers' v effect size:
print(f"{'Feature':<20} | {'P-Value':<10} | {'CramÃ©r V (Effect Size)':<9}")
print("-" * 60)

for col in categorical_features:
    contingency_table = pd.crosstab(train[col], train['diagnosed_diabetes'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    n = contingency_table.values.sum()
    r, k = contingency_table.shape
    cramers_v = np.sqrt(chi2 / (n * (min(r, k) - 1)))
    print(f"{col:<20} | {p:.5f}    | {cramers_v:.3f}")




#-------nominal variables (one-hot encoding):
nominal_columns.remove("smoking_status")  #not statistically significant
train = pd.get_dummies(train, columns = nominal_columns, drop_first = True, dtype = int)
test = pd.get_dummies(test, columns = nominal_columns, drop_first = True, dtype = int)

#-------ordinal variables (label-encoding):
#education:
mapping_education = {"No formal":0 , "Highschool":1, "Graduate":2, "Postgraduate":3}
train["education_level"] = train["education_level"].map(mapping_education)
test["education_level"] = test["education_level"].map(mapping_education)

#income-level:
mapping_income = {"Low":0, "Lower-Middle":1, "Middle":2, "Upper-Middle":3, "High":4}
train["income_level"] = train["income_level"].map(mapping_income)
test["income_level"] = test["income_level"].map(mapping_income)


#------------------some feature engineering (you can do much more here):
train["pulse_pressure"] = train["systolic_bp"] - train["diastolic_bp"]
test["pulse_pressure"]  = test["systolic_bp"]  - test["diastolic_bp"]

train["chol_hdl_ratio"] = train["cholesterol_total"] / train["hdl_cholesterol"]
test["chol_hdl_ratio"] = test["cholesterol_total"] / test["hdl_cholesterol"]

train["tg_hdl_ratio"] = train["triglycerides"] / train["hdl_cholesterol"]
test["tg_hdl_ratio"] = test["triglycerides"] / test["hdl_cholesterol"]

def lifestyle_risk(row):
    risk = 0
    if row["physical_activity_minutes_per_week"] < 150:
        risk += 1
    if row["screen_time_hours_per_day"] > 6:
        risk += 1
    if row["sleep_hours_per_day"] < 6:
        risk += 1
    if row["diet_score"] < 5:
        risk += 1
    if row["alcohol_consumption_per_week"]:
        risk += 1
    return risk

train["lifestyle_risk_store"] = train.apply(lifestyle_risk, axis = 1)
test["lifestyle_risk_store"] = test.apply(lifestyle_risk, axis = 1)


drop = ["id", "diagnosed_diabetes", "smoking_status"]
features = [c for c in train.columns if c not in drop]

X = train[features].copy()
y = train[target]

#---------scaling:
scaler = StandardScaler()
cols_to_scale = [
    c for c in features
    if not c.startswith(("gender_", "ethnicity_", "employment_status_"))
    and c not in ["income_level", "education_level", 
                  "hypertension_history", "cardiovascular_history", "family_history_diabetes"]
]

X[cols_to_scale] = X[cols_to_scale].astype(float)
X.loc[:, cols_to_scale] = scaler.fit_transform(X[cols_to_scale])



# ---------------- CV ----------------
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---------------- Parameters ----------------
xgb_params = dict(
    n_estimators=1000, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=1,
    gamma=0.1, objective="binary:logistic", eval_metric="auc",
    random_state=42
)

lgb_params = dict(
    n_estimators=1000, learning_rate=0.03, num_leaves=31,
    subsample=0.7, colsample_bytree=0.7, min_child_samples=20,
    objective="binary", metric="auc", random_state=42,
    n_jobs=-1, verbosity=-1
)

# ---------------- Storage ----------------
xgb_scores, lgb_scores, xgb_imps, lgb_imps = [], [], [], []

print(f"{'Fold':<5} | {'XGB AUC':<10} | {'LGB AUC':<10}")
print("-" * 35)

# ---------------- CV Loop ----------------
for fold, (tr, va) in enumerate(kf.split(X, y)):
    X_t, y_t, X_v, y_v = X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va]
    
    # ----- XGBoost -----
    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(X_t, y_t)
    xgb_scores.append(roc_auc_score(y_v, model_xgb.predict_proba(X_v)[:, 1]))
    xgb_imps.append(pd.DataFrame({
        "Feature": X.columns,
        "Importance": model_xgb.feature_importances_,  # GAIN
        "Model": "XGBoost"
    }))
    
    # ----- LightGBM -----
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(X_t, y_t)
    lgb_scores.append(roc_auc_score(y_v, model_lgb.predict_proba(X_v)[:, 1]))
    lgb_imps.append(pd.DataFrame({
        "Feature": X.columns,
        "Importance": model_lgb.booster_.feature_importance(importance_type="gain"),
        "Model": "LightGBM"
    }))
    
    print(f"{fold+1:<5} | {xgb_scores[-1]:.5f}    | {lgb_scores[-1]:.5f}")

# ---------------- Results ----------------
print("-" * 35)
print(f"Mean XGBoost  : {np.mean(xgb_scores):.5f}")
print(f"Mean LightGBM : {np.mean(lgb_scores):.5f}")

# ---------------- Feature Importance ----------------
df_xgb, df_lgb = pd.concat(xgb_imps), pd.concat(lgb_imps)
order_xgb = df_xgb.groupby("Feature")["Importance"].mean().sort_values(ascending=False).index
order_lgb = df_lgb.groupby("Feature")["Importance"].mean().sort_values(ascending=False).index

fig, axes = plt.subplots(1, 2, figsize=(18, 10))

sns.barplot(data=df_xgb, x="Importance", y="Feature", order=order_xgb,
            color="steelblue", ax=axes[0], errorbar=None)
axes[0].set_title("XGBoost Feature Importance (GAIN)")

sns.barplot(data=df_lgb, x="Importance", y="Feature", order=order_lgb,
            color="darkorange", ax=axes[1], errorbar=None)
axes[1].set_title("LightGBM Feature Importance (GAIN)")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()



#Train on the entire dataset:
X_full = X.copy()
# For test dataset prediction:
X_test_final = test[features].copy()
X_test_final[cols_to_scale] = scaler.transform(X_test_final[cols_to_scale])

#train model on full dataset
print("Training XGBoost on full data...")
model_xgb = xgb.XGBClassifier(**xgb_params)
model_xgb.fit(X_full, y)

print("Training LightGBM on full data...")
model_lgb = lgb.LGBMClassifier(**lgb_params)
model_lgb.fit(X_full, y)

#predict on test dataset
print("Predicting with XGBoost...")
pred_xgb = model_xgb.predict_proba(X_test_final)[:, 1]

print("Predicting with LightGBM...")
pred_lgb = model_lgb.predict_proba(X_test_final)[:, 1]

#Ensemble the two models (We use a 50/50 blend. Use Stacking Technique if needed!)
final_preds = (0.5 * pred_xgb) + (0.5 * pred_lgb)

#Save Submission ---
submission = pd.DataFrame({
    'id': test['id'],
    'prediction': final_preds
})

submission.to_csv('submission.csv', index=False)




