import pandas as pd
from ydata_profiling import ProfileReport
import numpy as np


df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
profile = ProfileReport(df, title='Predicting Loan Payback')
profile.to_file("my_report.html")


from catboost import CatBoostClassifier, Pool
import pandas as pd
from sklearn.model_selection import train_test_split

# Load your cleaned dataset
df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")

#Feature Engineering
df["high_dti_flag"] = (df["debt_to_income_ratio"] > 0.58).astype(int)

# Define target and features
col_to_drop = ["loan_paid_back", 'id']
X = df.drop(columns=col_to_drop)
y = df['loan_paid_back']

# Identify categorical columns (non-numeric ones)
cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Split into train/validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create CatBoost Pool (handles categorical features efficiently)
train_pool = Pool(X_train, y_train, cat_features=cat_features)
val_pool = Pool(X_val, y_val, cat_features=cat_features)

# Train a quick model
model = CatBoostClassifier(
    iterations=10000,
    learning_rate=0.05,
    depth=6,
    eval_metric='AUC',
    early_stopping_rounds=100,
    random_seed=42,
    verbose=100
)

model.fit(train_pool, eval_set=val_pool, use_best_model=True)

# Evaluate
print("Validation AUC:", model.get_best_score()['validation']['AUC'])

# # Feature importance
# feature_importance = pd.DataFrame({
#     'Feature': model.feature_names_,
#     'Importance': model.get_feature_importance(train_pool)
# }).sort_values(by='Importance', ascending=False)

# print(feature_importance)

# # Optional: visualize
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 6))
# plt.barh(feature_importance['Feature'], feature_importance['Importance'])
# plt.gca().invert_yaxis()
# plt.title("CatBoost Feature Importance")
# plt.show()


import shap

# Initialize SHAP explainer for CatBoost
explainer = shap.TreeExplainer(model)

# Compute SHAP values on validation data
shap_values = explainer.shap_values(X_val)

# Summary plot (global importance + direction)
shap.summary_plot(shap_values, X_val, plot_type="dot")

# Optional: bar chart version (simpler overview)
shap.summary_plot(shap_values, X_val, plot_type="bar")

# Individual prediction explanation (example)
i = 5  # pick any row index
shap.force_plot(explainer.expected_value, shap_values[i], X_val.iloc[i, :])



import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x='loan_paid_back', y='debt_to_income_ratio', data=df)
plt.title('Debt-to-Income Ratio vs Loan Paid Back')
plt.show()


import numpy as np
import pandas as pd

thresholds = np.arange(df['debt_to_income_ratio'].min(), df['debt_to_income_ratio'].max(), 0.01)
ratios = []

for t in thresholds:
    defaults = (df.loc[df['debt_to_income_ratio'] > t, 'loan_paid_back'] == 0).mean()
    ratios.append(defaults)

best_t = thresholds[np.argmax(ratios)]
print("Threshold where default proportion is highest:", best_t)



df['dti_bin'] = pd.qcut(df['debt_to_income_ratio'], 10)
default_rate = df.groupby('dti_bin')['loan_paid_back'].apply(lambda x: (x==0).mean())

default_rate.plot(kind='bar', figsize=(10,5))
plt.title('Default Rate by Debt-to-Income Decile')
plt.ylabel('Default Rate')
plt.show()


test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
test["high_dti_flag"] = (test["debt_to_income_ratio"] > 0.58).astype(int)
X_test = test.drop(['id'], axis=1)
test["prediction"] = model.predict_proba(X_test)[:, 1]

# Save submission file
submission = test[["id", "prediction"]]
submission.to_csv("submission_2.csv", index=False)

print("✅ submission_2.csv created successfully!")
print(submission.head())

