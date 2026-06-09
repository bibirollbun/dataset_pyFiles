import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import shap
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')



print(train.info())


# Save ID and drop
test_id = test["id"]
train.drop("id", axis=1, inplace=True)
test.drop("id", axis=1, inplace=True)


X = train.drop("Fertilizer Name", axis=1)
y = train["Fertilizer Name"]



# Encode categorical target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Encode categorical features
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    test[col] = le.transform(test[col])


# Define model
model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.1,
    reg_lambda=1,
    reg_alpha=0,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)


# Stratified K-Fold Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y_encoded, cv=cv, scoring='neg_log_loss')
print("Average Log Loss (5-fold CV):", -np.mean(scores))


# Fit final model
model.fit(X, y_encoded)

# Predict test set probabilities
probs = model.predict_proba(test)

# Top-3 class predictions
top3 = np.argsort(probs, axis=1)[:, -3:][:, ::-1]
top3_labels = label_encoder.inverse_transform(top3.ravel()).reshape(top3.shape)
preds_top3 = [' '.join(row) for row in top3_labels]


plt.figure(figsize=(10, 6))
xgb.plot_importance(model, max_num_features=15, importance_type='gain')
plt.title("Top Features")
plt.tight_layout()
plt.show()


# Prepare submission
submission = pd.DataFrame({'id': test_id, 'Fertilizer Name': preds_top3})
submission.to_csv("submission.csv", index=False)
submission.head(5)


# SHAP Plot
#explainer = shap.TreeExplainer(model)
#shap_values = explainer.shap_values(X)
#shap.summary_plot(shap_values, X)

# Sample 500 rows to make SHAP faster
sampled_X = X.sample(n=500, random_state=42)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sampled_X)

shap.summary_plot(shap_values, sampled_X)


