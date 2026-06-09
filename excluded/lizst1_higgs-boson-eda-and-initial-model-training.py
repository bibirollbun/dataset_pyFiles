import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


os.getcwd()


# Load the data
train = pd.read_csv(r'/kaggle/input/higgs-boson-detection-2025/train.csv')
test = pd.read_csv(r'/kaggle/input/higgs-boson-detection-2025/test.csv')
sample_submission = pd.read_csv(r'/kaggle/input/higgs-boson-detection-2025/sample_submission.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print(train.columns)


X = train.iloc[:,1:] #features
y = train.iloc[:,0] #label

#Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X


from scipy.stats import pointbiserialr

corrs = []
for feature in train.columns[1:]:  # skip the 'Label'
    corr, _ = pointbiserialr(train['label'], train[feature])
    corrs.append((feature, corr))

# Sort by absolute correlation
sorted_corrs = sorted(corrs, key=lambda x: abs(x[1]), reverse=True)

# Plot
import matplotlib.pyplot as plt
import seaborn as sns

features = [x[0] for x in sorted_corrs]
values = [x[1] for x in sorted_corrs]

plt.figure(figsize=(10, 8))
sns.barplot(x=values, y=features, palette='coolwarm')
plt.title("Correlation of Features with Label (Point-Biserial)")
plt.xlabel("Correlation with Label")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()



from sklearn.model_selection import train_test_split
# Separate features and target
X = train.drop('label', axis=1)
y = train['label']

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


from sklearn.ensemble import RandomForestClassifier

# Initialize model
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1,          
    class_weight='balanced'  # Because labels might be a bit imbalanced
)

# Train
rf.fit(X_train, y_train)



from sklearn.metrics import roc_auc_score, accuracy_score

# Predict probabilities and classes
y_pred_proba = rf.predict_proba(X_val)[:, 1]
y_pred = rf.predict(X_val)

# Metrics
print("ROC AUC:", roc_auc_score(y_val, y_pred_proba))
print("Accuracy:", accuracy_score(y_val, y_pred))


from sklearn import metrics

y_pred_proba = rf.predict_proba(X_val)[:, 1]  # Take probability of positive class (usually class '1')
# Compute False Positive Rate, True Positive Rate
fpr, tpr, thresholds = metrics.roc_curve(y_val, y_pred_proba, pos_label=1)

# Compute AUC
auc_score = metrics.auc(fpr, tpr)
print(f"AUC Score: {auc_score:.4f}")

# Plot ROC Curve
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend(loc="lower right")
plt.show()


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Predict probabilities
y_pred_proba = rf.predict_proba(X_val)[:, 1]


threshold = 0.5
y_pred_label = (y_pred_proba >= threshold).astype(int)

# Confusion matrix
cm = confusion_matrix(y_val, y_pred_label)

print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix (Threshold = {threshold})')
plt.show()



#Feature Importance Check
#Feature Importance Check
import pandas as pd
import matplotlib.pyplot as plt

# Feature importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot
plt.figure(figsize=(12, 8))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.show()



from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train, y_train)

y_pred_proba_lr = lr.predict_proba(X_val)[:, 1]
print("LogReg ROC AUC:", roc_auc_score(y_val, y_pred_proba_lr))



import xgboost as xgb

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

y_pred_proba_xgb = xgb_model.predict_proba(X_val)[:, 1]
print("XGBoost ROC AUC:", roc_auc_score(y_val, y_pred_proba_xgb))



import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(class_weight='balanced', random_state=42)
lgb_model.fit(X_train, y_train)

y_pred_proba_lgb = lgb_model.predict_proba(X_val)[:, 1]
print("LightGBM ROC AUC:", roc_auc_score(y_val, y_pred_proba_lgb))



lgb.plot_importance(lgb_model,importance_type='gain',title='LGB Feature Importance Gain')


from sklearn.svm import SVC

svm = SVC(probability=True, class_weight='balanced', random_state=42)
svm.fit(X_train[:5000], y_train[:5000])  # Use subset due to slowness

y_pred_proba_svm = svm.predict_proba(X_val[:5000])[:, 1]
print("SVM ROC AUC (subset):", roc_auc_score(y_val[:5000], y_pred_proba_svm))


