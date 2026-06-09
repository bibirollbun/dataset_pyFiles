!pip install -U autogluon > /dev/null

from autogluon.tabular import TabularDataset, TabularPredictor
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from catboost import CatBoostClassifier, Pool
from catboost.utils import eval_metric
from imblearn.over_sampling import SMOTENC
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# USE_DATA_LEAK='Y' 
##
RAND_VAL=27
num_folds=10 ## Number of folds
n_est=10000 ## Number of estimators


df_train = pd.read_csv('/kaggle/input/playground-series-s4e1/train.csv')
print(df_train.columns)
df_train.head()


df_test = pd.read_csv('/kaggle/input/playground-series-s4e1/test.csv')
df_test_ov = df_test.copy()
df_test.head()


df_test_val = pd.read_csv('/kaggle/input/newewew/train.csv')
df_test_val.head()


scale_cols = ['Age','CreditScore', 'Balance','EstimatedSalary']

for c in scale_cols:
    min_value = df_train[c].min()
    max_value = df_train[c].max()
    df_train[c+"_scaled"] = (df_train[c] - min_value) / (max_value - min_value)
    df_test[c+"_scaled"] = (df_test[c] - min_value) / (max_value - min_value)
    df_test_val[c+"_scaled"] = (df_test_val[c] - min_value) / (max_value - min_value)


df_train.head()


# Define features and target
X = df_train.drop(['id', 'CustomerId', 'Surname', 'Exited'], axis=1, errors='ignore')
y = df_train['Exited']


cat_cols = X.select_dtypes(['object', 'category']).columns
cat_idx  = [X.columns.get_loc(c) for c in cat_cols]
print("\nCategorical columns:", cat_cols)


# Plot distribution before SMOTENC
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(x=y, palette='Set2')
plt.title('Distribution of Exited (Before SMOTENC)')
plt.xlabel('Exited')
plt.ylabel('Count')


sm = SMOTENC(categorical_features=cat_idx, sampling_strategy=0.667, random_state=42)
X_smotenc, y_smotenc = sm.fit_resample(X, y)

# Convert resampled data back to DataFrame
df_train_smotenc = pd.DataFrame(X_smotenc, columns=X.columns)
df_train_smotenc['Exited'] = y_smotenc

# Plot distribution after SMOTENC
plt.subplot(1, 2, 2)
sns.countplot(x=y_smotenc, palette='Set2')
plt.title('Distribution of Exited (After SMOTENC, 6:4 Ratio)')
plt.xlabel('Exited')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


# Print class distribution
print("\nClass distribution before SMOTENC:")
print(y.value_counts())
print("\nClass distribution after SMOTENC:")
print(pd.Series(y_smotenc).value_counts())


# Restore original columns needed for feature engineering
for col in ['id', 'CustomerId', 'Surname']:
    if col in df_train.columns:
        df_train_smotenc[col] = df_train[col].iloc[:len(df_train_smotenc)].values


def getFeats(df):
    
    df['IsSenior'] = df['Age'].apply(lambda x: 1 if x >= 60 else 0)
    df['IsActive_by_CreditCard'] = df['HasCrCard'] * df['IsActiveMember']
    df['Products_Per_Tenure'] =  df['Tenure'] / df['NumOfProducts']
    df['AgeCat'] = np.round(df.Age/20).astype('int').astype('category')
    df['Sur_Geo_Gend_Sal'] = df['Surname']+df['Geography']+df['Gender']+np.round(df.EstimatedSalary).astype('str')

    
    return df


df_train = getFeats(df_train)
df_test = getFeats(df_test)
df_test_val = getFeats(df_test_val)
df_train_smotenc = getFeats(df_train_smotenc) 

feat_cols=df_train.columns.drop(['id','CustomerId', 'Surname','Exited'])
feat_cols=feat_cols.drop(scale_cols)
print(feat_cols)
df_train.head()


X=df_train[feat_cols]
y=df_train['Exited']

X_smotenc = df_train_smotenc[feat_cols]
y_smotenc = df_train_smotenc['Exited']

cat_features = np.where(X.dtypes != np.float64)[0]
cat_features


# # Step 1: Apply RandomUnderSampler to reduce majority class
# undersampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)  # ~2:1 ratio
# X_under, y_under = undersampler.fit_resample(X, y)
# print("\nClass distribution after undersampling:")
# print(pd.Series(y_under).value_counts())


# # Step 2: Apply SMOTENC with 0.2 ratio on undersampled data
# smotenc = SMOTENC(categorical_features=categorical_indices, sampling_strategy=0.7, random_state=42)
# X_smotenc, y_smotenc = smotenc.fit_resample(X_under, y_under)

# # Plot distribution after undersampling + SMOTENC
# plt.subplot(1, 2, 2)
# sns.countplot(x=y_smotenc, palette='Set2')
# plt.title('Distribution of Exited (After Undersampling + SMOTENC)')
# plt.xlabel('Exited')
# plt.ylabel('Count')
# plt.tight_layout()
# plt.show()


# # Step 3: Apply SMOTENC with 6:4 ratio
# # smotenc = SMOTENC(categorical_features=categorical_indices, sampling_strategy=0.4, random_state=42)
# # X_smotenc, y_smotenc = smotenc.fit_resample(X, y)

# sm = SMOTENC(categorical_features=cat_idx,
#              sampling_strategy=0.667,
#              random_state=42)

# X_smotenc, y_smotenc = sm.fit_resample(X, y)

# # Plot distribution after SMOTENC
# plt.subplot(1, 2, 2)
# sns.countplot(x=y_smotenc, palette='Set2')
# plt.title('Distribution of Exited (After SMOTENC, 6:4 Ratio)')
# plt.xlabel('Exited')
# plt.ylabel('Count')
# plt.tight_layout()
# plt.show()


# # Khởi tạo SMOTENC bên trong SMOTETomek
# # Đặt smote_sampling_strategy=1.0 để đạt được tỷ lệ gần 1:1
# smote_nc_tomek = SMOTETomek(
#     smote=SMOTENC(categorical_features=categorical_indices, random_state=42),
#     random_state=42,
#     sampling_strategy='auto' # Hoặc một số cụ thể nếu bạn muốn kiểm soát chi tiết hơn
# )

# print("\n--- Áp dụng SMOTETomek với SMOTENC ---")
# # Áp dụng lên dữ liệu gốc của bạn
# X_smotenc, y_smotenc = smote_nc_tomek.fit_resample(X, y)

# print("\nPhân bố lớp sau khi áp dụng SMOTETomek (với SMOTENC):")
# print(Counter(y_smotenc))

# # Vẽ biểu đồ phân bố
# plt.figure(figsize=(10, 5))
# sns.countplot(x=y_smotenc, palette='viridis')
# plt.title('Phân bố lớp sau SMOTETomek (với SMOTENC)')
# plt.xlabel('Exited')
# plt.ylabel('Số lượng')
# plt.show()


# # Print class distribution
# print("\nClass distribution before SMOTENC:")
# print(y.value_counts())
# print("\nClass distribution after SMOTENC:")
# print(pd.Series(y_smotenc).value_counts())


folds = StratifiedKFold(n_splits=num_folds,random_state=RAND_VAL,shuffle=True)
test_preds = np.empty((num_folds, len(df_test)))
auc_vals=[]

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_smotenc, y_smotenc)):
    
    X_train, y_train = X_smotenc.iloc[train_idx], y_smotenc.iloc[train_idx]
    X_val, y_val = X_smotenc.iloc[valid_idx], y_smotenc.iloc[valid_idx]
    
    train_pool = Pool(X_train, y_train,cat_features=cat_features)
    val_pool = Pool(X_val, y_val,cat_features=cat_features)
    
    clf = CatBoostClassifier(
    eval_metric='AUC',
    task_type='GPU',
    learning_rate=0.01,
    iterations=n_est)
    clf.fit(train_pool, eval_set=val_pool,verbose=300)
    
    y_pred_val = clf.predict_proba(X_val[feat_cols])[:,1]
    auc_val = roc_auc_score(y_val, y_pred_val)
    print("AUC for fold ",n_fold,": ",auc_val)
    auc_vals.append(auc_val)
    
    y_pred_test = clf.predict_proba(df_test[feat_cols])[:,1]
    test_preds[n_fold, :] = y_pred_test
    print("----------------")


"Mean AUC: ",np.mean(auc_vals)


import shap
shap.initjs()
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(train_pool)
shap.summary_plot(shap_values, X_train, plot_type="bar")


catboost_predictions = test_preds.mean(axis=0)
df_sub = pd.DataFrame({'id': df_test['id'], 'Exited': catboost_predictions})


from autogluon.tabular import TabularPredictor

# Prepare data for AutoGluon
df_train_AG = df_train.copy()
df_train_AG.pop('id')
df_train_AG.head(3)

df_test_AG = df_test.copy()
df_test_AG.pop('id')
df_test_AG.head(3)

df_test_val_AG = df_test_val.copy()
df_test_val_AG.pop('id')


# Train AutoGluon model
automl = TabularPredictor(
    label='Exited',
    problem_type='binary',
    eval_metric='roc_auc')
automl.fit(df_train_AG, presets='best_quality')


# View AutoGluon leaderboard
automl.leaderboard()


# Get predictions from AutoGluon
predictions = automl.predict_proba(df_test_AG)
predictions0 = predictions[1]  # AutoGluon's Best Model prediction

# Print best model name
best_model_name = automl.leaderboard().iloc[0]['model']
print(f"AutoGluon Best Model: {best_model_name}")


# Ensemble: Simple average of CatBoost and AutoGluon predictions
ensemble_predictions = (predictions0 + catboost_predictions) / 2

# Create submission file
output_sample = pd.read_csv('/kaggle/input/playground-series-s4e1/sample_submission.csv')
output = pd.DataFrame({'id': output_sample.id, 'Exited': ensemble_predictions})
output.to_csv('submission.csv', index=False, sep=',')


# Visualize final predictions
output.hist(column='Exited', bins=20, range=[0,1], figsize=(12,6))
plt.title('Distribution of Ensemble Predictions')
plt.show()


# Step 7: Predict on df_test_val with CatBoost
print("\nPredicting on df_test_val with CatBoost...")
test_val_pool = Pool(df_test_val[feat_cols], cat_features=cat_features)
y_pred_test_val_catboost = clf.predict(test_val_pool)
y_pred_test_val_proba_catboost = clf.predict_proba(test_val_pool)[:, 1]

# Step 8: Predict on df_test_val with AutoGluon
print("\nPredicting on df_test_val with AutoGluon...")
y_pred_test_val_autogluon = automl.predict(df_test_val_AG)
y_pred_test_val_proba_autogluon = automl.predict_proba(df_test_val_AG)[1]

# Step 9: Ensemble predictions
print("\nCreating ensemble predictions (CatBoost + AutoGluon) / 2...")
y_pred_test_val_proba_ensemble = (y_pred_test_val_proba_catboost + y_pred_test_val_proba_autogluon) / 2
y_pred_test_val_ensemble = (y_pred_test_val_proba_ensemble > 0.5).astype(int)


# Confusion Matrix for Ensemble
cm_ensemble = confusion_matrix(df_test_val['Exited'], y_pred_test_val_ensemble, labels=[0, 1])
print("\nConfusion Matrix on df_test_val (Ensemble):")
print(cm_ensemble)

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_ensemble, display_labels=['Not Exited (0)', 'Exited (1)'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix on df_test_val (Ensemble)')
plt.show()

# Classification Report for Ensemble
print("\nClassification Report on df_test_val (Ensemble):")
print(classification_report(df_test_val['Exited'], y_pred_test_val_ensemble, target_names=['Not Exited (0)', 'Exited (1)']))

# ROC Curve for Ensemble
fpr, tpr, _ = roc_curve(df_test_val['Exited'], y_pred_test_val_proba_ensemble)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve on df_test_val (Ensemble)')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Precision-Recall Curve for Ensemble
precision, recall, _ = precision_recall_curve(df_test_val['Exited'], y_pred_test_val_proba_ensemble)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='purple', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve on df_test_val (Ensemble)')
plt.legend(loc="lower left")
plt.grid(True)
plt.show()


# Step 10: Compare individual models (optional)
# CatBoost Confusion Matrix
cm_catboost = confusion_matrix(df_test_val['Exited'], y_pred_test_val_catboost, labels=[0, 1])
print("\nConfusion Matrix on df_test_val (CatBoost):")
print(cm_catboost)

# AutoGluon Confusion Matrix
cm_autogluon = confusion_matrix(df_test_val['Exited'], y_pred_test_val_autogluon, labels=[0, 1])
print("\nConfusion Matrix on df_test_val (AutoGluon):")
print(cm_autogluon)

# ROC AUC for individual models
auc_catboost = roc_auc_score(df_test_val['Exited'], y_pred_test_val_proba_catboost)
auc_autogluon = roc_auc_score(df_test_val['Exited'], y_pred_test_val_proba_autogluon)
auc_ensemble = roc_auc_score(df_test_val['Exited'], y_pred_test_val_proba_ensemble)
print(f"\nROC AUC Scores:")
print(f"CatBoost: {auc_catboost:.4f}")
print(f"AutoGluon: {auc_autogluon:.4f}")
print(f"Ensemble: {auc_ensemble:.4f}")

