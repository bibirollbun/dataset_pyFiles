import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import lightgbm as lgb
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import boxcox
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel
from warnings import filterwarnings
filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


# Display initial data overview
train_df.head().style.background_gradient(cmap='plasma')


test_df.head().style.background_gradient(cmap='plasma')


sub.head().style.background_gradient(cmap='plasma')


print("\nTrain Data Info:")
train_df.info()


print("\nTest Data Info:")
test_df.info()


train_df.describe().style.background_gradient(cmap='tab20c')


test_df.describe().style.background_gradient(cmap='tab20c')


# Handle missing values (using median imputation - robust to outliers)
imputer_num = SimpleImputer(strategy='median')  # Only impute numerical features
numerical_cols = train_df.select_dtypes(include=np.number).columns.tolist()
numerical_cols.remove('id') # exclude id
numerical_cols.remove('rainfall') # exclude target


train_df[numerical_cols] = imputer_num.fit_transform(train_df[numerical_cols])
test_df[numerical_cols] = imputer_num.transform(test_df[numerical_cols]) # use same transformer


# Feature Engineering
train_df['year'] = 2023 # Assuming all data is for the same year -  add a fixed year
test_df['year'] = 2023
# The error was in the format string. '%Y-%j' expects the day of the year to be an integer.
train_df['date'] = pd.to_datetime(train_df['year'].astype(str) + '-' + train_df['day'].astype(int).astype(str), format='%Y-%j')
test_df['date'] = pd.to_datetime(test_df['year'].astype(str) + '-' + test_df['day'].astype(int).astype(str), format='%Y-%j')
train_df['month'] = train_df['date'].dt.month
test_df['month'] = test_df['date'].dt.month
train_df['day_of_week'] = train_df['date'].dt.dayofweek
test_df['day_of_week'] = test_df['date'].dt.dayofweek # Monday=0, Sunday=6
train_df['day_of_year'] = train_df['date'].dt.dayofyear
test_df['day_of_year'] = test_df['date'].dt.dayofyear
train_df.drop(['date', 'year', 'day'], axis=1, inplace=True)  # Remove original day and date feature
test_df.drop(['date', 'year', 'day'], axis=1, inplace=True)



print("\nEngineered Train Data Head:")
train_df.head().style.background_gradient(cmap='YlOrBr')


print("\nEngineered Test Data Head:")
test_df.head().style.background_gradient(cmap='YlOrBr')


# More Feature engineering
train_df['temp_range'] = train_df['maxtemp'] - train_df['mintemp']
test_df['temp_range'] = test_df['maxtemp'] - test_df['mintemp']
train_df['temp_avg'] = (train_df['maxtemp'] + train_df['mintemp']) / 2
test_df['temp_avg'] = (test_df['maxtemp'] + test_df['mintemp']) / 2
train_df['pressure_humidity'] = train_df['pressure'] * train_df['humidity']
test_df['pressure_humidity'] = test_df['pressure'] * test_df['humidity']



# Data Scaling
numerical_cols = train_df.select_dtypes(include=np.number).columns.tolist()
numerical_cols.remove('id') # exclude id
numerical_cols.remove('rainfall') # exclude target


scaler = StandardScaler()
train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])


# ----------------------- Feature Visualization -----------------------
features = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
            'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(16, 20))
fig.suptitle('Rainfall Indicator vs. Weather Features', fontsize=20)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] # Nice color palette

for i, feature in enumerate(features):
    row = i // 2
    col = i % 2
    sns.boxplot(x='rainfall', y=feature, data=train_df, ax=axes[row, col], palette=[colors[0], colors[i%len(colors)]])  #Binary target, boxplot
    axes[row, col].set_title(f'{feature} vs. Rainfall')
    axes[row, col].set_xlabel('Rainfall (0: No, 1: Yes)')
    axes[row, col].set_ylabel(feature)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for the suptitle
plt.show()



# ----------------------- Model Training and Evaluation -----------------------

X = train_df.drop(['id', 'rainfall'], axis=1)
y = train_df['rainfall']
X_test = test_df.drop('id', axis=1)  # Keep test IDs for submission

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



# --- Class Imbalance Handling (SMOTE) ---
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train) # resample the training data

# --- Logistic Regression ---
lr = LogisticRegression(solver='liblinear', random_state=42)
lr.fit(X_train_smote, y_train_smote) #Use resampled data for training LR.
lr_pred_proba = lr.predict_proba(X_val)[:, 1]
lr_roc_auc = roc_auc_score(y_val, lr_pred_proba)
print(f"Logistic Regression ROC AUC: {lr_roc_auc}")

# --- Random Forest ---
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_smote, y_train_smote)  #Train RF on resampled data.
rf_pred_proba = rf.predict_proba(X_val)[:, 1]
rf_roc_auc = roc_auc_score(y_val, rf_pred_proba)
print(f"Random Forest ROC AUC: {rf_roc_auc}")

# --- LightGBM ---
lgbm = lgb.LGBMClassifier(random_state=42)
lgbm.fit(X_train_smote, y_train_smote)  #Train LGBM on resampled data
lgbm_pred_proba = lgbm.predict_proba(X_val)[:, 1]
lgbm_roc_auc = roc_auc_score(y_val, lgbm_pred_proba)
print(f"LightGBM ROC AUC: {lgbm_roc_auc}")

# --- Ensemble (VotingClassifier) ---
# Weight models based on their individual ROC AUC scores
total_auc = lr_roc_auc + rf_roc_auc + lgbm_roc_auc
lr_weight = lr_roc_auc / total_auc
rf_weight = rf_roc_auc / total_auc
lgbm_weight = lgbm_roc_auc / total_auc

voting_clf = VotingClassifier(estimators=[('lr', lr), ('rf', rf), ('lgbm', lgbm)],
                                voting='soft',
                                weights=[lr_weight, rf_weight, lgbm_weight])

voting_clf.fit(X_train_smote, y_train_smote) #Train ensemble on resampled data

ensemble_pred_proba = voting_clf.predict_proba(X_val)[:, 1]
ensemble_roc_auc = roc_auc_score(y_val, ensemble_pred_proba)
print(f"Ensemble ROC AUC: {ensemble_roc_auc}")


# --- Plot ROC Curve for all models ---
plt.figure(figsize=(10, 8))
fpr_lr, tpr_lr, _ = roc_curve(y_val, lr_pred_proba)
fpr_rf, tpr_rf, _ = roc_curve(y_val, rf_pred_proba)
fpr_lgbm, tpr_lgbm, _ = roc_curve(y_val, lgbm_pred_proba)
fpr_ensemble, tpr_ensemble, _ = roc_curve(y_val, ensemble_pred_proba)

plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {lr_roc_auc:.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_roc_auc:.2f})')
plt.plot(fpr_ensemble, tpr_ensemble, label=f'Ensemble (AUC = {ensemble_roc_auc:.2f})', linestyle='--')

plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# --- Make Predictions on Test Data and Create Submission File ---
#Use the ensemble for final prediction

test_pred_proba = voting_clf.predict_proba(X_test)[:, 1]
submission_df = pd.DataFrame({'id': test_df['id'], 'rainfall': test_pred_proba})
submission_df.to_csv('submission.csv', index=False)

# --- Display Submission Head ---
print("\nSubmission File Head:")
submission_df.head()


# --- Feature Importance Visualization (Using Random Forest) ---
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis') #Use viridis color palette
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()




