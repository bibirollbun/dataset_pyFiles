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


#Load data
TRAIN_PATH = '/kaggle/input/GiveMeSomeCredit/cs-training.csv'
TEST_PATH  = '/kaggle/input/GiveMeSomeCredit/cs-test.csv'
SAMPLE_PATH = '/kaggle/input/GiveMeSomeCredit/sampleEntry.csv'

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample = pd.read_csv(SAMPLE_PATH)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()



train.columns.tolist()
print("\nTarget distribution (train):")
print(train['SeriousDlqin2yrs'].value_counts(normalize=True))
print("\nTrain numeric summary:")
display(train.describe().T)


train.info()
train.describe()


import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x=train["SeriousDlqin2yrs"])
plt.title("Target Distribution (Default vs No Default)")
plt.show()

train["SeriousDlqin2yrs"].value_counts(normalize=True)


train.isnull().sum()


#Outlier Analysis
numerical_cols = train.columns.drop(['SeriousDlqin2yrs', 'unamed: 0'], errors='ignore')
train[numerical_cols].boxplot(figsize=(14,6))
plt.xticks(rotation=90)
plt.title("Outlier Check")
plt.show()


#Correlation analysis
plt.figure(figsize=(10,7))
sns.heatmap(train.corr(), annot=False)
plt.title("Correlation Heatmap")
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer


#removing id columns
train = train.drop("Unnamed: 0", axis=1, errors='ignore')
test = test.drop("Unnamed: 0", axis=1, errors='ignore')
test = test.drop("SeriousDlqin2yrs", axis=1, errors='ignore')

# Seperate X and y target
X = train.drop('SeriousDlqin2yrs', axis=1)
y = train['SeriousDlqin2yrs']


# Imputing missing values
imputer= SimpleImputer(strategy='median')
X_imputed= imputer.fit_transform(X)
test_imputed= imputer.transform(test)


# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
test_scaled= scaler.transform(test_imputed)


X_train, X_val, y_train, y_val= train_test_split( X_scaled, y, test_size=0.2, random_state=42,stratify=y)


lr = LogisticRegression(class_weight="balanced", max_iter=300)
lr.fit(X_train, y_train)
proba_lr = lr.predict_proba(X_val)[:, 1]

pred_lr = lr.predict(X_val)
print(classification_report(y_val, pred_lr))



rf= RandomForestClassifier(n_estimators=200,max_depth=12, class_weight='balanced', random_state= 12)
rf.fit(X_train, y_train)
predict_rf = rf.predict(X_val)
proba_val = rf.predict_proba(X_val)[:, 1]

print(classification_report(y_val, predict_rf))
print("ROC-AUC Score:", roc_auc_score(y_val, proba_val))




from sklearn.metrics import roc_curve

print("\n" + "="*70)
print("ROC CURVE VISUALIZATION")
print("="*70)

# Calculate ROC curve for both models
fpr_lr, tpr_lr, _ = roc_curve(y_val, proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_val, proba_val)

# Plot
plt.figure(figsize=(10, 7))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_score(y_val, proba_lr):.3f})', 
         linewidth=2, color='blue')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_val, proba_val):.3f})', 
         linewidth=2, color='red')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing', linewidth=1)

plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve Comparison', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()




print("=" * 70)
print("BUSINESS IMPACT ANALYSIS")
print("=" * 70)

# Business costs
avg_loan = 50000
default_loss_rate = 0.70  # Bank loses 70% on default
fp_cost = 500  # Cost of rejecting good customer
fn_cost = avg_loan * default_loss_rate  # $35,000

# Calculate costs from confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_val, predict_rf)
tn, fp, fn, tp = cm.ravel()

print(f"\nConfusion Matrix Breakdown:")
print(f"True Negatives (Good loans correctly approved): {tn:,}")
print(f"False Positives (Good customers incorrectly rejected): {fp:,}")
print(f"False Negatives (Bad loans incorrectly approved): {fn:,}")
print(f"True Positives (Bad loans correctly caught): {tp:,}")

total_cost = (fp * fp_cost) + (fn * fn_cost)
prevented_loss = tp * fn_cost
false_positive_cost = fp * fp_cost
false_negative_cost = fn * fn_cost

print(f"\nBusiness Impact:")
print(f"Cost of False Positives: ${false_positive_cost,}")
print(f"Cost of False Negatives: ${false_negative_cost,}")
print(f"Total Cost: ${total_cost:,}")
print(f"Losses Prevented: ${prevented_loss:,}")
print(f"\nDefaults Caught: {tp}/{tp+fn} ({tp/(tp+fn)*100:.1f}%)")




from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precisions, recalls, thresholds = precision_recall_curve(y_val, proba_val)

# Calculate F1 scores for different thresholds
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"\nOptimal Threshold: {optimal_threshold:.3f}")
print(f"At this threshold:")
print(f"  Precision: {precisions[optimal_idx]:.3f}")
print(f"  Recall: {recalls[optimal_idx]:.3f}")
print(f"  F1-Score: {f1_scores[optimal_idx]:.3f}")

# Plot precision-recall vs threshold
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(thresholds, precisions[:-1], label='Precision')
plt.plot(thresholds, recalls[:-1], label='Recall')
plt.axvline(optimal_threshold, color='red', linestyle='--', label=f'Optimal: {optimal_threshold:.3f}')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall vs Threshold')
plt.legend()
plt.grid(True, alpha=0.3)

# Cost analysis at different thresholds
plt.subplot(1, 2, 2)
costs = []
for threshold in np.arange(0.1, 0.9, 0.01):
    preds_temp = (proba_val >= threshold).astype(int)
    cm_temp = confusion_matrix(y_val, preds_temp)
    tn, fp, fn, tp = cm_temp.ravel()
    cost = (fp * false_positive_cost) + (fn * false_negative_cost)
    costs.append(cost)

plt.plot(np.arange(0.1, 0.9, 0.01), costs)
plt.xlabel('Threshold')
plt.ylabel('Total Cost ($)')
plt.title('Business Cost vs Threshold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



def assign_risk_tier(probability):
    if probability < 0.10:
        return 'Very Low Risk'
    elif probability < 0.25:
        return 'Low Risk'
    elif probability < 0.50:
        return 'Medium Risk'
    elif probability < 0.75:
        return 'High Risk'
    else:
        return 'Very High Risk'

# Apply to validation set
risk_tiers = [assign_risk_tier(p) for p in proba_val]
risk_dist = pd.Series(risk_tiers).value_counts()

print("Risk Distribution in Validation Set:")
print(risk_dist)
print(f"\nTotal: {risk_dist.sum()} applicants")

# Calculate actual default rate per tier
tier_analysis = pd.DataFrame({
    'Predicted_Risk': risk_tiers,
    'Actual_Default': y_val
})

print("\nActual Default Rate by Risk Tier:")
for tier in ['Very Low Risk', 'Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']:
    tier_data = tier_analysis[tier_analysis['Predicted_Risk'] == tier]
    if len(tier_data) > 0:
        actual_default_rate = tier_data['Actual_Default'].mean()
        print(f"{tier}: {actual_default_rate*100:.1f}% default rate ({len(tier_data)} applicants)")


importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values().plot(kind="barh", figsize=(10,7))
plt.title("Feature Importance")
plt.show()



test_preds = rf.predict_proba(test_scaled)[:, 1]





print("\n" + "=" * 70)
print("TOP 5 DEFAULT RISK DRIVERS - BUSINESS INTERPRETATION")
print("=" * 70)

top_5 = importances.nlargest(5)

interpretations = {
    'RevolvingUtilizationOfUnsecuredLines': 
        'Maxing out credit cards signals financial stress and cash flow problems',
    'age': 
        'Younger borrowers typically have less financial stability and credit history',
    'NumberOfTimes90DaysLate': 
        'Past severe delinquency is the strongest predictor - behavior repeats',
    'DebtRatio': 
        'High debt-to-income ratio shows borrower is overextended financially',
    'NumberOfTime30-59DaysPastDueNotWorse': 
        'Recent payment issues indicate current financial difficulties',
    'MonthlyIncome':
        'Lower income increases vulnerability to financial shocks'
}

for i, (feature, importance) in enumerate(top_5.items(), 1):
    print(f"\n{i}. {feature}")
    print(f"   Importance: {importance:.1%}")
    print(f"   â†’ {interpretations.get(feature, 'Significant risk factor')}")




print("\n" + "="*70)
print("RISK SCORE SEGMENTATION")
print("="*70)

def assign_risk_tier(probability):
    if probability < 0.10:
        return 'Very Low Risk'
    elif probability < 0.25:
        return 'Low Risk'
    elif probability < 0.50:
        return 'Medium Risk'
    elif probability < 0.75:
        return 'High Risk'
    else:
        return 'Very High Risk'

# Apply to validation set
risk_tiers = [assign_risk_tier(p) for p in proba_val]
tier_df = pd.DataFrame({
    'Risk_Tier': risk_tiers,
    'Actual_Default': y_val
})

print("\nRisk Distribution:")
print(pd.Series(risk_tiers).value_counts().sort_index())

print("\nActual Default Rate by Tier:")
tier_order = ['Very Low Risk', 'Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
for tier in tier_order:
    tier_data = tier_df[tier_df['Risk_Tier'] == tier]
    if len(tier_data) > 0:
        default_rate = tier_data['Actual_Default'].mean()
        print(f"{tier:18s}: {default_rate*100:5.1f}% default ({len(tier_data):,} applicants)")

print("\nðŸ’¼ Recommended Actions:")
print("â€¢ Very Low Risk (<10%): Auto-approve, best rates")
print("â€¢ Low Risk (10-25%): Approve, standard rates")
print("â€¢ Medium Risk (25-50%): Manual review")
print("â€¢ High Risk (50-75%): Deny or require collateral")
print("â€¢ Very High Risk (>75%): Auto-deny")

# Visualize
colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#8e44ad']
risk_counts = pd.Series(risk_tiers).value_counts().reindex(tier_order)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
risk_counts.plot(kind='bar', color=colors, edgecolor='black')
plt.title('Risk Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Risk Tier')
plt.ylabel('Number of Applicants')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

plt.subplot(1, 2, 2)
default_rates = [tier_df[tier_df['Risk_Tier']==tier]['Actual_Default'].mean()*100 
                 for tier in tier_order if len(tier_df[tier_df['Risk_Tier']==tier]) > 0]
plt.bar(range(len(default_rates)), default_rates, color=colors, edgecolor='black')
plt.title('Actual Default Rate by Tier', fontsize=14, fontweight='bold')
plt.xlabel('Risk Tier')
plt.ylabel('Default Rate (%)')
plt.xticks(range(len(tier_order)), tier_order, rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('risk_segmentation.png', dpi=300, bbox_inches='tight')
plt.show()


submission = pd.DataFrame({
    "Id": test.index,
    "Probability": test_preds
})

# Save file
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)

submission.head(), submission_path

