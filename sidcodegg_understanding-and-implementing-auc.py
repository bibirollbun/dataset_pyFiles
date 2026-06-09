import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np

train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


train.head()


train['rainfall'].value_counts()


train.info()


for col in train.columns:
    if col not in ['rainfall', 'id']:
        plt.figure()
        sns.boxplot(x=train['rainfall'], y=train[col])
        plt.title(f'{col} vs Rainfall')
        plt.show()


selected_features = ['humidity', 'cloud', 'sunshine', 'windspeed']
target = 'rainfall'


X = train[selected_features]
y = train[target]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


print("Training set target distribution:")
print(y_train.value_counts())
print("\nTest set target distribution:")
print(y_test.value_counts())



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(random_state=42, solver='liblinear')
log_model.fit(X_train_scaled, y_train)

y_pred = log_model.predict(X_test_scaled)


coef_dict = dict(zip(selected_features, log_model.coef_[0]))
print("\nModel Coefficients:")
for feature, coef in coef_dict.items():
    print(f"{feature}: {coef:.4f}")



y_test_prob = log_model.predict_proba(X_test_scaled)[:, 1]



print("True Labels (first 5):", y_test[:5])


print("Predicted probabilites (first 5):", y_test_prob[:5])


unique_scores = np.sort(np.unique(y_test_prob))[::-1]
print("First 5 unique thresholds (sorted):", unique_scores[:5])
thresholds = np.concatenate(([np.inf], unique_scores, [-np.inf]))
print("First 5 unique thresholds with inf (sorted):", thresholds[:5])


tpr_list = []
fpr_list = []

for thresh in thresholds:
    y_pred = (y_test_prob >= thresh).astype(int)

    tp = np.sum((y_pred == 1) & (y_test == 1))
    fp = np.sum((y_pred == 1) & (y_test == 0))
    fn = np.sum((y_pred == 0) & (y_test == 1))
    tn = np.sum((y_pred == 0) & (y_test == 0))
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    tpr_list.append(tpr)
    fpr_list.append(fpr)

print("TPR list:", tpr_list[:5])
print("FPR list:", fpr_list[:5])
print('For thresholds:', thresholds[:5])



plt.figure(figsize=(8, 6))
plt.plot(fpr_list, tpr_list, marker='o', label='ROC Curve (from scratch)')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()



auc = np.trapz(tpr_list, fpr_list)
print("AUC", auc)



y_test_prob = log_model.predict_proba(X_test_scaled)[:, 1]

fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
auc_test = roc_auc_score(y_test, y_test_prob)

plt.figure(figsize=(10, 6))
plt.plot(fpr_test, tpr_test, label=f'Test ROC (AUC = {auc_test:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve sklearn')
plt.legend(loc='lower right')
plt.show()



public_submission = pd.read_csv('/kaggle/input/rainfallpublic/submission.csv')
public_submission.to_csv('submission.csv', index=False)


public_submission




