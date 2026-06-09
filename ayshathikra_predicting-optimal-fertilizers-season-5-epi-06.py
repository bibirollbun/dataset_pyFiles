import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


print("Dataset shape:", train_df.shape)
print("\nFirst 5 rows of the dataset")
print("------------------------------")
print(train_df.head())

print("\n\nData types")
print("-------------")
print(train_df.dtypes)

print("\n\nMissing values")
print("----------------")
print(train_df.isnull().sum())


print("\nSummary statistics")
print("---------------------")
print(train_df.describe())


print("\nUnique Crop Types")
print("-------------------")
print(train_df['Crop Type'].unique())

print("\n\nUnique Soil Types")
print("--------------------")
print(train_df['Soil Type'].unique())



fertilizer_names = sorted(train_df['Fertilizer Name'].unique())

print(" All the Unique Fertilizer Names")
print("---------------------------------")
for name in fertilizer_names:
    print(f"  - {name}")


fert_counts = train_df['Fertilizer Name'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(fert_counts, labels=fert_counts.index, autopct='%1.1f%%', 
        startangle=140)
plt.title('Fertilizer Type Distribution (Pie Chart)\n\n')
plt.axis('equal')
plt.tight_layout()
plt.show()



numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols = [col for col in numerical_cols if col not in ['id', 'target']]

plt.figure(figsize=(15, 12))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 3, i)
    sns.histplot(train_df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()

plt.suptitle('Distribution of Numerical Features', fontsize=16, y=1.02)
plt.show()



plt.figure(figsize=(8, 5))
sns.countplot(data=train_df, x='Fertilizer Name', order=train_df['Fertilizer Name'].value_counts().index)
plt.title('Fertilizer Class Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))
sns.countplot(data=train_df, x='Crop Type', order=train_df['Crop Type'].value_counts().index)
plt.title('Crop Type Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(7, 4))
sns.countplot(data=train_df, x='Soil Type', order=train_df['Soil Type'].value_counts().index)
plt.title('Soil Type Distribution')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(data=train_df, x='Fertilizer Name', y='Moisture')
plt.title('Moisture by Fertilizer')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


scatter_pairs = [
    ('Temparature', 'Moisture'),
    ('Humidity', 'Moisture'),
    ('Nitrogen', 'Phosphorous'),
    ('Potassium', 'Phosphorous'),
    ('Temparature', 'Humidity'),
    ('Nitrogen', 'Potassium'),
]

plt.figure(figsize=(15, 15))
for i, (x, y) in enumerate(scatter_pairs, 1):
    plt.subplot(3, 2, i)
    sns.scatterplot(data=train_df, x=x, y=y, hue='Fertilizer Name', alpha=0.6, palette='Set2')
    plt.title(f'{x} vs {y}', fontsize=12)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()

plt.suptitle('Scatter Plots of Key Feature Pairs', fontsize=16, y=1.02)
plt.show()



sample_df = train_df.sample(3000, random_state=42)

sns.pairplot(sample_df, vars=['Nitrogen', 'Phosphorous', 'Potassium', 'Moisture'],
             hue='Fertilizer Name', palette='tab10', diag_kind='kde')
plt.suptitle('Pairwise Feature Distributions by Fertilizer', y=1.02)
plt.show()



plt.figure(figsize=(10, 6))
sns.boxplot(data=train_df, x='Fertilizer Name', y='Nitrogen')
plt.title('Nitrogen by Fertilizer')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
sns.countplot(data=train_df, x='Soil Type', hue='Fertilizer Name')
plt.title('Fertilizer by Soil Type')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


train_df['Crop_Code'] = train_df['Crop Type'].astype('category').cat.codes
train_df['Soil_Code'] = train_df['Soil Type'].astype('category').cat.codes
corr = train_df[['Temparature', 'Humidity','Moisture','Soil_Code','Crop_Code','Nitrogen','Potassium','Phosphorous']].corr()

plt.figure(figsize=(20, 20))
sns.heatmap(corr, cmap='cividis', annot=True, annot_kws={"size": 16, "weight": "bold"})
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.title("\n\nCorrelation Heatmap\n\n", fontsize=18, fontweight='bold')
plt.show()


le_target = LabelEncoder()
train_df['target'] = le_target.fit_transform(train_df['Fertilizer Name'])

for df in [train_df, test_df]:
    df['Crop_Code'] = df['Crop Type'].astype('category').cat.codes
    df['Soil_Code'] = df['Soil Type'].astype('category').cat.codes

features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous', 'Crop_Code', 'Soil_Code']
X = train_df[features]
X_test = test_df[features]
y = train_df['target']


def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        if a in p[:k]:
            return 1.0 / (p[:k].index(a) + 1)
        return 0.0
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(train_df), len(le_target.classes_)))
test_preds = np.zeros((len(test_df), len(le_target.classes_)))
models = []
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nTraining fold {fold + 1}...")
    X_train, X_val = X.loc[train_idx], X.loc[val_idx]
    y_train, y_val = y.loc[train_idx], y.loc[val_idx]
    model = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                         use_label_encoder=False, eval_metric='mlogloss', random_state=42+fold)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)
    models.append(model)
    val_preds = model.predict_proba(X_val)
    oof_preds[val_idx] = val_preds
    test_preds += model.predict_proba(X_test) / skf.n_splits
    top3 = np.argsort(val_preds, axis=1)[:, -3:][:, ::-1]
    score = mapk(y_val.tolist(), top3.tolist(), k=3)
    fold_scores.append(score)
    print(f"Fold {fold + 1} MAP@3: {score:.5f}")

best_fold = np.argmax(fold_scores)
print(f"\n\nBest fold: {best_fold + 1} with MAP@3 = {fold_scores[best_fold]:.5f}")
print(f"\nAverage MAP@3: {np.mean(fold_scores):.5f}")


from sklearn.metrics import (
    log_loss, roc_auc_score, matthews_corrcoef,
    precision_score, recall_score, f1_score
)
import numpy as np

y_pred_labels = np.argmax(oof_preds, axis=1)

logloss = log_loss(y, oof_preds)
print(f"Log Loss: {logloss:.5f}")

try:
    y_onehot = np.eye(len(le_target.classes_))[y]
    roc_auc = roc_auc_score(y_onehot, oof_preds, multi_class='ovr')
    print(f"ROC-AUC Score (OvR): {roc_auc:.5f}")
except:
    print("ROC-AUC not supported due to missing class probabilities.")

mcc = matthews_corrcoef(y, y_pred_labels)
print(f"MCC Score: {mcc:.5f}")

precision = precision_score(y, y_pred_labels, average='macro')
recall = recall_score(y, y_pred_labels, average='macro')
f1 = f1_score(y, y_pred_labels, average='macro')

print(f"Precision (Macro): {precision:.5f}")
print(f"Recall (Macro): {recall:.5f}")
print(f"F1 Score (Macro): {f1:.5f}")



plt.figure(figsize=(8, 4))
plt.plot(range(1, 6), fold_scores, marker='o', label='Fold MAP@3')
plt.axhline(np.mean(fold_scores), linestyle='--', color='gray', label='Average')
plt.title('MAP@3 Score Per Fold')
plt.xlabel('Fold')
plt.ylabel('MAP@3')
plt.legend()
plt.grid(True)
plt.show()


y_pred_labels = np.argmax(oof_preds, axis=1)
cm = confusion_matrix(y, y_pred_labels)
class_names = le_target.classes_
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(xticks_rotation=45, cmap='Blues')
plt.title("Confusion Matrix - OOF Predictions")
plt.tight_layout()
plt.show()


top3_test_preds = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]
top3_labels = le_target.inverse_transform(top3_test_preds.ravel()).reshape(top3_test_preds.shape)
submission = pd.DataFrame({
    'id': test_df['id'],
    'Fertilizer Name': [' '.join(row) for row in top3_labels]
})
submission.to_csv('submission.csv', index=False)
print("✅ Submission file created successfully: submission.csv")


sub_check = submission.copy()
sub_check['Fertilizer Count'] = sub_check['Fertilizer Name'].apply(lambda x: len(set(x.split())))
repeated = sub_check[sub_check['Fertilizer Count'] < 3]
print(f"Rows with duplicate fertilizers: {len(repeated)}")


print("First 5 rows of submission:")
print(pd.read_csv('submission.csv').head())


print("Total rows:", len(submission))
print("Unique IDs:", submission['id'].nunique())


print("Random sample from submission:")
print(pd.read_csv('submission.csv').sample(5))


import joblib
joblib.dump(model, 'fertilizer_xgb_model.pkl')

