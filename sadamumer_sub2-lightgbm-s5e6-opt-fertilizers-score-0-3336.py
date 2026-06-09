import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import label_ranking_average_precision_score
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


print("\nTrain Shape:", train.shape)
print("Test Shape:", test.shape)


train.head()


print("\nMissing Values in Train:")
print(train.isnull().sum())


sns.countplot(y='Fertilizer Name', data=train)
plt.title('Fertilizer Distribution')
plt.show()


print("\nTarget Distribution:")
print(train['Fertilizer Name'].value_counts())


sns.countplot(y='Soil Type', data=train)
plt.title("Soil Type Distribution")
plt.show()


sns.countplot(y='Crop Type', data=train)
plt.title("Crop Type Distribution")
plt.show()


# Pair Plot for selected features
sns.pairplot(train[['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']])
plt.suptitle("Pair Plot of Numerical Features", y=1.02)
plt.show()


# Box plots to see distributions by target class
for col in ['Temparature', 'Humidity', 'Moisture']:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='Fertilizer Name', y=col, data=train)
    plt.xticks(rotation=90)
    plt.title(f"{col} Distribution by Fertilizer")
    plt.tight_layout()
    plt.show()


numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
sns.heatmap(train[numerical_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Numerical Feature")
plt.show()


cat_cols = ['Soil Type', 'Crop Type']
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# Encode target
target_le = LabelEncoder()
train['Fertilizer Name'] = target_le.fit_transform(train['Fertilizer Name'])

X = train.drop(columns=['id', 'Fertilizer Name'])
y = train['Fertilizer Name']
X_test = test.drop(columns=['id'])


best_params = {
    'learning_rate': 0.1,
    'max_depth': 5,
    'n_estimators': 300,
    'num_leaves': 63,
    'random_state': 42
}

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
final_preds = np.zeros((X_test.shape[0], len(np.unique(y))))
fold_scores = []


for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = LGBMClassifier(**best_params)
    model.fit(X_train, y_train)

    val_preds = model.predict_proba(X_val)
    y_val_binary = np.zeros_like(val_preds)
    for i, label in enumerate(y_val):
        y_val_binary[i, label] = 1

    map3 = label_ranking_average_precision_score(y_val_binary, val_preds)
    fold_scores.append(map3)
    print(f"Fold {fold + 1} MAP@3 Score: {map3:.5f}")

    final_preds += model.predict_proba(X_test) / kf.n_splits


fold_scores


print(f"\nAverage CV MAP@3 Score: {np.mean(fold_scores):.5f}")


top_3_preds = np.argsort(final_preds, axis=1)[:, -3:][:, ::-1]
final_labels = [' '.join(target_le.inverse_transform(row)) for row in top_3_preds]


submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': final_labels
})


submission.head()


submission.to_csv('submission.csv', index=False)
print("\nSubmission file created: submission.csv")

