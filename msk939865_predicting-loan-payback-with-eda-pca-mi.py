


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv', index_col='id')


# Displaying the first 5 rows of the dataset
train.head()


# Checking for missing values
print(train.isna().sum())
print()
print(test.isna().sum())


# Dropping target vector from test dataset
X = train.drop(['loan_paid_back'], axis=1)
y = train['loan_paid_back']


# Printing the columns of 'object' datatype
object_cols = X.select_dtypes(include='object').columns.tolist()
print(f"The object columns are: \n{object_cols}")

# Printing the columns of numerical datatype
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"The numerical columns are: \n{numerical_cols}")


sns.set_style('whitegrid')
sns.countplot(data=train, x='loan_paid_back')
plt.title('Distribution of loan_paid_back')
plt.xlabel('loan_paid_back')
plt.ylabel('Count')
plt.show()


for col in object_cols:
    plt.figure(figsize=(7, 4))
    sns.countplot(x=col, hue='loan_paid_back', data=train)
    plt.title(f'{col} vs loan_paid_back count')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


for col in numerical_cols:
    plt.figure(figsize=(6, 7))
    sns.boxplot(x='loan_paid_back', y=col, data=train)
    plt.title(f'Distribution of {col} by loan_paid_back')
    plt.tight_layout()
    plt.show()


plt.figure(figsize=(7, 6))
corr = X[numerical_cols].corr()

sns.heatmap(corr, annot=True, cmap='viridis_r', fmt=".2f")
plt.title("Correlation of Numeric columns", fontsize=16)
plt.tight_layout()
plt.show()


encoder = LabelEncoder()
for obj in object_cols:
    X[obj] = encoder.fit_transform(X[obj])
    test[obj] = encoder.transform(test[obj])


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)


target_variance = 0.99
pca = PCA(target_variance)
principalComponents = pca.fit(X_scaled)


print(f"The number of components to achieve {target_variance} variance is \
{principalComponents.n_components_}")


plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title(f'PCA: {principalComponents.n_components_} \
Components to Explain {target_variance:.0%} Variance')
plt.axhline(y=target_variance, color='r', linestyle='--')
plt.axvline(x=principalComponents.n_components_, color='g', linestyle='--',
            label=f'n_components={principalComponents.n_components_}')
plt.grid(True)
plt.legend()
plt.show()


pca2d = PCA(n_components=2)

principalComponents2d = pca2d.fit_transform(X_scaled)

principalDf2d = pd.DataFrame(
    data=principalComponents2d, 
    columns=['principalComponent1', 'principalComponent2']
)

# Concatenating the target vector y with 2 principal component values
finalDf2d = pd.concat([principalDf2d, y], axis=1)


fig = plt.figure(figsize = (12, 8))
ax = fig.add_subplot(1, 1, 1) 
ax.set_xlabel('Principal Component 1', fontsize = 10)
ax.set_ylabel('Principal Component 2', fontsize = 10)
ax.set_title('PCA - 2D Projection', fontsize = 20)

targets = [0, 1]
colors = ['#008080', '#FF6F61']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf2d['loan_paid_back'] == target
    ax.scatter(finalDf2d.loc[indicesToKeep, 'principalComponent1'], 
               finalDf2d.loc[indicesToKeep, 'principalComponent2'], 
               c = color, s = 20, label=f'Class {target}')
ax.legend()
ax.grid()


pca3d = PCA(n_components=3)

principalComponents3d = pca3d.fit_transform(X_scaled)

principalDf3d = pd.DataFrame(
    data=principalComponents3d, 
    columns=['principalComponent1', 'principalComponent2', 'principalComponent3']
)

# Concatenating the target vector y with 3 principal component values
finalDf3d = pd.concat([principalDf3d, y], axis=1)


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('Principal Component 1', fontsize=10)
ax.set_ylabel('Principal Component 2', fontsize=10)
ax.set_zlabel('Principal Component 3', fontsize=10)
ax.set_title('PCA - 3D Projection', fontsize=20)

targets = [0, 1]
colors = ['#008080', '#FF6F61']

for target, color in zip(targets, colors):
    indicesToKeep = finalDf3d['loan_paid_back'] == target
    ax.scatter(
        finalDf3d.loc[indicesToKeep, 'principalComponent1'],
        finalDf3d.loc[indicesToKeep, 'principalComponent2'],
        finalDf3d.loc[indicesToKeep, 'principalComponent3'],
        c=color,
        s=20,
        label=f'Class {target}'
    )

ax.legend()
plt.show()


mi_scores = mutual_info_classif(X_scaled, y, random_state=42)


mi_series = pd.Series(mi_scores, index=X.columns)
mi_series = mi_series.sort_values(ascending=True)

print(mi_series)


mi_series.plot(kind='barh', figsize=(9, 4), color='#FF6F61')
plt.title('Mutual Information Scores')
plt.ylabel('MI Score')
plt.xlabel('Features')
plt.tight_layout()
plt.show()


X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
X_scaled_df.head()


test_scaled_df = pd.DataFrame(test_scaled, columns=test.columns)
test_scaled_df.head()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


xgb = XGBClassifier(
    eval_metric='auc',
    verbosity=0,
    random_state=42,
    tree_method='hist',
    n_jobs=-1
)

xgb_params = {
    'n_estimators': randint(300, 1200),
    'max_depth': randint(3, 12),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.5, 0.5),
    'colsample_bytree': uniform(0.5, 0.5),
    'gamma': uniform(0, 1),
    'min_child_weight': randint(1, 10),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0.5, 1.5)
}

xgb_search = RandomizedSearchCV(estimator=xgb, param_distributions=xgb_params, n_iter=60,
    scoring='roc_auc', cv=3, n_jobs=-1, random_state=42, verbose=1)

xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_

print("Best params:", xgb_search.best_params_)


model = best_xgb
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print(f'LogisticRegression Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(f'ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}')
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')


sub = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')
test_preds = model.predict_proba(test)
test_preds_proba = test_preds[:, 1]
submission = pd.DataFrame({
    'id': sub['id'],
    'loan_paid_back': test_preds_proba
})

submission.to_csv('submission.csv', index=False)
print("submission.csv created successfully")




