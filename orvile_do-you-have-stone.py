!pip install -q LazyPredict 


import numpy as np 
import pandas as pd 
import os

import matplotlib.pyplot as plt
import seaborn as sns
# import plotly.express as px

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

from sklearn.metrics import precision_score, recall_score, f1_score


train = pd.read_csv('/kaggle/input/playground-series-s3e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s3e12/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s3e12/sample_submission.csv')

print('The dimession of the train dataset is:', train.shape)
print('The dimession of the test dataset is:', test.shape)


train.fillna(train.median(), inplace=True)
test.fillna(test.median(), inplace=True)


train.columns


train.head()


train.info()


train.describe()


print(train.isnull().sum())


train.isna().sum()


train['target'].value_counts()


test.head()


test.info()


test.describe()


test.isna().sum()


print(test.isnull().sum())


sns.countplot(x='target', data=train)
plt.title('Count of Target Classes')
plt.show()


# Feature-wise Boxplots for Target Classes
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
sns.boxplot(ax=axes[0, 0], x='target', y='gravity', data=train)
sns.boxplot(ax=axes[0, 1], x='target', y='ph', data=train)
sns.boxplot(ax=axes[0, 2], x='target', y='osmo', data=train)
sns.boxplot(ax=axes[1, 0], x='target', y='cond', data=train)
sns.boxplot(ax=axes[1, 1], x='target', y='urea', data=train)
sns.boxplot(ax=axes[1, 2], x='target', y='calc', data=train)
plt.suptitle("Feature Distributions by Target")
plt.show()


# Feature Distributions
train.drop(columns=['id', 'target']).hist(bins=20, figsize=(14, 10), layout=(3, 3))
plt.suptitle("Feature Distributions in Training Data")
plt.show()


plt.figure(figsize=(15, 5))
ax = plt.subplot()
ax.scatter(train[train['target'] == 1]['ph'], train[train['target'] == 1]['urea'], c='green', s=train[train['target'] == 1]['ph'])
ax.scatter(train[train['target'] == 0]['ph'], train[train['target'] == 0]['urea'], c='red', s=train[train['target'] == 0]['ph']);


# Creating the subplots
fig, axes = plt.subplots(1, 2, figsize=(22, 8))

# The boxplot for gravity
sns.boxplot(ax=axes[0], x='target', y='gravity', hue='target', data=train)
axes[0].set_title('Boxplot of Gravity by Target')
axes[0].set_xlabel('Target')
axes[0].set_ylabel('Gravity')

# The boxplot for ph
sns.boxplot(ax=axes[1], x='target', y='ph', hue='target', data=train)
axes[1].set_title('Boxplot of pH by Target')
axes[1].set_xlabel('Target')
axes[1].set_ylabel('pH')

plt.tight_layout()
plt.show()


# Check for inf values and replace with NaN
train.replace([np.inf, -np.inf], np.nan, inplace=True)
test.replace([np.inf, -np.inf], np.nan, inplace=True)

# 2. Prepare Data (as before)
train_vis = train.drop(columns=['id', 'target'], axis=1).reset_index(drop=True).copy()
test_vis = test.drop(columns=['id'], axis=1).reset_index(drop=True).copy()

train_vis['Dataset'] = 'Train'
test_vis['Dataset'] = 'Test'
data_tot = pd.concat([train_vis, test_vis], axis=0).reset_index(drop=True)

# 3. Improved Plotting (Addressing FutureWarning and better style)
fig, axes = plt.subplots(2, 3, figsize=(25, 15))

features = ['gravity', 'ph', 'osmo', 'cond', 'urea', 'calc']
titles = ['Gravity Distribution', 'pH Distribution', 'Osmolarity Distribution',
          'Conductivity Distribution', 'Urea Distribution', 'Calcium Distribution']

for ax, feature, title in zip(axes.flatten(), features, titles):
    sns.kdeplot(ax=ax, x=feature, hue='Dataset', data=data_tot, fill=True, common_norm=False) # common_norm=False is important
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(feature.capitalize(), fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    sns.despine(ax=ax)  # Adds a clean look (removes spines)

plt.tight_layout()
plt.show()


features = ['gravity', 'ph', 'osmo', 'cond', 'urea', 'calc']  # Exclude 'id' and 'target'

for feature in features:
    # Create a KDE plot 
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=train, x=feature, hue="target", multiple="layer", palette=["green", "red"])
    
    # Adding titles and labels
    plt.xlabel('pH Level')
    plt.ylabel('Density')
    plt.title(f'Density of {feature} Levels by Kidney Stone Presence')
    plt.legend(title='Kidney Stone Presence', labels=['Kidney Stones Present','No Kidney Stones'])
    plt.show();


# Creating subplots
fig, axes = plt.subplots(1, 3, figsize=(25, 7))

# Scatter plot for osmo vs. urea
sns.scatterplot(ax=axes[0], data=train, x='osmo', y='urea', hue='target', palette='coolwarm', edgecolor='k', s=100)
axes[0].set_title('Osmolarity vs Urea')
axes[0].set_xlabel('Osmolarity')
axes[0].set_ylabel('Urea')

# Scatter plot for osmo vs. cond
sns.scatterplot(ax=axes[1], data=train, x='osmo', y='cond', hue='target', palette='viridis', edgecolor='k', s=100)
axes[1].set_title('Osmolarity vs Conductivity')
axes[1].set_xlabel('Osmolarity')
axes[1].set_ylabel('Conductivity')

# Scatter plot for urea vs. gravity
sns.scatterplot(ax=axes[2], data=train, x='urea', y='gravity', hue='target', palette='plasma', edgecolor='k', s=100)
axes[2].set_title('Urea vs Gravity')
axes[2].set_xlabel('Urea')
axes[2].set_ylabel('Gravity')

plt.tight_layout()
plt.show()


# Correlation Analysis
corr = train.drop(columns=['id', 'target']).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()


features = ['gravity', 'ph', 'osmo', 'cond', 'urea', 'calc']  # Exclude 'id' and 'target'

for feature in features:
    plt.figure(figsize=(12, 6))  # Create a new figure for each feature
    for target in train['target'].unique():
        subset = train[train['target'] == target]
        plt.hist(subset[feature], bins=10, alpha=0.5, label=f'target {target}')

    plt.title(f'Distribution of {feature} by Target')  # Dynamic title
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


train_df = train.drop('id',axis=1)


X = train_df.drop('target',axis =1)
y = train_df['target']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, 
                                                  random_state=42, stratify=y)


print("X_train is:",X_train.shape)
print("y_train is:",y_train.shape)
print("X_val is: ",X_val.shape)
print("y_test is: ",y_val.shape)


scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)


test_scaled = scaler.transform(test.drop('id',axis=1))


from lazypredict.Supervised import LazyClassifier


clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)

models, predictions = clf.fit(X_train_scaled, X_val_scaled, y_train, y_val)

print(models)

# train test split içinde stratify yaparsan oran düşüyor!


from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


models = {
    "LogisticRegression": LogisticRegression(),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "ExtraTrees": ExtraTreesClassifier(),
    "HistGradientBoosting": HistGradientBoostingClassifier(),
    "KNeighbors": KNeighborsClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
    "SVC": SVC(probability=True),
    "LGBM": LGBMClassifier(),
    "XGBoost": XGBClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0)
}

def train_and_evaluate(models, X_train, y_train, X_val, y_val):
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None
        
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='binary')
        recall = recall_score(y_val, y_pred, average='binary')
        f1 = f1_score(y_val, y_pred, average='binary')
        roc_auc = roc_auc_score(y_val, y_pred_proba) if y_pred_proba is not None else None
        
        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC AUC": roc_auc
        })
        print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
    
    return pd.DataFrame(results)

# Eğitim ve değerlendirme
results_df = train_and_evaluate(models, X_train, y_train, X_val, y_val)
print(results_df)



# Initialize the classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the validation set
y_val_pred_proba = clf.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_val_pred_proba)
print("Validation ROC AUC Score:", roc_auc)

y_val_pred_pred = clf.predict(X_val)
accuracy_rfc = accuracy_score(y_val, y_val_pred_pred)
print(f'Accuracy: {accuracy_rfc}')


# xgboost
import xgboost as xgb

xgb_classifier = xgb.XGBClassifier(objective='binary:logistic', random_state=42).fit(X_train, y_train)
ypred=xgb_classifier.predict(X_val)
final = np.round(ypred)
print('accuracy of xgboost is:',accuracy_score(final, list(y_val)))


import xgboost as xgb
import optuna
import numpy as np
from sklearn.metrics import accuracy_score

# Optuna ile hiperparametre optimizasyonu
def objective(trial):
    param = {
        'objective': 'binary:logistic',
        'random_state': 42,
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
    }
    
    model = xgb.XGBClassifier(**param)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(np.round(y_pred), y_val)
    return accuracy

# Optuna çalıştırma
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# En iyi parametreler
good_params = study.best_params
print("Best parameters:", good_params)

# En iyi model ile yeniden eğitim
best_model = xgb.XGBClassifier(**good_params)
best_model.fit(X_train, y_train)
y_pred_final = best_model.predict(X_val)
accuracy_final = accuracy_score(np.round(y_pred_final), y_val)
print("Optimized XGBoost accuracy:", accuracy_final)



# En iyi model ile yeniden eğitim
best_model = xgb.XGBClassifier(**good_params)
best_model.fit(X_train, y_train)
y_pred_final = best_model.predict(X_val)
accuracy_final = accuracy_score(np.round(y_pred_final), y_val)
print("Optimized XGBoost accuracy:", accuracy_final)


y_pred_final = best_model.predict(test.drop('id',axis=1))
y_pred_final


submission_df = pd.DataFrame({
    'id': test['id'],
    'target': y_pred_final
})
submission_df


submission_df.to_csv('submission.csv', index=False)

