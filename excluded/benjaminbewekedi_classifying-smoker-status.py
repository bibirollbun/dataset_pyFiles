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


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve,auc,f1_score,precision_score,recall_score,ConfusionMatrixDisplay
import lightgbm as lgb
from sklearn.decomposition import PCA


df_submission = pd.read_csv('/kaggle/input/health-signal-analytics-classifying-smoker-status/sample_submission.csv')


df_train = pd.read_csv('/kaggle/input/health-signal-analytics-classifying-smoker-status/train.csv')
df_test = pd.read_csv('/kaggle/input/health-signal-analytics-classifying-smoker-status/test.csv')


df_train.head()


df_test.shape


df_train.head(5)


df_train.keys()


df_train['smoking'].value_counts().plot(kind='pie',autopct='%1.1f')
plt.title('Distribution of the target column: smoking')
plt.grid(True)
plt.show()


df_train.columns


df_train = df_train.drop('id',axis=1)


df_train.info()


scaler = StandardScaler()
df_train_scaled = scaler.fit_transform(df_train)


type(df_train_scaled)


len(df_train.columns)


pca = PCA(n_components=2)
pca.fit(df_train_scaled)


pcs = pca.components_
plt.figure(figsize=(8, 8))
for i, var in enumerate(df_train.columns):
    if i == 22: #index of the target variable
        plt.arrow(0, 0, pcs[0, i], pcs[1, i], head_width=0.05, color='red')
        plt.text(pcs[0, i]*1.15, pcs[1, i]*1.15, var, color='red')
    else:
        plt.arrow(0, 0, pcs[0, i], pcs[1, i], head_width=0.03, color='b')
        plt.text(pcs[0, i]*1.15, pcs[1, i]*1.15, var, color='b')
    

circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
plt.gca().add_artist(circle)
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.axhline(0, color='grey', lw=1)
plt.axvline(0, color='grey', lw=1)
plt.title("Correlations circle (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid()
plt.show()


def get_heatmap_from_df(df:pd.DataFrame,method:str,every_column:bool)->None:
    if not every_column:
        # Select only numeric columns if every_column is False
        df = df.select_dtypes(include=['float64', 'int64'])

    # Calculate the correlation matrix
    corr_matrix = df.corr(method=method)

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(f'Heatmap of Correlation Matrix ({method.capitalize()})')
    plt.show()



get_heatmap_from_df(df_train,method='spearman',every_column=True) #or pearson, kendall


columns_to_select = ['height(cm)','serum creatinine','hemoglobin','Gtp','HDL','smoking']
df_train_selected_features = df_train[columns_to_select].copy()


df_test.keys()


columns_to_select[:-1]


columns_to_select_for_test = columns_to_select[:-1]
df_test_selected_features = df_test[columns_to_select_for_test].copy()


sns.pairplot(df_train_selected_features, vars=columns_to_select, hue="smoking", diag_kind=None)
plt.title('Pairplot of training data with selected features')
plt.show()


features = df_train_selected_features.drop('smoking',axis=1)
target = df_train_selected_features['smoking'].copy()


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=42)


X_train.shape


X_test.shape


type(X_test)


# Création des sous-graphiques
fig, axes = plt.subplots(1,2, figsize=(12, 5))

# Tracer la distribution de y_train
sns.countplot(x=y_train, ax=axes[0])
axes[0].set_title('Distribution of y_train')
axes[0].set_xlabel('Classes')
axes[0].set_ylabel('Frequency')
axes[0].grid(True)
# Tracer la distribution de y_test
sns.countplot(x=y_test, ax=axes[1])
axes[1].set_title('Distribution of y_test')
axes[1].set_xlabel('Classes')
axes[1].set_ylabel('Frequency')
axes[1].grid(True)
plt.tight_layout()
plt.show()


class_weights = {0: 1, 1: len(y_train[y_train == 0]) / len(y_train[y_train == 1])}


class_weights


params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'scale_pos_weight': class_weights[1]  #class weights
}


train_data = lgb.Dataset(X_train, label=y_train)


# Entraînement du modèle
model = lgb.train(params, train_data, num_boost_round=100)

# Prédiction
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
y_pred_classes = [1 if p > 0.5 else 0 for p in y_pred]


def compute_metrics(y_true:np.array,y_pred:np.array)->dict:
    dict_to_return = {
        "f1":f1_score(y_true,y_pred),
        "recall":recall_score(y_true,y_pred),
        "precision":precision_score(y_true,y_pred)
    }
    return dict_to_return

def plot_confusion_matrix(y_true:np.array,y_pred:np.array):
    cm = confusion_matrix(y_true,y_pred)

    cm_display = ConfusionMatrixDisplay(cm, display_labels = ['not smoking', 'smoking'])
    
    cm_display.plot()
    plt.show()


compute_metrics(y_test,y_pred_classes)


plot_confusion_matrix(y_test,y_pred_classes)


def plot_roc_curve(y_true,y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    distances = np.sqrt((1 - tpr) ** 2 + fpr ** 2)

    
    optimal_idx = np.argmin(distances)
    optimal_threshold = thresholds[optimal_idx]
    
    # Tracer la courbe ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc="right")
    plt.text(0.4, -0.2, f'optimal threshold:{round(optimal_threshold,2)}', fontsize=8)
    plt.grid(True)
    plt.show()


plot_roc_curve(y_test,y_pred)


optimal_threshold = 0.57
y_pred_classes = [1 if p > optimal_threshold else 0 for p in y_pred]


compute_metrics(y_test,y_pred_classes)


df_test_selected_features.keys()


df_my_submission = pd.DataFrame()
df_my_submission['id'] = df_test['id'].copy()


submission_preds_prob = model.predict(df_test_selected_features)
y_submission_pred_classes = [1 if p > 0.5 else 0 for p in submission_preds_prob]


df_my_submission['smoking'] = y_submission_pred_classes


df_test['id'].head()


df_my_submission['smoking'].value_counts().plot(kind='barh')


df_my_submission.to_csv('submission.csv',index=False)

