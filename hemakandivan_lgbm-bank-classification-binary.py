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


#read files

train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
sample_df = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')


train_df.head()



#missing values

print(f'Missing values for training data: {train_df.isnull().sum()}')
print(f'Missing values for test data: {test_df.isnull().sum()}')


#training and test data shapes
print(f'Training data shape: {train_df.shape}')
print(f'Test data shape: {test_df.shape}')


#check duplicates

print(train_df.duplicated().sum())
print(test_df.duplicated().sum())


train_df.info()


train_df.describe()


import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

num_f = train_df.select_dtypes(include=['int64', 'float64']).columns
cat_f = train_df.select_dtypes(include=['object']).columns


sns.set_style('whitegrid')

    
#Numerical features statistics
if len(num_f) > 0:
    n_cols = 3
    n_rows = (len(num_f) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    fig.suptitle('Numerical features analysis', fontsize=16)

    if n_rows == 1:
        axes = [axes] if n_cols else axes
    else:
        axes = axes.flatten()

    for i, col in enumerate(num_f):
        sns.boxplot(x=train_df[col], orient='h', ax=axes[i])
        axes[i].set_title(f'Outliers in {col}', fontsize=12)
        axes[i].set_xlabel(col)

    for i in range(len(num_f), len(axes)):
        axes[i].set_visible(False)
        

    plt.tight_layout()
    plt.show()


#Categorical features statistics
if len(cat_f) > 0:
    n_cols = 2
    n_rows = (len(cat_f) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    fig.suptitle('Categorical features analysis', fontsize=16)

    if n_rows == 1:
        axes = [axes] if n_cols else axes
    else:
        axes = axes.flatten()

    for i, col in enumerate(cat_f):
        sns.countplot(data=train_df, x=col, hue='y', ax=axes[i])
        axes[i].set_title(f'Count of {col}', fontsize=12)
        axes[i].set_xlabel(col)
        axes[i].tick_params(axis='x', rotation=45)

        axes[i].legend(title='Target y', bbox_to_anchor=(1.05,1), loc='upper left')

    for i in range(len(cat_f), len(axes)):
        axes[i].set_visible(False)
        
    plt.tight_layout()
    plt.show()
        


corr = train_df[num_f].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap for Numeric Features')
plt.show()


#find outliers
#
def remove_outliers_per_feature(df, features, std_threshold=3):
    clean_traindf = df.copy()
    
    for feature in features:
        std = df[feature].std()
        mean = df[feature].mean()
        lower_bound = mean - std_threshold * std
        upper_bound = mean + std_threshold * std
        
        clean_traindf = clean_traindf[(clean_traindf[feature] >= lower_bound) & (clean_traindf[feature] <= upper_bound)]
    
    return clean_traindf


clean_traindf = remove_outliers_per_feature(train_df, num_f)
print(f"Original shape: {train_df.shape}")
print(f"After removing outliers: {clean_traindf.shape}")
print(f"Removed {train_df.shape[0] - clean_traindf.shape[0]} rows")



print(num_f.shape, cat_f.shape)


X = clean_traindf.drop(columns=['y'])
y = clean_traindf['y']  


#label encoding for categorical features

X_encoded = pd.get_dummies(X, columns=cat_f)
test_encoded = pd.get_dummies(test_df, columns=cat_f)

#Align columns (handle missing categories)
X_encoded, test_encoded = X_encoded.align(test_encoded, join='left', axis=1, fill_value=0)



#we will use a few classifiers to see which one gives best accuracy
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import recall_score, roc_auc_score, accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split, StratifiedKFold


models = {
'lr': LogisticRegression(max_iter=1000),
'tree': DecisionTreeClassifier(),
'forest': RandomForestClassifier(),
'adab': AdaBoostClassifier(),
'gb': GradientBoostingClassifier(),
'xgb': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
'lgbm': LGBMClassifier()
}


results = {}
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

X = X_encoded.values
if not isinstance(y, np.ndarray):
    y = y.values
    
for name, model in models.items():
    print(f'\n Evaluating {name} with 5-fold CV')
    roc_scores, acc_scores, f1_scores, prec_scores, rec_scores = [], [], [], [], [] 


    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            roc_scores.append(roc_auc_score(y_val, y_pred_proba))
        else:
            roc_scores.append(np.nan)
   

    
        acc_scores.append(accuracy_score(y_val, y_pred))
        prec_scores.append(precision_score(y_val, y_pred, average='weighted'))
        rec_scores.append(recall_score(y_val, y_pred, average='weighted'))
        f1_scores.append(f1_score(y_val, y_pred, average='weighted'))

    results[name] = {

        'Accuracy': np.mean(acc_scores),
        'Precision': np.mean(prec_scores),
        'Recall': np.mean(rec_scores),
        'F1 score': np.mean(f1_scores),
        'ROC AUC': np.nanmean(roc_scores),
        'ROC AUC std': np.nanstd(roc_scores)
    }



    print(f"{name} | ROC-AUC: {results[name]['ROC AUC']:.4f} ± {results[name]['ROC AUC std']:.4f} | F1: {results[name]['F1 score']:.4f} | Acc: {results[name]['Accuracy']:.4f}")


from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier, early_stopping
from sklearn.preprocessing import StandardScaler

lgbm = LGBMClassifier(
    objective='binary',
    random_state=42,
    n_jobs=-1,
    scale_pos_weight = (401621/46054)
)


param_dist = {
    'n_estimators': [200, 400],
    'max_depth': [5, 7, -1],  
    'learning_rate': [0.05, 0.1],
    'num_leaves': [31, 63, 127],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}


random_search = RandomizedSearchCV(
    estimator=lgbm,
    param_distributions=param_dist,
    n_iter=10,
    scoring='roc_auc',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)


X_scaled = X_encoded
test_scaled = test_encoded


random_search.fit(
    X_scaled, y,
    eval_set=[(X_scaled, y)],
    callbacks=[early_stopping(stopping_rounds=10)])



print("\nBest Parameters:", random_search.best_params_)
print("Best ROC-AUC (CV):", random_search.best_score_)


best_lgbm = random_search.best_estimator_

best_lgbm.fit(X_scaled, y)


final_predictions = best_lgbm.predict(test_scaled)
print("\nFinal predictions ready")



submission = pd.DataFrame({
    'id':test_df['id'],
    'y': final_predictions
})
submission.head()


submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Saved!")
print(os.listdir("/kaggle/working"))


submission.head()




