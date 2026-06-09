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


# ğŸ“¦ 1. ë�¼ì�´ë¸ŒëŸ¬ë¦¬ ë¶ˆëŸ¬ì˜¤ê¸°
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')


#íŒŒì�¼ ë¶ˆëŸ¬ì˜¤ê¸°
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

# ë�°ì�´í„° í�¬ê¸° í™•ì�¸
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Submission shape:", submission.shape)

# ì•�ë¶€ë¶„ ë¯¸ë¦¬ ë³´ê¸°
print(train.head())


# ğŸ”� 3. ìˆ˜ì¹˜í˜• ë³€ìˆ˜ ì�´ìƒ�ì¹˜ ë¹„ìœ¨ ê³„ì‚° & ê²°ì¸¡ì¹˜ ì²˜ë¦¬
numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                'Friends_circle_size', 'Post_frequency']

def detect_outliers_IQR(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_count = ((series < lower) | (series > upper)).sum()
    total = series.notnull().sum()
    return outlier_count / total

outlier_ratios = {col: detect_outliers_IQR(train[col]) for col in numeric_cols}
fill_strategies = {col: 'median' if outlier_ratios[col] > 0.1 else 'mean' for col in numeric_cols}

def fill_missing_values(df, strategy_map):
    df_copy = df.copy()
    for col, strategy in strategy_map.items():
        if strategy == 'median':
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
        elif strategy == 'mean':
            df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
    return df_copy

train_filled = fill_missing_values(train, fill_strategies)
test_filled = fill_missing_values(test, fill_strategies)


# ğŸ§© 4. ë²”ì£¼í˜• ë³€ìˆ˜ ê²°ì¸¡ì¹˜ ìµœë¹ˆê°’ ì±„ìš°ê¸° ë°� ì¡°í•©
for col in ['Stage_fear', 'Drained_after_socializing']:
    mode = train[col].mode()[0]
    train[col] = train[col].fillna(mode)
    test[col] = test[col].fillna(mode)

def combine_social_disposition(row):
    if row['Stage_fear'] == 'Yes' and row['Drained_after_socializing'] == 'Yes':
        return 'Both_Yes'
    elif row['Stage_fear'] == 'No' and row['Drained_after_socializing'] == 'No':
        return 'Both_No'
    else:
        return 'Mixed'

train['Social_disposition'] = train.apply(combine_social_disposition, axis=1)
test['Social_disposition'] = test.apply(combine_social_disposition, axis=1)

# íŒŒìƒ�ë³€ìˆ˜ ì¶”ê°€ ì˜ˆì‹œ
train['Fear_outside'] = train['Stage_fear'] + '_' + train['Going_outside'].astype(str)
test['Fear_outside'] = test['Stage_fear'] + '_' + test['Going_outside'].astype(str)

# ì œê±°í•  ì›�ë�˜ ì»¬ëŸ¼
train = train.drop(columns=['Stage_fear', 'Drained_after_socializing'])
test = test.drop(columns=['Stage_fear', 'Drained_after_socializing'])




# ğŸ�·ï¸� 5. ë²”ì£¼í˜• ì�¸ì½”ë”©
# Label Encoding
le = LabelEncoder()
train['Personality'] = le.fit_transform(train['Personality'])  # Extrovert: 0, Introvert: 1

train = pd.get_dummies(train, columns=['Social_disposition', 'Fear_outside'])
test = pd.get_dummies(test, columns=['Social_disposition', 'Fear_outside'])

# train/test ì»¬ëŸ¼ ì •ë ¬ ì�¼ì¹˜
X_columns = train.drop(columns=['Personality']).columns
test = test.reindex(columns=X_columns, fill_value=0)





# ğŸ“Š 6. Feature/Target ë¶„ë¦¬ ë°� í‘œì¤€í™”
X = train.drop(columns=['id', 'Personality'])
y = train['Personality']
X_test_final = test.drop(columns=['id'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test_final)


# ğŸ�¯ 7. SVM í•™ìŠµ ë°� ê²€ì¦�
X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

svm = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_train)


# LightGBM (RandomSearchCV ê²°ê³¼ë¡œ ìµœì � ëª¨ë�¸)
lgbm = lgb.LGBMClassifier(num_leaves=31, max_depth=-1, learning_rate=0.05, 
                          n_estimators=200, min_child_samples=50, random_state=42)
lgbm.fit(X_train, y_train)







#í•˜ì�´í�¼íŒŒë�¼ë¯¸í„° íŠœë‹�(ë‹¤í•­ íŠ¹ì„± + GridSearchCV)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)  # ëª¨ë“  ëª¨ë�¸ì—� í™•ì�¥ ê°€ëŠ¥
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

param_grid = [
    {'penalty': ['l1'], 'solver': ['liblinear'], 'C': [0.01, 0.1, 1, 10], 'max_iter': [200, 300, 500]},
    {'penalty': ['l2'], 'solver': ['liblinear', 'saga'], 'C': [0.01, 0.1, 1, 10], 'max_iter': [200, 300, 500]}
]

lr = LogisticRegression()

grid = GridSearchCV(estimator=lr, param_grid=param_grid, scoring='f1_macro', cv=5, verbose=1, n_jobs=-1)
grid.fit(X_train_poly, y_train)

print("Best Parameters:", grid.best_params_)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test_poly)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
print(classification_report(y_test, y_pred))











#svm í•˜ì�´í�¼íŒŒë�¼ë¯¸í„° íŠœë‹�

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly', 
               
               
               'sigmoid']
}

svm = SVC(probability=True, random_state=42)

grid_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)
grid_svm.fit(X_train, y_train)

print("Best SVM Params:", grid_svm.best_params_)

y_pred_svm = grid_svm.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("SVM F1 Score:", f1_score(y_test, y_pred_svm, average='macro'))



#LightGBM íŠœë‹� ë°� í�‰ê°€
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV

lgbm = lgb.LGBMClassifier(random_state=42)

param_grid_lgb = {
    'num_leaves': [31, 50, 100],
    'max_depth': [-1, 10, 20],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 500],
    'min_child_samples': [20, 50, 100]
}

random_lgb = RandomizedSearchCV(lgbm, param_distributions=param_grid_lgb,
                                n_iter=20, scoring='f1_macro', cv=5, random_state=42, n_jobs=-1, verbose=1)
random_lgb.fit(X_train, y_train)

print("Best LightGBM Params:", random_lgb.best_params_)

y_pred_lgb = random_lgb.predict(X_test)
print("LightGBM Accuracy:", accuracy_score(y_test, y_pred_lgb))
print("LightGBM F1 Score:", f1_score(y_test, y_pred_lgb, average='macro'))



from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# ìµœì � íŒŒë�¼ë¯¸í„°ë¡œ ë¡œì§€ìŠ¤í‹± íšŒê·€ ëª¨ë�¸ ë‹¤ì‹œ ì •ì�˜ (ì˜ˆ: Best Parameters ë°˜ì˜�)
lr_best = LogisticRegression(C=0.01, penalty='l2', solver='liblinear', max_iter=200, random_state=42)

# SVMê³¼ LightGBMë�„ ìœ„ íŠœë‹� ê²°ê³¼ë¡œ ê°�ê°� best_estimator_ ê°€ì ¸ì˜¤ê¸°
svm_best = grid_svm.best_estimator_
lgb_best = random_lgb.best_estimator_

voting_clf = VotingClassifier(
    estimators=[('lr', lr_best), ('svm', svm_best), ('lgb', lgb_best)],
    voting='soft', n_jobs=-1
)

voting_clf.fit(X_train, y_train)
y_pred_ensemble = voting_clf.predict(X_test)

print("Ensemble Accuracy:", accuracy_score(y_test, y_pred_ensemble))
print("Ensemble F1 Score:", f1_score(y_test, y_pred_ensemble, average='macro'))
print(classification_report(y_test, y_pred_ensemble))




# í…ŒìŠ¤íŠ¸ ë�°ì�´í„° ì˜ˆì¸¡
final_pred = voting_clf.predict(X_test_scaled)

# ìˆ«ì�� â†’ ë¬¸ì�� ë³µì›�
submission['Personality'] = le.inverse_transform(final_pred)

# ìº�ê¸€ ì œì¶œìš© íŒŒì�¼ ì €ì�¥
submission.to_csv('submission.csv', index=False)
print("âœ… ìº�ê¸€ ì œì¶œ íŒŒì�¼ ìƒ�ì„± ì™„ë£Œ: submission.csv")



