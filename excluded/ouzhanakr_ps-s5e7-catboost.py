# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    cross_val_score
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from catboost import CatBoostClassifier
import optuna



warnings.filterwarnings('ignore')


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
#submission_d = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


train.head()


missing_count = train.isnull().sum()
missing_percent = (missing_count/len(train))*100
missing_df = pd.DataFrame({'missing count':missing_count,'missing percent':missing_percent})
missing_df


train.info()


#train = train.dropna(subset=['Personality','Social_event_attendance','Post_frequency'])


numeric_cols = train.select_dtypes(include=['int64','float64']).columns.tolist()
num_imputer = SimpleImputer(strategy='median')
train[numeric_cols] = num_imputer.fit_transform(train[numeric_cols])

numeric_cols_test = test.select_dtypes(include=['int64','float64']).columns.tolist()
num_imputer_test = SimpleImputer(strategy='median')
test[numeric_cols_test] = num_imputer_test.fit_transform(test[numeric_cols_test])



cat_cols = train.select_dtypes(include=['object']).columns.tolist()
cat_imputer = SimpleImputer(strategy='most_frequent')
train[cat_cols] = cat_imputer.fit_transform(train[cat_cols])

cat_cols_test = test.select_dtypes(include=['object']).columns.tolist()
cat_imputer_test = SimpleImputer(strategy='most_frequent')
test[cat_cols_test] = cat_imputer_test.fit_transform(test[cat_cols_test])



train.isnull().sum()


mapping_yes_no = {'Yes': 1, 'No':0}
train['Stage_fear'] = train['Stage_fear'].map(mapping_yes_no)
test['Stage_fear'] = test['Stage_fear'].map(mapping_yes_no)

train['Drained_after_socializing'] = train['Drained_after_socializing'].map(mapping_yes_no)
test['Drained_after_socializing'] = test['Drained_after_socializing'].map(mapping_yes_no)



train.info()


test.head()


train = train.drop('id',axis=1)


test.isnull().sum()


# Bu kodu eksik deÄŸerleri doldurduktan sonra ekleyin
# Ã–rnek olarak birkaÃ§ yeni Ã¶zellik
train['Social_Activity_Score'] = train['Going_outside'] + train['Social_event_attendance'] - train['Time_spent_Alone']
train['Post_per_Friend'] = train['Post_frequency'] / (train['Friends_circle_size'] + 1) # 0'a bÃ¶lmeyi engellemek iÃ§in +1

# AynÄ± iÅŸlemleri test seti iÃ§in de yapÄ±n
test['Social_Activity_Score'] = test['Going_outside'] + test['Social_event_attendance'] - test['Time_spent_Alone']
test['Post_per_Friend'] = test['Post_frequency'] / (test['Friends_circle_size'] + 1)


# X['alone_times_friends_interaction'] = X['Time_spent_Alone'] * X['Friends_circle_size']
# test_features['alone_times_friends_interaction'] = (
#     test_features['Time_spent_Alone'] * test_features['Friends_circle_size']
# )


X = train.drop('Personality', axis = 1)
y = train['Personality']


# --- FEATURE ENGINEERING START ---
# train ve test DataFrame'leriniz hazÄ±r olduktan sonra:

# 1) EtkileÅŸim (multiplicative) terimleri
for df in [train, test]:
    df['alone_friends_mul']    = df['Time_spent_Alone'] * df['Friends_circle_size']
    df['alone_drained_mul']    = df['Time_spent_Alone'] * df['Drained_after_socializing']
    df['friends_drained_mul']  = df['Friends_circle_size'] * df['Drained_after_socializing']

# 2) Oran (ratio) terimleri
for df in [train, test]:
    df['social_per_friend']    = df['Social_event_attendance'] / (df['Friends_circle_size'] + 1)
    df['post_per_friend']      = df['Post_frequency'] / (df['Friends_circle_size'] + 1)
    df['going_per_social']     = df['Going_outside'] / (df['Social_event_attendance'] + 1)
    df['drained_per_social']   = df['Drained_after_socializing'] / (df['Social_event_attendance'] + 1)

# 3) Polinom (square) terimleri
numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
                'Friends_circle_size', 'Post_frequency']
for df in [train, test]:
    for col in numeric_cols:
        df[f'{col}_sq'] = df[col] ** 2

# 4) Binning (discretization) â€“ Ã¶rneÄŸin Time_spent_Alone iÃ§in
bins  = [ -0.1,  2, 5, 10, train['Time_spent_Alone'].max() ]
labels= ['very_low', 'low', 'medium', 'high']
for df in [train, test]:
    df['alone_bin'] = pd.cut(
        df['Time_spent_Alone'],
        bins=bins,
        labels=labels
    )

# 5) One-Hot Encoding for the new bin feature
train = pd.get_dummies(train, columns=['alone_bin'], prefix='alone')
test  = pd.get_dummies(test,  columns=['alone_bin'], prefix='alone')

# EÄŸer train/test sÃ¼tun sayÄ±larÄ± farklÄ±ysa, eksik kolonlarÄ± ekleyin
for col in set(train.columns) - set(test.columns):
    if col.startswith('alone_'):
        test[col] = 0
for col in set(test.columns) - set(train.columns):
    if col.startswith('alone_'):
        train[col] = 0

# --- FEATURE ENGINEERING END ---



# X['alone_times_friends_interaction'] = X['Time_spent_Alone'] * X['Friends_circle_size']
# test_features['alone_times_friends_interaction'] = (
#     test_features['Time_spent_Alone'] * test_features['Friends_circle_size']
# )






cat_model = CatBoostClassifier(iterations = 300, learning_rate = 0.1, depth = 4, random_seed = 42, verbose = False)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


cv_accuracy_scores = cross_val_score(cat_model, X, y, cv=cv, scoring='accuracy')
cv_f1_scores = cross_val_score(cat_model, X, y, cv=cv, scoring='f1_weighted')

print("CatBoost Model Cross-Validation Results")
print("ğŸ”� CV Accuracy Scores:", cv_accuracy_scores)
print("âœ… Mean CV Accuracy:", cv_accuracy_scores.mean())
print("ğŸ“‰ Std CV Accuracy:", cv_accuracy_scores.std())
print("ğŸ”� CV Weighted F1 Scores:", cv_f1_scores)
print("âœ… Mean CV Weighted F1 Score:", cv_f1_scores.mean())
print("ğŸ“‰ Std CV Weighted F1 Score:", cv_f1_scores.std())


cat_model.fit(X, y)

y_pred = cat_model.predict(X)

test_accuracy = accuracy_score(y, y_pred)
test_f1 = f1_score(y, y_pred, average='weighted')

print("\nğŸ§ª Full Data Accuracy:", test_accuracy)
print("ğŸ§ª Full Data Weighted F1 Score:", test_f1)
print("\nğŸ“‹ Classification Report:\n", classification_report(y, y_pred, digits=4))

cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cat_model.classes_)
disp.plot(cmap='Blues')
plt.title("CatBoost Confusion Matrix - Full Data")
plt.show()

if hasattr(cat_model, 'get_feature_importance'):
    importances = cat_model.get_feature_importance()
    feature_names = X.columns if hasattr(X, 'columns') else [f"Feature {i}" for i in range(X.shape[1])]
    
    feature_importance_df = (
        pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        .sort_values(by='Importance', ascending=False)
    )

    print("\nğŸ“Š CatBoost Feature Importances:")
    print(feature_importance_df)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title('CatBoost Top Feature Importances')
    plt.tight_layout()
    plt.show()
else:
    print("âš ï¸� CatBoost model does not support direct feature importance.")



from sklearn.preprocessing import StandardScaler, OrdinalEncoder


X.head()


test.head()


test_features = test.drop(columns=['id'], errors='ignore')


test_features



predictions = cat_model.predict(test_features)
#predictions = xgb_model.predict(test_features)




#le.fit(cat_model.classes_)


submission = pd.DataFrame({
    'id': test['id'],
    'Personality': predictions
})

# id'yi float â†’ int yap
submission['id'] = submission['id'].astype(int)

# Kaydet
submission.to_csv('submission.csv', index=False)

submission.to_csv("/kaggle/working/submission.csv", index=False)



ls


submission


# Ã‡alÄ±ÅŸtÄ±rÄ±n ve dosyanÄ±n listede olduÄŸundan emin olun
import os
print(os.listdir('/kaggle/working'))


import pandas as pd
import os

# Submission dosyasÄ±nÄ± oku ve kontrol et
if os.path.exists('/kaggle/working/submission.csv'):
    sub_check = pd.read_csv('/kaggle/working/submission.csv')
    print(f"âœ… Dosya bulundu! Boyut: {sub_check.shape}")
    print(sub_check.head())
else:
    print("â�Œ Dosya bulunamadÄ±! LÃ¼tfen yolu kontrol edin.")




