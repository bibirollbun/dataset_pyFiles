# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import compute_sample_weight
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s4e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e2/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s4e2/sample_submission.csv')


train.head()


missing_sum = train.isnull().sum()
missing_percent = (missing_sum/len(train))*100
missing_df = pd.DataFrame({'missing total':missing_sum,'missing percent':missing_percent})
missing_df


train.info()


train['NObeyesdad'].value_counts()


def object_plt(df, col, hue='NObeyesdad', palette='Set1'):
    plt.figure(figsize=(12,6))
    sns.countplot(
        data=df,
        x=col,
        hue=hue,
        palette=palette,
        edgecolor='black'
    )
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('count')
    plt.xticks(rotation=45, ha='right')
    plt.legend(
        title=hue,
        labels=list(df[hue].unique())
    )
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()



object_plt(train,'Gender')


mapping_gender = {'Male':0,'Female':1}
train['Gender'] = train['Gender'].map(mapping_gender)

test['Gender'] = test['Gender'].map(mapping_gender)


train['family_history_with_overweight'].value_counts()


object_plt(train,'family_history_with_overweight')


summary = (
    train
    .groupby(['NObeyesdad', 'family_history_with_overweight'])
    .size()
    .unstack(fill_value=0)
)
summary



mapping_yes_no = {'no':0,'yes':1}
train['family_history_with_overweight'] = train['family_history_with_overweight'].map(mapping_yes_no)

test['family_history_with_overweight'] = test['family_history_with_overweight'].map(mapping_yes_no)


object_plt(train,'FAVC')


train['FAVC'] = train['FAVC'].map(mapping_yes_no)

test['FAVC'] = test['FAVC'].map(mapping_yes_no)


train['CAEC'].value_counts()


object_plt(train,'CAEC')


summary2 = (
    train
    .groupby(['NObeyesdad', 'CAEC'])
    .size()
    .unstack(fill_value=0)
)
summary2



# ohe = OneHotEncoder(
#     sparse_output=False,
#     dtype=int,
#     handle_unknown='ignore'
# )

# caec_array = ohe.fit_transform(train[['CAEC']])

# caec_cols = ohe.get_feature_names_out(['CAEC'])
# caec_df   = pd.DataFrame(caec_array, columns=caec_cols, index=train.index)

# train = pd.concat([train, caec_df], axis=1)
# train.drop('CAEC', axis=1, inplace=True)


object_plt(train,'SMOKE')



train['SMOKE'] = train['SMOKE'].map(mapping_yes_no)

test['SMOKE'] = test['SMOKE'].map(mapping_yes_no)


object_plt(train,'SCC')


train['SCC'] = train['SCC'].map(mapping_yes_no)


def object_plt(df, col, hue='NObeyesdad', palette='Set1'):
    plt.figure(figsize=(6,4))
    sns.countplot(
        data=df,
        x=col,
        hue=hue,
        palette=palette,
        edgecolor='black'
    )
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('count')
    plt.xticks(rotation=45, ha='right')
    plt.legend(
        title=hue,
        labels=list(df[hue].unique())
    )
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()



train['SCC'] = train['SCC'].map(mapping_yes_no)

test['SCC'] = test['SCC'].map(mapping_yes_no)


train.info()


object_plt(train, 'CALC')
object_plt(train, 'MTRANS')


def onehot_encode(
    df,
    col,
    drop_original: bool = True,
    prefix: str | None = None,
    sparse_output: bool = False,
    dtype: type = int,
    handle_unknown: str = 'ignore'
) -> pd.DataFrame:
   
    ohe = OneHotEncoder(
        sparse_output=sparse_output,
        dtype=dtype,
        handle_unknown=handle_unknown
    )

    arr = ohe.fit_transform(df[[col]])

    feat_name = prefix or col
    new_cols = ohe.get_feature_names_out([feat_name])
    df_ohe = pd.DataFrame(arr, columns=new_cols, index=df.index)
    
    df_new = pd.concat([df, df_ohe], axis=1)
    if drop_original:
        df_new = df_new.drop(col, axis=1)
    return df_new

train = onehot_encode(train,'CAEC')
# train = onehot_encode(train,'CALC')
train = onehot_encode(train,'MTRANS')

test = onehot_encode(test,'CAEC')
# test = onehot_encode(test,'CALC')
test = onehot_encode(test,'MTRANS')


le = LabelEncoder()
train['CALC'] = le.fit_transform(train['CALC'])

test['CALC'] = le.fit_transform(test['CALC'])


train.drop('SCC',axis=1,inplace=True)
test.drop('SCC',axis=1,inplace=True)


train.info()


train['BMI'] = train['Weight'] / (train['Height'] ** 2)

train['meal_veg_ratio'] = train['NCP'] / (train['FCVC'] + 1e-3)


# train['Age_group'] = pd.cut(train['Age'], bins=[14,20,30,40,61],
#                              labels=['14‑19','20‑29','30‑39','40+'])
train['Height_group'] = pd.cut(train['Height'], bins=4, labels=False)


train['PseudoTarget'] = pd.cut(train['BMI'],
                               bins=[0,18.5,25,30,35,40,100],
                               labels=[0,1,2,3,4,5]).astype(int)


test['BMI'] = test['Weight'] / (test['Height'] ** 2)
test['meal_veg_ratio'] = test['NCP'] / (test['FCVC'] + 1e-3)



# test['Age_group'] = pd.cut(test['Age'], bins=[14,20,30,40,61],
                             # labels=['14‑19','20‑29','30‑39','40+'])
test['Height_group'] = pd.cut(test['Height'], bins=4, labels=False)


test['PseudoTarget'] = pd.cut(test['BMI'],
                               bins=[0,18.5,25,30,35,40,100],
                               labels=[0,1,2,3,4,5]).astype(int)


train['Metabolic_Risk'] = train['BMI'] * train['Age'] / 10
train['Activity_Score'] = train['FAF'] / (train['TUE'] + 1e-3)
train['Nutrition_Index'] = train['FCVC'] + train['NCP'] - train['FAVC']

test['Metabolic_Risk'] = test['BMI'] * test['Age'] / 10
test['Activity_Score'] = test['FAF'] / (test['TUE'] + 1e-3)
test['Nutrition_Index'] = test['FCVC'] + test['NCP'] - test['FAVC']


train['Diet_Habit'] = train['CAEC_Always'] + train['CALC']
test['Diet_Habit'] = test['CAEC_Always'] + test['CALC']

train['Genetic_Risk'] = train['family_history_with_overweight'] * train['Gender']
test['Genetic_Risk'] = test['family_history_with_overweight'] * test['Gender']


train.info()


test.info()


train.drop('id',axis=1,inplace=True)

test.drop('id',axis=1,inplace=True)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train['NObeyesdad'] = le.fit_transform(train['NObeyesdad'])



# import pandas as pd
# from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# from xgboost import XGBClassifier

# X = train.drop('NObeyesdad', axis=1)
# y = train['NObeyesdad']

# # Eğitim ve test setlerindeki sütunları eşitle
# train_features = X.columns.tolist()

# # Test setinde eksik sütunları ekle (0 değeriyle doldur)
# for col in train_features:
#     if col not in test.columns:
#         test[col] = 0

# # Sütun sıralamasını eğitim setiyle aynı yap
# test = test[train_features]

# # 3) Train/test split
# X_train, X_valid, y_train, y_valid = train_test_split(
#     X, y,
#     test_size=0.2,
#     stratify=y,
#     random_state=42
# )

# # 4) Modeli tanımla
# model = XGBClassifier(
#     n_estimators=1000,
#     learning_rate=0.05,
#     max_depth=6,
#     use_label_encoder=False,
#     eval_metric='mlogloss',
#     random_state=42
# )

# # 5) 5‑fold Stratified CV ile accuracy skoru
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# cv_scores = cross_val_score(
#     model,
#     X_train, y_train,
#     cv=skf,
#     scoring='accuracy'
# )
# print(f"CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# # 6) Modeli tüm eğitim verisiyle eğit ve validation skorunu yazdır
# model.fit(X_train, y_train)
# val_score = model.score(X_valid, y_valid)
# print(f"Validation accuracy: {val_score:.4f}")


# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

# # 1) Özellikleri / hedefi ayır
# X = train.drop('NObeyesdad', axis=1)
# y = train['NObeyesdad']

# # 2) Train / validation split (son eğitim için)
# X_train, X_valid, y_train, y_valid = train_test_split(
#     X, y,
#     test_size=0.2,
#     stratify=y,
#     random_state=42
# )

# import numpy as np
# from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.utils import class_weight

# # Sınıf ağırlıklarını hesapla (sample_weight için)
# class_weights = class_weight.compute_sample_weight(
#     class_weight='balanced',
#     y=y_train
# )

# # Modeli güncelle (scale_pos_weight kaldırıldı, early_stopping için eval_set eklendi)
# model = XGBClassifier(
#     n_estimators=1500,
#     learning_rate=0.03,
#     max_depth=7,
#     subsample=0.8,
#     colsample_bytree=0.7,
#     gamma=0.1,
#     reg_alpha=0.1,
#     reg_lambda=1.0,
#     objective='multi:softmax',
#     num_class=7,
#     eval_metric='merror',  # mloss yerine merror daha uygun
#     use_label_encoder=False,
#     random_state=42,
#     tree_method='gpu_hist'  
# )

# # 5) 5-fold CV - sample_weight kullanarak
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# cv_scores = []

# for train_idx, val_idx in skf.split(X_train, y_train):
#     X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
#     y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
#     fold_weights = class_weight.compute_sample_weight(
#         class_weight='balanced',
#         y=y_fold_train
#     )
    
#     model.fit(
#         X_fold_train, 
#         y_fold_train,
#         sample_weight=fold_weights,
#         eval_set=[(X_fold_val, y_fold_val)],
#         verbose=0
#     )
    
#     val_pred = model.predict(X_fold_val)
#     fold_acc = accuracy_score(y_fold_val, val_pred)
#     cv_scores.append(fold_acc)

# print(f"CV accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# # 6) Final model eğitimi (tüm eğitim seti ile)
# final_weights = class_weight.compute_sample_weight(
#     class_weight='balanced',
#     y=y_train
# )

# model.fit(
#     X_train,
#     y_train,
#     sample_weight=final_weights,
#     eval_set=[(X_valid, y_valid)],
#     verbose=100
# )

# val_pred = model.predict(X_valid)
# val_score = accuracy_score(y_valid, val_pred)
# print(f"Validation accuracy: {val_score:.4f}")


X = train.drop('NObeyesdad', axis=1)
y = train['NObeyesdad']
train_features = X.columns.tolist()

for col in train_features:
    if col not in test.columns:
        test[col] = 0

test = test[train_features]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

model = XGBClassifier(
    n_estimators=1500,          
    learning_rate=0.1,         
    max_depth=7,                 
    min_child_weight=3,         
    gamma=0.2,                   
    subsample=0.8,               
    colsample_bytree=0.8,       
    reg_alpha=0.1,             
    reg_lambda=1.0,              
    objective='multi:softmax',   
    num_class=7,                
    eval_metric='mlogloss',      
    use_label_encoder=False,
    random_state=42,
    early_stopping_rounds=100,  
    tree_method='hist',          
    n_jobs=-1                   
)

sample_weights = compute_sample_weight(
    class_weight='balanced',
    y=y_train
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

print("Starting 5-fold cross-validation...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\nFold {fold+1}/5")
    
    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    fold_weights = compute_sample_weight(
        class_weight='balanced',
        y=y_fold_train
    )
    
    model.fit(
        X_fold_train, 
        y_fold_train,
        sample_weight=fold_weights,
        eval_set=[(X_fold_val, y_fold_val)],
        verbose=0
    )
    
    val_pred = model.predict(X_fold_val)
    fold_acc = accuracy_score(y_fold_val, val_pred)
    cv_scores.append(fold_acc)
    print(f"Fold {fold+1} Accuracy: {fold_acc:.4f}")

print(f"\nCV accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

print("\nTraining final model on entire training set...")
final_weights = compute_sample_weight(
    class_weight='balanced',
    y=y_train
)

model.fit(
    X_train,
    y_train,
    sample_weight=final_weights,
    eval_set=[(X_valid, y_valid)],
    verbose=100
)

val_pred = model.predict(X_valid)
val_acc = accuracy_score(y_valid, val_pred)
print(f"\nValidation accuracy: {val_acc:.4f}")



train.info()


test.info()


test1 = pd.read_csv('/kaggle/input/playground-series-s4e2/test.csv')
ids  = test1['id']


X_test = test

preds_numeric = model.predict(X_test)


preds_labels = le.inverse_transform(preds_numeric)


submission = pd.DataFrame({
    'id':          ids,
    'NObeyesdad': preds_labels
})
submission.to_csv('submission.csv', index=False)
submission.head()



X_test.info()


train.info()

