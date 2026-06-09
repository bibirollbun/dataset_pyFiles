# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


print(f"Eğitim (Train) seti boyutu: {train_df.shape}")
print(f"Test seti boyutu: {test_df.shape}")


train_df.head()


train_df.info()


submission_df.head()


train_df = train_df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)


print("--- Hedef Değişken Dağılımı ---")
target_counts = train_df['diagnosed_diabetes'].value_counts(normalize=True)
print(target_counts)


plt.figure(figsize=(6, 4))
sns.countplot(x='diagnosed_diabetes', data=train_df)
plt.title('Diyabet Dağılımı (0: Yok, 1: Var)')
plt.show()


cat_cols = train_df.select_dtypes(include=['object']).columns


for col in cat_cols:
    print(f"{col}: {train_df[col].nunique()} farklı değer -> {train_df[col].unique()[:5]}") # İlk 5 örneği göster


education_map = {
    'No formal': 0,
    'Highschool': 1,
    'Graduate': 2,
    'Postgraduate': 3
}

income_map = {
    'Low': 0,
    'Lower-Middle': 1,
    'Middle': 2,
    'Upper-Middle': 3,
    'High': 4
}


for df in [train_df, test_df]:
    df['education_level'] = df['education_level'].map(education_map)
    df['income_level'] = df['income_level'].map(income_map)

print("Ordinal dönüşüm tamamlandı. Örnek:")
print(train_df[['education_level', 'income_level']].head())



nominal_cols = ['gender', 'ethnicity', 'smoking_status', 'employment_status']


train_df = pd.get_dummies(train_df, columns=nominal_cols, drop_first=True)
test_df = pd.get_dummies(test_df, columns=nominal_cols, drop_first=True)


print(f"Yeni Train Boyutu: {train_df.shape}")
print(f"Yeni Test Boyutu: {test_df.shape}")


print(train_df.dtypes.value_counts())


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


for col in train_df.select_dtypes(include=['bool']).columns:
    train_df[col] = train_df[col].astype(int)
    test_df[col] = test_df[col].astype(int)


X = train_df.drop('diagnosed_diabetes', axis=1)
y = train_df['diagnosed_diabetes']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)


val_preds = log_model.predict(X_val_scaled)
val_prob = log_model.predict_proba(X_val_scaled)[:, 1]


print(f"Doğruluk (Accuracy) Skoru: {accuracy_score(y_val, val_preds):.4f}")
print(f"ROC-AUC Skoru: {roc_auc_score(y_val, val_prob):.4f}")
print("\n--- Sınıflandırma Raporu ---")
print(classification_report(y_val, val_preds))


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


ada_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)


ada_model.fit(X_train_scaled, y_train)


ada_preds = ada_model.predict(X_val_scaled)
ada_prob = ada_model.predict_proba(X_val_scaled)[:, 1]

print(f"AdaBoost Doğruluk (Accuracy): {accuracy_score(y_val, ada_preds):.4f}")
print(f"AdaBoost ROC-AUC: {roc_auc_score(y_val, ada_prob):.4f}")


from xgboost import XGBClassifier
import matplotlib.pyplot as plt


xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    n_jobs=-1, # Tüm işlemci çekirdeklerini kullan
    eval_metric='auc' # Başarı kriteri olarak AUC kullan
)


xgb_model.fit(X_train_scaled, y_train)


xgb_preds = xgb_model.predict(X_val_scaled)
xgb_prob = xgb_model.predict_proba(X_val_scaled)[:, 1]


print(f"XGBoost Doğruluk (Accuracy): {accuracy_score(y_val, xgb_preds):.4f}")
print(f"XGBoost ROC-AUC: {roc_auc_score(y_val, xgb_prob):.4f}")


feature_importances = pd.Series(xgb_model.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
feature_importances.nlargest(10).plot(kind='barh')
plt.title('En Önemli 10 Özellik (Feature Importance)')
plt.show()


xgb_tuned = XGBClassifier(
    n_estimators=1500,     
    learning_rate=0.01,    
    max_depth=8,            
    subsample=0.7,          
    colsample_bytree=0.7,   
    gamma=1,                
    reg_alpha=0.1,         
    reg_lambda=1.0,         
    n_jobs=-1,              
    tree_method='hist',     
    random_state=42
)


xgb_tuned.fit(X_train_scaled, y_train)


tuned_preds = xgb_tuned.predict(X_val_scaled)
tuned_prob = xgb_tuned.predict_proba(X_val_scaled)[:, 1]


print("\n--- TUNED MODEL SONUÇLARI ---")
print(f"Yeni Doğruluk (Accuracy): {accuracy_score(y_val, tuned_preds):.4f}")
print(f"Yeni ROC-AUC: {roc_auc_score(y_val, tuned_prob):.4f}")


import lightgbm as lgb


X_train_eng = X_train.copy()
X_val_eng = X_val.copy()

def create_new_features(df):
    
    df['BMI_Age_Interact'] = df['bmi'] * df['age']
    
   
    df['Cardio_Risk'] = df['systolic_bp'] + df['diastolic_bp'] + df['heart_rate']
    
 
    df['Cholesterol_Ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1)
    
    
    df['Health_Score'] = (df['physical_activity_minutes_per_week'] + df['sleep_hours_per_day']*10) - (df['alcohol_consumption_per_week']*5)
    
    return df


print("Yeni özellikler türetiliyor...")
X_train_eng = create_new_features(X_train_eng)
X_val_eng = create_new_features(X_val_eng)


scaler_new = StandardScaler()
X_train_eng_scaled = scaler_new.fit_transform(X_train_eng)
X_val_eng_scaled = scaler_new.transform(X_val_eng)

print(f"Eski Sütun Sayısı: {X_train.shape[1]}")
print(f"Yeni Sütun Sayısı: {X_train_eng.shape[1]}")



print("\nLightGBM Modeli Eğitiliyor...")

lgb_model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=8,
    num_leaves=31,        
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary',
    random_state=42,
    n_jobs=-1,
    verbose=-1        
)

lgb_model.fit(X_train_eng_scaled, y_train)

# Tahminler
lgb_preds = lgb_model.predict(X_val_eng_scaled)
lgb_prob = lgb_model.predict_proba(X_val_eng_scaled)[:, 1]

print(f"\nLightGBM + Yeni Özellikler Accuracy: {accuracy_score(y_val, lgb_preds):.4f}")
print(f"LightGBM + Yeni Özellikler ROC-AUC: {roc_auc_score(y_val, lgb_prob):.4f}")



def process_data(df):

    df = df.copy()

  
    df["age_group"] = pd.cut(df["age"], 3, labels=["Youth", "Adult", "Elderly"]).astype("object")


    df["bp_diff"] = df["systolic_bp"] - df["diastolic_bp"]


    df["cholesterol_unkown"] = df["cholesterol_total"] - (df["hdl_cholesterol"] + df["ldl_cholesterol"])

    
    df["CRI"] = df["cholesterol_total"] + df["triglycerides"] + df["systolic_bp"]

  
    df["bmi_group"] = pd.cut(df["bmi"], 3, labels=["Low", "Moderate", "High"]).astype("object")
    df["heart_group"] = pd.cut(df["heart_rate"], 3, labels=["Low", "Moderate", "High"]).astype("object")

 
    df["smoking_status"] = df["smoking_status"].replace({"Never": 0, "Former": 1, "Current": 2})

    return df


train_full = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv', index_col='id')
test_full = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv', index_col='id')
sub_sample = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


train_processed = process_data(train_full)
test_processed = process_data(test_full)


target = 'diagnosed_diabetes'
train_dummies = pd.get_dummies(train_processed.drop(target, axis=1), drop_first=True)
test_dummies = pd.get_dummies(test_processed, drop_first=True)


train_dummies, test_dummies = train_dummies.align(test_dummies, join='left', axis=1, fill_value=0)

X = train_dummies
y = train_processed[target]


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_scaled = scaler.transform(test_dummies)


lgb_params = {
    'n_estimators': 863,
    'learning_rate': 0.0325,
    'num_leaves': 21,
    'max_depth': 12,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbose': -1
}


xgb_params = {
    'n_estimators': 838,
    'learning_rate': 0.143, 
    'max_depth': 4,
    'subsample': 0.95,
    'colsample_bytree': 0.56,
    'min_child_weight': 9,
    'gamma': 0.56,
    'random_state': 42,
    'eval_metric': 'auc'
}

print("\nModeller Eğitiliyor (Bu biraz sürebilir)...")


model_lgb = lgb.LGBMClassifier(**lgb_params)
model_lgb.fit(X_train_scaled, y_train)
val_preds_lgb = model_lgb.predict_proba(X_val_scaled)[:, 1]
print(f"LightGBM AUC: {roc_auc_score(y_val, val_preds_lgb):.5f}")


model_xgb = XGBClassifier(**xgb_params)
model_xgb.fit(X_train_scaled, y_train)
val_preds_xgb = model_xgb.predict_proba(X_val_scaled)[:, 1]
print(f"XGBoost AUC: {roc_auc_score(y_val, val_preds_xgb):.5f}")


val_preds_ensemble = (val_preds_lgb * 0.6) + (val_preds_xgb * 0.4) # LightGBM'e biraz daha güvendik
print(f"Ensemble (Birleştirilmiş) AUC: {roc_auc_score(y_val, val_preds_ensemble):.5f}")


test_preds_lgb = model_lgb.predict_proba(test_scaled)[:, 1]
test_preds_xgb = model_xgb.predict_proba(test_scaled)[:, 1]
test_preds_final = (test_preds_lgb * 0.6) + (test_preds_xgb * 0.4)




from catboost import CatBoostClassifier


cat_params = {
    'iterations': 1952,
    'depth': 4,
    'learning_rate': 0.127,
    'random_state': 42,
    'verbose': 0,
    'allow_writing_files': False 
}

print("CatBoost Eğitiliyor ")
model_cat = CatBoostClassifier(**cat_params)
model_cat.fit(X_train_scaled, y_train)


val_preds_cat = model_cat.predict_proba(X_val_scaled)[:, 1]
print(f"CatBoost AUC: {roc_auc_score(y_val, val_preds_cat):.5f}")




ensemble_preds = (val_preds_lgb * 0.30) + (val_preds_xgb * 0.40) + (val_preds_cat * 0.30)

print("-" * 30)
print(f"XGBoost Skorun: {roc_auc_score(y_val, val_preds_xgb):.5f}")
print(f"Üçlü Ensemble Skorun: {roc_auc_score(y_val, ensemble_preds):.5f}")
print("-" * 30)


test_preds_cat = model_cat.predict_proba(test_scaled)[:, 1]

final_preds = (test_preds_lgb * 0.30) + (test_preds_xgb * 0.40) + (test_preds_cat * 0.30)


y_pred_class = [1 if x > 0.5 else 0 for x in ensemble_preds]


print(f"Accuracy (Doğruluk) Skoru: {accuracy_score(y_val, y_pred_class):.5f}")


from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score


train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv', index_col='id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv', index_col='id')
sub = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


def feature_engineer(df):
    df["age_group"] = pd.cut(df["age"], 3, labels=["Youth", "Adult", "Elderly"]).astype(str)
    df["bp_diff"] = df["systolic_bp"] - df["diastolic_bp"]
    df["cholesterol_unkown"] = df["cholesterol_total"] - (df["hdl_cholesterol"] + df["ldl_cholesterol"])
    df["CRI"] = df["cholesterol_total"] + df["triglycerides"] + df["systolic_bp"]
    return df

train_df = feature_engineer(train_df)
test_df = feature_engineer(test_df)


cat_features = [col for col in train_df.columns if train_df[col].dtype == 'object']


le = LabelEncoder()
for col in cat_features:

    combined = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
    le.fit(combined)
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))


X = train_df.drop('diagnosed_diabetes', axis=1)
y = train_df['diagnosed_diabetes']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


print("Native CatBoost Eğitiliyor...")


native_cat = CatBoostClassifier(
    iterations=2000,
    depth=6,             
    learning_rate=0.05,
    loss_function='Logloss',
    eval_metric='AUC',
    cat_features=cat_features, 
    verbose=200,
    random_state=42
)

native_cat.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)


val_probs = native_cat.predict_proba(X_val)[:, 1]


best_acc = 0
best_thresh = 0.5

print("\n--- En İyi Eşik Değeri (Threshold) Aranıyor ---")
for thresh in np.arange(0.3, 0.7, 0.01):
    preds = (val_probs > thresh).astype(int)
    acc = accuracy_score(y_val, preds)
    if acc > best_acc:
        best_acc = acc
        best_thresh = thresh

print(f"EN İYİ ACCURACY: {best_acc:.5f}")
print(f"Bunu sağlayan Eşik Değeri: {best_thresh}")


test_probs = native_cat.predict_proba(test_df)[:, 1]

final_preds = (test_probs > best_thresh).astype(int) 






from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
def process_data(df):
    df = df.copy()
 
    df["age_group"] = pd.cut(df["age"], 3, labels=["Youth", "Adult", "Elderly"]).astype(str)
    df["cholesterol_unkown"] = df["cholesterol_total"] - (df["hdl_cholesterol"] + df["ldl_cholesterol"])
    df["bp_diff"] = df["systolic_bp"] - df["diastolic_bp"]
    df["smoking_status"] = df["smoking_status"].replace({"Never": 0, "Former": 1, "Current": 2})
    df["CRI"] = df["cholesterol_total"] + df["triglycerides"] + df["systolic_bp"]
    df["bmi_group"] = pd.cut(df["bmi"], 3, labels=["Low", "Moderate", "High"]).astype(str)
    df["heart_group"] = pd.cut(df["heart_rate"], 3, labels=["Low", "Moderate", "High"]).astype(str)
    return df


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv', index_col='id')
submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


X = process_data(train).drop('diagnosed_diabetes', axis=1)
y = train['diagnosed_diabetes']
X_test = process_data(test)


cat_cols = X.select_dtypes(include=['object']).columns
num_cols = X.select_dtypes(exclude=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])


lgb_params = {
    'n_estimators': 863,
    'learning_rate': 0.0325,
    'num_leaves': 21,
    'max_depth': 12,
    'min_child_samples': 20,
    'verbose': -1,
    'random_state': 42
}

xgb_params = {
    'n_estimators': 838,
    'learning_rate': 0.1438,
    'max_depth': 4,
    'subsample': 0.9577,
    'colsample_bytree': 0.5607,
    'min_child_weight': 9,
    'gamma': 0.5636,
    'random_state': 42,
    'eval_metric': 'auc'
}

cat_params = {
    'iterations': 1952,
    'depth': 4,
    'learning_rate': 0.1278,
    'verbose': 0,
    'random_state': 42,
    'allow_writing_files': False
}


clf1 = XGBClassifier(**xgb_params)
clf2 = LGBMClassifier(**lgb_params)
clf3 = CatBoostClassifier(**cat_params)


voting_clf = VotingClassifier(
    estimators=[('xgb', clf1), ('lgb', clf2), ('cat', clf3)],
    voting='soft'
)


pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', voting_clf)
])


X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X, y, test_size=0.2, random_state=42)

print("Voting Model Eğitiliyor (3 Model Aynı Anda)...")
pipeline.fit(X_train_split, y_train_split)


val_probs = pipeline.predict_proba(X_val_split)[:, 1]
val_preds = pipeline.predict(X_val_split)

print("-" * 40)
print(f"Voting ROC-AUC: {roc_auc_score(y_val_split, val_probs):.5f}")
print(f"Voting Accuracy: {accuracy_score(y_val_split, val_preds):.5f}")
print("-" * 40)

test_probs = pipeline.predict_proba(X_test)[:, 1]



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def process_data(df):
    df = df.copy()
    df["age_group"] = pd.cut(df["age"], 3, labels=["Youth", "Adult", "Elderly"]).astype(str)
    df["cholesterol_unkown"] = df["cholesterol_total"] - (df["hdl_cholesterol"] + df["ldl_cholesterol"])
    df["bp_diff"] = df["systolic_bp"] - df["diastolic_bp"]
    df["smoking_status"] = df["smoking_status"].replace({"Never": 0, "Former": 1, "Current": 2})
    df["CRI"] = df["cholesterol_total"] + df["triglycerides"] + df["systolic_bp"]
    df["bmi_group"] = pd.cut(df["bmi"], 3, labels=["Low", "Moderate", "High"]).astype(str)
    df["heart_group"] = pd.cut(df["heart_rate"], 3, labels=["Low", "Moderate", "High"]).astype(str)
    return df


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')

TARGET = 'diagnosed_diabetes'


X = process_data(train).drop(['id', TARGET], axis=1)
y = train[TARGET]
X_test = process_data(test).drop('id', axis=1)


cat_cols = X.select_dtypes(include=['object']).columns
num_cols = X.select_dtypes(exclude=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])


lgb_params = {'n_estimators': 863, 'learning_rate': 0.0325, 'num_leaves': 21, 'max_depth': 12, 'min_child_samples': 20, 'verbose': -1, 'random_state': 42}
xgb_params = {'n_estimators': 838, 'learning_rate': 0.1438, 'max_depth': 4, 'subsample': 0.9577, 'colsample_bytree': 0.5607, 'min_child_weight': 9, 'gamma': 0.5636, 'random_state': 42, 'eval_metric': 'auc'}
cat_params = {'iterations': 1952, 'depth': 4, 'learning_rate': 0.1278, 'verbose': 0, 'random_state': 42, 'allow_writing_files': False}

voting_clf = VotingClassifier(
    estimators=[
        ('xgb', XGBClassifier(**xgb_params)), 
        ('lgb', LGBMClassifier(**lgb_params)), 
        ('cat', CatBoostClassifier(**cat_params))
    ],
    voting='soft'
)

model = Pipeline([('preprocessor', preprocessor), ('classifier', voting_clf)])


FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

print(f"{FOLDS} Fold Cross-Validation Başlıyor...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]
    

    model.fit(X_train_fold, y_train_fold)
    

    val_p = model.predict_proba(X_val_fold)[:, 1]
    oof_preds[val_idx] = val_p
    
    
    test_p = model.predict_proba(X_test)[:, 1]
    test_preds += test_p / FOLDS
    
    score = roc_auc_score(y_val_fold, val_p)
    print(f"Fold {fold+1} ROC-AUC: {score:.5f}")

print("-" * 30)
print(f"GENEL OOF ROC-AUC SKORU: {roc_auc_score(y, oof_preds):.5f}")
print("-" * 30)


sample_sub[TARGET] = test_preds
sample_sub.to_csv('submission_cv_voting.csv', index=False)

oof_df = pd.DataFrame()
oof_df['id'] = train['id']
oof_df[TARGET] = y
oof_df['pred'] = oof_preds
oof_df.to_csv('oof_predictions.csv', index=False)

print('Dosyalar başarıyla kaydedildi: submission_cv_voting.csv')


plt.figure(figsize=(10, 5))
sns.kdeplot(oof_preds, label='OOF Predictions (Train)', fill=True, color='blue', alpha=0.3)
sns.kdeplot(test_preds, label='Test Predictions', fill=True, color='orange', alpha=0.3)
plt.title('Distribution of Predictions: OOF vs Test')
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.legend()
plt.show()


import pandas as pd
import os

# Mevcut dosyamızı okuyalım
df = pd.read_csv('submission_cv_voting.csv')

# Kaggle'ın istediği isimle tekrar kaydedelim
df.to_csv('submission.csv', index=False)






