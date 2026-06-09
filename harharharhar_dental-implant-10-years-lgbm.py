import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report
from sklearn.utils import resample
from lightgbm import LGBMClassifier


df_train = pd.read_csv('/kaggle/input/dental-implant-10-year-survival-prediction/train.csv')
df_test = pd.read_csv('/kaggle/input/dental-implant-10-year-survival-prediction/test.csv')
df_train.head()


df_train.info()


train_clean = df_train.drop(['patient_id'],axis=1)
test_clean = df_test.drop(['patient_id'],axis=1)


numerical = train_clean.select_dtypes(include='number')
numerical.describe()


plt.figure(figsize=(15,8))
numerical.hist(figsize=(20,20), edgecolor='lightblue', bins=20)
plt.show()


plt.figure(figsize=(20,10))
corr = numerical.corr()
mask = np.triu(np.ones_like(corr))
sns.heatmap(corr, cmap='Blues', annot=True, mask=mask)
plt.show()


def fetch_cols(df):
    numerical = df.columns[df.dtypes != object]
    categorical = df.columns[df.dtypes == object]

    return numerical, categorical


numerical, categorical = fetch_cols(train_clean)


ohe = OneHotEncoder(sparse_output=False)

encoded = ohe.fit_transform(train_clean[categorical])
encoded_cols = ohe.get_feature_names_out(categorical)
encoded_df = pd.DataFrame(encoded, columns=encoded_cols)


df_processed = train_clean.drop(columns=categorical)
df_final = pd.concat([df_processed, encoded_df], axis=1)


majority_class = df_final[df_final['implant_survival_10y'] == 1]
minority_class = df_final[df_final['implant_survival_10y'] == 0]
minority_upsampled = resample(minority_class,
                              replace=True,        
                              n_samples=len(majority_class),  
                              random_state=42)
balanced_data = pd.concat([majority_class, minority_upsampled])


X_balanced = balanced_data.drop('implant_survival_10y', axis=1)
y_balanced = balanced_data['implant_survival_10y']




target = df_final['implant_survival_10y']  
values = df_final.drop(columns=['implant_survival_10y'])


lgb = LGBMClassifier()
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv = cross_validate(lgb, X_balanced, y_balanced, cv=skf, scoring='roc_auc', n_jobs=-1, return_estimator=True)


cv


for fold, model in enumerate(cv['estimator']):
   
    val_idx = list(skf.split(X_balanced, y_balanced))[fold][1]
    X_val, y_val = X_balanced.iloc[val_idx], y_balanced.iloc[val_idx]
    y_pred = model.predict(X_val)
    
   
    print(f"Classification Report for Fold {fold+1}:")
    print(classification_report(y_val, y_pred))
    print("="*50)


features = pd.DataFrame({
    'feature': cv['estimator'][0].feature_name_,
    'importance': cv['estimator'][0].feature_importances_
}).sort_values('importance', ascending=False)

features


test_encoded = ohe.fit_transform(test_clean[categorical])
test_encoded_cols = ohe.get_feature_names_out(categorical)
test_encoded_df = pd.DataFrame(test_encoded, columns=test_encoded_cols)
df_test_processed = test_clean.drop(columns=categorical)
df_final_test = pd.concat([df_test_processed, test_encoded_df], axis=1)

test_preds = np.zeros((len(df_final_test), 10))  # 10 folds in StratifiedKFold

for fold, model in enumerate(cv['estimator']):
    
    test_preds[:, fold] = model.predict_proba(df_final_test)[:, 1]  # Probability for class 1 (positive class)
final_preds = test_preds.mean(axis=1)
df_final_test['predicted_probabilities'] = final_preds



submission = pd.DataFrame({
    'patient_id': df_test['patient_id'],  
    'implant_survival_10y': final_preds  
})


submission.head()


submission.to_csv('/kaggle/working/submission_implants.csv', index=False)

