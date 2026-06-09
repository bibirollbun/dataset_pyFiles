import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score,StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier



train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


train.head()


train.info()


train.select_dtypes(include='number').describe()


train_clean = train.drop(['id'],axis=1)
test_clean = test.drop(['id'],axis=1)


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

df_final.head()


test_encoded = ohe.fit_transform(test_clean[categorical])
test_encoded_cols = ohe.get_feature_names_out(categorical)
test_encoded_df = pd.DataFrame(test_encoded, columns=test_encoded_cols)
df_test_processed = test_clean.drop(columns=categorical)
df_final_test = pd.concat([df_test_processed, test_encoded_df], axis=1)


corr = df_final.corr()
#corr


plt.figure(figsize=(16,10))
mask = np.triu(np.ones_like(corr))
sns.heatmap(corr, cmap='Blues', vmin=0, linewidths=0.9,mask=mask)


df_final['bmi_age_interaction'] = df_final['bmi'] * df_final['age']
df_final_test['bmi_age_interaction'] = df_final_test['bmi'] * df_final_test['age']

df_final['bp_interaction'] = df_final['systolic_bp'] * df_final['diastolic_bp']
df_final_test['bp_interaction'] = df_final_test['systolic_bp'] * df_final_test['diastolic_bp']

df_final['diet_physical_activity'] = df_final['diet_score'] * df_final['physical_activity_minutes_per_week']
df_final_test['diet_physical_activity'] = df_final_test['diet_score'] * df_final_test['physical_activity_minutes_per_week']

df_final['cholesterol_ratio'] = df_final['cholesterol_total'] / (df_final['hdl_cholesterol'] + 1e-5)
df_final_test['cholesterol_ratio'] = df_final_test['cholesterol_total'] / (df_final_test['hdl_cholesterol'] + 1e-5)


df_final['log_cholesterol_total'] = np.log(df_final['cholesterol_total'] + 1)
df_final_test['log_cholesterol_total'] = np.log(df_final_test['cholesterol_total'] + 1)

df_final['log_hdl_cholesterol'] = np.log(df_final['hdl_cholesterol'] + 1)
df_final_test['log_hdl_cholesterol'] = np.log(df_final_test['hdl_cholesterol'] + 1)

df_final['log_ldl_cholesterol'] = np.log(df_final['ldl_cholesterol'] + 1)
df_final_test['log_ldl_cholesterol'] = np.log(df_final_test['ldl_cholesterol'] + 1)


df_final['bmi_category'] = pd.cut(df_final['bmi'], bins=[0, 18.5, 24.9, 29.9, 40], 
                                  labels=['underweight', 'normal', 'overweight', 'obese'])
df_final_test['bmi_category'] = pd.cut(df_final_test['bmi'], bins=[0, 18.5, 24.9, 29.9, 40], 
                                       labels=['underweight', 'normal', 'overweight', 'obese'])


bmi_ohe = ohe.fit_transform(df_final[['bmi_category']])
bmi_ohe_df = pd.DataFrame(bmi_ohe, columns=ohe.get_feature_names_out(['bmi_category']))

bmi_ohe_test = ohe.transform(df_final_test[['bmi_category']])
bmi_ohe_test_df = pd.DataFrame(bmi_ohe_test, columns=ohe.get_feature_names_out(['bmi_category']))


df_final = df_final.drop(columns=['bmi_category']).reset_index(drop=True)
df_final = pd.concat([df_final, bmi_ohe_df], axis=1)

df_final_test = df_final_test.drop(columns=['bmi_category']).reset_index(drop=True)
df_final_test = pd.concat([df_final_test, bmi_ohe_test_df], axis=1)

#Health Risk Score
df_final['health_risk_score'] = df_final['family_history_diabetes'] + df_final['hypertension_history'] + df_final['cardiovascular_history']
df_final_test['health_risk_score'] = df_final_test['family_history_diabetes'] + df_final_test['hypertension_history'] + df_final_test['cardiovascular_history']

# Age-Related Features
df_final['age_squared'] = df_final['age'] ** 2
df_final_test['age_squared'] = df_final_test['age'] ** 2

df_final['age_systolic_bp_interaction'] = df_final['age'] * df_final['systolic_bp']
df_final_test['age_systolic_bp_interaction'] = df_final_test['age'] * df_final_test['systolic_bp']

# Lifestyle Score (Composite Score)
lifestyle_columns = ['alcohol_consumption_per_week', 'physical_activity_minutes_per_week', 'diet_score', 'sleep_hours_per_day']
scaler = StandardScaler()
df_final[lifestyle_columns] = scaler.fit_transform(df_final[lifestyle_columns])
df_final_test[lifestyle_columns] = scaler.transform(df_final_test[lifestyle_columns])

df_final['lifestyle_score'] = df_final[lifestyle_columns].sum(axis=1)
df_final_test['lifestyle_score'] = df_final_test[lifestyle_columns].sum(axis=1)


values = df_final.drop(['diagnosed_diabetes'],axis=1)
target = df_final['diagnosed_diabetes']
values_scal = scaler.fit_transform(values)
pca = PCA(n_components=3) #top 3 components 
principalComponents = pca.fit_transform(values)
principalDf = pd.DataFrame(data=principalComponents,
                            columns=['principal component 1', 'principal component 2', 
                                     'principal component 3'])
principalDf['diagnosed_diabetes'] = target
principalDf.head()


correlation_with_target = df_final.corr()['diagnosed_diabetes'].sort_values(ascending=False)
correlation_with_target = values.corrwith(target).sort_values(ascending=False)

threshold = 0.1
strong_correlations = correlation_with_target[abs(correlation_with_target) > threshold]

plt.figure(figsize=(10, 6))
sns.barplot(x=strong_correlations.index, y=strong_correlations.values, palette='viridis')
plt.title(f'Strong Correlations (> {threshold}) with Diabetes')
plt.xticks(rotation=90)
plt.ylabel('Correlation Coefficient')
plt.xlabel('Features')
plt.show()


pca = PCA(n_components=3)
principalComponents = pca.fit_transform(values_scal)
pca_components = pd.DataFrame(pca.components_, columns=values.columns, 
                              index=[f'Principal Component {i+1}' for i in range(pca.n_components)])
pca_components




sorted_pca_components = pca_components.abs().T.sort_values(by='Principal Component 1', ascending=False)
print(sorted_pca_components.head(10))  # Display top 10 features 


plt.figure(figsize=(12, 8))
sns.heatmap(pca_components, cmap='viridis', fmt='.2f', linewidths=0.5)
plt.title('PCA: Feature contributions to each principal component')
plt.xlabel('Features')
plt.ylabel('Principal Components')
plt.show()


sns.countplot(data=df_final, x='diagnosed_diabetes').set_title('Diabetes')


sns.histplot(data=df_final, x='age', hue='diagnosed_diabetes', bins=20).set_title('Diabetes by age')


sns.histplot(data=df_final, x= 'bmi',hue='diagnosed_diabetes', bins=10).set_title('Diabetes by age groups')


sampled_data = df_final.sample(n=1000, random_state=42)
sns.jointplot(data=sampled_data, x='systolic_bp', y='bmi', hue='diagnosed_diabetes', kind='hist')



sampled_data = df_final.sample(n=1000, random_state=42)
sns.jointplot(data=sampled_data, x='age', y='bmi', hue='diagnosed_diabetes', kind='hist')


# majority_class = train_encoded[train_encoded['diagnosed_diabetes'] == 1]
# minority_class = train_encoded[train_encoded['diagnosed_diabetes'] == 0]
# minority_upsampled = resample(minority_class,
#                               replace=True,        
#                               n_samples=len(majority_class),  
#                               random_state=42)
# balanced_data = pd.concat([majority_class, minority_upsampled])


# X_balanced = balanced_data.drop('diagnosed_diabetes', axis=1)
# y_balanced = balanced_data['diagnosed_diabetes']


label = df_final.drop('diagnosed_diabetes', axis=1)
target = df_final['diagnosed_diabetes']
X_train, X_test, y_train, y_test = train_test_split(label, target, test_size=0.2, random_state=101, stratify=target)



xgboost_model = XGBClassifier(
    n_estimators=7500,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    early_stopping_rounds=50,
    alpha=1.0,  # L1 regularization term
    tree_method='hist',   
    device='cuda',        #GPU for training
)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

xgboost_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=500
)



y_pred_prob = xgboost_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_prob))



xgboost_preds = xgboost_model.predict_proba(df_final_test)[:, 1]


lgb = LGBMClassifier()
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
#cv = cross_validate(lgb, X_balanced, y_balanced, cv=skf, scoring='roc_auc', n_jobs=-1, return_estimator=True)
label = df_final.drop('diagnosed_diabetes', axis=1)
target = df_final['diagnosed_diabetes']
X_train, X_test, y_train, y_test = train_test_split(label, target, test_size=0.2, random_state=101)
cv = cross_validate(lgb, X_train, y_train, cv=skf, scoring='roc_auc', n_jobs=-1, return_estimator=True)


cv


for fold, model in enumerate(cv['estimator']):
   
    val_idx = list(skf.split(X_train, y_train))[fold][1]
    X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
    y_pred = model.predict(X_val)
    
   
    print(f"Classification Report for Fold {fold+1}:")
    print(classification_report(y_val, y_pred))
    print("="*50)


features = pd.DataFrame({
    'feature': cv['estimator'][0].feature_name_,
    'importance': cv['estimator'][0].feature_importances_
}).sort_values('importance', ascending=False)

features


test_preds = np.zeros((len(df_final_test), 10))  # 10 folds in StratifiedKFold

for fold, model in enumerate(cv['estimator']):
    
    test_preds[:, fold] = model.predict_proba(df_final_test)[:, 1]  # Probability for class 1 (positive class)
lgbm_preds = test_preds.mean(axis=1)
df_final_test['predicted_probabilities'] = lgbm_preds


final_pred_prob = (xgboost_preds + lgbm_preds) / 2



submission = pd.DataFrame({
    'id': test['id'],  
    'diagnosed_diabetes': final_pred_prob 
})

submission.to_csv('/kaggle/working/submission_diabetes_features_eng.csv', index=False)


submission.head()

