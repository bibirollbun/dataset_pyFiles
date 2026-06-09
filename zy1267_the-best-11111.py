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



import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from category_encoders import TargetEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.quantiles = {}
        self.mean_values = {}
    
    def fit(self, X, y=None):
        num_cols = ['Monthly_Spending', 'Total_Usage_Hours', 'Account_Age_Months',
                   'Support_Calls', 'Late_Payments', 'Streaming_Usage']
        for col in num_cols:
            if col in X.columns:
                self.quantiles[col] = {
                    'q25': X[col].quantile(0.25),
                    'q75': X[col].quantile(0.75)
                }
                self.mean_values[col] = X[col].mean()
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        
        # 基础比值特征
        if all(col in X.columns for col in ['Monthly_Spending', 'Total_Usage_Hours']):
            X['Spending_Per_Hour'] = X['Monthly_Spending'] / (X['Total_Usage_Hours'] + 1e-5)
        
        # 交互特征
        if all(col in X.columns for col in ['Gender', 'Location']):
            X['Gender_Location'] = X['Gender'] + '_' + X['Location']
        if all(col in X.columns for col in ['Subscription_Type', 'Last_Interaction_Type']):
            X['Subscription_Interaction'] = X['Subscription_Type'] + '_' + X['Last_Interaction_Type']
        
        # 分位数特征
        for col in self.quantiles:
            X[f'{col}_Above_Q75'] = (X[col] > self.quantiles[col]['q75']).astype(int)
            X[f'{col}_Below_Q25'] = (X[col] < self.quantiles[col]['q25']).astype(int)
        
        # 行为特征组合
        if 'Support_Calls' in X.columns and 'Complaint_Tickets' in X.columns:
            X['Total_Issues'] = X['Support_Calls'] + X['Complaint_Tickets']
        
        if 'Late_Payments' in X.columns and 'Account_Age_Months' in X.columns:
            X['Late_Payment_Rate'] = X['Late_Payments'] / (X['Account_Age_Months'] + 1)
        
        # 二值特征
        if 'Satisfaction_Score' in X.columns:
            X['Is_Low_Satisfaction'] = (X['Satisfaction_Score'] <= 2).astype(int)
        
        if 'Promo_Opted_In' in X.columns:
            X['Promo_Active'] = X['Promo_Opted_In'].astype(int)
        
        return X


train_data = pd.read_csv('/kaggle/input/ultimate-customer-churn-prediction-challenge/train.csv')
test_data = pd.read_csv('/kaggle/input/ultimate-customer-churn-prediction-challenge/test.csv')

# 提取ID和目标
train_ids = train_data['Customer_ID']
test_ids = test_data['Customer_ID']
y = train_data['Churn']




feature_engineer = FeatureEngineer()
train_features = train_data.drop(['Customer_ID', 'Churn'], axis=1)
test_features = test_data.drop(['Customer_ID'], axis=1)

all_data = pd.concat([train_features, test_features])
all_data_engineered = feature_engineer.fit_transform(all_data)

X = all_data_engineered.iloc[:len(train_data)]
X_test = all_data_engineered.iloc[len(train_data):]



categorical_features = [
    'Gender', 'Location', 'Subscription_Type', 
    'Last_Interaction_Type', 'Gender_Location',
    'Subscription_Interaction'
]

numerical_features = [col for col in X.columns 
                     if col not in categorical_features + ['Customer_ID', 'Churn']]

# 删除冗余特征
redundant_features = [
    'Monthly_Spending_ZScore', 
    'Total_Usage_Hours_ZScore',
    'Hourly_Value'
]

X = X.drop(redundant_features, axis=1, errors='ignore')
X_test = X_test.drop(redundant_features, axis=1, errors='ignore')
numerical_features = [col for col in numerical_features if col not in redundant_features]



numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(mutual_info_classif, k=25))
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', TargetEncoder(smoothing=30))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ]
)


base_models = [
    ('xgb', XGBClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=5,
        gamma=0.5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        eval_metric='auc',
        random_state=42
    )),
    ('cat', CatBoostClassifier(
        iterations=800,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=0
    ))
]

meta_model = LogisticRegression(
    C=0.1,
    penalty='elasticnet',
    solver='saga',
    l1_ratio=0.5,
    max_iter=1000,
    random_state=42
)

stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    passthrough=True
)

final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', stacking_model)
])




cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

print("开始交叉验证训练...")
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    final_pipeline.fit(X_train, y_train)
    val_pred = final_pipeline.predict_proba(X_val)[:, 1]
    
    auc = roc_auc_score(y_val, val_pred)
    cv_scores.append(auc)
    print(f"Fold {fold+1} | AUC: {auc:.5f}")

print(f"\n平均交叉验证AUC: {np.mean(cv_scores):.5f} ± {np.std(cv_scores):.5f}")



val_preds = cross_val_predict(
    final_pipeline, 
    X, 
    y, 
    method='predict_proba',
    cv=cv
)[:, 1]

precision, recall, thresholds = precision_recall_curve(y, val_preds)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"\n最佳F1阈值: {best_threshold:.4f}")

print("\n在全量数据上训练最终模型...")
final_pipeline.fit(X, y)
test_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]
test_pred = (test_pred_proba >= best_threshold).astype(int)


submission = pd.DataFrame({
    'Customer_ID': test_ids,
    'Churn': test_pred
})

print("\n预测结果分布:")
print(submission['Churn'].value_counts(normalize=True))

submission.to_csv('optimized_submission.csv', index=False)
print("\n提交文件已保存！")

