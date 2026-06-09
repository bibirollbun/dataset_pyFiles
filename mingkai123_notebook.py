print("学号：2024423310120，姓名：刘铭凯")

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')

def preprocess_data(train, test):
    X_train = train.drop(['id', 'Fertilizer Name'], axis=1)
    y_train = train['Fertilizer Name']
    X_test = test.drop(['id'], axis=1)
    
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    num_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean'))
    ])
    
    cat_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    
    X_train_processed = preprocessor.fit_transform(X_train).astype(np.float32)
    X_test_processed = preprocessor.transform(X_test).astype(np.float32)
    
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    
    return X_train_processed, y_train_encoded, X_test_processed, le

X, y, X_test, le = preprocess_data(train, test)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = LGBMClassifier(
    objective='multiclass',
    num_class=len(np.unique(y)),
    metric='multi_logloss',
    learning_rate=0.05,
    n_estimators=500,
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)]
)

y_probs = model.predict_proba(X_test)
top5_indices = np.argsort(y_probs, axis=1)[:, -5:][:, ::-1]

top5_labels = []
for row in top5_indices:
    top5_labels.append(le.inverse_transform(row))
top5_labels = np.array(top5_labels)

submission['id'] = test['id']
submission['Fertilizer Name'] = [' '.join(row) for row in top5_labels]
submission.to_csv('submission.csv', index=False)

print(f"训练完成，提交文件已保存为 'submission.csv'")
print(f"模型最佳迭代轮数: {model.best_iteration_}")
print(f"验证集最佳分数: {model.best_score_}")




