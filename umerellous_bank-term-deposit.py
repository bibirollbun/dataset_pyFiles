import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, RocCurveDisplay
import xgboost as xgb
import os


np.random.seed(42)


data_path = '/kaggle/input/playground-series-s5e8/'


train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
test_df = pd.read_csv(os.path.join(data_path, 'test.csv'))
sample_submission = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'))


print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")


train_df.head()


train_df.info()


train_df.isnull().sum()


plt.figure(figsize=(6, 4))
sns.countplot(x=train_df['y'])
plt.title('Distribution of the Target Variable')
plt.show()


X = train_df.drop(['id', 'y'], axis=1)
y = train_df['y']
X_test = test_df.drop('id', axis=1)


categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=np.number).columns.tolist()


print(f"Numerical features: {numerical_features}")
print(f"Categorical features: {categorical_features}")


numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


print(f"Training set size: {X_train.shape}")
print(f"Validation set size: {X_val.shape}")


scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Calculated scale_pos_weight for XGBoost: {scale_pos_weight:.2f}")


xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight
)


model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb_model)
])


model_pipeline.fit(X_train, y_train)


y_pred_proba = model_pipeline.predict_proba(X_val)[:, 1]


roc_auc = roc_auc_score(y_val, y_pred_proba)
print(f"ROC AUC Score on validation set: {roc_auc:.4f}")


fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_predictions(y_val, y_pred_proba, ax=ax)
plt.title('ROC Curve')
plt.show()


final_predictions = model_pipeline.predict_proba(X_test)[:, 1]


submission_df = pd.DataFrame({
    'id': test_df['id'],
    'y': final_predictions
})


submission_df.to_csv('submission.csv', index=False)


submission_df.head()




