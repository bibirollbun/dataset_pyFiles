import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import  GradientBoostingClassifier, VotingClassifier


train=pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sub=pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


train.head(5)


train.info()


train.columns


test.head(5)


test.info()


test.columns


sub.info()


import seaborn as sns
sns.countplot(x="y", data=train)
plt.title("Target Distribution")
plt.show()


# Age distribution 
sns.histplot(train['age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()


numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(10, 6))
sns.heatmap(train[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Numeric Feature Correlation")
plt.show()


train = train.dropna(subset=['y'])
y = train['y']
X = train.drop(['y', 'id'], axis=1)
X_test_final = test.drop(['id'], axis=1)


from sklearn.preprocessing import LabelEncoder, StandardScaler
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        X_test_final[col] = le.transform(X_test_final[col])


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test_final)


X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)



# Initialize models
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
gb = GradientBoostingClassifier(n_estimators=200, random_state=42)
xgb_tuned = XGBClassifier(
    max_depth=8,
    learning_rate=0.011192294909989933,
    min_child_weight=4,
    subsample=0.7326594105196521,
    n_estimators=100000,
    eval_metric='logloss',
    random_state=42,
    use_label_encoder=False,
    verbosity=0
)



from sklearn.ensemble import HistGradientBoostingClassifier

rf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', n_jobs=-1)
gb = HistGradientBoostingClassifier(max_iter=50, random_state=42)  # fast GB

models = {
    'Logistic Regression': LogisticRegression(max_iter=500, class_weight='balanced', random_state=42),
    'Random Forest': rf,
    'HistGradientBoosting': gb
}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(f"{name} Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print("-" * 40)
from sklearn.ensemble import HistGradientBoostingClassifier

rf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', n_jobs=-1)
gb = HistGradientBoostingClassifier(max_iter=50, random_state=42)  # fast GB

models = {
    'Logistic Regression': LogisticRegression(max_iter=500, class_weight='balanced', random_state=42),
    'Random Forest': rf,
    'HistGradientBoosting': gb
}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(f"{name} Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print("-" * 40)



from xgboost import XGBClassifier

xgb_fast = XGBClassifier(
    max_depth=5,
    learning_rate=0.05,
    n_estimators=300, 
    eval_metric='logloss',
    use_label_encoder=False,
    verbosity=1,
    random_state=42
)
# Train XGBoost with early stopping
print("Training XGBoost with early stopping...")
xgb_fast.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=20
)



y_pred_xgb = xgb_fast.predict(X_val)
print(f"XGBoost Accuracy: {accuracy_score(y_val, y_pred_xgb):.4f}")
print(classification_report(y_val, y_pred_xgb))
print("-" * 40)


ensemble = VotingClassifier(
    estimators=[
        ('log_reg', log_reg),
        ('rf', rf),
        ('gb', gb),
        ('xgb', xgb_fast)
    ],
    voting='soft'
)
ensemble.fit(X_train, y_train)
y_pred_ensemble = ensemble.predict(X_val)
print(f"Ensemble Accuracy: {accuracy_score(y_val, y_pred_ensemble):.4f}")
print(classification_report(y_val, y_pred_ensemble))


 ensemble.fit(X_scaled, y)
 y_test_pred = ensemble.predict(X_test_scaled)
 submission = pd.DataFrame({'id': test['id'], 'y': y_test_pred})
 submission.to_csv('submission.csv', index=False)


submission.head(5)




