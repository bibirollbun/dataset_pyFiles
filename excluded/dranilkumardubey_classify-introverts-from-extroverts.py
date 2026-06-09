# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_log_error, accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.impute import SimpleImputer

from skopt import BayesSearchCV
import warnings
warnings.filterwarnings('ignore')



# Step 2: Load Dataset
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')



# Step 3: Encode Target Label
le = LabelEncoder()
train['Personality'] = le.fit_transform(train['Personality'])
target = 'Personality'



# Step 4: EDA and Heatmap (Convert categorical to numeric temporarily for correlation)
eda_df = train.copy()
for col in eda_df.select_dtypes(include='object').columns:
    eda_df[col] = LabelEncoder().fit_transform(eda_df[col])

plt.figure(figsize=(12, 8))
sns.heatmap(eda_df.drop(columns=['id']).corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()


# Step 5: Identify Numerical and Categorical Columns
numerical = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical = [col for col in numerical if col not in ['id', target]]

categorical = train.select_dtypes(include='object').columns.tolist()



# Step 6: Preprocessing Pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical),
        ('cat', categorical_transformer, categorical)
    ])



# Step 7: Train-Test Split
X = train.drop(columns=['id', target])
y = train[target]
X_test = test.drop(columns='id')

X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)



# Step 8: XGBoost + Preprocessor + BayesSearchCV
xgb = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, random_state=42)

pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', xgb)
])

search_space = {
    'classifier__n_estimators': (100, 500),
    'classifier__max_depth': (3, 10),
    'classifier__learning_rate': (0.01, 0.3, 'log-uniform'),
    'classifier__subsample': (0.6, 1.0, 'uniform')
}

opt = BayesSearchCV(pipe, search_space, n_iter=30, cv=3, verbose=0, random_state=42, n_jobs=-1)
opt.fit(X_train, y_train)



# Step 9: Evaluation on Validation Set
y_pred_val = opt.predict(X_val)

print("Accuracy:", accuracy_score(y_val, y_pred_val))
print("F1 Score:", f1_score(y_val, y_pred_val, average='macro'))
print("Recall Score:", recall_score(y_val, y_pred_val, average='macro'))
print("RMSLE:", np.sqrt(mean_squared_log_error(y_val, y_pred_val)))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_val, y_pred_val), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()



# Step 10: Ensemble with StackingClassifier
base_models = [
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
    ('lgb', LGBMClassifier(random_state=42)),
    ('cat', CatBoostClassifier(verbose=0, random_state=42))
]

stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    cv=5
)

final_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('stacking', stack_model)
])

final_pipe.fit(X, y)



# Step 11: Final Predictions for Kaggle Submission
final_preds = final_pipe.predict(X_test)
sample_submission['Personality'] = le.inverse_transform(final_preds)
sample_submission.to_csv("submission.csv", index=False)





