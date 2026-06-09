import warnings
warnings.filterwarnings("ignore")

import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay, roc_auc_score
from sklearn.utils import compute_class_weight


SEED = 52


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv', index_col='id')
submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


train.head()


def create_summary(df):
    describe = df.describe().transpose()
    summary = pd.DataFrame(df.dtypes, columns=['DataType'])
    summary['MissingValues'] = df.isnull().sum()
    summary['UniqueValues'] = df.nunique()
    summary['FirstValue'] = df.iloc[0]
    summary['SecondValue'] = df.iloc[1]
    summary['ThirdValue'] = df.iloc[2]
    summary = pd.concat([summary, describe], axis=1)
    summary = summary.fillna('--')
    return summary

create_summary(train)


# All features
test.columns.tolist()


# Add addtional features from existing ones
def add_features(df):

    df['bmi_age_interaction'] = df['bmi'] * df['age']
    df['bp_cholesterol_interaction'] = df['systolic_bp'] * df['cholesterol_total']

    # Age-based feature engineering
    df['age_squared'] = df['age'] ** 2
    df['age_category'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 60, 100], labels=['young', 'young_adult', 'middle_aged', 'senior', 'elderly'])

    # Lifestyle risk score
    df['lifestyle_risk_score'] = (
        df['alcohol_consumption_per_week'] * 0.2 +
        (10 - df['diet_score']) * 0.3 +  # Higher diet score is better, so invert
        (df['screen_time_hours_per_day'] / 10) * 0.2 +
        (1 / (df['physical_activity_minutes_per_week'] + 1)) * 100 * 0.3  # Invert physical activity
    )

    # Health metrics combinations
    df['cholesterol_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1)
    df['bp_ratio'] = df['systolic_bp'] / df['diastolic_bp']
    df['sleep_screen_ratio'] = df['sleep_hours_per_day'] / df['screen_time_hours_per_day']

    # Binary age group
    df['is_elderly'] = (df['age'] >= 60).astype(int)
    df['is_overweight'] = (df['bmi'] >= 25).astype(int)
    df['high_bp'] = ((df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 80)).astype(int)

    return df

train = add_features(train)
test = add_features(test)


# Visualize distributions of both numerical and categorical features
ncols = 3
nrows = int(np.ceil(len(train.columns) / ncols))
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows*4))
ax = ax.flatten()

for i, column in enumerate(train.columns):
    sns.histplot(data=train, x=column, kde=False, ax=ax[i], hue='diagnosed_diabetes', multiple="dodge")

for i in range(len(train.columns), len(ax)):
    fig.delaxes(ax[i])

plt.tight_layout()
plt.show()


X = train.drop('diagnosed_diabetes', axis=1)
y = train['diagnosed_diabetes']
X_test = test.copy()


# Compute class weights to handle class imbalance
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(y),
                                     y=y)

class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
class_weight_dict


num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OrdinalEncoder(), cat_cols)
    ]
)


# Outlier removal using IsolationForest
iso = IsolationForest(contamination=0.03, random_state=SEED)

iso_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('outliers', iso)
])

outlier_mask = iso_pipeline.fit_predict(X) == 1

X = X[outlier_mask]
y = y[outlier_mask]

print(f"Removed {sum(~outlier_mask)} outliers from training data")
print(f"New training data shape: {X.shape}")


# # Test various models with default parameters
# models = {
#     'Logistic Regression': LogisticRegression(random_state=SEED),
#     'Random Forest': RandomForestClassifier(random_state=SEED),
#     'Gradient Boosting': GradientBoostingClassifier(random_state=SEED),
#     'AdaBoost': AdaBoostClassifier(random_state=SEED),
#     'K-Nearest Neighbors': KNeighborsClassifier(),
#     'Decision Tree': DecisionTreeClassifier(random_state=SEED),
#     'LightGBM': lgb.LGBMClassifier(random_state=SEED, verbosity=-1),
#     'XGBoost': xgb.XGBClassifier(random_state=SEED),
#     'CatBoost': cb.CatBoostClassifier(random_state=SEED, verbose=0)
# }

# for name, model in models.items():
#     pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                                ('classifier', model)])
#     scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
#     print(f"{name}: Mean ROC AUC = {scores.mean():.4f}, Std = {scores.std():.4f}")



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.8, random_state=SEED, stratify=y)


# Use optuna to optimize Catboost classifier
np.random.seed(SEED)

fixed_param = {
    'eval_metric': 'AUC',
    'random_state': SEED,
    'verbose': 0
}

def objective(trial):

    param = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-5, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_strength': trial.suggest_float('random_strength', 1e-5, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'class_weights': trial.suggest_categorical('class_weights', [class_weight_dict, None]), 
    }
    
    model = cb.CatBoostClassifier(cat_features=cat_cols, **param, **fixed_param)
    # pipeline = Pipeline(steps=[
    #     ('preprocessor', preprocessor),
    #     ('classifier', model)
    #     ]
    # )
    
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')

    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)


best_params = study.best_params
print("Best parameters:", best_params)


# saved_params = {
#     'iterations': 987, 
#     'depth': 3, 
#     'learning_rate': 0.2845052935458534, 
#     'l2_leaf_reg': 0.0003697272009356358, 
#     'border_count': 217, 
#     'random_strength': 0.2109179027568805, 
#     'bagging_temperature': 0.6365445976520845
# }


best_model = cb.CatBoostClassifier(cat_features=cat_cols, **best_params, **fixed_param)
# pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                            ('classifier', best_model)])

best_model.fit(X_train, y_train)
y_pred = best_model.predict_proba(X_valid)[:, 1]
y_pred_int = (y_pred >= 0.5).astype(int)
roc_auc = roc_auc_score(y_valid, y_pred)
print(f"Validation ROC AUC: {roc_auc:.4f}")


# Visualize ROC Curve and Confusion Matrix

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
RocCurveDisplay.from_predictions(y_valid, y_pred, ax=ax[0])
ConfusionMatrixDisplay.from_predictions(y_valid, y_pred_int, ax=ax[1])
plt.show()


# Fit model to all data
best_model = cb.CatBoostClassifier(cat_features=cat_cols, **best_params, **fixed_param)
# pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                            ('classifier', best_model)])

best_model.fit(X, y)
y_test_pred = best_model.predict_proba(X_test)[:, 1]


submission['diagnosed_diabetes'] = y_test_pred
submission.to_csv('submission.csv', index=False)


submission.head()




