# ğŸ“¦ Importing Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from catboost import CatBoostRegressor
from xgboost import XGBRegressor


# ğŸ“‚ Load Dataset
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


# ğŸ§  Encode Categorical Variables
le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])


# ğŸ”§ Feature Engineering
for df in [train, test]:
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
    df['HR_per_min'] = df['Heart_Rate'] / df['Duration']
    df['Temp_per_min'] = df['Body_Temp'] / df['Duration']
    df['Effort'] = df['Heart_Rate'] * df['Body_Temp'] * df['Duration']
    df['Age_Weight'] = df['Age'] * df['Weight']
    df['Weight_per_height'] = df['Weight'] / df['Height']
    df['log_Duration'] = np.log1p(df['Duration'])
    df['log_HR'] = np.log1p(df['Heart_Rate'])


# ğŸ§¾ Feature Selection
features = [
    'Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp',
    'BMI', 'HR_per_min', 'Temp_per_min', 'Effort',
    'Age_Weight', 'Weight_per_height', 'log_Duration', 'log_HR'
]

X = train[features]
y = np.log1p(train["Calories"])  # Target in log scale
X_test = test[features]


# ğŸ§¼ Preprocessing Pipeline
numeric_features = [col for col in X.columns if col != 'Sex']
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features)
], remainder='passthrough')


# âš™ï¸� Initialize Base Models with Optimized Hyperparameters
catboost_model = CatBoostRegressor(
    learning_rate=0.1366,
    depth=8,
    l2_leaf_reg=5.2575,
    random_strength=1.3585,
    bagging_temperature=0.9722,
    border_count=253,
    iterations=1000,
    early_stopping_rounds=50,
    verbose=0
)

xgb_model = XGBRegressor(
    learning_rate=0.0717,
    max_depth=10,
    min_child_weight=2,
    gamma=0.019,
    subsample=0.9832,
    colsample_bytree=0.6812,
    reg_alpha=0.9333,
    reg_lambda=1.3558,
    n_estimators=1000,
    random_state=42
)


# ğŸ”� K-Fold Cross-Validation for Out-of-Fold Predictions
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_cat = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
test_preds_cat = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))

for train_idx, valid_idx in kf.split(X):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    pipe_cat = Pipeline([
        ('pre', preprocessor),
        ('cat', catboost_model)
    ])
    
    pipe_xgb = Pipeline([
        ('pre', preprocessor),
        ('xgb', xgb_model)
    ])

    pipe_cat.fit(X_train, y_train)
    pipe_xgb.fit(X_train, y_train)

    oof_cat[valid_idx] = pipe_cat.predict(X_valid)
    oof_xgb[valid_idx] = pipe_xgb.predict(X_valid)

    test_preds_cat += pipe_cat.predict(X_test) / kf.n_splits
    test_preds_xgb += pipe_xgb.predict(X_test) / kf.n_splits


# ğŸ§  Meta-Learner with Predefined Weights (Do not retrain)
learned_weights = [0.64229672, 0.35776623]
test_meta_X = np.vstack([test_preds_cat, test_preds_xgb]).T

# Apply weights directly
final_pred_log = test_meta_X @ learned_weights
final_pred = np.expm1(final_pred_log)


# ğŸ“¤ Create Submission
submission = pd.DataFrame({
    "id": test["id"],
    "Calories": final_pred
})
submission.to_csv("submission.csv", index=False)
print("ğŸ“� Submission saved as 'submission.csv'")

