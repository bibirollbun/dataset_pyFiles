import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")



train.info()


train.head()


test.head()


train.drop("id", axis=1,inplace=True
          )


train


print("Number of duplicates:", train.duplicated().sum())


print(train.isnull().sum())



categorical_cols = ["job", "marital", "education", "default", 
                    "housing", "loan", "contact", "month", "poutcome"]

for col in categorical_cols:
    print(f"\n--- {col} ---")
    print(train[col].value_counts(dropna=False))



train['contacted_before'] = (train['pdays'] != -1).astype(int)
test['contacted_before'] = (test['pdays'] != -1).astype(int)

# 2. Create an interaction feature for balance per age
# We add 1 to age to avoid any potential division by zero
train['balance_per_age'] = train['balance'] / (train['age'] + 1)
test['balance_per_age'] = test['balance'] / (test['age'] + 1)

print("New features created: 'contacted_before', 'balance_per_age'")




# Define column groups
ohe_cols = ["job", "marital", "contact", "poutcome", "month"]
ord_cols = ["education"]
bin_cols = ["default", "housing", "loan"]
num_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous", "contacted_before", "balance_per_age"]

# One-hot encoding for high-cardinality categoricals
categorical_transformer_ohe = OneHotEncoder(handle_unknown='ignore')

# Ordinal encoding for education (assuming order: primary < secondary < tertiary)
categorical_transformer_ord = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ord', OrdinalEncoder(categories=[['primary', 'secondary', 'tertiary']], 
                           handle_unknown='use_encoded_value', unknown_value=-1))
])

# Binary columns (yes/no → 0/1)
binary_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# Nume

numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Final ColumnTransformer
preprocessor = ColumnTransformer([
    ('ohe', categorical_transformer_ohe, ohe_cols),
    ('ord', categorical_transformer_ord, ord_cols),
    ('bin', binary_transformer, bin_cols),
    ('num', numerical_transformer, num_cols)
])



print(train['y'].value_counts(normalize=True))






numerical_cols = [x for x in train.columns if train[x].dtype in ['float64','int64']]

corr_matrix = train[numerical_cols].corr()


plt.figure(figsize=(14,12))
colormap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(
    corr_matrix,
    cmap=colormap,
    square=True,
    annot=True,
    fmt=".2f",
    linewidths=0.1,
    linecolor='white',
    cbar_kws={'shrink':0.9},
    annot_kws={'fontsize':12},
    vmax=1.0
)


X=train.drop("y" , axis=1)
y=train["y"]
X_test=test.copy()


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)





from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier
import numpy as np

# 1. Compute class imbalance ratio
neg, pos = np.bincount(y)   # counts of 0s and 1s
scale_pos_weight = neg / pos
print("scale_pos_weight:", scale_pos_weight)

# 2. Define StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 3. Pipeline already has preprocessor + classifier
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        tree_method='hist',    # GPU: "gpu_hist" if available
        scale_pos_weight=scale_pos_weight
    ))
])

# 4. Hyperparameter search space
param_dist = {
    'classifier__n_estimators': [200, 400, 600],
    'classifier__max_depth': [3, 5, 7, 10],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'classifier__subsample': [0.6, 0.8, 1.0],
    'classifier__colsample_bytree': [0.6, 0.8, 1.0],
    'classifier__reg_alpha': [0, 0.1, 1],
    'classifier__reg_lambda': [1, 2, 5],
    'classifier__gamma': [0, 0.1, 0.5]
}

# 5. Randomized Search with StratifiedKFold
search = RandomizedSearchCV(
    pipe,
    param_distributions=param_dist,
    n_iter=20,                # number of configs to try
    scoring='roc_auc',
    cv=cv,                    # stratified kfold
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# 6. Fit
search.fit(X_train, y_train)

print("Best ROC-AUC:", search.best_score_)
print("Best Params:", search.best_params_)




# 7. Evaluate the best model on the hold-out validation set
print("\nEvaluating the best model on the unseen validation set...")
best_model = search.best_estimator_
y_val_pred_proba = best_model.predict_proba(X_val)[:, 1]
val_roc_auc = roc_auc_score(y_val, y_val_pred_proba)

print(f"Validation ROC-AUC score: {val_roc_auc:.5f}")






test_predictions_proba = best_model.predict_proba(X_test.drop("id", axis=1))[:, 1]
submission_df = pd.DataFrame({'id': test['id'], 'y': test_predictions_proba})

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully!")




