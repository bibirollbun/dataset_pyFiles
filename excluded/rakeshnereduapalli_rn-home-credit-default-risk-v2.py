### Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier, StackingClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score


### Data Loading
train_data = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
test_data = pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv')
bureau = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau.csv')
bureau_balance = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau_balance.csv')
pos_cash_balance = pd.read_csv('/kaggle/input/home-credit-default-risk/POS_CASH_balance.csv')
credit_card_balance = pd.read_csv('/kaggle/input/home-credit-default-risk/credit_card_balance.csv')
previous_application = pd.read_csv('/kaggle/input/home-credit-default-risk/previous_application.csv')
installments_payments = pd.read_csv('/kaggle/input/home-credit-default-risk/installments_payments.csv')


### Creating Aggregation Function
def aggregate_features(df, groupby_col, prefix):
    numeric_df = df.select_dtypes(include=[np.number])
    agg_df = numeric_df.groupby(groupby_col).agg(['mean', 'sum', 'max', 'min'])
    agg_df.columns = [f"{prefix}_{col[0]}_{col[1]}" for col in agg_df.columns]
    return agg_df.reset_index()


### Aggregating Features
bureau_agg = aggregate_features(bureau, 'SK_ID_CURR', 'bureau')
bureau_balance_agg = aggregate_features(bureau_balance, 'SK_ID_BUREAU', 'bureau_balance')
bureau_combined = bureau.merge(bureau_balance_agg, on='SK_ID_BUREAU', how='left')
bureau_combined_agg = aggregate_features(bureau_combined, 'SK_ID_CURR', 'bureau_combined')
pos_cash_balance_agg = aggregate_features(pos_cash_balance, 'SK_ID_CURR', 'pos_cash')
credit_card_balance_agg = aggregate_features(credit_card_balance, 'SK_ID_CURR', 'credit_card')
previous_application_agg = aggregate_features(previous_application, 'SK_ID_CURR', 'previous_app')
installments_payments_agg = aggregate_features(installments_payments, 'SK_ID_CURR', 'installments')


### Merging Aggregated Features into Main Dataset
train_data = train_data.merge(bureau_agg, on='SK_ID_CURR', how='left')
train_data = train_data.merge(bureau_combined_agg, on='SK_ID_CURR', how='left')
train_data = train_data.merge(pos_cash_balance_agg, on='SK_ID_CURR', how='left')
train_data = train_data.merge(credit_card_balance_agg, on='SK_ID_CURR', how='left')
train_data = train_data.merge(previous_application_agg, on='SK_ID_CURR', how='left')
train_data = train_data.merge(installments_payments_agg, on='SK_ID_CURR', how='left')

test_data = test_data.merge(bureau_agg, on='SK_ID_CURR', how='left')
test_data = test_data.merge(bureau_combined_agg, on='SK_ID_CURR', how='left')
test_data = test_data.merge(pos_cash_balance_agg, on='SK_ID_CURR', how='left')
test_data = test_data.merge(credit_card_balance_agg, on='SK_ID_CURR', how='left')
test_data = test_data.merge(previous_application_agg, on='SK_ID_CURR', how='left')
test_data = test_data.merge(installments_payments_agg, on='SK_ID_CURR', how='left')


### Handle Missing Values
train_data.fillna(0, inplace=True)
test_data.fillna(0, inplace=True)


### Defining Features and Target
X = train_data.drop(columns=['TARGET', 'SK_ID_CURR'])
y = train_data['TARGET']
X_test = test_data.drop(columns=['SK_ID_CURR'])


## Splitting Data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


### Preprocessing Pipelines
num_features = X.select_dtypes(include=['float64', 'int64']).columns
cat_features = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('to_string', FunctionTransformer(lambda x: x.astype(str), validate=False)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ]
)


import lightgbm as lgb
from scipy.stats import randint, uniform

lgb_model = lgb.LGBMClassifier(random_state=42)

# Define parameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': randint(100, 500),
    'learning_rate': uniform(0.01, 0.2),
    'num_leaves': randint(20, 50),
    'min_child_samples': randint(10, 50),
    'subsample': uniform(0.6, 0.9),
    'colsample_bytree': uniform(0.6, 0.9)
}



from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=param_dist,
    n_iter=10,
    scoring='roc_auc',
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=2
)


# Creating Pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', random_search)])


### Fit the Pipeline
pipeline.fit(X_train, y_train)


# Evaluate of the model 
from sklearn.metrics import roc_auc_score
y_val_preds = pipeline.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_val_preds)

print(f"Validation AUC: {val_auc}")


X_test = test_data.drop(columns=['SK_ID_CURR'])
test_preds = pipeline.predict_proba(X_test)[:, 1]

# Prepare the submission file
submission = pd.DataFrame({
    'SK_ID_CURR': test_data['SK_ID_CURR'],
    'TARGET': test_preds
})

submission.to_csv("submission.csv", index=False)
print("Submission file created successfully.")




