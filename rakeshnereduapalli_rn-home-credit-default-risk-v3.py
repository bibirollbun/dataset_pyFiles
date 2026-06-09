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


# installing CatBoost
!pip install catboost


from catboost import CatBoostClassifier, Pool

# List of categorical feature names (from your earlier step)
cat_features = X.select_dtypes(include=['object']).columns.tolist()

# Get column indices for CatBoost
cat_features_indices = [X.columns.get_loc(col) for col in cat_features]


# Initialize the CatBoostClassifier
cat_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric='AUC',
    loss_function='Logloss',
    cat_features=cat_features_indices,
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50
)



# Fit the model on training data
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))


from sklearn.metrics import accuracy_score, roc_auc_score

# Predictions
y_pred = cat_model.predict(X_val)
y_proba = cat_model.predict_proba(X_val)[:, 1]

# Evaluation
accuracy = accuracy_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_proba)

print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Validation AUC: {auc:.4f}")



# Predict probabilities on test data
test_preds = cat_model.predict_proba(X_test)[:, 1]

# Create submission dataframe
submission = pd.DataFrame({
    'SK_ID_CURR': test_data['SK_ID_CURR'],
    'TARGET': test_preds
})

# Save to CSV
submission.to_csv('catboost_submission.csv', index=False)


## Evaluating Feature Importance
import matplotlib.pyplot as plt
feature_importances = cat_model.get_feature_importance(prettified=True)
top_features = feature_importances.head(20)

plt.figure(figsize=(10,6))
plt.barh(top_features['Feature Id'], top_features['Importances'], color='teal')
plt.xlabel("Importance Score")
plt.title("Top 20 Feature Importances - CatBoost")
plt.gca().invert_yaxis()
plt.show()




