### Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
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


lightgbm = ('lightgbm', LGBMClassifier(random_state=42, n_estimators=200))
random_forest = ('random_forest', RandomForestClassifier(random_state=42, n_estimators=200))
logistic_regression = ('logistic', LogisticRegression(max_iter=1000))


## Stacking Classifier
stacked_model = StackingClassifier(
    estimators=[lightgbm, random_forest, logistic_regression],
    final_estimator=LogisticRegression(max_iter=500),
    cv=3,
    n_jobs=-1
)


## Full Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('stacked_model', stacked_model)
])


### Fit the Pipeline
from tqdm import tqdm
with tqdm(total=100, desc="Training Progress") as pbar:
    pipeline.fit(X_train, y_train)
    pbar.update(100)


## Evaluating the Model
y_pred = pipeline.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")


# Drop 'SK_ID_CURR' from test data to match the model input
X_test = test_data.drop(columns=['SK_ID_CURR'])

# Predict probabilities for the positive class (TARGET = 1)
test_preds = pipeline.predict_proba(X_test)[:, 1]

# Prepare the submission DataFrame
submission = pd.DataFrame({
    'SK_ID_CURR': test_data['SK_ID_CURR'],
    'TARGET': test_preds
})

# Save the submission to a CSV file
submission.to_csv("submission.csv", index=False)
print("Submission file created successfully.")




