import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


train_data = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/train.csv')
test_data = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/test.csv')


X_train = train_data.drop('PCOS', axis=1)
y_train = train_data['PCOS'].map({'Yes': 1, 'No': 0}) 


numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


model = RandomForestClassifier(random_state=42)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

pipeline.fit(X_train, y_train)


X_test = test_data.copy()
probabilities = pipeline.predict_proba(X_test)[:, 1]


submission = pd.DataFrame({
    'ID': test_data['ID'],
    'PCOS': probabilities 
})
submission.to_csv('submission.csv', index=False)

print("Submission file saved as 'submission.csv'")

