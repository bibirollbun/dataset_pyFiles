import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


# Separate features (X) and target (y)
X_train = train_df.drop(['id', 'Personality'], axis=1)
y_train = train_df['Personality']
X_test = test_df.drop('id', axis=1)
test_ids = test_df['id'] # Store test IDs for submission


numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train.select_dtypes(include='object').columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numerical_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])


label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)


model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', RandomForestClassifier(random_state=42))]) # Simplified n_estimators

model_pipeline.fit(X_train, y_train_encoded)


predictions_encoded = model_pipeline.predict(X_test)
predictions_personality = label_encoder.inverse_transform(predictions_encoded)


submission_df = pd.DataFrame({'id': test_ids, 'Personality': predictions_personality})
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully.")
print(submission_df.head())

