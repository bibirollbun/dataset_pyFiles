import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


# Loading dataset

df = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/train.csv')


df.info()

df.describe()


# import dtale

# d = dtale.show(df)
# d.open_browser()


# # Analysis of the dataset:
# # Features which are useful for the model based upon
# #   1. Correlation analysis
# #       1. |r| > 0.8 → Remove one of the correlated features (redundant).  
# #       2. 0.3 ≤ |r| ≤ 0.8 → Keep if useful for the model.  
# #       3. |r| < 0.3 → Likely not useful, consider removing.
# #   2. Class imbalance
# #   3. How well the features are defined

# # Selected features:
# #     1. Weight_kg
# #     2. Menstrual_Irregularity
# #     3. Hormonal_Imbalance
# #     4. Hyperandrogenism
# #     5. Hirsutism
# #     6. Stress_Level



df_selected_features = df[['Weight_kg', 'Hormonal_Imbalance', 'Hyperandrogenism', 'Hirsutism', 'PCOS']]

df_selected_features.info()


# d = dtale.show(df_selected_features)
# d.open_browser()


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df_selected_features.drop('PCOS', axis=1), df_selected_features['PCOS'], test_size=0.15, random_state=42)



# Preprocessing

numeric_features=['Weight_kg']
categorical_features=['Hormonal_Imbalance', 'Hyperandrogenism', 'Hirsutism']

imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()


numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('imputer', SimpleImputer(strategy='mean'))
])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
])

# Model for probability prediction
rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Fit the model
rf.fit(X_train, y_train)


# Predictions
y_pred = rf.predict(X_test)

# Model Evaluation
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

# Cross Validation
cv_score = cross_val_score(rf, X_train, y_train, cv=5)
print('Cross Validation Score:', cv_score.mean())


# creating a submission csv file for the test data set

df_test = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/test.csv')

predictions = rf.predict_proba(df_test)

submission = pd.DataFrame({"ID": df_test["ID"], "PCOS": predictions[:,1]})  
submission.to_csv("submission.csv", index=False)

