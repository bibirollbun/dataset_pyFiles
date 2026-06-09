# 1. Importing Libraries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import log_loss


# 2. Uploading data
train_df = pd.read_csv('/kaggle/input/multiclassificationtask/train.csv')
test_df = pd.read_csv('/kaggle/input/multiclassificationtask/test.csv')
submission_df = pd.read_csv('/kaggle/input/multiclassificationtask/sample_submission.csv')


# 3. Separating categorical and numeric columns
cat_features = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
num_features = ['N_Days', 'Age', 'Bilirubin', 'Cholesterol', 'Albumin',
                'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Stage']

target_col = 'Status'
status_mapping = {'C': 0, 'CL': 1, 'D': 2}
train_df[target_col] = train_df[target_col].map(status_mapping)

# Removing NaN values
train_df = train_df.dropna(subset=[target_col])


# 4. Cleaning data and filling in missing values
imputer_cat = SimpleImputer(strategy='most_frequent')
imputer_num = SimpleImputer(strategy='median')
train_df[cat_features] = imputer_cat.fit_transform(train_df[cat_features])
test_df[cat_features] = imputer_cat.transform(test_df[cat_features])
train_df[num_features] = imputer_num.fit_transform(train_df[num_features])
test_df[num_features] = imputer_num.transform(test_df[num_features])


# 5. Coding Categorical Columns 
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
train_encoded = pd.DataFrame(encoder.fit_transform(train_df[cat_features]))
test_encoded = pd.DataFrame(encoder.transform(test_df[cat_features]))
train_encoded.columns = encoder.get_feature_names_out()
test_encoded.columns = encoder.get_feature_names_out()


# 6. Adding new Features 
train_df['Bilirubin_Albumin_Ratio'] = train_df['Bilirubin'] / train_df['Albumin']
test_df['Bilirubin_Albumin_Ratio'] = test_df['Bilirubin'] / test_df['Albumin']
num_features.append('Bilirubin_Albumin_Ratio')


# 7. Normalization
scaler = StandardScaler()
train_scaled = pd.DataFrame(scaler.fit_transform(train_df[num_features]), columns=num_features)
test_scaled = pd.DataFrame(scaler.transform(test_df[num_features]), columns=num_features)


# 8. Concating data
X_train = pd.concat([train_scaled, train_encoded], axis=1)
X_test = pd.concat([test_scaled, test_encoded], axis=1)
y_train = train_df[target_col]


# 9. Preparing Model
X_train_split, X_valid, y_train_split, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_split, y_train_split)


# 10. Evaluating Model
y_pred_proba = model.predict_proba(X_valid)
logloss = log_loss(y_valid, y_pred_proba)
print(f'Validation Log Loss: {logloss}')


test_predictions = model.predict_proba(X_test)

if 'id' in test_df.columns:
    submission_df = pd.DataFrame({'id': test_df['id']})
else:
    submission_df = pd.DataFrame({'id': range(15000, 15000+len(test_df))})

submission_df[['Status_C', 'Status_CL', 'Status_D']] = test_predictions

submission_df.to_csv('submission.csv', index=False)
print(submission_df)




