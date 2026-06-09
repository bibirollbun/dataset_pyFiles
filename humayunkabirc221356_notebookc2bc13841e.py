import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Load training data
df_train = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv")

# Encode categorical features
encoders = {}
cat_features = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']

for feature in cat_features:
    encoder = LabelEncoder()
    df_train[feature] = encoder.fit_transform(df_train[feature].astype(str))
    encoders[feature] = encoder

# Separate input features and target
X = df_train.drop(columns=['Unnamed: 0', 'id', 'satisfaction'])
y = df_train['satisfaction']

# Fill missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Train-test split
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_tr, y_tr)

# Evaluate
val_preds = xgb_model.predict(X_val)
print(f"Validation Accuracy: {accuracy_score(y_val, val_preds):.4f}")

# Load test dataset
df_test = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/test_dataset_exam.csv")

# Apply encoding to test set
for feature in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
    df_test[feature] = encoders[feature].transform(df_test[feature].astype(str))

# Prepare test features
X_test = df_test.drop(columns=['Unnamed: 0', 'id'], errors='ignore')
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Predict
df_test['satisfaction'] = xgb_model.predict(X_test)
df_test['satisfaction'] = encoders['satisfaction'].inverse_transform(df_test['satisfaction'])




# Rename the column 'id' to 'ID' if needed
df_test.rename(columns={'id': 'ID'}, inplace=True)

# Save only the required columns to submission file
df_test[['ID', 'satisfaction']].to_csv("submission.csv", index=False)


