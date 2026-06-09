import pandas as pd
import numpy as np

# Load datasets
train = pd.read_csv('/kaggle/input/tfug-bbsr-kaggle-community-olympiad-decode-social-type/train.csv')
test = pd.read_csv('/kaggle/input/tfug-bbsr-kaggle-community-olympiad-decode-social-type/test.csv')
meta = pd.read_csv('/kaggle/input/tfug-bbsr-kaggle-community-olympiad-decode-social-type/metaData.csv')
sample_submission = pd.read_csv('/kaggle/input/tfug-bbsr-kaggle-community-olympiad-decode-social-type/sample_submission.csv')

print(train.head())
print(test.head())
print(meta.head())



# Work on copies
train_df = train.copy()
test_df = test.copy()

# Identify numeric & categorical columns
num_cols = train_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()

# Remove target from categorical columns
if 'Interaction profile' in cat_cols:
    cat_cols.remove('Interaction profile')

print("Numeric columns:", num_cols)
print("Categorical columns:", cat_cols)



from sklearn.preprocessing import LabelEncoder

for col in cat_cols:
    le = LabelEncoder()
    all_values = pd.concat([train_df[col].astype(str), test_df[col].astype(str)])
    le.fit(all_values)
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))



from sklearn.impute import SimpleImputer

# Drop rows with missing target
train_df = train_df.dropna(subset=['Interaction profile'])

# Impute numeric columns
num_imputer = SimpleImputer(strategy='median')
train_df[num_cols] = num_imputer.fit_transform(train_df[num_cols])
test_df[num_cols] = num_imputer.transform(test_df[num_cols])

# Impute categorical columns
cat_imputer = SimpleImputer(strategy='most_frequent')
train_df[cat_cols] = cat_imputer.fit_transform(train_df[cat_cols])
test_df[cat_cols] = cat_imputer.transform(test_df[cat_cols])



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
test_df[num_cols] = scaler.transform(test_df[num_cols])



from sklearn.model_selection import train_test_split

X = train_df.drop('Interaction profile', axis=1)
y = train_df['Interaction profile']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)

print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))



# Predict on test set
test_pred = clf.predict(test_df)

# Save submission
sample_submission['Interaction profile'] = test_pred
sample_submission.to_csv('submission.csv', index=False)
print("✅ Submission file saved!")


