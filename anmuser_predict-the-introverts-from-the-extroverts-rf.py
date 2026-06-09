import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder


# Load the data
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')  
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


train.isna().sum()


test.isna().sum()


# Handle missing data for numerical columns
numerical_cols = train.select_dtypes(include=['int64', 'float64']).columns

for col in numerical_cols:
    train[col].fillna(train[col].mean(), inplace=True) 

for col in numerical_cols:
    test[col].fillna(test[col].mean(), inplace=True) 


test[numerical_cols].isna().sum()


train[numerical_cols].isna().sum()


X = train.drop('Personality', axis=1)  # Features
y = train['Personality']  # Target variable


X.head(3)


# Handle missing data for categorical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

for col in categorical_cols:
    # Fill with the most frequent value
    X[col].fillna(X[col].mode()[0], inplace=True)
for col in categorical_cols:
    # Fill with the most frequent value
    test[col].fillna(test[col].mode()[0], inplace=True)


X[categorical_cols].isna().sum()


test[categorical_cols].isna().sum()


# One-hot encoding for categorical variables
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
test = pd.get_dummies(test, columns=categorical_cols, drop_first=True)


X.head(3)


test.head(3)


Y = pd.DataFrame(y)
Y


label_encoder = LabelEncoder()
Y['Personality'] = label_encoder.fit_transform(Y['Personality'])


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


model = xgb.XGBClassifier(max_depth=200,learning_rate=0.01,n_estimators=300)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print(classification_report(y_test, y_pred))


prediction = model.predict(test)


sample.head(3)


original_labels = label_encoder.inverse_transform(prediction)



sample['Personality'] = original_labels
sample.head(3)


sample.to_csv('submission.csv',index=False)




