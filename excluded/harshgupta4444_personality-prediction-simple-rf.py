import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


train.head()


test.head()


train.info()


test.info()


display(train['Personality'].value_counts())

numerical_cols = train.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = train[numerical_cols].corr()
display(correlation_matrix)



train[numerical_cols].hist(figsize=(15, 10))
plt.tight_layout()
plt.show()



categorical_cols = train.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col != 'Personality': # Exclude the target variable
        plt.figure(figsize=(8, 5))
        sns.countplot(data=train, x=col)
        plt.title(f'Distribution of {col}')
        plt.show()


missing_train = train.isnull().sum()
missing_train = missing_train[missing_train > 0]
print("Missing values in train dataset:")
print(missing_train)


missing_test = test.isnull().sum()
missing_test = missing_test[missing_test > 0]
print("\nMissing values in test dataset:")
print(missing_test)


# Impute numerical columns with the median
numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
for col in numerical_cols:
    train[col].fillna(train[col].median(), inplace=True)
    test[col].fillna(test[col].median(), inplace=True)


# Impute categorical columns with the mode
categorical_cols = ['Stage_fear', 'Drained_after_socializing']
for col in categorical_cols:
    train[col].fillna(train[col].mode()[0], inplace=True)
    test[col].fillna(test[col].mode()[0], inplace=True)


print("\nMissing values in train dataset after imputation:")
print(train.isnull().sum()[train.isnull().sum() > 0])


print("\nMissing values in test dataset after imputation:")
print(test.isnull().sum()[test.isnull().sum() > 0])


categorical_features = ['Stage_fear', 'Drained_after_socializing']
numerical_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']



# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')



# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])



# Apply the preprocessor to the training data
train_processed = preprocessor.fit_transform(train)



test_processed = preprocessor.transform(test)


feature_names = numerical_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))


train_processed_df = pd.DataFrame(train_processed, columns=feature_names)
test_processed_df = pd.DataFrame(test_processed, columns=feature_names)



train_processed_df.head()


test_processed_df.head()


X_train = train_processed_df
y_train = train['Personality']



model = RandomForestClassifier(random_state=42)


model.fit(X_train, y_train)


y_pred_train = model.predict(X_train)


accuracy = accuracy_score(y_train, y_pred_train)
precision = precision_score(y_train, y_pred_train, average='weighted')
recall = recall_score(y_train, y_pred_train, average='weighted')
f1 = f1_score(y_train, y_pred_train, average='weighted')


print(f"Accuracy on training data: {accuracy:.4f}")
print(f"Precision on training data: {precision:.4f}")
print(f"Recall on training data: {recall:.4f}")
print(f"F1-score on training data: {f1:.4f}")


test_predictions = model.predict(test_processed_df)


submission_df = pd.DataFrame({'id': test['id'], 'Personality': test_predictions})
submission_df.to_csv('submission.csv', index=False)
display(submission_df.head())




