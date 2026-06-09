# importing required libraries
import pandas as pd
import numpy as np
import scipy.stats as stats
from warnings import filterwarnings
filterwarnings('ignore')


file_path = '/kaggle/input/playground-series-s5e7/train.csv'
# reading the CSV file
df = pd.read_csv(file_path)
# Display the first few rows of the dataframe
print("First few rows of the dataframe:")
print(df.head())


# Display the summary of the dataframe
df.info()



# Display the data types of the dataframe
df.dtypes


# Display the shape of the dataframe
df.shape


# checking missing values
print("Missing values in each column:")
print(df.isnull().sum())


df.head()


# Filling missing values in 'Time_spent_Alone' with the mean
df['Time_spent_Alone'].fillna(df['Time_spent_Alone'].mean(), inplace=True)
# Filling missing values in 'Stage_fear' with the mode of 'Stage_fear'
df['Stage_fear'].fillna(df['Stage_fear'].mode()[0], inplace=True)
# Filling missing values in 'Social_event_attendance' with the mean
df['Social_event_attendance'].fillna(df['Social_event_attendance'].mean(), inplace=True)
# Filling missing values in 'Going_outside' with the mean
df['Going_outside'].fillna(df['Going_outside'].mean(), inplace=True)
# Filling missing values in 'Drained_after_socializing' with the mode
df['Drained_after_socializing'].fillna(df['Drained_after_socializing'].mode()[0], inplace=True)
# Filling missing values in 'Friends_circle_size' with the mean
df['Friends_circle_size'].fillna(df['Friends_circle_size'].mean(), inplace=True)
# Filling missing values in 'Post_frequencye' with the mean
df['Post_frequency'].fillna(df['Post_frequency'].mean(), inplace=True)







# Checking missing values after filling
df.isnull().sum()


# dropping duplicate values
df.drop_duplicates(inplace=True)


df.shape


cleaned_df = df.copy()



cleaned_df.sample(2)


from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder


scaler = MinMaxScaler()
cleaned_df[['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',]] = scaler.fit_transform(cleaned_df[['Time_spent_Alone', 'Social_event_attendance', 'Going_outside']])


cleaned_df.head()


le = LabelEncoder()
# Encoding categorical variables
cleaned_df[['Stage_fear', 'Drained_after_socializing']] = cleaned_df[['Stage_fear', 'Drained_after_socializing']].apply(le.fit_transform)


cleaned_df.head()


from scipy.stats import zscore
# Calculating z-scores for numerical columns
z_scores = np.abs(zscore(cleaned_df[['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']]))
outliers = (z_scores > 3).any(axis=1)
print(f"Here are outliers: {outliers}")
# Removing outliers
cleaned_df = cleaned_df[~outliers]
# Displaying the cleaned dataframe
print("Cleaned DataFrame after removing outliers:")
print(cleaned_df.head())
# Displaying the shape of the cleaned dataframe 
print("Shape of the cleaned DataFrame:")
print(cleaned_df.shape)


# Data visulization 
import matplotlib.pyplot as plt
import seaborn as sns
# Plotting the distribution of the target variable
plt.figure(figsize=(10, 6))
sns.countplot(x='Personality', data=df)
plt.title('Distribution of Target Variable')
plt.xlabel('Personality')
plt.ylabel('Count')
plt.show()
# Plotting the distribution of numerical features
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features):
    plt.subplot(3, 3, i + 1)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Statistical summary of numerical features
print("Statistical summary of numerical features:")
print(df.describe())



# Importing train_test_split from sklearn
from sklearn.model_selection import train_test_split
# Splitting the data into features and target variable
X = cleaned_df.drop('Personality', axis=1)
y = cleaned_df['Personality']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Displaying the shapes of the training and testing sets
print("Shapes of the training and testing sets:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")


# Displaying the first few rows of the training set
print("First few rows of the training set:")
print(X_train.head())
# Displaying the first few rows of the target variable
print("First few rows of the target variable:")
print(y_train.head())



# Importing models 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

model = LogisticRegression(max_iter=1000)
# Fitting the model on the training data
model.fit(X_train, y_train)

print("Model training completed.")




# Importing accuracy_score from sklearn
from sklearn.metrics import accuracy_score

# Making predictions on the test set
y_pred = model.predict(X_test)
# Calculating the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

print(f"Model accuracy: {accuracy * 100:.2f}")

# Importing confusion_matrix and classification_report from sklearn
from sklearn.metrics import confusion_matrix, classification_report
# Calculating confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Displaying confusion matrix
print("Confusion Matrix:")
print(cm)


# Calculating classification report
report = classification_report(y_test, y_pred)
# Displaying classification report
print("Classification Report:")
print(report)

# Importing joblib to save the model
import joblib
# Saving the trained model to a file
joblib.dump(model, 'personality_model.pkl')



# loading model from a file
loaded_model = joblib.load('personality_model.pkl')
# Making predictions using the loaded model
sample_data = X_test.sample(5)
predictions = loaded_model.predict(sample_data)
print(sample_data)


# Displaying the predictions
print("Predictions for the sample data:")
print(predictions)




# Predicting the personality type for a new sample
file_path = '/kaggle/input/playground-series-s5e7/test.csv'
# Reading the new sample data
new_data = pd.read_csv(file_path)
# pringint the first few rows of the new data
print("First few rows of the new data:")
print(new_data.head())


# Preprocessing the new data
pd.isnull(new_data).sum()


# handling missing values in the new data
# Filling missing values in 'Time_spent_Alone' with the mean
new_data['Time_spent_Alone'].fillna(new_data['Time_spent_Alone'].mean(), inplace=True)
# Filling missing values in 'Stage_fear' with the mode of 'Stage_fear'
new_data['Stage_fear'].fillna(new_data['Stage_fear'].mode()[0], inplace=True)
# Filling missing values in 'Social_event_attendance' with the mean
new_data['Social_event_attendance'].fillna(new_data['Social_event_attendance'].mean(), inplace=True)
# Filling missing values in 'Going_outside' with the mean
new_data['Going_outside'].fillna(new_data['Going_outside'].mean(), inplace=True)
# Filling missing values in 'Drained_after_socializing' with the mode
new_data['Drained_after_socializing'].fillna(new_data['Drained_after_socializing'].mode()[0], inplace=True)
# Filling missing values in 'Friends_circle_size' with the mean
new_data['Friends_circle_size'].fillna(new_data['Friends_circle_size'].mean(), inplace=True)
# Filling missing values in 'Post_frequencye' with the mean
new_data['Post_frequency'].fillna(new_data['Post_frequency'].mean(), inplace=True)


pd.isnull(new_data).sum()


scaller = MinMaxScaler()
new_data[['Time_spent_Alone', 'Social_event_attendance', 'Going_outside']] = scaller.fit_transform(new_data[['Time_spent_Alone', 'Social_event_attendance', 'Going_outside']])


le = LabelEncoder()
# Encoding categorical variables in the new data
new_data[['Stage_fear', 'Drained_after_socializing']] = new_data[['Stage_fear', 'Drained_after_socializing']].apply(le.fit_transform)



print("First few rows of the preprocessed new data:")
print(new_data.head())


# making prediction on the new data
new_predictions = loaded_model.predict(new_data)
# Displaying the predictions for the new data
print("Predictions for the new data:")
print(new_predictions)

# Saving the predictions to a CSV file
new_data['Predicted_Personality'] = new_predictions
new_data.to_csv('predicted_personality.csv', index=False)
print("Predictions saved to 'predicted_personality.csv'.")


