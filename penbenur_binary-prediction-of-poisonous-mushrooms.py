import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


df_train=pd.read_csv('/kaggle/input/playground-series-s4e8/train.csv')
df_test=pd.read_csv('/kaggle/input/playground-series-s4e8/test.csv')


df_train.head()


df_test.head()


df_train=df_train.drop('id', axis=1)
test_id=df_test['id']
df_test=df_test.drop('id', axis=1)


df_train.info()


df_train.shape


df_train.describe()


df_train.corr(numeric_only=True)


df_train['has-ring'].value_counts()


df_train['does-bruise-or-bleed'].value_counts()


df_train['habitat'].value_counts()


# function to remove outliers
def remove_outliers(df):
    # Remove outliers from numeric columns
    for column in df.select_dtypes(include=['number']).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
   
    return df


# Remove outliers and infrequent categories from the entire dataset
df = remove_outliers(df_train)


df.head()


df['gill-spacing'].value_counts()


def handle_infrequent_categories(df, threshold=0.03):
    # Loop through all categorical columns
    for column in df.select_dtypes(include=['object']).columns:
        # Calculate the frequency of each category
        freq = df[column].value_counts(normalize=True)
        
        # Identify categories to keep
        categories_to_keep = freq[freq >= threshold].index
        
        # Replace infrequent categories with 'Other'
        df[column] = df[column].where(df[column].isin(categories_to_keep), 'Other')
    
    return df


# Apply the function to the 'habitat' column
df = handle_infrequent_categories(df, threshold=0.03)


df.isnull().sum()


df.head()


plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='class', palette='viridis')
plt.title('Class Distribution')
plt.savefig('class_distribution.png')  # Save figure
plt.show()


# Map original class labels to new labels
df['class'] = df['class'].map({'e': 1, 'p': 2})


# Convert categorical variables to numerical
df_encoded = pd.get_dummies(df, drop_first=True)


df_encoded


X = df_encoded.drop('class', axis=1) 
y = df_encoded['class']


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


 # Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy)


# Classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)


# Feature importances
feature_importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("Feature Importances:\n", importance_df)


# Save the model
joblib.dump(model, 'random_forest_model.pkl')




