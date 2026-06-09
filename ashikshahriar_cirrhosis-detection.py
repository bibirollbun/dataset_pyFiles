import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


train_data=pd.read_csv('/kaggle/input/kaggle-biochallenge-cirrhosis-detection/train.csv')
train_data.head()


train_data.info()


import seaborn as sns
import matplotlib.pyplot as plt

# Visualize missing values as a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(train_data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()



# Visualizing the distribution of numerical features
numerical_cols = train_data.select_dtypes(include=['float64', 'int64']).columns

# Plot histograms for numerical features
train_data[numerical_cols].hist(bins=30, figsize=(15, 12))
plt.tight_layout()
plt.show()

# Boxplots for detecting outliers
for col in numerical_cols:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=train_data[col])
    plt.title(f'Boxplot for {col}')
    plt.show()



# Compute correlation matrix for numerical features
correlation_matrix = train_data[numerical_cols].corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()



# Distribution of the target variable
plt.figure(figsize=(8, 6))
sns.countplot(x=train_data['Status'])
plt.title('Target Variable Distribution (Status)')
plt.show()

# For regression target variable, plot distribution
plt.figure(figsize=(8, 6))
sns.histplot(train_data['Stage'], kde=True)
plt.title('Target Variable Distribution (Stage)')
plt.show()



# Visualizing categorical features
categorical_cols = train_data.select_dtypes(include=['object']).columns

for col in categorical_cols:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=train_data[col])
    plt.title(f'Countplot for {col}')
    plt.xticks(rotation=45)
    plt.show()



# Handling missing values (Example: Imputing with mean for numerical columns)
train_data[numerical_cols] = train_data[numerical_cols].fillna(train_data[numerical_cols].mean())

# Impute categorical data with the mode
train_data[categorical_cols] = train_data[categorical_cols].fillna(train_data[categorical_cols].mode().iloc[0])



train_data.isnull().sum()


# Visualizing the relationship between categorical and numerical features (using boxplots or violin plots)
for col in categorical_cols:
    if col != 'Status':  # Skip the target column if it is categorical
        for num_col in numerical_cols:
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=train_data[col], y=train_data[num_col])
            plt.title(f'{num_col} vs {col}')
            plt.show()



# Cross-tabulation of two categorical variables
cross_tab = pd.crosstab(train_data['Drug'], train_data['Ascites'])
print(cross_tab)

# Plotting the cross-tab as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cross_tab, annot=True, cmap='Blues')
plt.title('Cross Tabulation: Drug vs Ascites')
plt.show()



from sklearn.ensemble import RandomForestClassifier

# Drop non-numeric columns (except target column)
X = train_data.drop(columns=['id', 'Status'])  # Drop 'Status' if it is the target
y = train_data['Status']

# Convert categorical columns into numerical (if any)
X = pd.get_dummies(X, drop_first=True)

# Train a random forest classifier and analyze feature importances
model = RandomForestClassifier()
model.fit(X, y)

# Plot feature importance
feature_importance = pd.DataFrame(model.feature_importances_, index=X.columns, columns=['importance'])
feature_importance.sort_values(by='importance', ascending=False).head(10).plot(kind='barh', figsize=(10, 6))
plt.title('Top 10 Important Features')
plt.show()



from sklearn.decomposition import PCA

# Standardize the data before PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

# Plotting the PCA results
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['target'] = y
sns.scatterplot(x='PC1', y='PC2', hue='target', data=pca_df, palette='viridis')
plt.title('PCA of Features')
plt.show()



from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


le = LabelEncoder()

for col in categorical_cols:
    train_data[col] = le.fit_transform(train_data[col])


train_data['Status'].value_counts()


X = train_data.drop(columns=['id', 'Status'])
y = train_data['Status']


from sklearn.model_selection import train_test_split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)



model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# Classification report
print(classification_report(y_test, y_pred))



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Train a Logistic Regression model
log_reg = LogisticRegression(max_iter=10000, random_state=42)
log_reg.fit(X_train, y_train)

# Predictions
y_pred_log_reg = log_reg.predict(X_test)

# Model evaluation
print(f'Accuracy: {accuracy_score(y_test, y_pred_log_reg) * 100:.2f}%')
print(classification_report(y_test, y_pred_log_reg))



from sklearn.ensemble import GradientBoostingClassifier

# Train a Gradient Boosting Classifier
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)

# Predictions
y_pred_gb = gb.predict(X_test)

# Model evaluation
print(f'Accuracy: {accuracy_score(y_test, y_pred_gb) * 100:.2f}%')
print(classification_report(y_test, y_pred_gb))



from sklearn.neighbors import KNeighborsClassifier

# Train a KNN classifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Predictions
y_pred_knn = knn.predict(X_test)

# Model evaluation
print(f'Accuracy: {accuracy_score(y_test, y_pred_knn) * 100:.2f}%')
print(classification_report(y_test, y_pred_knn))



import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

# Train an XGBoost classifier
xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)

# Predictions
y_pred_xgb = xgb_model.predict(X_test)

# Model evaluation
print(f'Accuracy: {accuracy_score(y_test, y_pred_xgb) * 100:.2f}%')
print(classification_report(y_test, y_pred_xgb))



import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report

# Train a LightGBM classifier
lgb_model = lgb.LGBMClassifier(random_state=42)
lgb_model.fit(X_train, y_train)

# Predictions
y_pred_lgb = lgb_model.predict(X_test)

# Model evaluation
print(f'Accuracy: {accuracy_score(y_test, y_pred_lgb) * 100:.2f}%')
print(classification_report(y_test, y_pred_lgb))



test_data=pd.read_csv('/kaggle/input/kaggle-biochallenge-cirrhosis-detection/test.csv')
test_data.head()


# Step 5: Prepare the submission file

ids = test_data['id']




test_data.isnull().sum()


test_data.info()


# For numerical columns, fill missing values with the median
numerical_cols = ['N_Days', 'Age', 'Bilirubin', 
                  'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 
                  'Tryglicerides', 'Platelets', 'Prothrombin']

for col in numerical_cols:
    test_data[col].fillna(test_data[col].median(), inplace=True)



# For categorical columns, fill missing values with the mode (most frequent value)
categorical_cols = ['Drug', 'Sex','Ascites','Hepatomegaly','Edema','Spiders']  # Add any other categorical columns

for col in categorical_cols:
    test_data[col].fillna(test_data[col].mode().iloc[0], inplace=True)



for col in categorical_cols:
    test_data[col] = le.fit_transform(test_data[col])


test_data=test_data.drop(columns=['id'])


knn.predict(test_data)


test_data1=pd.read_csv('/kaggle/input/kaggle-biochallenge-cirrhosis-detection/test.csv')


pred_probabilities =knn.predict_proba(test_data)
# Step 6: Save the submission to a CSV file
# Reorder the columns to match the required format


submission = pd.DataFrame(pred_probabilities, columns=['Status_C', 'Status_CL', 'Status_D'])
submission['id']=test_data1['id']
submission = submission[['id', 'Status_C', 'Status_CL', 'Status_D']]
submission.to_csv('submission_knn.csv', index=False)

print("Submission file created successfully using KNN!")


submission.head()




