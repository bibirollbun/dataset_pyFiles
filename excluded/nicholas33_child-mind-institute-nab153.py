#installing all the necessary packages 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler  
from sklearn.decomposition import PCA #Principle component analysis to improve model performance 


# Load datasets

train_data = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')
test_data = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv')
train_data.head()


train_data.describe().transpose()


train_data.info()


train_data['sii'].value_counts()


#select columns with more than 50% non null values and filling the missing values to ensure that the selected columns are the most accurate 
threshold = 0.5 * len(train_data)
columns_with_data = train_data.columns[train_data.isnull().sum() < threshold]
train_data = train_data[columns_with_data]
#replace all missing values with 0 
train_data = train_data.fillna(0)


#define the target column 
target_column = 'sii'
train_data_cleaned = train_data.dropna(subset=[target_column])
#check the results 
train_data_cleaned.head()
train_data_cleaned.info()


#categorical columns in the dataset 
categorical_columns = ['Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season', 'FGC-Season', 'BIA-Season', 'PCIAT-Season', 'SDS-Season', 'PreInt_EduHx-Season']
#plotting boxplots for 'sii' against each categorical column 
plt.figure(figsize=(16,24))
for i, col in enumerate(categorical_columns, 1):
    plt.subplot(4,2,i)
    sns.boxplot(x=col, y='sii', data=train_data_cleaned)
    plt.xticks(rotation=45)
    plt.title(f"'sii' vs {col}")
plt.tight_layout()
plt.show()


#plot target column 'sii' with numerical columns 
#numerical against sii 
numerical_cols = train_data_cleaned.select_dtypes(include=['float', 'int64']).columns
#set the number of plots per row 
plots_per_row = 5
n_rows = (len(numerical_cols) + plots_per_row -1) // plots_per_row
plt.figure(figsize=(20, 4 * n_rows))
for i, col in enumerate(numerical_cols): 
    plt.subplot(n_rows, plots_per_row, i + 1)
    sns.boxplot(x='sii', y=col, data=train_data_cleaned)
    plt.title(col)
    plt.tight_layout()
plt.show()


#indentify categorical columns for seasons
season_cols = [
    'basic_Demos-Enroll_Season', 
    'CGAS-Season', 
    'Physical-Season', 
    'FGC-Season', 
    'BIA-Season', 
    'PCIAT-Season', 
    'SDS-Season', 
    'PreInt_EduHx-Season'  
]
#create a mapping dict for seasons 
season_mapping = {
    'Spring': 0, 
    'Summer': 1, 
    'Fall': 2,
    'Winter': 3
}
#Apply manual encoding to the categorical columns
for col in season_cols:
    if col in train_data_cleaned.columns:
        train_data_cleaned[col] = train_data_cleaned[col].replace(season_mapping)


#drop the id column if present 
train_data_no_id = train_data_cleaned.drop(columns=['id'], errors='ignore')
train_data_numeric = train_data_no_id.select_dtypes(include=['number'])
#calculate the correlation matrix 
correlation_matrix = train_data_numeric.corr()
#plot the heatmap 
plt.figure(figsize=(30,30))
sns.heatmap(correlation_matrix, annot=True, fmt='1f', cmap='coolwarm', square=True)
plt.title('correlation heatmap')
plt.show()


#Get the intersection of columns in teh train_data_cleaned and test_data 
# âœ… Get the intersection of column names
common_columns = train_data_cleaned.columns.intersection(test_data.columns)
# âœ… Filter both DataFrames to keep only the common columns
X = train_data_cleaned[common_columns].drop(columns=['id'])
y = train_data_cleaned['sii']


# âœ… Step 2: Split Data into Training & Validation Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2) # 20% test split


from sklearn.preprocessing import StandardScaler

# âœ… Step 1: Ensure all categorical features are converted to numeric
X_train_encoded = pd.get_dummies(X_train)  # Converts categorical to numeric
X_test_encoded = pd.get_dummies(X_test)


# âœ… Step 2: Align columns (ensure train & test have the same features)
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)


# âœ… Step 3: Scale the numeric data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)


# âœ… Print shape confirmation
print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)


#Apply PCA to reduce dimensionality and simplify the dataset PCA reduces dimensionality while preserving important patterns in the data.
# âœ… Step 2: Apply PCA

pca = PCA(n_components=0.95) #Keep 95% of the variance 
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)



#Create instance of Random Forest Classifier 
from sklearn.ensemble import RandomForestClassifier

# âœ… Step 1: Create Random Forest Classifier Instance
rf_classifier = RandomForestClassifier(
    n_estimators=100,  # Number of trees in the forest
    max_depth=10,  # No depth limit (fully grown trees)
    min_samples_split=10,  # Minimum samples needed to split a node
    min_samples_leaf=4,  # Minimum samples per leaf
    random_state=42,  # Ensures reproducibility
    n_jobs=-1  # Uses all available CPU cores for faster training
)

# âœ… Print Model Parameters
print(rf_classifier)


rf_classifier.fit(X_train_pca, y_train)


# Make predictions on the test set
y_pred = rf_classifier.predict(X_test_pca)
# Generate a classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
# Generate a confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# Calculate the accuracy on the test set
accuracy = rf_classifier.score(X_test_pca, y_test)
print(f"Model Accuracy: {accuracy}")


#indentify categorical columns for seasons
season_cols = [
    'basic_Demos-Enroll_Season', 
    'CGAS-Season', 
    'Physical-Season', 
    'FGC-Season', 
    'BIA-Season', 
    'PCIAT-Season', 
    'SDS-Season', 
    'PreInt_EduHx-Season'  
]
#create a mapping dict for seasons 
season_mapping = {
    'Spring': 0, 
    'Summer': 1, 
    'Fall': 2,
    'Winter': 3
}
#Apply manual encoding to the categorical columns
for col in season_cols:
    if col in test_data.columns:
        test_data[col] = test_data[col].map(season_mapping)


# âœ… Step 1: Handle Missing Values
test_data.fillna(0, inplace=True)
# âœ… Step 2: Find Common Columns
common_columns = train_data_cleaned.columns.intersection(test_data.columns)
# âœ… Step 3: Prepare the Test Data
X_test_data = test_data[common_columns].drop(columns=['id'])  #Drop the id col 
# âœ… Step 4: One-Hot Encode Categorical Columns
X_test_encoded = pd.get_dummies(X_test_data)
# âœ… Step 5: Align Columns to Match Training Data
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
# âœ… Step 6: Scale the Test Data Using the Fitted Scaler
X_test_scaled = scaler.transform(X_test_encoded)
# âœ… Step 7: Apply PCA
X_test_pca = pca.transform(X_test_scaled)
# âœ… Step 8: Make Predictions
predictions = rf_classifier.predict(X_test_pca)
# âœ… Step 9: Create Submission File
submission = pd.DataFrame({
    'id': test_data['id'], #include the id from test_data 
    'sii': predictions  #predictions from the model 
})
submission.to_csv('submission.csv', index=False)
#Check submission data frame 
print(submission.head())




