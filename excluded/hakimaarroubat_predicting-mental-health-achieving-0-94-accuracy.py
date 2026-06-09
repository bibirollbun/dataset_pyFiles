import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score, f1_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb


train = pd.read_csv("/kaggle/input/playground-series-s4e11/train.csv")


print("\nğŸ”� First 5 Rows of the Dataset:")
print(train.head())

print("\nğŸ“Š Dataset Information:")
train.info()

print("\nâ�“ Checking Missing Values:")
print(train.isnull().sum())

# ğŸ”„ Checking for Duplicates**

print("\nğŸ“Œ Number of Duplicate Rows:", train.duplicated().sum())


# Count the total number of students and working professionals
total_students = train[train['Working Professional or Student'] == 'Student'].shape[0]
total_professionals = train[train['Working Professional or Student'] == 'Working Professional'].shape[0]

# Print total counts
print(f"Total Students: {total_students}")
print(f"Total Working Professionals: {total_professionals}")

# Group by 'Working Professional or Student' and calculate missing values as percentage for each column
missing_values_by_status = train.groupby('Working Professional or Student').apply(
    lambda x: x.isnull().sum() / x.shape[0] * 100  # Calculate percentage of missing values
)

# Format the output to show count (percentage) for each column in both status categories
def format_missing_values(status):
    formatted = []
    for column in train.columns:
        missing_count = train[train['Working Professional or Student'] == status][column].isnull().sum()
        missing_percentage = (missing_count / train[train['Working Professional or Student'] == status].shape[0]) * 100
        formatted.append(f"{missing_count} ({missing_percentage:.2f}%)")
    return formatted

# Create a DataFrame to display the formatted results
missing_values_display = pd.DataFrame({
    'Column': train.columns,
    'Student': format_missing_values('Student'),
    'Working Professional': format_missing_values('Working Professional')
})

# Display the table
missing_values_display


class DataCleaner:
    def __init__(self, df):
        """Initialize with a dataset."""
        self.df = df.copy()

    def clean_data(self):
        """Apply all data cleaning steps efficiently."""
        
        # Handling 'Profession' column
        self.df.loc[self.df['Working Professional or Student'] == 'Student', 'Profession'] = self.df.loc[self.df['Working Professional or Student'] == 'Student', 'Profession'].fillna('Student')
        self.df.dropna(subset=['Profession'], inplace=True)

        # Handling 'Academic Pressure' & 'Work Pressure'
        for col, role in [('Academic Pressure', 'Working Professional'), ('Work Pressure', 'Student')]:
            self.df.loc[self.df['Working Professional or Student'] == role, col] = self.df.loc[self.df['Working Professional or Student'] == role, col].fillna(0)
            self.df.dropna(subset=[col], inplace=True)

        # Handling 'CGPA', 'Study Satisfaction', and 'Job Satisfaction'
        for col in ['CGPA', 'Study Satisfaction']:
            self.df.loc[self.df['Working Professional or Student'] == 'Working Professional', col] = self.df.loc[self.df['Working Professional or Student'] == 'Working Professional', col].fillna(0)
            self.df.dropna(subset=[col], inplace=True)

        self.df.loc[self.df['Working Professional or Student'] == 'Student', 'Job Satisfaction'] = self.df.loc[self.df['Working Professional or Student'] == 'Student', 'Job Satisfaction'].fillna(0)
        self.df.dropna(subset=['Job Satisfaction'], inplace=True)

        # Removing negligible missing values
        cols_to_drop = ['Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?', 'Work/Study Hours', 
                        'Financial Stress', 'Family History of Mental Illness']
        self.df.dropna(subset=cols_to_drop, inplace=True)

        return self.df



C = DataCleaner(train)
train_c = C.clean_data()
train_c.info()


# Plot the number of people by depression class
plt.figure(figsize=(6, 4))
sns.countplot(data=train_c, y='Depression', palette='Blues')


plt.figure(figsize=(6, 5))
sns.histplot(data=train, x='Working Professional or Student', hue='Depression', multiple='stack', shrink=0.8)
plt.title('Students vs Professionals and Depression')
plt.show()


# Convert age into categorical bins
train_c['Age Group'] = pd.cut(train['Age'], bins=[20, 30, 40, 50, 60], labels=['20-30', '30-40', '40-50', '50-60'])

# Stacked bar chart
plt.figure(figsize=(8, 5))
sns.histplot(data=train_c, x='Age Group', hue='Depression', multiple='stack', shrink=0.8)
plt.title('Age Group vs Depression')
plt.show()

plt.figure(figsize=(6, 5))
sns.histplot(data=train_c, x='Gender', hue='Depression', multiple='stack', shrink=0.8)
plt.title('Gender vs Depression')
plt.show()




def plot_sleep_distribution(train_c):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=train_c, x='Sleep Duration', order=train_c['Sleep Duration'].value_counts().index, palette='viridis')
    plt.xlabel("Sleep Duration")
    plt.ylabel("Number of Observations")
    plt.title("Distribution of Sleep Duration")
    plt.xticks(rotation=45)
    plt.show()

    # Unique values
    unique_counts = train_c['Sleep Duration'].value_counts()
    print("Number of unique values for each sleep duration category:")
    print(unique_counts)

# Call the function with the dataset
plot_sleep_distribution(train_c)


# Filter professionals only (excluding students)
professionals = train_c.dropna(subset=["Profession", "Work Pressure"])

# Select the top 10 most common professions
top_10_professions = professionals["Profession"].value_counts().nlargest(10).index
top_professionals = professionals[professionals["Profession"].isin(top_10_professions)]

# Compute the average stress level and depression rate per profession
profession_stats = top_professionals.groupby("Profession").agg(
    Avg_Stress=("Work Pressure", "mean"),
    Depression_Rate=("Depression", lambda x: x.mean() * 100)  # Depression percentage
).reset_index()

# Heatmap
plt.figure(figsize=(10, 6))
heatmap_data = profession_stats.set_index("Profession")
sns.heatmap(heatmap_data, cmap="coolwarm", annot=True, fmt=".1f")

plt.title("Stress & Depression Rate in Top 10 Professions")
plt.show()



# Countplot for cities vs depression
plt.figure(figsize=(12, 6))
sns.countplot(data=train_c, x='City', hue='Depression', palette='coolwarm')
plt.title('City-wise Count and Depression Distribution')
plt.xticks(rotation=90)  # Rotate city labels for better readability
plt.xlabel('City')
plt.ylabel('Count of Entries')
plt.show()


# Count the number of depressed vs non-depressed people per city
top_10_cities = train_c.groupby('City')['Depression'].value_counts().unstack().fillna(0)

# Sort cities by the total number of depressions in descending order
top_10_cities = top_10_cities[1].sort_values(ascending=False).head(10)

# Filter the top 10 cities data
top_10_cities_data = train_c[train_c['City'].isin(top_10_cities.index)]

# Grouped bar chart for top 10 cities showing depression vs non-depression
plt.figure(figsize=(12, 6))
sns.countplot(data=top_10_cities_data, x='City', hue='Depression', palette='coolwarm', order=top_10_cities.index)
plt.title('Top 10 Cities with the Highest Depression Counts (Descending Order)')
plt.xlabel('City')
plt.ylabel('Count of Entries')
plt.xticks(rotation=90)  # Rotate city labels for better readability
plt.show()


# Split the data into students and professionals
students_df = train_c[train_c['Working Professional or Student'] == 'Student'] 
professionals_df = train_c[train_c['Working Professional or Student'] == 'Working Professional']

# Select the numerical columns of interest
cols = ['Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction', 'Financial Stress', 'Depression']

# Compute the correlation matrices for each group
corr_students = students_df[cols].corr()
corr_professionals = professionals_df[cols].corr()

# Create the comparative plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Correlation Comparison: Students vs Professionals", fontsize=16)

# Heatmap for students
sns.heatmap(corr_students, annot=True, cmap='coolwarm', linewidths=0.5, ax=axes[0])
axes[0].set_title("Correlations for Students")

# Heatmap for professionals
sns.heatmap(corr_professionals, annot=True, cmap='coolwarm', linewidths=0.5, ax=axes[1])
axes[1].set_title("Correlations for Professionals")

plt.show()


# Create a copy of train_c and drop 'id' and 'Name' columns from the copy
train_p = train_c.copy()  # Make a copy of the original DataFrame
train_p.drop(columns=['id', 'Name'], inplace=True)  # Drop 'id' and 'Name' in the copy


# Select numerical features
numerical_cols = ['Age', 'Work Pressure','Academic Pressure','Job Satisfaction','Study Satisfaction', 'Financial Stress']

n_cols = len(numerical_cols)
n_rows = math.ceil(n_cols / 2)  # for 2 columns per row

plt.figure(figsize=(12, 6 * n_rows))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(n_rows, 2, i)  # 2 columns per row
    sns.boxplot(x=train_p[col], palette="Set2")
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()



# Modified function to return outliers
def get_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    return outliers

# Dictionary to store outliers for each numerical column
outliers_dict = {}

# Identify outliers for all numerical columns
for col in numerical_cols:
    outliers_dict[col] = get_outliers(train_p, col)

# Create a DataFrame to summarize the outliers
outliers_summary = pd.DataFrame({
    'Variable': [col for col in numerical_cols],
    'Number of Outliers': [len(outliers_dict[col]) for col in numerical_cols]
})

# Display the summary table
print(outliers_summary)


def detect_categorical_outliers(df, categorical_cols, threshold=0.01):
    """
    Detects potential outliers in categorical columns based on their frequency distribution.
    Categorical values that appear less than the given threshold (as a percentage of total records)
    will be flagged as potential outliers.
    
    :param df: DataFrame
    :param categorical_cols: List of categorical columns to check for outliers
    :param threshold: Minimum percentage frequency to consider a category as non-outlier
    :return: Dictionary with categorical column and their outliers
    """
    outliers = {}
    
    for col in categorical_cols:
        freq = df[col].value_counts(normalize=True)
        rare_categories = freq[freq < threshold].index.tolist()
        
        if rare_categories:
            outliers[col] = rare_categories
    
    return outliers

def handle_categorical_outliers(df, categorical_cols, threshold=0.01):
    """
    Replace rare categories in categorical columns with 'Other' if they appear less frequently than the threshold.
    """
    outliers = detect_categorical_outliers(df, categorical_cols, threshold)
    
    for col, rare_categories in outliers.items():
        df[col] = df[col].apply(lambda x: 'Other' if x in rare_categories else x)




# List of categorical columns
categorical_cols = ['Gender', 'City', 'Profession', 'Degree', 'Dietary Habits', 'Sleep Duration', 'Family History of Mental Illness', 'Working Professional or Student','Have you ever had suicidal thoughts ?' ]

# Detect categorical outliers
outliers = detect_categorical_outliers(train_p, categorical_cols, threshold=0.01)

# Show the potential outliers
outliers



handle_categorical_outliers(train_p, categorical_cols, threshold=0.01)
detect_categorical_outliers(train_p, categorical_cols, threshold=0.01)


def encode_categorical_data(df_train, df_test, categorical_cols, target_col, ordinal_cols=None):
    """
    Encodes categorical columns into numerical format using one-hot encoding or label encoding.
    The target column is removed before encoding to prevent data leakage.
    Additionally, boolean columns are converted to integers (True = 1, False = 0).
    """
    # Separate the target column from the training dataset
    y_train = df_train[target_col]
    df_train = df_train.drop(columns=[target_col])



    # Label Encoding for ordinal columns
    if ordinal_cols:
        for col in ordinal_cols:
            label_encoder = LabelEncoder()
            df_train[col] = label_encoder.fit_transform(df_train[col])

            # Handle unseen categories in the test set by assigning -1
            df_test[col] = df_test[col].apply(lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1)

    # One-Hot Encoding for nominal columns
    df_train_encoded = pd.get_dummies(df_train, columns=categorical_cols, drop_first=True)
    df_test_encoded = pd.get_dummies(df_test, columns=categorical_cols, drop_first=True)

    # Align columns of train and test sets (ensuring the same structure)
    for col in df_train_encoded.columns:
        if col not in df_test_encoded.columns:
            df_test_encoded[col] = 0  # Add missing columns in test with 0

    df_test_encoded = df_test_encoded[df_train_encoded.columns]  # Reorder test columns to match train
    
    # Convert boolean columns to integers (True = 1, False = 0)
    for col in df_train_encoded.select_dtypes(include=[bool]).columns:
        df_train_encoded[col] = df_train_encoded[col].astype(int)

    for col in df_test_encoded.select_dtypes(include=[bool]).columns:
        df_test_encoded[col] = df_test_encoded[col].astype(int)
    return df_train_encoded, y_train, df_test_encoded



test = pd.read_csv("/kaggle/input/playground-series-s4e11/test.csv")
C_test = DataCleaner(test)
test_c = C_test.clean_data()
test_ids = test_c[['id', 'Name']].copy()
test_p = test_c.copy()
test_p.drop(columns=['id', 'Name'], inplace=True)
handle_categorical_outliers(test_p, categorical_cols, threshold=0.01)


# For this example, treat 'Degree' and 'Sleep Duration' as ordinal columns
ordinal_cols = ['Degree', 'Sleep Duration']  # Sleep Duration is now considered ordinal

# Apply the encoding function
train_num, y, test_num =encode_categorical_data(df_train = train_p, df_test =  test_p, categorical_cols = categorical_cols, target_col = 'Depression', ordinal_cols=None)


# Force the types to be identical in both datasets
for col in train_num.columns:
    if col in test_num.columns:
        test_num[col] = test_num[col].astype(train_num[col].dtype)


set(train_num.columns) - set(test_num.columns)  # Columns in train but not in test
set(test_num.columns) - set(train_num.columns)  # Columns in test but not in train



train_num.info()
test_num.info()


# Remove the "Age Group" column from the training and testing datasets
train_num = train_num.drop(columns=['Age Group'])
test_num = test_num.drop(columns=['Age Group'])


train_num.info()
test_num.info()


# Select numerical columns
numerical_cols = train_num.select_dtypes(include=['int64', 'float64']).columns

# Initialize the scaler (Choose one: StandardScaler or MinMaxScaler)
scaler = StandardScaler()  # Standardization (zero mean, unit variance)
# scaler = MinMaxScaler()  # Normalization (scales between 0 and 1)

# Fit on training data and transform both train and test sets
train_num[numerical_cols] = scaler.fit_transform(train_num[numerical_cols])
test_num[numerical_cols] = scaler.transform(test_num[numerical_cols])

# Check the transformed dataset
train_num.head()


# Split the training dataset into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_num, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape}, Validation set size: {X_val.shape}")


# start with a simple Logistic Regression model to establish a baseline.
# Train a Logistic Regression model
model = LogisticRegression(max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
print(f"Baseline Model Accuracy: {accuracy:.4f}")


# We can now experiment with different models like Random Forest and XGBoost.

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Train an XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Evaluate both models
rf_accuracy = accuracy_score(y_val, rf_model.predict(X_val))
xgb_accuracy = accuracy_score(y_val, xgb_model.predict(X_val))



# To improve performance, let's use GridSearchCV for hyperparameter tuning.
# Define hyperparameters for tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

# Grid search for Random Forest
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Accuracy: {grid_search.best_score_:.4f}")



# âœ… Add class_weight='balanced' in LogisticRegression
# This automatically adjusts the importance of the classes. 

model_bl = LogisticRegression(class_weight='balanced', random_state=42)
model_bl.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model_bl.predict(X_val)

# Evaluate the model
accuracy_mod_bl = accuracy_score(y_val, y_pred)
print(f"Baseline Model Accuracy: {accuracy_mod_bl:.4f}")


# Get feature importance from Random Forest
feature_importance = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 5))
feature_importance[:10].plot(kind='bar')
plt.title("Top 10 Most Important Features")
plt.show()


# Predictions
y_pred1 = model.predict(X_val) # LogisticRegression
y_pred2 = rf_model.predict(X_val) # RandomForestClassifier
y_pred3 = xgb_model.predict(X_val) # XGBoost

# Calculate AUC-ROC
roc_auc1 = roc_auc_score(y_val, y_pred1)
roc_auc2 = roc_auc_score(y_val, y_pred2)
roc_auc3 = roc_auc_score(y_val, y_pred3)

# Calculate F1-Score
f1_score1 = f1_score(y_val, y_pred1)
f1_score2 = f1_score(y_val, y_pred2)
f1_score3 = f1_score(y_val, y_pred3)

# Print the results
print("Model LogisticRegression AUC-ROC: {:.4f}".format(roc_auc1))
print("Model RandomForestClassifier AUC-ROC: {:.4f}".format(roc_auc2))
print("Model XGBoost AUC-ROC: {:.4f}".format(roc_auc3))

print("Model LogisticRegression F1-Score: {:.4f}".format(f1_score1))
print("Model RandomForestClassifier F1-Score: {:.4f}".format(f1_score2))
print("Model XGBoost F1-Score: {:.4f}".format(f1_score3))


#âœ… Add class_weight='balanced' in LogisticRegression and train it
lg_model_bl = LogisticRegression(class_weight='balanced', random_state=42)
lg_model_bl.fit(X_train, y_train)

# do the same for Random Forest model
rf_model_bl = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_model_bl.fit(X_train, y_train)

# âœ… Add scale_pos_weight for XGBoost model
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])  # 80/20 = 4
xgb_model_bl = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42)
xgb_model_bl.fit(X_train, y_train)

# Evaluate both models
lg_model_bl = accuracy_score(y_val, lg_model_bl.predict(X_val))
rf_accuracy_bl = accuracy_score(y_val, rf_model_bl.predict(X_val))
xgb_accuracy_bl = accuracy_score(y_val, xgb_model_bl.predict(X_val))



print(f"Logistic Regression Accuracy (Balanced): {lg_model_bl:.4f}")
print(f"Random Forest Accuracy (Balanced): {rf_accuracy_bl:.4f}")
print(f"XGBoost Accuracy (Balanced): {xgb_accuracy_bl:.4f}")


#Making Predictions on the Test Set

# Predict on the test set
final_predictions = rf_model_bl.predict(test_num)

# Convert predictions into a DataFrame
submission = pd.DataFrame({'id': test_ids['id'], 'Depression': final_predictions})

# Save predictions
submission.to_csv('final_predictions.csv', index=False)

print("Predictions saved successfully!")


submission.head()

