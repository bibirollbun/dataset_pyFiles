import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# import library
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve


df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
df.info()


# Drop the 'ID' column
df = df.drop(columns=['id'])


# Check the range of numeric variables
# List of numeric variables
numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# Check min max for each numeric variable
for feature in numeric_features:
    min_feature = df[feature].min()
    max_feature = df[feature].max()
    print(f"{feature}: Min: {min_feature}, Max: {max_feature}")


# Filter out only numeric columns
num_col = df.select_dtypes(include=['number'])

# Calculate correlation matrix
corr_matrix = num_col.corr()

# create a heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap='BuPu', fmt=".2f")
plt.title('Correlation Heatmap of Variables')
plt.show()


# Drop the 'previous' due to high correlation between features
df = df.drop(columns=['previous'])


# Separate '-1' from pdays

# New binary column: was the client contacted before?
df['pdays_contacted_or_not'] = df['pdays'].apply(lambda x: 0 if x == -1 else 1)

# Replace -1 with a 0 for original column
df['pdays'] = df['pdays'].replace(-1, 0)


# List of continuous variables
numeric_features_2 = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays']

# Define plot color
hist_colors = sns.color_palette("Set2", len(numeric_features_2))

# Plot the histograms
for i, col in enumerate(numeric_features_2):
  if df[col].dtype in ['int64', 'float64']:
    plt.figure(figsize=(4, 3))
    plt.hist(df[col], color=hist_colors[i], bins=30, alpha=0.5)
    plt.title(col)
    plt.show()


# Bin 'day' to early/mid/late month
df['day_bin'] = pd.cut(df['day'], bins=[0, 10, 20, 31], labels=['early', 'mid', 'late'])

# Drop 'day'
df = df.drop(columns=['day'])


# Bin 'age' 
df['age_bin'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 65, 100],
                       labels=['<25', '25-35', '35-45', '45-55', '55-65', '65+'])

# Drop 'age'
df = df.drop(columns=['age'])


# Remove outlier mannually based on histogram insight
# For feature 'balance', 'duration', 'campaign', 'pdays'

df = df[(df['balance'] <= 15000) & (df['duration'] <= 1500) & (df['campaign'] <= 12) & (df['pdays'] <= 400)]


# Check unique value of categorical variables
# List of categorical variables
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']

# Check unique values for each categorical variable
for feature in categorical_features:
    unique_features = df[feature].unique()
    print(f"Unique values for {feature}: {unique_features}")


# Bin months to seasons
month_to_season = {
    'dec': 'winter', 'jan': 'winter', 'feb': 'winter',
    'mar': 'spring', 'apr': 'spring', 'may': 'spring',
    'jun': 'summer', 'jul': 'summer', 'aug': 'summer',
    'sep': 'fall', 'oct': 'fall', 'nov': 'fall'
}
df['season'] = df['month'].map(month_to_season)

# Drop 'month'
df = df.drop(columns=['month'])


# View 'job'

plt.figure(figsize=(4, 3))
sns.countplot(x='job', data=df, order=df['job'].value_counts().index)
plt.xticks(rotation=90)
plt.xlabel('Job')
plt.ylabel('Count')
plt.show()


# Group 'job' categories with <3% into 'other'
job_counts = df['job'].value_counts(normalize=True)
rare_jobs = job_counts[job_counts < 0.03].index
df['job'] = df['job'].apply(lambda x: 'other' if x in rare_jobs else x)

# View 'job'
plt.figure(figsize=(4, 3))
sns.countplot(x='job', data=df, order=df['job'].value_counts().index)
plt.xticks(rotation=90)
plt.xlabel('Job')
plt.ylabel('Count')
plt.show()


# Data splitting
X = df.drop(columns=['y']) # features
y = df['y'] # target variable

# Split into at 70-30 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Identify categorical columns
cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()


# Train CatBoost
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    eval_metric='AUC',
    verbose=100,
    random_state=42
)

model.fit(X_train, y_train, cat_features=cat_features, 
          eval_set=(X_test, y_test), early_stopping_rounds=50)


# Calculate ROC AUC score
y_prob = model.predict_proba(X_test)[:, 1]
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))


# Prepare to submit test set
df_test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
df_test.info()


# Preprocess df_test

# Keep the 'ID' column separate
id_test = df_test['id']  

# Drop the 'ID' & 'previous' column
df_test = df_test.drop(columns=['id', 'previous'])

# Preprocess 'pdays'
df_test['pdays_contacted_or_not'] = df_test['pdays'].apply(lambda x: 0 if x == -1 else 1)
df_test['pdays'] = df_test['pdays'].replace(-1, 0)

# Bin 'day', 'age' & 'month'
df_test['day_bin'] = pd.cut(df_test['day'], bins=[0, 10, 20, 31], labels=['early', 'mid', 'late'])
df_test['age_bin'] = pd.cut(df_test['age'], bins=[0, 25, 35, 45, 55, 65, 100],
                            labels=['<25', '25-35', '35-45', '45-55', '55-65', '65+'])
df_test['season'] = df_test['month'].map(month_to_season)
df_test = df_test.drop(columns=['day', 'age', 'month'])

# Not removing outlier in test data

# Preprocess 'job'
df_test['job'] = df_test['job'].apply(lambda x: 'other' if x in rare_jobs else x)


# Align test set columns with training data
X_final_test = df_test.copy()
X_final_test = X_final_test[model.feature_names_]


# Predict probabilities
y_prob_test = model.predict_proba(X_final_test)[:, 1]


# Create dataframe with 'ID' and 'y' columns
output = pd.DataFrame({'id': id_test, 'y': y_prob_test})
output.head()


# Save to CSV
output.to_csv('submission.csv', index=False)

