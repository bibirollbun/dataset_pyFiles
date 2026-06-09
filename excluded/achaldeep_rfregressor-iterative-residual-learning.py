# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# supressing warnings
import warnings
warnings.simplefilter(action='ignore')

# Loading Datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")


train.shape, test.shape # How big is the data ?


train.sample(5) # How does the training data look like ?


test.sample(5) # How does the testing data look like ?


train.info(), test.info() # What is the data type of columns ?


train.isna().sum(), test.isna().sum() # Are there any missing values ?


train.isin(["?", "None", "NA", "N/A", "-", 
            " ", "", "Unknown", "UNKNOWN", 
            "Invalid"]).sum(), test.isin(["?", "None", "NA", "N/A", "-", 
                                          " ", "", "Unknown", "UNKNOWN", "Invalid"]).sum() # checking for other values


train.duplicated().sum(), test.duplicated().sum() # Are there duplicate values ?


train.nunique(), test.nunique() # Unique entries per column


# drop columns which has unique value for each row
train.drop(['id'], axis=1, inplace=True)
test.drop(['id'], axis=1, inplace=True)
train.shape, test.shape


df = train.copy(deep=True) # DataFrame for EDA
df.sample(5)


# Downcasting Data Types for EDA DataFrame

# Float64 to appropriate data type
for col in df.select_dtypes(include=['float64']).columns:
    if np.all(np.modf(df[col])[0] == 0):  # Check if all decimal parts are 0
        df[col] = pd.to_numeric(df[col], downcast='integer')
    else:
        df[col] = pd.to_numeric(df[col], downcast='float')

# Object to Category DataType
df[df.select_dtypes(include='object').columns] = df.select_dtypes(include='object').apply(lambda x: x.astype('category'))


df.describe(include=['float32']) # Summary for numerical columns


# Distribution of Number_of_Ads
plt.figure(figsize=(12, 5))
ax = sns.countplot(data=df, x='Number_of_Ads', palette="viridis")
ax.bar_label(container=ax.containers[0])
plt.title('Distribution of Number_of_Ads')
plt.xlabel('Number_of_Ads')
plt.ylabel('Count')
plt.show()


num_hist = ['Episode_Length_minutes',	'Host_Popularity_percentage',	'Guest_Popularity_percentage',	'Listening_Time_minutes']

# Define bin sizes based on data distribution
bin_sizes = {
    'Episode_Length_minutes': 10, 
    'Host_Popularity_percentage': 5,
    'Guest_Popularity_percentage': 5, 
    'Listening_Time_minutes': 5
}

# Create subplots for histograms
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(16, 20)) 
axes = axes.flatten()  # Flatten for easy iteration

# Loop through each column and plot histograms
for i, col in enumerate(num_hist):
    bins = bin_sizes.get(col, 50)  # Default to 50 if not specified
    sns.histplot(df[col], bins=bins, kde=True, ax=axes[i], color='royalblue')

    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Frequency")

# Adjust layout
plt.tight_layout()
plt.show()


# Episode_Length_minutes vs Listening_Time_minutes
sns.scatterplot(data=df, x='Episode_Length_minutes', y='Listening_Time_minutes')
plt.title('Scatter Plot of Episode_Length_minutes vs Listening_Time_minutes')
plt.show()

# sns.scatterplot(
#     data=df,
#     x='Episode_Length_minutes',
#     y='Listening_Time_minutes',
#     hue='Number_of_Ads', # first remove outliers from this column
#     palette='viridis'  # or 'coolwarm', 'magma', etc.
# )
# plt.title('Episode Length vs Listening Time (Colored by No. of Ads)')
# plt.show()


# Host_Popularity_percentage vs Guest_Popularity_percentage
sns.scatterplot(data=df, x='Host_Popularity_percentage', y='Guest_Popularity_percentage')
plt.title('Scatter Plot of Host_Popularity_percentage vs Guest_Popularity_percentage')
plt.show()


num_continuous_cols = df.select_dtypes(include=['float32']).columns

fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(16, 24))
fig.suptitle("Boxplots of Numerical Columns", fontsize=14)

# Flatten axes array for easy iteration
axes = axes.flatten()

# Loop through each column and plot
for i, col in enumerate(num_continuous_cols):
    sns.boxplot(x=df[col], ax=axes[i])
    axes[i].set_title(f"Boxplot of {col}")

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Keeps suptitle from overlapping
plt.show()


df.corr(numeric_only=True) # Numerical correlations


# Correlation Heatmap
sns.heatmap(df.corr(numeric_only=True),annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


df.describe(include=['category']) # Summary for categorical columns


# Distribution of Podcast_Name
plt.figure(figsize=(20, 6))
ax = sns.countplot(data=df, x='Podcast_Name', palette="magma")
ax.bar_label(container=ax.containers[0])
plt.title('Distribution of Podcast_Name')
plt.xlabel('Podcast_Name')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.show()


# Distribution of Episode_Title
plt.figure(figsize=(26, 8))
ax = sns.countplot(data=df, x='Episode_Title', palette="magma")
ax.bar_label(container=ax.containers[0])
plt.title('Distribution of Episode_Title')
plt.xlabel('Episode_Title')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.show()


# Distribution of Genre
ax = sns.countplot(data=df, x='Genre', palette="magma")
ax.bar_label(container=ax.containers[0])
plt.title('Distribution of Genre')
plt.xlabel('Genre')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.show()


# Pie Chart of Publication_Day Column
col = 'Publication_Day'
df[col].value_counts().plot.pie(autopct='%1.2f%%', figsize=(6, 6), title=f'Distribution of {col} Column')
plt.ylabel("")  # Optional: remove y-axis label
plt.tight_layout()
plt.show()


# Pie Chart of Publication_Time Column
col = 'Publication_Time'
df[col].value_counts().plot.pie(autopct='%1.2f%%', figsize=(6, 6), title=f'Distribution of {col} Column')
plt.ylabel("")  # Optional: remove y-axis label
plt.tight_layout()
plt.show()


# Pie Chart of Episode_Sentiment Column
col = 'Episode_Sentiment'
df[col].value_counts().plot.pie(autopct='%1.2f%%', figsize=(6, 6), title=f'Distribution of {col} Column')
plt.ylabel("")  # Optional: remove y-axis label
plt.tight_layout()
plt.show()


def data_cleaning(df, train_medians=None):
    df = df.copy()

    # Invalid Values Handling
    df['Number_of_Ads'] = df['Number_of_Ads'].clip(upper=3.0)
    cols_to_cap = ['Host_Popularity_percentage', 'Guest_Popularity_percentage']
    df[cols_to_cap] = df[cols_to_cap].clip(upper=100)
    df['Episode_Length_minutes'] = df['Episode_Length_minutes'].clip(upper=119.98)

    # Handling Missing Values
    cols = ['Guest_Popularity_percentage', 'Episode_Length_minutes']
    if train_medians is None:
        # Calculate medians if training data
        train_medians = {col: df[col].median() for col in cols}
        
    for col in cols:
        df[col] = df[col].fillna(train_medians[col])

    # Feature Engineering
    df['Episode_Number'] = df['Episode_Title'].str.extract(r'(\d+)', expand=False).astype(int)
    df.drop(columns=['Episode_Title'], inplace=True)
    
    return df, train_medians  # Return both cleaned DF and medians

train, medians = data_cleaning(train)
test, _ = data_cleaning(test, train_medians=medians)  # Use train's medians for test


# One Hot Encoding 
ohe_cols = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time']

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe_train = ohe.fit_transform(train[ohe_cols])
ohe_test = ohe.transform(test[ohe_cols])

ohe_feature_names = ohe.get_feature_names_out(ohe_cols)
ohe_train_df = pd.DataFrame(ohe_train, columns=ohe_feature_names, index=train.index)
ohe_test_df = pd.DataFrame(ohe_test, columns=ohe_feature_names, index=test.index)

train = train.drop(ohe_cols, axis=1)
test = test.drop(ohe_cols, axis=1)
train = pd.concat([train, ohe_train_df], axis=1)
test = pd.concat([test, ohe_test_df], axis=1)

# Ordinal Encoding
ord_encoding = ['Episode_Sentiment']
sentiment_order = [['Negative', 'Neutral', 'Positive']]
oe = OrdinalEncoder(categories=sentiment_order)
train[ord_encoding] = oe.fit_transform(train[ord_encoding])
test[ord_encoding] = oe.transform(test[ord_encoding])


# splitting data
X = train.drop('Listening_Time_minutes', axis=1)
y = train['Listening_Time_minutes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# downcasting the datatypes

# Float64 to appropriate data type
for col in X_train.select_dtypes(include=['float64']).columns:
    if np.all(np.modf(X_train[col].values)[0] == 0):  # Check if all decimal parts are 0
        X_train[col] = pd.to_numeric(X_train[col], downcast='integer')
        X_test[col] = pd.to_numeric(X_test[col], downcast='integer')
        test[col] = pd.to_numeric(test[col], downcast='integer')
    else:
        X_train[col] = pd.to_numeric(X_train[col], downcast='float')
        X_test[col] = pd.to_numeric(X_test[col], downcast='float')
        test[col] = pd.to_numeric(test[col], downcast='float')

# Int64 to appropriate data types
for col in X_train.select_dtypes(include=['int64']).columns:
    X_train[col] = pd.to_numeric(X_train[col], downcast='integer')
    X_test[col] = pd.to_numeric(X_test[col], downcast='integer')
    test[col] = pd.to_numeric(test[col], downcast='integer')


# Iterative Residual Learning
model_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', RandomForestRegressor(random_state=42))
])
model_pipeline.fit(X_train, y_train)
predictions_train = model_pipeline.predict(X_train)
predictions_test = model_pipeline.predict(X_test)
print("Initial model training done")
    
models = [model_pipeline]  # To store all models

rmse = np.sqrt(mean_squared_error(y_test, predictions_test))
print(f"RandomForestRegressor Test RMSE: {rmse:.5f}")


# Residual learning loop (3 iterations)
for i in range(1, 4):
    residuals = y_train - predictions_train

    # Train on residuals
    residual_model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestRegressor(random_state=42))
    ])
    residual_model.fit(X_train, residuals)

    # Predict residuals and update predictions
    predictions_train += residual_model.predict(X_train)
    predictions_test += residual_model.predict(X_test)

    models.append(residual_model)
    print(f"Iteration {i} complete")

    rmse = np.sqrt(mean_squared_error(y_test, predictions_test))
    print(f"RandomForestRegressor Test RMSE with Iterative Residual Learning {i} : {rmse:.5f}")


y_pred_test = np.zeros(test.shape[0])

# Add predictions from each model in the order they were trained
for i, model in enumerate(models):
    step_preds = model.predict(test)
    y_pred_test += step_preds
    print(f"Model {i+1} predictions added")


# Submission
sample['Listening_Time_minutes'] = y_pred_test
sample['Listening_Time_minutes'] = sample['Listening_Time_minutes'].clip(lower=0)

sample.to_csv("submission.csv", index=False)

