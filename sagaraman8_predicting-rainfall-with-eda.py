# For data cleaning and visualisation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# This one hides warnings for better output vies
import warnings
warnings.filterwarnings('ignore')

# For data preprocessing
from sklearn.preprocessing import StandardScaler

# For feature reduction
from sklearn.decomposition import PCA

# Some machine learning models
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# For Matrics Evaluation
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
train_df.head()


print("The shape of Training Data: ", train_df.shape)
print("The shape of Test Data: ", test_df.shape)


print("Training Data Information: ")
print(train_df.info())
print("Test Data Information: ")
print(test_df.info())


print("Missing value in Training Data: ", train_df.isnull().sum().sum())
print("Missing value in Test Data: ", test_df.isnull().sum().sum())


sns.histplot(x='winddirection', data=train_df, kde=True)
plt.show()


test_df['winddirection'] = test_df['winddirection'].fillna(test_df['winddirection'].mode()[0])

print("Missing values in Test Data: ", test_df.isnull().sum().sum())
print("Missing values in Train Data: ", train_df.isnull().sum().sum())


def boxplot(df):
  cols_to_plot = [col for col in df.columns if col not in ['id', 'rainfall','day']]

  n_cols = 4
  n_rows = 3

  fig, axes = plt.subplots(n_rows, n_cols, figsize=(15,10))
  axes = axes.flatten()

  for i, col in enumerate(cols_to_plot):
      sns.boxplot(x=col, data=train_df, ax=axes[i])
      axes[i].set_title(f'Boxplot of {col}')
      axes[i].set_xlabel(col)

  for j in range(i + 1, len(axes)):
      fig.delaxes(axes[j])

  plt.tight_layout()
  plt.show()

print("Boxplots for Training Data:")
boxplot(train_df)


print("\nBoxplots for Test Data:")
boxplot(test_df)


# creates a list of column names from the training dataframe, excluding 'id', 'day', and 'rainfall'.
# These are the columns where outliers will be handled.
cols_to_handle_outliers = [col for col in train_df.columns if col not in ['id', 'day', 'rainfall']]

# Iterates through each column in the cols_to_handle_outliers list
for col in cols_to_handle_outliers:

    # Calculates the first quartile (25th percentile) for the current column in the training data
    Q1 = train_df[col].quantile(0.25)

    # Calculates the third quartile (75th percentile) for the current column in the training data.
    Q3 = train_df[col].quantile(0.75)

    # Calculates the Interquartile Range (IQR), which is the difference between the third and first quartiles.
    IQR = Q3 - Q1

    # Determines the lower bound for outlier detection. Any value below this bound is considered an outlier.
    lower_bound = Q1 - 1.5 * IQR

    # Determines the upper bound for outlier detection. Any value above this bound is considered an outlier.
    upper_bound = Q3 + 1.5 * IQR

    # Applies capping to the current column in the training.
    # Values below the lower_bound are replaced with the lower_bound, and
    # values above the upper_bound are replaced with the upper_bound.
    train_df[col] = train_df[col].clip(lower=lower_bound, upper=upper_bound)
    test_df[col] = test_df[col].clip(lower=lower_bound, upper=upper_bound)

print("Outliers handled using IQR-based capping for the following columns:")
print(cols_to_handle_outliers) # Gives the list of columns in which capping where applied.


print("Boxplots for Training Data after handling outliers:")
boxplot(train_df)


print("Boxplots for Test Data after handling outliers:")
boxplot(test_df)


cols_to_visualize = [col for col in train_df.columns if col not in ['id', 'day', 'rainfall']]
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(cols_to_visualize):
    sns.histplot(train_df[col], ax=axes[i], kde=True)
    axes[i].set_title(f'Distribution of {col}')

for j in range(i + 1, len(axes)):
      fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# Calculate the correlation matrix
numerical_cols = train_df.select_dtypes(include=np.number).columns.tolist()
cols_for_corr = [col for col in numerical_cols if col not in ['id', 'day']]
correlation_matrix = train_df[cols_for_corr].corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()


cols_for_pariplot = ['pressure','temparature','maxtemp','mintemp','dewpoint','winddirection']

sns.pairplot(train_df[cols_for_pariplot], diag_kind='kde')
plt.title("Pair plot of selected columns")
plt.show()


# Assuming the day 1 is the start of the year
# Create a list of days in each month (non-leap year)
days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

day_to_month = {}
current_day = 0
for month, days in enumerate(days_in_month, 1):
    for day in range(days):
        current_day += 1
        day_to_month[current_day] = month

# Function to map day to month
def get_month(day):
    return day_to_month.get(day, None)

# Apply the function to create the 'month' column in both dataframes
train_df['month'] = train_df['day'].apply(get_month)
test_df['month'] = test_df['day'].apply(get_month)

print("Training data with 'month' column:")
display(train_df.head())

print("\nTest data with 'month' column:")
display(test_df.head())


print("Unique values in month column in Training data: ",train_df['month'].unique())
print("Unique values in month column in Test data: ",test_df['month'].unique())


train_df['avg_temp'] = (train_df['maxtemp'] + train_df['mintemp'] + train_df['temparature']) / 3
test_df['avg_temp'] = (test_df['maxtemp'] + test_df['mintemp'] + test_df['temparature']) / 3

print("Training data with 'avg_temp' column:")
display(train_df.head())

print("\nTest data with 'avg_temp' column:")
display(test_df.head())


# Define the bins and labels for sunshine categories
# Assuming sunshine is in hours
bins = [-0.1, 0.1, 3, 7, train_df['sunshine'].max()] # -0.1 to 0.1 for 'no sun', then thresholds for others
labels = ['no sun', 'low sun', 'moderate sun', 'high sun']

# Create the 'sunshine_category' column using pd.cut
train_df['sunshine_category'] = pd.cut(train_df['sunshine'], bins=bins, labels=labels, right=True)
test_df['sunshine_category'] = pd.cut(test_df['sunshine'], bins=bins, labels=labels, right=True)

print("Training data with 'sunshine_category' column:")
display(train_df.head())

print("\nTest data with 'sunshine_category' column:")
display(test_df.head())


scaler = StandardScaler()

num_cols = [col for col in train_df.columns if col not in ['id','day','rainfall','month'] and train_df[col].dtype in ['float64', 'int64']]

train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
test_df[num_cols] = scaler.transform(test_df[num_cols])

print("Training data after scaling:")
display(train_df.head())

print("\nTest data after scaling:")
display(test_df.head())



train_df = pd.get_dummies(train_df, columns=['month','sunshine_category'], prefix='month', drop_first=True)
test_df = pd.get_dummies(test_df, columns=['month','sunshine_category'], prefix='month', drop_first=True)

print("Training data after creating dummy variables for 'month' and 'sunshine_category':")
display(train_df.head())

print("\nTest data after creating dummy variables for 'month' and 'sunshine_category':")
display(test_df.head())


# Separate features and target variable
X_train = train_df.drop(['id', 'day', 'rainfall'], axis=1)
y_train = train_df['rainfall']
X_test = test_df.drop(['id', 'day'], axis=1)

# Initialize PCA and fit on the training data
# We can start by keeping 95% of the variance
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train)

# Apply the same transformation to the test data
X_test_pca = pca.transform(X_test)

print(f"Original number of features: {X_train.shape[1]}")
print(f"Number of features after PCA: {X_train_pca.shape[1]}")

print("\nShape of X_train_pca:", X_train_pca.shape)
print("Shape of X_test_pca:", X_test_pca.shape)


models = [('Random Forest', RandomForestClassifier()),
          ('KNN', KNeighborsClassifier()),
          ('SVM', SVC()),
          ('Logistic Regression', LogisticRegression())]

for name, model in models:
    score = cross_val_score(model, X_train_pca, y_train, cv=5)
    print(f"{name} Accuracy: {score.mean()}")


param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': ['scale', 'auto'],
              'kernel': ['linear', 'rbf']}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train_pca, y_train)

# Print the best parameters and the best score
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation accuracy: ", grid_search.best_score_)

# Get the best model
best_svc_model = grid_search.best_estimator_


predictions = best_svc_model.predict(X_test_pca)


submission = pd.DataFrame({'id': test_df['id'], 'rainfall': predictions})
submission.to_csv('submission.csv', index=False)
print("Submissing file created.")

