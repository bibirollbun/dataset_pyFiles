import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Change the settings to show all rows
pd.set_option('display.max_rows', None)


# Load datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s4e1/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s4e1/test.csv')


# first 5 rows
train_df.head()


test_df.head()


# last 2 rows
train_df.tail(2)


# check the number of rows and columns
train_df.shape


test_df.shape


# Check general information about data
train_df.info()


# statistics 
train_df.describe() 


# check empty data
train_df.isnull().sum()


test_df.isnull().sum()


train_df['Geography'].value_counts()


train_df['Gender'].value_counts()


train_df['NumOfProducts'].value_counts()


train_df['Gender'].value_counts()


train_df['Tenure'].value_counts()


train_df['Exited'].value_counts()


# Calculate the distribution of the 'Exited' column
distribution = train_df['Exited'].value_counts()

# Create pie chart
plt.figure(figsize=(8, 6))
plt.pie(distribution, labels=['Not Exited (0)', 'Exited (1)'], autopct='%1.1f%%', startangle=90, colors=['lightblue', 'orange'])
plt.title('Distribution of Exited Customers')
plt.axis('equal') # Equal circle
plt.show()


from sklearn.utils import resample


# Separate positive and negative examples
df_majority = train_df[train_df['Exited'] == 0]
df_minority = train_df[train_df['Exited'] == 1]

# Resample the minority class
df_minority_resampled = resample(df_minority, 
                                  replace=True,    # Resampling with replacement
                                  n_samples=len(df_majority),  # Sample size equal to the majority class
                                  random_state=42)  # For reproducibility

# Create a new dataset after balancing
train_balanced = pd.concat([df_majority, df_minority_resampled])

# Shuffle the dataset
train_balanced = train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("Size of the balanced dataset:")
print(train_balanced['Exited'].value_counts())


# Dengeli veri setindeki `Exited` dağılımını hesapla
distribution = train_balanced['Exited'].value_counts()

# Pasta grafiğini oluştur
plt.figure(figsize=(8, 6))
plt.pie(distribution, labels=['Not Exited (0)', 'Exited (1)'], autopct='%1.1f%%', startangle=90, colors=['lightblue', 'orange'])
plt.title('Distribution of Exited Customers')
plt.axis('equal')  # Eşit oranlı daire
plt.show()


# Remove the Surname and id columns from a DataFrame
train_df = train_balanced.drop(columns=['Surname', 'CustomerId'])
test_df = test_df.drop(columns=['Surname', 'CustomerId'])


# Function to remove outliers based on IQR
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.15)
    Q3 = df[column].quantile(0.85)
    IQR = Q3 - Q1
    return df[(df[column] >= (Q1 - 1.5 * IQR)) & (df[column] <= (Q3 + 1.5 * IQR))]

# Remove outliers from EstimatedSalary and Balance for train_df
train_df = remove_outliers(train_df, 'EstimatedSalary')
train_df = remove_outliers(train_df, 'Balance')

# Remove outliers from EstimatedSalary and Balance for test_df
test_df = remove_outliers(test_df, 'EstimatedSalary')
test_df = remove_outliers(test_df, 'Balance')


# Apply get_dummies for categorical variables
train_dummies = pd.get_dummies(train_df, drop_first=True)
test_dummies = pd.get_dummies(test_df, drop_first=True)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score


# Split x and y
x = train_dummies.drop(['Exited', 'id'], axis=1)
y = train_dummies['Exited']

# Split into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define the model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(x_train, y_train)

# Make a prediction
predictions = model.predict(x_test)
predictions_proba = model.predict_proba(x_test)[:, 1] # Probabilities of the positive class

# Calculate the F1 score and AUC-ROC value
f1 = f1_score(y_test, predictions)
auc_roc = roc_auc_score(y_test, predictions_proba)

print("F1 Score:", f1)
print("AUC-ROC Score:", auc_roc)

