import pandas as pd


df = pd.read_csv('/kaggle/input/mercedes-benz-greener-manufacturing/train.csv.zip')


df


# Generate expected column names from X0 to X385
expected_cols = [f'X{i}' for i in range(386)]

# Get the actual column names from the DataFrame, excluding 'ID' and 'y'
actual_x_cols = [col for col in df.columns if col.startswith('X')]

# Find the missing columns
missing_cols = list(set(expected_cols) - set(actual_x_cols))

print("Missing 'X' columns:", missing_cols)


df.info()


df.describe()


import matplotlib.pyplot as plt
import seaborn as sns

# Get the frequency of each value in 'X0'
x0_counts = df['X0'].value_counts()

# Create a bar plot
plt.figure(figsize=(12, 6))
sns.barplot(x=x0_counts.index, y=x0_counts.values)
plt.title('Frequency of X0 Values')
plt.xlabel('X0 Values')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Get the frequency of each value in 'X0'
x8_counts = df['X8'].value_counts()

# Create a bar plot
plt.figure(figsize=(12, 6))
sns.barplot(x=x8_counts.index, y=x8_counts.values)
plt.title('Frequency of X8 Values')
plt.xlabel('X8 Values')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.show()


# Check for binary columns
binary_cols = [col for col in df.columns if df[col].nunique() == 2 and all(df[col].isin([0, 1]))]

print(f"Number of binary variables: {len(binary_cols)}")
print("Binary variable names:", binary_cols)


import matplotlib.pyplot as plt
import seaborn as sns

# Create a histogram of 'y' values
plt.figure(figsize=(10, 6))
sns.histplot(df['y'], kde=True, bins=30)
plt.title('Distribution of y Values')
plt.xlabel('y Values')
plt.ylabel('Frequency')
plt.show()


from sklearn.preprocessing import LabelEncoder

# Select object type columns (strings)
string_cols = df.select_dtypes(include='object').columns

# Create a copy to avoid modifying the original DataFrame
df_encoded = df.copy()

# Perform label encoding
for col in string_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])

# Display the first few rows of the encoded DataFrame
display(df_encoded.head())


df_encoded.describe()


df_encoded.corr()


# Calculate the correlation matrix
correlation_matrix = df_encoded.corr()

# Get the correlations with 'y' and sort them
y_correlations = correlation_matrix['y'].sort_values(ascending=False)

# Display the correlations
print("Correlations with 'y':")
print(y_correlations)


# Identify columns with zero variance
zero_variance_cols = df_encoded.columns[df_encoded.var() == 0]

print("Columns with zero variance (will result in NaN correlations):")
print(zero_variance_cols)


df_encoded = df_encoded.drop(columns=zero_variance_cols)


X = df_encoded[['X314','X261']]
Y = df_encoded['y']


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 42)


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, Y_train)


Y_pred = model.predict(X_test)


ss_res = ((Y_test - Y_pred) ** 2).sum()
ss_tot = ((Y_test - Y_test.mean()) ** 2).sum()
r2_manual = 1 - ss_res / ss_tot
print("Manual R²:", r2_manual)


mae = (abs(Y_test - Y_pred)).mean()
print("MAE:", mae)


import numpy as np
rmse = np.sqrt(((Y_test - Y_pred) ** 2).mean())
print("RMSE:", rmse)


test_df = pd.read_csv('/kaggle/input/mercedes-benz-greener-manufacturing/test.csv.zip')


# Select object type columns (strings)
string_cols = test_df.select_dtypes(include='object').columns

# Create a copy to avoid modifying the original DataFrame
test_df_encoded = test_df.copy()

# Perform label encoding
for col in string_cols:
    le = LabelEncoder()
    test_df_encoded[col] = le.fit_transform(test_df_encoded[col])

# Display the first few rows of the encoded DataFrame
display(test_df_encoded.head())


X2 = test_df_encoded[['X314','X261']]


Y_test_pred = model.predict(X2)


submission = pd.DataFrame({
    "ID": test_df['ID'],
    "y": Y_test_pred
})

submission.to_csv('submission.csv', index=False)

print("Submission file successfully created!")

