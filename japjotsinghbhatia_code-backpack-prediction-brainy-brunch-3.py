import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train=pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
train_extra=pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


train


print("Train DataSet Summary (First Rows,  Shape)")

display(train, train.shape)



train.dtypes


print("Training extra DataSet Summary (First Rows,  Shape,  Data Types)")

display(train_extra, train_extra.shape, train_extra.dtypes)



plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
sns.histplot(train['Price'], bins=50, kde=True, color='red')
plt.title("Train [Price] Distribution")
plt.xlabel("Price")

plt.subplot(1, 2, 2)
sns.histplot(train_extra['Price'], bins=50, kde=True, color='orange')
plt.title("Train_extra [Price] Distribution")
plt.xlabel("Price")

plt.tight_layout()
plt.show()


print("Numeric Data Distribution Across Train,Training_extra Datasets")

num_cols = test.select_dtypes(include=['number']).columns

plt.figure(figsize=(8, len(num_cols) * 3))

for i, col in enumerate(num_cols):
    plt.subplot(len(num_cols), 3, i*3 + 1)
    sns.histplot(train[col], bins=10, color='blue')
    plt.title(f"Train [{col}] Distribution")
    plt.xlabel(col)

    plt.subplot(len(num_cols), 3, i*3 + 3)
    sns.histplot(train_extra[col], bins=10, color='red')
    plt.title(f"Train_extra [{col}] Distribution")
    plt.xlabel(col)

plt.tight_layout()
plt.show()


print("Pie Chart Comparison of Categorical Variables in Train, Train_extra Datasets")

# Get the columns with object data type
obj_cols = train.select_dtypes(include=['object']).columns

for variable in obj_cols:
    sns.set_style('whitegrid')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Use 2 subplots instead of 3
    plt.subplots_adjust(wspace=0.3)

    # Pie Chart for Train data
    train[variable].value_counts().plot.pie(ax=axes[0], autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.3))
    axes[0].set_ylabel('')
    axes[0].set_title(f"train [{variable}]")

    # Pie Chart for Train_ex data
    train_extra[variable].value_counts().plot.pie(ax=axes[1], autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.3))
    axes[1].set_ylabel('')
    axes[1].set_title(f"train_ex [{variable}]")

    plt.show()



print("Missing Values Count for Train Dataset")

train.isnull().sum()



print("Missing Values Count for Train_ex Dataset")

train_extra.isnull().sum()



import matplotlib.pyplot as plt

print("Comparative Charts of Missing Data in Train and Train_extra Datasets")

# Calculate missing values
train_null = train.isnull().sum()
train_extra_null = train_extra.isnull().sum()

# Create subplots (2 rows, 2 columns)
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Train dataset: Pie Chart
axes[0, 0].pie(train_null, labels=train_null.index, autopct='%1.1f%%', startangle=90)
axes[0, 0].set_title('Missing Values in Train Dataset')

# Train dataset: Bar Chart
axes[0, 1].barh(train_null.index, train_null.values, color='skyblue')
axes[0, 1].set_title('Missing Values in Train Dataset')
axes[0, 1].set_xlabel('Count')
axes[0, 1].invert_yaxis()

# Train_extra dataset: Pie Chart
axes[1, 0].pie(train_extra_null, labels=train_extra_null.index, autopct='%1.1f%%', startangle=90)
axes[1, 0].set_title('Missing Values in Train_extra Dataset')

# Train_extra dataset: Bar Chart
axes[1, 1].barh(train_extra_null.index, train_extra_null.values, color='lightcoral')
axes[1, 1].set_title('Missing Values in Train_extra Dataset')
axes[1, 1].set_xlabel('Count')
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.show()



print("Missing Values Heatmap for Train and Train_extra Datasets")

# Create subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Train Missing Values Heatmap
sns.heatmap(train.isnull(), cmap="viridis", cbar=False, ax=axes[0])
axes[0].set_title("Missing Values in Train")


# Train_extra Missing Values Heatmap
sns.heatmap(train_extra.isnull(), cmap="viridis", cbar=False, ax=axes[1])
axes[1].set_title("Missing Values in Train_extra")


plt.tight_layout()
plt.show()


# Merging Train and Train_ex Data

train = pd.concat([train, train_extra], axis=0, ignore_index=True)



train


print("Updated Train Dataset Types")

train.dtypes



print("Missing Values Count for Updated Train Dataset")

train.isnull().sum()



# numerical columns are id compartments weight capacity price
#in which only weight capacity contains null values of 1808


#information related dataset
train.info()


#Unique Values in particular "Weight Capacity (kg)" column
train["Weight Capacity (kg)"].unique()


#particular values count of "Weight Capacity (kg)"
train["Weight Capacity (kg)"].value_counts()


#null vallues of "Weight Capacity (kg)" sum
train["Weight Capacity (kg)"].isnull().sum()


#Non null values count"Weight Capacity (kg)"
train["Weight Capacity (kg)"].count()


#Percentage of null from non null
train["Weight Capacity (kg)"].isnull().sum()*100/3994318



# Specify the columns
column1 = "Weight Capacity (kg)"  # Categorical column
column2 = "Style"  # Categorical column

# Convert categorical data to numerical using factorization (without modifying original dataset)
column1_encoded, _ = pd.factorize(train[column1])
column2_encoded, _ = pd.factorize(train[column2])

# Compute correlation
correlation = pd.Series(column1_encoded).corr(pd.Series(column2_encoded))

print(f'Correlation between {column1} and {column2}: {correlation}')



# No relation between Weight Capacity (kg) and Style



# Specify the columns
column1 = "Weight Capacity (kg)"  # Categorical column
column2 = "Size"  # Categorical column

# Convert categorical data to numerical using factorization (without modifying original dataset)
column1_encoded, _ = pd.factorize(train[column1])
column2_encoded, _ = pd.factorize(train[column2])

# Compute correlation
correlation = pd.Series(column1_encoded).corr(pd.Series(column2_encoded))

print(f'Correlation between {column1} and {column2}: {correlation}')



## No relation between Weight Capacity (kg) and Size


a=train["Weight Capacity (kg)"].mode()[0]
a



train["Weight Capacity (kg)"]=train["Weight Capacity (kg)"].fillna(a)



#null vallues of "Weight Capacity (kg)" sum after handling
train["Weight Capacity (kg)"].isnull().sum()


train['Brand'].unique()


train["Brand"].value_counts()


#almost of same ratios


train['Brand'].isna().sum()


train["Brand"].count()


train['Brand'].isna().sum()*100/3994318



# Specify the columns
column1 = "Brand"  # Categorical column
column2 = "Material"  # Categorical column

# Convert categorical data to numerical using factorization (without modifying original dataset)
column1_encoded, _ = pd.factorize(train[column1])
column2_encoded, _ = pd.factorize(train[column2])

# Compute correlation
correlation = pd.Series(column1_encoded).corr(pd.Series(column2_encoded))

print(f'Correlation between {column1} and {column2}: {correlation}')




# Specify the columns
column1 = "Brand"  # Categorical column
column2 = "Style"  # Categorical column

# Convert categorical data to numerical using factorization (without modifying original dataset)
column1_encoded, _ = pd.factorize(train[column1])
column2_encoded, _ = pd.factorize(train[column2])

# Compute correlation
correlation = pd.Series(column1_encoded).corr(pd.Series(column2_encoded))

print(f'Correlation between {column1} and {column2}: {correlation}')




# Specify the columns
column1 = "Brand"  # Categorical column
column2 = "Price"  # Categorical column

# Convert categorical data to numerical using factorization (without modifying original dataset)
column1_encoded, _ = pd.factorize(train[column1])
column2_encoded, _ = pd.factorize(train[column2])

# Compute correlation
correlation = pd.Series(column1_encoded).corr(pd.Series(column2_encoded))

print(f'Correlation between {column1} and {column2}: {correlation}')



value_probs = train["Brand"].value_counts(normalize=True)
value_probs


# Find indices where 'Material' is NaN
nan_indices = train[train["Brand"].isna()].index

# Generate random values based on the existing distribution
random_choices = np.random.choice(value_probs.index, size=len(nan_indices), p=value_probs.values)

# Assign the generated values to the NaN positions
train.loc[nan_indices, "Brand"] = random_choices



train["Brand"].value_counts()


train["Brand"].isna().sum()


train['Material'].unique()


train['Material'].isna().sum()



train['Material'].count()


train['Material'].isna().sum()*100/3994318



train['Material'].value_counts()



# Specify the columns
column1 = "Material"  # Categorical column
column2 = "Price"  # Categorical column

# Convert categorical data to numerical using factorization (without modifying original dataset)
column1_encoded, _ = pd.factorize(train[column1])
column2_encoded, _ = pd.factorize(train[column2])

# Compute correlation
correlation = pd.Series(column1_encoded).corr(pd.Series(column2_encoded))

print(f'Correlation between {column1} and {column2}: {correlation}')




# Specify the columns
column1 = "Material"  # Categorical column
column2 = "Waterproof"  # Categorical column

# Convert categorical data to numerical using factorization (without modifying original dataset)
column1_encoded, _ = pd.factorize(train[column1])
column2_encoded, _ = pd.factorize(train[column2])

# Compute correlation
correlation = pd.Series(column1_encoded).corr(pd.Series(column2_encoded))

print(f'Correlation between {column1} and {column2}: {correlation}')




# Specify the columns
column1 = "Material"  # Categorical column
column2 = "Style"  # Categorical column

# Convert categorical data to numerical using factorization (without modifying original dataset)
column1_encoded, _ = pd.factorize(train[column1])
column2_encoded, _ = pd.factorize(train[column2])

# Compute correlation
correlation = pd.Series(column1_encoded).corr(pd.Series(column2_encoded))

print(f'Correlation between {column1} and {column2}: {correlation}')



value_probs = train["Material"].value_counts(normalize=True)
value_probs


# Find indices where 'Material' is NaN
nan_indices = train[train["Material"].isna()].index

# Generate random values based on the existing distribution
random_choices = np.random.choice(value_probs.index, size=len(nan_indices), p=value_probs.values)

# Assign the generated values to the NaN positions
train.loc[nan_indices, "Material"] = random_choices



train['Material'].value_counts()


train['Material'].isna().sum()


train["Size"].unique()


train["Size"].isna().sum()


train['Size'].count()


train["Size"].isna().sum()*100/3994918


train["Size"].value_counts()



# Specify the columns
column1 = "Size"  # Categorical column
column2 = "Style"  # Categorical column

# Convert categorical data to numerical using factorization (without modifying original dataset)
column1_encoded, _ = pd.factorize(train[column1])
column2_encoded, _ = pd.factorize(train[column2])

# Compute correlation
correlation = pd.Series(column1_encoded).corr(pd.Series(column2_encoded))

print(f'Correlation between {column1} and {column2}: {correlation}')




# Specify the columns
column1 = "Size"  # Categorical column
column2 = "Compartments"  # Categorical column

# Convert categorical data to numerical using factorization (without modifying original dataset)
column1_encoded, _ = pd.factorize(train[column1])
column2_encoded, _ = pd.factorize(train[column2])

# Compute correlation
correlation = pd.Series(column1_encoded).corr(pd.Series(column2_encoded))

print(f'Correlation between {column1} and {column2}: {correlation}')




# Specify the columns
column1 = "Size"  # Categorical column
column2 = "Laptop Compartment"  # Categorical column

# Convert categorical data to numerical using factorization (without modifying original dataset)
column1_encoded, _ = pd.factorize(train[column1])
column2_encoded, _ = pd.factorize(train[column2])

# Compute correlation
correlation = pd.Series(column1_encoded).corr(pd.Series(column2_encoded))

print(f'Correlation between {column1} and {column2}: {correlation}')




# Specify the columns
column1 = "Size"  # Categorical column
column2 = "Weight Capacity (kg)"  # Categorical column

# Convert categorical data to numerical using factorization (without modifying original dataset)
column1_encoded, _ = pd.factorize(train[column1])
column2_encoded, _ = pd.factorize(train[column2])

# Compute correlation
correlation = pd.Series(column1_encoded).corr(pd.Series(column2_encoded))

print(f'Correlation between {column1} and {column2}: {correlation}')




# Specify the columns
column1 = "Size"  # Categorical column
column2 = "Price"  # Categorical column

# Convert categorical data to numerical using factorization (without modifying original dataset)
column1_encoded, _ = pd.factorize(train[column1])
column2_encoded, _ = pd.factorize(train[column2])

# Compute correlation
correlation = pd.Series(column1_encoded).corr(pd.Series(column2_encoded))

print(f'Correlation between {column1} and {column2}: {correlation}')



value_probs = train["Size"].value_counts(normalize=True)
value_probs


# Find indices where 'Material' is NaN
nan_indices = train[train["Size"].isna()].index

# Generate random values based on the existing distribution
random_choices = np.random.choice(value_probs.index, size=len(nan_indices), p=value_probs.values)

# Assign the generated values to the NaN positions
train.loc[nan_indices, "Size"] = random_choices



train["Size"].value_counts()


train["Size"].isna().sum()


train["Laptop Compartment"].unique()


train["Laptop Compartment"].isna().sum()


train["Laptop Compartment"].count()


train["Laptop Compartment"].isna().sum()*100/3994318


train["Laptop Compartment"].value_counts()



# Specify the columns
column1 = "Laptop Compartment"  # Categorical column
column2 = "Size"  # Categorical column

# Convert categorical data to numerical using factorization (without modifying original dataset)
column1_encoded, _ = pd.factorize(train[column1])
column2_encoded, _ = pd.factorize(train[column2])

# Compute correlation
correlation = pd.Series(column1_encoded).corr(pd.Series(column2_encoded))

print(f'Correlation between {column1} and {column2}: {correlation}')




# Specify the columns
column1 = "Laptop Compartment"  # Categorical column
column2 = "Weight Capacity (kg)"  # Categorical column

# Convert categorical data to numerical using factorization (without modifying original dataset)
column1_encoded, _ = pd.factorize(train[column1])
column2_encoded, _ = pd.factorize(train[column2])

# Compute correlation
correlation = pd.Series(column1_encoded).corr(pd.Series(column2_encoded))

print(f'Correlation between {column1} and {column2}: {correlation}')




# Specify the columns
column1 = "Laptop Compartment"  # Categorical column
column2 = "Price"  # Categorical column

# Convert categorical data to numerical using factorization (without modifying original dataset)
column1_encoded, _ = pd.factorize(train[column1])
column2_encoded, _ = pd.factorize(train[column2])

# Compute correlation
correlation = pd.Series(column1_encoded).corr(pd.Series(column2_encoded))

print(f'Correlation between {column1} and {column2}: {correlation}')



value_probs = train["Laptop Compartment"].value_counts(normalize=True)
value_probs


# Find indices where 'Material' is NaN
nan_indices = train[train["Laptop Compartment"].isna()].index

# Generate random values based on the existing distribution
random_choices = np.random.choice(value_probs.index, size=len(nan_indices), p=value_probs.values)

# Assign the generated values to the NaN positions
train.loc[nan_indices, "Laptop Compartment"] = random_choices



train["Laptop Compartment"].value_counts()


train["Laptop Compartment"].isna().sum()


train['Waterproof'].unique()


train["Waterproof"].isna().sum()


train["Waterproof"].count()


train["Waterproof"].isna().sum()*100/3994318


train["Waterproof"].value_counts()



# Specify the columns
column1 = "Waterproof"  # Categorical column
column2 = "Material"  # Categorical column

# Convert categorical data to numerical using factorization (without modifying original dataset)
column1_encoded, _ = pd.factorize(train[column1])
column2_encoded, _ = pd.factorize(train[column2])

# Compute correlation
correlation = pd.Series(column1_encoded).corr(pd.Series(column2_encoded))

print(f'Correlation between {column1} and {column2}: {correlation}')




# Specify the columns
column1 = "Waterproof"  # Categorical column
column2 = "Style"  # Categorical column

# Convert categorical data to numerical using factorization (without modifying original dataset)
column1_encoded, _ = pd.factorize(train[column1])
column2_encoded, _ = pd.factorize(train[column2])

# Compute correlation
correlation = pd.Series(column1_encoded).corr(pd.Series(column2_encoded))

print(f'Correlation between {column1} and {column2}: {correlation}')




# Specify the columns
column1 = "Waterproof"  # Categorical column
column2 = "Price"  # Categorical column

# Convert categorical data to numerical using factorization (without modifying original dataset)
column1_encoded, _ = pd.factorize(train[column1])
column2_encoded, _ = pd.factorize(train[column2])

# Compute correlation
correlation = pd.Series(column1_encoded).corr(pd.Series(column2_encoded))

print(f'Correlation between {column1} and {column2}: {correlation}')



value_probs = train["Waterproof"].value_counts(normalize=True)
value_probs


# Find indices where 'Material' is NaN
nan_indices = train[train["Waterproof"].isna()].index

# Generate random values based on the existing distribution
random_choices = np.random.choice(value_probs.index, size=len(nan_indices), p=value_probs.values)

# Assign the generated values to the NaN positions
train.loc[nan_indices, "Waterproof"] = random_choices



train["Waterproof"].value_counts()


train["Waterproof"].isna().sum()


train['Style'].unique()


train['Style'].isna().sum()


train['Style'].count()


train["Style"].value_counts()


train['Style'].isna().sum()*100/3994318



# Specify the columns
column1 = "Style"  # Categorical column
column2 = "Material"  # Categorical column

# Convert categorical data to numerical using factorization (without modifying original dataset)
column1_encoded, _ = pd.factorize(train[column1])
column2_encoded, _ = pd.factorize(train[column2])

# Compute correlation
correlation = pd.Series(column1_encoded).corr(pd.Series(column2_encoded))

print(f'Correlation between {column1} and {column2}: {correlation}')




# Specify the columns
column1 = "Style"  # Categorical column
column2 = "Size"  # Categorical column

# Convert categorical data to numerical using factorization (without modifying original dataset)
column1_encoded, _ = pd.factorize(train[column1])
column2_encoded, _ = pd.factorize(train[column2])

# Compute correlation
correlation = pd.Series(column1_encoded).corr(pd.Series(column2_encoded))

print(f'Correlation between {column1} and {column2}: {correlation}')




# Specify the columns
column1 = "Style"  # Categorical column
column2 = "Waterproof"  # Categorical column

# Convert categorical data to numerical using factorization (without modifying original dataset)
column1_encoded, _ = pd.factorize(train[column1])
column2_encoded, _ = pd.factorize(train[column2])

# Compute correlation
correlation = pd.Series(column1_encoded).corr(pd.Series(column2_encoded))

print(f'Correlation between {column1} and {column2}: {correlation}')




# Specify the columns
column1 = "Style"  # Categorical column
column2 = "Price"  # Categorical column

# Convert categorical data to numerical using factorization (without modifying original dataset)
column1_encoded, _ = pd.factorize(train[column1])
column2_encoded, _ = pd.factorize(train[column2])

# Compute correlation
correlation = pd.Series(column1_encoded).corr(pd.Series(column2_encoded))

print(f'Correlation between {column1} and {column2}: {correlation}')



value_probs = train["Style"].value_counts(normalize=True)
value_probs



# Find indices where 'Material' is NaN
nan_indices = train[train["Style"].isna()].index

# Generate random values based on the existing distribution
random_choices = np.random.choice(value_probs.index, size=len(nan_indices), p=value_probs.values)

# Assign the generated values to the NaN positions
train.loc[nan_indices, "Style"] = random_choices


train["Style"].value_counts()


train["Style"].isna().sum()


train['Color'].unique()


train['Color'].isna().sum()


train['Color'].count()


train['Color'].isna().sum()*100/3994318


train['Color'].value_counts()



# Specify the columns
column1 = "Color"  # Categorical column
column2 = "Brand"  # Categorical column

# Convert categorical data to numerical using factorization (without modifying original dataset)
column1_encoded, _ = pd.factorize(train[column1])
column2_encoded, _ = pd.factorize(train[column2])

# Compute correlation
correlation = pd.Series(column1_encoded).corr(pd.Series(column2_encoded))

print(f'Correlation between {column1} and {column2}: {correlation}')




# Specify the columns
column1 = "Color"  # Categorical column
column2 = "Material"  # Categorical column

# Convert categorical data to numerical using factorization (without modifying original dataset)
column1_encoded, _ = pd.factorize(train[column1])
column2_encoded, _ = pd.factorize(train[column2])

# Compute correlation
correlation = pd.Series(column1_encoded).corr(pd.Series(column2_encoded))

print(f'Correlation between {column1} and {column2}: {correlation}')




# Specify the columns
column1 = "Color"  # Categorical column
column2 = "Price"#Categorical column

# Convert categorical data to numerical using factorization (without modifying original dataset)
column1_encoded, _ = pd.factorize(train[column1])
column2_encoded, _ = pd.factorize(train[column2])

# Compute correlation
correlation = pd.Series(column1_encoded).corr(pd.Series(column2_encoded))

print(f'Correlation between {column1} and {column2}: {correlation}')



value_probs = train["Color"].value_counts(normalize=True)
value_probs


# Find indices where 'Material' is NaN
nan_indices = train[train["Color"].isna()].index

# Generate random values based on the existing distribution
random_choices = np.random.choice(value_probs.index, size=len(nan_indices), p=value_probs.values)

# Assign the generated values to the NaN positions
train.loc[nan_indices, "Color"] = random_choices


train["Color"].value_counts()


train["Color"].isna().sum()


train.isnull().sum()




print("Missing Values Heatmap for Datasets after handling it")

# Create a single subplot
fig, ax = plt.subplots(figsize=(12, 5))

# Train Missing Values Heatmap
sns.heatmap(train.isnull(), cmap="viridis", cbar=False, ax=ax)
ax.set_title("Missing Values in Train")

plt.tight_layout()
plt.show()





print("Missing Values in Train Dataset after handling null values")

# Calculate missing values
train_null = train.isnull().sum()

# Filter only columns with missing values
train_null = train_null[train_null > 0]

# Create bar plot for missing values
plt.figure(figsize=(10, 5))
plt.barh(train_null.index, train_null.values, color='skyblue')
plt.xlabel("Count")
plt.ylabel("Columns")
plt.title("Missing Values in Train Dataset")
plt.gca().invert_yaxis()  # Invert y-axis for better readability

plt.show()



# Create a copy of the dataset to avoid modifying the original
train_copy = train.copy()

# Convert categorical columns to numeric using factorization
for col in train_copy.select_dtypes(include=['object']).columns:
    train_copy[col], _ = pd.factorize(train_copy[col])

# Plot the heatmap using the modified copy
plt.figure(figsize=(9, 6))
heatmap = sns.heatmap(train_copy.corr(), annot=True, cmap='coolwarm', fmt=".4f", annot_kws={"size":9})
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=70, fontsize=9)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=9)
plt.title('Correlation Heatmap of Train Dataset')
plt.show()



a=test["Weight Capacity (kg)"].mode()[0]
a



test["Weight Capacity (kg)"]=test["Weight Capacity (kg)"].fillna(a)



#null vallues of "Weight Capacity (kg)" sum after handling
test["Weight Capacity (kg)"].isnull().sum()


value_probs = test["Brand"].value_counts(normalize=True)
value_probs


# Find indices where 'Material' is NaN
nan_indices = test[test["Brand"].isna()].index

# Generate random values based on the existing distribution
random_choices = np.random.choice(value_probs.index, size=len(nan_indices), p=value_probs.values)

# Assign the generated values to the NaN positions
test.loc[nan_indices, "Brand"] = random_choices



# Find indices where 'Material' is NaN
nan_indices = test[test["Material"].isna()].index

# Generate random values based on the existing distribution
random_choices = np.random.choice(value_probs.index, size=len(nan_indices), p=value_probs.values)

# Assign the generated values to the NaN positions
test.loc[nan_indices, "Material"] = random_choices



value_probs = test["Size"].value_counts(normalize=True)
value_probs


# Find indices where 'Material' is NaN
nan_indices = test[test["Size"].isna()].index

# Generate random values based on the existing distribution
random_choices = np.random.choice(value_probs.index, size=len(nan_indices), p=value_probs.values)

# Assign the generated values to the NaN positions
test.loc[nan_indices, "Size"] = random_choices



value_probs = test["Laptop Compartment"].value_counts(normalize=True)
value_probs


# Find indices where 'Material' is NaN
nan_indices = test[test["Laptop Compartment"].isna()].index

# Generate random values based on the existing distribution
random_choices = np.random.choice(value_probs.index, size=len(nan_indices), p=value_probs.values)

# Assign the generated values to the NaN positions
test.loc[nan_indices, "Laptop Compartment"] = random_choices



value_probs = test["Waterproof"].value_counts(normalize=True)
value_probs


# Find indices where 'Material' is NaN
nan_indices = test[test["Waterproof"].isna()].index

# Generate random values based on the existing distribution
random_choices = np.random.choice(value_probs.index, size=len(nan_indices), p=value_probs.values)

# Assign the generated values to the NaN positions
test.loc[nan_indices, "Waterproof"] = random_choices



value_probs = test["Style"].value_counts(normalize=True)
value_probs



# Find indices where 'Material' is NaN
nan_indices = test[test["Style"].isna()].index

# Generate random values based on the existing distribution
random_choices = np.random.choice(value_probs.index, size=len(nan_indices), p=value_probs.values)

# Assign the generated values to the NaN positions
test.loc[nan_indices, "Style"] = random_choices


value_probs = test["Color"].value_counts(normalize=True)
value_probs


# Find indices where 'Material' is NaN
nan_indices = test[test["Color"].isna()].index

# Generate random values based on the existing distribution
random_choices = np.random.choice(value_probs.index, size=len(nan_indices), p=value_probs.values)

# Assign the generated values to the NaN positions
test.loc[nan_indices, "Color"] = random_choices


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

print("Starting data preparation and model training...")

# Copy the dataset to preserve the original
train_copy = train.copy()

# Define the target variable
target_col = "Price"

# Separate features and target; drop 'id' if it exists
X = train_copy.drop(columns=[target_col, "id"], errors='ignore')
y = train_copy[target_col]

# Convert categorical features to numeric
for col in X.select_dtypes(include=['object']).columns:
    X[col], _ = pd.factorize(X[col])

# Split data into training and validation sets (80/20 split)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize XGBoost regressor with tuned hyperparameters
model = XGBRegressor(
    max_depth=4, 
    n_estimators=2500, 
    learning_rate=0.02, 
    subsample=0.9, 
    colsample_bytree=0.8,
    reg_lambda=2,
    reg_alpha=1,
    random_state=42,
    eval_metric='rmse'
)


# Train the model with validation monitoring
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=200
)

# Evaluate the model using RMSE on the validation set
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse:.4f}")

# Retrieve evaluation metrics (if supported) and identify the best iteration
evals_result = model.evals_result()
val_rmse = evals_result['validation_0']['rmse']
min_rmse = min(val_rmse)
min_index = val_rmse.index(min_rmse)

print("Model training completed successfully!")



try:
    evals_result = model.evals_result()
    val_rmse = evals_result['validation_0']['rmse']
    min_rmse = min(val_rmse)
    min_index = val_rmse.index(min_rmse)
    
    plt.figure(figsize=(10, 5))
    plt.plot(val_rmse, label='Validation RMSE', color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.title('Trends in Validation RMSE During Model Training')
    plt.scatter(min_index, min_rmse, color='red', s=50, label=f'Min RMSE: {min_rmse:.3f}')
    plt.text(min_index+50, min_rmse+0.02, f'Validation RMSE (min): {min_rmse:.3f}', 
             color='red', fontsize=11, ha='right')
    plt.legend()
    plt.grid(True)
    plt.show()
except Exception as e:
    print("Evaluation results not available for plotting:", e)


print("Starting predictions on test data...")

# Create a copy of the test DataFrame
test_copy = test.copy()

# Save the ID column for submission
ids = test_copy['id']

# Drop the 'id' column from the features before prediction
X_test = test_copy.drop(columns=['id'], errors='ignore')

# Convert categorical columns to numeric using factorize (same approach as training)
for col in X_test.select_dtypes(include=['object']).columns:
    X_test[col], _ = pd.factorize(X_test[col])

# Predict prices using the trained model
test_predictions = model.predict(X_test)

# Create a submission DataFrame with 'id' and predicted 'Price'
submission = pd.DataFrame({
    'id': ids,
    'Price': test_predictions
})

# Save submission to CSV (without index)
submission.to_csv("submission.csv", index=False)

print("Predictions completed and saved to 'submission.csv'.")




