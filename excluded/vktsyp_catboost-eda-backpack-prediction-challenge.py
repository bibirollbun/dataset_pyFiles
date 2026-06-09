import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings("ignore")


#Importing Train, Test, and Training Extra Datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv',index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv',index_col='id')
train_ex = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv",index_col='id')


print("Train DataSet Summary (First Rows, Shape, Data Types)")

display(train.head(), train.shape, train.dtypes)


print("Test DataSet Summary (First Rows, Shape, Data Types)")

display(test.head(), test.shape, test.dtypes)


print("Train_ex DataSet Summary (First Rows, Shape, Data Types)")

display(train_ex.head(), train_ex.shape, train_ex.dtypes)


print("Price Distributions in Train and Train_ex Datasets")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
sns.histplot(train['Price'], bins=50, kde=True, color='blue')
plt.title("Train [Price] Distribution")
plt.xlabel("Price")

plt.subplot(1, 2, 2)
sns.histplot(train_ex['Price'], bins=50, kde=True, color='green')
plt.title("Train_ex [Price] Distribution")
plt.xlabel("Price")

plt.tight_layout()
plt.show()


print("Numeric Data Distribution Across Train, Test, and Train_ex Datasets")

num_cols = test.select_dtypes(include=['number']).columns

plt.figure(figsize=(12, len(num_cols) * 3))

for i, col in enumerate(num_cols):
    plt.subplot(len(num_cols), 3, i*3 + 1)
    sns.histplot(train[col], bins=10, color='blue')
    plt.title(f"Train [{col}] Distribution")
    plt.xlabel(col)
    
    plt.subplot(len(num_cols), 3, i*3 + 2)
    sns.histplot(test[col], bins=10, color='green')
    plt.title(f"Test [{col}] Distribution")
    plt.xlabel(col)
    
    plt.subplot(len(num_cols), 3, i*3 + 3)
    sns.histplot(train_ex[col], bins=10, color='red')
    plt.title(f"Train_ex [{col}] Distribution")
    plt.xlabel(col)

plt.tight_layout()
plt.show()


print("Pie Chart Comparison of Categorical Variables in Train, Test, and Train_ex Datasets")

obj_cols = train.select_dtypes(include=['object']).columns

for variable in obj_cols:
    sns.set_style('whitegrid')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(wspace=0.3)
    
    # Pie Chart for Train
    train[variable].value_counts().plot.pie(ax=axes[0], autopct='%1.1f%%', startangle=90)
    axes[0].set_ylabel('')
    axes[0].set_title(f"train [{variable}]")
    
    # Pie Chart for Test
    test[variable].value_counts().plot.pie(ax=axes[1], autopct='%1.1f%%', startangle=90)
    axes[1].set_ylabel('')
    axes[1].set_title(f"test [{variable}]")
    
    # Pie Chart for Train_ex
    train_ex[variable].value_counts().plot.pie(ax=axes[2], autopct='%1.1f%%', startangle=90)
    axes[2].set_ylabel('')
    axes[2].set_title(f"train_ex [{variable}]")
    
    plt.show()



print("Missing Values Count for Train Dataset")

train.isnull().sum()


print("Missing Values Count for Test Dataset")

test.isnull().sum()


print("Missing Values Count for Train_ex Dataset")

train_ex.isnull().sum()


print("Comparative Charts of Missing Data in Train, Test, and Train_ex Datasets")

# Calculate the number of missing values
train_null = train.isnull().sum()
test_null = test.isnull().sum()
train_ex_null = train_ex.isnull().sum()

# Plot pie charts and bar plots
fig, axes = plt.subplots(3, 2, figsize=(12, 12))

# Missing values in Train dataset
axes[0, 0].pie(train_null, labels=train_null.index, autopct='%1.1f%%', startangle=90)
axes[0, 0].set_title('Missing Values in Train Dataset')
axes[0, 1].barh(train_null.index, train_null.values, color='skyblue')
axes[0, 1].set_title('Missing Values in Train Dataset')
axes[0, 1].set_xlabel('Count')
axes[0, 1].invert_yaxis()

# Missing values in Test dataset
axes[1, 0].pie(test_null, labels=test_null.index, autopct='%1.1f%%', startangle=90)
axes[1, 0].set_title('Missing Values in Test Dataset')
axes[1, 1].barh(test_null.index, test_null.values, color='skyblue')
axes[1, 1].set_title('Missing Values in Test Dataset')
axes[1, 1].set_xlabel('Count')
axes[1, 1].invert_yaxis()

# Missing values in Train_ex dataset
axes[2, 0].pie(train_ex_null, labels=train_ex_null.index, autopct='%1.1f%%', startangle=90)
axes[2, 0].set_title('Missing Values in Train_ex Dataset')
axes[2, 1].barh(train_ex_null.index, train_ex_null.values, color='skyblue')
axes[2, 1].set_title('Missing Values in Train_ex Dataset')
axes[2, 1].set_xlabel('Count')
axes[2, 1].invert_yaxis()

plt.tight_layout()
plt.show()


print("Visualization of Missing Data Locations in Train, Test, and Train_ex Datasets")

# Plot missing data matrix for Train dataset
msno.matrix(train, color=(0.0, 0.2, 0.4))
plt.title('Missing Data Locations in Train Dataset', fontsize=24)
plt.xlabel('Columns', fontsize=20)
plt.show()

# Plot missing data matrix for Test dataset
msno.matrix(test, color=(0.0, 0.4, 0.2))
plt.title('Missing Data Locations in Test Dataset', fontsize=24)
plt.xlabel('Columns', fontsize=20)
plt.show()

# Plot missing data matrix for Train_ex dataset
msno.matrix(train_ex, color=(0.6, 0.2, 0.0))
plt.title('Missing Data Locations in Train_ex Dataset', fontsize=24)
plt.xlabel('Columns', fontsize=20)
plt.show()


print("Visualizing Missing Values in Train, Test, and Train_ex Datasets")

# Function to highlight missing values in the DataFrame
def highlight_missing(val):
    if pd.isna(val):
        # Apply styling for missing values
        return 'background-color: SkyBlue; border: 1px solid red'
    else:
        return ''

# Function to get representative rows with missing values
def get_representative_rows(df):
    columns_with_issues = df.columns[df.isnull().sum() > 0]
    representative_rows = pd.concat(
        [df[df[col].isnull()].iloc[:1] for col in columns_with_issues]
    ).drop_duplicates()
    representative_rows_sorted = representative_rows.sort_values(by='id')
    return representative_rows_sorted

# Get representative rows with missing values for each dataset
train_representative = get_representative_rows(train)
test_representative = get_representative_rows(test)
train_ex_representative = get_representative_rows(train_ex)

# Apply styling to highlight missing values in each DataFrame
styled_train = train_representative.style.applymap(highlight_missing)
styled_test = test_representative.style.applymap(highlight_missing)
styled_train_ex = train_ex_representative.style.applymap(highlight_missing)

# Display the styled DataFrames separately
print("Missing Values in Train Dataset")
display(styled_train)

print("Missing Values in Test Dataset")
display(styled_test)

print("Missing Values in Train_ex Dataset")
display(styled_train_ex)



# Merging Train and Train_ex Data

train = pd.concat([train, train_ex], axis=0, ignore_index=True)



print("Updated Train Dataset")

train


print("Updated Train Dataset Types")

train.dtypes


print("Missing Values Count for Updated Train Dataset")

train.isnull().sum()


# Detect and Add Numeric Columns with Missing Values as Object Type

# Function to detect numeric columns with missing values
def detect_numeric_columns_with_missing(df):
    numeric_cols = df.select_dtypes(include=['number']).columns
    numeric_cols_with_missing = numeric_cols[df[numeric_cols].isnull().sum() > 0]
    return numeric_cols_with_missing

# Function to convert detected numeric columns to object type
def convert_to_object(df, cols_with_missing):
    for col in cols_with_missing:
        new_col_name = f"{col} (obj)"
        df[new_col_name] = df[col].astype('object')
    return df

# Detect numeric columns with missing values in the dataset
numeric_cols_with_missing = detect_numeric_columns_with_missing(test)

# Convert the detected columns to object type in both train and test datasets
train = convert_to_object(train, numeric_cols_with_missing)
test = convert_to_object(test, numeric_cols_with_missing)



print("Train DataSet Summary (First Rows, Shape, Data Types)")

display(train.head().T, train.shape, train.dtypes)


print("Test DataSet Summary (First Rows, Shape, Data Types)")

display(test.head().T, test.shape, test.dtypes)


# Impute missing numerical data with the median values from the TRAIN dataset

num_cols = test.select_dtypes(include=['number']).columns

imputation_value = train[num_cols].median()

train[num_cols] = train[num_cols].fillna(imputation_value)
test[num_cols] = test[num_cols].fillna(imputation_value)


print("Missing Values and Data Types for Train Dataset")

display(train.dtypes, train.isnull().sum())


print("Missing Values and Data Types for Test Dataset")

display(test.dtypes, test.isnull().sum())


# Impute Missing Values in Object Columns with 'None'

obj_cols = train.select_dtypes(include=['object']).columns

train[obj_cols] = train[obj_cols].fillna('None')
test[obj_cols] = test[obj_cols].fillna('None')


print("Missing Values and Data Types for Train Dataset")

display(train.dtypes, train.isnull().sum())


print("Missing Values and Data Types for Test Dataset")

display(test.dtypes, test.isnull().sum())


# Converting object type data to categorical type for compatibility with CatBoost.

obj_cols = train.select_dtypes(include=['object']).columns

train[obj_cols] = train[obj_cols].astype('string').astype('category')
test[obj_cols] = test[obj_cols].astype('string').astype('category')



print("Data Types for Train Dataset")

display(train.dtypes)


print("Data Types for Test Dataset")

display(test.dtypes)


# Set the target variable 'Price' as y and features as X for training data

X = train.drop(['Price'], axis=1)
y = train['Price']


print("Features X Summary (First Rows, Shape, Data Types)")

display(X.head(), X.shape, X.dtypes)


print("Target y Summary (First Rows, Shape, Data Type)")

display(y.head(), y.shape, y.dtypes)


# Initialize variables to store values
val_rmse_sav = []
y_val_sav = []
y_val_pred_sav = []
y_test_pred_sav = []

# Train CatBoost using K-Fold Cross-Validation
for train_id, val_id in KFold(5, shuffle=True, random_state=42).split(X, y):
    
    # Define CatBoost model
    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.24,
        depth=4,  
        task_type='GPU',        
        random_seed=42  )
    
    # Split training and validation data
    X_train,X_val,y_train,y_val = X.iloc[train_id],X.iloc[val_id],y.iloc[train_id],y.iloc[val_id]    
    
    # Train model
    model.fit(X_train, y_train, eval_set=(X_val,y_val), cat_features=list(obj_cols), verbose=1000)    

    # Predict values
    y_val_pred = model.predict(X_val)    
    y_test_pred = model.predict(test)    
    val_rmse = model.get_evals_result()['validation']['RMSE']
        
    # Save data
    y_val_sav.append(y_val)        
    y_val_pred_sav.append(y_val_pred)    
    y_test_pred_sav.append(y_test_pred)    
    val_rmse_sav.append(val_rmse)



print("Comparison of Validation True and Predicted Values")

y_true = [val for sublist in y_val_sav for val in sublist]
y_pred = [pred for sublist in y_val_pred_sav for pred in sublist]

# Plot preparation
plt.figure(figsize=(7, 5))
plt.scatter(y_true, y_pred, c=y_pred, cmap='viridis', s=20, alpha=0.7, linewidth=0.5)
cb = plt.colorbar()
#cb.set_label('Prediction values')

# Plot the diagonal line
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', linewidth=1.0)
plt.axis('equal')

plt.xlabel(f'True values (y_val)', fontsize=11)
plt.ylabel(f'Predicted values (y_val_pred)', fontsize=11)
plt.title('Comparison of True and Predicted Values (Validation)', fontsize=12)
plt.grid(True)
plt.show()



# Trends in Validation RMSE During Model Training

mean_val_rmse = np.mean(val_rmse_sav, axis=0)
min_val_rmse = np.min(mean_val_rmse)
min_val_rmse_iteration = np.argmin(mean_val_rmse)

plt.figure(figsize=(9, 5))
for idx, rmse_list in enumerate(val_rmse_sav):
    plt.plot(rmse_list, alpha=0.5, color='green', linestyle='--', label='Individual Validation RMSE' if idx == 0 else "")
plt.plot(mean_val_rmse, label='Validation RMSE (Mean)', color='blue')
plt.scatter(min_val_rmse_iteration, min_val_rmse, color='red')
plt.text(min_val_rmse_iteration, min_val_rmse+0.01, f'min Validation RMSE: {min_val_rmse:.3f}', ha='right', fontsize=11, color='red')
plt.xlabel('Iterations', fontsize=11)
plt.ylabel('RMSE', fontsize=11)
plt.title('Trends in Validation RMSE During Model Training', fontsize=12)
plt.legend(loc='upper right', fontsize=11)

plt.tight_layout()
plt.show()



print("Calculating Validation RMSE")

print(f"Validation RMSE: {min_val_rmse:.4f}")


# Distribution of Test Data Prediction

y_test_pred = np.mean(y_test_pred_sav, axis=0)

plt.figure(figsize=(6, 4))
sns.histplot(y_test_pred, bins=50, kde=True, color='blue')
plt.title("Distribution of Test Data Prediction")
plt.xlabel("Price")

plt.tight_layout()
plt.show()



submission = pd.DataFrame({'id': test.index, 'Price': y_test_pred})
submission.to_csv('submission.csv', index=False)
display(submission)

