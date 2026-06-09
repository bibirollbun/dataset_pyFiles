import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.model_selection import train_test_split
from cuml.preprocessing import TargetEncoder
from xgboost import XGBRegressor, plot_importance
import warnings
warnings.filterwarnings("ignore")


#Importing Train, Test, and Training Extra Datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv',index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv',index_col='id')
train_ex = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv",index_col='id')


print("Train DataSet Summary (First Rows,  Shape,  Data Types)")

display(train.head(), train.shape, train.dtypes)


print("Test DataSet Summary (First Rows,  Shape,  Data Types)")

display(test.head(), test.shape, test.dtypes)


print("Train_ex DataSet Summary (First Rows,  Shape,  Data Types)")

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


print("Donut Chart Comparison of Categorical Variables in Train, Test, and Train_ex Datasets")

# Get the columns with object data type
obj_cols = train.select_dtypes(include=['object']).columns

for variable in obj_cols:
    sns.set_style('whitegrid')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(wspace=0.3)
    
    # Donut Chart for Train data
    train[variable].value_counts().plot.pie(ax=axes[0], autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.3))
    axes[0].set_ylabel('')
    axes[0].set_title(f"train [{variable}]")
    
    # Donut Chart for Test data
    test[variable].value_counts().plot.pie(ax=axes[1], autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.3))
    axes[1].set_ylabel('')
    axes[1].set_title(f"test [{variable}]")
    
    # Donut Chart for Train_ex data
    train_ex[variable].value_counts().plot.pie(ax=axes[2], autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.3))
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



TE = TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='mean')

features = test.columns.tolist()

for col in features:
    TE.fit(train[col], train['Price'])
    train[col] = TE.transform(train[col])
    test[col] = TE.transform(test[col])



print("Train DataSet Summary (First Rows,  Shape,  Data Types)")

display(train.head(8).T, train.shape, train.dtypes)



print("Test DataSet Summary (First Rows,  Shape,  Data Types)")

display(test.head(8).T, test.shape, test.dtypes)



def plot_combined_boxplot_grid(train, test, columns):
    sns.set_style('whitegrid')
    
    num_columns = 3
    num_rows = (len(columns) + num_columns - 1) // num_columns
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(10, 3 * num_rows))
    plt.subplots_adjust(wspace=0.4, hspace=0.6)

    for idx, column in enumerate(columns):
        row = idx // num_columns
        col = idx % num_columns

        combined_data = pd.DataFrame({
            'Value': list(train[column]) + list(test[column]),
            'Dataset': ['Train'] * len(train[column]) + ['Test'] * len(test[column])
        })

        sns.boxplot(x='Dataset', y='Value', data=combined_data, ax=axes[row, col], palette='Set2', showfliers=False)
        axes[row, col].set_title(f'{column}')

    for idx in range(len(columns), num_rows * num_columns):
        axes[idx // num_columns, idx % num_columns].axis('off')
    
    plt.tight_layout()
    plt.show()

plot_combined_boxplot_grid(train, test, test.columns)



# Correlation Heatmap of Train Dataset

plt.figure(figsize=(9, 6))
heatmap=sns.heatmap(train.corr(), annot=True, cmap='coolwarm', fmt=".4f", annot_kws={"size":9})
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=70, fontsize=9)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=9)
plt.title('Correlation Heatmap of Train DataSet')
plt.show()


# Set the target variable 'Price' as y and features as X for training data

X = train.drop(['Price'], axis=1)
y = train['Price']


print("Features X Summary (First Rows,  Shape,  Data Types)")

display(X.head(), X.shape, X.dtypes)


print("Target y Summary (First Rows,  Shape,  Data Type)")

display(y.head(), y.shape, y.dtypes)


# Split the indices of the train data
train_id, val_id = train_test_split(train.index, test_size=0.2, random_state=42)

# Split train and validation data
X_train,X_val,y_train,y_val = X.iloc[train_id],X.iloc[val_id],y.iloc[train_id],y.iloc[val_id]    

# Define XGBoost model
model = XGBRegressor(
    device="cuda",
    max_depth=5,
    n_estimators=2000,
    learning_rate=0.015,
    random_state=42
)

# Train model
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    verbose=200
)


# Trends in Validation RMSE During Model Training

evals_result = model.evals_result()
min_rmse = min(evals_result['validation_0']['rmse'])
min_index = evals_result['validation_0']['rmse'].index(min_rmse)

plt.figure(figsize=(10, 5))
plt.plot(evals_result['validation_0']['rmse'], label='Validation RMSE', color='blue')
plt.xlabel('Iteration')
plt.ylabel('RMSE')
plt.title('Trends in Validation RMSE During Model Training')
plt.scatter(min_index, min_rmse, color='red', s=50)
plt.text(min_index+50, min_rmse+0.02, f'Validation RMSE(min): {min_rmse:.3f}', color='red', fontsize=11, ha='right')
plt.legend()
plt.grid(True)
plt.show()


# Displaying Minimum Validation RMSE

print(f"Validation RMSE(min): {min_rmse:.4f}")


print("XGBoost Feature Importance Analysis")

plot_importance(model)
plt.title('Feature Importance')
plt.show()



print("Comparison of Validation True and Predicted Values")

y_true = y_val
y_pred = model.predict(X_val)

# Plot preparation
plt.figure(figsize=(7, 5))
plt.scatter(y_true, y_pred, c=y_pred, cmap='viridis', s=20, alpha=0.7, linewidth=0.5)
cb = plt.colorbar()

# Plot the diagonal line
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', linewidth=1.0)
plt.axis('equal')

plt.xlabel(f'True values (y_val)', fontsize=11)
plt.ylabel(f'Predicted values (y_val_pred)', fontsize=11)
plt.title('Validation True and Predicted Values', fontsize=12)
plt.grid(True)
plt.show()



# Distribution of Test Data Prediction

y_test_pred = model.predict(test)

plt.figure(figsize=(6, 4))
sns.histplot(y_test_pred, bins=50, kde=True, color='blue')
plt.title("Distribution of Test Data Prediction")
plt.xlabel("Price")

plt.tight_layout()
plt.show()



submission = pd.DataFrame({'id': test.index, 'Price': y_test_pred})
submission.to_csv('submission.csv', index=False)
display(submission)

