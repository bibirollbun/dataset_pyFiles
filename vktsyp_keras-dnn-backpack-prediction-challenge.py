import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from cuml.preprocessing import TargetEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.utils import plot_model

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

plt.figure(figsize=(11.5, 4))

plt.subplot(1, 2, 1)
sns.histplot(train['Price'], bins=100, kde=True, color='blue')
plt.title("Train [Price] Distribution")
plt.xlabel("Price")

plt.subplot(1, 2, 2)
sns.histplot(train_ex['Price'], bins=100, kde=True, color='green')
plt.title("Train_ex [Price] Distribution")
plt.xlabel("Price")

plt.tight_layout()
plt.show()


print("Numeric Data Distribution Across Train, Test, and Train_ex Datasets")

num_cols = test.select_dtypes(include=['number']).columns

plt.figure(figsize=(11.5, len(num_cols) * 3))

for i, col in enumerate(num_cols):
    plt.subplot(len(num_cols), 3, i*3 + 1)
    sns.histplot(train[col], bins=19, color='blue')
    plt.title(f"Train [{col}] Distribution")
    plt.xlabel(col)
    
    plt.subplot(len(num_cols), 3, i*3 + 2)
    sns.histplot(test[col], bins=19, color='green')
    plt.title(f"Test [{col}] Distribution")
    plt.xlabel(col)
    
    plt.subplot(len(num_cols), 3, i*3 + 3)
    sns.histplot(train_ex[col], bins=19, color='red')
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
    train[variable].value_counts().plot.pie(ax=axes[0], autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.55), pctdistance=0.7)
    axes[0].set_ylabel('')
    axes[0].set_title(f"train [{variable}]")
    
    # Donut Chart for Test data
    test[variable].value_counts().plot.pie(ax=axes[1], autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.55), pctdistance=0.7)
    axes[1].set_ylabel('')
    axes[1].set_title(f"test [{variable}]")
    
    # Donut Chart for Train_ex data
    train_ex[variable].value_counts().plot.pie(ax=axes[2], autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.55), pctdistance=0.7)
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

# Plot donut charts and bar plots
fig, axes = plt.subplots(3, 2, figsize=(12.5, 12.5))

# Function to add value labels to bars
def add_value_labels(bars, ax):
    for bar in bars:
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{int(bar.get_width())}', va='center')

# Function to plot a single dataset
def plot_dataset(null_data, axes_row, title):
    axes_row[0].pie(null_data, labels=null_data.index, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.55), pctdistance=0.7)
    axes_row[0].set_title(f'Missing Values in {title} Dataset')
    bars = axes_row[1].barh(null_data.index, null_data.values, color='skyblue')
    axes_row[1].set_title(f'Missing Values in {title} Dataset')
    axes_row[1].set_xlabel('Count')
    axes_row[1].invert_yaxis()
    add_value_labels(bars, axes_row[1])

# Plot Charts of Missing Data for Train, Test, and Train_ex datasets
plot_dataset(train_null, axes[0], 'Train')
plot_dataset(test_null, axes[1], 'Test')
plot_dataset(train_ex_null, axes[2], 'Train_ex')
plt.tight_layout()
plt.show()



print("Visualization of Missing Data Locations in Train, Test, and Train_ex Datasets")

# Plot missing data matrix for Train dataset
msno.matrix(train, color=(0.1, 0.2, 0.4))
plt.title('Missing Data Locations in Train Dataset', fontsize=24)
plt.xlabel('Columns', fontsize=20)
plt.show()

# Plot missing data matrix for Test dataset
msno.matrix(test, color=(0.1, 0.4, 0.2))
plt.title('Missing Data Locations in Test Dataset', fontsize=24)
plt.xlabel('Columns', fontsize=20)
plt.show()

# Plot missing data matrix for Train_ex dataset
msno.matrix(train_ex, color=(0.6, 0.2, 0.1))
plt.title('Missing Data Locations in Train_ex Dataset', fontsize=24)
plt.xlabel('Columns', fontsize=20)
plt.show()


print("Visualizing Missing Values in Train, Test, and Train_ex Datasets\n")

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



# Data Normalization Using StandardScaler

columns=test.columns
scaler = StandardScaler()
train[columns] = scaler.fit_transform(train[columns])
test[columns] = scaler.transform(test[columns])



print("Train DataSet Summary (First Rows,  Shape,  Data Types)")

display(train.head(8).T, train.shape, train.dtypes)


print("Test DataSet Summary (First Rows,  Shape,  Data Types)")

display(test.head(8).T, test.shape, test.dtypes)


print("Boxplot Analysis of Normalized Train and Test Data")

def plot_combined_boxplot_grid(train, test, columns, palette='Set2'):
    sns.set_style('whitegrid')
    
    num_columns = 3
    num_rows = (len(columns) + num_columns - 1) // num_columns
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(10.5, 3 * num_rows))
    plt.subplots_adjust(wspace=0.4, hspace=0.6)

    for idx, column in enumerate(columns):
        row = idx // num_columns
        col = idx % num_columns

        combined_data = pd.DataFrame({
            'Value': list(train[column]) + list(test[column]),
            'Dataset': ['Train'] * len(train[column]) + ['Test'] * len(test[column])
        })

        sns.boxplot(x='Dataset', y='Value', data=combined_data, ax=axes[row, col], palette=palette, showfliers=False, width=0.53)
        axes[row, col].set_title(f'{column}')

    for idx in range(len(columns), num_rows * num_columns):
        axes[idx // num_columns, idx % num_columns].axis('off')
    
    plt.tight_layout()
    plt.show()

custom_palette = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
plot_combined_boxplot_grid(train, test, test.columns, palette=custom_palette)



#Correlation Heatmap Analysis for Train and Test Datasets

def plot_correlation_heatmap(data, title):
    plt.figure(figsize=(8.5, 6))
    heatmap = sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".4f", annot_kws={"size":9})
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=70, fontsize=9)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=9)
    plt.title(title)
    plt.show()

# Correlation Heatmap of Train Dataset
plot_correlation_heatmap(train, 'Correlation Heatmap of Train DataSet')

# Correlation Heatmap of Test Dataset
plot_correlation_heatmap(test, 'Correlation Heatmap of Test DataSet')



# Set the target variable 'Price' as y and features as X for training data

X = train.drop(['Price'], axis=1)
y = train['Price']


print("Features X Summary (First Rows,  Shape,  Data Types)")

display(X.head(), X.shape, X.dtypes)


print("Target y Summary (First Rows,  Shape,  Data Type)")

display(y.head(), y.shape, y.dtypes)


# Split train and validation data
train_id, val_id = train_test_split(X.index, test_size=0.2, random_state=42)
X_train, X_val = X.iloc[train_id], X.iloc[val_id]
y_train, y_val = y.iloc[train_id], y.iloc[val_id]

# DNN model
model = Sequential([
    Input(shape=(X_train.shape[1],)),  
    Dense(128, kernel_regularizer=l2(0.0001)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.1),
    
    Dense(128, kernel_regularizer=l2(0.0001)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.1),
    
    Dense(64, kernel_regularizer=l2(0.0001)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.1),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[RootMeanSquaredError()])

# Show the model summary
model.summary()
plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True, dpi=63)



# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f'Epoch {epoch+1}/{self.params["epochs"]} - Val RMSE: {logs["val_root_mean_squared_error"]}')

# Model Training
search = model.fit(
    X_train, 
    y_train, 
    epochs=50,
    batch_size=2048,
    callbacks=[early_stopping, CustomCallback()],
    validation_data=(X_val, y_val),
    verbose=0
)

# Best Val RMSE
best_rmse = min(search.history['val_root_mean_squared_error'])  
print("\nBest Val RMSE : ", best_rmse)



print("Plot the evolution of RMSE during training")

val_rmse_array = np.array(search.history['val_root_mean_squared_error'])

best_val_epoch = val_rmse_array.argmin()
best_val_rmse = val_rmse_array[best_val_epoch]

# Create figure and plot RMSE
plt.figure(figsize=(8.8, 3.6))
plt.plot(search.history['root_mean_squared_error'], label='Training RMSE')
plt.plot(search.history['val_root_mean_squared_error'], label='Validation RMSE')

plt.plot(best_val_epoch, best_val_rmse, 'ro')
plt.text(best_val_epoch+3, best_val_rmse+1, f'Best Val RMSE:{best_val_rmse:.4f}',
         fontsize=10, color='red', verticalalignment='bottom', horizontalalignment='right')

plt.title('Evolution of RMSE During Training')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.show()



print("Comparison of Validation True and Predicted Values\n")

y_true = y_val
y_pred = model.predict(X_val)

fig = plt.figure(figsize=(8, 7))
grid = plt.GridSpec(4, 4, hspace=0.05, wspace=0.05)
ax_main = fig.add_subplot(grid[1:4, 0:3])
ax_xhist = fig.add_subplot(grid[0, 0:3], sharex=ax_main)
ax_yhist = fig.add_subplot(grid[1:4, 3], sharey=ax_main)
ax_main.scatter(y_true, y_pred, alpha=0.6, edgecolors='w', linewidth=0.5)
ax_main.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', linewidth=1.0)

ax_xhist.hist(y_true, bins=30, color='blue', alpha=0.7)
ax_xhist.set_ylabel('Count')
ax_yhist.hist(y_pred, bins=30, orientation='horizontal', color='green', alpha=0.7)
ax_yhist.set_xlabel('Count')
plt.setp(ax_xhist.get_xticklabels(), visible=False)
plt.setp(ax_yhist.get_yticklabels(), visible=False)

ax_main.set_xlabel('Validation (True)', fontsize=11)
ax_main.set_ylabel('Validation (Pred)', fontsize=11)
ax_main.grid(True)
plt.show()



# Distribution of Test Data Prediction

y_test_pred = model.predict(test)

fig = plt.figure(figsize=(7, 6))
grid = plt.GridSpec(4, 1, hspace=0.1) 
ax_hist = fig.add_subplot(grid[0:3, 0]) 
ax_box = fig.add_subplot(grid[3, 0], sharex=ax_hist)

sns.histplot(y_test_pred, bins=50, kde=True, color='blue', ax=ax_hist, legend=False)
ax_hist.set_title("Distribution of Test Data Prediction")
ax_hist.set_xlabel("")

sns.boxplot(x=y_test_pred, ax=ax_box, color='blue')
ax_box.set_xlabel("Price")

plt.setp(ax_hist.get_xticklabels(), visible=False)
plt.tight_layout()
plt.show()



y_test_pred = np.array(y_test_pred).flatten() 
submission = pd.DataFrame({'id': test.index, 'Price': y_test_pred})
submission.to_csv('submission.csv', index=False)
display(submission)

