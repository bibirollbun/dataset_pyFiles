import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from keras.models import Sequential
from keras.layers import Dense, Input, BatchNormalization, Activation, Dropout, LeakyReLU
from keras.regularizers import l2
from keras.metrics import RootMeanSquaredError
from keras.callbacks import EarlyStopping, Callback
from keras.utils import plot_model

import warnings
warnings.filterwarnings("ignore")


# Load the Train and Test Datasets

train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv',index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv',index_col='id')


print("Train Dataset Summary (First Rows,  Shape,  Data Types)")

display(train.head(10).T, train.shape, train.dtypes)


print("Test Dataset Summary (First Rows,  Shape,  Data Types)")

display(test.head(10).T, test.shape, test.dtypes)


print("Missing Values Count for Train Dataset")

train.isnull().sum()


print("Missing Values Count for Test Dataset")

test.isnull().sum()


# Distribution of Target Variable

y_train = train['accident_risk']

fig = plt.figure(figsize=(10, 5))
grid = plt.GridSpec(4, 1, hspace=0.1) 
ax_hist = fig.add_subplot(grid[0:3, 0]) 
ax_box = fig.add_subplot(grid[3, 0], sharex=ax_hist)

sns.histplot(y_train, bins=30, kde=True, color='blue', ax=ax_hist, legend=False)
ax_hist.set_title("Distribution of accident_risk (Target Variable)")
ax_hist.set_xlabel("")

sns.boxplot(x=y_train, ax=ax_box, color='blue')
ax_box.set_xlabel("accident_risk")

plt.setp(ax_hist.get_xticklabels(), visible=False)
plt.tight_layout()
plt.show()


print("Distribution of Numeric Features (Train vs Test)")

num_cols = test.select_dtypes(include=['number']).columns

plt.figure(figsize=(10, len(num_cols) * 2.5))

for i, col in enumerate(num_cols):
    plt.subplot(len(num_cols), 2, i*2 + 1)
    sns.histplot(train[col], bins=24, color='blue')
    plt.title(f"Train [{col}]")
    plt.xlabel(col)
    
    plt.subplot(len(num_cols), 2, i*2 + 2)
    sns.histplot(test[col], bins=24, color='green')
    plt.title(f"Test [{col}]")
    plt.xlabel(col)
    
plt.tight_layout()
plt.show()


print("Inspecting Unique Values in Categorical Columns\n")

categorical_columns = train.select_dtypes(include=['object', 'bool']).columns

for col in categorical_columns:
    print(f"{col} : {train[col].unique()}")
    print("-" * 50)


print("Donut Chart Comparison of Categorical Variables in Train, Test Datasets")

# Get the columns with object data type
obj_cols = train.select_dtypes(include=['object', 'bool']).columns

for variable in obj_cols:
    sns.set_style('whitegrid')
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plt.subplots_adjust(wspace=0.3)
    
    # Donut Chart for Train data
    train[variable].value_counts().plot.pie(ax=axes[0], autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.6), pctdistance=0.7)
    axes[0].set_ylabel('')
    axes[0].set_title(f"train [{variable}]", fontsize=11)
    
    # Donut Chart for Test data
    test[variable].value_counts().plot.pie(ax=axes[1], autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.6), pctdistance=0.7)
    axes[1].set_ylabel('')
    axes[1].set_title(f"test [{variable}]", fontsize=11)
        
    plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# Select the first 12 features excluding 'accident_risk'
feature_cols = [col for col in train.columns if col != 'accident_risk'][:12]

# Create a 4×3 grid of subplots
fig, axes = plt.subplots(4, 3, figsize=(10, 10))
axes = axes.flatten()

for i, col in enumerate(feature_cols):
    if col == 'curvature':
        # Use scatter plot with linear regression line for 'curvature'
        sns.regplot(
            data=train,
            x=col,
            y='accident_risk',
            ax=axes[i],
            scatter=True,
            scatter_kws={
                's': 20,               # Point size
                'alpha': 0.5,          # Transparency
                'edgecolor': 'w',      # White edge around points
                'linewidths': 0.5      # Edge thickness
            },
            line_kws={
                'color': 'red',        # Regression line color
                'linewidth': 1.5       # Regression line thickness
            }
        )
    else:
        # Use boxplot for all other features
        sns.boxplot(data=train, x=col, y='accident_risk', ax=axes[i])
    
    # Set title and axis labels
    axes[i].set_title(f'{col} vs accident_risk', fontsize=11)
    axes[i].set_xlabel(col, fontsize=10)
    axes[i].set_ylabel('accident_risk', fontsize=10)

# Add overall title
plt.suptitle('Accident Risk Distribution by Feature — Train Dataset', fontsize=14, y=1.02)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()



# List of columns that will be encoded with ordinal values
ordinal_columns = [
    'road_type',
    'lighting',
    'weather',
    'time_of_day',
]

# Specify the order of categories for each column
ordinal_categories = [
    ['rural', 'urban', 'highway'],         # road_type
    ['daylight', 'dim', 'night'],          # lighting
    ['clear', 'rainy', 'foggy'],           # weather
    ['morning', 'afternoon', 'evening'],   # time_of_day
]

# Create an instance of OrdinalEncoder with the specified categories
ordinal_encoder = OrdinalEncoder(categories=ordinal_categories)

# Apply the OrdinalEncoder to the training data and test data
train[ordinal_columns] = ordinal_encoder.fit_transform(train[ordinal_columns]).astype('int8')
test[ordinal_columns] = ordinal_encoder.transform(test[ordinal_columns]).astype('int8')



print("Train Dataset Summary (First Rows,  Shape,  Data Types)")

display(train.head(10).T, train.shape, train.dtypes)


print("Test Dataset Summary (First Rows,  Shape,  Data Types)")

display(test.head(10).T, test.shape, test.dtypes)


# Data Normalization Using StandardScaler

columns=test.columns
scaler = StandardScaler()
train[columns] = scaler.fit_transform(train[columns])
test[columns] = scaler.transform(test[columns])


print("Train Dataset Summary (First Rows,  Shape,  Data Types)")

display(train.head(8).T, train.shape, train.dtypes)


print("Test Dataset Summary (First Rows,  Shape,  Data Types)")

display(test.head(8).T, test.shape, test.dtypes)


# Correlation Heatmap of Train Dataset

plt.figure(figsize=(11, 7))
heatmap=sns.heatmap(train.corr(), annot=True, cmap='coolwarm', fmt=".4f", annot_kws={"size":9})
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=70, fontsize=9)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=9)
plt.title('Correlation Heatmap of Train Dataset')
plt.show()


# Correlation Heatmap of Test Dataset

plt.figure(figsize=(11, 7))
heatmap=sns.heatmap(test.corr(), annot=True, cmap='coolwarm', fmt=".4f", annot_kws={"size":9})
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=70, fontsize=9)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=9)
plt.title('Correlation Heatmap of Test DataSet')
plt.show()


# Target variable
target_column = 'accident_risk'

# Compute correlation matrix
correlation_matrix = train.corr()

# Extract and sort absolute correlations with the target
correlation_with_target = correlation_matrix[target_column].drop(target_column).abs()
correlation_with_target_sorted = correlation_with_target.sort_values(ascending=False)

# Plot horizontal bar chart of feature correlations
plt.figure(figsize=(10, 6))
ax = correlation_with_target_sorted.plot(kind='barh', color='steelblue', edgecolor='black')

for index, value in enumerate(correlation_with_target_sorted):
    if not (pd.isna(value) or value == float('inf') or value == float('-inf')):
        plt.text(value + 0.002, index, f"{value:.4f}", va='center', fontsize=10)

ax.invert_yaxis()
plt.xlim(0, correlation_with_target_sorted.max() + 0.01)
plt.title("Feature Importance Relative to Target", fontsize=12)
plt.xlabel("Correlation Coefficient", fontsize=11)
plt.ylabel("Feature", fontsize=10)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout(pad=2)
plt.show()



# Prepare input data
X_train = train.drop(['accident_risk'], axis=1)
y_train = train['accident_risk']

# Split the dataset into training and validation data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


print("X_train Summary: First Rows, Shape, Dtypes")

display(X_train.head(10).T, X_train.shape, X_train.dtypes)


print("y_train Summary (First Rows,  Shape,  Data Types)")

display(y_train.head(10), y_train.shape)


#===================================================================
# DNN Model : Dropout Removed to Encourage Overfitting
#           : Activation Function Changed: ReLU → LeakyReLU
#===================================================================
# MLP-based DNN Model
model = Sequential([
    Input(shape=(X_train.shape[1],)),  
    Dense(128, kernel_regularizer=l2(0.0001)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1), 
#    Dropout(0.1),
    
    Dense(64, kernel_regularizer=l2(0.0001)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1), 
#    Dropout(0.1),
    
    Dense(128, kernel_regularizer=l2(0.0001)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1), 
#    Dropout(0.1),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[RootMeanSquaredError()])

# Show the model summary
model.summary()
plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True, dpi=63)


# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 20 == 0:  
            print(f'Epoch {epoch+1}/{self.params["epochs"]} - Val RMSE: {logs["val_root_mean_squared_error"]:.6f}')

# Model Training
search = model.fit(
    X_train, 
    y_train, 
    epochs=200,
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
plt.text(best_val_epoch+0, best_val_rmse*2.0, f'Best Val RMSE:{best_val_rmse:.4f}',
         fontsize=10, color='red', verticalalignment='bottom', horizontalalignment='center')

plt.title('Evolution of RMSE During Training')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.show()



# Predict on training and validation data
y_train_pred = model.predict(X_train)
y_val_pred   = model.predict(X_val)

# Calculate RMSE
RMSE_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
RMSE_val   = np.sqrt(mean_squared_error(y_val, y_val_pred))

print(f"RMSE_train : {RMSE_train:.4f}")
print(f"RMSE_val   : {RMSE_val:.4f}")


print("Comparison of True vs Predicted Values (Validation Set)")

y_true = y_val
y_pred = y_val_pred

fig = plt.figure(figsize=(9, 8))
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

ax_main.set_xlabel('Actual Values (Validation)', fontsize=11)
ax_main.set_ylabel('Predicted Values (Validation)', fontsize=11)
ax_main.grid(True)
plt.show()


# Predictions on Test Data

y_test_pred = model.predict(test)


print("Prediction Summary (Values, Shape, Dtype)\n")

print(y_test_pred.reshape(-1, 1), "\n")
print("Shape:", y_test_pred.shape, "\n")
print("Dtype:", y_test_pred.dtype)


print("Distribution of Predicted Values on Test Data")

fig = plt.figure(figsize=(10, 5))
grid = plt.GridSpec(4, 1, hspace=0.1) 
ax_hist = fig.add_subplot(grid[0:3, 0]) 
ax_box = fig.add_subplot(grid[3, 0], sharex=ax_hist)

sns.histplot(y_test_pred, bins=30, kde=True, color='blue', ax=ax_hist, legend=False)
ax_hist.set_title("Distribution of Test Predictions")
ax_hist.set_xlabel("")

sns.boxplot(x=y_test_pred, ax=ax_box, color='blue')
ax_box.set_xlabel("accident_risk")

plt.setp(ax_hist.get_xticklabels(), visible=False)
plt.tight_layout()
plt.show()




print("Distribution of Predictions and True Values (Train / Validation / Test)")

fig, axes = plt.subplots(3, 2, figsize=(9, 7))  # 3 rows × 2 columns
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Define data and corresponding subplot titles
data_pairs = [
    (y_train_pred, y_train, "Train"),
    (y_val_pred, y_val, "Validation"),
    (y_test_pred, None, "Test")
]

for i, (pred, true, label) in enumerate(data_pairs):
    row = i

    # Plot Predictions
    ax_pred = axes[row][0]
    sns.histplot(pred, bins=30, kde=True, color='blue', ax=ax_pred, legend=False)
    ax_pred.set_xlim(0, 1)
    ax_pred.set_xlabel("accident_risk")
    ax_pred.set_title(f"{label} Predictions", fontsize=10)

    # Get y-axis limit from prediction plot
    y_max = ax_pred.get_ylim()[1]

    # Plot True Values
    ax_true = axes[row][1]
    if true is not None:
        sns.histplot(true, bins=30, kde=True, color='green', ax=ax_true, legend=False)
        ax_true.set_xlim(0, 1)
        ax_true.set_ylim(0, y_max)  # Match y-axis to prediction
        ax_true.set_xlabel("accident_risk")
        ax_true.set_title(f"{label} True Values", fontsize=10)
    else:
        ax_true.axis('off')  # Hide empty subplot

plt.tight_layout()
plt.show()


# Load sample submission template
submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

# Display the sample submission DataFrame
display(submission)


# Insert predicted values into the 'accident_risk' column
submission['accident_risk'] = y_test_pred

# Save the completed submission file as CSV
submission.to_csv('submission.csv', index=False)

# Display the submission DataFrame
display(submission)

