# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np, pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv', index_col = 0)
train_ext = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv',index_col = 0) 
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv', index_col = 0)


print(f'Shape of train data is {train.shape}')
print(f'Shape of train extra data is {train_ext.shape}')
print(f'Shape of test is {train_ext.shape}')
train.head()


all_train = pd.concat([train, train_ext])
print(all_train.shape)
all_train.head()


print(f'Columns are: {train.columns}')


all_train.dtypes


num_columns = ['Weight Capacity (kg)','Compartments']
target_columns = ['Price']
cat_columns = ['Brand', 'Material','Laptop Compartment','Waterproof', 'Color','Style', 'Size']


train_miss = all_train.isna().sum()
n = len(all_train)
train_miss.index

miss_value = pd.DataFrame({'columns': train.columns, 'train_miss_count': train_miss.values, '% train_miss': 100*train_miss.values/n})

test_miss = test.isna().sum()
test_miss_value = pd.DataFrame({'columns': test.columns, 'test_miss_count': test_miss.values, '% test_miss': 100*test_miss.values/n})

miss_value = pd.merge(miss_value,test_miss_value, on= 'columns', how = 'left')
miss_value


print(f'Number of duplicate in train set is {all_train.duplicated().sum()}')
print(f'Number of duplicate in test set is {test.duplicated().sum()}')


def create_distribution(data):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
    
    # Plot histogram with density
    sns.histplot(data, bins=30, kde=True, color="royalblue", edgecolor="black", alpha=0.7, ax = axes[0])
    
    # Customize appearance
    axes[0].set_title("Histogram & Distribution", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Value", fontsize=12)
    axes[0].set_ylabel("Density", fontsize=12)
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)
    
    # Add vertical lines for mean and median
    axes[0].axvline(np.mean(data), color="red", linestyle="dashed", linewidth=2, label=f"Mean: {np.mean(data):.2f}")
    axes[0].axvline(np.median(data), color="green", linestyle="dashed", linewidth=2, label=f"Median: {np.median(data):.2f}")
    
    # Show legend
    axes[0].legend()
    
    sns.boxplot(data, ax = axes[1], color = 'orange',orient='h')
    # Customize boxplot appearance
    axes[1].set_title("Box Plot", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Value", fontsize=12)
    axes[1].grid(axis="x", linestyle="--", alpha=0.7)
    
    # Show the plot
    plt.show()


create_distribution(train['Weight Capacity (kg)'])
create_distribution(train['Compartments'])


def create_bar_plot(data, column_name, target_name):
    
    value_counts = data[column_name].value_counts(dropna=False)
    categories = np.array(value_counts.index.values).astype(str)
    # Convert counts to percentages
    percentages = value_counts / value_counts.sum() * 100
    
    # Create the figure
    fig, ax=plt.subplots(nrows=1, ncols=2, figsize=(14, 4))
    
    # Create bar plot
    sns.barplot(x=categories, y=percentages, ax = ax[0])
    
    # Annotate bars with percentage values
    for i, value in enumerate(percentages):
        ax[0].text(i, value + 1, f"{value:.1f}%", ha='center', fontsize=12, fontweight='bold')
    
    # Customize appearance
    ax[0].set_title(column_name, fontsize=12, fontweight='bold')
    ax[0].set_xlabel("Category", fontsize=10)
    ax[0].set_ylabel("Percentage (%)", fontsize=10)
    ax[0].set_ylim(0, max(percentages) + 5)  # Adjust y-axis to fit labels
    ax[0].grid(axis="y", linestyle="--", alpha=0.7)
    
    sns.boxplot(x=column_name, y=target_name, data=data, showfliers=True, ax = ax[1])
    
    # Customize appearance
    ax[1].set_title(f"{target_name} Distribution by {column_name}", fontsize=12, fontweight='bold')
    ax[1].set_xlabel(column_name, fontsize=10)
    ax[1].set_ylabel(target_name, fontsize=10)
    ax[1].grid(axis="y", linestyle="--", alpha=0.7)
    
    # Show the plot
    plt.show()


for cl in cat_columns:
    create_bar_plot(all_train, cl, "Price")


for col in cat_columns:
    col_mode = all_train[col].mode()[0]
    all_train[col].fillna(col_mode, inplace=True)
    test[col].fillna(col_mode, inplace=True)

for col in num_columns:
    col_mean = all_train[col].median()
    all_train[col].fillna(col_mean, inplace=True)
    test[col].fillna(col_mean, inplace=True)


all_train.isna().sum()


test.isna().sum()


from scipy.stats import f_oneway


def influence_analize(data, column, target_colum):
    categories = data[column].unique()
    cat_groups = [data[data[column] == cat][target_colum] for cat in categories]
    
    f_stat, p_value = f_oneway(*cat_groups)
    # Compute Effect Size (η²)
    ss_between = sum(len(g) * (np.mean(g) - np.mean(data[target_colum]))**2 for g in cat_groups)
    ss_total = sum((data[target_colum] - np.mean(data[target_colum]))**2)
    eta_squared = ss_between / ss_total
    
    # Print results
    print(f"{column}")
    print(f"ANOVA F-statistic: {f_stat:.4f}, p-value: {p_value:.10f}")
    print(f"Effect Size (η²): {eta_squared:.4f}")
    
    # Interpretation
    if eta_squared > 0.14:
        print(f"Large effect size → {column} strongly influences {target_colum}.")
    elif eta_squared > 0.06:
        print(f"Medium effect size → {column} has some influence on {target_colum}.")
    else:
        print(f"Small effect size → {column} has little influence on {target_colum}.")
    print("\n")


for cl in cat_columns:
    influence_analize(all_train, cl, "Price")


from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# Apply LabelEncoder to all categorical columns
label_encoders = {}  # Store encoders for inverse transformation if needed

for col in  cat_columns:
    le = LabelEncoder()
    all_train[col] = le.fit_transform(all_train[col])  # Encode column
    test[col] = le.transform(test[col])
    label_encoders[col] = le  # Store the encoder

num_scaler = StandardScaler()
all_train[num_columns] = num_scaler.fit_transform(all_train[num_columns])
test[num_columns] = num_scaler.transform(test[num_columns])



Y_all = all_train['Price']
X_all = all_train.drop('Price', axis=1)
X_all.shape, Y_all.shape


# Define Cross-Validation strategy
kf = KFold(n_splits=10, shuffle=True, random_state=42)

catboost_params = {
    "iterations": 300,
    "learning_rate": 0.1,
    "depth": 8,
    "verbose": 0,
    "random_seed": 42
}

# Lists to store results
rmse_scores = []
mae_scores = []
oof_preds = np.zeros(len(X_all))
test_preds_cb = np.zeros(len(test))

# Store feature importances
feature_list = np.zeros(X_all.shape[1])

# Perform K-Fold Cross Validation
print("Training using Cross-Validation...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X_all)):
    print(f"\nTraining Fold {fold+1}...")

    X_train, X_val = X_all.iloc[train_idx], X_all.iloc[val_idx]
    y_train, y_val = Y_all.iloc[train_idx], Y_all.iloc[val_idx]

    # Define model
    cb_model = CatBoostRegressor(**catboost_params)

    # Train model
    cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=0)

    # Predict on validation set
    val_preds_cb = cb_model.predict(X_val)
    oof_preds[val_idx] = val_preds_cb

    # Calculate and store scores
    rmse = np.sqrt(mean_squared_error(y_val, val_preds_cb))
    mae = mean_absolute_error(y_val, val_preds_cb)
    rmse_scores.append(rmse)
    mae_scores.append(mae)

    print(f"Fold {fold+1} RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # Accumulate feature importances
    #feature_importance_list += cb_model.get_feature_importance() / kf.get_n_splits()

    # Predict on test data and average across folds
    test_preds_cb += cb_model.predict(test) / kf.get_n_splits()


mean_rmse = np.mean(rmse_scores)
mean_mae = np.mean(mae)
print(f'Mean RMSE {mean_rmse}')
print(f'Mean MAE {mean_mae}')


plt.figure(figsize=(8, 5))
plt.plot(range(1, len(rmse_scores) + 1), rmse_scores, marker='o', linestyle='--', color='b', label='RMSE per Fold')
plt.axhline(y=mean_rmse, color='r', linestyle='-', label=f'Avg RMSE: {mean_mae:.4f}')
plt.xlabel('Fold')
plt.ylabel('RMSE')
plt.title('RMSE per Fold')
plt.legend()
plt.show()


print(len(test_preds_cb))
test_preds_cb


submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv', index_col = 0)


submission['Price'] = test_preds_cb


submission


submission.to_csv("submission.csv")




