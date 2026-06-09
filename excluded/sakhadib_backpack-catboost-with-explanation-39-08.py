import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv").drop('id', axis=1)
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv").drop('id', axis=1)

train.info()


# Define categorical columns
cat_cols = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment',
            'Waterproof', 'Style', 'Color']

# Convert columns to 'category' type and then add 'None' to the categories before filling missing values
for col in cat_cols:
    # Convert column to 'category' type if not already
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')
    
    # Add 'None' to the categories for these columns
    train[col] = train[col].cat.add_categories('None')
    test[col] = test[col].cat.add_categories('None')
    
    # Fill missing values with 'None'
    train[col] = train[col].fillna('None')
    test[col] = test[col].fillna('None')

# Handle missing values for 'Weight Capacity (kg)' and convert it to string
median_weight = train['Weight Capacity (kg)'].median()
train['Weight Capacity (kg)'] = train['Weight Capacity (kg)'].fillna(median_weight).astype('string')
test['Weight Capacity (kg)'] = test['Weight Capacity (kg)'].fillna(median_weight).astype('string')

# Create 'laptop_and_waterproof' column based on the logical AND operation
train['laptop_and_waterproof'] = (train['Laptop Compartment'] == 'Yes') & (train['Waterproof'] == 'Yes')
train['laptop_and_waterproof'] = train['laptop_and_waterproof'].map({True: 'Yes', False: 'No'})

test['laptop_and_waterproof'] = (test['Laptop Compartment'] == 'Yes') & (test['Waterproof'] == 'Yes')
test['laptop_and_waterproof'] = test['laptop_and_waterproof'].map({True: 'Yes', False: 'No'})

# Display updated DataFrame structure
train.info()

# Show the first few rows of the train dataset
train.head()



test[cat_cols] = test[cat_cols].fillna('None').astype('string').astype('category')
test['Weight Capacity (kg)'] = test['Weight Capacity (kg)'].fillna(median_weight).astype('string')

test.info()


# Convert 'Weight Capacity (kg)' and 'Compartments' to numeric values and handle errors
for df in [train, test]:
    df['Weight Capacity (kg)'] = pd.to_numeric(df['Weight Capacity (kg)'], errors='coerce')
    df['Compartments'] = pd.to_numeric(df['Compartments'], errors='coerce')

    # Create 'weight per compartment' while handling division by zero and missing values
    df['weight per compartment'] = df['Weight Capacity (kg)'] / df['Compartments']

    # Replace NaN with 0 (for missing values) and infinity with 0 (for division by zero)
    df['weight per compartment'] = df['weight per compartment'].fillna(0)
    df['weight per compartment'] = df['weight per compartment'].replace([float('inf'), -float('inf')], 0)



train.head()


# Convert 'Weight Capacity (kg)' to a numeric type (float)
train['Weight Capacity (kg)'] = pd.to_numeric(train['Weight Capacity (kg)'], errors='coerce')
test['Weight Capacity (kg)'] = pd.to_numeric(test['Weight Capacity (kg)'], errors='coerce')

# Calculate median weight for categorization
median_weight = train['Weight Capacity (kg)'].median()

# Create 'Material_Weight' column by categorizing based on the weight
train['Material_Weight'] = train['Material'].astype(str) + "_" + (train['Weight Capacity (kg)'] > median_weight).map({True: 'Heavy', False: 'Light'})
test['Material_Weight'] = test['Material'].astype(str) + "_" + (test['Weight Capacity (kg)'] > median_weight).map({True: 'Heavy', False: 'Light'})

# Check the updated DataFrame structure
train.info()

# Show the first few rows of the train dataset
train.head()


train['laptop_and_waterproof'] = train['laptop_and_waterproof'].astype('category')
train['Material_Weight'] = train['Material_Weight'].astype('category')

test['laptop_and_waterproof'] = test['laptop_and_waterproof'].astype('category')
test['Material_Weight'] = test['Material_Weight'].astype('category')

train.info()


sns.set(style="whitegrid", palette="rocket", font_scale=1)

plt.figure(figsize=(8, 6))
sns.histplot(train['Price'], kde=True, bins=30, color='skyblue')
plt.title("Target distribution")
plt.xlabel("Price")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


# Create the subplots
fig, axes = plt.subplots(1, 5, figsize=(20, 6))

# List of column names and titles for each subplot
columns = ['Brand', 'Material', 'Size', 'Style', 'Color']
titles = ['Brand vs Price', 'Materials vs Price', 'Size vs Price', 'Style vs Price', 'Color vs Price']

# Create the box plots
for i, col in enumerate(columns):
    sns.boxplot(data=train, x=col, y='Price', ax=axes[i])
    axes[i].set_title(titles[i])
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Price')
    axes[i].tick_params(axis='x', rotation=45)

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()


fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 16))

for ax, col in zip(axes.flatten(), cat_cols):
    train[col].value_counts().plot(
        kind='barh', 
        color='g', 
        title=f'Backpacks {col}',
        ax=ax 
    )

plt.tight_layout() 
plt.show()


print(train.info())


from sklearn.preprocessing import LabelEncoder

# Categorical columns to be label encoded
categorical_columns = [
    'Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style',
    'Color', 'laptop_and_waterproof', 'Material_Weight'
]

# Initialize the label encoder
label_encoder = LabelEncoder()

# Apply label encoding to the categorical columns in both train and test dataframes
for col in categorical_columns:
    train[col] = label_encoder.fit_transform(train[col])
    test[col] = label_encoder.transform(test[col])

train.info()


train.head()


X = train.drop('Price', axis=1)
y = train.Price


catboost_params = {
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'learning_rate': 0.05550266178302702,
        'iterations': 2000,
        'depth': 4,
        'random_strength': 0,
        'l2_leaf_reg': 5.189087598805998,
        'task_type':'GPU',
        'random_seed': 42,
        'verbose': False    
    }


# Define KFold cross-validation
cv = KFold(5, shuffle=True, random_state=0)
cv_splits = cv.split(X, y)
scores = []
test_preds = []

# No need to specify categorical columns as all columns are numerical
X_test_pool = Pool(test)





for train_idx, val_idx in cv_splits:
    model = CatBoostRegressor(**catboost_params)
    
    X_train_fold, X_val_fold = X.loc[train_idx], X.loc[val_idx]
    y_train_fold, y_val_fold = y.loc[train_idx], y.loc[val_idx]
    
    X_train_pool = Pool(X_train_fold, y_train_fold)
    X_valid_pool = Pool(X_val_fold, y_val_fold)
    
    model.fit(X=X_train_pool, eval_set=X_valid_pool, verbose=100, early_stopping_rounds=200)
    
    val_pred = model.predict(X_valid_pool)
    
    score = np.sqrt(mean_squared_error(y_val_fold, val_pred))
    
    scores.append(score)
    
    test_pred = model.predict(X_test_pool)
    
    test_preds.append(test_pred)
    
print(f'Cross-validated RMSE score: {np.mean(scores):.3f} +/- {np.std(scores):.3f}')
print(f'Max RMSE score: {np.max(scores):.3f}')
print(f'Min RMSE score: {np.min(scores):.3f}')


sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
sample_submission['Price'] = np.mean(test_preds, axis=0)
sample_submission.to_csv('submission.csv', index=False)
sample_submission.head(10)

