import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb


PATH = '/kaggle/input/playground-series-s5e12/'

train = pd.read_csv(PATH + 'train.csv')
test = pd.read_csv(PATH + 'test.csv')


print('Train:', train.shape)
print('Test:' ,test.shape)

#Train: 70000 rows, 26 columns
#Test: 30000 rows, 25 columns


train.head()


train.info()
test.info()


train_encoded = pd.get_dummies(train, columns=['gender','ethnicity','education_level','income_level','smoking_status','employment_status'])

correlation_matrix = train_encoded.corr()

TARGET_COLUMN = 'diagnosed_diabetes'

target_corr = correlation_matrix[TARGET_COLUMN].sort_values(ascending=False)
target_corr = target_corr.drop(TARGET_COLUMN)


plt.figure(figsize=(10,8))

sns.barplot(x=target_corr.values,y=target_corr.index,palette='vlag')

plt.show()


# 1. Identify and Filter Relevant Numerical Columns
numeric_cols = train.select_dtypes(include=['number']).columns
# Remove identifier and target columns, if they exist
numeric_cols = numeric_cols.drop(['id', 'diagnosed_diabetes'], errors='ignore')

# 2. Determine the Grid Layout
num_plots = len(numeric_cols)
N_COLS = 3  # Based on your image, 3 columns is a good layout
# Calculate the required number of rows
N_ROWS = (num_plots + N_COLS - 1) // N_COLS

# 3. Create the Large Figure and Iterate
# Set the figure size based on the number of plots (e.g., 5 inches wide, 4 inches high per plot)
plt.figure(figsize=(N_COLS * 5, N_ROWS * 4))

print(f"Generating {num_plots} histograms in a {N_ROWS}x{N_COLS} grid...")

# Loop through the columns and their index (starting at 1 for subplot)
for i, col in enumerate(numeric_cols, 1):
    # Set the current subplot location: (rows, columns, plot_number)
    plt.subplot(N_ROWS, N_COLS, i)

    # Generate the histogram for the current column
    # Use .dropna() to handle any missing values safely
    plt.hist(train[col].dropna(), bins=30, edgecolor='black')

    # Set plot titles and labels
    plt.title(f'Histogram of {col}', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

# 4. Final Display
# Adjust subplot parameters for a tight layout, preventing titles/labels from overlapping
plt.tight_layout()
plt.show()
#


categorical_cols = train.select_dtypes(include=['object', 'category']).columns


num_cols = len(categorical_cols)
rows = 1
cols = num_cols

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5))

if cols == 1:
    axes = [axes]
else:
    axes = axes.flatten()


for i, col in enumerate(categorical_cols):
    sns.countplot(x=col, data=train, order=train[col].value_counts().index, ax=axes[i])
    axes[i].set_title(f'Count Plot of {col}', fontsize=14)
    axes[i].set_xlabel(col, fontsize=12)
    axes[i].set_ylabel('Count', fontsize=12)
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


for df in [train,test]:
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    #df['map'] = df['diastolic_bp'] + (1/3) * df['pulse_pressure'] #mean arterial pressure
    #df['rpp'] = df['systolic_bp'] * df['heart_rate'] #rate pressure product

    df['total_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1e-9)
    df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1e-9)
    #df['non_hdl_cholesterol'] = df ['cholesterol_total'] - df['hdl_cholesterol']
    
    df['age_bmi_interaction'] = df['age'] * df['bmi'] ##people with high age and high bmi are more likely to have diabetes
    df['physical_sleep_interaction'] = df['physical_activity_minutes_per_week'] * df['sleep_hours_per_day'] #people with less physical activity and less sleep tend to be more unhealth

 


train.info()
test.info()


X = train.drop(columns=['id','diagnosed_diabetes'])
y = train['diagnosed_diabetes']
X_test = test.drop(columns=['id'])

categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
for col in categorical_cols:
    # Set the type on both train and test sets
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_preds = np.zeros(len(X_test)) 


for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"--- Training Fold {fold + 1} ---")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05, # Slightly lower learning rate for better stability
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(100)],
        categorical_feature=categorical_cols
    )

    test_preds += model.predict_proba(X_test)[:, 1] / n_splits



fig, ax = plt.subplots(1, 2, figsize=(16, 8))

# Plot by Split (Frequency)
lgb.plot_importance(model, max_num_features=20, importance_type='split', ax=ax[0])
ax[0].set_title("Feature Importance: Split")

# Plot by Gain (Information Value)
lgb.plot_importance(model, max_num_features=20, importance_type='gain', ax=ax[1])
ax[1].set_title("Feature Importance: Gain")

plt.tight_layout()
plt.show()


submission = pd.read_csv(PATH + 'sample_submission.csv')
submission['diagnosed_diabetes'] = test_preds
submission.to_csv('/kaggle/working/submission.csv', index=False)

