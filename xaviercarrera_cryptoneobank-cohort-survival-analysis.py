import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from scipy import stats

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import log_loss, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
import shap
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Don't do the following without adult supervision :)
pd.set_option('display.max_columns', 500) 
pd.options.display.float_format = '{:.3f}'.format

import warnings
warnings.filterwarnings("ignore")


# Open and concat training files
file_pattern = '/kaggle/input/neo-bank-non-sub-churn-prediction/train_*.parquet' 

all_files = glob.glob(file_pattern)

raw_data = pd.concat([pd.read_parquet(file) for file in all_files])
raw_data


# Check for coincidences in the churns columns

coincidences = len(raw_data['churn_due_to_fraud'] == raw_data['model_predicted_fraud'])
total_data = len(raw_data)
percent_coincidences = (coincidences / total_data) * 100

print(f'Total data: {total_data}')
print(f'Total coincidences: {coincidences}')
print(f'% of coincidences: {percent_coincidences:.2f}%')


raw_data.info()


raw_data.isnull().mean()


# Count unique customers

total_customers = raw_data['customer_id'].nunique()

total_customers


raw_data.describe().T


raw_data['churn_due_to_fraud'].value_counts()


# Calculate the number of unique values per column for each customer
unique_values_per_col = raw_data.drop(columns=['Id', 'date', 'touchpoints', 'csat_scores']) \
                                .groupby('customer_id').nunique()

# Sum unique values across all customers for each column
values_differences = pd.DataFrame(unique_values_per_col.sum(), columns=['Total Different Values'])

# Calculate the mean frequency of change per column
values_differences['Frequency of Change'] = values_differences['Total Different Values'] / total_customers

values_differences


df_cohort = raw_data.copy()

df_cohort['year_month'] = df_cohort['date'].dt.to_period('M')

activity_columns = [
    'atm_transfer_in', 'atm_transfer_out',
    'bank_transfer_in', 'bank_transfer_out',
    'crypto_in', 'crypto_out',
    'bank_transfer_in_volume', 'bank_transfer_out_volume',
    'crypto_in_volume', 'crypto_out_volume'
]

monthly_activity = (
    df_cohort.groupby(['customer_id', 'year_month'], as_index=False)[activity_columns]
      .sum()
)

monthly_activity['active'] = (
    monthly_activity[activity_columns].sum(axis=1).gt(0).astype(int)
)

monthly_activity = monthly_activity.sort_values(['customer_id','year_month'])
monthly_activity['next_month_active'] = monthly_activity.groupby('customer_id')['active'].shift(-1)
monthly_activity['churn'] = (
    (monthly_activity['active'] == 1) &
    (monthly_activity['next_month_active'].fillna(0) == 0)
).astype(int)

monthly_activity


# Step 1: Assign cohorts (first active month per customer)
monthly_activity['cohort'] = monthly_activity.groupby('customer_id')['year_month'].transform('min')

# Step 2: Calculate the period since the cohort start
monthly_activity['period'] = (monthly_activity['year_month'] - monthly_activity['cohort']).apply(lambda x: x.n)

# Step 3: Aggregate active customers by cohort and period
cohort_data = (
    monthly_activity
    .groupby(['cohort', 'period'])['customer_id']
    .nunique()  # Count unique active customers
    .unstack(1)  # Pivot to get periods as columns
    .fillna(0)   # Fill NaNs with 0
)

# Step 4: Calculate survival rates
cohort_sizes = cohort_data.iloc[:, 0]  # Size of each cohort at period 0
survival_rates = cohort_data.divide(cohort_sizes, axis=0) * 100  # Convert to percentages


# Convert the first column to represent cohort sizes (assume it corresponds to cohort size at period 0)
cohort_data.columns = ['Period_0'] + [f'Period_{i}' for i in range(1, cohort_data.shape[1])]

# Calculate survival rates as percentages
cohort_sizes = cohort_data['Period_0']
survival_rates = cohort_data.div(cohort_sizes, axis=0) * 100

# Calculate median survival rates for key periods to understand trends
median_survival_by_period = survival_rates.median()

# Plot the median survival rates over time to observe overall trends
plt.figure(figsize=(12, 6))
median_survival_by_period.plot(kind='line', marker='o')
plt.title("Median Survival Rate Over Time")
plt.xlabel("Periods Since Cohort Start")
plt.ylabel("Median Survival Rate (%)")
plt.grid()
plt.show()

# Visualize the full cohort survival matrix using a heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(
    survival_rates,
    cmap="Blues",
    annot=False,  # Optional: set to True if you want to annotate each cell
    fmt=".1f",
    cbar_kws={'label': 'Survival Rate (%)'}
)
plt.title("Cohort Survival Heatmap")
plt.xlabel("Periods Since Cohort Start")
plt.ylabel("Cohort (Starting Period)")
plt.show()


# Calculate the last active period per customer
last_active_periods = monthly_activity[monthly_activity['active'] == 1].groupby('customer_id')['period'].max()

# Calculate the average last active period
print(f'On average, it takes {last_active_periods.mean():.2f} periods for a customer to reach their last activity')
print(f'Median: {last_active_periods.median():.2f}')
print(f'Record low: {last_active_periods.min():.2f}')
print(f'Record high: {last_active_periods.max():.2f}')



# Variables of interest
variables = [
    'atm_transfer_in', 'atm_transfer_out', 'bank_transfer_in', 'bank_transfer_out',
    'crypto_in', 'crypto_out', 'bank_transfer_in_volume', 'bank_transfer_out_volume',
    'crypto_in_volume', 'crypto_out_volume'
]

# Group by customer and time (e.g., year_month) to calculate monthly totals
monthly_trends = monthly_activity.groupby(['customer_id', 'year_month'])[variables].sum()

# Reset index for easier visualization
monthly_trends = monthly_trends.reset_index()

monthly_trends


# Step 1: Assign cohorts and periods
monthly_activity['cohort'] = monthly_activity.groupby('customer_id')['year_month'].transform('min')
monthly_activity['period'] = (monthly_activity['year_month'] - monthly_activity['cohort']).apply(lambda x: x.n)

# Step 2: Aggregate variables by cohort and period
aggregated_data = (
    monthly_activity
    .groupby(['cohort', 'period'])[variables]
    .mean()
    .reset_index()
)

# Step 3: Calculate the mean and confidence intervals
# Group by period to calculate statistics
stats = aggregated_data.groupby('period')[variables].agg(['median', 'std', 'count'])

# Add confidence intervals (95%)
for var in variables:
    stats[(var, 'ci_upper')] = stats[(var, 'median')] + 1.96 * stats[(var, 'std')] / np.sqrt(stats[(var, 'count')])
    stats[(var, 'ci_lower')] = stats[(var, 'median')] - 1.96 * stats[(var, 'std')] / np.sqrt(stats[(var, 'count')])

# Step 4: Create subplots for each variable
fig, axes = plt.subplots(len(variables), 1, figsize=(12, 4 * len(variables)), sharex=True)

for i, variable in enumerate(variables):
    ax = axes[i]
    
    # Extract mean and confidence intervals
    mean = stats[(variable, 'median')]
    ci_upper = stats[(variable, 'ci_upper')]
    ci_lower = stats[(variable, 'ci_lower')]
    
    # Plot mean with confidence intervals
    ax.plot(stats.index, mean, label=f'Mean {variable}', color='blue')
    ax.fill_between(stats.index, ci_lower, ci_upper, color='blue', alpha=0.2, label='95% CI')
    
    # Configure subplot
    ax.set_title(f"Trends Over Periods: {variable}")
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid()

# Configure shared x-axis
axes[-1].set_xlabel('Periods Since Start')
plt.tight_layout()
plt.show()


# Step 1: Define the reference date
reference_date = raw_data['date'].max()

# Step 2: Calculate the last recorded activity per customer
last_activity = raw_data.groupby('customer_id')['date'].max().reset_index()
last_activity.columns = ['customer_id', 'last_activity_date']

# Step 3: Calculate the difference in days
last_activity['days_since_last_activity'] = (reference_date - last_activity['last_activity_date']).dt.days

# Step 4: Define churn flag
churn_threshold_days = 6 * 30  # n months
last_activity['churn'] = (last_activity['days_since_last_activity'] > churn_threshold_days).astype(int)

# Step 5: Get the list of churned customer_ids
churned_customers = last_activity[last_activity['churn'] == 1]['customer_id'].tolist()

print(f'Total churned customers: {len(churned_customers)}')


df_model = raw_data.copy()

df_model = df_model.groupby(['customer_id']).agg(
    ## Crypto IN
    total_crypto_in=('crypto_in', 'sum'),
    mean_crypto_in=('crypto_in', 'mean'),
    max_crypto_in=('crypto_in', 'max'),
    ## Crypto OUT
    total_crypto_out=('crypto_out', 'sum'),
    mean_crypto_out=('crypto_out', 'mean'),
    max_crypto_out=('crypto_out', 'max'),
    ## Bank Transfer IN
    total_bank_transfer_in_volume=('bank_transfer_in_volume', 'sum'),
    mean_bank_transfer_in_volume=('bank_transfer_in_volume', 'mean'),
    max_bank_transfer_in_volume=('bank_transfer_in_volume', 'max'),
    ## Bank Transfer OUT
    total_bank_transfer_out_volume=('bank_transfer_out_volume', 'sum'),
    mean_bank_transfer_out_volume=('bank_transfer_out_volume', 'mean'),
    max_bank_transfer_out_volume=('bank_transfer_out_volume', 'max'),
    ## Crypto Vol IN
    total_crypto_in_volume=('crypto_in_volume', 'sum'),
    mean_crypto_in_volume=('crypto_in_volume', 'mean'),
    max_crypto_in_volume=('crypto_in_volume', 'max'),
    ## Crypto Vol OUT
    total_crypto_out_volume=('crypto_out_volume', 'sum'),
    mean_crypto_out_volume=('crypto_out_volume', 'mean'),
    max_crypto_out_volume=('crypto_out_volume', 'max'),
    ## Dates
    first_date=('date', 'min'),
    last_date=('date', 'max')
).reset_index()

# Compute the time since the customer's first and last activity
df_model['days_since_opening'] = (reference_date - df_model['first_date']).dt.days
df_model['days_since_last_activity'] = (reference_date - df_model['last_date']).dt.days

df_model.drop(columns=['first_date','last_date'], inplace=True)

# Build target variable
df_model['target'] = np.where(df_model['customer_id'].isin(churned_customers), 1, 0)

df_model


df_model['target'].value_counts()


independent_variables = df_model.drop(['customer_id','target'], axis=1).columns

for var in independent_variables:

    sns.boxplot(x='target', y=var, data=df_model)
    ax = sns.boxplot(x='target', y=var, data=df_model)
    plt.title(f'Boxplot for {var}')
    plt.xlabel('Churn')
    plt.ylabel(f'{var}')
    plt.show()


for var in independent_variables:
    plt.figure(figsize=(8, 6))  
    
    # Plot histogram
    sns.histplot(data=df_model, x=var, hue='target', kde=True, bins=30)
    
    # Add titles and labels
    plt.title(f'Histogram for {var}')
    plt.xlabel(f'{var})')
    plt.ylabel('Frequency')
    
    plt.show()


from scipy import stats

# Ensure no missing data
df_model[independent_variables] = df_model[independent_variables].fillna(0)

# Create empty lists
t_test_results = []
z_test_results = []

# Loop through each variable
for var in independent_variables:
    churned = df_model[df_model['target'] == 1][var].values
    not_churned = df_model[df_model['target'] == 0][var].values

    # Skip if insufficient data
    if len(churned) < 2 or len(not_churned) < 2:
        continue

    # Perform t-test
    t_stat, t_p_value = stats.ttest_ind(churned, not_churned, equal_var=False, nan_policy='omit')
    t_test_results.append((var, t_stat, t_p_value))

    # Perform z-test
    std_churned = churned.std()
    std_not_churned = not_churned.std()
    n_churned = len(churned)
    n_not_churned = len(not_churned)

    if n_churned > 1 and n_not_churned > 1:
        z_stat = (std_churned - std_not_churned) / np.sqrt(
            (std_churned**2 / (2 * n_churned)) + (std_not_churned**2 / (2 * n_not_churned))
        )
        z_p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        z_test_results.append((var, z_stat, z_p_value))

# Convert to DataFrames
t_test_df = pd.DataFrame(t_test_results, columns=['Variable', 'T-Statistic', 'P-Value'])
z_test_df = pd.DataFrame(z_test_results, columns=['Variable', 'Z-Statistic', 'P-Value'])


t_test_df


z_test_df


X = df_model[independent_variables]
y = df_model['target']

# Apply SMOTE for balancing
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Convert back to a DataFrame
df_balanced = pd.DataFrame(X_balanced, columns=independent_variables)
df_balanced['target'] = y_balanced

df_balanced.head()


df_balanced.shape


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)


# Logistic Regression
logistic_model = LogisticRegression(random_state=42, solver='lbfgs', max_iter=1000)
logistic_model.fit(X_train, y_train)

# Cross-validation
logloss_lr = -cross_val_score(logistic_model, X_balanced, y_balanced, cv=5, scoring='neg_log_loss')
print(f"Logistic Regression Logloss (CV): {logloss_lr.mean():.4f}")

# Metrics
y_pred_lr = logistic_model.predict(X_test)
y_pred_proba_lr = logistic_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred_lr))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_lr):.4f}")

# SHAP values
explainer = shap.Explainer(logistic_model, X_test)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)


# CatBoost
catboost_model = CatBoostClassifier(loss_function='Logloss', eval_metric='Logloss', verbose=0, random_seed=42)
catboost_model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)

# Cross-validation
logloss_catboost = -cross_val_score(catboost_model, X_balanced, y_balanced, cv=5, scoring='neg_log_loss')
print(f"CatBoost Logloss (CV): {logloss_catboost.mean():.4f}")

# Metrics
y_pred_catboost = catboost_model.predict(X_test)
y_pred_proba_catboost = catboost_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred_catboost))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_catboost):.4f}")

# SHAP values
explainer = shap.Explainer(catboost_model, X_test)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)


# XGBoost
xgboost_model = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
xgboost_model.fit(X_train, y_train)

# Cross-validation
logloss_xgboost = -cross_val_score(xgboost_model, X_balanced, y_balanced, cv=5, scoring='neg_log_loss')
print(f"XGBoost Logloss (CV): {logloss_xgboost.mean():.4f}")

# Metrics
y_pred_xgboost = xgboost_model.predict(X_test)
y_pred_proba_xgboost = xgboost_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred_xgboost))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_xgboost):.4f}")

# SHAP values
explainer = shap.Explainer(xgboost_model, X_test)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)


# Define the Neural Network
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compile the model with binary crossentropy loss and Adam optimizer
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = nn_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
loss, accuracy = nn_model.evaluate(X_test, y_test, verbose=0)
print(f"Neural Network Logloss: {loss:.4f}")

# Plot Loss Progression
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Neural Network Loss Progression')
plt.yscale('log') 
plt.xlabel('Epochs')
plt.ylabel('Logloss')
plt.legend()
plt.show()

# Plot Accuracy Progression
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Neural Network Accuracy Progression')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Predictions for ROC-AUC and Classification Report
y_pred_proba_nn = nn_model.predict(X_test).ravel()
print(classification_report(y_test, (y_pred_proba_nn > 0.5).astype(int)))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_nn):.4f}")

# Compute SHAP values for the Neural Network
explainer = shap.Explainer(nn_model, X_test)
shap_values = explainer(X_test)

# Plot SHAP summary plot
shap.summary_plot(shap_values, X_test, feature_names=X_test.columns)

