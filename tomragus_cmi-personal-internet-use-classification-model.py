# load pandas
import pandas as pd

# load sample submission
sample = pd.read_csv("/kaggle/input/child-mind-institute-problematic-internet-use/sample_submission.csv")

# display sample submission
print("Sample submission")
print(f"Submission shape: {sample.shape}")
sample


import warnings

# load train set
train = pd.read_csv("/kaggle/input/child-mind-institute-problematic-internet-use/train.csv")

# display first 5 rows of train set
print("""Train set: where the 'features' live""")
print(f"Train shape: {train.shape}")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    display(train.head())



# load test set
test = pd.read_csv("/kaggle/input/child-mind-institute-problematic-internet-use/test.csv")

# display first 5 rows of test set
print("""Test set: what we will evaluate our models on""")
print(f"Test shape: {test.shape}")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    display(test.head())


# load data dictionary
data_dict = pd.read_csv("/kaggle/input/child-mind-institute-problematic-internet-use/data_dictionary.csv")

# display first 5 rows of data dictionary
print("""Data Dictionary: what each feature means""")
print(f"Data Dictionary shape: {data_dict.shape}")
display(data_dict.head())


# load all other libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import xgboost as xgb
import lightgbm as lgb
from pandas.api.types import is_numeric_dtype, is_object_dtype, is_categorical_dtype, CategoricalDtype
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
warnings.filterwarnings("ignore", category=RuntimeWarning)


# isolating train-only features
train_cols = set(train.columns)
test_cols = set(test.columns)
columns_not_in_test = sorted(list(train_cols - test_cols))

# addind additional information using data dictionary
data_dict[data_dict['Field'].isin(columns_not_in_test)]


# calculate max and min
pciat_min_max = train.groupby('sii', observed=True)['PCIAT-PCIAT_Total'].agg(['min', 'max'])
pciat_min_max = pciat_min_max.rename(columns={'min': 'Minimum PCIAT total Score', 'max': 'Maximum total PCIAT Score'})
display(pciat_min_max)

# print range for each level of severity
print(data_dict[data_dict['Field'] == 'PCIAT-PCIAT_Total']['Value Labels'].iloc[0])


# display section with missing values highlighted
train_with_sii = train[train['sii'].notna()][columns_not_in_test]
train_with_sii[train_with_sii.isna().any(axis=1)].head().style.map(lambda x: 'background-color: #FFC0CB' if pd.isna(x) else '')


# isolate PCIAT columns
PCIAT_cols = [f'PCIAT-PCIAT_{i+1:02d}' for i in range(20)]

# define function to recalculate SII accounting for missing PCIAT values
def recalculate_sii(row):
    if pd.isna(row['PCIAT-PCIAT_Total']):
        return np.nan
    max_possible = row['PCIAT-PCIAT_Total'] + row[PCIAT_cols].isna().sum() * 5
    if row['PCIAT-PCIAT_Total'] <= 30 and max_possible <= 30:
        return 0
    elif 31 <= row['PCIAT-PCIAT_Total'] <= 49 and max_possible <= 49:
        return 1
    elif 50 <= row['PCIAT-PCIAT_Total'] <= 79 and max_possible <= 79:
        return 2
    elif row['PCIAT-PCIAT_Total'] >= 80 and max_possible >= 80:
        return 3
    return np.nan
train['recalc_sii'] = train.apply(recalculate_sii, axis=1)

# overwriting SII with recalc_SII and adding labels 'missing', 'none', 'mild', 'moderate', 'severe'
train['sii'] = train['recalc_sii']
train['complete_resp_total'] = train['PCIAT-PCIAT_Total'].where(train[PCIAT_cols].notna().all(axis=1), np.nan)
sii_map = {0: '0 (None)', 1: '1 (Mild)', 2: '2 (Moderate)', 3: '3 (Severe)'}
train['sii'] = train['sii'].map(sii_map).fillna('Missing')
sii_order = ['Missing', '0 (None)', '1 (Mild)', '2 (Moderate)', '3 (Severe)']
train['sii'] = pd.Categorical(train['sii'], categories=sii_order, ordered=True)
train.drop(columns='recalc_sii', inplace=True)


# remove rows with no SII
initial_rows = len(train)
train = train[train['sii'] != 'Missing']
train['sii'] = train['sii'].cat.remove_unused_categories()
removed_rows = initial_rows - len(train)
print(f"Removed {removed_rows} rows with 'Missing' SII values.")
print(f"Train shape: {train.shape}")

# check for/ remove duplicate rows (if present)
duplicate_count = train.duplicated().sum()
print(f"Duplicate rows: {duplicate_count}")


# define helper function for gathering information from given column(s)
def calculate_stats(data, columns):
    if isinstance(columns, str):
        columns = [columns]
    stats = []
    for col in columns:
        if data[col].dtype in ['object', 'category']:
            counts = data[col].value_counts(dropna=False, sort=False)
            percents = data[col].value_counts(normalize=True, dropna=False, sort=False) * 100
            formatted = counts.astype(str) + ' (' + percents.round(2).astype(str) + '%)'
            stats_col = pd.DataFrame({'count (%)': formatted})
            stats.append(stats_col)
        else:
            stats_col = data[col].describe().to_frame().transpose()
            stats_col['missing'] = data[col].isnull().sum()
            stats_col.index.name = col
            stats.append(stats_col)
    return pd.concat(stats, axis=0)


warnings.filterwarnings("ignore", message=".*use_inf_as_na option is deprecated.*")

# helper variables for EDA
sii_counts = train['sii'].value_counts().reset_index()
sii_counts.columns = ['SII', 'Count']
total = sii_counts['Count'].sum()
sii_counts['percentage'] = (sii_counts['Count'] / total) * 100
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# generate custom color pallete: bright yellow, lavender, mint green, light pink, sky blue
custom_palette = ['#FFFB46', '#E2A0FF', '#B7FFD8', '#FFC1CF', '#C4F5FC']

# create pie chart: distribution of participant SII
axes[0].pie(sii_counts['Count'], labels=sii_counts['SII'], autopct='%1.1f%%', colors=custom_palette, startangle=140, wedgeprops={'edgecolor': 'black'})
axes[0].set_title('Distribution of Severity Impairment Index (SII)', fontsize=14)
axes[0].axis('equal')

# create histogram: distribution of participant PCIAT_Total
sns.histplot(train['complete_resp_total'].dropna(), bins=20, color='#E2A0FF', ax=axes[1])
axes[1].set_title('Distribution of PCIAT_Total', fontsize=14)
axes[1].set_xlabel('PCIAT_Total')
axes[1].set_ylabel('Participants')
plt.tight_layout()
plt.show()


assert train['Basic_Demos-Age'].isna().sum() == 0
assert train['Basic_Demos-Sex'].isna().sum() == 0

# create table: distribution of participant Age Group
train['Age Group'] = pd.cut(train['Basic_Demos-Age'], bins=[4, 12, 18, 22], labels=['Children (5-12)', 'Adolescents (13-18)', 'Adults (19-22)'])
calculate_stats(train, 'Age Group')


# create table: distribution of participant Sex
sex_map = {0: 'Male', 1: 'Female'}
train['Basic_Demos-Sex'] = train['Basic_Demos-Sex'].map(sex_map)
calculate_stats(train, 'Basic_Demos-Sex')


# create table: distribution of SII by Age Group
stats_age = train.groupby(['Age Group', 'sii'], observed=False).size().unstack(fill_value=0)
stats_age_prop = stats_age.div(stats_age.sum(axis=1), axis=0) * 100
stats_age = stats_age.astype(str) +' (' + stats_age_prop.round(1).astype(str) + '%)'
stats_age


# create table: distribution of SII by Sex
stats_sex = train.groupby(['Basic_Demos-Sex', 'sii'], observed=False).size().unstack(fill_value=0)
stats_sex_prop = stats_sex.div(stats_sex.sum(axis=1), axis=0) * 100
stats_sex = stats_sex.astype(str) +' (' + stats_sex_prop.round(1).astype(str) + '%)'
stats_sex


warnings.filterwarnings("ignore", message=".*observed=False is deprecated.*")
fig, axes = plt.subplots(1, 3, figsize=(24, 6)) 

# create bar chart: distribution of Hours of Internet Use
ax1 = sns.countplot(x='PreInt_EduHx-computerinternet_hoursday', data=train, palette=custom_palette[:4], ax=axes[0], edgecolor='black', linewidth=0.8)
axes[0].set_title('Distribution of Hours of Internet Use')
axes[0].set_xlabel('Hours per Day Group')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', which='both', bottom=False, labelbottom=True)

# create boxplot: Hours of Internet Use vs Age
sns.boxplot(y=train['Basic_Demos-Age'], x=train['PreInt_EduHx-computerinternet_hoursday'], palette=custom_palette[:4], ax=axes[1])
axes[1].set_title('Hours of Internet Use by Age')
axes[1].set_ylabel('Age')
axes[1].set_xlabel('Hours per Day Group')

# create boxplot: Hours of Internet Use vs Age Group
sns.boxplot(y='PreInt_EduHx-computerinternet_hoursday', x='Age Group', data=train, palette=custom_palette[:3], ax=axes[2])
axes[2].set_title('Internet Use by Age Group')
axes[2].set_ylabel('Hours per Day')
axes[2].set_xlabel('Age Group');


warnings.filterwarnings("ignore", message=".*observed=False is deprecated.*")
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5])

# create boxplot: SII vs Hours of Internet Use
ax1 = fig.add_subplot(gs[0, 0])
sns.boxplot(x='sii', y='PreInt_EduHx-computerinternet_hoursday', data=train, ax=ax1, palette=custom_palette[:4])
ax1.set_title('SII by Internet Use')
ax1.set_ylabel('Hours per Day')
ax1.set_xlabel('SII')

# create boxplot: PCIAT_Total vs Hours of Internet Use
ax2 = fig.add_subplot(gs[0, 1])
sns.boxplot(x='PreInt_EduHx-computerinternet_hoursday', y='complete_resp_total', data=train, palette=custom_palette[:4], ax=ax2)
ax2.set_title('PCIAT_Total by Internet Use')
ax2.set_ylabel('PCIAT_Total')
ax2.set_xlabel('Hours per Day');


# covert SII values to float type
train['sii'] = train['sii'].str[0].astype(float)

# isolate physical features
physical_cols = ['Physical-BMI', 'Physical-Height', 'Physical-Weight', 'Physical-Waist_Circumference', 'Physical-Diastolic_BP', 'Physical-HeartRate', 'Physical-Systolic_BP']
data_subset = train[physical_cols + ['sii']]

# generate correlation heatmap matrix for physical features
corr_matrix = data_subset.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()


plt.figure(figsize=(18, 5))

# create scatterplot: Height vs Age
plt.subplot(1, 3, 1)
sns.scatterplot(x='Basic_Demos-Age', y='Physical-Height', data=train)
plt.title('Physical-Height by Age')
plt.xlabel('Age')
plt.ylabel('Height (cm)')

# create scatterplot: Weight vs Age
plt.subplot(1, 3, 2)
sns.scatterplot(x='Basic_Demos-Age', y='Physical-Weight', data=train)
plt.title('Physical-Weight by Age')
plt.xlabel('Age')
plt.ylabel('Weight (kg)')
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# create boxplot: SII vs BMI
sns.boxplot(x='sii', y='Physical-BMI', data=train, ax=axes[0], palette=custom_palette[:4])
axes[0].set_title('SII vs BMI')
axes[0].set_xlabel('SII Category')
axes[0].set_ylabel('BMI')

# create boxplot: SII vs Height
sns.boxplot(x='sii', y='Physical-Height', data=train, ax=axes[1], palette=custom_palette[:4])
axes[1].set_title('SII vs Height')
axes[1].set_xlabel('SII Category')
axes[1].set_ylabel('Height (cm)')

# create boxplot: SII vs Weight
sns.boxplot(x='sii', y='Physical-Weight', data=train, ax=axes[2], palette=custom_palette[:4])
axes[2].set_title('SII vs Weight')
axes[2].set_xlabel('SII Category')
axes[2].set_ylabel('Weight (kg)')
plt.tight_layout()
plt.show()


# isolate non-PCIAT features
train_subset = train.drop(train.columns[55:76].tolist() + ['complete_resp_total'], axis=1)
numerical_features = train_subset.select_dtypes(include=[np.number]).columns.tolist()

# calculate Pearson correlations with 'sii'
target_correlations = []
for feature in numerical_features:
    if feature != 'sii':
        corr = train_subset[[feature, 'sii']].corr().iloc[0, 1]
        if not np.isnan(corr):
            target_correlations.append((feature, abs(corr)))

# sort and select top 20
top_corr = sorted(target_correlations, key=lambda x: x[1], reverse=True)[:20]
features, corrs = zip(*top_corr)

# create bar chart of features by correlation with SII
plt.figure(figsize=(12, 8))
plt.barh(range(len(features)), corrs, color='#E2A0FF', edgecolor='black')
plt.yticks(range(len(features)), features)
plt.xlabel('Absolute Correlation with SII')
plt.title('Top 20 Features by Correlation with SII')
plt.gca().invert_yaxis()  # Optional: highest at top
plt.tight_layout()
plt.show()


# permanentlyremove all PCIAT columns
pciat_cols = [col for col in train.columns if 'PCIAT' in col]
train = train.drop(columns=pciat_cols)

# remove "complete_resp" and "age group" columns
train = train.drop("complete_resp_total", axis=1)
train = train.drop("Age Group", axis=1) 

# remove id column
train = train.drop("id", axis=1)


# separate SII from features
X = train.drop('sii', axis=1)
y = train['sii']

# create sets for categorical and numerical features
categorical_cols = []
numerical_cols = []
for col in X.columns:
    if X[col].dtype == 'object':
        categorical_cols.append(col)
    else:
        numerical_cols.append(col)

# re-calculate Pearson correlation coefficients for numerical columns (corr. with SII)
def calculate_correlations(X, y, numerical_cols):
    correlations = {}
    for col in numerical_cols:
        mask = ~(X[col].isnull() | y.isnull())
        if mask.sum() > 1:
            corr, p_value = pearsonr(X[col][mask], y[mask])
            correlations[col] = {'correlation': corr, 'p_value': p_value}
    return correlations
correlations = calculate_correlations(X, y, numerical_cols)

# sort features by correlation
sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)
print("\nTop 20 correlations with SII (absolute correlation):")
for i, (col, stats) in enumerate(sorted_correlations[:20]):
    print(f"{i+1:2d}. {col:<40} | Corr: {stats['correlation']:6.3f} | p-value: {stats['p_value']:.4f}")


# define function to engineer new features
def create_engineered_features(df, correlations=None):
    df_eng = df.copy()
    
    # BMI-related features
    if 'Physical-BMI' in df_eng.columns and 'BIA-BIA_BMI' in df_eng.columns:
        df_eng['BMI_difference'] = df_eng['Physical-BMI'] - df_eng['BIA-BIA_BMI']
    
    # fitness ratios
    if 'Fitness_Endurance-Time_Mins' in df_eng.columns and 'Fitness_Endurance-Time_Sec' in df_eng.columns:
        df_eng['Total_Fitness_Time'] = df_eng['Fitness_Endurance-Time_Mins'] * 60 + df_eng['Fitness_Endurance-Time_Sec']
    
    # body composition ratios
    if 'BIA-BIA_Fat' in df_eng.columns and 'BIA-BIA_FFM' in df_eng.columns:
        df_eng['Fat_to_FFM_ratio'] = df_eng['BIA-BIA_Fat'] / (df_eng['BIA-BIA_FFM'] + 1e-8)
    
    # physical health composite
    if all(col in df_eng.columns for col in ['Physical-HeartRate', 'Physical-Systolic_BP', 'Physical-Diastolic_BP']):
        health_cols = ['Physical-HeartRate', 'Physical-Systolic_BP', 'Physical-Diastolic_BP']
        for col in health_cols:
            if df_eng[col].notna().sum() > 0:
                mean_val = df_eng[col].mean()
                std_val = df_eng[col].std()
                df_eng[f'{col}_normalized'] = (df_eng[col] - mean_val) / (std_val + 1e-8)
    
    # age-related
    if 'Basic_Demos-Age' in df_eng.columns:
        df_eng['Age_squared'] = df_eng['Basic_Demos-Age'] ** 2
        df_eng['Age_group'] = pd.cut(df_eng['Basic_Demos-Age'], 
                                   bins=[0, 8, 12, 16, 22], 
                                   labels=['child', 'preteen', 'teen', 'young_adult'])
    
    # features based on correlation
    if correlations is not None:
        top_corr_features = [col for col, stats in sorted(correlations.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)[:10] if col in df_eng.columns]
        for i, col1 in enumerate(top_corr_features[:5]):
            for col2 in top_corr_features[i+1:5]:
                if col1 in df_eng.columns and col2 in df_eng.columns:
                    if df_eng[col1].dtype in ['int64', 'float64'] and df_eng[col2].dtype in ['int64', 'float64']:
                        df_eng[f'{col1}_x_{col2}'] = df_eng[col1] * df_eng[col2]
    return df_eng

# apply feature engineering function to train set
X_engineered = create_engineered_features(X, correlations)
print("New features created:")
new_features = set(X_engineered.columns) - set(X.columns)
for feat in new_features:
    print(f"  - {feat}")


# re-create categorical and numerical sets to include engineered features
categorical_cols_eng = []
numerical_cols_eng = []
for col in X_engineered.columns:
    if is_object_dtype(X_engineered[col]) or isinstance(X_engineered[col].dtype, CategoricalDtype):
        categorical_cols_eng.append(col)
    elif is_numeric_dtype(X_engineered[col]):
        numerical_cols_eng.append(col)
print(f"\nFinal categorical columns: {len(categorical_cols_eng)}")
print(f"Final numerical columns: {len(numerical_cols_eng)}")

# recalculate correlations with engineered features
correlations_eng = calculate_correlations(X_engineered, y, numerical_cols_eng)

# sort by absolute correlation
sorted_correlations_eng = sorted(correlations_eng.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)
print("\nTop 15 correlations with SII (after feature engineering):")
for i, (col, stats) in enumerate(sorted_correlations_eng[:15]):
    print(f"{i+1:2d}. {col:<40} | Corr: {stats['correlation']:6.3f} | p-value: {stats['p_value']:.4f}")


# split engineered data into train and validation sets, 80/ 20
X_train, X_val, y_train, y_val = train_test_split(X_engineered, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")


# RIDGE REGRESSION PREPROCESSING PIPELINE
print("\n" + "="*50)
print("RIDGE REGRESSION PREPROCESSING")
print("="*50)

# define function for feature selection based on threshold
def select_features_by_correlation(X, y, numerical_cols, threshold=0.05):
    correlations = calculate_correlations(X, y, numerical_cols)
    selected_features = []
    for col, stats in correlations.items():
        if abs(stats['correlation']) >= threshold and stats['p_value'] < 0.05:
            selected_features.append(col)
    return selected_features

# define ridge model preprocessing function
def preprocess_for_ridge(X_train, X_val, categorical_cols, numerical_cols, correlation_threshold=0.05, use_correlation_filtering=True):
    
    # create copies of sets
    X_train_processed = X_train.copy()
    X_val_processed = X_val.copy()
    
    # missing values (fill numerical with median, categorical with mode)
    for col in numerical_cols:
        if col in X_train_processed.columns:
            median_val = X_train_processed[col].median()
            X_train_processed[col] = X_train_processed[col].fillna(median_val)
            X_val_processed[col] = X_val_processed[col].fillna(median_val)
    for col in categorical_cols:
        if col in X_train_processed.columns:
            mode_val = X_train_processed[col].mode()[0] if not X_train_processed[col].mode().empty else 'Unknown'
            X_train_processed[col] = X_train_processed[col].fillna(mode_val)
            X_val_processed[col] = X_val_processed[col].fillna(mode_val)
    
    # feature selection based on correlation
    selected_numerical_features = numerical_cols
    if use_correlation_filtering:
        selected_numerical_features = select_features_by_correlation(X_train_processed, y_train, numerical_cols, correlation_threshold)
        print(f"Selected {len(selected_numerical_features)} numerical features out of {len(numerical_cols)} based on correlation > {correlation_threshold}")
        print("Selected features:", selected_numerical_features[:10], "..." if len(selected_numerical_features) > 10 else "")
        
        # keep only selected numerical features
        features_to_keep = selected_numerical_features + categorical_cols
        X_train_processed = X_train_processed[features_to_keep]
        X_val_processed = X_val_processed[features_to_keep]
    
    # one-hot encoding for categorical variables
    X_train_encoded = pd.get_dummies(X_train_processed, columns=categorical_cols, drop_first=True)
    X_val_encoded = pd.get_dummies(X_val_processed, columns=categorical_cols, drop_first=True)
    
    # validate continuity between sets
    missing_cols_val = set(X_train_encoded.columns) - set(X_val_encoded.columns)
    missing_cols_train = set(X_val_encoded.columns) - set(X_train_encoded.columns)
    for col in missing_cols_val:
        X_val_encoded[col] = 0
    for col in missing_cols_train:
        X_train_encoded[col] = 0
    X_val_encoded = X_val_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
    
    # scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_val_scaled = scaler.transform(X_val_encoded)
    encoded_cols = list(X_train_encoded.columns)
    return X_train_scaled, X_val_scaled, scaler, encoded_cols, selected_numerical_features

# train Ridge models with different correlation thresholds
correlation_thresholds = [0.01, 0.03, 0.05, 0.1]
ridge_results = {}
print("\nTesting different correlation thresholds for Ridge:")
for threshold in correlation_thresholds:
    print(f"\n--- Testing threshold: {threshold} ---")
    
    # preprocessing
    X_train_ridge, X_val_ridge, scaler, encoded_cols, selected_features = preprocess_for_ridge(X_train, X_val, categorical_cols_eng, numerical_cols_eng, correlation_threshold=threshold, use_correlation_filtering=True)
    print(f"Ridge - Train shape after preprocessing: {X_train_ridge.shape}")
    
    # train
    ridge_model = Ridge(alpha=1.0, random_state=42)
    ridge_model.fit(X_train_ridge, y_train)
    
    # predictions and evaluation
    y_train_pred_ridge = ridge_model.predict(X_train_ridge)
    y_val_pred_ridge = ridge_model.predict(X_val_ridge)
    ridge_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_ridge))
    ridge_val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred_ridge))
    ridge_results[threshold] = {'train_rmse': ridge_train_rmse, 'val_rmse': ridge_val_rmse, 'num_features': X_train_ridge.shape[1], 'selected_features': selected_features}
    print(f"Train RMSE: {ridge_train_rmse:.4f}")
    print(f"Validation RMSE: {ridge_val_rmse:.4f}")

# test without correlation fitting
print(f"\n--- Testing without correlation filtering ---")
X_train_ridge_no_filter, X_val_ridge_no_filter, ridge_scaler_no_filter, ridge_columns_no_filter, _ = preprocess_for_ridge(X_train, X_val, categorical_cols_eng, numerical_cols_eng, use_correlation_filtering=False)
print(f"Ridge - Train shape after preprocessing: {X_train_ridge_no_filter.shape}")
ridge_model_no_filter = Ridge(alpha=1.0, random_state=42)
ridge_model_no_filter.fit(X_train_ridge_no_filter, y_train)

# evaluate without correlation fitting
y_train_pred_ridge_no_filter = ridge_model_no_filter.predict(X_train_ridge_no_filter)
y_val_pred_ridge_no_filter = ridge_model_no_filter.predict(X_val_ridge_no_filter)
ridge_train_rmse_no_filter = np.sqrt(mean_squared_error(y_train, y_train_pred_ridge_no_filter))
ridge_val_rmse_no_filter = np.sqrt(mean_squared_error(y_val, y_val_pred_ridge_no_filter))
ridge_results['no_filter'] = {'train_rmse': ridge_train_rmse_no_filter, 'val_rmse': ridge_val_rmse_no_filter, 'num_features': X_train_ridge_no_filter.shape[1], 'selected_features': None}
print(f"Train RMSE: {ridge_train_rmse_no_filter:.4f}")
print(f"Validation RMSE: {ridge_val_rmse_no_filter:.4f}")

# select best Ridge configuration for final model
best_ridge_config = min(ridge_results.items(), key=lambda x: x[1]['val_rmse'])
print(f"\nBest Ridge configuration: {best_ridge_config[0]} (Validation RMSE: {best_ridge_config[1]['val_rmse']:.4f})")
if best_ridge_config[0] == 'no_filter':
    X_train_ridge_final, X_val_ridge_final = X_train_ridge_no_filter, X_val_ridge_no_filter
    ridge_model_final = ridge_model_no_filter
    final_ridge_train_rmse, final_ridge_val_rmse = ridge_train_rmse_no_filter, ridge_val_rmse_no_filter
else:
    X_train_ridge_final, X_val_ridge_final, _, _, _ = preprocess_for_ridge(X_train, X_val, categorical_cols_eng, numerical_cols_eng, correlation_threshold=best_ridge_config[0], use_correlation_filtering=True)
    ridge_model_final = Ridge(alpha=1.0, random_state=42)
    ridge_model_final.fit(X_train_ridge_final, y_train)
    y_train_pred_final = ridge_model_final.predict(X_train_ridge_final)
    y_val_pred_final = ridge_model_final.predict(X_val_ridge_final)
    final_ridge_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_final))
    final_ridge_val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred_final))
selected_features = best_ridge_config[1]['selected_features']


# LIGHTGBM PREPROCESSING PIPELINE
print("\n" + "="*50)
print("LIGHTGBM PREPROCESSING")
print("="*50)

def preprocess_for_lightgbm(X_train, X_val, categorical_cols, numerical_cols):
    
    # create copies of sets
    X_train_processed = X_train.copy()
    X_val_processed = X_val.copy()
    
    # missing values for numerical columns
    for col in numerical_cols:
        if col in X_train_processed.columns:
            median_val = X_train_processed[col].median()
            X_train_processed[col] = X_train_processed[col].fillna(median_val)
            X_val_processed[col] = X_val_processed[col].fillna(median_val)
    
    # label encoding for categorical variables
    label_encoders = {}
    for col in categorical_cols:
        if col in X_train_processed.columns:
            le = LabelEncoder()
            
            # fill missing values first
            mode_val = X_train_processed[col].mode()[0] if not X_train_processed[col].mode().empty else 'Unknown'
            X_train_processed[col] = X_train_processed[col].fillna(mode_val)
            X_val_processed[col] = X_val_processed[col].fillna(mode_val)

            # fit on train data
            X_train_processed[col] = le.fit_transform(X_train_processed[col]).astype(int)

            # handle unseen categories in validation
            train_categories = set(le.classes_)
            X_val_processed[col] = X_val_processed[col].map(lambda x: le.transform([x])[0] if x in train_categories else 0).astype(int)
            label_encoders[col] = le
    return X_train_processed, X_val_processed, label_encoders

# preprocess for LightGBM
X_train_lgb, X_val_lgb, lgb_encoders = preprocess_for_lightgbm(X_train, X_val, categorical_cols_eng, numerical_cols_eng)
categorical_feature_names = [col for col in categorical_cols_eng if col in X_train_lgb.columns]
print(f"LightGBM - Train shape after preprocessing: {X_train_lgb.shape}")
print(f"LightGBM - Validation shape after preprocessing: {X_val_lgb.shape}")

# define LightGBM parameters
lgb_params = {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'num_leaves': 31, 'learning_rate': 0.05, 'colsample_bytree': 0.9, 'subsample': 0.8, 'bagging_freq': 5, 'verbose': 0, 'random_state': 42, 'force_col_wise': True}

# create train and validation sets for LightGBM
train_data = lgb.Dataset(X_train_lgb, label=y_train, categorical_feature=categorical_feature_names)
val_data = lgb.Dataset(X_val_lgb, label=y_val, reference=train_data, categorical_feature=categorical_feature_names)

# train LightGBM model
lgb_model = lgb.train(lgb_params, train_data, valid_sets=[train_data, val_data], num_boost_round=1000, callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)])

# evaluate trained model on train and validation sets for result
y_train_pred_lgb = lgb_model.predict(X_train_lgb, num_iteration=lgb_model.best_iteration)
y_val_pred_lgb = lgb_model.predict(X_val_lgb, num_iteration=lgb_model.best_iteration)
lgb_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_lgb))
lgb_val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred_lgb))
print(f"\nLightGBM Results:")
print(f"Train RMSE: {lgb_train_rmse:.4f}")
print(f"Validation RMSE: {lgb_val_rmse:.4f}")


# print overfitting analysis
print("\nOverfitting Analysis:")
ridge_overfit = final_ridge_val_rmse - final_ridge_train_rmse
lgb_overfit = lgb_val_rmse - lgb_train_rmse
print(f"   - Ridge overfitting: {ridge_overfit:.4f}")
print(f"   - LightGBM overfitting: {lgb_overfit:.4f}")
if ridge_overfit < lgb_overfit:
    print("   - Ridge is more robust (less overfitting)")
else:
    print("   - LightGBM is more robust (less overfitting)")


# determine best alpha for Ridge
ridge = Ridge(random_state=42)
param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_root_mean_squared_error')
grid_search.fit(X_train_ridge_final, y_train)
print(f"Best alpha: {grid_search.best_params_['alpha']}")

# retrain with best alpha from CV
best_alpha = grid_search.best_params_['alpha']
ridge_model_final = Ridge(alpha=best_alpha, random_state=42)
ridge_model_final.fit(X_train_ridge_final, y_train)

# re-evaluate Ridge model on validation set with best alpha
y_train_pred = ridge_model_final.predict(X_train_ridge_final)
y_val_pred = ridge_model_final.predict(X_val_ridge_final)

# calculate train and validation RMSE
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"\nFinal Ridge Model with alpha={best_alpha}:")
print(f"  Train RMSE: {train_rmse:.4f}")
print(f"  Validation RMSE: {val_rmse:.4f}")


# define hyperparameters for tuning
param_grid = {'learning_rate': [0.01, 0.05], 'num_leaves': [15, 31], 'max_depth': [-1, 5], 'colsample_bytree': [0.8, 1.0], 'subsample': [0.8, 1.0], 'min_child_samples': [20, 50]}
lgb_params['verbose'] = -1

# define new model for parameter testing
lgb_model = LGBMRegressor(objective='regression', random_state=42, n_estimators=1000, verbose=-1)
grid_search = GridSearchCV(estimator=lgb_model, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=3, verbose=0, n_jobs=-1)

# fit new parameters on train lgb data
grid_search.fit(X_train_lgb, y_train)
print(f"Best Parameters:\n{grid_search.best_params_}")
print(f"Best CV RMSE: {-grid_search.best_score_:.4f}")
best_lgb_params = grid_search.best_params_

# retrain final model with tuned parameters
final_lgb_model = LGBMRegressor(**best_lgb_params, objective='regression', random_state=42, n_estimators=1000)

# fit final model on full training set
final_lgb_model.fit(X_train_lgb, y_train,eval_set=[(X_val_lgb, y_val)], eval_metric='rmse', callbacks=[early_stopping(stopping_rounds=50), log_evaluation(0)])

# evaluate tuned/ trained model on train and validation sets for final result
y_train_pred = final_lgb_model.predict(X_train_lgb)
y_val_pred = final_lgb_model.predict(X_val_lgb)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"\nTuned LightGBM Results:")
print(f"  Train RMSE: {train_rmse:.4f}")
print(f"  Validation RMSE: {val_rmse:.4f}")


# convert sex binary to string (we changed this in the train set in the EDA section)
test['Basic_Demos-Sex'] = test['Basic_Demos-Sex'].map({0: 'Male', 1: 'Female'})

# separate id from test set
test_ids = test["id"].copy()
test = test.drop("id", axis=1)

# update test set with engineered features
X_test_engineered = create_engineered_features(test, correlations)


# define helper function to round predictions
def clip_predictions(predictions):
    rounded = np.round(predictions)
    clipped = np.clip(rounded, 0, 3)
    return clipped

# define Ridge preprocessing function for test set
def preprocess_test_for_ridge(X_test, categorical_cols, numerical_cols, selected_numerical_features, scaler, encoded_column_names):
    X_test_processed = X_test.copy()
    
    # missing values (fill numerical with median, categorical with mode)
    for col in numerical_cols:
        if col in X_test_processed.columns:
            median_val = X_test_processed[col].median()
            X_test_processed[col] = X_test_processed[col].fillna(median_val)
    for col in categorical_cols:
        if col in X_test_processed.columns:
            mode_val = X_test_processed[col].mode()[0] if not X_test_processed[col].mode().empty else 'Unknown'
            X_test_processed[col] = X_test_processed[col].fillna(mode_val)

    # keep selected features and one-hot encode
    features_to_keep = selected_numerical_features + categorical_cols
    X_test_processed = X_test_processed[features_to_keep]
    X_test_encoded = pd.get_dummies(X_test_processed, columns=categorical_cols, drop_first=True)

    # add absent columns, reorder columns and scale
    for col in encoded_column_names:
        if col not in X_test_encoded.columns:
            X_test_encoded[col] = 0
    X_test_encoded = X_test_encoded[encoded_column_names]
    X_test_scaled = scaler.transform(X_test_encoded)
    return X_test_scaled

# apply Ridge preprocessing function to test set
X_test_scaled = preprocess_test_for_ridge(X_test_engineered, categorical_cols = categorical_cols_eng, numerical_cols = numerical_cols_eng, selected_numerical_features = selected_features, scaler = scaler, encoded_column_names = encoded_cols)

# make predictions
predictions_ridge = ridge_model_final.predict(X_test_scaled)
y_pred_ridge = clip_predictions(predictions_ridge)

# generate Ridge predictions as dataframe
submission_ridge = pd.DataFrame({'id': test_ids, 'SII': y_pred_ridge})
display(submission_ridge)


# define LightGBM preprocessing function for test set
def preprocess_test_for_lightgbm(X_test, categorical_cols, numerical_cols, label_encoders):
    X_test_processed = X_test.copy()
    
    # missing values for numerical columns
    for col in numerical_cols:
        if col in X_test_processed.columns:
            median_val = X_test_processed[col].median()
            X_test_processed[col] = X_test_processed[col].fillna(median_val)
    
    # label encode using training encoders
    for col in categorical_cols:
        if col in X_test_processed.columns:
            le = label_encoders.get(col)
            if le:
                mode_val = X_test_processed[col].mode()[0] if not X_test_processed[col].mode().empty else 'Unknown'
                X_test_processed[col] = X_test_processed[col].fillna(mode_val)
                train_categories = set(le.classes_)
                X_test_processed[col] = X_test_processed[col].map(lambda x: le.transform([x])[0] if x in train_categories else 0).astype(int)
            else:
                X_test_processed[col] = 0
    return X_test_processed

# apply LightGBM preprocessing function to test set
X_test_lgb = preprocess_test_for_lightgbm(X_test_engineered, categorical_cols = categorical_cols_eng, numerical_cols = numerical_cols_eng, label_encoders = lgb_encoders)

# make predictions
predictions_lgb = final_lgb_model.predict(X_test_lgb)
y_pred_lgb = clip_predictions(predictions_lgb)

# generate LightGBM predictions as dataframe
submission_gbm = pd.DataFrame({'id': test_ids, 'SII': y_pred_lgb})
display(submission_gbm)


# calcuate weighted average of predictions
submission_ridge.to_csv('submission.csv', index=False)

