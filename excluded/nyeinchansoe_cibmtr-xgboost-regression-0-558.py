import pandas as pd

train_df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
test_df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")

train_df.shape, test_df.shape


train_df_set = set(train_df.columns)
test_df_set = set(test_df.columns)

test_missing_cols = train_df_set - test_df_set
test_extra_cols = test_df_set - train_df_set

print(f'Missing cols in test.csv: {test_missing_cols}')
print(f'Extra cols in test.csv: {test_extra_cols}')


efs_dist = train_df['efs'].value_counts(normalize=True)
efs_dist


import matplotlib.pyplot as plt
import seaborn as sns

# identify class imbalance
# the model might become biased toward predicting the dominant class

plt.figure(figsize=(6, 4))
sns.countplot(x='efs', data=train_df)
plt.title('Distribution of efs')
plt.xlabel('Event-Free Survival (efs)')
plt.ylabel('Count')
plt.show()


# efs_time: Time to event-free survival, months
efs_time = train_df['efs_time']
efs_time.describe()


plt.figure(figsize=(8, 4))
sns.histplot(efs_time)
plt.title('Distribution of efs_time')
plt.xlabel('Event-Free Survival Time (Months)')
plt.ylabel('Frequency')
plt.show()


from scipy.stats import pointbiserialr

# efs: binary value
# efs_time: continuous 
# efs and efs_time: point-biserial correlation is used to measure their relationship
# relationship between a continuous variable (efs_time) and a dichotomous variable (efs)

corr_coef, p_value = pointbiserialr(train_df['efs'], train_df['efs_time'])
print(f'efs and efs_time corr coefficient: {corr_coef:.4f}')
print(f'statistically significant: {p_value:.4f}')


plt.figure(figsize=(6, 5))
sns.boxplot(x='efs', y='efs_time', data=train_df)
plt.title('Boxplot of EFS Time by EFS Status')
plt.xlabel('EFS')
plt.ylabel('EFS Time (Months)')
plt.show()


missing_data = train_df.isnull().mean() * 100
missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
missing_data.count()


missing_data_test = test_df.isnull().mean() * 100
missing_data_test = missing_data_test[missing_data_test > 0].sort_values(ascending=False)
missing_data_test.count()


# columns with high missingness > 40%
HIGH_MISSINGNESS_PERC = 40

high_missing_data = missing_data[missing_data > HIGH_MISSINGNESS_PERC].sort_values(ascending=False)
# decide later to impute or exclude
# all of these features with high missing values are categorical features
high_missing_data


high_missing_data_test = missing_data_test[missing_data_test > HIGH_MISSINGNESS_PERC].sort_values(ascending=False)
# decide later to impute or exclude
# all of these features with high missing values are categorical features
high_missing_data_test


plt.figure(figsize=(14, 10))
sns.heatmap(train_df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Pattern (Train)")
plt.xlabel("Features")
plt.ylabel("Samples")
plt.show()


plt.figure(figsize=(10, 4))
sns.heatmap(test_df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Pattern (Test)")
plt.xlabel("Features")
plt.ylabel("Samples")
plt.show()


plt.figure(figsize=(10, 8))
missing_data.plot(kind='barh')
plt.title('Percentage of Missing Values by Feature (Train)')
plt.xlabel('Percentage of Missing Values')
plt.ylabel('Features')
plt.show()


plt.figure(figsize=(8, 6))
missing_data_test.plot(kind='barh')
plt.title('Percentage of Missing Values by Feature (Test)')
plt.xlabel('Percentage of Missing Values')
plt.ylabel('Features')
plt.show()


numerical_features = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_features.remove('ID')
numerical_features.remove('efs')
numerical_features.remove('efs_time')

print(f'Numerical features count: {len(numerical_features)}')


numerical_features_test = test_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_features_test.remove('ID')

print(f'Numerical features (Test) count: {len(numerical_features_test)}')


plt.figure(figsize=(14, 10))
sns.heatmap(train_df[numerical_features].isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Pattern Numerical (Train)")
plt.xlabel("Features")
plt.ylabel("Samples")
plt.show()


plt.figure(figsize=(10, 4))
sns.heatmap(test_df[numerical_features_test].isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Pattern Numerical (Test)")
plt.xlabel("Features")
plt.ylabel("Samples")
plt.show()


missing_data_numerical = train_df[numerical_features].isnull().mean() * 100
missing_data_numerical = missing_data_numerical[missing_data_numerical > 0].sort_values(ascending=False)


plt.figure(figsize=(10, 8))
missing_data_numerical.plot(kind='barh')
plt.title('Percentage of Missing Values Numerical (Train)')
plt.xlabel('Percentage of Missing Values')
plt.ylabel('Features')
plt.show()


missing_data_numerical_test = test_df[numerical_features_test].isnull().mean() * 100
missing_data_numerical_test = missing_data_numerical_test[missing_data_numerical_test > 0].sort_values(ascending=False)


plt.figure(figsize=(6, 4))
missing_data_numerical_test.plot(kind='barh')
plt.title('Percentage of Missing Values Numerical (Test)')
plt.xlabel('Percentage of Missing Values')
plt.ylabel('Features')
plt.show()


numerical_summary_desc = train_df[numerical_features].describe().T
numerical_summary_desc


# 3 standard deviations from the mean
def has_outliers(series):
    mean, std = series.mean(), series.std()
    return ((series > mean + 3 * std) | (series < mean - 3 * std)).any()


numerical_summary = pd.DataFrame({
    'Feature': numerical_features,
    'Skewness': train_df[numerical_features].skew(),
    'Has Outliers': train_df[numerical_features].apply(has_outliers)
})

numerical_summary


SKEW_THRESHOLD = 0.75


skewed_features = numerical_summary[numerical_summary['Skewness'].abs() > SKEW_THRESHOLD]['Feature'].tolist()

features_with_outliers = numerical_summary[numerical_summary['Has Outliers']]['Feature'].tolist()

normally_distributed_features = numerical_summary[
    (numerical_summary['Skewness'].abs() <= SKEW_THRESHOLD) & (~numerical_summary['Has Outliers'])
]['Feature'].tolist()

print("Skewed Features:", skewed_features)
print("Features with Outliers:", features_with_outliers)
print("Normally Distributed Features:", normally_distributed_features)


numerical_corr_with_target = train_df[numerical_features].corrwith(train_df['efs']).sort_values(ascending=False)
numerical_corr_with_target


top_correlated_features = numerical_corr_with_target.head(6).index.tolist()
negatively_correlated_features = numerical_corr_with_target.tail(2).index.tolist()

selected_features = top_correlated_features + negatively_correlated_features
selected_features


for feature in selected_features:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [1, 2]})
    sns.boxplot(x=train_df[feature], ax=axes[0], color='lightblue')
    axes[0].set_title(f'Box Plot of {feature}')
    sns.histplot(train_df[feature], bins=20, kde=True, ax=axes[1])
    axes[1].set_title(f'Histogram of {feature}')
    plt.tight_layout()
    plt.show()


categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()
print(f'Categorical features count: {len(categorical_features)}')


categorical_features_test = test_df.select_dtypes(include=['object']).columns.tolist()
print(f'Categorical features (Test) count: {len(categorical_features_test)}')


plt.figure(figsize=(14, 10))
sns.heatmap(train_df[categorical_features].isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Pattern Categorical (Train)")
plt.xlabel("Features")
plt.ylabel("Samples")
plt.show()


plt.figure(figsize=(10, 4))
sns.heatmap(test_df[categorical_features_test].isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Pattern Categorical (Test)")
plt.xlabel("Features")
plt.ylabel("Samples")
plt.show()


missing_data_categorical = train_df[categorical_features].isnull().mean() * 100
missing_data_categorical = missing_data_categorical[missing_data_categorical > 0].sort_values(ascending=False)


plt.figure(figsize=(10, 8))
missing_data_categorical.plot(kind='barh')
plt.title('Percentage of Missing Values Categorical (Train)')
plt.xlabel('Percentage of Missing Values')
plt.ylabel('Features')
plt.show()


missing_data_categorical_test = test_df[categorical_features_test].isnull().mean() * 100
missing_data_categorical_test = missing_data_categorical_test[missing_data_categorical_test > 0].sort_values(ascending=False)


plt.figure(figsize=(6, 4))
missing_data_categorical_test.plot(kind='barh')
plt.title('Percentage of Missing Values Categorical (Test)')
plt.xlabel('Percentage of Missing Values')
plt.ylabel('Features')
plt.show()


import pandas as pd

# Thresholds
DOMINANT_THRESHOLD = 0.8 
RARE_THRESHOLD = 0.05

dominant_rare_info = {}

for feature in categorical_features:
    freq_counts = train_df[feature].value_counts(normalize=True)
    cumulative_percentage = freq_counts.cumsum()

    # Dominant categories
    dominant = freq_counts[cumulative_percentage <= DOMINANT_THRESHOLD].index.tolist()
    dominant_coverage = cumulative_percentage.loc[dominant[-1]] if dominant else 0.0

    # Rare categories
    rare = freq_counts[freq_counts < RARE_THRESHOLD].index.tolist()
    rare_coverage = freq_counts.loc[rare].sum() if rare else 0.0

    dominant_rare_info[feature] = {
        'total': len(freq_counts),
        'num_dominant': len(dominant),
        'num_rare': len(rare),
        'dominant_coverage': dominant_coverage * 100,
        'rare_coverage': rare_coverage * 100,
        'dominant_cats': dominant,
        'rare_cats': rare
    }

dominant_rare_df = pd.DataFrame(dominant_rare_info).T
dominant_rare_df


dominant_rare_info_test = {}

for feature in categorical_features_test:
    freq_counts = test_df[feature].value_counts(normalize=True)
    cumulative_percentage = freq_counts.cumsum()

    dominant = freq_counts[cumulative_percentage <= DOMINANT_THRESHOLD].index.tolist()
    dominant_coverage = cumulative_percentage.loc[dominant[-1]] if dominant else 0.0

    rare = freq_counts[freq_counts < RARE_THRESHOLD].index.tolist()
    rare_coverage = freq_counts.loc[rare].sum() if rare else 0.0

    dominant_rare_info_test[feature] = {
        'total': len(freq_counts),
        'num_dominant': len(dominant),
        'num_rare': len(rare),
        'dominant_coverage': dominant_coverage * 100,
        'rare_coverage': rare_coverage * 100,
        'dominant_cats': dominant,
        'rare_cats': rare
    }

dominant_rare_df_test = pd.DataFrame(dominant_rare_info_test).T
dominant_rare_df_test


# for feature in categorical_features:
#     plt.figure(figsize=(10, 5))
#     sns.countplot(data=train_df, x=feature, hue='efs', order=train_df[feature].value_counts().index)
#     plt.title(f'Relationship Between {feature} and efs')
#     plt.xlabel(feature)
#     plt.ylabel('Count')
#     plt.xticks(rotation=45, ha='right')
#     plt.show()


from scipy.stats import chi2_contingency

chi_square_results = {}

# Apply the Chi-Square test to the categorical columns with high missingness 
for feature in high_missing_data.index.to_list():
    if feature in train_df.columns:
        contingency_table = pd.crosstab(train_df[feature], train_df['efs'])
        chi2, p, _, _ = chi2_contingency(contingency_table)
        chi_square_results[feature] = {'Chi2': chi2, 'P-Value': p}

chi_square_df = pd.DataFrame(chi_square_results).T
chi_square_df.index.name = 'Feature'
chi_square_df.reset_index(inplace=True)

chi_square_df


# Features to drop due to high missing percentages
features_to_drop = ['tce_match', 'mrd_hct']

train_df_dropped = train_df.drop(columns=features_to_drop)
test_df_dropped = test_df.drop(columns=features_to_drop)

categorical_features_dropped = train_df_dropped.select_dtypes(include=['object']).columns.tolist()


train_df_dropped = train_df_dropped.fillna(train_df_dropped[categorical_features_dropped].mode().iloc[0])
test_df_dropped = test_df_dropped.fillna(test_df_dropped[categorical_features_dropped].mode().iloc[0])


median_imputed_cols = list(set(skewed_features + features_with_outliers))
mean_imputed_cols = normally_distributed_features

train_df_dropped = train_df_dropped.fillna(train_df_dropped[median_imputed_cols].median())
train_df_dropped = train_df_dropped.fillna(train_df_dropped[mean_imputed_cols].mean())
test_df_dropped = test_df_dropped.fillna(test_df_dropped[median_imputed_cols].median())
test_df_dropped = test_df_dropped.fillna(test_df_dropped[mean_imputed_cols].mean())


train_df_imputed = train_df_dropped.copy()
test_df_imputed = test_df_dropped.copy()


fig, axes = plt.subplots(1, 2, figsize=(16, 8))

sns.heatmap(train_df_imputed.isnull(), cbar=False, cmap='viridis', ax=axes[0])
axes[0].set_title("Missing Data Pattern (Train)")
axes[0].set_xlabel("Features")
axes[0].set_ylabel("Samples")
sns.heatmap(test_df_imputed.isnull(), cbar=False, cmap='viridis', ax=axes[1])
axes[1].set_title("Missing Data Pattern (Test)")
axes[1].set_xlabel("Features")
axes[1].set_ylabel("Samples")
plt.tight_layout()
plt.show()


HIGH_CARDINALITY_THRESHOLD = 7

unique_count = train_df_imputed[categorical_features_dropped].nunique()

high_cardinality_features = unique_count[unique_count > HIGH_CARDINALITY_THRESHOLD].index.tolist()
low_cardinality_features = unique_count[unique_count <= HIGH_CARDINALITY_THRESHOLD].index.tolist()

print(f'High Cardinality Features: {high_cardinality_features}')
print(f'Low Cardinality Features: {low_cardinality_features}')


train_df_preprocessed = train_df_imputed.copy()
test_df_preprocessed = test_df_imputed.copy()


!pip install category-encoders


# from category_encoders import TargetEncoder

# target_encoder = TargetEncoder()

# train_df_preprocessed[high_cardinality_features] = target_encoder.fit_transform(
#     train_df_imputed[high_cardinality_features], train_df_imputed['efs']
# )

# test_df_preprocessed[high_cardinality_features] = target_encoder.transform(
#     test_df_imputed[high_cardinality_features]
# )


# from sklearn.preprocessing import OrdinalEncoder

# ordinal_features_order = {
#     'cyto_score': ['Poor', 'Intermediate', 'Favorable'],
#     'cyto_score_detail': ['Poor', 'Intermediate', 'Favorable', 'TBD', 'Not tested'],
#     'conditioning_intensity': ['NMA', 'RIC', 'MAC', 'TBD', 'No drugs reported', 'N/A, F(pre-TED) not submitted'],
#     'pulm_severe': ['No', 'Yes', 'Not done'],
#     'hepatic_severe': ['No', 'Yes', 'Not done'],
#     'renal_issue': ['No', 'Yes', 'Not done'],
#     'obesity': ['No', 'Yes', 'Not done'],
#     'pulm_moderate': ['No', 'Yes', 'Not done'],
#     'hepatic_mild': ['No', 'Yes', 'Not done'],
# }

# for feature, order in ordinal_features_order.items():
#     encoder = OrdinalEncoder(categories=[order], handle_unknown='use_encoded_value', unknown_value=-1)
#     train_df_preprocessed[feature + '_ordinal'] = encoder.fit_transform(train_df_dropped[[feature]])
#     test_df_preprocessed[feature + '_ordinal'] = encoder.transform(test_df_dropped[[feature]])


from sklearn.preprocessing import LabelEncoder


# label_features = list(set(low_cardinality_features) - set(ordinal_features_order.keys()))
for feature in categorical_features_dropped:
    encoder = LabelEncoder()
    train_df_preprocessed[feature] = encoder.fit_transform(train_df_dropped[feature].astype(str))
    test_df_preprocessed[feature] = encoder.transform(test_df_dropped[feature].astype(str))


# train_df_preprocessed = train_df_preprocessed.drop(columns=low_cardinality_features)
# test_df_preprocessed = test_df_preprocessed.drop(columns=low_cardinality_features)


train_df_preprocessed_set = set(train_df_preprocessed.columns)
test_df_preprocessed_set = set(test_df_preprocessed.columns)

test_missing_cols = train_df_preprocessed_set - test_df_preprocessed_set
test_extra_cols = test_df_preprocessed_set - train_df_preprocessed_set

print(f'Missing cols in test.csv: {test_missing_cols}')
print(f'Extra cols in test.csv: {test_extra_cols}')

print(f'Train preprocessed dataset shape: {train_df_preprocessed.shape}')
print(f'Test preprocessed dataset shape: {test_df_preprocessed.shape}')


from sklearn.model_selection import train_test_split

X_preprocessed = train_df_preprocessed.drop(columns=['efs', 'efs_time'])
y_preprocessed =train_df_preprocessed['efs_time']

X_pp_train, X_pp_test, y_pp_train, y_pp_test = train_test_split(
    X_preprocessed, y_preprocessed, test_size=0.2, random_state=42
)


from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(random_state=42, n_estimators=100)
regressor.fit(X_pp_train, y_pp_train)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_pp_predict = regressor.predict(X_pp_test)

mae = mean_absolute_error(y_pp_test, y_pp_predict)
mse = mean_squared_error(y_pp_test, y_pp_predict)
rmse = np.sqrt(mse)
r2 = r2_score(y_pp_test, y_pp_predict)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")


X_preprocessed_test = test_df_preprocessed.copy()
predicted_efs_time = regressor.predict(X_preprocessed_test)
predicted_efs_time


test_df_preprocessed['efs_time'] = predicted_efs_time
test_df_preprocessed.head()


# train_df_preprocessed.to_csv('data/train_preprocessed.csv', index=True)
# test_df_preprocessed.to_csv('data/test_preprocessed.csv', index=True)


X = train_df_preprocessed.drop(columns=['ID', 'efs'])
y = train_df_preprocessed['efs']

X_test = test_df_preprocessed.drop(columns=['ID'])

# Ensure the columns in X_test are in the same order as in X
X_test = X_test[X.columns]

print(f'X dataset shape: {X.shape}')
print(f'X_test dataset shape: {X_test.shape}')


X_test_missing_cols = set(X.columns) - set(X_test.columns)
X_test_extra_cols = set(X_test.columns) - set(X.columns)

print(f'Missing cols in test.csv: {X_test_missing_cols}')
print(f'Extra cols in test.csv: {X_test_extra_cols}')


from sklearn.model_selection import StratifiedKFold

# Create stratified K-fold object
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=5000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'LightGBM': lgb.LGBMClassifier(random_state=42),
}


from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, make_scorer

scoring_metrics = {
    'roc_auc': make_scorer(roc_auc_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score)
}


!pip install xgboost


from sklearn.model_selection import cross_validate

# Evaluate each model using cross-validation
results = {}
for name, model in models.items():
    scores = cross_validate(model, X, y, cv=skf, scoring=scoring_metrics)

    roc_auc = scores['test_roc_auc'].mean()
    precision = scores['test_precision'].mean()
    recall = scores['test_recall'].mean()
    f1 = scores['test_f1'].mean()

    results[name] = {
        'ROC-AUC': roc_auc,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

results_df = pd.DataFrame(results).T
print("Baseline Model Performance:")
print(results_df)



fitted_models = {}
for name, model in models.items():
    model.fit(X, y)
    fitted_models[name] = model


best_model = fitted_models['XGBoost']


assert test_df_preprocessed['ID'].min() == 28800, "ID column should start from 28800"


assert list(X.columns) == list(X_test.columns), "Feature order mismatch!"


test_risk_scores = best_model.predict_proba(X_test)[:, 1]
submission = pd.DataFrame({
    'ID': test_df_preprocessed['ID'],
    'prediction': test_risk_scores
})

# Save the submission file in CSV format
submission_file_path = 'submission.csv'
submission.to_csv(submission_file_path, index=False)


test_risk_scores


print('DONE!')

