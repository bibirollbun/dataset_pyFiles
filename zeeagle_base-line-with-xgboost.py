# 1. Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score

!pip install xgboost
import xgboost as xgb


# 2. Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


categorical_columns = train.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_columns = train.select_dtypes(include=['number']).columns.tolist()


import matplotlib.pyplot as plt

for col in categorical_columns:
    plt.figure(figsize=(10, 6))
    train[col].value_counts().plot(kind='bar')
    plt.title(f'Frequency of each category in {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()


from scipy import stats
import matplotlib.pyplot as plt

# Exclude 'id' column from numerical columns
numerical_columns = [col for col in numerical_columns if col != 'id']

# Box plot for numerical columns
for col in numerical_columns:
    plt.figure(figsize=(10, 6))
    train.boxplot(column=col)
    plt.title(f'Box plot for {col}')
    plt.show()

# Z-score method for outlier detection
z_scores = train[numerical_columns].apply(stats.zscore)
outliers_z = (z_scores.abs() > 3).any(axis=1)
print(f'Number of outliers detected by Z-score method: {outliers_z.sum()}')

# IQR method for outlier detection
Q1 = train[numerical_columns].quantile(0.25)
Q3 = train[numerical_columns].quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = ((train[numerical_columns] < (Q1 - 1.5 * IQR)) | (train[numerical_columns] > (Q3 + 1.5 * IQR))).any(axis=1)
print(f'Number of outliers detected by IQR method: {outliers_iqr.sum()}')

# Display rows identified as outliers by either method
outliers_combined = outliers_z | outliers_iqr
display(train[outliers_combined])


# 3. Encode target
le = LabelEncoder()
train["Personality_encoded"] = le.fit_transform(train["Personality"])


# 4. Prepare features
X = train.drop(columns=["id", "Personality", "Personality_encoded"])
y = train["Personality_encoded"]
X_test = test.drop(columns=["id"])


# 5. Encode categorical columns
combined = pd.concat([X, X_test], axis=0)
cat_cols = combined.select_dtypes(include="object").columns.tolist()
encoder = OrdinalEncoder()
combined[cat_cols] = encoder.fit_transform(combined[cat_cols])

X = combined.iloc[:len(X)].reset_index(drop=True)
X_test = combined.iloc[len(X):].reset_index(drop=True)


X


# !pip install missingno
import missingno as msno
import scipy.stats as stats

# Visualize missingness pattern
msno.matrix(X)
display(msno.heatmap(X))

# Little's MCAR test (using statsmodels' implementation)
from statsmodels.imputation import mice

# Little's MCAR test is not directly available in Python, but we can use a workaround:
def littles_mcar_test(df):
    from numpy import isnan
    from scipy.stats import chi2
    import numpy as np

    df = df.copy()
    mask = df.isnull()
    n, p = df.shape
    patterns = mask.drop_duplicates()
    patterns_idx = [np.where((mask == pat).all(axis=1))[0] for _, pat in patterns.iterrows()]
    means = df.mean()
    cov = df.cov()
    stat = 0
    dfree = 0
    for idx in patterns_idx:
        sub = df.iloc[idx]
        obs = ~mask.iloc[idx[0]]
        n_g = len(idx)
        if n_g == 0 or obs.sum() == 0:
            continue
        sub_mean = sub.loc[:, obs].mean()
        mean_diff = sub_mean - means[obs]
        cov_g = cov.loc[obs, obs]
        try:
            inv_cov_g = np.linalg.inv(cov_g)
        except np.linalg.LinAlgError:
            continue
        stat += n_g * mean_diff.T @ inv_cov_g @ mean_diff
        dfree += obs.sum()
    p_value = 1 - chi2.cdf(stat, dfree)
    return stat, dfree, p_value

stat, dfree, p_value = littles_mcar_test(X)
display({"Little's MCAR test statistic": stat, "df": dfree, "p-value": p_value})


# Visualize missingness pattern with a bar plot
msno.bar(X)
display(msno.bar(X))

# Visualize missingness pattern with a dendrogram
msno.dendrogram(X)
display(msno.dendrogram(X))


# Interpretation of Little's MCAR test results
if p_value < 0.05:
    interpretation = "The missing data is not Missing Completely At Random (MCAR)."
else:
    interpretation = "The missing data is likely Missing Completely At Random (MCAR)."

display({"Interpretation": interpretation})


from statsmodels.imputation.mice import MICEData
import pandas as pd

# Convert the DataFrame to a format suitable for MICE
mice_data = MICEData(X)

# Perform multiple imputations
mice_data.update_all()

# Check if the imputed values are significantly different from the observed values
imputed_data = pd.DataFrame(mice_data.data, columns=X.columns)

# Compare the distributions of the observed and imputed data
comparison = {}
for column in X.columns:
    observed = X[column].dropna()
    imputed = imputed_data[column][X[column].isna()]
    comparison[column] = {
        "observed_mean": observed.mean(),
        "imputed_mean": imputed.mean(),
        "observed_std": observed.std(),
        "imputed_std": imputed.std()
    }

display(comparison)


threshold = 0.1  # Define a threshold for relative mean difference

mar_status = {}
for column, stats in comparison.items():
    observed_mean = stats["observed_mean"]
    imputed_mean = stats["imputed_mean"]
    if pd.isna(observed_mean) or observed_mean == 0:
        rel_diff = abs(imputed_mean - observed_mean)
    else:
        rel_diff = abs(imputed_mean - observed_mean) / abs(observed_mean)
    
    if pd.isna(imputed_mean):
        mar_status[column] = "No Missing Values"  
    elif rel_diff > threshold:
        mar_status[column] = "Likely MAR (Missing At Random) or MNAR (Not Missing At Random)"
    else:
        mar_status[column] = "Likely MCAR (Missing Completely At Random)"

display(mar_status)


from scipy.stats import ttest_ind

mar_status = {}
for column, stats in comparison.items():
    observed = X[column].dropna()
    imputed = imputed_data[column][X[column].isna()]

    observed_mean = stats["observed_mean"]
    imputed_mean = stats["imputed_mean"]
    
    p_value = ttest_ind(observed, imputed, nan_policy='omit').pvalue

    if pd.isna(imputed_mean):
        mar_status[column] = "No Missing Values"    
    elif p_value <= 0.01:
        mar_status[column] = f"Likely MAR (Missing At Random) or MNAR (Not Missing At Random)"
        mar_status[column + '_p_value'] = f"{p_value}"
    else:
        mar_status[column] = "Likely MCAR (Missing Completely At Random)"
        mar_status['p_value'] = f"{p_value}"


display(mar_status)


from scipy.stats import ttest_ind

columns_to_test = ["Social_event_attendance", "Going_outside", "Post_frequency"]
mar_mnar_results = {}

for col in columns_to_test:
    missing_mask = X[col].isna()
    for other_col in X.columns:
        if other_col == col:
            continue
        observed = X.loc[~missing_mask, other_col].dropna()
        missing = X.loc[missing_mask, other_col].dropna()
        if len(observed) > 0 and len(missing) > 0:
            stat, pval = ttest_ind(observed, missing, equal_var=False)
            if pval < 0.05:
                mar_mnar_results.setdefault(col, []).append(
                    f"Missingness in {col} is associated with {other_col} (p={pval:.3g}) → Likely MAR"
                )
    if col not in mar_mnar_results:
        mar_mnar_results[col] = ["No significant association with other variables → Possibly MNAR"]

display(mar_mnar_results)


from sklearn.impute import KNNImputer

# Select columns to impute and their associated columns
mar_columns = ["Social_event_attendance", "Going_outside", "Post_frequency"]
associated_columns = {col: [assoc.split(" is associated with ")[1].split(" (p=")[0] 
                           for assoc in mar_mnar_results[col] if "associated with" in assoc]
                      for col in mar_columns}

# Impute each MAR column using KNNImputer with its associated columns
for col in mar_columns:
    cols_for_impute = [col] + associated_columns[col]
    # Ensure columns exist and are numeric
    cols_for_impute = [c for c in cols_for_impute if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
    mar_imputer = KNNImputer(n_neighbors=5)
    X[cols_for_impute] = mar_imputer.fit_transform(X[cols_for_impute])

display(X[mar_columns])


# Impute the corresponding columns in X_test using the mar_imputer trained on X
for col in mar_columns:
    cols_for_impute = [col] + associated_columns[col]
    # Ensure columns exist and are numeric
    cols_for_impute = [c for c in cols_for_impute if c in X_test.columns and pd.api.types.is_numeric_dtype(X_test[c])]
    if cols_for_impute:
        X_test[cols_for_impute] = mar_imputer.fit_transform(X_test[cols_for_impute])

display(X_test[mar_columns])


# Identify columns with missing values that were not imputed (i.e., MCAR columns)
imputed_cols = set(["Social_event_attendance", "Going_outside", "Post_frequency"])
mcar_columns = [col for col in X.columns if X[col].isna().any() and col not in imputed_cols and pd.api.types.is_numeric_dtype(X[col])]

if mcar_columns:
    mcar_imputer = KNNImputer(n_neighbors=5)
    X[mcar_columns] = pd.DataFrame(mcar_imputer.fit_transform(X[mcar_columns]), columns=mcar_columns)



if mcar_columns:
    X_test[mcar_columns] = pd.DataFrame(mcar_imputer.transform(X_test[mcar_columns]), columns=mcar_columns)


X.info()


X_test.info()


display(train['Personality'].value_counts())


personality_counts = train['Personality'].value_counts(normalize=True) * 100
introvert_percentage = personality_counts.get('Introvert', 0)
extrovert_percentage = personality_counts.get('Extrovert', 0)

introvert_percentage, extrovert_percentage


introverts = train[train['Personality'] == 'Introvert']
extroverts_sample = train[train['Personality'] == 'Extrovert'].sample(frac=0.35, random_state=42)

train_balanced = pd.concat([introverts, extroverts_sample]).sort_index()
display(train_balanced)


X_balanced = X.loc[train_balanced.index]
display(X_balanced)


y_balanced = y.loc[train_balanced.index]
display(y_balanced)


# 6. Setup XGBoost
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}


# 7. Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_balanced))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_balanced, y_balanced)):
    X_train, X_val = X_balanced.iloc[train_idx], X_balanced.iloc[val_idx]
    y_train, y_val = y_balanced.iloc[train_idx], y_balanced.iloc[val_idx]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    model = xgb.train(params, dtrain, num_boost_round=100,
                      evals=[(dval, "valid")],
                      early_stopping_rounds=10, verbose_eval=False)
    
    oof_preds[val_idx] = model.predict(dval) > 0.5
    test_preds += model.predict(dtest) / skf.n_splits


# 8. Evaluate
cv_acc = accuracy_score(y_balanced, oof_preds)
print(f"Cross-Validation Accuracy: {cv_acc:.4f}")

# 9. Create submission
final_preds = (test_preds > 0.5).astype(int)
submission["Personality"] = le.inverse_transform(final_preds)
submission.to_csv("submission.csv", index=False)
submission.head()



display(submission)




