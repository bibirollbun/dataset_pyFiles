import numpy as np 
import pandas as pd
import joblib

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
print(train.head())


print(train.info())


print(train.isnull().sum())


print(train["efs"].value_counts(normalize=True))


import matplotlib.pyplot as plt
missing_values = train.isnull().sum()  # Count missing values per column

plt.figure(figsize=(15, 6))
missing_values.plot(kind='bar', color='red', alpha=0.7)

plt.axhline(y=28800, color='blue', linestyle='--', label="Total Entries")

plt.title("Missing Values Per Column Compared to Total Entries (28,800)")
plt.xlabel("Columns")
plt.ylabel("Number of Missing Values")
plt.xticks(rotation=90)  # Rotate column names for better visibility
plt.legend()
plt.show()





from scipy.stats import chi2_contingency
categorical_cols = ["pulm_moderate","cardiac","hepatic_mild","tce_div_match","donor_related","melphalan_dose","race_group","sex_match","rheum_issue","gvhd_proph","peptic_ulcer","prior_tumor","ethnicity","conditioning_intensity","cyto_score_detail","prod_type","graft_type","vent_hist","renal_issue","pulm_severe","prim_disease_hct", "dri_score","psych_disturb","cyto_score","diabetes","tbi_status","arrhythmia","cmv_status","tce_imm_match","rituximab","obesity","mrd_hct","in_vivo_tcd","tce_match","hepatic_severe"]
p_values = {}

# Perform Chi-Square Test for each categorical feature
for col in categorical_cols:
    contingency_table = pd.crosstab(train[col], train["efs"])
    _, p_value, _, _ = chi2_contingency(contingency_table)
    p_values[col] = p_value

# Convert results to DataFrame
p_values_df = pd.DataFrame(list(p_values.items()), columns=['Feature', 'p_value'])
p_values_df.sort_values(by="p_value", ascending=True, inplace=True)

# Plot the p-values
plt.figure(figsize=(12, 6))
bars = plt.bar(p_values_df["Feature"], p_values_df["p_value"], color=["red" if p < 0.05 else "gray" for p in p_values_df["p_value"]])

plt.axhline(y=0.05, color='blue', linestyle='--', label="Significance Threshold (p=0.05)")

plt.xticks(rotation=90, ha="right")
plt.ylabel("Chi-Square Test p-value")
plt.title("Chi-Square Test Results for Categorical Features vs. EFS")
plt.legend()

# Highlight significant features in red
for bar, p in zip(bars, p_values_df["p_value"]):
    if p < 0.05:
        bar.set_color("red")

plt.show()


numerical_cols = ["hla_match_c_high", "hla_high_res_8", "hla_low_res_6", "hla_high_res_6", "hla_high_res_10", "hla_match_dqb1_high", "hla_nmdp_6", "hla_match_c_low", "hla_match_drb1_low", "hla_match_dqb1_low", "hla_match_a_high", "donor_age", "hla_match_b_low", "age_at_hct", "hla_match_a_low", "hla_match_b_high", "comorbidity_score", "karnofsky_score", "hla_low_res_8", "hla_match_drb1_high", "hla_low_res_10"]
# Include both target columns
correlation_results = train[numerical_cols + ["efs_time", "efs"]].corr()

# Extract correlations with both target variables
efs_time_corr = correlation_results["efs_time"].drop("efs_time").sort_values(ascending=False)
efs_corr = correlation_results["efs"].drop("efs").sort_values(ascending=False)

# Print both correlation results
print("Correlation with efs_time:")
print(efs_time_corr)

print("\nCorrelation with efs:")
print(efs_corr)


from scipy.stats import f_oneway

p_values_anova = {}

# Perform ANOVA Test for each numerical feature vs. efs
for col in numerical_cols:
    groups = train.groupby("efs")[col]
    p_value = f_oneway(*[group.dropna() for _, group in groups])[1]  # Extract p-value
    p_values_anova[col] = p_value

# Convert to DataFrame
p_values_anova_df = pd.DataFrame(list(p_values_anova.items()), columns=['Feature', 'p_value'])
p_values_anova_df.sort_values(by="p_value", ascending=True, inplace=True)

# Plot ANOVA results (Fixed Scaling & Labels)
plt.figure(figsize=(14, 6))
bars = plt.bar(p_values_anova_df["Feature"], p_values_anova_df["p_value"], 
               color=["red" if p < 0.05 else "gray" for p in p_values_anova_df["p_value"]])

plt.axhline(y=0.05, color='blue', linestyle='--', label="Significance Threshold (p=0.05)")

# Set log scale to better visualize small differences
plt.yscale("log")  
plt.xticks(rotation=90, ha="right")
plt.ylabel("ANOVA Test p-value (Log Scale)")
plt.title("ANOVA Test Results for Numerical Features vs. EFS")
plt.legend()

# Add value labels on top of bars
for bar, p in zip(bars, p_values_anova_df["p_value"]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{p:.3f}", 
             ha="center", va="bottom", fontsize=9, color="black", rotation=90)

plt.show()




# Drop the irrelevant columns
columns_to_drop = ["rheum_issue", "melphalan_dose", "rituximab", "vent_hist"]
train.drop(columns=columns_to_drop, inplace=True)
# Confirm they are removed
print("Remaining columns:", train.columns)



#smart imputation code
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier


# Numerical - KNN imputer (after isolating numerical data)
knn_imputer = KNNImputer(n_neighbors=5)
train[numerical_cols] = knn_imputer.fit_transform(train[numerical_cols])


# Categorical - mode imputation or random forest
for col in categorical_cols:
    if col not in train.columns:
        print(f"Warning: Column '{col}' not found in train DataFrame. Skipping...")
        continue

    if train[col].isnull().sum() == 0:
        continue
    # Use mode if too many missing values or low cardinality
    if train[col].nunique() <= 5 or train[col].isnull().mean() > 0.3:
        mode_val = train[col].mode()[0]
        train[col].fillna(mode_val, inplace=True)
    else:
        # Use RF Classifier on most frequent class
        train_notnull = train[train[col].notnull()]
        train_null = train[train[col].isnull()]
        
        if train_notnull.shape[0] > 100 and train_null.shape[0] > 0:
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            X_train = train_notnull.drop([col, 'efs', 'efs_time'], axis=1)
            y_train = train_notnull[col]
            X_pred = train_null.drop([col, 'efs', 'efs_time'], axis=1)

            # Drop non-numeric cols for RF model
            X_train = X_train.select_dtypes(include=[np.number]).fillna(-1)
            X_pred = X_pred.select_dtypes(include=[np.number]).fillna(-1)
            
            rf.fit(X_train, y_train)
            train.loc[train[col].isnull(), col] = rf.predict(X_pred)
        else:
            # Fallback to mode
            train[col].fillna(train[col].mode()[0], inplace=True)


from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Ensure all categorical columns exist in the DataFrame
categorical_cols = [col for col in categorical_cols if col in train.columns]

# Check if there are categorical columns left after filtering
if categorical_cols:
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(train[categorical_cols])
    encoded_train = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=train.index)

    # Concatenate with original DataFrame
    train = pd.concat([train, encoded_train], axis=1)
else:
    print("Warning: No categorical columns found for encoding.")



# Final check
print("Final shape:", train.shape)
print("Missing values remaining:\n", train.isnull().sum().sort_values(ascending=False).head(10))


print(train.isnull().sum()[train.isnull().sum() > 0])


# study for outliers 
from scipy.stats import zscore
print("Z-SCORE OUTLIERS:")
z_thresh = 3
z_scores = train[numerical_cols].apply(zscore)
z_outliers = (np.abs(z_scores) > z_thresh)
outlier_counts_z = z_outliers.sum().sort_values(ascending=False)
print(outlier_counts_z[outlier_counts_z > 0])


def winsorize_series(series, lower=0.01, upper=0.99):
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    return np.clip(series, lower_bound, upper_bound)

# Apply winsorization to outlier columns only
outlier_cols = ["comorbidity_score", "karnofsky_score", "hla_match_dqb1_low", "hla_match_c_high",
                "hla_match_c_low", "hla_match_dqb1_high", "hla_match_b_high", "hla_match_drb1_high",
                "hla_match_b_low", "hla_match_a_high", "hla_match_a_low"]

for col in outlier_cols:
    train[col] = winsorize_series(train[col])


print("Z-SCORE OUTLIERS:")
z_thresh = 3
z_scores = train[numerical_cols].apply(zscore)
z_outliers = (np.abs(z_scores) > z_thresh)
outlier_counts_z = z_outliers.sum().sort_values(ascending=False)
print(outlier_counts_z[outlier_counts_z > 0])


#For comorbidity_score (still 559 outliers)
#This likely indicates right skew or a non-normal distribution 
#Use a log transform to compress the upper tail
train['comorbidity_score_log'] = np.log1p(train['comorbidity_score'])  # safer version of log



print("Z-SCORE OUTLIERS:")
z_thresh = 3
z_scores = train[numerical_cols].apply(zscore)
z_outliers = (np.abs(z_scores) > z_thresh)
outlier_counts_z = z_outliers.sum().sort_values(ascending=False)
print(outlier_counts_z[outlier_counts_z > 0])


z_scores_log = zscore(train['comorbidity_score_log'])
print("Outliers in log version:", (np.abs(z_scores_log) > 3).sum())



#For hla_high_res_* and hla_low_res_* columns (1–2 outliers each)
#apply slight winsorization (0.5% - 99.5%) just for those columns
cols_to_adjust = ['hla_high_res_6', 'hla_high_res_8', 'hla_high_res_10', 'hla_low_res_8']
for col in cols_to_adjust:
    train[col] = winsorize_series(train[col], lower=0.005, upper=0.995)


print("Z-SCORE OUTLIERS:")
z_thresh = 3
z_scores = train[numerical_cols].apply(zscore)
z_outliers = (np.abs(z_scores) > z_thresh)
outlier_counts_z = z_outliers.sum().sort_values(ascending=False)
print(outlier_counts_z[outlier_counts_z > 0])


from sklearn.preprocessing import LabelEncoder

# Identify categorical columns (object dtype)
object_cols = train.select_dtypes(include='object').columns
print(object_cols)
# Encode with LabelEncoder
label_encoders = {}
for col in object_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    label_encoders[col] = le
joblib.dump(label_encoders, 'label_encoders.pkl')


encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded = encoder.fit_transform(train[categorical_cols])
joblib.dump(encoder, '/kaggle/working/encoder.pkl')



!pip install lifelines --quiet


from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index

# Ensure 'efs' is binary and 'efs_time' is positive
assert set(train['efs'].unique()).issubset({0, 1})
assert (train['efs_time'] > 0).all()


# Step 1: Kaplan-Meier Estimator
kmf = KaplanMeierFitter()
kmf.fit(durations=train['efs_time'], event_observed=train['efs'])

# Plot survival curve
kmf.plot_survival_function()
plt.title("Kaplan-Meier Survival Curve")
plt.xlabel("Time")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.show()


#PROBLEMATIC COLUMNS CHECK DUE TO ERROR IN MODELING: 
events = train['efs'].astype(bool)
print("Variance when event occurred:", train.loc[events, 'gvhd_proph_FK+- others(not MMF,MTX)'].var())
print("Variance when no event:", train.loc[~events, 'gvhd_proph_FK+- others(not MMF,MTX)'].var())



# Step 1: Identify near-zero variance columns (excluding target columns)
# Manually drop features with near-zero variance
low_variance_cols = []

for col in train.select_dtypes(include='number').columns:
    if col not in ['efs', 'efs_time']:
        var = train[col].var()
        if var < 1e-4:
            low_variance_cols.append(col)

print("Dropping these low variance columns:", low_variance_cols)

# Drop and proceed with survival model
train_filtered = train.drop(columns=low_variance_cols)


#Prepare data for Cox model
covariates_cleaned = train_filtered.drop(columns=['efs', 'efs_time'])
data_for_cox = train_filtered[['efs_time', 'efs']].join(covariates_cleaned)


# Initialize the CoxPHFitter with a small penalizer
cph = CoxPHFitter(penalizer=0.1)

# Fit the model
cph.fit(data_for_cox, duration_col='efs_time', event_col='efs')

# Display the summary
cph.print_summary()



#test deepsurv with pycox

!pip install torchtuples pycox --quiet



import torchtuples as tt
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from sklearn.preprocessing import StandardScaler
import torch

# Step 1: Prepare and one-hot encode
df = train.copy()
df = df.drop(columns=['ID'], errors='ignore')
df = pd.get_dummies(df, drop_first=True)


# After one-hot encoding:
df = df.drop(columns=['ID'], errors='ignore')
X = df.drop(columns=['efs', 'efs_time'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype('float32')



# Step 2: Extract features and targets
y_time = df['efs_time'].values
y_event = df['efs'].values


# Step 3: Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype('float32')



# Save train columns and scaler
train_columns = df.drop(columns=['efs', 'efs_time']).columns
np.save('/kaggle/working/train_columns.npy', train_columns)
joblib.dump(scaler, '/kaggle/working/scaler.pkl')



# Step 4: Build neural net
in_features = X_scaled.shape[1]
net = tt.practical.MLPVanilla(in_features, [64, 32], 1, batch_norm=True, dropout=0.1)

model = CoxPH(net, tt.optim.Adam)
model.fit(X_scaled, (y_time, y_event), batch_size=256, epochs=100, verbose=True)




# Compute baseline hazards (from training set)
model.compute_baseline_hazards(X_scaled, (y_time, y_event))

# Compute baseline cumulative hazards
model.compute_baseline_cumulative_hazards()


# Save baseline hazards only (best practice)
joblib.dump(model.baseline_hazards_, 'baseline_hazards.pkl')
joblib.dump(model.baseline_cumulative_hazards_, 'baseline_cumulative_hazards.pkl')




# Step 5: Predict and evaluate

# Required before using `predict_surv_df`
model.compute_baseline_hazards()
surv = model.predict_surv_df(X_scaled)
ev = EvalSurv(surv, y_time, y_event, censor_surv='km')
c_index = ev.concordance_td('antolini')

print(f" DeepSurv C-index: {c_index:.4f}")



# Save to Kaggle output directory
model_path = "/kaggle/working/deepsurv_model.pth"
torch.save({
    'model_state_dict': model.net.state_dict(),
    'optimizer_state_dict': model.optimizer.state_dict(),
}, model_path)

print(f"Model saved to: {model_path}")


test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
test.drop(columns=columns_to_drop, inplace=True)


#smart imputation code
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier

# Numerical - KNN imputer (after isolating numerical data)
knn_imputer = KNNImputer(n_neighbors=5)
test[numerical_cols] = knn_imputer.fit_transform(test[numerical_cols])


# Mode/RF for categorical
for col in categorical_cols:
    if col not in test.columns:
        continue
    if test[col].isnull().sum() == 0:
        continue
    if test[col].nunique() <= 5 or test[col].isnull().mean() > 0.3:
        test[col] = test[col].fillna(test[col].mode()[0])


def winsorize_series(series, lower=0.01, upper=0.99):
    return np.clip(series, series.quantile(lower), series.quantile(upper))

for col in outlier_cols:
    test[col] = winsorize_series(test[col])


for col in cols_to_adjust:
    test[col] = winsorize_series(test[col], lower=0.005, upper=0.995)



test['comorbidity_score_log'] = np.log1p(test['comorbidity_score'])


label_encoders = joblib.load('/kaggle/working/label_encoders.pkl')
for col in object_cols:
    if col in test.columns:
        le = label_encoders[col]
        test[col] = le.transform(test[col].astype(str))


df = test.copy()
encoder = joblib.load('/kaggle/working/encoder.pkl')
encoded = encoder.transform(test[categorical_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=test.index)

# Combine with other numerical/log features
test_final = pd.concat([test.drop(columns=categorical_cols + ['ID'], errors='ignore'), encoded_df], axis=1)
train_columns = np.load("/kaggle/working/train_columns.npy", allow_pickle=True)

# Add missing columns with 0s
for col in train_columns:
    if col not in test_final.columns:
        test_final[col] = 0

# Drop extra columns
test_final = test_final[[col for col in train_columns if col in test_final.columns]]

# Optional: print mismatches for debugging
missing = set(train_columns) - set(test_final.columns)
extra = set(test_final.columns) - set(train_columns)
print("Missing:", missing)
print("Extra:", extra)



# Add missing columns with 0s
for col in train_columns:
    if col not in df.columns:
        df[col] = 0
# Remove extra columns that were not in training
df = df[[col for col in train_columns if col in df.columns]]
df = df.drop(columns=['ID', 'efs', 'efs_time'], errors='ignore')


print("Shape after alignment:", df.shape)  # Should be (3, 206)


scaler = joblib.load('/kaggle/working/scaler.pkl')
X_scaled = scaler.transform(df).astype('float32')


# 1. Prepare the test features
X_test_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# 2. Rebuild the same network architecture
# Same model architecture
in_features = X_scaled.shape[1]
net = tt.practical.MLPVanilla(in_features, [64, 32], 1, batch_norm=True, dropout=0.1)
model = CoxPH(net)


# Load model weights
checkpoint = torch.load("/kaggle/working/deepsurv_model.pth", map_location='cpu')
model.net.load_state_dict(checkpoint['model_state_dict'])


# Load baseline hazards computed from training
model.baseline_hazards_ = joblib.load('/kaggle/working/baseline_hazards.pkl')
model.baseline_cumulative_hazards_ = joblib.load('/kaggle/working/baseline_cumulative_hazards.pkl')


# Now you can predict survival curves
surv = model.predict_surv_df(X_test_tensor)
risk_scores = -model.predict(X_test_tensor)


# 6. Prepare submission
submission = pd.DataFrame({
    'ID': test['ID'],
    'prediction': risk_scores.numpy().flatten()
})




# 9. Save to CSV
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("Submission saved to submission.csv")


