import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')


train.head(3)


test.head(3)


def chart(patient_id, ax):
    data = train[train['Patient'] == patient_id]
    x = data['Weeks']
    y = data['FVC']
    ax.set_title(patient_id)
    ax = sns.regplot(x, y, ax=ax, ci=None, line_kws={'color':'red'})
    

f, axes = plt.subplots(1, 3, figsize=(15, 5))
chart('ID00419637202311204720264', axes[0])
chart('ID00009637202177434476278', axes[1])
chart('ID00010637202177584971671', axes[2])


# Kaggle, please add Pyro/PyTorch support!
import pymc3 as pm
import theano
import arviz as az
from sklearn import preprocessing


# Very simple pre-processing: adding patient class
def patient_class(row):
    if row['Sex'] == 'Male':
        if row['SmokingStatus'] == 'Currently smokes':
            return 0
        elif row['SmokingStatus'] == 'Ex-smoker':
            return 1
        elif row['SmokingStatus'] == 'Never smoked':
            return 2
    else:
        if row['SmokingStatus'] == 'Currently smokes':
            return 3
        elif row['SmokingStatus'] == 'Ex-smoker':
            return 4
        elif row['SmokingStatus'] == 'Never smoked':
            return 5

train['Class'] = train.apply(patient_class, axis=1)


train['Patient'].nunique()


aux = train[['Patient', 'Weeks']].groupby('Patient')\
    .min().reset_index()
aux = pd.merge(aux, train[['Patient', 'Weeks', 'FVC']], how='left', 
               on=['Patient', 'Weeks'])

# aux = aux.groupby('Patient').mean().reset_index()
a =aux.groupby(['Patient'],as_index=False)['Weeks'].count()
aux[aux['Patient']=='ID00048637202185016727717']


# Very simple pre-processing: adding FVC and week baselines
aux = train[['Patient', 'Weeks']].groupby('Patient')\
    .min().reset_index()
aux = pd.merge(aux, train[['Patient', 'Weeks', 'FVC']], how='left', 
               on=['Patient', 'Weeks'])
aux = aux.groupby('Patient').mean().reset_index()
aux['Weeks'] = aux['Weeks'].astype(int)
aux['FVC'] = aux['FVC'].astype(int)
aux
train = pd.merge(train, aux, how='left', on='Patient', suffixes=('', '_base'))


train


# Very simple pre-processing: creating patient indexes
le = preprocessing.LabelEncoder()
train['PatientID'] = le.fit_transform(train['Patient'])

patients = train[['Patient', 'PatientID', 'Age', 'Class', 'Weeks_base', 'FVC_base']].drop_duplicates()
fvc_data = train[['Patient', 'PatientID', 'Weeks', 'FVC']]

# patients


fvc_data.head()


len(fvc_data['Weeks'])



FVC_b = patients['FVC_base'].values
w_b = patients['Weeks_base'].values
age = patients['Age'].values
patient_class = patients['Class'].values

t = fvc_data['Weeks'].values
FVC_obs = fvc_data['FVC'].values
patient_id = fvc_data['PatientID'].values

with pm.Model() as hierarchical_model:
    # Hyperpriors for Alpha
    beta_int = pm.Normal('beta_int', 0, sigma=100)
    sigma_int = pm.HalfNormal('sigma_int', 100)
    
    # Alpha
    mu_alpha = FVC_b + beta_int * w_b
    alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_int, 
                      shape=train['Patient'].nunique())
    
    # Hyperpriors for Beta
    sigma_s = pm.HalfNormal('sigma_s', 100)
    alpha_s = pm.Normal('alpha_s', 0, sigma=100)
    beta_cs = pm.Normal('beta_cs', 0, sigma=100, shape=6)
    
    # Beta
    mu_beta = alpha_s + age * beta_cs[patient_class]
    beta = pm.Normal('beta', mu=mu_beta, sigma=sigma_s,
                     shape=train['Patient'].nunique())
    
    # Model variance
    sigma = pm.HalfNormal('sigma', 200)
    
    # Model estimate
    FVC_est = alpha[patient_id] + beta[patient_id] * t
    
    # Data likelihood
    FVC_like = pm.Normal('FVC_like', mu=FVC_est,
                          sigma=sigma, observed=FVC_obs)


# Inference button (TM)!
with hierarchical_model:
    trace = pm.sample(2000, tune=2000, target_accept=.9)


len(trace['sigma'])


# Very simple pre-processing: adding patient class
def patient_class(row):
    if row['Sex'] == 'Male':
        if row['SmokingStatus'] == 'Currently smokes':
            return 0
        elif row['SmokingStatus'] == 'Ex-smoker':
            return 1
        elif row['SmokingStatus'] == 'Never smoked':
            return 2
    else:
        if row['SmokingStatus'] == 'Currently smokes':
            return 3
        elif row['SmokingStatus'] == 'Ex-smoker':
            return 4
        elif row['SmokingStatus'] == 'Never smoked':
            return 5

test['Class'] = test.apply(patient_class, axis=1)
test = test.rename(columns={'FVC': 'FVC_base', 'Weeks': 'Weeks_base'})
test.head()


# prepare submission dataset
submission = []
for i, patient in enumerate(test['Patient'].unique()):
    df = pd.DataFrame(columns=['Patient', 'Weeks', 'FVC'])
    df['Weeks'] = np.arange(-12, 134)
    df['Patient'] = patient
    df['PatientID'] = i
    df['FVC'] = 0
    submission.append(df)
    
submission = pd.concat(submission).reset_index(drop=True)
submission.head()


trace['beta_cs'].mean()


FVC_b = test['FVC_base'].values
w_b = test['Weeks_base'].values
age = test['Age'].values
patient_class = test['Class'].values
t = submission['Weeks'].values
patient_id = submission['PatientID'].values
            
with pm.Model() as new_model:
    # Hyperpriors for Alpha
    beta_int = pm.Normal('beta_int', 
                         trace['beta_int'].mean(), 
                         sigma=trace['beta_int'].std())
    sigma_int = pm.TruncatedNormal('sigma_int', 
                                   trace['sigma_int'].mean(),
                                   sigma=trace['sigma_int'].std(),
                                   lower=0)
    
    # Alpha
    mu_alpha = FVC_b + beta_int * w_b
    alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_int, 
                      shape=test['Patient'].nunique())
    
    # Hyperpriors for Beta
    sigma_s = pm.TruncatedNormal('sigma_s', 
                                 trace['sigma_s'].mean(),
                                 sigma=trace['sigma_s'].std(),
                                 lower=0)
    alpha_s = pm.Normal('alpha_s', 
                        trace['alpha_s'].mean(), 
                        sigma=trace['alpha_s'].std())
    cov = np.zeros((6, 6))
    np.fill_diagonal(cov, trace['beta_cs'].var(axis=0))
    beta_cs = pm.MvNormal('beta_cs',
                          mu=trace['beta_cs'].mean(axis=0),
                          cov=cov,
                          shape=6)
    
    # Beta
    mu_beta = alpha_s + age * beta_cs[patient_class]
    beta = pm.Normal('beta', mu=mu_beta, sigma=sigma_s,
                     shape=test['Patient'].nunique())
    
    # Model variance
    sigma = pm.TruncatedNormal('sigma', 
                               trace['sigma'].mean(),
                               sigma=trace['sigma'].std(),
                               lower=0)
    
    # Model estimate
    # Here, there are 2 ways of estimating FVC. One is deterministic, the other
    # stochastic. Assuming FVC is deterministic, we calculate sigma later, by
    # evaluating std dev over the 4000 different models. This yields a higher 
    # confidence (lower sigmas). Assuming FVC is stochastic (commented code below)
    # yields irregular lines. The mean FVC values are about the same, but the 
    # confidence is much lower (higher sigmas, about 2x the first case). Let's
    # try submitting both cases, starting by the 1st assumption.
    FVC_est = pm.Deterministic('FVC_est', alpha[patient_id] + beta[patient_id] * t)
    
    # sigma = pm.HalfNormal('sigma', 200)
    # FVC_like = pm.Normal('FVC_like', mu=alpha[patient_id] + beta[patient_id] * t, 
    #                      sigma=sigma,
    #                      shape=submission.shape[0])


with new_model:
    trace2 = pm.sample(2000, tune=2000, target_accept=.9)


trace2['FVC_est']


trace2


preds = pd.DataFrame(data=trace2['FVC_est'].T)
submission = pd.merge(submission, preds, left_index=True, right_index=True)
submission['Patient_Week'] = submission['Patient'] + '_' + submission['Weeks'].astype(str)
submission = submission.drop(columns=['Patient', 'Weeks', 'FVC', 'PatientID'])

FVC = submission.iloc[:, :-1].mean(axis=1)
confidence = submission.iloc[:, :-1].std(axis=1)
submission['FVC'] = FVC
submission['Confidence'] = confidence
submission = submission[['Patient_Week', 'FVC', 'Confidence']]
submission.to_csv('submission.csv', index=False)
submission.head()


# Define function to calculate patient class
def calculate_patient_class(sex, smoking_status):
    if sex == 'Male':
        if smoking_status == 'Currently smokes':
            return 0
        elif smoking_status == 'Ex-smoker':
            return 1
        elif smoking_status == 'Never smoked':
            return 2
    else:
        if smoking_status == 'Currently smokes':
            return 3
        elif smoking_status == 'Ex-smoker':
            return 4
        elif smoking_status == 'Never smoked':
            return 5

# Define function to predict FVC for upcoming weeks
def predict_fvc_for_patient(patient_id, age, sex, smoking_status, current_week, upcoming_weeks):
    # Check if patient exists in the test dataset
    if patient_id not in test['Patient'].values:
        print(f"Patient with ID '{patient_id}' not found in the test dataset.")
        return None
    
    # Calculate patient class
    patient_class = calculate_patient_class(sex, smoking_status)
    
    # Get patient index
    patient_index = test[test['Patient'] == patient_id].index[0]
    
    # Predict FVC for upcoming weeks
    predicted_fvc = []
    for week in upcoming_weeks:
        FVC_est = trace['alpha'][:, patient_index] + trace['beta'][:, patient_index] * week
        FVC_est += trace['beta_int'] * current_week  # Adjust FVC for baseline week
        FVC_est += trace['alpha_s'] + trace['beta_cs'][:, patient_class] * age  # Adjust FVC for age and class
        predicted_fvc.append(FVC_est.mean())  # Taking mean FVC from posterior samples
    
    # Create a DataFrame for results
    results_df = pd.DataFrame({
        'Week': upcoming_weeks,
        'Predicted_FVC': predicted_fvc
    })
    
    return results_df

# Example usage:
patient_id = 'ID00421637202311550012437'
age = 50
sex = 'Male'
smoking_status = 'Ex-smoker'
current_week = 50
upcoming_weeks = [51, 52, 53]  # Weeks for which we want to predict FVC

# Call the function to predict FVC for upcoming weeks
prediction_results = predict_fvc_for_patient(patient_id, age, sex, smoking_status, current_week, upcoming_weeks)

# Display the prediction results
if prediction_results is not None:
    print(prediction_results)




print(test.columns.tolist())
print (train.columns.tolist())
print(train.head())
print (test.head())


from sklearn import preprocessing
from sklearn.metrics import r2_score
import numpy as np

# --- 0ï¸�âƒ£ Encode PatientID in test (same as train) ---
le = preprocessing.LabelEncoder()
le.fit(train['Patient'])
test['PatientID'] = le.transform(test['Patient'])

# --- 1ï¸�âƒ£ Observed FVC ---
y_true = test['FVC_base'].values

# --- 2ï¸�âƒ£ Prepare indices ---
patient_indices = test['PatientID'].values
patient_classes = test['Class'].values
weeks = test['Weeks_base'].values  # use Weeks_base if you only have that
ages = test['Age'].values

# --- 3ï¸�âƒ£ Posterior means from PyMC trace ---
alpha_mean = trace['alpha'].mean(axis=0)
beta_mean = trace['beta'].mean(axis=0)
beta_int_mean = trace['beta_int'].mean()
alpha_s_mean = trace['alpha_s'].mean()
beta_cs_mean = trace['beta_cs'].mean(axis=0)

# --- 4ï¸�âƒ£ Predicted FVC ---
y_pred = (
    alpha_mean[patient_indices] +
    beta_mean[patient_indices] * weeks +
    beta_int_mean +
    alpha_s_mean +
    beta_cs_mean[patient_classes] * ages
)

# --- 5ï¸�âƒ£ Compute RÂ² ---
r2 = r2_score(y_true, y_pred)
print("RÂ² score:", r2)



# -------------------------------
# 0ï¸�âƒ£ Imports
# -------------------------------
import pandas as pd
import numpy as np
import pymc3 as pm
import arviz as az
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# -------------------------------
# 1ï¸�âƒ£ Load data
# -------------------------------
train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

# -------------------------------
# 2ï¸�âƒ£ Preprocessing
# -------------------------------
# Add baseline week and FVC per patient


train['Weeks_base'] = train.groupby('Patient')['Weeks'].transform('min')
train['FVC_base'] = train.groupby('Patient')['FVC'].transform('first')

# For test, create Weeks_base and FVC_base if missing


if 'Weeks_base' not in test.columns:
    test['Weeks_base'] = test.groupby('Patient')['Weeks'].transform('min')
if 'FVC_base' not in test.columns:
    test['FVC_base'] = test.groupby('Patient')['FVC'].transform('first')

# Encode PatientID
le = LabelEncoder()
train['PatientID'] = le.fit_transform(train['Patient'])
test['PatientID'] = le.transform(test['Patient'])

# Define patient class
def calculate_patient_class(sex, smoking_status):
    if sex == 'Male':
        return {'Currently smokes':0,'Ex-smoker':1,'Never smoked':2}[smoking_status]
    else:
        return {'Currently smokes':3,'Ex-smoker':4,'Never smoked':5}[smoking_status]

train['Class'] = train.apply(lambda row: calculate_patient_class(row['Sex'], row['SmokingStatus']), axis=1)
test['Class'] = test.apply(lambda row: calculate_patient_class(row['Sex'], row['SmokingStatus']), axis=1)

# -------------------------------
# 3ï¸�âƒ£ Map unique patients to indices
# -------------------------------
unique_patients = train[['PatientID', 'FVC_base', 'Weeks_base', 'Age', 'Class']].drop_duplicates().sort_values('PatientID')
patient_idx_map = dict(zip(unique_patients['PatientID'], range(len(unique_patients))))

train['pid_idx'] = train['PatientID'].map(patient_idx_map)
test['pid_idx'] = test['PatientID'].map(patient_idx_map)

# -------------------------------
# 4ï¸�âƒ£ Prepare patient-level data
# -------------------------------

FVC_base_pat = unique_patients['FVC_base'].values
Weeks_base_pat = unique_patients['Weeks_base'].values
Age_pat = unique_patients['Age'].values
Class_pat = unique_patients['Class'].values

# -------------------------------
# 5ï¸�âƒ£ Build PyMC3 Hierarchical Model
# -------------------------------
with pm.Model() as hierarchical_model:
    # Hyperpriors for alpha
    beta_int = pm.Normal('beta_int', 0, sigma=100)
    sigma_alpha = pm.HalfNormal('sigma_alpha', 100)
    alpha = pm.Normal('alpha', mu=FVC_base_pat, sigma=sigma_alpha, shape=len(FVC_base_pat))
    
    # Hyperpriors for beta
    sigma_beta = pm.HalfNormal('sigma_beta', 100)
    alpha_s = pm.Normal('alpha_s', 0, sigma=100)
    beta_cs = pm.Normal('beta_cs', 0, sigma=100, shape=6)
    gamma_cs = pm.Normal('gamma_cs', 0, sigma=10, shape=6)  # optional quadratic effect
    
    # Beta for each patient
    beta_mu = alpha_s + Age_pat * beta_cs[Class_pat]  # linear age effect
    beta = pm.Normal('beta', mu=beta_mu, sigma=sigma_beta, shape=len(FVC_base_pat))
    
    # Model variance
    sigma = pm.HalfNormal('sigma', 200)
    
    # FVC estimate per row in train
    FVC_est = alpha[train['pid_idx'].values] + beta[train['pid_idx'].values] * train['Weeks'].values
    FVC_like = pm.Normal('FVC_like', mu=FVC_est, sigma=sigma, observed=train['FVC'].values)
    
    # Sample from posterior
    trace = pm.sample(500, tune=500, target_accept=0.95, cores=2)

# -------------------------------
# 6ï¸�âƒ£ Predict on test set
# -------------------------------
alpha_mean = trace['alpha'].mean(axis=0)
beta_mean = trace['beta'].mean(axis=0)

y_pred = alpha_mean[test['pid_idx'].values] + beta_mean[test['pid_idx'].values] * test['Weeks_base'].values
y_true = test['FVC_base'].values

# -------------------------------
# 7ï¸�âƒ£ Compute RÂ²
# -------------------------------
r2 = r2_score(y_true, y_pred)
print("âœ… RÂ² score:", r2)

# -------------------------------
# 8ï¸�âƒ£ Save model trace
# -------------------------------
az.to_netcdf(trace, 'fvc_model_trace.nc')
print("âœ… Model saved as fvc_model_trace.nc")



# -------------------------------
# 9ï¸�âƒ£ Compute Correlation Metrics
# -------------------------------
from scipy.stats import pearsonr, spearmanr

# Pearson correlation
pearson_corr, pearson_p = pearsonr(y_true, y_pred)
print(f"âœ… Pearson correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4e})")

# Spearman correlation
spearman_corr, spearman_p = spearmanr(y_true, y_pred)
print(f"âœ… Spearman correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4e})")



# -------------------------------
# 9ï¸�âƒ£ Correlation Metrics Table
# -------------------------------
from scipy.stats import pearsonr, spearmanr
import pandas as pd

# Calculate correlations
pearson_corr, pearson_p = pearsonr(y_true, y_pred)
spearman_corr, spearman_p = spearmanr(y_true, y_pred)

# Create a results table
metrics_table = pd.DataFrame({
    'Metric': ['RÂ²', 'Pearson', 'Spearman'],
    'Value': [r2, pearson_corr, spearman_corr],
    'p-value': [None, pearson_p, spearman_p]
})

print("ğŸ“Š Correlation Metrics:")
print(metrics_table)



# -------------------------------
# Correlation Matrix Heatmap
# -------------------------------
import seaborn as sns
import matplotlib.pyplot as plt

# Select numeric features
features = ['Weeks', 'FVC', 'Percent', 'Age']

# Compute correlation matrix
corr_matrix = train[features].corr()

# Plot heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, linewidths=1, linecolor='white')
plt.title("Correlation Matrix of Features", fontsize=14)
plt.show()



# -------------------------------
# 0ï¸�âƒ£ Imports
# -------------------------------
import pandas as pd
import numpy as np
import pymc3 as pm
import arviz as az
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# -------------------------------
# 1ï¸�âƒ£ Load data
# -------------------------------
train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

# -------------------------------
# 2ï¸�âƒ£ Preprocessing
# -------------------------------
# Add baseline week and FVC per patient
train['Weeks_base'] = train.groupby('Patient')['Weeks'].transform('min')
train['FVC_base'] = train.groupby('Patient')['FVC'].transform('first')

# For test, create Weeks_base and FVC_base if missing
if 'Weeks_base' not in test.columns:
    test['Weeks_base'] = test.groupby('Patient')['Weeks'].transform('min')
if 'FVC_base' not in test.columns:
    test['FVC_base'] = test.groupby('Patient')['FVC'].transform('first')

# Encode PatientID
le = LabelEncoder()
train['PatientID'] = le.fit_transform(train['Patient'])
test['PatientID'] = le.transform(test['Patient'])

# Define patient class
def calculate_patient_class(sex, smoking_status):
    if sex == 'Male':
        return {'Currently smokes':0,'Ex-smoker':1,'Never smoked':2}[smoking_status]
    else:
        return {'Currently smokes':3,'Ex-smoker':4,'Never smoked':5}[smoking_status]

train['Class'] = train.apply(lambda row: calculate_patient_class(row['Sex'], row['SmokingStatus']), axis=1)
test['Class'] = test.apply(lambda row: calculate_patient_class(row['Sex'], row['SmokingStatus']), axis=1)

# -------------------------------
# 3ï¸�âƒ£ Map unique patients to indices
# -------------------------------
unique_patients = train[['PatientID', 'FVC_base', 'Weeks_base', 'Age', 'Class']].drop_duplicates().sort_values('PatientID')
patient_idx_map = dict(zip(unique_patients['PatientID'], range(len(unique_patients))))

train['pid_idx'] = train['PatientID'].map(patient_idx_map)
test['pid_idx'] = test['PatientID'].map(patient_idx_map)

# -------------------------------
# 4ï¸�âƒ£ Prepare patient-level data
# -------------------------------
FVC_base_pat = unique_patients['FVC_base'].values
Weeks_base_pat = unique_patients['Weeks_base'].values
Age_pat = unique_patients['Age'].values
Class_pat = unique_patients['Class'].values

# -------------------------------
# 5ï¸�âƒ£ PyMC3 Hierarchical Model with nonlinear effects
# -------------------------------
with pm.Model() as hierarchical_model:
    # Hyperpriors for alpha (patient-specific intercept)
    sigma_alpha = pm.HalfNormal('sigma_alpha', 100)
    alpha = pm.Normal('alpha', mu=FVC_base_pat, sigma=sigma_alpha, shape=len(FVC_base_pat))
    
    # Hyperpriors for beta (patient-specific slope)
    sigma_beta = pm.HalfNormal('sigma_beta', 100)
    
    # Age Ã— Class effect
    beta_cs = pm.Normal('beta_cs', 0, sigma=100, shape=6)
    
    # Weeks Ã— Class linear effect
    beta_week_cs = pm.Normal('beta_week_cs', 0, sigma=10, shape=6)
    
    # WeeksÂ² Ã— Class nonlinear effect
    gamma_week_cs = pm.Normal('gamma_week_cs', 0, sigma=5, shape=6)
    
    # Compute beta for each patient
    beta = (Age_pat * beta_cs[Class_pat] +
            Weeks_base_pat * beta_week_cs[Class_pat] +
            (Weeks_base_pat**2) * gamma_week_cs[Class_pat])
    
    # Model variance
    sigma = pm.HalfNormal('sigma', 200)
    
    # FVC estimate per training row
    FVC_est = (alpha[train['pid_idx'].values] +
               beta[train['pid_idx'].values] * train['Weeks'].values +
               gamma_week_cs[Class_pat[train['pid_idx'].values]] * (train['Weeks'].values**2))
    
    # Likelihood
    FVC_like = pm.Normal('FVC_like', mu=FVC_est, sigma=sigma, observed=train['FVC'].values)
    
    # Sample from posterior
    trace = pm.sample(500, tune=500, target_accept=0.95, cores=2)

# -------------------------------
# 6ï¸�âƒ£ Predict on test set
# -------------------------------
alpha_mean = trace['alpha'].mean(axis=0)
beta_cs_mean = trace['beta_cs'].mean(axis=0)
beta_week_cs_mean = trace['beta_week_cs'].mean(axis=0)
gamma_week_cs_mean = trace['gamma_week_cs'].mean(axis=0)

y_pred = (
    alpha_mean[test['pid_idx'].values] +
    (test['Age'].values * beta_cs_mean[test['Class'].values]) +
    (test['Weeks_base'].values * beta_week_cs_mean[test['Class'].values]) +
    ((test['Weeks_base'].values**2) * gamma_week_cs_mean[test['Class'].values])
)
y_true = test['FVC_base'].values

# -------------------------------
# 7ï¸�âƒ£ Compute RÂ²
# -------------------------------
r2 = r2_score(y_true, y_pred)
print("âœ… RÂ² score:", r2)

# -------------------------------
# 8ï¸�âƒ£ Save model trace
# -------------------------------
az.to_netcdf(trace, 'fvc_model_trace1.nc')
print("âœ… Model saved as fvc_model_trace1.nc")


