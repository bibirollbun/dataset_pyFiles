!pip install -q lifelines
!pip install -q esda
!pip install -q semopy
!pip install -q cliffs-delta
!pip install --upgrade -q libpysal esda

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Load data 
df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
print("Import Success")


# Initial exploration
print(df.info())
print(df.describe())


for col in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
    if df[col].notna().sum() > 0:
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()
        
        # Shapiro-Wilk normality test
        stat, p = stats.shapiro(df[col].dropna())
        print(f'{col}: Shapiro-Wilk p-value = {p:.4f}')


df['smiles_length'] = df['SMILES'].apply(len)
median_length = df['smiles_length'].median()
group1 = df[df['smiles_length'] > median_length]['FFV'].dropna()
group2 = df[df['smiles_length'] <= median_length]['FFV'].dropna()

# T-test
t_stat, p_val = stats.ttest_ind(group1, group2)
print(f"Independent t-test: t = {t_stat:.3f}, p = {p_val:.4f}")


# Mann-Whitney U test (non-parametric alternative)
u_stat, p_val = stats.mannwhitneyu(group1, group2)
print(f"Mann-Whitney U test: U = {u_stat:.3f}, p = {p_val:.4f}")


corr_matrix = df[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].corr(method='pearson')

# Spearman correlation (non-parametric)
spearman_matrix = df[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].corr(method='spearman')

# Test significance of correlations
# Pearson correlation with proper NaN handling
for col1 in corr_matrix.columns:
    for col2 in corr_matrix.columns:
        if col1 != col2:
            # Drop rows where either column has NaN
            valid_rows = df[[col1, col2]].dropna()
            if len(valid_rows) > 30:  # Ensure sufficient sample size
                r, p = stats.pearsonr(valid_rows[col1], valid_rows[col2])
                print(f"Pearson {col1} vs {col2}: r = {r:.3f}, p = {p:.4f}, n = {len(valid_rows)}")


from itertools import combinations

def calculate_correlations(df, cols, min_sample=30):
    results = []
    for col1, col2 in combinations(cols, 2):
        valid_rows = df[[col1, col2]].dropna()
        n = len(valid_rows)
        
        if n >= min_sample:
            # Pearson
            r_pearson, p_pearson = stats.pearsonr(valid_rows[col1], valid_rows[col2])
            
            # Spearman
            r_spearman, p_spearman = stats.spearmanr(valid_rows[col1], valid_rows[col2])
            
            # Confidence intervals
            ci_pearson = pearson_ci(r_pearson, n)
            ci_spearman = spearman_ci(r_spearman, n)
            
            results.append({
                'Variable 1': col1,
                'Variable 2': col2,
                'Pearson r': r_pearson,
                'Pearson p': p_pearson,
                'Pearson CI lower': ci_pearson[0],
                'Pearson CI upper': ci_pearson[1],
                'Spearman Ï�': r_spearman,
                'Spearman p': p_spearman,
                'Spearman CI lower': ci_spearman[0],
                'Spearman CI upper': ci_spearman[1],
                'Sample Size': n
            })
    
    return pd.DataFrame(results)

def pearson_ci(r, n, confidence=0.95):
    """Calculate confidence interval for Pearson correlation"""
    z = np.arctanh(r)
    se = 1/np.sqrt(n-3)
    z_crit = stats.norm.ppf(1-(1-confidence)/2)
    lo_z, hi_z = z-z_crit*se, z+z_crit*se
    return np.tanh((lo_z, hi_z))

def spearman_ci(rho, n, confidence=0.95):
    """Approximate CI for Spearman correlation"""
    # Using Fisher transformation approximation
    return pearson_ci(rho, n, confidence)

# Usage
cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
correlation_results = calculate_correlations(df, cols)
pd.set_option('display.max_columns', None)
print(correlation_results.sort_values('Pearson p'))


# Example: Test if proportion of high FFV polymers differs between groups
threshold = df['FFV'].median()
group1_high = (group1 > threshold).mean()
group2_high = (group2 > threshold).mean()

# Two-proportion z-test
from statsmodels.stats.proportion import proportions_ztest
count = [sum(group1 > threshold), sum(group2 > threshold)]
nobs = [len(group1), len(group2)]
z_stat, p_val = proportions_ztest(count, nobs)
print(f"Proportion z-test: z = {z_stat:.3f}, p = {p_val:.4f}")


# Correlation between SMILES length and properties
for prop in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
    if df[prop].notna().sum() > 30:
        r, p = stats.spearmanr(df['smiles_length'], df[prop])
        print(f"SMILES length vs {prop}: Ï� = {r:.3f}, p = {p:.4f}")


df['has_ester'] = df['SMILES'].str.contains('COC=O')
group1 = df[df['has_ester']]['FFV'].dropna()
group2 = df[~df['has_ester']]['FFV'].dropna()

# Welch's t-test (unequal variances)
t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
print(f"Ester group FFV comparison: t = {t_stat:.3f}, p = {p_val:.4f}")


# Calculate required sample size for future experiments
from statsmodels.stats.power import TTestIndPower

effect_size = 0.5  # Medium effect
alpha = 0.05
power = 0.8

analysis = TTestIndPower()
sample_size = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha)
print(f"Required sample size per group: {sample_size:.1f}")


# Bootstrap median FFV
n_bootstraps = 1000
boot_medians = []
for _ in range(n_bootstraps):
    sample = df['FFV'].dropna().sample(frac=1, replace=True)
    boot_medians.append(sample.median())

ci_lower = np.percentile(boot_medians, 2.5)
ci_upper = np.percentile(boot_medians, 97.5)
print(f"Bootstrap 95% CI for median FFV: ({ci_lower:.3f}, {ci_upper:.3f})")


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Prepare numeric data (example with FFV and Tc)
numeric_df = df[['FFV', 'Tc']].dropna()
X = StandardScaler().fit_transform(numeric_df)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)

# Scree plot
plt.plot(pca.explained_variance_ratio_)
plt.title('Scree Plot')
plt.ylabel('Explained Variance Ratio')
plt.show()

# Biplot
sns.scatterplot(x=principal_components[:,0], y=principal_components[:,1])
for i, feature in enumerate(numeric_df.columns):
    plt.arrow(0, 0, pca.components_[0,i], pca.components_[1,i], color='r')
    plt.text(pca.components_[0,i], pca.components_[1,i], feature, color='r')
plt.title('PCA Biplot')
plt.show()


import pymc as pm  # PyMC v5 (successor to PyMC3)
import numpy as np
import pandas as pd
import arviz as az

# Prepare data
numeric_df = df[['FFV', 'Tc']].dropna()
centered_data = numeric_df.values - numeric_df.mean().values

with pm.Model():
    # Priors
    mu = pm.Normal('mu', mu=0, sigma=1, shape=2)
    sigma = pm.HalfNormal('sigma', sigma=1, shape=2)
    rho = pm.Uniform('rho', -1, 1)
    
    # Covariance matrix
    cov = pm.Deterministic('cov', pm.math.stack([[sigma[0]**2, sigma[0]*sigma[1]*rho],
                                               [sigma[0]*sigma[1]*rho, sigma[1]**2]]))
    
    # Likelihood
    mv_normal = pm.MvNormal('mv_normal', mu=mu, cov=cov, 
                          observed=centered_data)
    
    trace = pm.sample(2000, tune=1000)
    
az.plot_posterior(trace, var_names=['rho'])


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# If you had time-series experimental data
plot_acf(df['FFV'].dropna(), lags=20)
plt.title('Autocorrelation Function')
plt.show()

plot_pacf(df['FFV'].dropna(), lags=20)
plt.title('Partial Autocorrelation Function')
plt.show()


# Compare distributions of two groups
stat, p = stats.ks_2samp(group1, group2)
print(f"KS Test: D = {stat:.3f}, p = {p:.4f}")


import numpy as np
from scipy import stats

def cohens_d(x, y):
    """
    Calculate Cohen's d effect size
    Args:
        x: array-like - First group's data
        y: array-like - Second group's data
    Returns:
        d: float - Cohen's d effect size
    """
    # Ensure inputs are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Remove any remaining NaN values
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    
    # Calculate pooled standard deviation
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)
    
    # Calculate Cohen's d
    return (np.mean(x) - np.mean(y)) / pooled_std

def cliffs_delta(x, y):
    """
    Calculate Cliff's delta effect size (non-parametric)
    Args:
        x: array-like - First group's data
        y: array-like - Second group's data
    Returns:
        delta: float - Cliff's delta effect size (-1 to 1)
        magnitude: str - Interpretation string
    """
    # Ensure inputs are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Remove any remaining NaN values
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    
    # Implementation based on original 1993 paper
    nx, ny = len(x), len(y)
    pairwise_diffs = np.sign(np.subtract.outer(x, y))
    delta = np.sum(pairwise_diffs) / (nx * ny)
    
    # Interpretation
    magnitude = "negligible"
    if abs(delta) >= 0.147:
        magnitude = "small"
    if abs(delta) >= 0.33:
        magnitude = "medium"
    if abs(delta) >= 0.474:
        magnitude = "large"
    
    return delta, magnitude

# Example usage:
group1 = df[df['has_ester'] == 1]['FFV'].dropna().values
group2 = df[df['has_ester'] == 0]['FFV'].dropna().values

# Cohen's d
d = cohens_d(group1, group2)
print(f"Cohen's d: {d:.3f} ({'small' if abs(d)<0.5 else 'medium' if abs(d)<0.8 else 'large'} effect)")

# Cliff's delta
delta, magnitude = cliffs_delta(group1, group2)
print(f"Cliff's delta: {delta:.3f} ({magnitude} effect)")


from statsmodels.stats.multitest import multipletests

p_values = [0.01, 0.04, 0.03, 0.20, 0.001]
rejected, corrected_p, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

print("Original p-values:", p_values)
print("BH-corrected p-values:", corrected_p)
print("Rejected:", rejected)


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X = pd.get_dummies(df[['smiles_length', 'has_ester']])  # Example features
y = df['FFV'].dropna()
X = X.loc[y.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# Permutation importance
from sklearn.inspection import permutation_importance
result = permutation_importance(model, X_test, y_test, n_repeats=30)

for i in result.importances_mean.argsort()[::-1]:
    print(f"{X.columns[i]:<8} "
          f"{result.importances_mean[i]:.3f} Â± {result.importances_std[i]:.3f}")


import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter

# Create synthetic stability data based on FFV thresholds
# Assuming higher FFV correlates with better stability
threshold = df['FFV'].median()
df['stable'] = (df['FFV'] > threshold).astype(int)

# Create synthetic time data (e.g., experimental observation periods)
np.random.seed(42)
df['observation_time'] = np.random.uniform(1, 100, size=len(df))

# For polymers with FFV below threshold, assume some failed during observation
df['failed'] = 0
df.loc[(df['FFV'] <= threshold) & 
       (df['observation_time'] > np.random.uniform(0, 50, size=len(df))), 
       'failed'] = 1

# Survival analysis
kmf = KaplanMeierFitter()
kmf.fit(df['observation_time'], event_observed=df['failed'], label='All Polymers')

plt.figure(figsize=(10, 6))
kmf.plot_survival_function()
plt.title('Polymer Stability by FFV Threshold')
plt.ylabel('Probability of Remaining Stable')
plt.xlabel('Observation Time (arbitrary units)')
plt.grid(True)

# Add threshold comparison
for value, label in [(0, 'Low FFV'), (1, 'High FFV')]:
    kmf.fit(df[df['stable'] == value]['observation_time'], 
            event_observed=df[df['stable'] == value]['failed'], 
            label=label)
    kmf.plot_survival_function()

plt.legend()
plt.show()


stability_corr = df[['FFV', 'Tg', 'Tc', 'Density']].corr(method='spearman')
print("Property Correlations with Stability Proxy (FFV):")
print(stability_corr['FFV'].sort_values(ascending=False))

# Stability classification model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X = df[['Tg', 'Tc', 'Density', 'smiles_length']].dropna()
y = (df.loc[X.index, 'FFV'] > df['FFV'].median()).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

print(f"\nStability Classifier Accuracy: {clf.score(X_test, y_test):.2f}")


import numpy as np
import libpysal
from esda.moran import Moran

# 1. Prepare your data - ensure no NaN values in coordinates
analysis_df = df[['smiles_length', 'FFV', 'Tc']].dropna()

# 2. Create normalized coordinates
coordinates = np.column_stack([
    analysis_df['smiles_length'].rank(pct=True).values,  # X coordinate
    analysis_df['FFV'].rank(pct=True).values             # Y coordinate
])

# 3. Create weights matrix with proper data handling
try:
    # Convert coordinates to float64 if needed
    coordinates = coordinates.astype(np.float64)
    
    # Create weights matrix
    w = libpysal.weights.DistanceBand.from_array(
        coordinates, 
        threshold=0.3,  # Adjust this based on your data distribution
        binary=True,     # Simpler binary weights
        silence_warnings=True
    )
    
    # 4. Calculate Moran's I
    moran = Moran(analysis_df['Tc'].values, w)
    print(f"Moran's I: {moran.I:.3f}")
    print(f"P-value: {moran.p_sim:.4f}")
    print(f"Expected I: {moran.EI:.3f}")
    
    # 5. Interpretation
    if moran.p_sim < 0.05:
        if moran.I > moran.EI:
            print("Significant positive spatial autocorrelation")
        else:
            print("Significant negative spatial autocorrelation")
    else:
        print("No significant spatial autocorrelation")

except ValueError as e:
    print(f"Error creating weights matrix: {e}")
    print("Possible solutions:")
    print("1. Check for NaN/infinite values in coordinates")
    print("2. Try adjusting the threshold parameter")
    print("3. Normalize your coordinates differently")

