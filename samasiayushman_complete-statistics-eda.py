!pip install -q scipy statsmodels
!pip install -q scipy pingouin 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pingouin as pg

# Load the data
train = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/train.csv')
star_info = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/train_star_info.csv')
merged_data = pd.merge(train, star_info, on='planet_id')

# Create wavelength summary statistics
wl_columns = [col for col in train.columns if col.startswith('wl_')]


desc_stats = merged_data[['Rs', 'Ms', 'Ts', 'Mp', 'P', 'sma', 'i']].describe().T
desc_stats['skewness'] = merged_data[['Rs', 'Ms', 'Ts', 'Mp', 'P', 'sma', 'i']].skew()
desc_stats['kurtosis'] = merged_data[['Rs', 'Ms', 'Ts', 'Mp', 'P', 'sma', 'i']].kurt()
display(desc_stats)


wl_stats = train[wl_columns].agg(['mean', 'std', 'min', 'max', 'skew', 'kurtosis']).T
wl_stats['cv'] = wl_stats['std'] / wl_stats['mean']  # Coefficient of variation

plt.figure(figsize=(12, 6))
plt.plot(wl_stats.index, wl_stats['mean'], label='Mean')
plt.fill_between(wl_stats.index, 
                wl_stats['mean'] - wl_stats['std'], 
                wl_stats['mean'] + wl_stats['std'], 
                alpha=0.2, label='Â±1 Std Dev')
plt.title('Mean Wavelength Flux with Standard Deviation')
plt.xlabel('Wavelength Index')
plt.ylabel('Normalized Flux')
plt.legend()
plt.show()


plt.figure(figsize=(10, 8))
corr_matrix = merged_data[['Rs', 'Ms', 'Ts', 'Mp', 'P', 'sma', 'i']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of System Parameters')
plt.show()


def get_ci(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    h = std_err * stats.t.ppf((1 + confidence) / 2, n-1)
    return mean, mean-h, mean+h

params = ['Ts', 'Mp', 'P']
ci_results = {}

for param in params:
    ci_results[param] = get_ci(merged_data[param])
    
ci_df = pd.DataFrame(ci_results, index=['mean', 'lower_ci', 'upper_ci']).T
display(ci_df)


# Create temperature groups
merged_data['temp_group'] = pd.qcut(merged_data['Ts'], q=3, labels=['Cool', 'Medium', 'Hot'])

# Compare planet masses
hot_cool = merged_data[merged_data['temp_group'].isin(['Hot', 'Cool'])]
print(stats.ttest_ind(
    hot_cool[hot_cool['temp_group'] == 'Hot']['Mp'],
    hot_cool[hot_cool['temp_group'] == 'Cool']['Mp'],
    equal_var=False
))

# Visual comparison
plt.figure(figsize=(10, 6))
sns.boxplot(x='temp_group', y='Mp', data=hot_cool)
plt.title('Planet Mass Distribution by Star Temperature Group')
plt.ylabel('Planet Mass (Mj)')
plt.xlabel('Star Temperature Group')
plt.show()


normality_results = {}
for param in ['Ts', 'Mp', 'P', 'sma']:
    stat, p = stats.shapiro(merged_data[param])
    normality_results[param] = {'statistic': stat, 'p-value': p}
    
normality_df = pd.DataFrame(normality_results).T
display(normality_df)


# One-way ANOVA for planet mass across temperature groups
anova_result = pg.anova(data=merged_data, dv='Mp', between='temp_group', detailed=True)
display(anova_result)

# Post-hoc test if ANOVA is significant
if anova_result['p-unc'][0] < 0.05:
    print("\nPost-hoc Tukey HSD:")
    tukey = pairwise_tukeyhsd(endog=merged_data['Mp'],
                             groups=merged_data['temp_group'],
                             alpha=0.05)
    print(tukey)


# Kruskal-Wallis test (non-parametric alternative to ANOVA)
print("Kruskal-Wallis Test:")
print(stats.kruskal(
    merged_data[merged_data['temp_group'] == 'Cool']['Mp'],
    merged_data[merged_data['temp_group'] == 'Medium']['Mp'],
    merged_data[merged_data['temp_group'] == 'Hot']['Mp']
))

# Mann-Whitney U test for two groups
print("\nMann-Whitney U Test (Hot vs Cool):")
print(stats.mannwhitneyu(
    merged_data[merged_data['temp_group'] == 'Hot']['Mp'],
    merged_data[merged_data['temp_group'] == 'Cool']['Mp']
))


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize wavelength data
scaler = StandardScaler()
wl_scaled = scaler.fit_transform(train[wl_columns])

# Perform PCA
pca = PCA(n_components=3)
pca_results = pca.fit_transform(wl_scaled)

# Plot explained variance
plt.figure(figsize=(10, 5))
plt.bar(range(3), pca.explained_variance_ratio_)
plt.title('PCA Explained Variance Ratio')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()

# Plot PC1 vs PC2
plt.figure(figsize=(10, 8))
sc = plt.scatter(pca_results[:, 0], pca_results[:, 1], 
                c=merged_data['Ts'], cmap='viridis')
plt.colorbar(sc, label='Star Temperature (K)')
plt.title('PCA of Wavelength Data (PC1 vs PC2)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Prepare data
X = merged_data[['Rs', 'Ms', 'Ts']]
y = merged_data['Mp']

# Fit model
model = LinearRegression().fit(X, y)

# Print results
print(f"Intercept: {model.intercept_:.4f}")
print("Coefficients:")
for name, coef in zip(X.columns, model.coef_):
    print(f"{name}: {coef:.4f}")

# Calculate R-squared
r2 = r2_score(y, model.predict(X))
print(f"\nR-squared: {r2:.4f}")

# Plot results
plt.figure(figsize=(8, 8))
plt.scatter(y, model.predict(X))
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Planet Mass')
plt.ylabel('Predicted Planet Mass')
plt.title('Linear Regression (scikit-learn)')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.signal import correlate

def custom_acf(x, nlags=40):
    """Manual implementation of autocorrelation function"""
    x = np.asarray(x) - np.mean(x)
    corr = correlate(x, x, mode='full')[-len(x):-len(x)+nlags+1]
    return corr / corr[0]

def custom_pacf(x, nlags=40):
    """Manual implementation of partial autocorrelation using Yule-Walker"""
    x = np.asarray(x) - np.mean(x)
    pacf = [1.0]
    for k in range(1, nlags+1):
        r = custom_acf(x, nlags=k)
        # Solve Yule-Walker equations
        R = np.array([r[:k]])
        r_k = r[1:k+1]
        phi = np.linalg.solve(np.toeplitz(R), r_k)
        pacf.append(phi[-1])
    return np.array(pacf)

def robust_wavelength_analysis(planet_index=0):
    """Completely self-contained time series analysis"""
    try:
        # Get sample planet data
        sample_planet = train.iloc[planet_index][wl_columns].values
        planet_id = train.iloc[planet_index]["planet_id"]
        
        # Basic statistics
        print(f"\nAnalysis for Planet {planet_id}")
        print(f"Mean flux: {np.mean(sample_planet):.4f}")
        print(f"Std dev: {np.std(sample_planet):.4f}")
        
        # Manual ACF/PACF
        nlags = min(40, len(sample_planet)//3)  # Conservative lag value
        acf_vals = custom_acf(sample_planet, nlags=nlags)
        pacf_vals = custom_pacf(sample_planet, nlags=nlags)
        
        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # ACF plot
        ax1.stem(acf_vals)
        conf = norm.ppf(1 - 0.05/2) / np.sqrt(len(sample_planet))
        ax1.axhspan(-conf, conf, alpha=0.2, color='blue')
        ax1.set_title(f'Autocorrelation (Planet {planet_id})')
        
        # PACF plot
        ax2.stem(pacf_vals)
        ax2.axhspan(-conf, conf, alpha=0.2, color='blue')
        ax2.set_title(f'Partial Autocorrelation (Planet {planet_id})')
        
        plt.tight_layout()
        plt.show()
        
        # Basic stationarity check
        diff = np.diff(sample_planet)
        print("\nStationarity indicators:")
        print(f"Original variance: {np.var(sample_planet):.4f}")
        print(f"1st difference variance: {np.var(diff):.4f}")
        
    except Exception as e:
        print(f"Error analyzing planet {planet_index}: {str(e)}")

# Run analysis on first 3 planets
for i in range(3):
    robust_wavelength_analysis(i)


def spectral_analysis(planet_index=0):
    """Basic periodogram analysis"""
    from scipy.signal import periodogram
    
    sample_planet = train.iloc[planet_index][wl_columns].values
    freqs, psd = periodogram(sample_planet)
    
    plt.figure(figsize=(12,4))
    plt.semilogy(freqs, psd)
    plt.title('Power Spectral Density')
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.show()

spectral_analysis(0)


!pip install -q scikit-fda

from skfda import FDataGrid
from skfda.preprocessing.dim_reduction import FPCA

# Convert wavelength data to functional objects
fd = FDataGrid(
    data_matrix=train[wl_columns].values,
    grid_points=np.arange(len(wl_columns))
)

# Functional PCA
fpca = FPCA(n_components=3)
fpca.fit(fd)
scores = fpca.transform(fd)


fig = plt.figure(figsize=(12, 6))
for i in range(3):
    fd.mean().plot(label='Mean')
    (fd.mean() + fpca.components_[i]*np.sqrt(fpca.explained_variance_[i])).plot(
        label=f'FPC {i+1}')
    plt.title(f'Functional Principal Component {i+1}')
    plt.legend()
    plt.show()


import networkx as nx
from community import community_louvain

# Create similarity network
corr_matrix = train[wl_columns].T.corr()
G = nx.Graph()

threshold = 0.9
for i in range(len(corr_matrix)):
    for j in range(i+1, len(corr_matrix)):
        if corr_matrix.iloc[i,j] > threshold:
            G.add_edge(i, j, weight=corr_matrix.iloc[i,j])

# Community detection
partition = community_louvain.best_partition(G)

# Visualize
pos = nx.spring_layout(G)
plt.figure(figsize=(12, 12))
nx.draw_networkx_nodes(G, pos, node_size=50, 
                      cmap=plt.cm.viridis,
                      node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.1)
plt.title('Planetary System Similarity Network')
plt.show()

