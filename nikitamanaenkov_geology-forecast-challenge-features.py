import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import kstest
from scipy.fft import fft
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", None)


df = pd.read_csv('/kaggle/input/geology-forecast-challenge-open/data/train.csv')


input_cols = [str(i) for i in range(-299, 1)] 
target_cols = [str(i) for i in range(1, 301)]  

imputer_X = SimpleImputer(strategy="mean")
X_raw_s = imputer_X.fit_transform(df[input_cols])

imputer_y = SimpleImputer(strategy="mean")
y_raw_s = imputer_y.fit_transform(df[target_cols])


y_raw = pd.DataFrame(y_raw_s, columns=target_cols)
X_raw = pd.DataFrame(X_raw_s, columns=input_cols)
y = y_raw.mean(axis=1)
X_filled = X_raw.T.interpolate(limit_direction='both').T.fillna(0)

X_np = X_filled.to_numpy()

feature_df = pd.DataFrame(index=df.index)
feature_df['mean'] = X_filled.mean(axis=1)
feature_df['std'] = X_filled.std(axis=1)
feature_df['min'] = X_filled.min(axis=1)
feature_df['max'] = X_filled.max(axis=1)
feature_df['median'] = X_filled.median(axis=1)
feature_df['range'] = feature_df['max'] - feature_df['min']
feature_df['missing_ratio'] = X_raw.isna().sum(axis=1) / X_raw.shape[1]

positions = np.arange(-299, 1).reshape(-1, 1)

slopes = []
for row in X_raw_s:
    lr = LinearRegression().fit(positions, row.reshape(-1, 1))
    slopes.append(lr.coef_[0][0])  

feature_df["slope_lr"] = slopes

first_deriv = np.diff(X_np, n=1, axis=1)
feature_df["deriv_mean"] = first_deriv.mean(axis=1)
feature_df["deriv_std"] = first_deriv.std(axis=1)

second_deriv = np.diff(X_np, n=2, axis=1)
feature_df["second_deriv_mean"] = second_deriv.mean(axis=1)
feature_df["second_deriv_std"] = second_deriv.std(axis=1)

fft_coeffs = np.abs(np.fft.rfft(X_np, axis=1))
fft_top5 = np.sort(fft_coeffs, axis=1)[:, -5:]
for i in range(5):
    feature_df[f"fft_top_{i+1}"] = fft_top5[:, i]

wavelet_feats = []
for row in X_np:
    coeffs = pywt.wavedec(row, 'db1', level=3)
    flat = np.hstack(coeffs)
    wavelet_feats.append([
        np.mean(flat),
        np.std(flat),
        np.max(flat),
        np.min(flat)
    ])
wavelet_feats = np.array(wavelet_feats)
feature_df["wavelet_mean"] = wavelet_feats[:, 0]
feature_df["wavelet_std"] = wavelet_feats[:, 1]
feature_df["wavelet_max"] = wavelet_feats[:, 2]
feature_df["wavelet_min"] = wavelet_feats[:, 3]

feature_df['skew'] = X_filled.skew(axis=1)
feature_df['kurtosis'] = X_filled.kurtosis(axis=1)

feature_df['slope_mean'] = np.gradient(X_filled, axis=1).mean(axis=1)
feature_df['slope_std'] = np.gradient(X_filled, axis=1).std(axis=1)

feature_df['sign_changes'] = (np.diff(np.sign(X_filled.values), axis=1) != 0).sum(axis=1)

feature_df['zero_frac'] = (X_filled == 0).mean(axis=1)
feature_df['neg_frac'] = (X_filled < 0).mean(axis=1)

weights = np.tile(np.arange(X_filled.shape[1]), (X_filled.shape[0], 1))
feature_df['center_of_mass'] = (X_filled.values * weights).sum(axis=1) / (X_filled.sum(axis=1) + 1e-6)

def low_freq_energy(row, k=10):
    spectrum = np.abs(fft(row))
    return np.sum(spectrum[:k]) / (np.sum(spectrum) + 1e-6)

feature_df['fft_low_energy'] = X_filled.apply(low_freq_energy, axis=1)

feature_df['coef_var'] = feature_df['std'] / (feature_df['mean'] + 1e-6)

feature_df['auc'] = np.trapz(X_np, axis=1)

first_deriv = np.diff(X_np, axis=1)
ratio = np.abs(first_deriv[:, 1:] / (first_deriv[:, :-1] + 1e-6))
feature_df['change_ratio'] = ratio.mean(axis=1)

sym_diff = np.abs(X_np - X_np[:, ::-1])
feature_df['symmetry'] = sym_diff.mean(axis=1)

ks_stats = []
for row in X_np:
    m, s = row.mean(), row.std()
    stat = kstest((row - m) / (s + 1e-6), 'norm').statistic
    ks_stats.append(stat)
feature_df['ks_gauss'] = ks_stats

local_max = []
local_min = []
for row in X_np:
    maxima = np.sum((row[1:-1] > row[:-2]) & (row[1:-1] > row[2:]))
    minima = np.sum((row[1:-1] < row[:-2]) & (row[1:-1] < row[2:]))
    local_max.append(maxima)
    local_min.append(minima)
feature_df['local_max'] = local_max
feature_df['local_min'] = local_min

kmeans_inertia = []
kmeans_center_range = []
for row in X_np:
    km = KMeans(n_clusters=3, random_state=42).fit(row.reshape(-1, 1))
    kmeans_inertia.append(km.inertia_)
    centers = km.cluster_centers_.flatten()
    kmeans_center_range.append(centers.max() - centers.min())
feature_df['kmeans_inertia'] = kmeans_inertia
feature_df['kmeans_center_range'] = kmeans_center_range

pca = PCA(n_components=2)
pca_scores = pca.fit_transform(X_filled)
feature_df['pca1'] = pca_scores[:, 0]
feature_df['pca2'] = pca_scores[:, 1]
feature_df['variance_ratio_pc1'] = pca.explained_variance_ratio_[0]

poly = PolynomialFeatures(degree=2, include_bias=False)
poly_feats = poly.fit_transform(feature_df)
poly_names = poly.get_feature_names_out(feature_df.columns)
poly_df = pd.DataFrame(poly_feats, index=feature_df.index, columns=poly_names)
interaction_cols = [c for c in poly_names if ' ' in c]
feature_df = pd.concat([feature_df, poly_df[interaction_cols]], axis=1)

rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf.fit(feature_df, y)
leaf_idx = rf.apply(feature_df)
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
leaf_ohe = ohe.fit_transform(leaf_idx)
leaf_cols = [f"leaf_{i}_{cat}" for i, cats in enumerate(ohe.categories_) for cat in cats]
leaf_df = pd.DataFrame(leaf_ohe, index=feature_df.index, columns=leaf_cols)
feature_df = pd.concat([feature_df, leaf_df], axis=1)


y_raw = y_raw.dropna()

X_raw = X_raw.loc[y_raw.index]

y = y_raw.mean(axis=1)

X_scaled = StandardScaler().fit_transform(feature_df)

mi_scores = mutual_info_regression(X_scaled, y)
mi_df = pd.Series(mi_scores, index=feature_df.columns)

top_30_mi_df = mi_df.sort_values(ascending=False).head(30)

print("Top 30 Mutual Information Scores:")
print(top_30_mi_df.to_string(float_format="%.4f"))

plt.figure(figsize=(10, 8))
top_30_mi_df.plot(kind='barh', color='cornflowerblue')
plt.title("Top 30 Mutual Information: Geological Features vs Target")
plt.xlabel("Mutual Information Score")
plt.ylabel("Feature")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


top_features = top_30_mi_df.head(6).index
sns.pairplot(pd.concat([feature_df[top_features], y], axis=1), 
             diag_kind="kde", 
             corner=True,
             plot_kws={"alpha": 0.5}, 
             height=2.2)
plt.suptitle("Top 6 Features and Their Interactions (colored by target)", y=1.02)
plt.tight_layout()
plt.show()


top_feature_corr = feature_df[top_30_mi_df.index].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(top_feature_corr, annot=False, cmap='coolwarm', center=0)
plt.title("Correlation Heatmap of Top 30 Features")
plt.tight_layout()
plt.show()


pearson_corr = feature_df.corrwith(y)
spearman_corr = feature_df.corrwith(y, method='spearman')
kendall_corr = feature_df.corrwith(y, method='kendall')

f_scores, p_values = f_regression(X_scaled, y)
f_df = pd.Series(f_scores, index=feature_df.columns)

metrics = {
    "Pearson Correlation": pearson_corr,
    "Spearman Correlation": spearman_corr,
    "Kendall Correlation": kendall_corr,
    "F-test Score": f_df
}

for name, series in metrics.items():
    top_25 = series.abs().sort_values(ascending=False).head(25)  

    print(f"\nTop 25 Features by {name}:")
    print(top_25.to_string(float_format="%.4f"))

    plt.figure(figsize=(10, 8))
    top_25.sort_values().plot(kind='barh', color='skyblue')
    plt.title(f"Top 25 Features by {name}")
    plt.xlabel(name)
    plt.ylabel("Feature")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

