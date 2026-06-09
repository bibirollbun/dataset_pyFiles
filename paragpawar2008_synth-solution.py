!pip install -q git+https://github.com/S-G-mathematics/genuity_os.git
!pip install -q scikit-learn scipy numpy pandas deap lightgbm

import pandas as pd
import numpy as np
from scipy import stats, linalg
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import warnings
import random
from deap import base, creator, tools, algorithms

warnings.filterwarnings('ignore')
from genuity_os.core_generator.ctgan.ctgan.utils.api import CTGANAPI
from genuity_os.core_generator.dp.differential_privacy import DifferentialPrivacyProcessor

# Seed everything for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
seed_everything(42)

print("âœ… Ready for OPTIMIZED_PIPELINE_V2")


# Load data
df = pd.read_csv('/kaggle/input/genuityxethos/real_0.6.csv')

if 'Series' in df.columns:
    df = df.drop('Series', axis=1)

orig_cols = df.columns.tolist()
value_cols = [c for c in orig_cols if c not in ['t', 'row_id']]

# 1. Handle Categorical Columns (The Fix)
# We must convert strings (like 'Symbol') to numbers before anything else
object_cols = df.select_dtypes(include=['object']).columns
for col in object_cols:
    # Use Pandas Categorical codes (handles NaNs by assigning -1, which we fix later)
    df[col] = pd.Categorical(df[col]).codes

# 2. Handle Missing Values
for col in df.columns:
    if df[col].isnull().any():
        df[col].fillna(method='ffill', inplace=True)
        # If ffill didn't catch everything (start of series), use median
        df[col].fillna(df[col].median(), inplace=True)

# 3. Gaussian Transformation (QuantileTransformer)
# Now safe to apply because all data is numeric
print("Applying QuantileTransformer (Gaussian)...")
qt = QuantileTransformer(output_distribution='normal', 
                         n_quantiles=max(min(len(df), 1000), 100), 
                         random_state=42)

df_transformed = df.copy()
df_transformed[value_cols] = qt.fit_transform(df[value_cols])

print(f"Data Transformed: {df_transformed.shape}")
display(df_transformed.head())


print("="*70)
print("CTGAN TRAINING (On Gaussian Data)")
print("="*70)

# Identify columns (CTGAN needs to know categorical indices, though usually few in Time Series)
# In the transformed data, everything is continuous except potentially 'row_id' or 't' if treated as cat
# We treat all value_cols as continuous here because of QuantileTransform
cont_idx = [df_transformed.columns.get_loc(c) for c in value_cols]
cat_idx = [i for i in range(len(df_transformed.columns)) if i not in cont_idx]

ctgan = CTGANAPI()

print("Training CTGAN (800 epochs)...") # Increased slightly for convergence
losses = ctgan.fit(
    data=df_transformed.values, # <--- CRITICAL FIX: Passing transformed data
    continuous_cols=cont_idx,
    categorical_cols=cat_idx,
    epochs=800,
    batch_size=128,
)

# Generate massive pool for filtering
POOL_SIZE = 50000 
print(f"Generating {POOL_SIZE} candidates in latent space...")

candidates_list = []
# Generate in batches to save RAM
batch_size = 5000
for i in range(POOL_SIZE // batch_size):
    batch = ctgan.generate(batch_size)
    candidates_list.append(batch)
    print(f".", end="")

raw_candidates = np.vstack(candidates_list)
df_candidates_gaussian = pd.DataFrame(raw_candidates, columns=orig_cols)

print(f"\nâœ… Generated {len(df_candidates_gaussian)} candidates (Gaussian Space)")


print("="*70)
print("ADVERSARIAL FILTERING (Discriminator Selection)")
print("="*70)

# 1. Prepare Data for Discriminator
# Real data is labeled 1, Synthetic is labeled 0
X_real = df_transformed[value_cols].values
X_fake = df_candidates_gaussian[value_cols].values

X_adv = np.vstack([X_real, X_fake])
y_adv = np.hstack([np.ones(len(X_real)), np.zeros(len(X_fake))])

# 2. Train LightGBM Discriminator
print("Training Discriminator to detect fakes...")
clf = lgb.LGBMClassifier(random_state=42, n_estimators=100)
clf.fit(X_adv, y_adv)

# 3. Predict 'Realness' of Synthetic Data
probs = clf.predict_proba(X_fake)[:, 1] # Probability of being class 1 (Real)
df_candidates_gaussian['realness_score'] = probs

# 4. Select the rows that fooled the classifier the most
# We want 3322 rows, but let's take top 8000 first for the GA to refine later
top_candidates_gaussian = df_candidates_gaussian.nlargest(8000, 'realness_score').drop('realness_score', axis=1)

# 5. Inverse Transform back to Original Scale
print("Inverse Transforming selected candidates...")
df_candidates_inversed = top_candidates_gaussian.copy()
df_candidates_inversed[value_cols] = qt.inverse_transform(top_candidates_gaussian[value_cols])

# Clip to min/max of original to prevent exploding values from inverse transform
for col in value_cols:
    df_candidates_inversed[col] = df_candidates_inversed[col].clip(df[col].min(), df[col].max())

print(f"âœ… Selected top 8000 candidates based on Adversarial Score")


print("="*70)
print("ROBUST CORRELATION ALIGNMENT (Whitening & Coloring)")
print("="*70)

# 1. Setup Data
real_data = df[value_cols].values
syn_data = df_candidates_inversed[value_cols].values

# 2. Calculate Statistics
real_mean = np.mean(real_data, axis=0)
real_cov = np.cov(real_data, rowvar=False)

syn_mean = np.mean(syn_data, axis=0)
syn_cov = np.cov(syn_data, rowvar=False)

# 3. Whitening Transform (Remove synthetic correlation)
# Z = (X - mu) * L_inv
try:
    # Use SVD for stability instead of Cholesky
    U, S, Vh = np.linalg.svd(syn_cov)
    epsilon = 1e-5
    # Inverse square root of covariance
    whiten_matrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))
    
    # Whiten the synthetic data
    centered_syn = syn_data - syn_mean
    whitened = np.dot(centered_syn, whiten_matrix)
    
    # 4. Coloring Transform (Inject real correlation)
    # X_new = Z * L_real + mu_real
    U_r, S_r, Vh_r = np.linalg.svd(real_cov)
    color_matrix = np.dot(U_r, np.dot(np.diag(np.sqrt(S_r + epsilon)), U_r.T))
    
    # Color the data
    aligned_data = np.dot(whitened, color_matrix) + real_mean
    
    # 5. update DataFrame
    df_refined = df_candidates_inversed.copy()
    df_refined[value_cols] = aligned_data
    
    # Clip to bounds
    for col in value_cols:
        df_refined[col] = df_refined[col].clip(df[col].min(), df[col].max())
        
    print("âœ… Correlation Matrix Aligned via SVD (Robust)")
    
except Exception as e:
    print(f"â�Œ Alignment failed: {e}")
    print("Using raw candidates (expect lower correlation score)")
    df_refined = df_candidates_inversed.copy()


print("="*70)
print("FINAL SELECTION VIA GENETIC ALGORITHM (Sort-Aware)")
print("="*70)

# --- RE-DEFINE ACF FUNCTION ---
def compute_acf(series, nlags=10):
    series = np.asarray(series)
    series = series[np.isfinite(series)]
    if len(series) < 2: return np.ones(nlags + 1)
    
    mean = np.mean(series)
    var = np.var(series)
    if var < 1e-10: return np.ones(nlags + 1)
    
    n = len(series)
    acf_vals = [1.0]
    for lag in range(1, min(nlags + 1, n)):
        # Efficient covariance calculation
        c = np.sum((series[:-lag] - mean) * (series[lag:] - mean)) / n
        acf_vals.append(c / var)
    return np.array(acf_vals)

# Pre-compute Real ACF
real_acf = {col: compute_acf(df[col].values) for col in value_cols}

# --- GA SETUP ---
if hasattr(creator, "FitnessMin"): del creator.FitnessMin
if hasattr(creator, "Individual"): del creator.Individual

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# We select indices from our refined pool
toolbox.register("indices", random.sample, range(len(df_refined)), 3322)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eval_fitness(individual):
    # 1. Materialize the dataframe
    sample = df_refined.iloc[list(individual)].copy()
    
    # CRITICAL FIX: Sort by 't' to restore time structure
    # If 't' was transformed, we use the original 't' column if available or infer it
    # We assume 't' is present in df_refined (it should be carried over)
    if 't' in sample.columns:
        sample = sample.sort_values('t')
    
    # 2. Moments Error (Mean/Std) - Weight: 1.0
    L_m = 0.0
    for col in value_cols:
        L_m += abs(sample[col].mean() - df[col].mean()) / (df[col].std() + 1e-6)
        L_m += abs(sample[col].std() - df[col].std()) / (df[col].std() + 1e-6)
    
    # 3. Correlation Error (Frobenius) - Weight: 0.5 (Already fixed by Matrix step)
    # We keep it light just to prevent the GA from breaking the matrix alignment
    real_corr = df[value_cols].corr().values
    samp_corr = sample[value_cols].corr().fillna(0).values
    L_c = np.linalg.norm(real_corr - samp_corr)
    
    # 4. ACF Error - Weight: 2.0 (High priority now)
    L_a = 0.0
    for col in value_cols:
        s_acf = compute_acf(sample[col].values)
        r_acf = real_acf[col]
        L_a += np.mean(np.abs(s_acf - r_acf))
        
    return (1.0*L_m + 0.5*L_c + 2.0*L_a,)

toolbox.register("evaluate", eval_fitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# --- RUN GA ---
print("Running Sort-Aware GA (Focus: ACF Restoration)...")
pop = toolbox.population(n=60) # Increased population
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min)

# Run for 60 generations
pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=60, 
                               stats=stats, verbose=True)

best_ind = tools.selBest(pop, 1)[0]
final_df = df_refined.iloc[list(best_ind)].reset_index(drop=True)

# Final Sort ensure
if 't' in final_df.columns:
    final_df = final_df.sort_values('t').reset_index(drop=True)

print("âœ… GA Finished (ACF Optimized)")


print("="*70)
print("TRAJECTORY MATCHING V2 (Smoothed Target)")
print("="*70)

from scipy.optimize import linear_sum_assignment
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist

# 1. Create the "Silky Smooth" Ideal Path
# We take the Real Data and smooth it aggressively to get the pure trend
real_data = df[value_cols].values
real_smooth = np.zeros_like(real_data)
for i in range(real_data.shape[1]):
    # Window 51, Poly 3 -> Strong smoothing
    real_smooth[:, i] = savgol_filter(real_data[:, i], 51, 3)

# 2. Interpolate to Target Size (3322 rows)
original_steps = np.linspace(0, 1, len(real_data))
target_steps = np.linspace(0, 1, len(final_df))
interpolator = interp1d(original_steps, real_smooth, kind='cubic', axis=0)
ideal_path = interpolator(target_steps)

# 3. Compute Cost Matrix & Solve Assignment
# We match Synthetic rows to this Smooth Ideal Path
scaler_sort = StandardScaler()
X_syn_scaled = scaler_sort.fit_transform(final_df[value_cols])
X_ideal_scaled = scaler_sort.transform(ideal_path)

print("Solving Optimal Transport (Hungarian Algorithm)...")
# Calculate distance matrix (float32 for speed/RAM)
cost_matrix = cdist(X_syn_scaled, X_ideal_scaled, metric='euclidean').astype('float32')
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# 4. Apply Sort
sort_map = pd.DataFrame({'syn_idx': row_ind, 'time_rank': col_ind})
sort_map = sort_map.sort_values('time_rank')
final_df = final_df.iloc[sort_map['syn_idx'].values].reset_index(drop=True)

# 5. Re-Stamp Time
if 't' in df.columns:
    t_min = df['t'].min()
    t_range = df['t'].max() - t_min
    extended_max = t_min + t_range * (len(final_df) / len(df))
    new_t = np.linspace(t_min, extended_max, len(final_df))
    final_df['t'] = new_t.astype(int)

print("âœ… Rows reordered to match Smoothed Real Trajectory")


print("="*70)
print("JOINT OPTIMIZATION LOOP (Iterative Ping-Pong)")
print("="*70)

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# Configuration
MAX_ITER = 15
CORR_TARGET = 1.95  # Slightly tighter than 2.0 to be safe
ACF_TARGET = 0.048  # Slightly tighter than 0.05
best_score = float('inf')
best_df = final_df.copy()

# --- HELPER FUNCTIONS ---

def get_stats(curr_df):
    """Calculate current error metrics"""
    # 1. Correlation Error
    real_corr = df[value_cols].corr().fillna(0).values
    syn_corr = curr_df[value_cols].corr().fillna(0).values
    c_err = np.linalg.norm(real_corr - syn_corr)
    
    # 2. ACF Error
    a_err = 0.0
    for col in value_cols:
        r_acf = [df[col].autocorr(l) for l in range(1, 11)]
        s_acf = [curr_df[col].autocorr(l) for l in range(1, 11)]
        a_err += np.mean(np.abs(np.nan_to_num(r_acf) - np.nan_to_num(s_acf)))
    a_err /= len(value_cols)
    return c_err, a_err

def fix_correlation(curr_df):
    """Apply SVD Coloring"""
    try:
        real_cov = np.cov(df[value_cols].values, rowvar=False)
        syn_data = curr_df[value_cols].values
        syn_mean = np.mean(syn_data, axis=0)
        syn_cov = np.cov(syn_data, rowvar=False)
        
        # Whitening
        U, S, Vh = np.linalg.svd(syn_cov)
        epsilon = 1e-6
        whiten = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T))
        whitened = np.dot(syn_data - syn_mean, whiten)
        
        # Coloring
        U_r, S_r, Vh_r = np.linalg.svd(real_cov)
        color = np.dot(U_r, np.dot(np.diag(np.sqrt(S_r + epsilon)), U_r.T))
        aligned = np.dot(whitened, color) + np.mean(df[value_cols].values, axis=0)
        
        curr_df[value_cols] = aligned
        for col in value_cols:
            curr_df[col] = curr_df[col].clip(df[col].min(), df[col].max())
        return curr_df
    except:
        return curr_df

def fix_acf(curr_df):
    """Apply FFT Magnitude Injection"""
    for col in value_cols:
        real_fft = np.fft.rfft(df[col].values)
        real_mag = np.abs(real_fft)
        
        syn_vals = curr_df[col].values
        syn_fft = np.fft.rfft(syn_vals)
        syn_phase = np.angle(syn_fft)
        
        # Interpolate Magnitude
        real_idx = np.linspace(0, 1, len(real_mag))
        syn_idx = np.linspace(0, 1, len(syn_phase))
        interp = interp1d(real_idx, real_mag, kind='linear')
        target_mag = interp(syn_idx)
        
        # Inject
        new_fft = target_mag * np.exp(1j * syn_phase)
        new_vals = np.fft.irfft(new_fft, n=len(curr_df))
        
        # Restore Variance
        t_std = df[col].std()
        c_std = new_vals.std()
        if c_std > 1e-9:
            new_vals = (new_vals - new_vals.mean()) * (t_std / c_std) + df[col].mean()
            
        curr_df[col] = np.clip(new_vals, df[col].min(), df[col].max())
    return curr_df

# --- THE OPTIMIZATION LOOP ---

print(f"Starting Ping-Pong Optimization (Max {MAX_ITER} loops)...")
print(f"Targets: Corr < {CORR_TARGET}, ACF < {ACF_TARGET}")

for i in range(1, MAX_ITER + 1):
    # 1. Ping: Fix Correlation
    final_df = fix_correlation(final_df)
    
    # 2. Pong: Fix ACF
    final_df = fix_acf(final_df)
    
    # 3. Check Scores
    c_err, a_err = get_stats(final_df)
    
    # 4. Composite Score (for Best-Save)
    # We weight them by their difficulty (Corr is usually larger)
    comp_score = (c_err / 2.0) + (a_err / 0.05)
    
    print(f"  Loop {i:02d}: Corr={c_err:.4f}, ACF={a_err:.4f}", end="")
    
    if comp_score < best_score:
        best_score = comp_score
        best_df = final_df.copy()
        print(" (New Best!)", end="")
    
    print()
    
    # 5. Success Check
    if c_err < CORR_TARGET and a_err < ACF_TARGET:
        print(f"âœ… CONVERGED at Loop {i}!")
        break

# Restore Best
final_df = best_df.copy()
print(f"\nğŸ�† Final Selected Stats: Corr={get_stats(final_df)[0]:.4f}, ACF={get_stats(final_df)[1]:.4f}")

# --- FINAL MICRO-NOISE (Must be tiny!) ---
print("Applying Final Micro-Noise (0.2%)...")
for col in value_cols:
    noise = np.random.normal(0, 0.0000001 * df[col].std(), len(final_df))
    final_df[col] += noise
    final_df[col] = final_df[col].clip(df[col].min(), df[col].max())

print("âœ… Optimization Complete.")


print("="*70)
print("TRIPLE-LOCK FINISHER (Dist + DCT + SVD)")
print("="*70)

from scipy.fftpack import dct, idct
from scipy.stats import rankdata

# Configuration
ACF_TARGET = 0.045
MAX_DCT_CUTOFF = 0.5  # Nyquist
MIN_DCT_CUTOFF = 0.01

def get_col_acf_error(real_col, syn_col):
    r_a = [real_col.autocorr(l) for l in range(1, 11)]
    s_a = [pd.Series(syn_col).autocorr(l) for l in range(1, 11)]
    return np.mean(np.abs(np.nan_to_num(r_a) - np.nan_to_num(s_a)))

print("Optimizing: Distribution -> Frequency (DCT) -> Correlation...")

for col in value_cols:
    # --- LOCK 1: DISTRIBUTION MATCHING ---
    # Force Syn values to follow the EXACT distribution of Real values
    # This fixes Mean, Std, Skew, Kurtosis instantly.
    real_vals_sorted = np.sort(df[col].values)
    # Stretch real values to match synthetic length
    if len(final_df) != len(df):
        real_vals_sorted = np.interp(
            np.linspace(0, 1, len(final_df)),
            np.linspace(0, 1, len(df)),
            real_vals_sorted
        )
    
    # Map Syn ranks to Real values
    syn_ranks = rankdata(final_df[col].values, method='ordinal') - 1
    # Safety clamp
    syn_ranks = np.clip(syn_ranks, 0, len(real_vals_sorted) - 1)
    final_df[col] = real_vals_sorted[syn_ranks]
    
    # --- LOCK 2: DCT SPECTRAL FILTERING ---
    # Convert to Frequency Domain
    y = final_df[col].values
    y_dct = dct(y, norm='ortho')
    
    best_y = y.copy()
    best_err = get_col_acf_error(df[col], best_y)
    
    # Progressive Low-Pass Filter
    # We zero out high frequencies until ACF is perfect
    n_coeffs = len(y_dct)
    for keep_ratio in np.arange(0.95, 0.05, -0.05):
        cutoff_idx = int(n_coeffs * keep_ratio)
        
        # Zero out high freq
        y_dct_filt = y_dct.copy()
        y_dct_filt[cutoff_idx:] = 0
        
        # Inverse DCT
        y_filt = idct(y_dct_filt, norm='ortho')
        
        # Check Error
        err = get_col_acf_error(df[col], y_filt)
        
        if err < best_err:
            best_err = err
            best_y = y_filt
        
        # Stop if we hit the magic number
        if best_err < ACF_TARGET:
            break
            
    final_df[col] = best_y
    print(f"  {col[:15]}: Final ACF={best_err:.4f}")

# --- LOCK 3: CORRELATION TOUCH-UP ---
# Only apply if Correlation drifted above 1.8 (Save the ACF work!)
real_corr = df[value_cols].corr().fillna(0).values
syn_corr = final_df[value_cols].corr().fillna(0).values
c_err = np.linalg.norm(real_corr - syn_corr)

if c_err > 1.8:
    print(f"Refining Correlation (Current: {c_err:.2f})...")
    try:
        # Gentle SVD Mix (50% strength) to avoid breaking ACF
        # We blend the Aligned version with the Current version
        syn_data = final_df[value_cols].values
        syn_mean = np.mean(syn_data, axis=0)
        syn_cov = np.cov(syn_data, rowvar=False)
        real_cov = np.cov(df[value_cols].values, rowvar=False)
        
        # Whitening
        U, S, Vh = np.linalg.svd(syn_cov)
        whiten = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + 1e-6)), U.T))
        whitened = np.dot(syn_data - syn_mean, whiten)
        
        # Coloring
        U_r, S_r, Vh_r = np.linalg.svd(real_cov)
        color = np.dot(U_r, np.dot(np.diag(np.sqrt(S_r + 1e-6)), U_r.T))
        aligned = np.dot(whitened, color) + np.mean(df[value_cols].values, axis=0)
        
        # 50% Blend
        final_df[value_cols] = 0.5 * final_df[value_cols] + 0.5 * aligned
        print("âœ… Correlation Soft-Lock applied")
    except:
        pass
else:
    print(f"âœ… Correlation Safe ({c_err:.4f}), skipping SVD to preserve ACF.")

# Clip finally
for col in value_cols:
    final_df[col] = final_df[col].clip(df[col].min(), df[col].max())

print("âœ… Triple-Lock Complete.")


print("="*70)
print("DIAGNOSTICS: INTERNAL SCORE ESTIMATION")
print("="*70)

# 1. Correlation Error (Frobenius Norm)
real_corr = df[value_cols].corr().fillna(0).values
syn_corr = final_df[value_cols].corr().fillna(0).values
corr_error = np.linalg.norm(real_corr - syn_corr)

# 2. ACF Error (Auto-Correlation)
def quick_acf(series, lag=10):
    return [pd.Series(series).autocorr(l) for l in range(1, lag+1)]

acf_error = 0.0
for col in value_cols:
    r_acf = np.nan_to_num(quick_acf(df[col]))
    s_acf = np.nan_to_num(quick_acf(final_df[col]))
    acf_error += np.mean(np.abs(r_acf - s_acf))
acf_error /= len(value_cols)

print(f"ğŸ“‰ Correlation Matrix Error: {corr_error:.4f} (Target: < 2.0)")
print(f"ğŸ“‰ Average ACF Error:        {acf_error:.4f} (Target: < 0.05)")

if corr_error < 2.0 and acf_error < 0.05:
    print("\nâœ… GREEN LIGHT: Statistics look excellent.")
else:
    print("\nâš ï¸� YELLOW LIGHT: Statistics are drifting. Consider re-running GA.")


# Final cleanup
final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
for col in final_df.columns:
    if final_df[col].isnull().any():
        final_df[col].fillna(df[col].median(), inplace=True)

# Integer handling
for col in final_df.columns:
    if col in df.columns and df[col].dtype in ['int64', 'int32']:
        final_df[col] = final_df[col].round().astype('int64')

# Clip T and Sort
if 't' in final_df.columns:
    final_df['t'] = final_df['t'].clip(df['t'].min(), df['t'].max()).astype('int64')
    final_df = final_df.sort_values('t').reset_index(drop=True)

# Add Row ID
final_df = final_df.reset_index(drop=True)
if 'row_id_column_name' not in final_df.columns:
    final_df.insert(0, 'row_id_column_name', np.arange(len(final_df)))
else:
    final_df['row_id_column_name'] = np.arange(len(final_df))

# Validate
assert len(final_df) == 3322
assert final_df.columns[0] == 'row_id_column_name'

final_df.to_csv('submission.csv', index=False)
print("âœ… Saved submission.csv")
print(final_df.head())




