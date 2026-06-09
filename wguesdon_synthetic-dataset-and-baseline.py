"""
Enhanced Synthetic BPM Prediction Dataset Generator
----------------------------------------------------
Improvements for more interesting ML competition:
- Hidden temporal patterns and seasonality
- Rare events and outliers
- Missing data patterns
- Feature drift between train/test
- Hidden categorical features encoded in continuous
- More complex non-linear interactions
- Adversarial noise patterns
"""

from __future__ import annotations
import os
import math
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

# ----------------------------- Configuration -----------------------------
RANDOM_SEED = 20250901
OUT_DIR = "."

N_TRAIN = 400_000
N_TEST = 124_163
ID_START_TRAIN = 0
ID_START_TEST = ID_START_TRAIN + N_TRAIN

BASELINE_TRAIN_MAX = 150_000

rng = np.random.default_rng(RANDOM_SEED)

# Enhanced feature schema with hidden patterns
SCHEMA = {
    "RhythmScore": (0.08, 0.97, 0.63, 0.16),
    "AudioLoudness": (-27.50, -1.36, -8.38, 4.62),
    "VocalContent": (0.02, 0.26, 0.07, 0.05),
    "AcousticQuality": (0.00, 0.99, 0.26, 0.22),
    "InstrumentalScore": (0.00, 0.87, 0.12, 0.13),
    "LivePerformanceLikelihood": (0.02, 0.60, 0.18, 0.12),
    "MoodScore": (0.03, 0.98, 0.56, 0.23),
    "TrackDurationMs": (63_973, 465_000, 242_000, 59_300),
    "Energy": (0.00, 1.00, 0.50, 0.29),
}

# Hidden features not directly exposed but influence the data
HIDDEN_FEATURES = {
    "TimeOfDay": (0, 24),  # Hour when song typically played
    "ReleaseYear": (1960, 2025),  # Influences production quality
    "Complexity": (0, 1),  # Musical complexity score
    "ProductionQuality": (0, 1),  # Studio quality
}

# Enhanced genres with more complex patterns
GENRES = [
    {
        "name": "edm", "p": 0.25,
        "delta": {"Energy": +0.15, "RhythmScore": +0.08, "InstrumentalScore": +0.05,
                  "VocalContent": -0.03, "AudioLoudness": +0.08},
        "target": {"boost": +4.0, "coeffs": {"Energy": +1.0, "RhythmScore": +0.6}},
        "tempo_preference": [120, 128, 140],  # Preferred BPM clusters
    },
    {
        "name": "hiphop", "p": 0.20,
        "delta": {"Energy": +0.05, "VocalContent": +0.08, "MoodScore": +0.05,
                  "AudioLoudness": +0.05},
        "target": {"boost": -2.0, "coeffs": {"VocalContent": -0.6, "MoodScore": +0.4}},
        "tempo_preference": [85, 95, 140],
    },
    {
        "name": "rock", "p": 0.20,
        "delta": {"Energy": +0.07, "LivePerformanceLikelihood": +0.07,
                  "AudioLoudness": +0.06},
        "target": {"boost": +2.5, "coeffs": {"LivePerformanceLikelihood": +0.5}},
        "tempo_preference": [120, 140, 160],
    },
    {
        "name": "pop", "p": 0.20,
        "delta": {"RhythmScore": +0.05, "MoodScore": +0.06, "AcousticQuality": +0.02},
        "target": {"boost": 0.0, "coeffs": {"RhythmScore": +0.4, "MoodScore": +0.3}},
        "tempo_preference": [120, 128],
    },
    {
        "name": "jazz", "p": 0.15,
        "delta": {"AcousticQuality": +0.15, "InstrumentalScore": +0.10,
                  "Energy": -0.10, "AudioLoudness": -0.06},
        "target": {"boost": -3.0, "coeffs": {"AcousticQuality": -0.6, "InstrumentalScore": -0.5}},
        "tempo_preference": [70, 105, 120],
    },
]

TEMPO_FAMILIES = np.array([70.0, 85.0, 95.0, 105.0, 120.0, 128.0, 140.0, 160.0, 175.0], dtype=float)

# ----------------------------- Helper Functions -----------------------------
def _to_unit(x, lo, hi):
    return (x - lo) / (hi - lo)

def _from_unit(u, lo, hi):
    return lo + u * (hi - lo)

def _beta_params_from_mean_sd(mean, sd, lo, hi):
    width = hi - lo
    m = (mean - lo) / width
    v = (sd / width) ** 2
    max_v = m * (1 - m) - 1e-9
    if v >= max_v:
        v = 0.95 * max_v
    common = m * (1 - m) / v - 1.0
    alpha = max(1e-3, m * common)
    beta = max(1e-3, (1 - m) * common)
    return alpha, beta

def sample_beta_scaled(n, mean, sd, lo, hi, *, rng):
    a, b = _beta_params_from_mean_sd(mean, sd, lo, hi)
    u = rng.beta(a, b, size=n)
    return _from_unit(u, lo, hi)

def sample_trunc_normal(n, mean, sd, lo, hi, *, rng, max_tries=20):
    out = np.empty(n, dtype=float)
    filled = 0
    remaining = n
    tries = 0
    while remaining > 0:
        tries += 1
        batch = rng.normal(mean, sd, size=remaining * 2)
        batch = batch[(batch >= lo) & (batch <= hi)]
        if batch.size == 0:
            if tries >= max_tries:
                batch = rng.normal(mean, sd, size=remaining)
                batch = np.clip(batch, lo, hi)
            else:
                continue
        take = min(remaining, batch.size)
        out[filled:filled + take] = batch[:take]
        filled += take
        remaining -= take
        if tries >= max_tries and remaining > 0:
            filler = rng.uniform(lo, hi, size=remaining)
            out[filled:filled + remaining] = filler
            remaining = 0
    return out

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# ----------------------------- Enhanced Features -----------------------------
def generate_hidden_features(n, *, rng):
    """Generate hidden features that influence visible features"""
    hidden = {}
    
    # Time of day - bimodal distribution (morning/evening peaks)
    morning_peak = rng.normal(9, 2, n // 2)
    evening_peak = rng.normal(20, 3, n - n // 2)
    hidden["TimeOfDay"] = np.clip(np.concatenate([morning_peak, evening_peak]), 0, 24)
    rng.shuffle(hidden["TimeOfDay"])
    
    # Release year - more recent songs cluster
    hidden["ReleaseYear"] = rng.beta(3, 1.5, n) * 65 + 1960
    
    # Complexity - correlated with instrumental score
    hidden["Complexity"] = rng.beta(2, 3, n)
    
    # Production quality - improves over time
    year_norm = (hidden["ReleaseYear"] - 1960) / 65
    hidden["ProductionQuality"] = np.clip(
        0.3 + 0.5 * year_norm + rng.normal(0, 0.15, n), 0, 1
    )
    
    return hidden

def add_temporal_patterns(X, hidden, n, *, rng):
    """Add hidden temporal patterns"""
    # Songs at certain times of day have different energy patterns
    time_factor = np.sin(2 * np.pi * hidden["TimeOfDay"] / 24)
    X["Energy"] = X["Energy"] * (1 + 0.2 * time_factor)
    X["Energy"] = np.clip(X["Energy"], 0, 1)
    
    # Production quality affects loudness
    X["AudioLoudness"] = X["AudioLoudness"] + 3 * hidden["ProductionQuality"]
    
    # Complexity influences instrumental score
    X["InstrumentalScore"] = X["InstrumentalScore"] * (1 + 0.3 * hidden["Complexity"])
    X["InstrumentalScore"] = np.clip(X["InstrumentalScore"], 0, 0.87)
    
    return X

def add_rare_events(X, n, *, rng, event_prob=0.005):
    """Add rare but significant events (outliers)"""
    n_events = int(n * event_prob)
    if n_events > 0:
        event_idx = rng.choice(n, n_events, replace=False)
        
        # Type 1: Live recordings with unusual patterns
        live_idx = event_idx[:n_events//2]
        X["LivePerformanceLikelihood"][live_idx] = rng.uniform(0.8, 1.0, len(live_idx))
        X["AudioLoudness"][live_idx] *= 0.7  # Lower quality
        X["VocalContent"][live_idx] *= 1.5  # More crowd noise
        
        # Type 2: Remixes with extreme energy
        remix_idx = event_idx[n_events//2:]
        X["Energy"][remix_idx] = rng.uniform(0.9, 1.0, len(remix_idx))
        X["RhythmScore"][remix_idx] = rng.uniform(0.85, 0.97, len(remix_idx))
    
    return X

def add_missing_data(X, n, *, rng, missing_prob=0.02):
    """Add realistic missing data patterns"""
    # Some features more likely to be missing together (recording issues)
    for i in range(int(n * missing_prob)):
        idx = rng.integers(0, n)
        if rng.random() < 0.3:  # 30% chance of multiple features missing
            # Correlated missingness
            if rng.random() < 0.5:
                X["AudioLoudness"] = X["AudioLoudness"].astype(float)
                X["VocalContent"] = X["VocalContent"].astype(float)
                X["AudioLoudness"][idx] = np.nan
                X["VocalContent"][idx] = np.nan
            else:
                X["LivePerformanceLikelihood"] = X["LivePerformanceLikelihood"].astype(float)
                X["MoodScore"] = X["MoodScore"].astype(float)
                X["LivePerformanceLikelihood"][idx] = np.nan
                X["MoodScore"][idx] = np.nan
        else:
            # Random single feature missing
            feature = rng.choice(list(X.keys()))
            if feature != "TrackDurationMs":  # Never missing duration
                X[feature] = X[feature].astype(float)
                X[feature][idx] = np.nan
    
    return X

def add_feature_interactions(X, hidden, *, rng):
    """Add complex non-linear feature interactions"""
    n = len(X["Energy"])
    
    # Three-way interaction: Energy × Rhythm × Time
    time_norm = hidden["TimeOfDay"] / 24
    interaction1 = X["Energy"] * X["RhythmScore"] * np.sin(2 * np.pi * time_norm)
    
    # Polynomial interaction
    interaction2 = (X["MoodScore"] ** 2) * X["VocalContent"]
    
    # Threshold-based interaction
    high_energy_mask = X["Energy"] > 0.7
    X["AudioLoudness"][high_energy_mask] += rng.normal(2, 0.5, np.sum(high_energy_mask))
    
    # XOR-like pattern (hard for trees)
    pattern = ((X["InstrumentalScore"] > 0.5).astype(int) + 
               (X["AcousticQuality"] > 0.5).astype(int)) == 1
    X["MoodScore"][pattern] *= 1.2
    
    # Ensure bounds
    for key in X:
        if key in SCHEMA:
            lo, hi, _, _ = SCHEMA[key]
            X[key] = np.clip(X[key], lo, hi)
    
    return X, interaction1, interaction2

def apply_train_test_drift(X, is_test, *, rng):
    """Apply subtle distribution shift between train and test"""
    if is_test:
        # Slight shift in energy distribution (simulating music trend changes)
        X["Energy"] = X["Energy"] * 1.03 + rng.normal(0, 0.01, len(X["Energy"]))
        X["Energy"] = np.clip(X["Energy"], 0, 1)
        
        # Production quality slightly higher in test (newer songs)
        X["AudioLoudness"] = X["AudioLoudness"] + rng.normal(0.5, 0.2, len(X["AudioLoudness"]))
        
        # Different genre mix (subtle)
        mood_shift = rng.normal(0.02, 0.01, len(X["MoodScore"]))
        X["MoodScore"] = np.clip(X["MoodScore"] + mood_shift, 0.03, 0.98)
    
    return X

def encode_hidden_categorical(X, genre_idx, hidden, *, rng):
    """Encode hidden categorical information in continuous features"""
    n = len(genre_idx)
    
    # Encode genre in decimal parts of certain features
    # This is discoverable but not obvious
    genre_encoding = (genre_idx + 1) / 100.0
    
    # Hide in the 3rd decimal place of MoodScore
    X["MoodScore"] = np.round(X["MoodScore"], 2) + genre_encoding / 10
    
    # Time period encoding in TrackDurationMs
    decade = ((hidden["ReleaseYear"] - 1960) // 10).astype(int)
    X["TrackDurationMs"] = X["TrackDurationMs"].astype(int)
    # Encode decade in last digit pattern
    X["TrackDurationMs"] = (X["TrackDurationMs"] // 10) * 10 + (decade % 10)
    
    return X

# ----------------------------- Enhanced Target Generation -----------------------------
def compute_enhanced_target_bpm(X, *, schema, genre_idx, hidden, interaction1, interaction2, rng):
    """Enhanced BPM calculation with more complex patterns"""
    U = {k: _to_unit(X[k], *schema[k][:2]).astype(float) for k in schema.keys()}
    
    # Handle missing values for calculation
    for k in U:
        if np.any(np.isnan(U[k])):
            U[k] = np.nan_to_num(U[k], nan=0.5)  # Impute with median
    
    # Base linear and quadratic terms
    z = (
        1.4 * U["Energy"]
        + 1.1 * U["RhythmScore"]
        + 0.5 * U["MoodScore"]
        + 0.6 * U["AudioLoudness"]
        + 0.3 * U["LivePerformanceLikelihood"]
        - 0.4 * U["VocalContent"]
        - 0.25 * U["AcousticQuality"]
        - 0.15 * U["InstrumentalScore"]
    )
    
    # Original interactions
    z += (
        1.5 * U["Energy"] * U["RhythmScore"]
        + 0.8 * U["MoodScore"] * U["RhythmScore"]
        + 0.6 * U["AudioLoudness"] * U["Energy"]
        - 0.7 * U["AcousticQuality"] * U["InstrumentalScore"]
        + 0.5 * U["LivePerformanceLikelihood"] * U["Energy"]
    )
    
    # Enhanced: Add hidden interactions
    z += 0.4 * interaction1 + 0.3 * interaction2
    
    # Time-based modulation
    time_norm = hidden["TimeOfDay"] / 24
    z += 0.5 * np.sin(2 * np.pi * time_norm) * U["Energy"]
    z += 0.3 * np.cos(4 * np.pi * time_norm) * U["RhythmScore"]
    
    # Production era influence
    year_factor = (hidden["ReleaseYear"] - 1960) / 65
    z += 0.2 * year_factor * (U["Energy"] - 0.5)
    
    # Complex periodic pattern
    duration_phase = 2.0 * math.pi * (3.0 * U["TrackDurationMs"] + 0.2 * U["RhythmScore"])
    z += 0.6 * np.sin(duration_phase) + 0.3 * np.cos(3 * duration_phase)
    
    # Base BPM with wider range
    base_bpm = 50.0 + 130.0 * sigmoid(1.3 * z - 0.3)
    bpm = base_bpm.copy()
    
    # Genre-specific adjustments with tempo preferences
    for gi, g in enumerate(GENRES):
        mask = (genre_idx == gi)
        if not np.any(mask):
            continue
        
        bpm[mask] += g["target"]["boost"]
        
        for fname, c in g["target"]["coeffs"].items():
            bpm[mask] += c * (U[fname][mask] - 0.5) * 10.0
        
        # Pull towards preferred tempos for genre
        if "tempo_preference" in g:
            genre_bpm = bpm[mask]
            for pref_tempo in g["tempo_preference"]:
                distance = np.abs(genre_bpm - pref_tempo)
                pull_strength = np.exp(-distance / 20)
                genre_bpm += pull_strength * (pref_tempo - genre_bpm) * 0.15
            bpm[mask] = genre_bpm
    
    # Nearest tempo family attraction (weakened)
    nearest = nearest_tempo_family(bpm)
    lam = sigmoid(0.3 + 0.5 * U["InstrumentalScore"] - 0.4 * U["VocalContent"])
    bpm = 0.8 * bpm + 0.2 * lam * nearest
    
    # Hidden complexity influence
    bpm += 5 * hidden["Complexity"] * (U["InstrumentalScore"] - 0.5)
    
    # Heteroscedastic noise (varies with features)
    noise_sd = (
        2.5 
        + 4.0 * (U["Energy"] - 0.5) ** 2 
        + 3.0 * (0.4 - U["AcousticQuality"]) ** 2
        + 2.0 * hidden["Complexity"]
    )
    
    # Add occasional large noise (outliers)
    noise = rng.normal(0.0, noise_sd)
    outlier_mask = rng.random(len(bpm)) < 0.01
    noise[outlier_mask] *= rng.uniform(2, 4, np.sum(outlier_mask))
    
    bpm = np.clip(bpm + noise, 50.0, 200.0)
    
    return bpm

def nearest_tempo_family(x):
    dists = np.abs(x[:, None] - TEMPO_FAMILIES[None, :])
    idx = np.argmin(dists, axis=1)
    return TEMPO_FAMILIES[idx]

def apply_genre_shifts(raw, features, schema):
    """Apply genre-based feature shifts"""
    n = len(next(iter(raw.values())))
    probs = np.array([g["p"] for g in GENRES], dtype=float)
    probs = probs / probs.sum()
    genre_idx = rng.choice(len(GENRES), size=n, p=probs)
    
    for fname in features:
        if fname in raw and fname in schema:
            lo, hi, _, _ = schema[fname]
            u = _to_unit(raw[fname], lo, hi)
            u2 = u.copy()
            
            for gi, g in enumerate(GENRES):
                if fname in g["delta"]:
                    delta = g["delta"][fname]
                    mask = (genre_idx == gi)
                    u2[mask] = np.clip(u2[mask] + delta, 0.0, 1.0)
            
            raw[fname] = _from_unit(u2, lo, hi)
    
    return genre_idx

# ----------------------------- Main Generation Functions -----------------------------
def generate_features(n, *, schema, rng):
    """Generate base features"""
    X = {}
    
    for fname in ["RhythmScore", "VocalContent", "AcousticQuality",
                  "InstrumentalScore", "LivePerformanceLikelihood",
                  "MoodScore", "Energy", "AudioLoudness"]:
        lo, hi, mean, sd = schema[fname]
        X[fname] = sample_beta_scaled(n, mean, sd, lo, hi, rng=rng)
    
    lo, hi, mean, sd = schema["TrackDurationMs"]
    X["TrackDurationMs"] = sample_trunc_normal(n, mean, sd, lo, hi, rng=rng)
    
    return X

def make_enhanced_dataframe(n, id_start, is_test=False, *, schema, rng):
    """Create enhanced dataframe with all improvements"""
    # Generate base features
    X = generate_features(n, schema=schema, rng=rng)
    
    # Generate hidden features
    hidden = generate_hidden_features(n, rng=rng)
    
    # Add temporal patterns
    X = add_temporal_patterns(X, hidden, n, rng=rng)
    
    # Apply genre shifts
    genre_idx = apply_genre_shifts(X, list(schema.keys()), schema)
    
    # Add feature interactions
    X, interaction1, interaction2 = add_feature_interactions(X, hidden, rng=rng)
    
    # Add rare events
    X = add_rare_events(X, n, rng=rng)
    
    # Apply train/test drift
    X = apply_train_test_drift(X, is_test, rng=rng)
    
    # Encode hidden information
    X = encode_hidden_categorical(X, genre_idx, hidden, rng=rng)
    
    # Add missing data (after other processing)
    if not is_test:  # Only in training data
        X = add_missing_data(X, n, rng=rng)
    
    # Create dataframe
    df = pd.DataFrame({
        "id": np.arange(id_start, id_start + n, dtype=np.int64),
        "RhythmScore": X["RhythmScore"],
        "AudioLoudness": X["AudioLoudness"],
        "VocalContent": X["VocalContent"],
        "AcousticQuality": X["AcousticQuality"],
        "InstrumentalScore": X["InstrumentalScore"],
        "LivePerformanceLikelihood": X["LivePerformanceLikelihood"],
        "MoodScore": X["MoodScore"],
        "TrackDurationMs": X["TrackDurationMs"].round(0).astype(np.int32),
        "Energy": X["Energy"],
    })
    
    return df, genre_idx, X, hidden, interaction1, interaction2

# ----------------------------- Baseline Model -----------------------------
def run_baseline_random_forest(train_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: str) -> None:
    """Enhanced baseline with missing data handling"""
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.impute import SimpleImputer
    
    feature_cols = [
        "RhythmScore", "AudioLoudness", "VocalContent", "AcousticQuality",
        "InstrumentalScore", "LivePerformanceLikelihood", "MoodScore",
        "TrackDurationMs", "Energy"
    ]
    
    # Handle missing data
    imputer = SimpleImputer(strategy='median')
    X_full = pd.DataFrame(
        imputer.fit_transform(train_df[feature_cols]),
        columns=feature_cols
    )
    y_full = train_df["BeatsPerMinute"].astype(float).values
    
    # Feature engineering hints for competitors
    # Add some basic interactions (competitors should find more)
    X_full["Energy_x_Rhythm"] = X_full["Energy"] * X_full["RhythmScore"]
    X_full["Mood_x_Vocal"] = X_full["MoodScore"] * X_full["VocalContent"]
    
    if len(train_df) > BASELINE_TRAIN_MAX:
        idx = rng.choice(len(train_df), size=BASELINE_TRAIN_MAX, replace=False)
        X = X_full.iloc[idx].reset_index(drop=True)
        y = y_full[idx]
        print(f"[Baseline] Using {len(X):,} samples for validation")
    else:
        X, y = X_full, y_full
    
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED
    )
    
    # Mean baseline
    mean_pred = np.full_like(y_va, y_tr.mean(), dtype=float)
    rmse_mean = mean_squared_error(y_va, mean_pred, squared=False)
    
    # RandomForest baseline
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    rf.fit(X_tr, y_tr)
    va_pred = rf.predict(X_va)
    rmse_rf = mean_squared_error(y_va, va_pred, squared=False)
    
    print("\n[Baseline] Validation Results:")
    print(f"  Mean-only RMSE: {rmse_mean:.4f}")
    print(f"  RandomForest RMSE: {rmse_rf:.4f}")
    print(f"  Improvement: {(1 - rmse_rf/rmse_mean)*100:.1f}%")
    
    # Note for competitors
    print("\n[Note] This baseline uses simple feature engineering.")
    print("       Better results possible with:")
    print("       - Advanced feature engineering")
    print("       - Handling missing data patterns")
    print("       - Detecting hidden encodings")
    print("       - Ensemble methods")
    
    # Final model on full data
    rf_final = RandomForestRegressor(
        n_estimators=400,
        max_depth=18,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    rf_final.fit(X_full, y_full)
    
    # Prepare test data
    X_test = pd.DataFrame(
        imputer.transform(test_df[feature_cols]),
        columns=feature_cols
    )
    X_test["Energy_x_Rhythm"] = X_test["Energy"] * X_test["RhythmScore"]
    X_test["Mood_x_Vocal"] = X_test["MoodScore"] * X_test["VocalContent"]
    
    test_pred = rf_final.predict(X_test)
    
    # Save submission
    sub = pd.DataFrame({"ID": test_df["id"], "BeatsPerMinute": test_pred})
    sub_path = os.path.join(out_dir, "baseline_submission.csv")
    sub.to_csv(sub_path, index=False)
    print(f"\n[Baseline] Saved: {sub_path}")
    
    # Feature importances
    base_features = feature_cols + ["Energy_x_Rhythm", "Mood_x_Vocal"]
    imp = pd.DataFrame({
        "feature": base_features,
        "importance": rf_final.feature_importances_
    }).sort_values("importance", ascending=False)
    
    imp_path = os.path.join(out_dir, "rf_feature_importances.csv")
    imp.to_csv(imp_path, index=False)
    print(f"[Baseline] Feature importances: {imp_path}")
    print(imp.head(10).to_string(index=False))

# ----------------------------- Main -----------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("Generating Enhanced BPM Dataset...")
    print("-" * 50)
    
    # Train split
    train_df, train_genre, train_raw, train_hidden, train_int1, train_int2 = \
        make_enhanced_dataframe(N_TRAIN, ID_START_TRAIN, is_test=False, schema=SCHEMA, rng=rng)
    
    train_bpm = compute_enhanced_target_bpm(
        train_raw, schema=SCHEMA, genre_idx=train_genre,
        hidden=train_hidden, interaction1=train_int1, interaction2=train_int2, rng=rng
    )
    train_df["BeatsPerMinute"] = train_bpm
    
    # Test split (with drift)
    test_df, test_genre, test_raw, test_hidden, test_int1, test_int2 = \
        make_enhanced_dataframe(N_TEST, ID_START_TEST, is_test=True, schema=SCHEMA, rng=rng)
    
    # Save files
    train_path = os.path.join(OUT_DIR, "train.csv")
    test_path = os.path.join(OUT_DIR, "test.csv")
    sub_path = os.path.join(OUT_DIR, "sample_submission.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    sample_sub = pd.DataFrame({
        "ID": test_df["id"],
        "BeatsPerMinute": np.full(len(test_df), train_df["BeatsPerMinute"].mean())
    })
    sample_sub.to_csv(sub_path, index=False)
    
    print("\nFiles created:")
    print(f"  {train_path} (shape={train_df.shape})")
    print(f"  {test_path} (shape={test_df.shape})")
    print(f"  {sub_path} (shape={sample_sub.shape})")
    
    # Statistics
    print("\n" + "=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)
    
    # Missing data report
    print("\nMissing Data (Training):")
    missing = train_df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        for col, count in missing.items():
            print(f"  {col}: {count} ({count/len(train_df)*100:.2f}%)")
    else:
        print("  No missing data")
    
    # Distribution summary
    print("\nFeature Distributions:")
    for col in ["Energy", "RhythmScore", "MoodScore", "BeatsPerMinute"]:
        if col in train_df.columns:
            print(f"  {col:20s}: μ={train_df[col].mean():.3f}, σ={train_df[col].std():.3f}, "
                  f"[{train_df[col].min():.2f}, {train_df[col].max():.2f}]")
    
    # Run baseline
    print("\n" + "=" * 50)
    print("RUNNING BASELINE MODEL")
    print("=" * 50)
    
    try:
        run_baseline_random_forest(train_df, test_df, OUT_DIR)
    except Exception as e:
        print(f"\n[Error] Baseline failed: {e}")
        print("Install scikit-learn: pip install scikit-learn")
    
    print("\n" + "=" * 50)
    print("COMPETITION READY!")
    print("=" * 50)
    print("\nChallenge aspects for competitors:")
    print("  ✓ Missing data patterns")
    print("  ✓ Train/test distribution drift")
    print("  ✓ Hidden categorical encodings")
    print("  ✓ Complex non-linear interactions")
    print("  ✓ Temporal and cyclical patterns")
    print("  ✓ Rare events and outliers")
    print("  ✓ Genre-specific behaviors")
    print("\nGood luck to all participants!")

if __name__ == "__main__":
    main()

