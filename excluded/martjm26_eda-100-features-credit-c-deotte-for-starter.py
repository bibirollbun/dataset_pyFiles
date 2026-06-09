%load_ext cudf.pandas

import numpy as np, pandas as pd
import matplotlib.pyplot as plt


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
train = pd.concat([train, train_extra], axis=0, ignore_index=True)


original_df = pd.read_csv("/kaggle/input/student-bag-price-prediction-dataset/Noisy_Student_Bag_Price_Prediction_Dataset.csv")
original_df = original_df.groupby("Weight Capacity (kg)").Price.mean()
original_df.name = "original_Price"
train = train.merge(original_df, on="Weight Capacity (kg)", how="left")
test = test.merge(original_df, on="Weight Capacity (kg)", how="left")


# merge features from original to train and test df's
original_df = pd.read_csv("/kaggle/input/student-bag-price-prediction-dataset/Noisy_Student_Bag_Price_Prediction_Dataset.csv")
original_df = original_df.loc[(original_df["Weight Capacity (kg)"]>5)&(original_df["Weight Capacity (kg)"]<30)]
original_df.columns = [f"original_{c}" for c in original_df.columns]
train = train.merge(original_df.iloc[:,:-1], left_on="Weight Capacity (kg)", right_on="original_Weight Capacity (kg)", how="left")
#train_df = train_df.drop("id",axis=1)
test = test.merge(original_df.iloc[:,:-1], left_on="Weight Capacity (kg)", right_on="original_Weight Capacity (kg)", how="left")
display(train.info(), test.info())


import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from statsmodels.nonparametric.kde import KDEUnivariate

def analyze_dataset_noise(data, verbose=True):
    """
    Analyze a dataset to estimate noise levels and signal strength
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset to analyze
    verbose : bool
        Whether to print analysis results
        
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    results = {}
    
    # 1. Basic statistical analysis
    if verbose:
        print("1. BASIC STATISTICAL ANALYSIS")
        print("-" * 50)
    
    # Check for missing values
    missing_percent = data.isnull().mean() * 100
    results['missing_values'] = missing_percent
    
    if verbose:
        print(f"Missing values percentage per column:")
        for col, pct in missing_percent.items():
            print(f"  {col}: {pct:.2f}%")
        print()
    
    # Check for outliers using Z-score
    numeric_data = data.select_dtypes(include=[np.number]).dropna()
    if not numeric_data.empty:
        z_scores = stats.zscore(numeric_data, nan_policy='omit')
        outliers_z = (np.abs(z_scores) > 3).any(axis=1).sum()
        outlier_percent_z = outliers_z / len(data) * 100
        results['outliers_zscore_percent'] = outlier_percent_z
        
        if verbose:
            print(f"Potential outliers (Z-score > 3): {outliers_z} rows ({outlier_percent_z:.2f}%)")
            print()
    else:
        if verbose:
            print("No numeric data available for outlier analysis")
            print()
    
    # 2. Distribution analysis
    if verbose:
        print("2. DISTRIBUTION ANALYSIS")
        print("-" * 50)
    
    # Calculate skewness and kurtosis for numerical columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        dist_stats = pd.DataFrame({
            'skewness': data[numeric_cols].skew(),
            'kurtosis': data[numeric_cols].kurtosis()
        })
        results['distribution_stats'] = dist_stats
        
        if verbose:
            print("Distribution statistics (high absolute values may indicate noise or signal):")
            print(dist_stats)
            print()
    else:
        if verbose:
            print("No numeric data available for distribution analysis")
            print()
    
    # 3. Signal-to-noise estimation using PCA
    if verbose:
        print("3. SIGNAL-TO-NOISE ESTIMATION USING PCA")
        print("-" * 50)
    
    # Standardize data for PCA
    numeric_data = data.select_dtypes(include=[np.number]).dropna()
    if len(numeric_data) > 0 and len(numeric_data.columns) > 1:  # Need at least 2 columns for PCA
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        # Apply PCA
        pca = PCA()
        pca.fit(scaled_data)
        
        # Calculate explained variance ratio
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # Estimate signal based on principal components
        # (Components that explain significant variance likely contain signal)
        n_components_90pct = np.argmax(cumulative_variance >= 0.9) + 1
        signal_ratio = sum(explained_variance[:n_components_90pct]) / sum(explained_variance)
        noise_ratio = 1 - signal_ratio
        
        results['pca_signal_ratio'] = signal_ratio
        results['pca_noise_ratio'] = noise_ratio
        results['pca_n_components_90pct'] = n_components_90pct
        results['pca_explained_variance'] = explained_variance
        
        if verbose:
            print(f"PCA Analysis Results:")
            print(f"  Estimated signal ratio: {signal_ratio:.2f} ({signal_ratio*100:.1f}%)")
            print(f"  Estimated noise ratio: {noise_ratio:.2f} ({noise_ratio*100:.1f}%)")
            print(f"  Number of components to explain 90% variance: {n_components_90pct}")
            print(f"  Variance explained by top 5 components: {explained_variance[:min(5, len(explained_variance))]}")
            print()
    else:
        if verbose:
            print("Not enough numeric data for PCA analysis (need at least 2 numeric columns)")
            print()
    
    # 4. Anomaly detection using Isolation Forest
    if verbose:
        print("4. ANOMALY DETECTION")
        print("-" * 50)
    
    if len(numeric_data) > 0 and len(numeric_data.columns) > 0:
        # Use Isolation Forest to identify anomalies (potential noise)
        try:
            if len(numeric_data.columns) > 1:
                # Multi-dimensional data
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                scaled_data = StandardScaler().fit_transform(numeric_data)
                outliers = iso_forest.fit_predict(scaled_data)
            else:
                # One-dimensional data
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                scaled_data = StandardScaler().fit_transform(numeric_data.values.reshape(-1, 1))
                outliers = iso_forest.fit_predict(scaled_data)
                
            outlier_count = (outliers == -1).sum()
            outlier_percent = outlier_count / len(numeric_data) * 100
            
            results['isolation_forest_outliers'] = outlier_count
            results['isolation_forest_outlier_percent'] = outlier_percent
            
            if verbose:
                print(f"Isolation Forest anomaly detection:")
                print(f"  Detected anomalies: {outlier_count} samples ({outlier_percent:.2f}%)")
                print()
        except Exception as e:
            if verbose:
                print(f"Error during anomaly detection: {str(e)}")
                print()
    else:
        if verbose:
            print("Not enough numeric data for anomaly detection")
            print()
    
    # 5. Feature correlation analysis
    if verbose:
        print("5. FEATURE CORRELATION ANALYSIS")
        print("-" * 50)
    
    # Calculate correlation matrix
    if len(numeric_data.columns) > 1:
        try:
            corr_matrix = numeric_data.corr()
            
            # Get average absolute correlation per feature
            # Fix: Check if corr_matrix is a DataFrame before using sort_values
            mean_corr = np.abs(corr_matrix).mean()
            if isinstance(mean_corr, pd.Series):
                avg_corr = mean_corr.sort_values(ascending=False)
            else:
                # For the case where mean_corr is a scalar (one column)
                avg_corr = mean_corr
                
            results['average_correlation'] = avg_corr
            
            if verbose:
                print("Average absolute correlation per feature (higher suggests stronger signal):")
                if isinstance(avg_corr, pd.Series):
                    for col, val in avg_corr.items():
                        print(f"  {col}: {val:.4f}")
                else:
                    print(f"  Single feature correlation: {avg_corr:.4f}")
                print()
        except Exception as e:
            if verbose:
                print(f"Error during correlation analysis: {str(e)}")
                print()
    else:
        if verbose:
            print("Not enough numeric columns for correlation analysis (need at least 2)")
            print()
    
    # Create summary of noise/signal indicators
    if verbose:
        print("SUMMARY")
        print("-" * 50)
        
        if 'pca_signal_ratio' in results:
            print(f"Estimated signal-to-noise ratio (PCA-based): {results['pca_signal_ratio']:.2f}:{results['pca_noise_ratio']:.2f}")
            print(f"Estimated signal percentage: {results['pca_signal_ratio']*100:.1f}%")
            print(f"Estimated noise percentage: {results['pca_noise_ratio']*100:.1f}%")
        
        if 'isolation_forest_outlier_percent' in results:
            print(f"Estimated noise from anomaly detection: {results['isolation_forest_outlier_percent']:.1f}%")
    
    return results


def plot_noise_analysis(data, results):
    """Plot visualizations of the noise analysis results"""
    # Set up the matplotlib figure
    fig = plt.figure(figsize=(15, 12))
    
    # Check if we have enough numeric data to plot
    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.empty:
        plt.figtext(0.5, 0.5, "Not enough numeric data for visualization", 
                   ha="center", va="center", fontsize=14)
        plt.tight_layout()
        plt.show()
        return
    
    # Plot counter to track subplot positions
    plot_counter = 1
    
    # 1. PCA explained variance plot
    if 'pca_explained_variance' in results:
        ax1 = plt.subplot(2, 2, plot_counter)
        plot_counter += 1
        
        exp_var = results['pca_explained_variance']
        plt.bar(range(1, len(exp_var) + 1), exp_var, alpha=0.8, label='Individual explained variance')
        plt.step(range(1, len(exp_var) + 1), np.cumsum(exp_var), where='mid', label='Cumulative explained variance')
        plt.axhline(y=0.9, color='r', linestyle='-', alpha=0.5, label='90% threshold')
        plt.xlabel('Principal Components')
        plt.ylabel('Explained variance ratio')
        plt.title('PCA: Explained Variance by Components')
        plt.legend(loc='best')
    
    # 2. Distribution of numeric features
    if len(numeric_data.columns) > 0:
        ax2 = plt.subplot(2, 2, plot_counter)
        plot_counter += 1
        
        # Standardize data for comparison
        try:
            scaler = StandardScaler()
            scaled_data = pd.DataFrame(scaler.fit_transform(numeric_data), columns=numeric_data.columns)
            
            # Plot distribution of each numeric column
            for column in scaled_data.columns:
                sns.kdeplot(scaled_data[column], label=column)
            
            plt.xlabel('Standardized values')
            plt.ylabel('Density')
            plt.title('Distribution of Standardized Features')
            if len(scaled_data.columns) <= 10:  # Only show legend if not too many columns
                plt.legend()
        except Exception as e:
            plt.figtext(0.5, 0.5, f"Error plotting distributions: {str(e)}", 
                       ha="center", va="center", fontsize=10)
    
    # 3. Skewness and Kurtosis
    if 'distribution_stats' in results:
        ax4 = plt.subplot(2, 2, plot_counter)
        plot_counter += 1
        
        try:
            dist_stats = results['distribution_stats']
            dist_stats.plot(kind='bar', ax=ax4)
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.title('Skewness and Kurtosis by Feature')
            plt.ylabel('Value')
        except Exception as e:
            plt.figtext(0.5, 0.5, f"Error plotting distribution stats: {str(e)}", 
                       ha="center", va="center", fontsize=10)
    
    plt.tight_layout()
    plt.show()


results = analyze_dataset_noise(train)
plot_noise_analysis(train, results)


CATS = list(train.drop(columns=["Price", "id", "Weight Capacity (kg)", "original_Weight Capacity (kg)"]).columns)
print(f"There are {len(CATS)} categorical columns:")
print( CATS )
print(f"There are 2 numerical columns:")
print( ["Weight Capacity (kg)", "original_Weight Capacity (kg)"] )


COMBO = []
for i,c in enumerate(CATS):
    #print(f"{c}, ",end="")
    combine = pd.concat([train[c],test[c]],axis=0)
    combine,_ = pd.factorize(combine)
    train[c] = combine[:len(train)]
    test[c] = combine[len(train):]
    n = f"{c}_wc"
    train[n] = train[c]*100 + train["original_Weight Capacity (kg)"]
    test[n] = test[c]*100 + test["original_Weight Capacity (kg)"]
    COMBO.append(n)
for i,c in enumerate(CATS):
    #print(f"{c}, ",end="")
    combine = pd.concat([train[c],test[c]],axis=0)
    combine,_ = pd.factorize(combine)
    train[c] = combine[:len(train)]
    test[c] = combine[len(train):]
    n = f"{c}_orig_wc"
    train[n] = train[c]*100 + train["original_Weight Capacity (kg)"]
    test[n] = test[c]*100 + test["original_Weight Capacity (kg)"]
    COMBO.append(n)
print()
print(f"We engineer {len(COMBO)} new columns!")
print( COMBO )


FEATURES = CATS + ["Weight Capacity (kg)", "original_Weight Capacity (kg)"] + COMBO
print(f"We now have {len(FEATURES)} columns:")
print( FEATURES )
train.info()


train_cols = train.select_dtypes(include=['float64']).columns
train[train_cols] = train[train_cols].astype('float32')
train_cols = train.select_dtypes(include=['int64']).columns
train[train_cols] = train[train_cols].astype('int16')

test_cols = test.select_dtypes(include=['float64']).columns
test[test_cols] = test[test_cols].astype('float32')
test_cols = test.select_dtypes(include=['int64']).columns
test[test_cols] = test[test_cols].astype('int16')
train.info()


from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import xgboost as xgb
import cudf
import cupy as cp
print(f"XGBoost version",xgb.__version__)


# STATISTICS TO AGGEGATE FOR OUR FEATURE GROUPS
STATS = ["mean","std","count","nunique","median","min","max","skew"]
STATS2 = ["mean","std"]


def optimize_dtypes(df):
    """Convert float64→float32 and int64→int32 to reduce memory usage."""
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int16')
    return df


FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof = np.zeros(len(train))
pred = np.zeros(len(test))

# OUTER K-FOLD
for i, (train_index, test_index) in enumerate(kf.split(train)):
    print(f"### OUTER Fold {i+1} ###")

    X_train = train.loc[train_index, FEATURES + ['Price']].reset_index(drop=True).copy()
    y_train = train.loc[train_index, 'Price']

    X_valid = train.loc[test_index, FEATURES].reset_index(drop=True).copy()
    y_valid = train.loc[test_index, 'Price']

    X_test = test[FEATURES].reset_index(drop=True).copy()

    # INNER K-FOLD (TO PREVENT LEAKAGE WHEN USING PRICE)
    kf2 = KFold(n_splits=FOLDS, shuffle=True, random_state=42)   
    for j, (train_index2, test_index2) in enumerate(kf2.split(X_train)):
        print(f" ## INNER Fold {j+1} (Outer Fold {i+1}) ##")

        X_train2 = X_train.loc[train_index2, FEATURES + ['Price']].copy()
        X_valid2 = X_train.loc[test_index2, FEATURES].copy()

        ### FEATURE SET 1 (Uses Price) ###
        col = "Weight Capacity (kg)"
        tmp = X_train2.groupby(col).Price.agg(STATS)
        tmp.columns = [f"TE1_wc_{s}" for s in STATS]
        X_valid2 = X_valid2.merge(tmp, on=col, how="left")
        for c in tmp.columns:
            X_train.loc[test_index2, c] = X_valid2[c].values

        col = "original_Weight Capacity (kg)"
        tmp = X_train2.groupby(col).Price.agg(STATS)
        tmp.columns = [f"TE1_orig_wc_{s}" for s in STATS]
        X_valid2 = X_valid2.merge(tmp, on=col, how="left")
        for c in tmp.columns:
            X_train.loc[test_index2, c] = X_valid2[c].values

        ### FEATURE SET 2 (Uses Price) ###
        for col in COMBO:
            tmp = X_train2.groupby(col).Price.agg(STATS2)
            tmp.columns = [f"TE2_{col}_{s}" for s in STATS2]
            X_valid2 = X_valid2.merge(tmp, on=col, how="left")
            for c in tmp.columns:
                X_train.loc[test_index2, c] = X_valid2[c].values

    ### FEATURE SET 1 (Uses Price) ###
    col = "Weight Capacity (kg)"
    tmp = X_train.groupby(col).Price.agg(STATS)
    tmp.columns = [f"TE1_wc_{s}" for s in STATS]
    X_valid = X_valid.merge(tmp, on=col, how="left")
    X_test = X_test.merge(tmp, on=col, how="left")

    col = "original_Weight Capacity (kg)"
    tmp = X_train.groupby(col).Price.agg(STATS)
    tmp.columns = [f"TE1_orig_wc_{s}" for s in STATS]
    X_valid = X_valid.merge(tmp, on=col, how="left")
    X_test = X_test.merge(tmp, on=col, how="left")

    ### FEATURE SET 2 (Uses Price) ###
    for col in COMBO:
        tmp = X_train.groupby(col).Price.agg(STATS2)
        tmp.columns = [f"TE2_{col}_{s}" for s in STATS2]
        X_valid = X_valid.merge(tmp, on=col, how="left")
        X_test = X_test.merge(tmp, on=col, how="left")

    # Convert newly created columns to optimized dtypes
    X_train = optimize_dtypes(X_train)
    X_valid = optimize_dtypes(X_valid)
    X_test = optimize_dtypes(X_test)

    # CONVERT TO CATS SO XGBOOST RECOGNIZES THEM
    X_train[CATS] = X_train[CATS].astype("category")
    X_valid[CATS] = X_valid[CATS].astype("category")
    X_test[CATS] = X_test[CATS].astype("category")

    # DROP PRICE THAT WAS USED FOR TARGET ENCODING
    X_train = X_train.drop(['Price'], axis=1)

    # Convert to CuPy (for GPU acceleration)
    X_train = cp.asarray(X_train)
    X_valid = cp.asarray(X_valid)
    X_test = cp.asarray(X_test)

    y_train = cp.asarray(y_train)
    y_valid = cp.asarray(y_valid)

    # Convert to XGBoost DMatrix (GPU enabled)
    dtrain = xgb.DMatrix(X_train, label=y_train, nthread=-1)
    dvalid = xgb.DMatrix(X_valid, label=y_valid, nthread=-1)
    dtest = xgb.DMatrix(X_test, nthread=-1)

    # Set XGBoost parameters (GPU enabled)
    params = {
        "max_depth": 6,
        "colsample_bytree": 0.5,
        "subsample": 0.5,
        "learning_rate": 0.02,
        "min_child_weight": 10,
        "tree_method": "hist",
        "device":"cuda"
    }
    
    # Train the model
    evallist = [(dtrain, "train"), (dvalid, "valid")]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=10_000,  # Instead of n_estimators
        evals=evallist,
        early_stopping_rounds=100,
        verbose_eval=300
    )


    # Predict OOF and Test using DMatrix
    oof[test_index] = model.predict(dvalid)
    pred += model.predict(dtest)

pred /= FOLDS


# COMPUTE OVERALL CV SCORE
true = train.Price.values
s = np.sqrt(np.mean( (oof-true)**2.0 ) )
print(f"=> Overall CV Score = {s}")


# SAVE OOF TO DISK FOR ENSEMBLES
VER = 2
np.save(f"oof_v{VER}",oof)
print("Saved oof to disk")


print(f"\nIn total, we used {dtrain.num_col()} features, Wow!\n")


import xgboost as xgb
fig, ax = plt.subplots(figsize=(10, 20))
xgb.plot_importance(model, importance_type='gain',ax=ax)
plt.title("Top Feature Importances (XGBoost)")
plt.show()


sub = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
sub.Price = pred
sub.to_csv(f"submission_v{VER}.csv",index=False)
sub.head()


plt.figure(figsize=(6,4))
plt.hist(sub.Price,bins=100)
plt.title("Test Predictions")
plt.show()




