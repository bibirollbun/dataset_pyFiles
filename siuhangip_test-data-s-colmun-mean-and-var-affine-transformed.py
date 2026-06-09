import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print("numpy version: ", np.__version__)
print("pandas version: ", pd.__version__)
df = pd.read_parquet('drw-crypto-market-prediction/train.parquet')
df_test = pd.read_parquet('drw-crypto-market-prediction/test.parquet')


print(df.dtypes.value_counts())
print(df_test.dtypes.value_counts())


# Find columns with NaN and their percentage
for col in df.columns:
   if df[col].isna().any():
       nan_count = df[col].isna().sum()
       total_count = len(df[col])
       nan_percentage = (nan_count / total_count) * 100
       print(f"Column '{col}': {nan_percentage:.2f}% NaN ({nan_count}/{total_count})")

# Find columns with NaN and their percentage in test set
for col in df_test.columns:
    if df_test[col].isna().any():
        nan_count = df_test[col].isna().sum()
        total_count = len(df_test[col])
        nan_percentage = (nan_count / total_count) * 100
        print(f"Column '{col}': {nan_percentage:.2f}% NaN ({nan_count}/{total_count})")


print("\nInf values in train set:")
# Find columns with inf and their percentage
for col in df.columns:
   if np.isinf(df[col]).any():
       inf_count = np.isinf(df[col]).sum()
       total_count = len(df[col])
       inf_percentage = (inf_count / total_count) * 100
       print(f"Column '{col}': {inf_percentage:.2f}% inf ({inf_count}/{total_count})")

print("\nInf values in test set:")
# Find columns with inf and their percentage in test set
for col in df_test.columns:
    if np.isinf(df_test[col]).any():
        inf_count = np.isinf(df_test[col]).sum()
        total_count = len(df_test[col])
        inf_percentage = (inf_count / total_count) * 100
        print(f"Column '{col}': {inf_percentage:.2f}% inf ({inf_count}/{total_count})")


df.describe()


df_test.describe()


def plot_x_columns(df, title_suffix=''):
    # Filter columns whose names start with 'X'
    x_cols = df.columns[df.columns.str.startswith('X')]

    # Calculate all statistics for these columns
    x_means = df[x_cols].mean()
    x_vars = df[x_cols].var()  # Changed back to variance
    x_ranges = df[x_cols].max() - df[x_cols].min()
    
    # Quantiles
    x_q0 = df[x_cols].min()  # 0th percentile (minimum)
    x_q25 = df[x_cols].quantile(0.25)  # 25th percentile
    x_q50 = df[x_cols].median()  # 50th percentile (median)
    x_q75 = df[x_cols].quantile(0.75)  # 75th percentile
    x_q100 = df[x_cols].max()  # 100th percentile (maximum)
    
    # IQR
    x_iqr = x_q75 - x_q25
    
    # Second moment (variance)
    x_second_moment = df[x_cols].var()
    
    # Third moment (for skewness calculation)
    x_third_moment = df[x_cols].apply(lambda x: ((x - x.mean()) ** 3).mean())
    
    # Fourth moment (for kurtosis calculation)
    x_fourth_moment = df[x_cols].apply(lambda x: ((x - x.mean()) ** 4).mean())

    # Create colors based on column position (red to blue)
    num_cols = len(x_cols)
    colors = range(num_cols)

    # Create figure with single column layout (now 12 subplots)
    fig, axs = plt.subplots(12, 1, figsize=(10, 32))

    # Plot means
    sc = axs[0].scatter(range(num_cols), x_means.values, c=colors, cmap='coolwarm')
    axs[0].set_title(f'Mean - {title_suffix}')
    axs[0].set_ylabel('Mean Value')
    axs[0].set_ylim(-1, 1)
    fig.colorbar(sc, ax=axs[0])

    # Plot variances
    sc = axs[1].scatter(range(num_cols), x_vars.values, c=colors, cmap='coolwarm')
    axs[1].set_title(f'Variance - {title_suffix}')
    axs[1].set_ylabel('Variance')
    axs[1].set_ylim(0, 2)
    fig.colorbar(sc, ax=axs[1])

    # Plot ranges
    sc = axs[2].scatter(range(num_cols), x_ranges.values, c=colors, cmap='coolwarm')
    axs[2].set_title(f'Range - {title_suffix}')
    axs[2].set_ylabel('Range')
    axs[2].set_ylim(0, 300)
    fig.colorbar(sc, ax=axs[2])

    # Plot Q0 (min)
    sc = axs[3].scatter(range(num_cols), x_q0.values, c=colors, cmap='coolwarm')
    axs[3].set_title(f'Q0 (Min) - {title_suffix}')
    axs[3].set_ylabel('Q0')
    axs[3].set_ylim(-160, 10)
    fig.colorbar(sc, ax=axs[3])

    # Plot Q25
    sc = axs[4].scatter(range(num_cols), x_q25.values, c=colors, cmap='coolwarm')
    axs[4].set_title(f'Q25 - {title_suffix}')
    axs[4].set_ylabel('Q25')
    axs[4].set_ylim(-2, 1)
    fig.colorbar(sc, ax=axs[4])

    # Plot Q50 (median)
    sc = axs[5].scatter(range(num_cols), x_q50.values, c=colors, cmap='coolwarm')
    axs[5].set_title(f'Q50 (Median) - {title_suffix}')
    axs[5].set_ylabel('Q50')
    axs[5].set_ylim(-1, 1)
    fig.colorbar(sc, ax=axs[5])

    # Plot Q75
    sc = axs[6].scatter(range(num_cols), x_q75.values, c=colors, cmap='coolwarm')
    axs[6].set_title(f'Q75 - {title_suffix}')
    axs[6].set_ylabel('Q75')
    axs[6].set_ylim(-0.75, 1.5)
    fig.colorbar(sc, ax=axs[6])

    # Plot Q100 (max)
    sc = axs[7].scatter(range(num_cols), x_q100.values, c=colors, cmap='coolwarm')
    axs[7].set_title(f'Q100 (Max) - {title_suffix}')
    axs[7].set_ylabel('Q100')
    axs[7].set_ylim(-1, 300)
    fig.colorbar(sc, ax=axs[7])

    # Plot IQR
    sc = axs[8].scatter(range(num_cols), x_iqr.values, c=colors, cmap='coolwarm')
    axs[8].set_title(f'IQR - {title_suffix}')
    axs[8].set_ylabel('IQR')
    axs[8].set_ylim(-0.5, 2.5)
    fig.colorbar(sc, ax=axs[8])

    # Plot Second Moment
    sc = axs[9].scatter(range(num_cols), x_second_moment.values, c=colors, cmap='coolwarm')
    axs[9].set_title(f'Second Moment - {title_suffix}')
    axs[9].set_ylabel('Second Moment')
    axs[9].set_ylim(0, 2.5)
    fig.colorbar(sc, ax=axs[9])

    # Plot Third Moment
    sc = axs[10].scatter(range(num_cols), x_third_moment.values, c=colors, cmap='coolwarm')
    axs[10].set_title(f'Third Moment - {title_suffix}')
    axs[10].set_ylabel('Third Moment')
    axs[10].set_ylim(-10, 300)
    fig.colorbar(sc, ax=axs[10])

    # Plot Fourth Moment
    sc = axs[11].scatter(range(num_cols), x_fourth_moment.values, c=colors, cmap='coolwarm')
    axs[11].set_title(f'Fourth Moment - {title_suffix}')
    axs[11].set_ylabel('Fourth Moment')
    axs[11].set_ylim(0, 5000)
    fig.colorbar(sc, ax=axs[11])

    # Set x-axis labels for all plots
    for i in range(12):
        axs[i].set_xticks(range(num_cols))
        axs[i].set_xticklabels(x_cols, rotation=45)
        axs[i].set_xlabel('Column Index')

    # Layout so plots do not overlap
    fig.tight_layout()

    plt.show()


plot_x_columns(df, title_suffix='Training Set')


plot_x_columns(df_test, title_suffix='Test Set')


import numpy as np
from sklearn.linear_model import LinearRegression

def find_affine_transform(train_stats, test_stats, stat_name):
    """
    Find the best affine transformation: test = a * train + b
    """
    
    # Fit linear regression: test = a * train + b
    X = train_stats.values.reshape(-1, 1)
    y = test_stats.values
    
    reg = LinearRegression().fit(X, y)
    a = reg.coef_[0]  # slope
    b = reg.intercept_  # intercept
    r_squared = reg.score(X, y)
    
    # Calculate mean absolute error
    predicted_test = a * train_stats + b
    mae = np.mean(np.abs(test_stats - predicted_test))
    
    # Calculate correlation
    correlation = np.corrcoef(train_stats, test_stats)[0, 1]
    
    print(f"\n{stat_name} Affine Transform Analysis:")
    print(f"  Best affine transform: test = {a:.6f} * train + {b:.6f}")
    print(f"  R-squared: {r_squared:.6f}")
    print(f"  Correlation: {correlation:.6f}")
    print(f"  Mean Absolute Error: {mae:.6f}")
    
    # Interpret the transformation
    if abs(a + 1) < 0.01:  # Close to -1
        print(f"  → Near-perfect reflection around y = {b/2:.6f}")
    elif abs(a - 1) < 0.01:  # Close to 1
        print(f"  → Near-perfect shift by {b:.6f}")
    elif abs(a) < 0.01:  # Close to 0
        print(f"  → Near-constant value {b:.6f}")
    else:
        print(f"  → General linear scaling and shift")
    
    return a, b, r_squared, mae, correlation

def analyze_affine_transform(df_train, df_test, title_suffix=''):
    x_cols = df_train.columns[df_train.columns.str.startswith('X')]
    
    # Calculate statistics
    train_means = df_train[x_cols].mean()
    test_means = df_test[x_cols].mean()
    
    train_vars = df_train[x_cols].var()
    test_vars = df_test[x_cols].var()
    
    # Find affine transforms
    mean_a, mean_b, mean_r2, mean_mae, mean_corr = find_affine_transform(train_means, test_means, "Mean")
    var_a, var_b, var_r2, var_mae, var_corr = find_affine_transform(train_vars, test_vars, "Variance")
    
    # Plot the analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Mean affine transform plot
    ax1.scatter(train_means, test_means, alpha=0.7, label='Actual', s=50)
    x_range = np.linspace(train_means.min(), train_means.max(), 100)
    y_pred = mean_a * x_range + mean_b
    ax1.plot(x_range, y_pred, 'r-', linewidth=2, label=f'Fit: y = {mean_a:.3f}x + {mean_b:.3f}')
    ax1.plot(x_range, x_range, 'k--', alpha=0.5, label='y = x (identity)')
    ax1.set_xlabel('Train Mean')
    ax1.set_ylabel('Test Mean')
    ax1.set_title(f'Mean Affine Transform\nR² = {mean_r2:.6f}, r = {mean_corr:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Variance affine transform plot
    ax2.scatter(train_vars, test_vars, alpha=0.7, label='Actual', s=50)
    x_range = np.linspace(train_vars.min(), train_vars.max(), 100)
    y_pred = var_a * x_range + var_b
    ax2.plot(x_range, y_pred, 'r-', linewidth=2, label=f'Fit: y = {var_a:.3f}x + {var_b:.3f}')
    ax2.plot(x_range, x_range, 'k--', alpha=0.5, label='y = x (identity)')
    ax2.set_xlabel('Train Variance')
    ax2.set_ylabel('Test Variance')
    ax2.set_title(f'Variance Affine Transform\nR² = {var_r2:.6f}, r = {var_corr:.3f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'mean': {'a': mean_a, 'b': mean_b, 'r2': mean_r2, 'mae': mean_mae, 'correlation': mean_corr},
        'variance': {'a': var_a, 'b': var_b, 'r2': var_r2, 'mae': var_mae, 'correlation': var_corr}
    }

# Usage:
affine_results = analyze_affine_transform(df, df_test, 'Your Data')


def align_training_to_test(df_train, df_test):
    """
    Transform training data to match test data statistics
    Much better for deployment!
    """
    x_cols = df_train.columns[df_train.columns.str.startswith('X')]
    df_train_aligned = df_train.copy()
    
    for col in x_cols:
        # Target statistics (from TEST - the "reality")
        target_mean = df_test[col].mean()
        target_std = df_test[col].std()
        
        # Current statistics (from TRAIN - what we modify)
        current_mean = df_train[col].mean()
        current_std = df_train[col].std()
        
        # Transform train column to match test column's mean/std
        if current_std > 0:
            df_train_aligned[col] = ((df_train[col] - current_mean) / current_std) * target_std + target_mean
        else:
            df_train_aligned[col] = df_train[col] - current_mean + target_mean
    
    return df_train_aligned

# Transform training data instead
df_train_aligned = align_training_to_test(df, df_test)



plot_x_columns(df_train_aligned, title_suffix='Train Set Aligned')

