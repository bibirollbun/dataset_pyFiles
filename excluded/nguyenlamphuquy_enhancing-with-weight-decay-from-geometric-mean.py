import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Style

from sklearn.model_selection import TimeSeriesSplit



def custom_score(y_true, y_pred, eps=1e-12):
    """Scoring function of the competition as defined on the competition overview page.
    
    Parameters:
    -----------
    y_true : array-like
    y_pred : array-like
    eps : float, optional (exact value doesn't matter)

    Return value:
    -------------
    dict with keys 'score', 'good_rate' and 'str'
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.size == 0:
        raise ValueError('empty array')

    if (y_true < 0).any():
        raise ValueError('negative y_true')

    if (~ np.isfinite(y_pred)).any():
        raise ValueError('infinite y_pred')

    ape = np.abs((y_true - y_pred) / np.maximum(y_true, eps))

    good_mask = ape <= 1.0
    good_rate = good_mask.mean()
    if good_rate < 0.7:
        return {'score': 0, 'good_rate': good_rate, 'str': f"{Fore.RED}score={0:.3f} {good_rate=:.3f}{Style.RESET_ALL}"}

    good_ape = ape[good_mask]
    mape = np.mean(good_ape)

    scaled_mape = mape / good_rate
    score = 1 - scaled_mape
    # score = max(0.0, score)
    return {'score': score, 'good_rate': good_rate, 'str': f"{score=:.3f} {good_rate=:.3f}"}


# We read all the data although this baseline notebook ignores most of it
# We convert the string-encoded months to integer values (time is 0..66 for train and 67..78 for test)

ci = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/city_indexes.csv') # one row per year
csi = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/city_search_index.csv') # several rows per training month
sp = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/sector_POI.csv') # at most one row per sector

train_lt = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/land_transactions.csv')
train_ltns = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/land_transactions_nearby_sectors.csv')
train_pht = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions.csv')
train_phtns = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions_nearby_sectors.csv')
train_nht = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions.csv')
train_nhtns = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions_nearby_sectors.csv')
test = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/test.csv')

month_codes = {
    'Jan': 1,
    'Feb': 2,
    'Mar': 3,
    'Apr': 4,
    'May': 5,
    'Jun': 6,
    'Jul': 7,
    'Aug': 8,
    'Sep': 9,
    'Oct': 10,
    'Nov': 11,
    'Dec': 12
}

test_id = test.id.str.split('_', expand=True)
test['month'] = test_id[0]
test['sector'] = test_id[1]
del test_id

for df in [train_lt, train_ltns, train_pht, train_phtns, train_nht, train_nhtns, csi, sp, test]:
    if df is not csi:
        df['sector_id'] = df.sector.str.slice(7, None).astype(int)
        # print(df.sector_id.min(), df.sector_id.max(), len(np.unique(df.sector_id)), len(df))
    if df is not sp:
        df['year'] = df.month.str.slice(0, 4).astype(int)
        df['month'] = df.month.str.slice(5, None).map(month_codes)
        df['time'] = (df['year'] - 2019) * 12 + df['month'] - 1 # min=0, max=66
        print(df['time'].min(), df['time'].max())



amount_new_house_transactions = train_nht.set_index(['time', 'sector_id']).amount_new_house_transactions.unstack()
# Missing values must be filled with zero:
amount_new_house_transactions = amount_new_house_transactions.fillna(0)
# We add sector 95, which has no transactions during the training period:
amount_new_house_transactions[95] = 0
amount_new_house_transactions = amount_new_house_transactions[np.arange(1, 97)]
amount_new_house_transactions.astype(int)


amount_pre_owned_house_transactions = train_pht.set_index(['time', 'sector_id']).amount_pre_owned_house_transactions.unstack()
# Missing values must be filled with zero:
amount_pre_owned_house_transactions = amount_pre_owned_house_transactions.fillna(0)

# Create full dataframe with all sectors 1-96, fill missing sectors with 0
full_sectors = np.arange(1, 97)
for sector in full_sectors:
    if sector not in amount_pre_owned_house_transactions.columns:
        amount_pre_owned_house_transactions[sector] = 0

# Reorder columns to match sector order 1-96
amount_pre_owned_house_transactions = amount_pre_owned_house_transactions[full_sectors]
amount_pre_owned_house_transactions.astype(int)



plt.title('Extrapolating a time series')
plt.plot(amount_new_house_transactions.sum(axis=1),
         color='b',
         label='total amount'
        )
plt.scatter(np.arange(11, 67, 12),
            amount_new_house_transactions.sum(axis=1).iloc[np.arange(11, 67, 12)],
            color='b',
            label='year-end peak')
plt.text(68, 2500000, '?', fontsize=96, color='b')
plt.xticks(np.arange(0, 80, 12))
plt.xlim(-2, 80)
plt.xlabel('time (months)')
plt.ylabel('Total amount_new_house_transactions')
plt.legend()
plt.show()


# WEIGHTED GEOMETRIC MEAN MODEL
def weighted_geometric_mean_model(
    n_lags=6,
    weight_type="linear",  # "linear", "exponential", "square"
    alpha=0.5,             # Parameter for exponential weights
    t2=6                   # Baseline condition check
):
    """
    Weighted Geometric Mean: weights increase over time
    
    Parameters:
    -----------
    weight_type : str
        "linear": w = [1, 2, 3, 4, 5, 6]
        "exponential": w = [alpha^5, alpha^4, alpha^3, alpha^2, alpha^1, alpha^0] 
        "square": w = [1^2, 2^2, 3^2, 4^2, 5^2, 6^2]
    alpha : float
        Parameter cho exponential weights (0 < alpha < 1)
    """
    
    # Generate weights
    if weight_type == "linear":
        weights = np.arange(1, n_lags + 1)  # [1, 2, 3, 4, 5, 6]
    elif weight_type == "exponential":
        weights = np.array([alpha**(n_lags-1-i) for i in range(n_lags)])  # [alpha^5, alpha^4, ..., alpha^0]
    elif weight_type == "square":
        weights = np.arange(1, n_lags + 1) ** 2  # [1, 4, 9, 16, 25, 36]
    else:
        raise ValueError("weight_type must be 'linear', 'exponential', or 'square'")
    
    # Normalize weights
    weights = weights / weights.sum()
    
    print(f"Weighted Geometric Mean Model:")
    print(f"Lag window: {n_lags} months")
    print(f"Weight type: {weight_type}")
    if weight_type == "exponential":
        print(f"Alpha: {alpha}")
    print(f"Weights: {weights.round(3)}")
    print(f"Baseline check: {t2} months")
    
    cv = TimeSeriesSplit(n_splits=4, test_size=12)
    true_results, oof_results = [], []
    
    for fold, (idx_tr, idx_va) in enumerate(cv.split(amount_new_house_transactions)):
        print(f"# Fold {fold}: train on months {idx_tr.min()}..{idx_tr.max()}, validate on months {idx_va.min()}..{idx_va.max()}")
        a_tr = amount_new_house_transactions.iloc[idx_tr]
        a_va = amount_new_house_transactions.iloc[idx_va]
        
        a_pred = pd.DataFrame(index=idx_va, columns=a_tr.columns, dtype=float)
        
        for sector in a_tr.columns:
            # Check baseline condition
            if (a_tr.tail(t2)[sector].min() == 0) or (a_tr[sector].sum() == 0):
                a_pred[sector] = 0
            else:
                # Get last n_lags values
                recent_vals = a_tr.tail(n_lags)[sector].values
                
                # Handle zeros and negative values
                if len(recent_vals) == n_lags and (recent_vals > 0).any():
                    # Only use positive values and corresponding weights
                    positive_mask = recent_vals > 0
                    positive_vals = recent_vals[positive_mask]
                    corresponding_weights = weights[positive_mask]
                    
                    if len(positive_vals) > 0:
                        # Renormalize weights for positive values
                        corresponding_weights = corresponding_weights / corresponding_weights.sum()
                        
                        # Weighted geometric mean
                        # Formula: (x1^w1 * x2^w2 * ... * xk^wk)^(1/sum(weights))
                        # In log space: exp(sum(wi * log(xi)) / sum(wi))
                        log_vals = np.log(positive_vals)
                        weighted_log_mean = np.sum(corresponding_weights * log_vals) / corresponding_weights.sum()
                        weighted_geom_mean = np.exp(weighted_log_mean)
                        
                        a_pred[sector] = weighted_geom_mean
                    else:
                        a_pred[sector] = 0
                else:
                    a_pred[sector] = 0
        
        a_pred.index.rename('time', inplace=True)
        score_result = custom_score(a_va, a_pred)
        print(f"# Fold {fold}: {score_result['str']}")
        
        true_results.append(a_va)
        oof_results.append(a_pred)
    
    overall_score = custom_score(pd.concat(true_results), pd.concat(oof_results))
    print(f"# Weighted Geometric Mean Overall: {overall_score['str']}")
    
    return true_results, oof_results, overall_score

# WEIGHTED GEOMETRIC MEAN EXPERIMENTS
print("WEIGHTED GEOMETRIC MEAN EXPERIMENTS:")
print("=" * 60)

experiments = []


# Manual calculation for equal weights
cv = TimeSeriesSplit(n_splits=4, test_size=12)
true_equal, oof_equal = [], []
for fold, (idx_tr, idx_va) in enumerate(cv.split(amount_new_house_transactions)):
    a_tr = amount_new_house_transactions.iloc[idx_tr]
    a_va = amount_new_house_transactions.iloc[idx_va]
    a_pred = pd.DataFrame(
        {time: np.exp(np.log(a_tr.tail(6)).mean(axis=0)) for time in idx_va}
    ).T
    a_pred.loc[:, a_tr.tail(6).min(axis=0) == 0] = 0
    a_pred.index.rename('time', inplace=True)
    true_equal.append(a_va)
    oof_equal.append(a_pred)
score_equal = custom_score(pd.concat(true_equal), pd.concat(oof_equal))
experiments.append(("Equal Weights", score_equal))

# Experiment: Square Weights
print("\nExperiment: Square Weights")
true_square, oof_square, score_square = weighted_geometric_mean_model(
    n_lags=6,
    weight_type="square"
)
experiments.append((f"Square", score_square))

# Experiment: Linear Weights
print("\nExperiment: Linear Weights")
true_linear, oof_linear, score_linear = weighted_geometric_mean_model(
    n_lags=6,
    weight_type="linear"
)
experiments.append((f"Linear", score_linear))


# Experiment: Exponential Weights
print("\nExperiment: Exponential Weights")
for alpha in [i/1000 for i in range(400,600,5)]:
    true_exp8, oof_exp8, score_exp8 = weighted_geometric_mean_model(
        n_lags=6,
        weight_type="exponential",
        alpha=alpha
    )
    experiments.append((f"Exponential alpha={alpha}", score_exp8))


# RESULTS SUMMARY
print("\nWEIGHTED GEOMETRIC MEAN RESULTS:")
print("=" * 70)
print("Rank | Method                   | Score  | Good Rate | vs Equal")
print("-" * 65)

# Sort by score
sorted_experiments = sorted(experiments, key=lambda x: x[1]['score'], reverse=True)
equal_score = score_equal['score']

for i, (name, score) in enumerate(sorted_experiments, 1):
    improvement = score['score'] - equal_score
    improvement_pct = improvement / equal_score * 100 if equal_score > 0 else 0
    print(f"{i:2d}   | {name:25s} | {score['score']:.3f} | {score['good_rate']:.3f} | {improvement:+.3f} ({improvement_pct:+.1f}%)")

# Find best method
best_name, best_score = sorted_experiments[0]
print(f"\nBEST METHOD: {best_name}")
print(f"Score improvement: {best_score['score'] - equal_score:+.3f}")
print(f"Score: {best_score['score']:.3f}")
print(f"Good rate: {best_score['good_rate']:.3f}")


# WEIGHTED GEOMETRIC MEAN FINAL SUBMISSION
import pandas as pd
import numpy as np

def generate_weighted_geom_final_submission(
    n_lags=6,
    weight_type="exponential",  
    alpha=0.5,                 # Best parameter from experiments
    t2=6,                      # Baseline condition check months
    filename_suffix="weighted_geom"
):
    """
    Generate final submission using exponential Weighted Geometric Mean with alpha = 0.5
    
    Parameters:
    -----------
    n_lags : int
        Number of months to use for geometric mean
    weight_type : str
        Type of weights ("exponential", "linear", "square")
    alpha : float
        Exponential decay parameter (0.5 is optimal)
    t2 : int
        Months to check for baseline condition
    """
    
    weights = np.array([alpha**(n_lags-1-i) for i in range(n_lags)])  
    
    
    # Normalize weights
    weights = weights / weights.sum()
    
    print(f"Weighted Geometric Mean Final Submission:")
    print(f"Lag window: {n_lags} months")
    print(f"Weight type: {weight_type}")
    print(f"Alpha: {alpha}")
    print(f"Weights: {weights.round(3)}")
    print(f"Baseline check: {t2} months")
    
    a_tr = amount_new_house_transactions
    a_pred_final = pd.DataFrame(index=np.arange(67, 79), columns=a_tr.columns, dtype=float)
    
    # Generate predictions for all sectors
    for sector in a_tr.columns:
        sector_data = a_tr[sector].values
        
        # Check baseline condition
        if (a_tr.tail(t2)[sector].min() == 0) or (sector_data.sum() == 0):
            a_pred_final[sector] = 0
            continue
        
        # Get last n_lags values for weighted geometric mean
        recent_vals = a_tr.tail(n_lags)[sector].values
        
        # Handle zeros and negative values
        if len(recent_vals) == n_lags and (recent_vals > 0).any():
            # Only use positive values và corresponding weights
            positive_mask = recent_vals > 0
            positive_vals = recent_vals[positive_mask]
            corresponding_weights = weights[positive_mask]
            
            if len(positive_vals) > 0:
                # Renormalize weights cho positive values
                corresponding_weights = corresponding_weights / corresponding_weights.sum()
                
                # Weighted geometric mean
                # Formula: exp(sum(wi * log(xi)) / sum(wi))
                log_vals = np.log(positive_vals)
                weighted_log_mean = np.sum(corresponding_weights * log_vals) / corresponding_weights.sum()
                weighted_geom_mean = np.exp(weighted_log_mean)
                
                # Apply same prediction to all test months (static prediction)
                a_pred_final[sector] = weighted_geom_mean
            else:
                a_pred_final[sector] = 0
        else:
            a_pred_final[sector] = 0
    
    a_pred_final.index.rename('time', inplace=True)
    
    # Create submission
    test['new_house_transaction_amount'] = a_pred_final.T.unstack().values
    
    # Generate filename
    if weight_type == "exponential":
        weight_suffix = f"_exp{alpha}"
    elif weight_type == "linear":
        weight_suffix = "_linear"
    elif weight_type == "square":
        weight_suffix = "_square"
    
    filename = f'submission_{filename_suffix}_lag{n_lags}{weight_suffix}.csv'
    test[['id', 'new_house_transaction_amount']].to_csv("/kaggle/working/submission.csv", index=False)
    
    return a_pred_final

# GENERATE BEST WEIGHTED GEOMETRIC MEAN SUBMISSION
print("GENERATING OPTIMAL WEIGHTED GEOMETRIC MEAN SUBMISSION:")
print("=" * 70)

# Generate with best configuration from experiments
final_weighted_predictions = generate_weighted_geom_final_submission(
    n_lags=6,
    weight_type="exponential",
    alpha=0.5,  # Best parameter from experiments
    t2=6,
    filename_suffix="best_weighted_geom"
)

