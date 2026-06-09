#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

def custom_score(y_true, y_pred, eps=1e-12):
    """Competition scoring function"""
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
        return {'score': 0, 'good_rate': good_rate, 'str': f"score=0.000 good_rate={good_rate:.3f}"}

    good_ape = ape[good_mask]
    mape = np.mean(good_ape)

    scaled_mape = mape / good_rate
    score = 1 - scaled_mape
    return {'score': score, 'good_rate': good_rate, 'str': f"score={score:.3f} good_rate={good_rate:.3f}"}

def month_str_to_time(month_str):
    """Convert month string to time integer - EXACT from improved_solution.py"""
    if '-' in month_str:
        year, month = month_str.split('-')
        year = int(year)
        month_map = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
    else:
        parts = month_str.split()
        year = int(parts[0])
        month_name = parts[1]
        month_map = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
    
    month_num = month_map[month_name if '-' not in month_str else month]
    time_int = (year - 2019) * 12 + month_num
    return time_int

def main():
    print("=== Kaggle Improved Real Estate Prediction Solution ===")
    
    # KAGGLE DATA PATHS
    TRAIN_PATH = "/kaggle/input/china-real-estate-demand-prediction/train"
    TEST_PATH = "/kaggle/input/china-real-estate-demand-prediction"
    
    # Load data with Kaggle paths
    print("Loading data from Kaggle input...")
    train_nht = pd.read_csv(f'{TRAIN_PATH}/new_house_transactions.csv')
    test = pd.read_csv(f'{TEST_PATH}/test.csv')
    
    print(f"âœ… Loaded {len(train_nht)} training records")
    print(f"âœ… Loaded {len(test)} test records")
    
    # Convert month strings to time integers - EXACT COPY
    print("Processing time data...")
    train_nht['time'] = train_nht['month'].apply(month_str_to_time)
    train_nht['sector_num'] = train_nht['sector'].str.extract(r'(\d+)').astype(int)
    
    print(f"Training time range: {train_nht['time'].min()} to {train_nht['time'].max()}")
    print(f"Test time range: {test['id'].str.split('_').str[0].apply(month_str_to_time).min()} to {test['id'].str.split('_').str[0].apply(month_str_to_time).max()}")
    
    # Create time series matrix - EXACT COPY
    amount_new_house_transactions = train_nht.pivot(
        index='time', columns='sector_num', values='amount_new_house_transactions'
    ).fillna(0)
    
    # Add missing sector 95 if not present
    if 95 not in amount_new_house_transactions.columns:
        amount_new_house_transactions[95] = 0
    # Ensure we have all sectors 1-96
    amount_new_house_transactions = amount_new_house_transactions.reindex(columns=range(1, 97), fill_value=0)
    
    print(f"Time series matrix shape: {amount_new_house_transactions.shape}")
    print(f"Sectors with all zeros: {(amount_new_house_transactions.sum(axis=0) == 0).sum()}")
    
    # Cross-validation with multiple strategies - EXACT COPY
    def evaluate_strategy(lookback_months, zero_check_months, use_geometric_mean=True):
        """Evaluate a prediction strategy using TimeSeriesSplit"""
        cv = TimeSeriesSplit(n_splits=4, test_size=12)
        true, oof = [], []
        
        for fold, (idx_tr, idx_va) in enumerate(cv.split(amount_new_house_transactions)):
            a_tr = amount_new_house_transactions.iloc[idx_tr]
            a_va = amount_new_house_transactions.iloc[idx_va]
            
            if use_geometric_mean:
                # Geometric mean for positive values, regular mean for others
                recent_data = a_tr.tail(lookback_months)
                positive_mask = recent_data > 0
                
                # Calculate predictions
                predictions = []
                for sector in recent_data.columns:
                    sector_data = recent_data[sector]
                    positive_values = sector_data[sector_data > 0]
                    
                    if len(positive_values) > 0:
                        # Geometric mean of positive values
                        pred_value = np.exp(np.log(positive_values).mean())
                    else:
                        pred_value = 0
                    
                    predictions.append(pred_value)
                
                predictions = np.array(predictions)
            else:
                # Simple mean
                predictions = a_tr.tail(lookback_months).mean(axis=0).values
            
            # Create prediction dataframe
            a_pred = pd.DataFrame(
                {time: predictions for time in idx_va}
            ).T
            a_pred.columns = amount_new_house_transactions.columns
            
            # Set to zero if any of the last zero_check_months were zero
            zero_sectors = a_tr.tail(zero_check_months).min(axis=0) == 0
            a_pred.loc[:, zero_sectors] = 0
            
            a_pred.index.rename('time', inplace=True)
            
            score_result = custom_score(a_va, a_pred)
            print(f"Fold {fold}: {score_result['str']}")
            
            true.append(a_va)
            oof.append(a_pred)
        
        overall_score = custom_score(pd.concat(true), pd.concat(oof))
        return overall_score, pd.concat(oof)
    
    print("\n=== Cross-Validation Results ===")
    
    # Test different strategies - EXACT COPY
    strategies = [
        (6, 6, True, "Geometric mean of last 6 months, zero if any of last 6 = 0"),
        (6, 3, True, "Geometric mean of last 6 months, zero if any of last 3 = 0"),  # CHAMPION
        (12, 6, True, "Geometric mean of last 12 months, zero if any of last 6 = 0"),
        (6, 6, False, "Simple mean of last 6 months, zero if any of last 6 = 0"),
    ]
    
    best_score = -1
    best_strategy = None
    best_params = None
    
    for lookback, zero_check, use_geo, description in strategies:
        print(f"\nTesting: {description}")
        score_result, _ = evaluate_strategy(lookback, zero_check, use_geo)
        print(f"Overall: {score_result['str']}")
        
        if score_result['score'] > best_score:
            best_score = score_result['score']
            best_strategy = (lookback, zero_check, use_geo)
            best_params = description
    
    print(f"\n=== Best Strategy ===")
    print(f"Strategy: {best_params}")
    print(f"Score: {best_score:.3f}")
    
    # Generate final predictions using best strategy - EXACT COPY
    print(f"\n=== Generating Final Predictions ===")
    lookback_months, zero_check_months, use_geometric_mean = best_strategy
    
    a_tr = amount_new_house_transactions
    
    if use_geometric_mean:
        recent_data = a_tr.tail(lookback_months)
        predictions = []
        
        for sector in recent_data.columns:
            sector_data = recent_data[sector]
            positive_values = sector_data[sector_data > 0]
            
            if len(positive_values) > 0:
                pred_value = np.exp(np.log(positive_values).mean())
            else:
                pred_value = 0
            
            predictions.append(pred_value)
        
        predictions = np.array(predictions)
    else:
        predictions = a_tr.tail(lookback_months).mean(axis=0).values
    
    # Create prediction dataframe for test period
    test_times = range(67, 79)  # Aug 2024 to Jul 2025
    a_pred = pd.DataFrame(
        {time: predictions for time in test_times}
    ).T
    a_pred.columns = amount_new_house_transactions.columns
    
    # Set to zero if any of the last zero_check_months were zero
    zero_sectors = a_tr.tail(zero_check_months).min(axis=0) == 0
    a_pred.loc[:, zero_sectors] = 0
    
    a_pred.index.rename('time', inplace=True)
    
    print(f"Prediction matrix shape: {a_pred.shape}")
    print(f"Sectors predicted as zero: {(a_pred == 0).all(axis=0).sum()}")
    print(f"Prediction range: {a_pred.values.min():.2f} to {a_pred.values.max():.2f}")
    print(f"Mean prediction: {a_pred.values.mean():.2f}")
    
    # Generate submission format - EXACT COPY
    submission_list = []
    
    for _, row in test.iterrows():
        test_id = row['id']
        month_str, sector_str = test_id.split('_')
        sector_num = int(sector_str.split()[1])
        time_val = month_str_to_time(month_str)
        
        if time_val in a_pred.index and sector_num in a_pred.columns:
            pred_value = a_pred.loc[time_val, sector_num]
        else:
            pred_value = 0
        
        submission_list.append({
            'id': test_id,
            'amount_new_house_transactions': pred_value  # KAGGLE EXPECTED FORMAT
        })
    
    submission = pd.DataFrame(submission_list)
    
    print(f"\nSubmission shape: {submission.shape}")
    print("First 10 predictions:")
    print(submission.head(10))
    
    # Save submission for Kaggle
    submission.to_csv('submission.csv', index=False)
    print("\nâœ… Submission saved as 'submission.csv' for Kaggle")
    
    print("\n=== Final Prediction Analysis ===")
    print(f"Zero predictions: {(submission['amount_new_house_transactions'] == 0).sum()}")
    print(f"Non-zero predictions: {(submission['amount_new_house_transactions'] > 0).sum()}")
    print("Prediction statistics:")
    print(submission['amount_new_house_transactions'].describe())
    
    print(f"\nğŸ�¯ Expected Score: ~0.548 (based on CV: {best_score:.3f})")
    print(f"ğŸ�† Champion Strategy: {best_params}")
    
    return submission

if __name__ == "__main__":
    submission = main()








