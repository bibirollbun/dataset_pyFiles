import pandas as pd
from sklearn.model_selection import KFold

def create_cv_folds_csv(data_path, n_folds=5, random_state=42, save_path='cv_folds.csv'):
    """
    Create consistent CV folds and save as CSV.
    """
    # Load the data to get the number of rows
    train_data = pd.read_parquet(data_path)
    train_data.dropna(inplace=True)
    n_samples = len(train_data)
    
    # Create KFold object
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Create a DataFrame with index and fold assignment
    fold_df = pd.DataFrame({
        'index': range(n_samples),
        'fold': -1
    })
    
    # Assign fold numbers
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(n_samples))):
        fold_df.loc[val_idx, 'fold'] = fold
    
    # Save as CSV
    fold_df.to_csv(save_path, index=False)
    
    print(f"Created {n_folds} folds:")
    for fold in range(n_folds):
        fold_size = len(fold_df[fold_df['fold'] == fold])
        print(f"Fold {fold}: {fold_size} samples")
    
    print(f"Fold assignments saved to: {save_path}")
    return fold_df


data_path = '/kaggle/input/hill-of-towie-wind-turbine-power-prediction/training_dataset.parquet'

create_cv_folds_csv(data_path=data_path)




