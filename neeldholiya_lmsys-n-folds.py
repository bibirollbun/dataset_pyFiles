import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.filterwarnings('ignore')



train = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/train.csv')

print("Original Train Shape:", train.shape)



print("\n" + "="*80)
print("TARGET DISTRIBUTION")
print("="*80)
train['winner'] = train.apply(
    lambda row: 0 if row['winner_model_a'] == 1 
    else (1 if row['winner_model_b'] == 1 else 2),
    axis=1
)

print(train['winner'].value_counts().sort_index())
print("\nPercentage Distribution:")
print(train['winner'].value_counts(normalize=True).sort_index() * 100)



n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

train['fold'] = -1

for fold, (train_idx, val_idx) in enumerate(skf.split(train, train['winner'])):
    train.loc[val_idx, 'fold'] = fold



print("\n" + "="*80)
print("FOLD DISTRIBUTION")
print("="*80)

for fold in range(n_splits):
    fold_data = train[train['fold'] == fold]
    print(f"\nFold {fold}:")
    print(f"  Total Samples: {len(fold_data)}")
    print(f"  Winner Distribution:")
    
    for winner in range(3):
        count = (fold_data['winner'] == winner).sum()
        pct = count / len(fold_data) * 100
        winner_name = ['Model A', 'Model B', 'Tie'][winner]
        print(f"    {winner_name}: {count} ({pct:.2f}%)")



print("\n" + "="*80)
print("FOLD VALIDATION")
print("="*80)

for fold in range(n_splits):
    val_data = train[train['fold'] == fold]
    train_data = train[train['fold'] != fold]
    
    print(f"\nFold {fold}:")
    print(f"  Train Size: {len(train_data)}, Validation Size: {len(val_data)}")
    
    train_dist = train_data['winner'].value_counts(normalize=True).sort_index()
    val_dist = val_data['winner'].value_counts(normalize=True).sort_index()
    
    print(f"  Train Distribution: {train_dist.values}")
    print(f"  Val Distribution: {val_dist.values}")
    
    diff = np.abs(train_dist.values - val_dist.values).max()
    print(f"  Max Distribution Difference: {diff:.4f}")



output_path = '/kaggle/working/train_folds.csv'
train.to_csv(output_path, index=False)

print("\n" + "="*80)
print(f"SAVED: {output_path}")
print("="*80)



print("\n" + "="*80)
print("SAMPLE DATA WITH FOLDS")
print("="*80)

print(train[['id', 'prompt', 'winner_model_a', 'winner_model_b',
             'winner_tie', 'winner', 'fold']].head(10))

print("\n" + "="*80)
print("FOLD CREATION COMPLETED")
print("="*80)


