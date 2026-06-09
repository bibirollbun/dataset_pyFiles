import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.filterwarnings('ignore')


# Load dataset
df = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/train.csv')
print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

# Create target variable from winner columns
def assign_winner(row):
    if row['winner_model_a'] == 1:
        return 0
    elif row['winner_model_b'] == 1:
        return 1
    else:
        return 2

df['winner'] = df.apply(assign_winner, axis=1)


# Display class distribution
print("\n" + "-" * 70)
print("CLASS DISTRIBUTION ANALYSIS")
print("-" * 70)

winner_counts = df['winner'].value_counts().sort_index()
winner_pcts = (df['winner'].value_counts(normalize=True).sort_index() * 100).round(2)

print("\nAbsolute counts:")
for idx, (count, pct) in enumerate(zip(winner_counts, winner_pcts)):
    label = ['Model A Wins', 'Model B Wins', 'Tie'][idx]
    print(f"  {label}: {count} ({pct}%)")


# Setup stratified k-fold cross-validation
NUM_FOLDS = 5
stratified_kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

# Initialize fold column
df['fold'] = -1

# Assign fold numbers
for fold_num, (_, val_indices) in enumerate(stratified_kfold.split(df, df['winner'])):
    df.loc[val_indices, 'fold'] = fold_num


# Verify fold distributions
print("\n" + "-" * 70)
print("STRATIFIED FOLD SUMMARY")
print("-" * 70)

for fold_num in range(NUM_FOLDS):
    fold_subset = df[df['fold'] == fold_num]
    print(f"\nFold {fold_num}: {len(fold_subset)} samples")
    
    for winner_class in range(3):
        class_count = (fold_subset['winner'] == winner_class).sum()
        class_pct = (class_count / len(fold_subset) * 100)
        class_label = ['Model A', 'Model B', 'Tie'][winner_class]
        print(f"  {class_label}: {class_count} ({class_pct:.2f}%)")


# Validate train/val split balance
print("\n" + "-" * 70)
print("TRAIN-VALIDATION SPLIT VERIFICATION")
print("-" * 70)

for fold_num in range(NUM_FOLDS):
    val_subset = df[df['fold'] == fold_num]
    train_subset = df[df['fold'] != fold_num]
    
    train_props = train_subset['winner'].value_counts(normalize=True).sort_index().values
    val_props = val_subset['winner'].value_counts(normalize=True).sort_index().values
    
    max_diff = np.abs(train_props - val_props).max()
    
    print(f"\nFold {fold_num}:")
    print(f"  Train: {len(train_subset)} | Val: {len(val_subset)}")
    print(f"  Train proportions: [{', '.join([f'{p:.4f}' for p in train_props])}]")
    print(f"  Val proportions: [{', '.join([f'{p:.4f}' for p in val_props])}]")
    print(f"  Max difference: {max_diff:.6f}")


# Save to CSV
output_file = '/kaggle/working/train_folds.csv'
df.to_csv(output_file, index=False)
print("\n" + "-" * 70)
print(f"File saved successfully: {output_file}")
print("-" * 70)

# Display sample records
print("\n" + "-" * 70)
print("PREVIEW OF PROCESSED DATA")
print("-" * 70)
print(df[['id', 'prompt', 'winner_model_a', 'winner_model_b', 
          'winner_tie', 'winner', 'fold']].head(10))

print("\n" + "-" * 70)
print("PROCESS COMPLETED SUCCESSFULLY")
print("-" * 70)

