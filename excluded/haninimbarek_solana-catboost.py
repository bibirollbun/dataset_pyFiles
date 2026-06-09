import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedShuffleSplit
from catboost import CatBoostClassifier


df_train = pd.read_csv('/kaggle/input/solana-skill-sprint-memcoin-graduation/train.csv')
test = pd.read_csv('/kaggle/input/solana-skill-sprint-memcoin-graduation/test_unlabeled.csv')

df_train.drop(columns=['slot_graduated','mint','Unnamed: 0','is_valid'],inplace=True)
df_train['has_graduated'] = df_train['has_graduated'].astype(int)


df = df_train.copy()
target_col = 'has_graduated'


# Step 1: Separate classes
class_0 = df[df['has_graduated'] == 0]
class_1 = df[df['has_graduated'] == 1]

# Step 2: Count occurrences of each slot_min value in each class
count_0 = class_0['slot_min'].value_counts()
count_1 = class_1['slot_min'].value_counts()

# Step 3: Find common slot_min values between the two classes
common_slot_min = count_0.index.intersection(count_1.index)

# Step 4: Build a DataFrame to show counts in both classes
overlap_df = pd.DataFrame({
    'count_class_0': count_0[common_slot_min],
    'count_class_1': count_1[common_slot_min]
})

# Optional: Sort by count in class 1 or total
overlap_df['total'] = overlap_df['count_class_0'] + overlap_df['count_class_1']
overlap_df = overlap_df.sort_values(by='total', ascending=False)

# Display top overlaps
print(overlap_df.head(10))


# Remove overlapping slot_min values from the class with fewer or equal counts (prefer keeping majority class)
to_remove_indices = []

for slot_val, row in overlap_df.iterrows():
    if row['count_class_0'] >= row['count_class_1']:
        # Remove from class 1
        indices = df[(df['has_graduated'] == 1) & (df['slot_min'] == slot_val)].index
    else:
        # Remove from class 0
        indices = df[(df['has_graduated'] == 0) & (df['slot_min'] == slot_val)].index
    to_remove_indices.extend(indices)

# Drop selected indices
df = df.drop(index=to_remove_indices).reset_index(drop=True)

# Optional: View class balance
print(df['has_graduated'].value_counts())


def engineer_slot_min_features(data, slot_col='slot_min', bins=180):
    df = data.copy()

    # Frequency
    value_counts = df[slot_col].value_counts()
    df['slot_min_prob'] = df[slot_col].map(value_counts)
 
    # Gap (difference from previous value)
    df['slot_min_gap'] = df[slot_col].diff().fillna(0).astype(int)

    # Multiply frequency by adjusted gap
    df['slot_min_score'] = df['slot_min_prob'] * df['slot_min_gap']

    # CDF
    cdf_values = df[slot_col].rank(method='average') / len(df)
    df['slot_min_cdf'] = cdf_values

    # PDF (density estimate)
    hist, bin_edges = np.histogram(df[slot_col], bins='sturges', density=True)
    bin_indices = np.digitize(df[slot_col], bins=bin_edges, right=True)
    pdf_values = [hist[i - 1] if 0 < i <= len(hist) else 0 for i in bin_indices]
    df['slot_min_pdf'] = pdf_values

    # slot_min_cdf Ã— slot_min_prob
    df['cdf_prob'] = df['slot_min_cdf'] * df['slot_min_prob']

    # slot_min_pdf Ã— slot_min_gap
    df['pdf_gap'] = df['slot_min_pdf'] * df['slot_min_gap']

    # Fill any missing
    df = df.fillna(0)

    return df


df = engineer_slot_min_features(df)
df_test = engineer_slot_min_features(test)


# Identify features automatically
target_col = 'has_graduated'

feature_cols = ['slot_min_prob', 'cdf_prob', 'slot_min_cdf', 'pdf_gap', 'slot_min_gap',
        'slot_min_score', 'slot_min']
               
strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Use indices for split
for train_idx, val_idx in strat_split.split(df, df[target_col]):
    train_data = df.loc[train_idx]
    val_data = df.loc[val_idx]

# Final split
X_train = train_data[feature_cols]
y_train = train_data[target_col]
X_val = val_data[feature_cols]
y_val = val_data[target_col]

# Show result
print("âœ… Stratified + Temporal-aware split complete.")
print(f"Train: {X_train.shape} | Pos: {y_train.sum()} | Neg: {(y_train==0).sum()}")
print(f"Val:   {X_val.shape} | Pos: {y_val.sum()} | Neg: {(y_val==0).sum()}")


# Initialize CatBoost (silent mode)
cat_model = CatBoostClassifier(
    loss_function='Logloss',
    eval_metric='Logloss',
    random_state=42, 
    verbose=0, 
    bagging_temperature = 0,
    random_strength = 1,
    depth =  4, 
    learning_rate = 0.1,
    l2_leaf_reg =  1,
    iterations = 1000,
    border_count = 32
)

# Train on your feature set
cat_model.fit(X_train, y_train)

# Predict and evaluate
train_pred_cb = cat_model.predict_proba(X_train)[:, 1]
val_pred_cb = cat_model.predict_proba(X_val)[:, 1]

print("ðŸ“Š CatBoost Train Log Loss:", log_loss(y_train, train_pred_cb))
print("ðŸ“Š CatBoost Validation Log Loss:", log_loss(y_val, val_pred_cb))


X_test = df_test[feature_cols]


cat_pred = cat_model.predict_proba(X_test)[:,1]


# Create submission DataFrame
submission = pd.DataFrame({
    'mint': df_test['mint'],  
    'has_graduated':cat_pred
})

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("âœ… Submission file saved as 'submission.csv'")
submission.head()

