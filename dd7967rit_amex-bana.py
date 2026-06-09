import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split


# Load feather files
train_data = pd.read_feather('/kaggle/input/amexfeather/train_data.ftr')
test_data = pd.read_feather('/kaggle/input/amexfeather/test_data.ftr')


train_data_cols = train_data.columns.tolist()
#print("train data columns -- ")
#print(train_data_cols)
test_data_cols = train_data.columns.tolist()
#print("test data columns -- ")
#print(test_data_cols)


# ======================
# 1. Feature Aggregation
# ======================


# Define aggregations for key features
aggs = {
    'P_2': ['mean', 'last', 'max', 'min', 'std'],
    'D_39': ['sum', 'mean'],
    'B_1': ['mean', 'max', 'min'],
    'S_2': [lambda x: (x.max() - x.min()).days],
    'D_41': ['sum', 'mean'],
    'R_1': ['mean', 'max']
}

# Aggregate time-series data per customer
train_agg = train_data.groupby('customer_ID').agg(aggs)
test_agg = test_data.groupby('customer_ID').agg(aggs)

# Flatten multi-index columns
train_agg.columns = ['_'.join(col) for col in train_agg.columns]
test_agg.columns = ['_'.join(col) for col in test_agg.columns]


# ======================
# 2. Prepare Targets
# ======================
# Extract target from train data
y_train = train_data.groupby('customer_ID')['target'].first()


# ======================
# 3. Train/Validation Split
# ======================
from sklearn.model_selection import train_test_split

X_train, X_val, y_train_split, y_val = train_test_split(
    train_agg, 
    y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=42
)


# ======================
# 4. LightGBM Setup
# ======================
import lightgbm as lgb

# Sample weights for class imbalance
sample_weights = np.where(y_train_split == 0, 20, 1)

# Define custom metric (must match competition metric)
def amex_metric_lgb(y_true, y_pred):
    import numpy as np
    import pandas as pd

    df = pd.DataFrame({'target': y_true, 'prediction': y_pred})

    # 1. Calculate Default Rate at 4% (D)
    df = df.sort_values('prediction', ascending=False).reset_index(drop=True)
    df['weight'] = np.where(df['target'] == 0, 20, 1)
    four_pct_cutoff = int(0.04 * df['weight'].sum())
    df['weight_cumsum'] = df['weight'].cumsum()
    df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
    d = df_cutoff['target'].sum() / df['target'].sum()

    # 2. Calculate Normalized Gini Coefficient (G)
    df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
    total_pos = (df['target'] * df['weight']).sum()
    df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
    df['lorentz'] = df['cum_pos_found'] / total_pos
    df['gini'] = (df['lorentz'] - df['random']) * df['weight']
    g = df['gini'].sum()

    # 3. Normalize against perfect model
    g_max = weighted_gini(df['target'], df['target'])
    g_normalized = g / g_max if g_max != 0 else 0

    # 4. Combine metrics
    m = 0.5 * (g_normalized + d)

    return 'amex_metric', m, True

def weighted_gini(y_true, y_pred):
    import numpy as np
    import pandas as pd
    df = pd.DataFrame({'target': y_true, 'prediction': y_pred})
    df = df.sort_values('prediction', ascending=False)
    df['weight'] = np.where(df['target'] == 0, 20, 1)
    df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
    total_pos = (df['target'] * df['weight']).sum()
    df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
    df['lorentz'] = df['cum_pos_found'] / total_pos
    df['gini'] = (df['lorentz'] - df['random']) * df['weight']
    return df['gini'].sum()



# Model parameters
params = {
    'objective': 'binary',
    'metric': 'custom',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 127,
    'min_child_samples': 2400,
    'feature_fraction': 0.8,
    'bagging_freq': 1,
    'verbosity': -1
}

# Train model
model = lgb.LGBMClassifier(**params)
model.fit(
    X_train, y_train_split,
    sample_weight=sample_weights,
    eval_set=[(X_val, y_val)],
    eval_metric=amex_metric_lgb,
    callbacks=[lgb.early_stopping(100)]
)



# ======================
# 5. Generate Predictions
# ======================
# Test set predictions
test_preds = model.predict_proba(test_agg)[:, 1]

# Create submission
submission = pd.DataFrame({
    'customer_ID': test_agg.index,
    'prediction': test_preds
})
submission.to_csv('submission.csv', index=False)

