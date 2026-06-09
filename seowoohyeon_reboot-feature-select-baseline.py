import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime
import os
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import gc
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cycler
from collections import Counter
from matplotlib.colors import LinearSegmentedColormap
colors = ["#068D9D", "#53599A", "#607BB0", "#6D9DC5", "#77BECF", "#80DED9", "#AEECEF"]
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True, grid=True, prop_cycle=cycler('color', colors))
SEED=42


def reduce_mem_usage(dataframe,dataset):
    print("Reducing memory usage fo:",dataset)
    initial_mem_usage=dataframe.memory_usage().sum()/1024**2
    for col in dataframe.columns:
        col_type=dataframe[col].dtype
        c_min=dataframe[col].min()
        c_max=dataframe[col].max()
        if str(col_type)[:3]=='int':
            if c_min>np.iinfo(np.int8).min and c_max<np.iinfo(np.int8).max:
                dataframe[col]=dataframe[col].astype(np.int8)
            elif c_min>np.iinfo(np.int16).min and c_max<np.iinfo(np.int16).max:
                dataframe[col]=dataframe[col].astype(np.int16)
            elif c_min>np.iinfo(np.int32).min and c_max<np.iinfo(np.int32).max:
                dataframe[col]=fataframe[col].astype(np.int32)
            elif c_min>np.iinfo(np.int64).min and c_max<np.iinfo(np.int64).max:
                dataframe[col]=dataframe[col].astype(np.int64)
        else:
            if c_min>np.finfo(np.float16).min and c_min<np.finfo(np.float16).max:
                dataframe[col]=dataframe[col].astype(np.float16)
            elif c_min>np.finfo(np.float32).min and c_min<np.finfo(np.float32).max():
                dataframe[col]=dataframe[col].astype(np.float32)
            else:
                dataframe[col]=dataframe[col].astype(np.float64)
    final_mem_usage=dataframe.memory_usage().sum()/1024**2
    print("--memory usage before: {:.2f}MB".format(initial_mem_usage))
    print("--memory usage after: {:.2f}MB".format(final_mem_usage))
    print("--decreased memory usage by {:.2f}MB%\n".format(100*(initial_mem_usage-final_mem_usage)/initial_mem_usage))
    return dataframe



train_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
train_df=reduce_mem_usage(train_df,"train")
train_df=train_df.reset_index()
test_df=pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')
TARGET=train_df['label']


def plot_feature_trend_by_split_range_from_df(df, subset_features, target='label',
                                              top_n=200, start_split=6, end_split=10, min_freq=4):
    """
    Given a DataFrame, finds features with high correlation to the target in each split,
    and visualizes features that appear at least `min_freq` times within the specified split range.
    
    Parameters:
    - df: The entire DataFrame
    - subset_features: List of features to use (e.g., ['X1', 'X2', ...])
    - target: Name of the target variable
    - top_n: Number of top features to select based on correlation in each split
    - start_split: Starting split number
    - end_split: Ending split number
    - min_freq: Minimum number of appearances within the split range
    
    Returns:
    - List of features that meet the specified condition
    """
    from collections import Counter
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    n_splits = 10
    split_size = len(df) // n_splits
    split_corr_dict = {}

    # Splitë³„ë¡œ ìƒ�ê´€ê³„ìˆ˜ ë†’ì�€ top_n í”¼ì²˜ ê³„ì‚°
    for i in range(n_splits):
        start_idx = i * split_size
        if i == n_splits - 1:
            df_split = df.iloc[start_idx:]
        else:
            df_split = df.iloc[start_idx:start_idx + split_size]

        corr_list = []
        for col in subset_features:
            if df_split[col].nunique() > 1:
                corr = df_split[[col, target]].corr().iloc[0, 1]
                corr_list.append((col, abs(corr)))
            else:
                corr_list.append((col, np.nan))

        corr_series = pd.Series({k: v for k, v in corr_list if not np.isnan(v)})
        top_features = corr_series.sort_values(ascending=False).head(top_n)
        split_corr_dict[f'Split_{i+1}'] = top_features

    # ì „ì²´ í”¼ì²˜ ëª©ë¡�
    all_top_features = set()
    for s in split_corr_dict.values():
        all_top_features.update(s.index.tolist())
    all_top_features = list(all_top_features)

    # Split x Feature ìƒ�ê´€ê³„ìˆ˜ í…Œì�´ë¸”
    corr_trend_df = pd.DataFrame(index=[f'Split_{i+1}' for i in range(n_splits)], columns=all_top_features)
    for split_name, corr_series in split_corr_dict.items():
        for feature in all_top_features:
            corr_trend_df.loc[split_name, feature] = corr_series.get(feature, np.nan)
    corr_trend_df = corr_trend_df.astype(float)

    # Step 1: ì§€ì •ë�œ split ë²”ìœ„ ë¦¬ìŠ¤íŠ¸ ìƒ�ì„±
    target_splits = [f'Split_{i}' for i in range(start_split, end_split + 1)]

    # Step 2: í”¼ì²˜ ë“±ì�¥ íšŸìˆ˜ ê³„ì‚°
    partial_counter = Counter()
    for split_name in target_splits:
        partial_counter.update(split_corr_dict[split_name].index.tolist())

    # Step 3: ì¡°ê±´ì—� ë§�ëŠ” í”¼ì²˜ í•„í„°ë§�
    selected_features = [f for f, count in partial_counter.items() if count >= min_freq]

    # ê²°ê³¼ ì¶œë ¥
    print(f"\nğŸ“Œ Splits {start_split}~{end_split}ì—�ì„œ {min_freq}ë²ˆ ì�´ìƒ� ë“±ì�¥í•œ í”¼ì²˜ ìˆ˜: {len(selected_features)}")
    print(" ì˜ˆì‹œ í”¼ì²˜ë“¤:", selected_features[:])

    # Step 4: ì‹œê°�í™”
    x = list(range(start_split, end_split + 1))
    plt.figure(figsize=(15, 8))
    for feature in selected_features:
        plt.plot(x, corr_trend_df.loc[[f'Split_{i}' for i in x], feature], marker='o', label=feature)

    plt.xticks(x, [f'Split_{i}' for i in x], rotation=45)
    plt.xlabel('Data Split')
    plt.ylabel('Absolute Correlation with Target')
    plt.title(f'Feature Correlation with Target (Splits {start_split} to {end_split})\n(Features appeared â‰¥ {min_freq} times)')
    #plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    del corr_trend_df
    del split_corr_dict
    del all_top_features
    del target_splits
    del partial_counter
    gc.collect()


    return selected_features

def remove_highly_correlated_features(df, features, threshold=0.9):
    """
    Removes features that are highly correlated with others based on the specified threshold.

    Parameters:
    - df: The input DataFrame containing the features.
    - features: A list of feature column names to evaluate.
    - threshold: Correlation threshold above which one of the features will be removed (default is 0.9).

    Returns:
    - A list of selected features with high-correlation features removed.
    """
    # Compute absolute correlation matrix
    corr_matrix = df[features].corr().abs()

    # Get the upper triangle of the correlation matrix (excluding self-correlations)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Identify features to drop: any feature with correlation > threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    print(f"ğŸ§¹ Removed {len(to_drop)} features due to high correlation (>{threshold})")

    # Explicitly delete large objects
    del corr_matrix, upper
    gc.collect()

    # Return filtered feature list
    return [f for f in features if f not in to_drop]
    
def generate_interaction_features(df, base_features, selected_features, eps=1e-3):
    """
    Create derived interaction features between selected_features and base_features using:
    - Multiplication
    - Division (with epsilon to avoid division by zero)
    - Addition
    - Subtraction

    Parameters:
    - df: Input DataFrame
    - base_features: List of all base features (e.g., X1 ~ X780)
    - selected_features: List of important features to combine with others
    - eps: Small constant to prevent division by zero (default=1e-3)

    Returns:
    - df_new: DataFrame containing the new derived features only
    """
    
    new_feature_dict = {}

    for sel in selected_features:
        for base in base_features:
            if sel == base:
                continue

            # Define new feature names
            new_feature_dict[f'{sel}_mul_{base}'] = df[sel] * df[base]
            #new_feature_dict[f'{sel}_div_{base}'] = df[sel] / (df[base] + eps)
            #new_feature_dict[f'{sel}_add_{base}'] = df[sel] + df[base]

    # Combine all columns at once
    df_new = pd.concat(new_feature_dict, axis=1)

    print(f" Generated {df_new.shape[1]} features from {len(selected_features)} Ã— {len(base_features)} combinations.")
    return df_new

def create_interaction_features(df, feature_list, eps=1e-3):
    """
    Given a DataFrame and a list of interaction feature names like 'X219_mul_X751',
    create these features by performing the indicated operations on the columns.

    Parameters:
    - df: Input DataFrame
    - feature_list: List of interaction feature names (e.g. 'X219_mul_X751')
    - eps: Small constant to avoid division by zero

    Returns:
    - DataFrame with the new interaction features
    """
    import numpy as np
    df_new = pd.DataFrame(index=df.index)

    for feat in feature_list:
        if '_mul_' in feat:
            left, right = feat.split('_mul_')
            if left in df.columns and right in df.columns:
                df_new[feat] = df[left] * df[right]

        elif '_div_' in feat:
            left, right = feat.split('_div_')
            if left in df.columns and right in df.columns:
                df_new[feat] = df[left] / (df[right] + eps)

        elif '_add_' in feat:
            left, right = feat.split('_add_')
            if left in df.columns and right in df.columns:
                df_new[feat] = df[left] + df[right]

        elif '_sub_' in feat:
            left, right = feat.split('_sub_')
            if left in df.columns and right in df.columns:
                df_new[feat] = df[left] - df[right]

    return df_new


from tqdm import tqdm

subset_features = [f"X{i}" for i in range(1, 781)]
batch_size = 100

# ë°°ì¹˜ë³„ë¡œ ì œê±° í›„ ì‚´ì•„ë‚¨ì�€ í”¼ì²˜ ì €ì�¥
kept_features_total = []

for i in tqdm(range(0, len(subset_features), batch_size)):
    batch_feats = subset_features[i:i+batch_size]
    kept_feats = remove_highly_correlated_features(train_df, batch_feats, 0.95)
    kept_features_total.extend(kept_feats)

# ìµœì¢… í”¼ì²˜ë“¤ë¡œ DataFrame êµ¬ì„±
train_df_filtered = train_df[kept_features_total + ['label']]  # íƒ€ê²Ÿ í�¬í•¨
print(f" ìµœì¢… ì‚¬ìš© í”¼ì²˜ ìˆ˜: {len(kept_features_total)}")
print(train_df_filtered.shape)


subset_features = [col for col in train_df_filtered.columns if col.startswith('X')]

selected_features = plot_feature_trend_by_split_range_from_df(
    df=train_df_filtered,
    subset_features=subset_features,
    target='label',
    top_n=120,
    start_split=6,
    end_split=10,
    min_freq=4
)


selected_features2 = plot_feature_trend_by_split_range_from_df(
    df=train_df_filtered,
    subset_features=subset_features,
    target='label',
    top_n=180,
    start_split=1,
    end_split=10,
    min_freq=9
)


Features=list(set(selected_features+selected_features2))
base_feature=remove_highly_correlated_features(train_df_filtered, Features)
test_df_filtered = test_df[base_feature].copy()
df_new_features = generate_interaction_features(
    train_df_filtered,              
    subset_features[::4],          
    base_feature                   
)
df_new_features = pd.concat([df_new_features, TARGET], axis=1, join='inner')
subset_features = [col for col in df_new_features.columns if col.startswith('X')]


subset_features_4_multiple = df_new_features.columns[::4].tolist()

selected_features3 = plot_feature_trend_by_split_range_from_df(
    df=df_new_features,
    subset_features=subset_features_4_multiple,
    target='label',
    top_n=150,
    start_split=1,
    end_split=10,
    min_freq=8
)
selected_features


derived_feature=remove_highly_correlated_features(df_new_features, selected_features3, 0.95)
df_new_test_features = create_interaction_features(test_df, derived_feature)


additional_features = [
    "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume",
    "X752", "X287", "X298", "X759", "X302", "X55", "X56",
    "X52", "X303", "X51", "X598", "X385", "X603", "X674",
    "X415", "X345", "X174", "X178", "X168", "X612",
]

# ê¸°ì¡´ ë¦¬ìŠ¤íŠ¸ + ì¶”ê°€ ë¦¬ìŠ¤íŠ¸ë¥¼ í•©ì¹œ í›„ setìœ¼ë¡œ ì¤‘ë³µ ì œê±°
base_feature = list(set(base_feature + additional_features))
test_df = pd.concat([test_df[base_feature], df_new_test_features], axis=1)



from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from scipy.stats import pearsonr
import numpy as np
import pandas as pd

# 2. 3-Fold Cross Validation í•™ìŠµ ë°� í�‰ê°€ (Pearson correlation ì‚¬ìš©)
kf = KFold(n_splits=3, shuffle=False)
pearson_list = []
fold = 1

for train_idx, val_idx in kf.split(train_df):
    X_train = pd.concat([
        train_df.iloc[train_idx][base_feature].reset_index(drop=True),
        df_new_features.iloc[train_idx][derived_feature].reset_index(drop=True)
    ], axis=1)
    y_train = TARGET.iloc[train_idx].reset_index(drop=True)

    X_val = pd.concat([
        train_df.iloc[val_idx][base_feature].reset_index(drop=True),
        df_new_features.iloc[val_idx][derived_feature].reset_index(drop=True)
    ], axis=1)
    y_val = TARGET.iloc[val_idx].reset_index(drop=True)

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=3,
        gamma=1,
        subsample=0.2,
        colsample_bytree=0.7,
        reg_alpha=20,
        reg_lambda=20,
        random_state=42,
        tree_method='hist',
        verbosity=0
    )
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    pearson_corr, _ = pearsonr(y_val, val_preds)
    print(f"Fold {fold} Pearson Correlation: {pearson_corr:.4f}")
    pearson_list.append(pearson_corr)
    fold += 1

print(f"\nAverage Pearson Correlation over 3 folds: {np.mean(pearson_list):.4f}")

# 3. ìµœì¢… ëª¨ë�¸ ì „ì²´ ë�°ì�´í„°ë¡œ í•™ìŠµ
X_full = pd.concat([
    train_df[base_feature].reset_index(drop=True),
    df_new_features[derived_feature].reset_index(drop=True)
], axis=1)
y_full = TARGET.reset_index(drop=True)

final_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    min_child_weight=3,
    gamma=1,
    subsample=0.2,
    colsample_bytree=0.7,
    reg_alpha=20,
    reg_lambda=20,
    random_state=42,
    tree_method='hist',
    verbosity=0
)
final_model.fit(X_full, y_full)


# 5. ì˜ˆì¸¡ ë°� ì œì¶œ íŒŒì�¼ ìƒ�ì„±
test_preds = final_model.predict(test_df)

submission = pd.DataFrame({
    'ID': test_df.index,  # ì�¸ë�±ìŠ¤ë¥¼ idë¡œ ì‚¬ìš©
    'prediction': test_preds
})
submission.to_csv('submission.csv', index=False)
print("âœ… submission.csv saved!")


