import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime
import os
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cycler
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


train_df.label


train_df['exp_855P289M125'] = np.exp(train_df['X855'] + train_df['X289'] - train_df['X125'])
train_df['868xexp_289M125'] = train_df['X868'] * np.exp(train_df['X289'] - train_df['X125'])
train_df['289xexp_289M125'] = train_df['X289'] * np.exp(train_df['X289'] - train_df['X125'])
train_df['exp_862P289M125'] = np.exp(train_df['X862'] + train_df['X289'] - train_df['X125'])
train_df['302xexp_289M125'] = train_df['X302'] * np.exp(train_df['X289'] - train_df['X125'])
train_df['exp_856P289M125'] = np.exp(train_df['X856'] + train_df['X289'] - train_df['X125'])
train_df['exp_868P855P289'] = np.exp(train_df['X868'] + train_df['X855'] + train_df['X289'])
train_df['exp_302P289M125'] = np.exp(train_df['X302'] + train_df['X289'] - train_df['X125'])
train_df['exp_289P855P21'] = np.exp(train_df['X289'] + train_df['X855'] + train_df['X21'])
train_df['385xexp_289M125'] = train_df['X385'] * np.exp(train_df['X289'] - train_df['X125'])
train_df['465x862x465'] = train_df['X465'] * train_df['X862'] * train_df['X465']
train_df['301xexp_289M125'] = train_df['X301'] * np.exp(train_df['X289'] - train_df['X125'])
train_df['exp_786P289M125'] = np.exp(train_df['X786'] + train_df['X289'] - train_df['X125'])
train_df['125x862x465'] = train_df['X125'] * train_df['X862'] * train_df['X465']
train_df['exp_603P863P153'] = np.exp(train_df['X603'] + train_df['X863'] + train_df['X153'])
train_df['exp_465P863P153'] = np.exp(train_df['X465'] + train_df['X863'] + train_df['X153'])
train_df['125x855x289'] = train_df['X125'] * train_df['X855'] * train_df['X289']
train_df['exp_598P868P855P289'] = np.exp(train_df['X598'] + train_df['X868'] + train_df['X855'] + train_df['X289'])
train_df['exp_862P868P855P289'] = np.exp(train_df['X862'] + train_df['X868'] + train_df['X855'] + train_df['X289'])
train_df['exp_860P868P855P289'] = np.exp(train_df['X860'] + train_df['X868'] + train_df['X855'] + train_df['X289'])
train_df['exp_612P868P855P289'] = np.exp(train_df['X612'] + train_df['X868'] + train_df['X855'] + train_df['X289'])
train_df['exp_868P855P289'] = np.exp(train_df['X868'] + train_df['X855'] + train_df['X289'])
train_df['852x868x855x289'] = train_df['X852'] * train_df['X868'] * train_df['X855'] * train_df['X289']
train_df['exp_174P868P855P289'] = np.exp(train_df['X174'] + train_df['X868'] + train_df['X855'] + train_df['X289'])
train_df['exp_465P868P855P289'] = np.exp(train_df['X465'] + train_df['X868'] + train_df['X855'] + train_df['X289'])
train_df['301x868x855x289'] = train_df['X301'] * train_df['X868'] * train_df['X855'] * train_df['X289']
train_df['302x868x855x289'] = train_df['X302'] * train_df['X868'] * train_df['X855'] * train_df['X289']
train_df['exp_168P868P855P289'] = np.exp(train_df['X168'] + train_df['X868'] + train_df['X855'] + train_df['X289'])
train_df['289x868x855x289'] = train_df['X289'] * train_df['X868'] * train_df['X855'] * train_df['X289']
train_df['855x868x855x289'] = train_df['X855'] * train_df['X868'] * train_df['X855'] * train_df['X289']
train_df['exp_603P868P855P289'] = np.exp(train_df['X603'] + train_df['X868'] + train_df['X855'] + train_df['X289'])
train_df['exp_856P868P855P289'] = np.exp(train_df['X856'] + train_df['X868'] + train_df['X855'] + train_df['X289'])
train_df['612x868x855x289'] = train_df['X612'] * train_df['X868'] * train_df['X855'] * train_df['X289']
train_df['168x868x855x289'] = train_df['X168'] * train_df['X868'] * train_df['X855'] * train_df['X289']
train_df['exp_125P862P289M125'] = np.exp(train_df['X125'] + train_df['X862'] + train_df['X289'] - train_df['X125'])
train_df['21x868x855x289'] = train_df['X21'] * train_df['X868'] * train_df['X855'] * train_df['X289']
train_df['868x868x855x289'] = train_df['X868'] * train_df['X868'] * train_df['X855'] * train_df['X289']


from collections import Counter
target = 'label'

subset_features = [
    "X863", "X856", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
    "X415", "X345", "X855", "X174", "X302", "X178", "X168", "X612",
    "buy_qty", "sell_qty", "volume", "X888", "X421", "X333",
    'X465', 'X153', 'X289', 'X125', 'X21',
    'exp_855P289M125',
    '868xexp_289M125',
    '289xexp_289M125',
    'exp_862P289M125',
    '302xexp_289M125',
    'exp_856P289M125',
    'exp_868P855P289',
    'exp_302P289M125',
    'exp_289P855P21',
    '385xexp_289M125',
    '465x862x465',
    '301xexp_289M125',
    'exp_786P289M125',
    '125x862x465',
    'exp_603P863P153',
    'exp_465P863P153',
    '125x855x289',
    'exp_598P868P855P289',
    'exp_862P868P855P289',
    'exp_860P868P855P289',
    'exp_612P868P855P289',
    'exp_868P855P289',
    '852x868x855x289',
    'exp_174P868P855P289',
    'exp_465P868P855P289',
    '301x868x855x289',
    '302x868x855x289',
    'exp_168P868P855P289',
    '289x868x855x289',
    '855x868x855x289',
    'exp_603P868P855P289',
    'exp_856P868P855P289',
    '612x868x855x289',
    '168x868x855x289',
    'exp_125P862P289M125',
    '21x868x855x289',
    '868x868x855x289'
]

n_splits = 10
split_size = len(train_df) // n_splits
split_corr_dict = {}

for i in range(n_splits):
    start_idx = i * split_size
    if i == n_splits - 1:
        df_split = train_df.iloc[start_idx:]
    else:
        df_split = train_df.iloc[start_idx:start_idx + split_size]

    # í•´ë‹¹ splitì—�ì„œ ìƒ�ê´€ê´€ê³„ ê³„ì‚°
    corr_matrix = df_split[subset_features + [target]].corr()
    target_corr = corr_matrix[target].drop(target).abs().sort_values(ascending=False)
    top50 = target_corr.head(50)
    split_corr_dict[f'Split_{i+1}'] = top50

# ëª¨ë“  splitì—�ì„œ ë‚˜ì˜¨ ìƒ�ìœ„ í”¼ì²˜ë“¤ ëª¨ìœ¼ê¸°
all_top_features = set()
for s in split_corr_dict.values():
    all_top_features.update(s.index.tolist())
all_top_features = list(all_top_features)

# ë¹ˆ DataFrame ìƒ�ì„± (indexëŠ” Split_1~Split_10, columnsëŠ” all_top_features)
corr_trend_df = pd.DataFrame(index=[f'Split_{i+1}' for i in range(n_splits)], columns=all_top_features)

# ìƒ�ê´€ê³„ìˆ˜ ê°’ ì±„ìš°ê¸°
for split_name, corr_series in split_corr_dict.items():
    for feature in all_top_features:
        value = corr_series.get(feature, np.nan)
        # valueê°€ Seriesë‚˜ list í˜•íƒœì�¼ ê²½ìš° NaNìœ¼ë¡œ ë³€ê²½
        if isinstance(value, (pd.Series, list, np.ndarray)):
            value = np.nan
        corr_trend_df.loc[split_name, feature] = value

# ìˆ«ì��í˜•ìœ¼ë¡œ ë³€í™˜ (NaN ìœ ì§€)
corr_trend_df = corr_trend_df.astype(float)

# ê°� featureê°€ ëª‡ ë²ˆ ë“±ì�¥í–ˆëŠ”ì§€ ì¹´ìš´íŠ¸
feature_counter = Counter()
for s in split_corr_dict.values():
    feature_counter.update(s.index.tolist())

# 10ë²ˆ ì�´ìƒ� ë“±ì�¥í•œ feature í•„í„°ë§�
top_features_by_frequency = [f for f, count in feature_counter.items() if count >= 10]

print(f"10ë²ˆ ì�´ìƒ� ë“±ì�¥í•œ feature ê°œìˆ˜: {len(top_features_by_frequency)}")

x = range(1, n_splits+1)
plt.figure(figsize=(15, 8))
for feature in top_features_by_frequency:
    plt.plot(x, corr_trend_df.loc[:, feature], marker='o', label=feature)

plt.xticks(x, corr_trend_df.index, rotation=45)
plt.xlabel('Data Split')
plt.ylabel('Absolute Correlation with Target')
plt.title('Feature Correlation with Target over 10 Data Splits')
plt.legend(loc='best', fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.show()


split_corr_dict


qualified_features = [f for f, count in feature_counter.items() if count == 10]
print(f"\nğŸ”¢ 10ê°œ ë¶„í• ì—�ì„œë§Œ ë“±ì�¥í•œ í”¼ì²˜ ìˆ˜: {len(qualified_features)}")
print(qualified_features)
qualified_features = [f for f, count in feature_counter.items() if count == 9]
print(f"\nğŸ”¢ 9ê°œ ë¶„í• ì—�ì„œë§Œ ë“±ì�¥í•œ í”¼ì²˜ ìˆ˜: {len(qualified_features)}")
print(qualified_features)
qualified_features = [f for f, count in feature_counter.items() if count == 8]
print(f"\nğŸ”¢ 8ê°œ ë¶„í• ì—�ì„œë§Œ ë“±ì�¥í•œ í”¼ì²˜ ìˆ˜: {len(qualified_features)}")
print(qualified_features)

