# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os


data_path = '/kaggle/input/aeroclub-recsys-2025/'

print(os.listdir(data_path))


import duckdb
import pandas as pd
import numpy as np
import os

print("Libraries loaded.")

data_path = '/kaggle/input/aeroclub-recsys-2025/'
train_parquet_path = data_path + 'train.parquet'
test_parquet_path = data_path + 'test.parquet'

con = duckdb.connect(database=':memory:', read_only=False)
print("DuckDB connection established successfully.")


row_count_query = f"SELECT COUNT(*) FROM '{train_parquet_path}'"
total_rows = con.execute(row_count_query).fetchone()[0]
print(f"\nTotal number of rows in the training set: {total_rows:,}")

schema_query = f"DESCRIBE SELECT * FROM '{train_parquet_path}'"
schema_df = con.execute(schema_query).fetchdf()
print("\nData Schema (Columns and Types):")
display(schema_df)

head_query = f"SELECT * FROM '{train_parquet_path}' LIMIT 5"
train_head_df = con.execute(head_query).fetchdf()
print("\nFirst 5 sample rows from training data:")
display(train_head_df)

print("\nFirst look at data with DuckDB completed. No memory issues!")


print("Preparing query: Which are the most preferred airlines?")

query = f"""
SELECT
    companyID,
    COUNT(*) AS total_options,
    SUM(selected) AS total_selections
FROM '{train_parquet_path}'
GROUP BY
    companyID
ORDER BY
    total_selections DESC
LIMIT 20;
"""

print("DuckDB runs the query...")
top_airlines_df = con.execute(query).fetchdf()
print("The query is completed and the results are stored.")


epsilon = 1e-9
top_airlines_df['selection_rate_%'] = (top_airlines_df['total_selections'] / (top_airlines_df['total_options'] + epsilon)) * 100

print("\nTop 20 Most Preferred Airlines:")
display(top_airlines_df.sort_values(by='total_selections', ascending=False))


import matplotlib.pyplot as plt
import seaborn as sns
print("Queries are being prepared: Price distributions of selected and unselected flights...")

query_selected = f"""
SELECT totalPrice
FROM '{train_parquet_path}'
WHERE selected = 1;
"""
selected_prices_df = con.execute(query_selected).fetchdf()


query_not_selected_sample = f"""
SELECT totalPrice
FROM '{train_parquet_path}'
WHERE selected = 0 AND random() < 0.1;
"""
not_selected_prices_df = con.execute(query_not_selected_sample).fetchdf()

print("Queries completed. Price data stored.")


print("Creating scatter plot (histogram)...")

plt.figure(figsize=(14, 7))

sns.histplot(not_selected_prices_df['totalPrice'], color='skyblue', label='Flights Not Selected (Sample 10%)', kde=True, log_scale=True)

sns.histplot(selected_prices_df['totalPrice'], color='red', label='Selected Flights', kde=True, log_scale=True)

plt.title('Price Distribution of Selected and Unselected Flights (Logarithmic Scale)', fontsize=16)
plt.xlabel('Total Price - Logarithmic Scale', fontsize=12)
plt.ylabel('Frequency (Number of Flights)', fontsize=12)
plt.legend()
plt.grid(True, which="both", ls="--", c='0.7')
plt.show()


print("Queries are being prepared: Times in 'HH:MM:SS' format will be converted to minutes...")

duration_calculation = """
    (
        COALESCE(EPOCH(CAST(legs0_segments0_duration AS INTERVAL)) / 60, 0) +
        COALESCE(EPOCH(CAST(legs0_segments1_duration AS INTERVAL)) / 60, 0) +
        COALESCE(EPOCH(CAST(legs0_segments2_duration AS INTERVAL)) / 60, 0) +
        COALESCE(EPOCH(CAST(legs1_segments0_duration AS INTERVAL)) / 60, 0) +
        COALESCE(EPOCH(CAST(legs1_segments1_duration AS INTERVAL)) / 60, 0) +
        COALESCE(EPOCH(CAST(legs1_segments2_duration AS INTERVAL)) / 60, 0)
    ) AS total_duration
"""

query_selected_duration = f"""
SELECT {duration_calculation}
FROM '{train_parquet_path}'
WHERE selected = 1;
"""
selected_duration_df = con.execute(query_selected_duration).fetchdf()


query_not_selected_duration = f"""
SELECT {duration_calculation}
FROM '{train_parquet_path}'
WHERE selected = 0 AND random() < 0.1;
"""
not_selected_duration_df = con.execute(query_not_selected_duration).fetchdf()

print("Queries completed. Flight time data stored.")


print("Creating scatter plot (histogram)...")

plt.figure(figsize=(14, 7))

sns.histplot(not_selected_duration_df['total_duration'], color='skyblue', label='Seçilmeyen Uçuşlar (Örneklem %10)', kde=True)
sns.histplot(selected_duration_df['total_duration'], color='red', label='Seçilen Uçuşlar', kde=True)

plt.title('Duration Distribution of Selected and Unselected Flights', fontsize=16)
plt.xlabel('Calculated Total Flight Time (Minutes)', fontsize=12)
plt.ylabel('Frequency (Number of Flights)', fontsize=12)
plt.legend()
plt.grid(True, which="both", ls="--", c='0.7')
plt.xlim(0, 2000)
plt.show()


print("Preparing test query: New features will be created...")

duration_calculation = """
    (
        COALESCE(EPOCH(CAST(legs0_segments0_duration AS INTERVAL)) / 60, 0) +
        COALESCE(EPOCH(CAST(legs0_segments1_duration AS INTERVAL)) / 60, 0) +
        COALESCE(EPOCH(CAST(legs0_segments2_duration AS INTERVAL)) / 60, 0) +
        COALESCE(EPOCH(CAST(legs1_segments0_duration AS INTERVAL)) / 60, 0) +
        COALESCE(EPOCH(CAST(legs1_segments1_duration AS INTERVAL)) / 60, 0) +
        COALESCE(EPOCH(CAST(legs1_segments2_duration AS INTERVAL)) / 60, 0)
    )
"""

feature_query = f"""
WITH DataWithDuration AS (
    SELECT *, {duration_calculation} AS total_duration
    FROM '{train_parquet_path}'
)
SELECT
    Id,
    ranker_id,
    companyID,
    totalPrice,
    total_duration,
    selected,
    -- YENİ ÖZELLİK 1: Grup içindeki fiyat sırası (en ucuz = 1)
    RANK() OVER(PARTITION BY ranker_id ORDER BY totalPrice ASC) AS price_rank_in_group,
    -- YENİ ÖZELLİK 2: Grup içindeki süre sırası (en kısa = 1)
    RANK() OVER(PARTITION BY ranker_id ORDER BY total_duration ASC) AS duration_rank_in_group,
    -- YENİ ÖZELLİK 3: Direkt uçuş mu? (1 = Evet, 0 = Hayır)
    CAST(CASE WHEN legs0_segments1_duration IS NULL AND legs1_segments0_duration IS NULL THEN 1 ELSE 0 END AS TINYINT) AS is_direct_flight

FROM DataWithDuration
"""

test_ids = "'ce0dabf6964640b63079fbafd42cbe', '4a26a333c5d64e999b8673a5a71141a3', '6e81f72744f445458066f774314bee4f'"
test_query = f"""
{feature_query}
WHERE ranker_id IN ({test_ids})
ORDER BY ranker_id, price_rank_in_group;
"""

print("Running test query for only 3 groups...")
featured_test_df = con.execute(test_query).fetchdf()

print("\nTest Output with New Features Added:")
display(featured_test_df)


print("For testing, 3 ranker_id are randomly taken from the dataset...")
get_ids_query = f"SELECT DISTINCT ranker_id FROM '{train_parquet_path}' LIMIT 3;"
test_ids_df = con.execute(get_ids_query).fetchdf()
test_ids_list = test_ids_df['ranker_id'].tolist()
test_ids_sql_format = ", ".join([f"'{id_}'" for id_ in test_ids_list])
print(f"IDs selected for testing: {test_ids_list}")


test_query = f"""
{feature_query}
WHERE ranker_id IN ({test_ids_sql_format})
ORDER BY ranker_id, price_rank_in_group;
"""

print("\nRunning test query with dynamic IDs...")
featured_test_df = con.execute(test_query).fetchdf()

print("\nTest Output with New Features Added:")
display(featured_test_df.head(15)) 


print("\n--- FULL VERSION LAUNCHED ---")
print("Running feature engineering query for all training data...")
print("This may take a few minutes, please wait...")

full_featured_df = con.execute(feature_query).fetchdf()

print("\nFeature engineering complete!")
print(f"Size of new dataset: {full_featured_df.shape}")

output_path = "train_featured.parquet"
print(f"processed data '{output_path}' saving to file...")
full_featured_df.to_parquet(output_path)

print("\nProcess completed! Now we can use the 'train_featured.parquet' file for modeling.")
display(full_featured_df.head())


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import gc

print("Libraries for modeling have been loaded.")

df = pd.read_parquet('train_featured.parquet')
print("Feature engineered data 'train_featured.parquet' loaded.")
print(f"Data size: {df.shape}")


def calculate_hitrate3(df_preds):
    group_sizes = df_preds.groupby('ranker_id')['Id'].count()
    valid_groups = group_sizes[group_sizes > 10].index
    df_filtered = df_preds[df_preds['ranker_id'].isin(valid_groups)]
    
    if df_filtered.empty:
        return 0.0

    df_filtered['rank'] = df_filtered.groupby('ranker_id')['prediction'].rank(method='first', ascending=False)
    correctly_selected = df_filtered[df_filtered['selected'] == 1]
    hits = (correctly_selected['rank'] <= 3).sum()
    hit_rate = hits / len(valid_groups.unique())
    return hit_rate


features = [
    'totalPrice', 'total_duration', 'price_rank_in_group',
    'duration_rank_in_group', 'is_direct_flight', 'companyID'
]
target = 'selected'
group_col = 'ranker_id'

print(f"\nFeatures to Use: {features}")

gkf = GroupKFold(n_splits=5)
groups = df[group_col]

scores = []

lgbm_ranker = lgb.LGBMRanker(
    objective="lambdarank", metric="ndcg", n_estimators=2000,
    learning_rate=0.05, random_state=42, n_jobs=-1,
    colsample_bytree=0.8, device='gpu'
)

for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df[target], groups=groups)):
    print(f"\n===== FOLD {fold+1} BAŞLADI =====")
    
    train_fold_df = df.iloc[train_idx]
    val_fold_df = df.iloc[val_idx]
    
    train_group = train_fold_df.groupby(group_col).size().to_numpy()
    val_group = val_fold_df.groupby(group_col).size().to_numpy()
    
    X_train, y_train = train_fold_df[features], train_fold_df[target]
    X_val, y_val = val_fold_df[features], val_fold_df[target]
    
    print("The model is being trained...")
    lgbm_ranker.fit(
        X_train, y_train, group=train_group,
        eval_set=[(X_val, y_val)], eval_group=[val_group],
        eval_at=[3], callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    
    print("Predictions are being made...")
    val_predictions = lgbm_ranker.predict(X_val)
    
    val_preds_df = val_fold_df[['Id', 'ranker_id', 'selected']].copy()
    val_preds_df['prediction'] = val_predictions
    
    score = calculate_hitrate3(val_preds_df)
    scores.append(score)
    print(f"FOLD {fold+1} HitRate@3 Score: {score:.5f}")
    
    del train_fold_df, val_fold_df, X_train, y_train, X_val, y_val, val_preds_df
    gc.collect()
    
print("\n===== CROSS VERIFICATION COMPLETED =====")
print(f"Average HitRate@3 Score: {np.mean(scores):.5f} (+/- {np.std(scores):.5f})")




