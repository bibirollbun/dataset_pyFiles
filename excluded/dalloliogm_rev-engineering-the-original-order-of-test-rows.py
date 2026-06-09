%%capture

#!pip install -q scikit-learn==1.3.2 autogluon==0.8.2
!pip install -q autogluon


import os
import pandas as pd
import polars as pl
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

def is_interactive_session():
    return os.environ.get('KAGGLE_KERNEL_RUN_TYPE','') == 'Interactive'

is_interactive_session()

config = {
    "autogluon_time": 3600,
    #"reduce_features": 0, # Set to >0 to use only the first n features
    "tail_rows": 0 # Set to >0 to use only the last n rows in the file
}

if is_interactive_session():
    print("Interactive session")
    config["autogluon_time"] = 100
    #config["reduce_features"] = 200
    config["tail_rows"] = 2000
    print(config)
else:
    print("running as job")
    print(config)




# Load the data
train_df = pl.read_parquet("/kaggle/input/drw-crypto-market-prediction/train.parquet")
train_df = train_df.select(pl.all().shrink_dtype())
train_df = train_df.to_pandas()
#train_df = train_df.sort_values('timestamp').reset_index(drop=True)
train_df['time_rank'] = np.arange(len(train_df)) / len(train_df)  # normalized 0â€“1
train_df.head()

# No need to read test yet
#test_df = pl.read_parquet("/kaggle/input/drw-crypto-market-prediction/test.parquet#")
#test_df = test_df.select(pl.all().shrink_dtype())
#test_df = test_df.to_pandas()
#test_df.head()



if config["tail_rows"]>0:
    train_df = train_df.tail(config["tail_rows"])



from autogluon.tabular import TabularPredictor

# --- Step 3: Clean features ---
train_df = train_df.replace([np.inf, -np.inf], np.nan).fillna(0)
train_df = train_df.drop(columns=['timestamp', 'label'], errors='ignore')

# Optional: reduce rows for memory
# train_df = train_df.tail(200_000)

# --- Step 4: Train AutoGluon model ---
predictor = TabularPredictor(label='time_rank', eval_metric='r2').fit(
    train_df,
    presets='medium_quality',  # or 'best_quality' if memory allows
    time_limit=config["autogluon_time"],  # Optional: 30 minutes max
    excluded_model_types=['NN_TORCH', 'CATBOOST']  # lighter memory
)


fi = predictor.feature_importance(train_df)
fi['importance'].head(20).plot(kind='barh', figsize=(8, 6), title='Top Temporal Features')



# --- Step 3: Predict and evaluate ---
y_true = val_data['time_rank'].values
y_pred = predictor.predict(val_data.drop(columns='time_rank'))

r2 = r2_score(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
rank_corr, _ = spearmanr(y_true, y_pred)

print(f"ðŸ“ˆ RÂ²: {r2:.4f}")
print(f"ðŸ“‰ RMSE: {rmse:.4f}")
print(f"ðŸ“Š Spearman Rank Correlation: {rank_corr:.4f}")

# --- Step 4: Plot true vs. predicted time_rank ---
plt.figure(figsize=(8, 6))
plt.scatter(y_true, y_pred, alpha=0.2, s=10)
plt.plot([0, 1], [0, 1], 'r--', lw=1)
plt.xlabel("True time_rank")
plt.ylabel("Predicted time_rank")
plt.title("True vs Predicted time_rank")
plt.grid(True)
plt.tight_layout()
plt.show()


# --- Load and shrink test set ---
test_df = pl.read_parquet("/kaggle/input/drw-crypto-market-prediction/test.parquet")
test_df = test_df.select(pl.all().shrink_dtype()).to_pandas()

# --- Clean test data ---
test_df = test_df.replace([np.inf, -np.inf], np.nan).fillna(0)
test_X = test_df.drop(columns='label', errors='ignore')

# --- Predict time ---
test_df['predicted_time'] = predictor.predict(test_X)

# --- Reorder test rows ---
test_df_ordered = test_df.sort_values('predicted_time').reset_index(drop=True)


