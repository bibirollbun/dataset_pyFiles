import pandas as pd
import polars as pl


%%capture

!pip install -q autogluon


# Load the data
train_df = pl.read_parquet("/kaggle/input/drw-crypto-market-prediction/train.parquet")
train_df = train_df.select(pl.all().shrink_dtype())
train_df = train_df.to_pandas()
train_df.head()


train_df.dtypes.value_counts()


test_df = pl.read_parquet("/kaggle/input/drw-crypto-market-prediction/test.parquet")
test_df = test_df.select(pl.all().shrink_dtype())
test_df = test_df.to_pandas()
test_df.head()



sample_submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
sample_submission.head()


train_df = train_df.tail(200_000)


from sklearn.feature_selection import VarianceThreshold

X = train_df.drop(columns=['timestamp', 'label'])
y = train_df['label']




# Remove low variance features

import numpy as np

# Replace inf values with NaN
X = X.replace([np.inf, -np.inf], np.nan)

# Drop columns with all NaNs
X = X.dropna(axis=1, how='all')

# Option 1: Fill NaNs with 0 (or use mean imputation)
X = X.fillna(0)

# Now apply VarianceThreshold
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)
X_reduced = pd.DataFrame(selector.fit_transform(X), columns=X.columns[selector.get_support()])

# Combine with label
train_filtered = pd.concat([X_reduced, y.reset_index(drop=True)], axis=1)






# ---- Time-aware validation split ----
split_idx = int(0.8 * len(train_filtered))
train_data = train_filtered.iloc[:split_idx]
val_data = train_filtered.iloc[split_idx:]


import pandas as pd
from autogluon.tabular import TabularPredictor



# Define feature and label columns
label = 'label'
ignore_cols = ['timestamp', 'ID'] 


predictor = TabularPredictor(label='label', eval_metric='pearsonr').fit(
    train_data.drop(columns="timestamp", errors="ignore"),
    tuning_data=val_data,
    presets='medium_quality',
    excluded_model_types=['NN_TORCH', 'CATBOOST'],
    time_limit=1800
)


# Predict on test set
preds = predictor.predict(test_df.drop(columns=ignore_cols + [label], errors="ignore"))

# Create submission
submission = sample_submission.copy()
submission['label'] = preds
submission.to_csv('submission.csv', index=False)



# ---- Feature Importance ----
fi = predictor.feature_importance(val_data)
fi['importance'].head(30).plot(kind='barh', figsize=(8, 10), title='Top 30 Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ---- Prepare test set ----

test_df = test_df.fillna(0)

test_X = test_df.drop(columns=['label'], errors='ignore')
test_X = test_X.replace([np.inf, -np.inf], np.nan).fillna(0)
test_X_reduced = pd.DataFrame(selector.transform(test_X), columns=X_reduced.columns)





# ---- Predict and Save ----
preds = predictor.predict(test_X_reduced)
submission = sample_submission.copy()
submission['label'] = preds
submission.to_csv('submission.csv', index=False)

