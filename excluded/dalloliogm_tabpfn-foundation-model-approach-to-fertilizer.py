pip install -q tabpfn



# Imports
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tabpfn import TabPFNClassifier
import torch


# ğŸ“¦ Imports
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
import torch

# ğŸ§¹ Feature Engineering Transformer
class FertilizerFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['N_P_ratio'] = X['Nitrogen'] / (X['Phosphorous'] + 1)
        X['N_K_ratio'] = X['Nitrogen'] / (X['Potassium'] + 1)
        X['P_K_ratio'] = X['Phosphorous'] / (X['Potassium'] + 1)
        X['Temp_Humidity'] = X['Temparature'] * X['Humidity']
        X['Soil_Crop'] = X['Soil Type'] + '_' + X['Crop Type']
        return X

# ğŸ“‚ Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# ğŸ�¯ Limit to top 100 most common classes
target = 'Fertilizer Name'
top_100_classes = train[target].value_counts().nlargest(100).index
train_subset = train[train[target].isin(top_100_classes)]

# ğŸ”€ Subsample to 10,000 rows
train_subset = train_subset.sample(n=10_000, random_state=42).reset_index(drop=True)

# â�— Split into X and y
X = train_subset.drop(columns=target)
y = train_subset[target]

# ğŸ› ï¸� Pipeline
pipeline = Pipeline([
    ('fe', FertilizerFeatureEngineer())
])

X_proc = pipeline.fit_transform(X)
X_test_proc = pipeline.transform(test)

# ğŸ�¨ Encode target
le = LabelEncoder()
y_enc = le.fit_transform(y)

# ğŸ”¢ Keep only numeric features for tabpfn
X_proc = X_proc.select_dtypes(include=np.number)
X_test_proc = X_test_proc.select_dtypes(include=np.number)

# âœ‚ï¸� Train/val split
X_tr, X_val, y_tr, y_val = train_test_split(X_proc, y_enc, test_size=0.2, stratify=y_enc, random_state=42)




# ğŸš€ TabPFN
clf = TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu')
clf.fit(X_tr.to_numpy().astype(np.float32), y_tr)




def predict_tabpfn_in_batches(model, X, batch_size=256):
    all_probs = []
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size]
        probs = model.predict_proba(X_batch.to_numpy().astype(np.float32))
        all_probs.append(probs)
    return np.vstack(all_probs)

# âœ… Use it
proba = predict_tabpfn_in_batches(clf, X_test_proc)



# ğŸ”® Predict top-3 for test
top3_idx = np.argsort(-proba, axis=1)[:, :3]
top3_labels = le.inverse_transform(top3_idx.flatten()).reshape(top3_idx.shape)
top3_joined = [' '.join(row) for row in top3_labels]

# ğŸ“¤ Submission
submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': top3_joined
})
submission.to_csv('submission.csv', index=False)

