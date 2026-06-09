!pip install hnswlib --quiet


import os
import itertools
import pandas as pd
import numpy as np
import hnswlib
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, roc_auc_score


base = "/kaggle/input/playground-series-s5e8"
data_train = pd.read_csv(os.path.join(base, 'train.csv'))
data_test = pd.read_csv(os.path.join(base, 'test.csv'))


data_train.head()


data_train.info()


train_X, train_y = data_train.drop(['y'], axis=1), data_train['y']
X_train, X_val, y_train, y_val = train_test_split(
    train_X, train_y, train_size=0.2, shuffle=True, stratify=train_y, random_state=77)


hnsw_scaler = StandardScaler()
hnsw_ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
del_cols = ['id']
cat_cols = None
num_cols = None

# ***** HNSW preprocessing  *****
def data_transform_hnsw(inp: pd.DataFrame, train: bool = True) -> pd.DataFrame:
    global cat_cols, num_cols
    
    data = inp.copy()
    data.drop(del_cols, axis=1, inplace=True, errors="ignore")
    
    if train:
        cat_cols = data.select_dtypes(exclude='number').columns.tolist()
        num_cols = data.select_dtypes(include='number').columns.tolist()
        data[num_cols] = hnsw_scaler.fit_transform(data[num_cols])
        encoded = hnsw_ohe.fit_transform(data[cat_cols])
    else:
        data[num_cols] = hnsw_scaler.transform(data[num_cols])
        encoded = hnsw_ohe.transform(data[cat_cols])
    
    encoded_df = pd.DataFrame(encoded, index=data.index, columns=hnsw_ohe.get_feature_names_out(cat_cols))
    data = pd.concat([data[num_cols], encoded_df], axis=1)
        
    return data


class HNSWClassifier:
    
    def __init__(self, k=11, space='l2', M=32, ef_construction=200, ef=64, num_threads=0):
        self.k = k
        self.space = space
        self.M = M
        self.ef_construction = ef_construction
        self.ef = ef
        self.num_threads = num_threads
        self.hnsw = None
        self.y = None

    
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int32)

        self.hnsw = hnswlib.Index(space=self.space, dim=X.shape[1])
        self.hnsw.init_index(max_elements=X.shape[0], M=self.M, ef_construction=self.ef_construction)
        self.hnsw.set_num_threads(self.num_threads)

        ids = np.arange(X.shape[0], dtype=np.int64)
        self.hnsw.add_items(X, ids)
        self.hnsw.set_ef(self.ef)

        return self

    
    def kneighbors(self, X):
        X = np.asarray(X, dtype=np.float32)
        labels, distances = self.hnsw.knn_query(X, k=self.k)
        
        return self.y[labels]
        

    def predict_proba(self, X):
        neigh_y = self.kneighbors(X)
        p1 = neigh_y.mean(axis=1)
        return np.vstack([1.0 - p1, p1]).T

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(np.int32)


params = {
    'k': 11,
    'space': 'l2',
    'M': 16,
    'ef_construction': 128,
    'ef': 128,
    'num_threads': 0
}

add_ef = [200, 600, 1000]
add_ef_construction = [128, 256, 1024]
add_k = [128, 256, 512, 1024]
add_M = [16, 32, 128]


def find_best_model():
    best_score = -1
    best_model = None
    best_params = None
    
    X_train_hnsw = data_transform_hnsw(X_train)
    X_val_hnsw = data_transform_hnsw(X_val)
    
    print(f"Number of iterations: {len(add_ef) * len(add_ef_construction) * len(add_M) * len(add_k)}")
    
    for (iteration, (ef, ef_k, M, k)) in tqdm(enumerate(itertools.product(add_ef, add_ef_construction, add_M, add_k)), desc="Iteration: "):
        current_params = params | {'ef': ef, 'k': k, 'M': M, 'ef_construction': ef_k}
        hnsw = HNSWClassifier(**current_params)
        hnsw.fit(X_train_hnsw, y_train)
        proba = hnsw.predict_proba(X_val_hnsw)[:, 1]
        auc = roc_auc_score(y_val, proba)
        
        print(f"Iteration: {iteration}, ROC AUC (HNSW): {auc}, ef: {ef}, k: {k}, M: {M}, ef_construction: {ef_k}.")
        if best_score < auc:
            best_score = auc
            best_params = current_params
            best_model = hnsw
    return best_score, best_model, best_params

# best_score, best_model, best_params = find_best_model()
# print(f"best_score: {best_score}\nbest params: {best_params}")


best_params =  {
  'k': 128,
  'space': 'l2',
  'M': 16,
  'ef_construction': 1024,
  'ef': 600,
  'num_threads': 0
}


hnsw = HNSWClassifier(**best_params)
hnsw.fit(data_transform_hnsw(train_X), train_y)


import os
def save_results(model, X, ids=None):
    pred = model.predict_proba(X)
    df = pd.DataFrame({
        'id': ids,
        'y': pred[:, 1]
    })
    df.to_csv('submission.csv', index=False)


# score is 0.94956
save_results(hnsw, data_transform_hnsw(data_test), data_test.id)

