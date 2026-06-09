!pip install -U catboost


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm_notebook, tqdm
import warnings
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
%matplotlib inline

warnings.simplefilter('ignore')
warnings.filterwarnings("ignore")

seed = 52
random.seed(seed)
np.random.seed(seed)


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype.name

        if col_type not in ['object', 'category']:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


data = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
data_org = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')


data


data_org


data.isnull().sum()


data_org.isnull().sum()


data = data.drop(columns=['id'])


data['Fertilizer Name'].value_counts()


data_org['Fertilizer Name'].value_counts()


data = pd.concat([data, data_org], ignore_index=True)


data.drop('Fertilizer Name', axis=1).select_dtypes(include=['object', 'bool']).columns.tolist()


data = data.drop_duplicates()


fer_name_to_num = {
    '14-35-14': 0,
    '10-26-26': 1,
    '17-17-17': 2,
    '28-28': 3,
    '20-20': 4,
    'DAP': 5,
    'Urea': 6
}
num_to_fer_name = {fer_name_to_num[i]: i for i in fer_name_to_num}


data['Fertilizer Name'] = data['Fertilizer Name'].apply(lambda x: fer_name_to_num[x])


data['Soil Type'].value_counts()


data['Crop Type'].value_counts()


sns.set(rc={'figure.figsize': (11, 8)})
correlation_matrix = pd.get_dummies(data, columns=data.drop('Fertilizer Name', axis=1).select_dtypes(include=['object', 'bool']).columns.tolist()).corr()
sns.heatmap(correlation_matrix)


data = reduce_mem_usage(data)


def map_at_3(y_true, y_pred):
    map_scores = []
    
    for true_labels, pred_labels in zip(y_true, y_pred):
        pred_labels = pred_labels[:3]
        
        average_precision = 0
        num_correct = 0
        correct_labels = set(true_labels)
        
        for k, pred in enumerate(pred_labels, 1):
            if pred in correct_labels:
                num_correct += 1
                precision_at_k = num_correct / k
                average_precision += precision_at_k
                correct_labels.remove(pred)
        
        if len(true_labels) > 0:
            average_precision /= min(len(true_labels), 3)
        
        map_scores.append(average_precision)
    
    return np.mean(map_scores) if map_scores else 0


X_train, X_valid, Y_train, Y_valid = train_test_split(
    data.drop(columns=['Fertilizer Name']),
    data['Fertilizer Name'],
    test_size=0.1,
    shuffle=True,
    stratify=data['Fertilizer Name'],
    random_state=seed
)


X_train.shape, X_valid.shape


model = CatBoostClassifier(
    iterations=4096 * 2,
    learning_rate=0.02,
    l2_leaf_reg=0.16,
    loss_function='MultiClass',
    eval_metric='MultiClass',
    depth=8,
    use_best_model=True,
    random_state=seed,
    bagging_temperature=0.24,
    border_count=128,
    task_type='GPU',
    verbose=1000
)
model.fit(
    X_train,
    Y_train,
    eval_set=(X_valid, Y_valid),
    cat_features=['Soil Type', 'Crop Type'],
    early_stopping_rounds=100
)


map_at_3([[i] for i in Y_valid], np.argsort(-model.predict_proba(X_valid), axis=1)[:, :3])


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
score_mean = 0
models = []

for train_idx, valid_idx in tqdm(list(skf.split(data.drop(columns=['Fertilizer Name']), data['Fertilizer Name']))):
    X_tr, X_val = data.drop(columns=['Fertilizer Name']).iloc[train_idx], data.drop(columns=['Fertilizer Name']).iloc[valid_idx]
    y_tr, y_val = data['Fertilizer Name'].iloc[train_idx], data['Fertilizer Name'].iloc[valid_idx]
    print('Train size:', len(y_tr), 'Valid size:', len(y_val))
    model = CatBoostClassifier(
        iterations=4096 * 2,
        learning_rate=0.02,
        l2_leaf_reg=0.16,
        loss_function='MultiClass',
        eval_metric='MultiClass',
        depth=8,
        use_best_model=True,
        random_state=seed,
        bagging_temperature=0.24,
        border_count=128,
        task_type='GPU',
        verbose=1000
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=(X_val, y_val),
        cat_features=['Soil Type', 'Crop Type'],
        early_stopping_rounds=100
    )
    map3 = map_at_3([[i] for i in y_val], np.argsort(-model.predict_proba(X_val), axis=1)[:, :3])
    print('MAP@3:', map_at_3([[i] for i in y_val], np.argsort(-model.predict_proba(X_val), axis=1)[:, :3]))
    print()
    score_mean += map3
    models.append(model)
print('Mean MAP@3:', score_mean / 10)


test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
test = test[X_train.columns]
test = reduce_mem_usage(test)
test


submit = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')
submit


pred = sum([m.predict_proba(test) for m in tqdm(models)]) / 10 # model.predict_proba(test)
pred = np.argsort(-pred, axis=1)[:, :3]
pred = [' '.join([num_to_fer_name[j] for j in i]) for i in pred]


submit['Fertilizer Name'] = pred


submit


submit.to_csv('Baseline_pred-_with_origTrain_v01_StartKFold10_reduce_mem_usage.csv', index=False)

