import numpy as np 
import polars as pl
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss


train = pl.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pl.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
orig_data = pl.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
train.head()


train.drop_in_place("id")
test.drop_in_place("id")


train = pl.concat([train, orig_data])


le = LabelEncoder()
encoded = le.fit_transform(train['Fertilizer Name'])

train = train.with_columns([
    pl.Series(name="Fertilizer Name", values=encoded)
])


y = train['Fertilizer Name'] 
X = train.drop(['Fertilizer Name'])


for col in ['Soil Type', 'Crop Type']:
    le_feature = LabelEncoder()
    combined = pl.concat([X[col], test[col]])
    le_feature.fit(combined)
    X_trans = le_feature.transform(X[col])
    test_trans = le_feature.transform(test[col])
    X = X.with_columns([
    pl.Series(name=col, values=X_trans)
    ])
    test = test.with_columns([
    pl.Series(name=col, values=test_trans)
    ])



FOLDS = 7
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)


oof = np.zeros((len(train), len(np.unique(y))))
pred = np.zeros((len(test), len(np.unique(y))))
logloss = []


for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"\n{'#'*10} Fold {i+1} {'#'*10}")
    
    x_train = X[train_idx].clone()
    y_train = y[train_idx]
    x_valid = X[valid_idx].clone()
    y_valid = y[valid_idx]
    x_test = test.clone()

    
    dtrain = xgb.DMatrix(x_train, label=y_train, enable_categorical=True)
    dvalid = xgb.DMatrix(x_valid, label=y_valid, enable_categorical=True)
    dtest = xgb.DMatrix(x_test, enable_categorical=True)

   
    params = {
        'objective': 'multi:softprob', 
        'num_class': len(np.unique(y)),  
        'max_depth': 10,
        'learning_rate': 0.03,
        'min_child_weight' : 2,
        'alpha': 0.8, 
        'reg_lambda': 4.0, 
        'colsample_bytree': 0.5,
        'subsample': 0.7,
        'max_bin': 128,
        'colsample_bytree': 0.5, 
        'colsample_bylevel': 1,  
        'colsample_bynode': 1,
        'tree_method': 'hist',  
        'random_state': 42,
        'eval_metric': 'mlogloss',
        "device":  "cuda"

               
    }

   
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        evals=[(dvalid, 'valid')],
        early_stopping_rounds=50,
        verbose_eval=200
    )

    
    oof[valid_idx] = model.predict(dvalid)
    pred += model.predict(dtest)

    log_loss_value = log_loss(y_valid, oof[valid_idx])
    print(f"Fold {i+1} log_loss: {log_loss_value:.4f}")
    logloss.append(log_loss_value)


pred /= FOLDS
log_loss_value = np.mean(logloss)

print(f"\nFinal CV log_loss: {log_loss_value:.4f}")


top_preds = np.argsort(pred, axis=1)[:, -3:][:, ::-1]  
actual = [[label] for label in y]

def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])
score = mapk(actual, top_preds)
print(f"Score: {score:.5f}")


top3_categories = le.inverse_transform(top_preds.flatten()).reshape(-1, 3)


df = pl.DataFrame({
    'id': np.arange(750000, 1000000),
    'Fertilizer name': [" ".join(map(str, row)) for row in top3_categories]
})
df.write_csv('submission.csv')


df




