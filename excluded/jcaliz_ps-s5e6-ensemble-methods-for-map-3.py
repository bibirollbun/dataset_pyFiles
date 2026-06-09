import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.stats import rankdata
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

def map_3(label, preds, sort=True):
    sorted_predictions = np.argsort(preds, axis=1)[:, -3:][:, ::-1]

    # Disclaimer: this is not optimized
    map_at_3 = 0
    for i in range(3):
        map_at_3 += (sorted_predictions[:, i] == y_vl).sum() / (i+1)

    return map_at_3 / len(preds)


train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv', index_col=0)

# Cast as categorical
categorical_columns = ['Soil Type', 'Crop Type', 'Fertilizer Name']
train_df[categorical_columns] = train_df[categorical_columns].astype('category')

# Cast label as integer
label_categories = train_df['Fertilizer Name'].cat.categories
train_df['Fertilizer Name'] = train_df['Fertilizer Name'].cat.codes
 
# Split and train
X_tr, X_vl, y_tr, y_vl = train_test_split(
    train_df.drop('Fertilizer Name', axis=1), train_df['Fertilizer Name'], test_size=0.2, random_state=0)


model_1 = XGBClassifier(objective='multi:softmax', enable_categorical=True)
model_2 = Pipeline([
    ('Preprocessing', ColumnTransformer([
        ("OneHot", OneHotEncoder(), ['Soil Type', 'Crop Type']),
        ("Scaler", StandardScaler(), [
            'Temparature', 'Humidity', 'Moisture',
            'Nitrogen', 'Potassium', 'Phosphorous'])
    ])),
    ('Model', LogisticRegression())
])
model_3 = LGBMClassifier(objective='multiclass', verbose=0)


model_1.fit(X_tr, y_tr)
model_2.fit(X_tr, y_tr)
model_3.fit(X_tr, y_tr)

vl_preds_1 = pd.DataFrame(model_1.predict_proba(X_vl), index=X_vl.index, columns=label_categories)
vl_preds_2 = pd.DataFrame(model_2.predict_proba(X_vl), index=X_vl.index, columns=label_categories)
vl_preds_3 = pd.DataFrame(model_3.predict_proba(X_vl), index=X_vl.index, columns=label_categories)


print('Model XGB  logloss: ', '{:.6}'.format(log_loss(y_vl, vl_preds_1)))
print('Model LR   logloss: ', '{:.6}'.format(log_loss(y_vl, vl_preds_2)))
print('Model LGBM logloss: ', '{:.6}'.format(log_loss(y_vl, vl_preds_3)))


print('Model XGB map@3:    ', f"{map_3(y_vl, vl_preds_1):.6f}")
print('Model LR map@3:     ', f"{map_3(y_vl, vl_preds_2):.6f}")
print('Model XGB+LR map@3: ', f"{map_3(y_vl, 99 * vl_preds_1 + 1 * vl_preds_2):.6f}")


ranked_vl_preds_1 = rankdata(vl_preds_1, axis=1)
ranked_vl_preds_2 = rankdata(vl_preds_2, axis=1)
ranked_vl_preds_3 = rankdata(vl_preds_3, axis=1)

print('Model XGB map@3:          ', f"{map_3(y_vl, vl_preds_1):.6f}")
print('Model LR map@3:           ', f"{map_3(y_vl, vl_preds_2):.6f}")
print('Model LGM map@3:          ', f"{map_3(y_vl, vl_preds_3):.6f}")
print('Model rank XGB+LGBM map@3:', f"{map_3(y_vl, 0.1 * ranked_vl_preds_1 + 3 * vl_preds_3):.6f}")


def fun(x, preds):
    x = x.reshape(len(x), 1, 1)
    preds = np.sum(np.multiply(preds, x), axis=0)
    return -map_3(y_vl, preds)

np_preds = np.array([
    vl_preds_1,
    vl_preds_2,
    vl_preds_3
])

coefs = minimize(fun, x0=[1/len(np_preds)] * len(np_preds), args=(np_preds), method='Nelder-Mead').x
coefs = coefs.reshape(len(coefs), 1, 1)

print('Model XGB  map@3:           ', f"{map_3(y_vl, vl_preds_1):.6f}")
print('Model LR   map@3:           ', f"{map_3(y_vl, vl_preds_2):.6f}")
print('Model LGM  map@3:           ', f"{map_3(y_vl, vl_preds_3):.6f}")
print('Optimize Nelder-Mead  map@3:', f"{map_3(y_vl, np.sum(np.multiply(np_preds, coefs), axis=0)):.6f}")

