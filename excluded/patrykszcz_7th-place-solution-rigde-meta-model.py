import pandas as pd 
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv',index_col = 'id')
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv',index_col = 'id')


X = train.drop('accident_risk',axis = 1)
y = train['accident_risk']


oof = np.load('/kaggle/input/model-oof-predictions/oof_1_xgb.npz')
oof_1 = oof['oof']
preds_1 = oof['test_oof'] 

oof_2 = np.array(pd.read_csv('/kaggle/input/model-oof-predictions/oof_2_nn.csv')['accident_risk'])
preds_2 = np.array(pd.read_csv('/kaggle/input/model-oof-predictions/preds_2_test_nn.csv')['accident_risk'])

oof_3 = np.load('/kaggle/input/model-oof-predictions/oof_3.npy')
preds_3 = np.load('/kaggle/input/model-oof-predictions/pred_3_test.npy')

oof_4 = np.array(pd.read_csv('/kaggle/input/model-oof-predictions/oof_4.csv')['accident_risk'])
preds_4 = np.array(pd.read_csv('/kaggle/input/model-oof-predictions/preds_4.csv')['accident_risk'])

oof_5 = np.array(pd.read_csv('/kaggle/input/model-oof-predictions/oof_5.csv')['accident_risk'])
preds_5 = np.array(pd.read_csv('/kaggle/input/model-oof-predictions/preds_5.csv')['accident_risk'])

oof_6 = np.array(pd.read_csv('/kaggle/input/model-oof-predictions/oof_6.csv')['YDF'])
preds_6 = np.array(pd.read_csv('/kaggle/input/model-oof-predictions/preds_6.csv')['YDF'])

oof_7 = np.array(pd.read_csv('/kaggle/input/model-oof-predictions/oof_7.csv')['accident_risk'])
preds_7 = np.array(pd.read_csv('/kaggle/input/model-oof-predictions/preds_7.csv')['accident_risk'])

oof_8 = np.array(pd.read_csv('/kaggle/input/model-oof-predictions/oof_8.csv').iloc[:,0])
preds_8 = np.array(pd.read_csv('/kaggle/input/model-oof-predictions/preds_8.csv')['accident_risk'])

oof_9 = np.array(pd.read_csv('/kaggle/input/model-oof-predictions/oof_9.csv').iloc[:,0])
preds_9 = np.array(pd.read_csv('/kaggle/input/model-oof-predictions/preds_9.csv')['accident_risk'])


preds = {
    1: preds_1,
    2: preds_2,
    3: preds_3,
    4: preds_4,
    5: preds_5,
    6: preds_6,
    7: preds_7,
    8: preds_8,
    9: preds_9
}

for i in range(1, 10):
    print(f"preds_{i}.shape =", preds[i].shape)


oofs = {
    1: oof_1,
    2: oof_2,
    3: oof_3,
    4: oof_4,
    5: oof_5,
    6: oof_6,
    7: oof_7,
    8: oof_8,
    9: oof_9
}

for i in range(1, 10):
    print(f"oofs_{i} RMSE {mean_squared_error(y, oofs[i], squared = False)}")


X_meta = np.column_stack([oof_1,oof_2,oof_3,oof_4,oof_5,oof_6,oof_7,oof_8,oof_9])
x_test = np.column_stack([preds_1,preds_2,preds_3,preds_4,preds_5,preds_6,preds_7,preds_8,preds_9])


kf = KFold(n_splits=15, shuffle=True, random_state=42)
oof_10 = np.zeros(len(train))
preds_10 = np.zeros(len(test))
for i, (train_idx, valid_idx) in enumerate(kf.split(X_meta,y)):
    print('#' * 15, f" FOLD {i+1} ", '#' * 15)

    X_train, X_valid = X_meta[train_idx], X_meta[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
    model = Ridge(**{
        'alpha': 0.0765505252086428,
        'tol': 0.0002159268285591614, 
        'solver': 'sag', 
        'max_iter': 1728,
        'fit_intercept':True
    })
    model.fit(X_train, y_train)

    y_preds = model.predict(X_valid)
    oof_10[valid_idx] = y_preds

    preds_10 += model.predict(x_test) / kf.n_splits
    fold_rmse = mean_squared_error(y_valid, y_preds, squared=False)
    print(f"âœ… FOLD {i+1}: RMSE = {fold_rmse:.5f}")

        
score = mean_squared_error(y, oof_10, squared=False)
print(f"ðŸ’¡ FINAL RMSE: {score:.7f}")



submission = pd.DataFrame({
    'id': test.index,
    'accident_risk': preds_10
})
submission.to_csv(f'submission.csv', index=False)
print("âœ… Submission file saved as 'submission.csv'")
submission.head()

