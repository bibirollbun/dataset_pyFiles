import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


lgb_preds = np.load("/kaggle/input/fork-of-s5e12-feature-en-lightgbm-968e65/lgb_test.npy")
lgb_oof= np.load("/kaggle/input/fork-of-s5e12-feature-en-lightgbm-968e65/lgb_oof.npy")

xgb_oof = np.load("/kaggle/input/s5e12-xgboost-diabetes-prediction-0-6962/xgb_oof.npy")
xgb_preds = np.load("/kaggle/input/s5e12-xgboost-diabetes-prediction-0-6962/xgb_preds.npy")

nn_oof = np.load("/kaggle/input/s5e12-ann-cv-deep-learning/nn_oof.npy")
nn_preds = np.load("/kaggle/input/s5e12-ann-cv-deep-learning/nn_preds.npy")


df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


y = df['diagnosed_diabetes']


print(lgb_preds.shape,lgb_oof.shape)
print(xgb_preds.shape,xgb_oof.shape)
print(nn_preds.shape,nn_oof.shape)


print("XGB OOF AUC:", roc_auc_score(y, xgb_oof))
print("LGB OOF AUC:", roc_auc_score(y, lgb_oof))
print("NN? OOF AUC:",roc_auc_score(y,nn_oof))


best_auc = 0
best_w = None

for w1 in np.arange(0.1, 0.9, 0.1):
    for w2 in np.arange(0.1, 0.9, 0.1):
        w3 = 1 - (w1 + w2)
        if w3 <= 0:
            continue

        blend = w1*xgb_oof + w2*lgb_oof + w3*nn_oof
        auc = roc_auc_score(y, blend)

        if auc > best_auc:
            best_auc = auc
            best_w = (w1, w2, w3)

print(best_w, best_auc)



best_w


final_pred = best_w[0] * xgb_preds + best_w[1]* lgb_preds + best_w[2] * nn_preds


submission = pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": final_pred
})

submission.to_csv("submission_blend.csv", index=False)



submission.head()




