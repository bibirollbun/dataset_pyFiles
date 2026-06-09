# TabPFN extension package
!pip install -q --no-index --find-links /kaggle/input/tabpfn-v2-install/tabpfn-extensions/dist tabpfn-extensions
!mkdir -p /usr/local/lib/python3.10/dist-packages/tabpfn_extensions/hpo/hpo_models
!cp /kaggle/input/tabpfn-v2-install/tabpfn-v2-*.ckpt /usr/local/lib/python3.10/dist-packages/tabpfn_extensions/hpo/hpo_models
# TabPFN package
!pip install -q --no-index --find-links /kaggle/input/tabpfn-v2-install tabpfn
!mkdir -p /root/.cache/tabpfn/
!cp /kaggle/input/tabpfn-v2-install/tabpfn-v2-*.ckpt /root/.cache/tabpfn/


#from tabpfn import TabPFNClassifier
import numpy as np
import pandas as pd
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import (
    AutoTabPFNClassifier,
    AutoTabPFNRegressor,
)
classifier = AutoTabPFNClassifier(max_time=30)
#classifier = TabPFNClassifier(n_estimators=64)



#from xgboost import XGBClassifier
#classifier = XGBClassifier()


# from lightgbm import LGBMClassifier
# classifier = LGBMClassifier()


#from catboost import CatBoostClassifier
#classifier = CatBoostClassifier()


def cleaning(dataset):
    dataset["EJ"] = dataset["EJ"].replace({"B":0,"A":1})
    #TabPFN V2 natively supports NaNs
    #dataset = dataset.fillna(dataset.mean(numeric_only=True))
    return dataset


# adapted from https://www.kaggle.com/code/muelsamu/simple-tabpfn-approach-for-score-of-15-in-1-min
def rebalance(p):
    class_0_est_instances = p[:,0].sum()
    others_est_instances  = p[:,1:].sum()
    new_p = p * np.array([[1/(class_0_est_instances if i==0 else others_est_instances) for i in range(p.shape[1])]])
    p = new_p / np.sum(new_p,axis=1,keepdims=1)
    return p


dataset = pd.read_csv("/kaggle/input/icr-identify-age-related-conditions/train.csv", index_col="Id")


X = dataset
X = X.drop(["Class"], axis=1) 
X = cleaning(X)
y = dataset["Class"]


%%time

classifier.fit(X, y)


X_test = pd.read_csv("/kaggle/input/icr-identify-age-related-conditions/test.csv", index_col="Id")
X_test = cleaning(X_test)


%%time

y_pred = classifier.predict_proba(X_test)


y_pred  = rebalance(y_pred)
class_0 = y_pred[:,0]
class_1 = 1-class_0


sample_submission = pd.read_csv("/kaggle/input/icr-identify-age-related-conditions/sample_submission.csv")
sample_submission["class_0"] = class_0
sample_submission["class_1"] = class_1
sample_submission.to_csv('submission.csv',index=False)




