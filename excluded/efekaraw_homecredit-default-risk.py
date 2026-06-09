# 1. PACKAGES
# -----------------------------------------------------------
# Base
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Model
from lightgbm import LGBMClassifier

# Configuration
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:,.2f}'.format


df = pd.read_feather("../input/homecredit-default-risk-step-by-step-1st-notebook/applications_traintest_feather")
pos = pd.read_feather("../input/homecredit-default-risk-step-by-step-1st-notebook/poscashbalance_agg_feather")
bb = pd.read_feather("../input/homecredit-default-risk-step-by-step-1st-notebook/bureau_bureaubalance_agg_feather")
cc = pd.read_feather("../input/homecredit-default-risk-step-by-step-1st-notebook/cc_feather")
ins = pd.read_feather("../input/homecredit-default-risk-step-by-step-1st-notebook/installments_payments_agg_feather")
prev = pd.read_feather("../input/homecredit-default-risk-step-by-step-1st-notebook/previous_applications_agg_feather")

print(df.shape, pos.shape, bb.shape, cc.shape, ins.shape, prev.shape)

for i in [pos, bb, cc, ins, prev]:
    df = pd.merge(df, i , how = "left", on = "SK_ID_CURR")
    
print(df.shape)

del pos, bb, ins, cc, prev


# Train-Test Split
df.columns = list(map(lambda x: str(x).replace(" ", "_").replace("-", "_").replace("_/_", "_").upper(), df.columns))
import re
df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))


train = df[df.TARGET.isnull() == False]
test = df[df.TARGET.isnull()]

x_train = train.drop(["TARGET", "SK_ID_CURR"], axis = 1)
x_test = test.drop(["TARGET", "SK_ID_CURR"], axis = 1)
y_train = train.TARGET


# LightGBM parameters found by Bayesian optimization
clf = LGBMClassifier(
    nthread=4,
    n_estimators=10000,
    learning_rate=0.02,
    num_leaves=34,
    colsample_bytree=0.9497036,
    subsample=0.8715623,
    max_depth=8,
    reg_alpha=0.041545473,
    reg_lambda=0.0735294,
    min_split_gain=0.0222415,
    min_child_weight=39.3259775,
    silent=-1,
    verbose=-1, )

clf.fit(x_train, y_train, eval_set=[(x_train, y_train)],
        eval_metric='auc', verbose=200)



submission = pd.DataFrame({
    "SK_ID_CURR": test.SK_ID_CURR,
    "TARGET": clf.predict_proba(x_test)[:,1]
})
submission.to_csv("submission.csv", index = None)

