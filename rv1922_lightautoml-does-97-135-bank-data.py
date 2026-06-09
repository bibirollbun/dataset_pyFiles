pip install LightAutoML


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task
from sklearn.metrics import roc_auc_score
from lightautoml.report.report_deco import ReportDeco, ReportDecoUtilized
from lightautoml.addons.tabular_interpretation import SSWARM
import warnings
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv", index_col='id')
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv", index_col='id')
submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
original = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv',sep=';')


train.head()


cat_cols = train.select_dtypes(include=['object']).columns.tolist()
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col]  = le.transform(test[col].astype(str))
    le_dict[col] = le


X = train.drop(columns=['y'])
y = train['y']


task = Task('binary')
roles = {
    'target': 'y'
}
automl = TabularAutoML(
    task = task,
    timeout = 5000,
    cpu_limit = 6,
    reader_params = {'n_jobs': 6, 'cv': 5, 'random_state': 42}
)


out_of_fold_predictions = automl.fit_predict(train, roles = roles, verbose = 2)
                                             
X_test = train.drop('y',axis=1)
y_test = train['y']


test_predictions = automl.predict(X_test).data[:, 0]  

roc_auc = roc_auc_score(y_test, test_predictions)

print(f"ROC-AUC: {roc_auc:.4f}")


predictions = automl.predict(test).data[:, 0]

submission_ids = submission['id']


predictions


submission = pd.DataFrame({
    'id': submission_ids,
    'loan_status': predictions  
})


submission


submission.to_csv('submission.csv', index=False)

print("File Saved!!")

