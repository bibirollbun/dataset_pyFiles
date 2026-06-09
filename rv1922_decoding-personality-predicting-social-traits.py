import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import h2o
from h2o.automl import H2OAutoML
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt  
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


h2o.init(max_mem_size="5G")


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


train.head()


num_col =  ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
       'Friends_circle_size', 'Post_frequency']
cat_col = ['Stage_fear', 'Drained_after_socializing']
target_col = 'Personality'


hf = h2o.H2OFrame(pd.concat([train[num_col+cat_col+[target_col]],]))
hf[target_col] = hf[target_col].asfactor()
for c in cat_col:
    hf[c] = hf[c].asfactor()


train, valid = hf.split_frame(ratios=[0.9], seed=123)


aml = H2OAutoML(
    max_runtime_secs=2000, 
    seed=1, 
    sort_metric='accuracy',
    balance_classes=True,      
    verbosity="info"
)


aml.train(x=num_col+cat_col, y=target_col, training_frame=train, validation_frame=valid)


lb = aml.leaderboard.as_data_frame()
print(lb.head(10))


hf_test = h2o.H2OFrame(test[num_col+cat_col])
for c in cat_col:
    hf_test[c] = hf_test[c].asfactor()


pred = aml.predict(hf_test)
pred_df = pred.as_data_frame()


if target_col in test:
    y_true = test[target_col].values  # 'Introvert' / 'Extrovert'
    y_pred = pred_df['predict'].values
    acc = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {acc:.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=['Introvert', 'Extrovert'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Introvert', 'Extrovert'])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()


submission[target_col] = pred_df['predict']
submission.to_csv('submission.csv', index=False)
print("File saved")


submission.head()

