import numpy as np 
import pandas as pd


sub1=pd.read_csv("/kaggle/input/ps-s5-e4-division-attention/submission.csv")
sub2=pd.read_csv("/kaggle/input/predict-podcast-listening-time-eda-ml/submission.csv")


sub=pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
sub['Listening_Time_minutes']=  0.5500 * sub1['Listening_Time_minutes'] +\
                                0.4500 * sub2['Listening_Time_minutes'] 


sub.to_csv('submission.csv', index=False)




