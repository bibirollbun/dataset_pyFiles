import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport , compare


train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv' , index_col = 'id')
test_df  = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv' , index_col = 'id')
original_df = pd.read_csv('/kaggle/input/calories-burnt-prediction/calories.csv' , index_col = 'User_ID')


original_df.rename(columns = {'Gender' : 'Sex'} , inplace = True)


Train_Report = ProfileReport(train_df , title = 'Train' , explorative = True)
Test_Report = ProfileReport(test_df , title = 'Test' ,  explorative=True)
Original_Report = ProfileReport(original_df , title = 'Original' ,  explorative=True)


Train_Report


Test_Report


Original_Report


Comparison_Report = compare([Train_Report, Original_Report])


Comparison_Report.to_notebook_iframe()




