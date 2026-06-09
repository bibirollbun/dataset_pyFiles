import numpy as np 
import pandas as pd 


test = pd.read_csv('/kaggle/input/nclab-competition/test.csv')


test.to_csv('submission.csv', index = False)




