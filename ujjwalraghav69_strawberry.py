import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns


test = pd.read_csv('/kaggle/input/regression-dataset/test.csv')
train = pd.read_csv('/kaggle/input/regression-dataset/train.csv')
train.head(5)


train.shape


train.info


train.nunique()


train.isnull().sum().sort_values(ascending=False)


print(train.columns)




train = pd.read_csv('/kaggle/input/regression-dataset/train.csv')


train.drop('MaxOfUpperTRange', inplace=True, axis=1, errors='ignore')
train.drop('MinOfUpperTRange', inplace=True, axis=1, errors='ignore')
train.drop('MaxOfLowerTRange', inplace=True, axis=1, errors='ignore')
train.drop('MinOfLowerTRange', inplace=True, axis=1, errors='ignore')
train.drop('RainingDays', inplace=True, axis=1, errors='ignore')
train.drop('fruitmass', inplace=True, axis=1, errors='ignore')
train.head(5)




#start training and fitting the model from here


#to be used only after training the model on the training set.
def clean(test):
    test.drop('MaxOfUpperTRange', inplace=True, axis=1, errors='ignore')
    test.drop('MinOfUpperTRange', inplace=True, axis=1, errors='ignore')
    test.drop('MaxOfLowerTRange', inplace=True, axis=1, errors='ignore')
    test.drop('MinOfLowerTRange', inplace=True, axis=1, errors='ignore')
    test.drop('RainingDays', inplace=True, axis=1, errors='ignore')
    test.drop('fruitmass', inplace=True, axis=1, errors='ignore')


