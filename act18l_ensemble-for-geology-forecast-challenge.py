import numpy as np 
import pandas as pd 


sub = pd.read_csv("/kaggle/input/geology-forecast-challenge-open/data/sample_submission.csv")
sub.head()


sub_42 = pd.read_csv("/kaggle/input/geo-submission/42.csv")
sub_43 = pd.read_csv("/kaggle/input/geo-submission/43.csv")
sub_44 = pd.read_csv("/kaggle/input/geo-submission/44.csv")
sub_45 = pd.read_csv("/kaggle/input/geo-submission/45.csv")
sub_46 = pd.read_csv("/kaggle/input/geo-submission/46.csv")
sub_47 = pd.read_csv("/kaggle/input/geo-submission/47.csv")
sub_48 = pd.read_csv("/kaggle/input/geo-submission/48.csv")

sub_25 = pd.read_csv("/kaggle/input/geo-submission/25.csv")


sub.iloc[:,1:]=(sub_42.iloc[:,1:]+
                sub_43.iloc[:,1:]+
                sub_44.iloc[:,1:]+
                sub_45.iloc[:,1:]+
                sub_46.iloc[:,1:]+
                sub_47.iloc[:,1:]+
                sub_48.iloc[:,1:]
               )/7
sub


sub.to_csv("submission.csv",index=None)
sub.head()




