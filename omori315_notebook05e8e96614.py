# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load



import numpy as np 
import pandas as pd 
import lightgbm as lgb
import ast # 文字列をPythonのオブジェクトとして評価するためのライブラリ
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



