# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv('/kaggle/input/cayleypy-cube1-submit/combined_submhgjhissio6n.csv',index_col = 0 )
df


df.to_csv('submission.csv')


df.iloc[150,0]#.split('.')


df.iloc[151,0]#.split('.')


for k in range(150,150+60):
    s = df.iloc[k,0]
    l = len(s.split('.'))
    print(l)


fn = '/kaggle/input/santa-2023/puzzles.csv'
dp = pd.read_csv(fn,index_col = 0)
print('Total puzzles:', dp.shape[0])
print()
dp['puzzle_type_brief'] = dp['puzzle_type'].apply(lambda x: x.split('_')[0])
display(dp['puzzle_type_brief'].value_counts() )
print()
# display(dp['puzzle_type'].value_counts() )
dp['state_len'] = dp['solution_state'].apply(lambda x: len(x.split(';')) )
dp['state_n_unique_symb'] = dp['solution_state'].apply(lambda x: len(np.unique(x.split(';'))) )
dp_groupby_save = dp.groupby('puzzle_type').agg(count=('puzzle_type','count'), state_len =('state_len','mean'), state_median_unique_symb = ('state_n_unique_symb','median'),
      state_min_unique_symb = ('state_n_unique_symb','min'), state_max_unique_symb = ('state_n_unique_symb','max'),
     state_nunique_unique_symb = ('state_n_unique_symb','nunique'),).astype(int).sort_values('count', ascending = False)
display(dp_groupby_save)  # ['state_len'].value_counts() )
print()

display(dp['num_wildcards'].value_counts() )
print()


display(dp.head(50))
display(dp.tail(50))

display(dp.sort_values('num_wildcards', ascending = False).head(50))


display(dp.iloc[150:210,:])


mask = dp['num_wildcards'].iloc[150:200] ==0


len( df.iloc[150:200][mask])


df.iloc[150:200][mask]


LL = len(df.iloc[150:200][mask])
list_lens = []
for i in range(LL):
    s = df.iloc[150:200][mask].iloc[i,0]
    l = len( s.split('.'))
    print(l)
    list_lens.append(l)
print()
print( list_lens     )
    




