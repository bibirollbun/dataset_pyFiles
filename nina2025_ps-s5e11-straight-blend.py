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


df_92600 = pd.read_csv('/kaggle/input/8-november-2025-ps-s5e11/submission 0.92600.csv')
df_92620 = pd.read_csv('/kaggle/input/4-november-2025-ps-s5e11/submission 0.92620.csv')
df_92668 = pd.read_csv('/kaggle/input/4-november-2025-ps-s5e11/submission 0.92668.csv')

df1 = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')

df1['loan_paid_back'] = \
    df_92600['loan_paid_back'] * 0.33333 + \
    df_92620['loan_paid_back'] * 0.33333 + \
    df_92668['loan_paid_back'] * 0.33334

df1


df_92603 = pd.read_csv('/kaggle/input/8-november-2025-ps-s5e11/submission 0.92603.csv')
df_92632 = pd.read_csv('/kaggle/input/8-november-2025-ps-s5e11/submission 0.92632.csv')
df_92643 = pd.read_csv('/kaggle/input/03-november-2025-ps-s5e11/submission 0.92643.csv')

df2 = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')

df2['loan_paid_back'] = \
    df_92603['loan_paid_back'] * 0.33333 + \
    df_92632['loan_paid_back'] * 0.33333 + \
    df_92643['loan_paid_back'] * 0.33334

df2


df_92684 = pd.read_csv('/kaggle/input/03-november-2025-ps-s5e11/submission 0.92684.csv')
df_92683 = pd.read_csv('/kaggle/input/03-november-2025-ps-s5e11/submission 0.92683.csv')
df_92672 = pd.read_csv('/kaggle/input/03-november-2025-ps-s5e11/submission 0.92672.csv')
df_92601 = pd.read_csv('/kaggle/input/03-november-2025-ps-s5e11/submission 0.92601.csv')

df3 = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')

df3['loan_paid_back'] = \
    df_92684['loan_paid_back'] * 0.25 + \
    df_92683['loan_paid_back'] * 0.25 + \
    df_92672['loan_paid_back'] * 0.25 + \
    df_92601['loan_paid_back'] * 0.25

df3


df_92664 = pd.read_csv('/kaggle/input/8-november-2025-ps-s5e11/submission 0.92664.csv')
df_92677 = pd.read_csv('/kaggle/input/4-november-2025-ps-s5e11/submission 0.92677.csv')
df_92655 = pd.read_csv('/kaggle/input/03-november-2025-ps-s5e11/submission 0.92655.csv')
df_92657 = pd.read_csv('/kaggle/input/8-november-2025-ps-s5e11/submission 0.92657.csv')

df4 = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')

df4['loan_paid_back'] = \
    df_92664['loan_paid_back'] * 0.25 + \
    df_92677['loan_paid_back'] * 0.25 + \
    df_92655['loan_paid_back'] * 0.25 + \
    df_92657['loan_paid_back'] * 0.25

df4


df_92643 = pd.read_csv('/kaggle/input/03-november-2025-ps-s5e11/submission 0.92643.csv')
df_92712 = pd.read_csv('/kaggle/input/8-november-2025-ps-s5e11/submission 0.92712.csv')
df_92603 = pd.read_csv('/kaggle/input/8-november-2025-ps-s5e11/submission 0.92603.csv')
df_92353 = pd.read_csv('/kaggle/input/8-november-2025-ps-s5e11/submission 0.92353.csv')

df5 = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')

df5['loan_paid_back'] = \
    df_92643['loan_paid_back'] * 0.25 + \
    df_92712['loan_paid_back'] * 0.25 + \
    df_92603['loan_paid_back'] * 0.25 + \
    df_92353['loan_paid_back'] * 0.25

df5


df1_0_92760 = pd.read_csv('/kaggle/input/ps-s5e11-h-blend-third-glance-1/submission.csv')
df2_0_92756 = pd.read_csv('/kaggle/input/ps-s5e11-h-blend-third-glance-2/submission.csv')
df3_0_92600 = pd.read_csv('/kaggle/input/8-november-2025-ps-s5e11/submission 0.92600.csv')

df7 = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')

df7['loan_paid_back'] = \
    df1_0_92760['loan_paid_back'] * 0.500 + \
    df2_0_92756['loan_paid_back'] * 0.493 + \
    df3_0_92600['loan_paid_back'] * 0.007

df7


df1_0_92760 = pd.read_csv('/kaggle/input/ps-s5e11-h-blend-third-glance-1/submission.csv')
df2_0_92761 = pd.read_csv('/kaggle/input/ps-s5e11-cycle-1/submission.csv')

df8 = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')

df8['loan_paid_back'] = \
    df1_0_92760['loan_paid_back'] * 0.50 + \
    df2_0_92761['loan_paid_back'] * 0.50

df8


df1_0_92762 = pd.read_csv('/kaggle/input/ps-s5e11-h-blend-third-glance-1/submission.csv')
df2_0_92558 = pd.read_csv('/kaggle/input/ps5e11-pytorch-nn-baseline-with-gqa/submission_9.csv')

df9 = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')

df9['loan_paid_back'] = \
    df1_0_92762['loan_paid_back'] * 0.9995 + \
    df2_0_92558['loan_paid_back'] * 0.0005

display(pd.merge(df9, df1_0_92762, on='id'))

df9


df1_0_92756 = pd.read_csv('/kaggle/input/11-november-2025-ps-s5e11/submission 0.92756.csv')
df2_0_92740 = pd.read_csv('/kaggle/input/11-november-2025-ps-s5e11/submission 0.92740.csv')
df3_0_92558 = pd.read_csv('/kaggle/input/11-november-2025-ps-s5e11/submission 0.92558.csv')

df10 = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')

df10['loan_paid_back'] = \
    df1_0_92756['loan_paid_back'] * 0.600 + \
    df2_0_92740['loan_paid_back'] * 0.230 + \
    df3_0_92558['loan_paid_back'] * 0.170

df = pd.merge(df1_0_92756, pd.merge(df2_0_92740, df3_0_92558, on='id'), on='id')

df = df.rename(columns={
    'loan_paid_back'  :'0_92756',
    'loan_paid_back_x':'0_92740',
    'loan_paid_back_y':'0_92558'}
)

display(df)

df10


df1_0_92764 = pd.read_csv('/kaggle/input/8-november-2025-ps-s5e11/submission 0.92764.csv')
df2_0_92766 = pd.read_csv('/kaggle/input/8-november-2025-ps-s5e11/submission 0.92766.csv')


df11 = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')

df11['loan_paid_back'] = \
    df1_0_92764['loan_paid_back'] * 0.03 + \
    df2_0_92766['loan_paid_back'] * 0.97

df = pd.merge(df1_0_92764, df2_0_92766, on='id')

df = df.rename(columns={
    'loan_paid_back_x':'0_92764',
    'loan_paid_back_y':'0_92766'}
)

display(df)

df11


df = df1  # LB=0.92_698
df = df2  # LB=0.89_088
df = df3  # LB=0.92_732
df = df4  # LB=0.92_694
df = df5  # LB=0.92_661
df = df7  # LB=0.92_758
df = df8  # LB=0.92_761
df = df9  # LB=0.92_760
df = df10 # LB=0.92_734

df = df11


df.to_csv('submission.csv',index=False)
df

