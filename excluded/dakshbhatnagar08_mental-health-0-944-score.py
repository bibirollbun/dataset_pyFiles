# CREDITS TO: ITASPS(turtle) [https://www.kaggle.com/code/itasps/0-94397-autogluon-model-depression-health-data]


# IMPORTS
import numpy as np
import pandas as pd
import os
import random
import warnings


# WARNING SUPPRESSION (convenience)
warnings.filterwarnings('ignore')


# PRINTING ALL FILES IN KAGGLE INPUT FOR REFERENCE AND/OR DEBUGING
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# LOADING MODEL SUBMISSION + TEST/SAMPLE FILES
sub1 = pd.read_csv('/kaggle/input/fork-of-autogluon-model-depression-health-data/submission.csv')
sub2 = pd.read_csv('/kaggle/input/0-94434-ensemble-exploring-mental-health/submission.csv')
sample_submission = pd.read_csv("/kaggle/input/playground-series-s4e11/sample_submission.csv")
test_data = pd.read_csv('/kaggle/input/playground-series-s4e11/test.csv')


# ENSURING REPRODUCIBILITY OF RANDOM FUNCTION
random.seed(100)


# COUNTING HOW MANY PREDICTIONS DIFFER BETWEEN BOTH SUBMISSIONS
disagreement_count = 0
for i in range(len(sub1)):
    if sub1.iloc[i]['Depression'] != sub2.iloc[i]['Depression']:
        disagreement_count += 1

# PRINTING TOTAL DISAGREEMENTS 
print(disagreement_count)


# COUNTING HOW MANY TIMES WE CHOOSE SUB1 OVER SUB2
switched_count = 0

# i. Default to sub2's prediction, with 10% chance of switching to sub1 if there's disagreement
for i in range(len(sub1)):
    sample_submission.loc[i, 'Depression'] = sub2.iloc[i]['Depression']
    
    if sub1.iloc[i]['Depression'] != sub2.iloc[i]['Depression']:
        if random.random() < 0.1:  # 10% probability
            sample_submission.loc[i, 'Depression'] = sub1.iloc[i]['Depression']
            switched_count += 1

# ii. Printing no. of switches to sub1
print(switched_count)


# SAVING SUBMISSION.CSV
sample_submission.to_csv('submission.csv', index=False)
print('submission.csv saved.')

