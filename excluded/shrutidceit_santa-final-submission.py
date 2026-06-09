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


perms = pd.read_csv('/kaggle/input/santa-shuffle-permutation/submission.csv')
bs = pd.read_csv('/kaggle/input/santa-beam-search/BS.csv')
hc = pd.read_csv('/kaggle/input/santa-hill-climbing-final/HC.csv')
sa = pd.read_csv('/kaggle/input/santa-simulated-annealing-final/SA.csv')


bs


perms


hc


sample_submission = pd.read_csv('/kaggle/input/santa-2024/sample_submission.csv')


sample_submission[['BS_text','BS_score']] = bs[['BS_text','BS_score']]
sample_submission[['HC_text','HC_score']] = hc[['HC_text','HC_score']]


sample_submission


sample_submission.loc[sample_submission['BS_score'] <= sample_submission['HC_score'], 'text_BH_f'] = sample_submission.loc[sample_submission['BS_score'] <= sample_submission['HC_score'], 'BS_text']
sample_submission.loc[sample_submission['HC_score'] <  sample_submission['BS_score'], 'text_BH_f'] = sample_submission.loc[sample_submission['HC_score'] <  sample_submission['BS_score'], 'HC_text']

sample_submission.loc[sample_submission['BS_score'] <= sample_submission['HC_score'], 'score_BH_f'] = sample_submission.loc[sample_submission['BS_score'] <= sample_submission['HC_score'], 'BS_score']
sample_submission.loc[sample_submission['HC_score'] <  sample_submission['BS_score'], 'score_BH_f'] = sample_submission.loc[sample_submission['HC_score'] <  sample_submission['BS_score'], 'HC_score']


sample_submission = sample_submission[['id','text_BH_f','score_BH_f']]


sample_submission


sa


sample_submission[['SA_text','SA_score']] = sa[['SA_text','SA_score']]
sample_submission


sample_submission.loc[sample_submission['score_BH_f'] <= sample_submission['SA_score'], 'text_BHS_f'] = sample_submission.loc[sample_submission['score_BH_f'] <= sample_submission['SA_score'], 'text_BH_f']
sample_submission.loc[sample_submission['SA_score'] <  sample_submission['score_BH_f'], 'text_BHS_f'] = sample_submission.loc[sample_submission['SA_score'] <  sample_submission['score_BH_f'], 'SA_text']

sample_submission.loc[sample_submission['score_BH_f'] <= sample_submission['SA_score'], 'score_BHS_f'] = sample_submission.loc[sample_submission['score_BH_f'] <= sample_submission['SA_score'], 'score_BH_f']
sample_submission.loc[sample_submission['SA_score'] <  sample_submission['score_BH_f'], 'score_BHS_f'] = sample_submission.loc[sample_submission['SA_score'] <  sample_submission['score_BH_f'], 'SA_score']


sample_submission


sample_submission = sample_submission[['id','text_BHS_f','score_BHS_f']]


#sample_submission[['perm_text','perm_score']] = perms[['perm_text','perm_score']]
sample_submission


#sample_submission.loc[sample_submission['score_BHS_f'] <= sample_submission['perm_score'], 'text'] = sample_submission.loc[sample_submission['score_BHS_f'] <= sample_submission['perm_score'], 'text_BHS_f']
#sample_submission.loc[sample_submission['perm_score'] <  sample_submission['score_BHS_f'], 'text'] = sample_submission.loc[sample_submission['perm_score'] <  sample_submission['score_BHS_f'], 'perm_text']

#sample_submission.loc[sample_submission['score_BHS_f'] <= sample_submission['perm_score'], 'score'] = sample_submission.loc[sample_submission['score_BHS_f'] <= sample_submission['perm_score'], 'score_BHS_f]
#sample_submission.loc[sample_submission['perm_score'] <  sample_submission['score_BHS_f'], 'score'] = sample_submission.loc[sample_submission['perm_score'] <  sample_submission['score_BHS_f'], 'perm_score']


#sample_submission


sample_submission = sample_submission[['id', 'text_BHS_f']].rename(columns={'text_BHS_f': 'text'})




sample_submission


perms


sample_submission.loc[0, "text"] = perms.loc[0, "text"]


sample_submission


sample_submission.to_csv('submission.csv', index=False)

