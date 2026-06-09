import pandas as pd
import numpy as np


sub_1=pd.read_csv('/kaggle/input/submission-2/nn_ver_4.csv') #0.69832 roc score
sub_2=pd.read_csv('/kaggle/input/lgbm-diabetics/submission.csv') #0.70164 roc score


rank_1 = sub_1['diagnosed_diabetes'].rank(pct=True)
rank_2 = sub_2['diagnosed_diabetes'].rank(pct=True)


roc_score_1 =0.68254
roc_score_2 =0.70164


tot=roc_score_1+roc_score_2


weight_1=roc_score_1/tot
weight_2=roc_score_2/tot


final = (rank_1 * weight_1) + (rank_2 * weight_2)


np.corrcoef(sub_1['diagnosed_diabetes'], sub_2['diagnosed_diabetes'])


submission = pd.DataFrame({
    'id': sub_1['id'],
    'diagnosed_diabetes': final
})

submission.to_csv('submission.csv', index=False)
submission.head(2)

