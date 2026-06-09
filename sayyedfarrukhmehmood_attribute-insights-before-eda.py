import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings
warnings.filterwarnings('ignore')

train=pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
train_demograph = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")    


train.loc[train['sequence_id']=='SEQ_009199'].loc[:,['sequence_id','sequence_counter']].head()


# demographic information of a particular subject
train_demograph.loc[train_demograph.subject=='SUBJ_040733']


behavior_mapping = {
    'Relaxes and moves hand to target location': 'Transition',
    'Moves hand to target location': 'Transition',
    'Hand at target location': 'Gesture',
    'Performs gesture': 'Gesture'
}


train.orientation.value_counts()


train.sequence_type.value_counts(normalize=True)*100


train.loc[train.phase=='Transition'].behavior.unique()


train.loc[train.phase=='Gesture'].behavior.unique()


train.gesture.unique()




