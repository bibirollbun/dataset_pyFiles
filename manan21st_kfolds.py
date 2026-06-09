import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import model_selection



def create_folds(data, num_splits):
    data["kfold"] = -1

    kf = model_selection.StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.label.values)):
        data.loc[v_, 'kfold'] = f

    return data



df = pd.read_csv("/kaggle/input/lmsys-chatbot-arena/train.csv")

# prepare label for model
df.loc[:, 'label'] = np.argmax(df[['winner_model_a','winner_model_b','winner_tie']].values, axis=1)

# Display data
df.head()



df_5 = create_folds(df, num_splits=5)
df_5.to_csv("train_5folds.csv", index=False)
print(df_5['kfold'].value_counts())
df_5.head()



df_10 = create_folds(df, num_splits=10)
df_10.to_csv("train_10folds.csv", index=False)
print(df_10['kfold'].value_counts())
df_10.head()


