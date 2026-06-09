# Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Read data
train = pd.read_csv('/kaggle/input/tst-day-2-upsolving/train.csv')

X = train.drop(columns=['id']).values







def_feat=train[["sliding_tackle", "defensive_awareness", "interceptions","standing_tackle","heading_accuracy" ]]
atack_feat=train[["penalties", "positioning","shot_power", "finishing","volleys" ]]
passing=train[["short_passing", "long_passing", "vision", "fk_accuracy", "curve", "crossing"]] 
gk_feat=train[["gk_kicking", "gk_diving", "gk_handling"]]
phy=train[["strength", "stamina", "jumping", "aggression"]]
pace=train[["acceleration", "sprint_speed"]]
dribbling=train[["dribbling", "agility", "balance", "ball_control", "reactions", "composure"]]
train_ext=pd.DataFrame()
for name, df in [
    ("def", def_feat),
    ("attack", atack_feat),
    ("passing", passing),
    ("gk", gk_feat),
    ("phy", phy),
    ("pace", pace),
    ("dribbling", dribbling)
]:
    p = PCA(n_components=1)
    train_ext[name] = p.fit_transform(df).ravel()


from sklearn.cluster import AgglomerativeClustering
final_model=KMeans(n_clusters=3)




final_model.fit(train_ext)



plt.scatter(train_ext["def"], train_ext["attack"], c=final_model.labels_)


submission = pd.DataFrame({
    'id': train['id'],
    'cluster': final_model.labels_
})
submission.to_csv('submission.csv', index=False)

