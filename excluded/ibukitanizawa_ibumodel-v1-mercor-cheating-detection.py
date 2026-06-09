import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn
import networkx as nx
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

# Path
path = ("")
test_path = ("")
social_graph_path = ("")

# Graph analysis
graph_raw = pd.read_csv(social_graph_path)
G = nx.from_pandas_edgelist(
    graph_raw, 
    source='user_a', 
    target='user_b',
    create_using=nx.Graph() 
)
degree_centrality = nx.degree_centrality(G)
pagerank = nx.pagerank(G)


degree_series = pd.Series(degree_centrality, name='graph_degree')
pagerank_series = pd.Series(pagerank, name='graph_pagerank')


graph_features = pd.concat([degree_series, pagerank_series], axis=1)
graph_features = graph_features.reset_index().rename(columns={'index': 'user_hash'})

# Data loading
raw_data = pd.read_csv(path)
test_data = pd.read_csv(test_path)

raw_data = pd.merge(raw_data, graph_features, on='user_hash', how='left')
test_data = pd.merge(test_data, graph_features, on='user_hash', how='left')

data = raw_data.dropna(subset=['is_cheating'])

# Data processing
basic_youso = ["feature_001","feature_002","feature_003","feature_004","feature_005","feature_006","feature_007","feature_008","feature_009","feature_010","feature_011","feature_012","feature_013","feature_014","feature_015","feature_016","feature_017","feature_018"]
graph_youso = ['graph_degree', 'graph_pagerank']
feature = basic_youso + graph_youso

x = data[feature]
y = data.is_cheating
y = y.astype(int)
x_test = test_data[feature]

# Weight adjustment of elements
count_neg = (data['is_cheating'] == 0).sum()
count_pos = (data['is_cheating'] == 1).sum()

scale_pos_weight_value = count_neg / count_pos

print(f"non cheat (0) : {count_neg}")
print(f"cheat (1) : {count_pos}")
print(f"scale_pos_weight: {scale_pos_weight_value:.2f}")

# fit
model = xgb.XGBClassifier(
    objective='binary:logistic',
    max_depth=3,
    scale_pos_weight=scale_pos_weight_value,
    tree_method='hist' 
)

model.fit(x,y)

pred_test = model.predict(x_test)

# predict
pred_test_proba = model.predict_proba(x_test)[:, 1]

outnum = test_data["user_hash"]

output = pd.DataFrame({"user_hash": outnum, "prediction": pred_test_proba})

output.to_csv("", index=False)




