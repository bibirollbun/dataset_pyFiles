!pip install polars

## GPU Version

!pip install torch==1.12.0+cu116 -f https://download.pytorch.org/whl/cu116/torch_stable.html  
!pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
!pip install pyg-lib -f https://data.pyg.org/whl/torch-1.12.0+cu116.html   
## CPU Version

#!pip install torch==1.12.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
#!pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cpu.html


import numpy as np
import gc
import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import GCNConv,SAGEConv,GAE
from torch_geometric.data import Data
from tqdm import tqdm
import polars as pl
from collections import defaultdict
from sklearn.preprocessing import StandardScaler


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


node2vec_embeddings  = np.load("/kaggle/input/gnn-outputs/node2vec_embeddings.npy")


aid_features = pl.read_parquet("/kaggle/input/graph-edges-features-agg/aid_features.parquet").fill_null(0)
aid_features_agg = pl.read_parquet("/kaggle/input/graph-edges-features-agg/aids_features_aggregation.parquet").fill_null(0)
aid_features_all = aid_features.join(aid_features_agg,on="aid",how="inner").drop("aid").to_numpy()


features_and_embeddings = np.concatenate((node2vec_embeddings,aid_features_all),axis=1)


std = StandardScaler().fit(features_and_embeddings)

node_features_scaled = std.transform(features_and_embeddings)


del features_and_embeddings,aid_features_all,aid_features_agg,aid_features
gc.collect()


del node2vec_embeddings
gc.collect()


#edges_tensor = torch.load("/kaggle/input/graph-edges-features-agg/otto-graph-edges.pt") # extracts all edges even duplicated one to biased the sampling !!
#edges_tensor = torch.load("/kaggle/input/graph-edges-features-agg/all_edges_without_duplicates_90m.pt") 
edges_tensor = torch.load("/kaggle/input/otto-top-neighbors/top_k10_neighbors.pt")


data = Data(x=torch.tensor(node_features_scaled),
            edge_index=edges_tensor,
             )
data.n_id = torch.arange(data.num_nodes)


del edges_tensor,node_features_scaled#,features_and_embeddings,aid_features_all,aid_features_agg,aid_features
gc.collect()


data


from torch_geometric.loader import NeighborLoader

gSAGE_loader = NeighborLoader(
    data,
    # Sample 30 neighbors for each node for 2 iterations
    num_neighbors=[10,10],#, neg_sample_size
    # Use a batch size of 128 for sampling training nodes
    batch_size=512,
)


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, 60, aggr="mean",project=False) # cached only for transductive learning
        self.conv2 = SAGEConv(60, 32, aggr="sum",project=False) # cached only for transductive learning

    def forward(self, x, edge_index):
        x = torch.nn.ELU()(self.conv1(x, edge_index))

        return  torch.nn.ELU()(self.conv2(x, edge_index))


out_channels = 32
num_features = data.x.shape[1]
model = GAE(GCNEncoder(num_features, out_channels))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005,weight_decay=1e-5)


 def train(loader):
        total_loss = 0
        for subgraph in tqdm(loader):
            optimizer.zero_grad()
            z = model.encode(subgraph.x.float().to(device),subgraph.edge_index.to(device))
            loss = model.recon_loss(z, pos_edge_index=subgraph.edge_index.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader), model



for epoch in range(0,10):
    
    loss,model = train(gSAGE_loader)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
torch.save(model,"graphSage_model")


model = torch.load("graphSage_model")


np_embeddings = np.zeros((data.num_nodes,32))
for subgraph in tqdm(gSAGE_loader):
    np_embeddings[subgraph.input_id] = model.encoder(subgraph.x.float().to(device),subgraph.edge_index.to(device)).cpu().detach().numpy()[:len(subgraph.input_id)]


del data, model
gc.collect()
del gSAGE_loader
gc.collect()
torch.cuda.empty_cache()


## Check the sampling encoding
'''
embeddings_all_batchs = model.encode(data.x.float().to(device),data.edge_index.to(device)).detach().cpu().numpy()[0]
embeddings_all_batchs[0]
np_embeddings[0]
'''


%%time
from annoy import AnnoyIndex

index = AnnoyIndex(32, 'angular')

for idx,idx_embedding in enumerate(np_embeddings):
    index.add_item(idx, idx_embedding)
    
index.build(10)



del np_embeddings
gc.collect()



def evaluate(path,mode="validation",n_neighbors=20):


    test = pl.read_parquet(path)

    session_types = ['clicks', 'carts', 'orders']
    test_session_AIDs = test.to_pandas().reset_index(drop=True).groupby('session')['aid'].apply(list)
    test_session_types = test.to_pandas().reset_index(drop=True).groupby('session')['type'].apply(list)

    del test
    gc.collect()
    labels = []

    type_weight_multipliers = {0: 1, 1: 6, 2: 3}

    for AIDs, types in zip(test_session_AIDs, test_session_types):
        if len(AIDs) >= 20:
                # if we have enough aids (over equals 20) we don't need to look for candidates! we just use the old logic
            weights=np.logspace(0.1,1,len(AIDs),base=2, endpoint=True)-1
            aids_temp=defaultdict(lambda: 0)
            for aid,w,t in zip(AIDs,weights,types): 
                aids_temp[aid]+= w * type_weight_multipliers[t]

            sorted_aids=[k for k, v in sorted(aids_temp.items(), key=lambda item: -item[1])]
            labels.append(sorted_aids[:20])
        else:
            # here we don't have 20 aids to output -- we will use word2vec embeddings to generate candidates!
            AIDs = list(dict.fromkeys(AIDs[::-1]))

            # let's grab the most recent aid
            most_recent_aid = AIDs[0]

            # and look for some neighbors!
            nns = [i for i in index.get_nns_by_item(most_recent_aid, n_neighbors)[1:]]


            labels.append((AIDs+nns)[:n_neighbors])

    labels_as_strings = [' '.join([str(l) for l in lls]) for lls in labels]

    predictions = pd.DataFrame(data={'session_type': test_session_AIDs.index, 'labels': labels_as_strings})

    prediction_dfs = []

    for st in session_types:
        modified_predictions = predictions.copy()
        modified_predictions.session_type = modified_predictions.session_type.astype('str') + f'_{st}'
        prediction_dfs.append(modified_predictions)

    sub = pd.concat(prediction_dfs).reset_index(drop=True)
    
    del prediction_dfs, predictions,labels_as_strings, labels, test_session_types,test_session_AIDs
    gc.collect()
    if mode=="test":
        sub.to_csv("submission.csv",index=False)
        return sub
    else:

        sub['labels_2'] = sub['labels'].apply(lambda x : [int(s) for s in x.split(' ')])
        submission = pd.DataFrame()
        submission['session'] = sub.session_type.apply(lambda x: int(x.split('_')[0]))
        submission['type'] = sub.session_type.apply(lambda x: x.split('_')[1])
        submission['labels'] = sub.labels_2.apply(lambda x : [item for item in x[:] ]) #.apply(lambda x: [int(i) for i in x.split(',')[:20]])
        test_labels = pd.read_parquet('/kaggle/input/otto-train-and-test-data-for-local-validation/test_labels.parquet')
        test_labels = test_labels.merge(submission, how='left', on=['session', 'type'])
        del sub,submission
        gc.collect()
        gc.collect()
        test_labels['hits'] = test_labels.apply(lambda df: len(set(df.ground_truth).intersection(set(df.labels))), axis=1)
        test_labels['gt_count'] = test_labels.ground_truth.str.len().clip(0,20)
        recall_per_type = test_labels.groupby(['type'])['hits'].sum() / test_labels.groupby(['type'])['gt_count'].sum() 
        score = (recall_per_type * pd.Series({'clicks': 0.1, 'carts': 0.30, 'orders': 0.60})).sum()

        return score



path = "/kaggle/input/otto-train-and-test-data-for-local-validation/test.parquet"
validation_score = evaluate(path,mode="validation",n_neighbors=19)
print(validation_score)


path = "/kaggle/input/otto-full-optimized-memory-footprint/test.parquet"
test_submission = evaluate(path,mode="test",n_neighbors=20)




