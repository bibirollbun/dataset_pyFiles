import numpy as np
import pandas as pd
from time import time

# import pytorch and set dgl backend to pytorch
import os
os.environ['DGLBACKEND'] = 'pytorch'
import torch
import torch.nn  as nn
import torch.nn.functional as F

try:
    import dgl
except ModuleNotFoundError:
    !pip install dgl
    import dgl

import networkx as nx
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


transactions_df = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
num_transactions = transactions_df.shape[0]
print("Number of transactions: ", num_transactions)
print(f"Proportion of fraudulent transaction: {transactions_df.isFraud.value_counts(normalize=True)[1]:.4f}")
identity_df = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')
print("Number of transaction with ID data: ",identity_df.shape[0])
print('-'*40)
print("Dataframe head:")
display(transactions_df.head())
display(identity_df.head())


# user set the ratio
TRAIN_VAL_RATIO = 0.75

# determine number of training records
n_train = int(transactions_df.shape[0]*TRAIN_VAL_RATIO)

# train/val split : split by time, training set preceeds val set.
train_ids = transactions_df.TransactionID[:n_train]
val_ids = transactions_df.TransactionID[n_train:]


import os

output_dir = "/kaggle/working/preprocessed_data"
os.makedirs(output_dir, exist_ok=True)

# get id columns from transactions_df
id_cols = ['card1','card2','card3','card4','card5','card6','ProductCD','addr1','addr2','P_emaildomain','R_emaildomain'] 
# get categorical columns for node features
cat_cols = ['M1','M2','M3','M4','M5','M6','M7','M8','M9']   
# get features and labels
transactions_non_features = ['isFraud','TransactionDT'] + id_cols
features_cols = [col for col in transactions_df.columns if col not in transactions_non_features]
# create df for transaction node features
features_df = pd.get_dummies(transactions_df[features_cols], columns=cat_cols).fillna(0)
# take log of transaction amount
features_df['TransactionAmt'] = features_df['TransactionAmt'].apply(np.log10)
# show new distribution
features_df['TransactionAmt'].plot(kind='hist', bins=[0,0.5,1,1.5,2,2.5,3,3.5,4])
plt.title('Distribution of transaction amounts')
plt.xlabel('Log of transaction amount')
plt.show()
features_df.to_csv(os.path.join('/kaggle/working/preprocessed_data', 'features.csv'), index=False, header=True)
# create a df for the labels
labels_df = transactions_df[['TransactionID','isFraud']]
labels_df[['TransactionID', 'isFraud']].to_csv(os.path.join('/kaggle/working/preprocessed_data', 'tags.csv'), index=False, header=True)
# create a list of all node types except TransactionID
node_types = id_cols + list(identity_df.columns)
node_types.remove('TransactionID')
# join the dfs to get a table with all ID data
full_identity_df = identity_df.merge(transactions_df[id_cols+['TransactionID']], on='TransactionID', how='right')
# create a dictionary of df that will determine the edges in the graph
edge_dfs = {}
for ntype  in node_types:
    edge_dfs[ntype] = full_identity_df[['TransactionID', ntype]].dropna()
    edge_dfs[ntype].to_csv(os.path.join(output_dir, 'relation_{}_edgelist.csv').format(ntype), index=False, header=True)


# initialize the dictionary to store each ID to dgl node index dictionary
id_to_node = {}

# First get dgl indices for TransactionID/target nodes
id_to_node['target'] = dict([(v,k) for k,v in dict(transactions_df['TransactionID']).items()])

# Then cycle through the other ID types and add those to the list (dict)
for ntype in node_types:
    new_nodes_ids = edge_dfs[ntype][ntype].unique()
    new_nodes_dgl = np.arange(len(new_nodes_ids)+1)
    id_to_node[ntype] = { a:b for a,b in zip(new_nodes_ids, new_nodes_dgl)}


# for example:
id_to_node['card4']


# instantiate the edge list dictionary
edgelists = {}
num_nodes_dict = {}

for ntype in node_types:
    # prepare each edge type triple and its reverse
    edge_type = ('target','target<>'+ntype,ntype)
    rev_edge_type = (ntype, ntype+'<>target','target')
    # get list of initial nodes and destination nodes
    source_nodes = edge_dfs[ntype]['TransactionID'].apply(lambda a : id_to_node['target'][a]).to_numpy()
    destination_nodes = edge_dfs[ntype][ntype].apply(lambda a : id_to_node[ntype][a]).to_numpy()
    # add to dict
    edgelists[edge_type] = ( source_nodes, destination_nodes )
    edgelists[rev_edge_type] = (destination_nodes, source_nodes)
    # get number of nodes of this type
    num_nodes_dict[ntype] = len(np.unique(destination_nodes))
    
# add self-loops for target nodes
source_nodes = edge_dfs[ntype]['TransactionID'].apply(lambda a : id_to_node['target'][a]).to_numpy()
edgelists[('target','target<>target','target')] = (source_nodes,source_nodes)
num_nodes_dict['target'] = num_transactions


# create the graph
g = dgl.heterograph(edgelists, num_nodes_dict)


# Visualizing the metagraph (sanity check: it should be star-shaped, with the target node in the center and a node for each id feature)
meta = g.metagraph()
nx.draw(meta)


features_df.head()


# create pytorch tensor consisting of features for each node
feature_tensor = torch.from_numpy(features_df.drop('TransactionID', axis=1).to_numpy())
# add feature data to graph
g.nodes['target'].data['features'] = feature_tensor


# sanity check
# for random target node, compare expected features with the feature of the node
trial = 1515

# data from the table
orig_feat = torch.from_numpy(features_df.iloc[trial,1:].to_numpy())
# data from the graph
graph_feat = g.ndata['features']['target'][trial]

assert max(orig_feat - graph_feat) == 0
print('Feature vectors match. Check is good!')


train_mask = [ id_to_node['target'][x] for x in train_ids]
val_mask = [ id_to_node['target'][x] for x in val_ids]


labels = torch.tensor(labels_df['isFraud'].to_numpy()).float()


mean = torch.mean(g.ndata['features']['target'], axis=0)
std = torch.sqrt(torch.sum((g.ndata['features']['target'] - mean)**2, axis=0)/g.ndata['features']['target'].shape[0])


g.ndata['features']['target'] = (g.ndata['features']['target'] - mean)/std


# Conv layers
INPUT_DIM = 16
HIDDEN_DIM = 16
TARGET_OUT_DIM = 16
CONV_LAYERS = 6 # should be at least 2; only have 3 'distinct' layers, weights for the middle ones are shared with all layers except first and last.
# pre/post processing of target
TARGET_PREPROCESSING_HIDDEN_DIM = 64
TARGET_PREPROCESSING_NO_LAYERS = 3
TARGET_POSTPROCESSING_HIDDEN_DIM = 16
TARGET_POSTPROCESSING_NO_LAYERS = 4
# training
LEARNING_RATE = 0.001
LOSS_MULTIPLIER = 5
NUM_EPOCHS = 120


from dgl.nn.pytorch import HeteroGraphConv, HeteroEmbedding, GraphConv
from torch.nn import Linear

# get dimension of target features
target_feature_dim = g.ndata['features']['target'].shape[1]

# create linear embeddings for the non-target nodes into R^IN_DIM space
num_embeddings_dict = { src : g.num_nodes(src) for (src,etype,dst) in g.canonical_etypes if (dst == 'target' and src != 'target')}

# define a NN for pre/postprocessing the target data
class ff_block(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers):
        super().__init__()
        self.input_layer = Linear(in_dim, hidden_dim)
        self.hidden_layer = Linear(hidden_dim, hidden_dim)
        self.output_layer = Linear(hidden_dim, out_dim)
        self.n_layers = n_layers
        
    def forward(self, in_feats):
        h = self.input_layer(in_feats)
        h = nn.ReLU()(h)
        for i in range(1, self.n_layers):
            h = self.hidden_layer(h)
            h = nn.ReLU()(h)
        h = self.output_layer(h)        
        return h
            

# define the model class
class RGCN(nn.Module):
    def __init__(self, target_feature_dim, in_dim, hidden_dim, conv_out_dim, num_conv_layers, num_embeddings, target_pre_h_dim, target_pre_layers, target_post_h_dim, target_post_layers):
        super().__init__()
        # create dictionaries for HeteroGraphConv
        entry_module_dict = { etype : GraphConv(in_feats=in_dim, out_feats=hidden_dim) for etype in g.etypes}

        hidden_model_dict = { etype : GraphConv(in_feats=hidden_dim, out_feats=hidden_dim) for etype in g.etypes}

        final_model_dict1 = { etype : GraphConv(in_feats=hidden_dim, out_feats=conv_out_dim) for src,etype,dst in g.canonical_etypes if dst == 'target'}
        final_model_dict2 = { etype : GraphConv(in_feats=hidden_dim, out_feats=1) for src,etype,dst in g.canonical_etypes if dst != 'target'}
        final_model_dict = {**final_model_dict1, **final_model_dict2}
        
        self.num_conv_layers = num_conv_layers
        
        self.embed_layer = HeteroEmbedding(
            num_embeddings,
            in_dim
        )
        self.target_preprocessing = ff_block(
            target_feature_dim, 
            target_pre_h_dim, 
            in_dim, 
            target_pre_layers)
        self.conv1 = HeteroGraphConv(
            entry_module_dict,
            aggregate = 'sum'
        )
        self.conv2 = HeteroGraphConv(
                hidden_model_dict,
                aggregate = 'sum'
        )

        self.conv3 = HeteroGraphConv(
            final_model_dict,
            aggregate = 'sum'
        )
        self.target_postprocessing = ff_block(
            conv_out_dim, 
            target_post_h_dim, 
            1, 
            target_post_layers
        )
        
    def forward(self, graph, input_features):
        embeds = self.embed_layer({ ntype : graph.nodes(ntype) for ntype in node_types })
        input_features = input_features.to(dtype=torch.float32)
        target_features = self.target_preprocessing(input_features)
        embeds['target'] = target_features
        h = self.conv1(graph, embeds)
        h = {k: F.relu(v) for k, v in h.items()}
        for i in range(2, self.num_conv_layers):
            h = self.conv2(graph, h)
            h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv3(graph, h)
        h['target'] = self.target_postprocessing(h['target'])
        
        return h


# create the model
model = RGCN(
    target_feature_dim = target_feature_dim,
    in_dim = INPUT_DIM, 
    hidden_dim = HIDDEN_DIM, 
    conv_out_dim = TARGET_OUT_DIM,
    num_conv_layers = CONV_LAYERS,
    num_embeddings = num_embeddings_dict, 
    target_pre_h_dim = TARGET_PREPROCESSING_HIDDEN_DIM, 
    target_pre_layers = TARGET_PREPROCESSING_NO_LAYERS,
    target_post_h_dim = TARGET_POSTPROCESSING_HIDDEN_DIM,
    target_post_layers = TARGET_POSTPROCESSING_NO_LAYERS
)
print('Number of parameters in the model: ')
print(' '*33,sum(param.numel() for param in model.parameters()))


class FocalLossWithWeight(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets, weight=None):
        # BCE with logits but per-sample (no reduction)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )

        # pt = exp(-BCE)
        pt = torch.exp(-bce_loss)

        # focal loss
        focal = self.alpha * (1 - pt) ** self.gamma * bce_loss

        # apply weight vector if available
        if weight is not None:
            focal = focal * weight

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal



# prepare weight vector for loss fn
weight_vector = (torch.ones(labels[train_mask].shape)+labels[train_mask]*LOSS_MULTIPLIER).reshape((labels[train_mask].shape[0],1))
val_weight_vector = (torch.ones(labels[val_mask].shape)+labels[val_mask]*LOSS_MULTIPLIER).reshape((labels[val_mask].shape[0],1))

# Set optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = FocalLossWithWeight(alpha=0.5, gamma=2.0)   # cho training
val_loss_fn = FocalLossWithWeight(alpha=0.5, gamma=2.0)


def train_one_epoch(epoch_no, model, g, features, labels, train_mask, val_mask, threshold, return_probs=False):
    t0 = time()
    # Forward pass
    logits_dict = model(g, features)
    logits = logits_dict['target']
    del logits_dict
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
       
    labels = labels.reshape_as(preds)
    
    # compute training and validation loss
    loss = loss_fn(logits[train_mask], labels[train_mask], weight_vector[train_mask])
    with torch.no_grad():
        val_loss = val_loss_fn(logits[val_mask], labels[val_mask], val_weight_vector[val_mask])

    
    # compute accuracies
    train_acc = (preds[train_mask] == labels[train_mask]).float().mean()
    val_acc = (preds[val_mask] == labels[val_mask]).float().mean()
    
    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if return_probs:
        return loss, val_loss, train_acc, val_acc, time()-t0, probs, preds
    else:
        return loss, val_loss, train_acc, val_acc, time()-t0


# function for visualizing loss history
def loss_history_plot(loss_history, val_loss_history=None, small=False):
    if small:
        size = (5,2)
    else:
        size = (7,4)
    fig, ax =plt.subplots(1,1,figsize=size)
    ax.plot(loss_history, label='Train')
    if val_loss_history:
        ax.plot(val_loss_history, label='Val')
        ax.legend()
    plt.show()


def train(model, g, num_epochs, labels, train_mask, val_mask, threshold):
    best_val_acc = 0
    epoch_times = []
    history = {}
    history['loss'] = []
    history['val_loss'] = []
    history['val_acc'] = []
    
    features = g.nodes['target'].data['features']
    
    for epoch in range(num_epochs):
        
        if epoch < num_epochs-1:
            loss, val_loss, train_acc, val_acc, epoch_time = train_one_epoch(epoch, model, g, features, labels, train_mask, val_mask, threshold)
        else:
            loss, val_loss, train_acc, val_acc, epoch_time, final_probs, final_preds = train_one_epoch(epoch, model, g, features, labels, train_mask, val_mask, threshold, return_probs=True)
            
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['loss'].append(loss.detach().numpy())
        epoch_times.append(epoch_time)
        
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            
        if epoch % 10 == 10-1:
            loss_rate_of_change = (history['loss'][-1]-history['loss'][-5])/5
            print(f"Epoch {epoch+1} loss: {loss:.3f}, (rate of change: {loss_rate_of_change:.4f}), val accuracy: {val_acc:.3f} (best: {best_val_acc:.3f})\n\
            -- Average time per epoch: {np.mean(epoch_times):.1f}sec (last 5: {np.mean(epoch_times[-5:]):.1f}sec).\
            Estimated time to end: {(num_epochs-epoch-1)*np.mean(epoch_times[-5:])/60:.0f} mins")
            
        if epoch % 25 == 25-1:
            loss_history_plot(history['loss'][-25:], history['val_loss'][-25:], small=True)
            loss_history_plot(history['val_acc'][-25:], small=True)
            
    print('-'*60)
    print(f"Training complete. \
    Final loss: {loss:.3f}, \
    final val accuracy: {val_acc:.3f}, (best: {best_val_acc:.3f}).")
    
    return final_probs, final_preds, history


QUICK_TEST = False

if QUICK_TEST:
    n_ep = 2
else:
    n_ep = NUM_EPOCHS

probs, preds, history = train(model, g, n_ep, labels, train_mask, val_mask, 0.5)


print('Training set: mean of probabilities, predictions, and true labels:')
print(probs[train_mask].mean())
print(preds[train_mask].mean())
print(labels[train_mask].mean())

print('\nValidation set:')
print(probs[val_mask].mean())
print(preds[val_mask].mean())
print(labels[val_mask].mean())


print('Evaluation of validation set results:\n','-'*50)

print(f"Proportion of transactions predicted as fraud: {torch.mean(preds[val_mask])*100:.2f} %.")
cm = confusion_matrix(preds[val_mask], labels[val_mask])
print(cm)
fp_rate = cm[1,0]/(cm[1,0]+cm[1,1])
print(f"False positive rate: {fp_rate*100:.1f} %.")
fn_rate = cm[0,1]/(cm[0,1]+cm[0,0])
print(f"False negative rate: {fn_rate*100:.2f} %.")
tn_rate = 1-fp_rate
print(f"True negative rate: {tn_rate*100:.2f} % (specificity).")
tp_rate = 1- fn_rate
print(f"True positive rate: {tp_rate*100:.2f} % (sensitivity).")


loss_history_plot(history['loss'], history['val_loss'])
loss_history_plot(history['val_acc'])

