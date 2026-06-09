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


# XGBoost GPU
!pip install xgboost==1.7.6


import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv("/kaggle/input/mercor-cheating-detection/train.csv")
test  = pd.read_csv("/kaggle/input/mercor-cheating-detection/test.csv")

print(train.shape, test.shape)


train.head()


null = {'Dtype':train.dtypes,
        'Total Values':len(train),
        'Unique Values': train.nunique(),
        'Null Values': train.isnull().sum(),
        '% of Null Values': np.round((train.isnull().sum()/len(train))*100,2)
    
}
pd.DataFrame(null)


# Target distribution
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(6,4))
train['is_cheating'].value_counts().plot(kind='bar')
plt.title("Target Distribution â€” is_cheating")
plt.xlabel("Cheating")
plt.ylabel("Count")
plt.show()

print(train['is_cheating'].value_counts(normalize=True))


train.describe().T


plt.figure(figsize=(12,8))
num_cols = train.select_dtypes(include=['float64','int64']).columns

corr = train[num_cols].corr()

sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.show()


plt.figure(figsize=(6,4))
train['high_conf_clean'].fillna(0).value_counts().plot(kind='bar')
plt.title("Distribution: high_conf_clean (filled NaN=0)")
plt.show()

print(train['high_conf_clean'].fillna(0).value_counts(normalize=True))


# STEP 1 â€” LOAD SOCIAL GRAPH & BUILD ADJACENCY LIST
# --------------------------------------------------------------

# Load the social graph edge list (each row = a connection between two users)
edges = pd.read_csv("/kaggle/input/mercor-cheating-detection/social_graph.csv")

# ---------------------------------------------------------------------
# WHAT IS AN ADJACENCY LIST?
# ---------------------------------------------------------------------
# An adjacency list is a data structure that represents a graph.
# For each node (user), we store a list of all users they are connected to.
#
# Example:
# If user 1 is connected to users [4, 7, 9],
# then adjacency_list[1] = [4, 7, 9]
#
# It is efficient for:
#   - Running BFS/DFS
#   - Generating random walks (Node2Vec)
#   - Finding neighbors 
#   - Graph analytics
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# STEP A â€” Collect ALL unique users present in the graph
# ---------------------------------------------------------------------
# The social graph contains two columns: user_a and user_b.
# Both sides contain user IDs, so we combine them and take unique values.

all_users = pd.Index(
    pd.concat([edges["user_a"], edges["user_b"]]).unique()
)

# Map each user_hash â†’ an integer index  (0 ... N-1)
# This is necessary because graph libraries work with integer node IDs.
uid2idx = {u: i for i, u in enumerate(all_users)}

# Number of total unique users in the graph
n = len(all_users)

# ---------------------------------------------------------------------
# STEP B â€” CREATE EMPTY ADJACENCY LIST
# ---------------------------------------------------------------------
# Initialize adjacency list with `n` empty lists.
# adj[i] will hold all neighbors of user with index i.
adj = [[] for _ in range(n)]

# ---------------------------------------------------------------------
# STEP C â€” FILL THE ADJACENCY LIST
# ---------------------------------------------------------------------
# The social graph is undirected:
# If A is connected to B, then B is also connected to A.
#
# For each row in the edges file:
#   - Convert user_a â†’ integer IA
#   - Convert user_b â†’ integer IB
#   - Add IB to IA's neighbor list
#   - Add IA to IB's neighbor list
# ---------------------------------------------------------------------

for a, b in zip(edges["user_a"], edges["user_b"]):
    ia = uid2idx[a]   # index of user A
    ib = uid2idx[b]   # index of user B

    adj[ia].append(ib)  # B is a neighbor of A
    adj[ib].append(ia)  # A is a neighbor of B

print("Adjacency list built. Total users:", n)


import numpy as np

# ------------------------------------------------------------
# Create an empty feature matrix "h0"
# ------------------------------------------------------------
# We want to build simple graph features for each user/node.
#
# h0 will be a matrix with:
#   - n rows (one row per user)
#   - 8 columns (we will create 8 graph-based features later)
#
# Example shape:
#   If we have 500,000 users --> h0.shape = (500000, 8)
#
# We initialize with zeros; we will fill each column later.
# ------------------------------------------------------------

h0 = np.zeros((n, 8), dtype=np.float32)



# ------------------------------------------------------------
# FEATURE 0 = NODE DEGREE
# ------------------------------------------------------------
# "Degree" of a node = how many connections (edges) it has.
# In the adjacency list:
#   adj[i] is a Python list containing all neighbors of user i.
#   So len(adj[i]) = number of neighbors â†’ degree of node i.
#
# We fill column 0 of h0 with degree values.
# ------------------------------------------------------------

for i in range(n):
    h0[i, 0] = len(adj[i])


device = "cuda" if torch.cuda.is_available() else "cpu"

h0 = torch.tensor(h0, device=device)


def sage_layer(features, adj):
    # ------------------------------------------------------------------
    # features: tensor of shape (n, d)
    #   - n = number of nodes
    #   - d = feature dimension per node
    #
    # adj: adjacency list
    #   - adj[i] = list of neighbor node indices for node i
    #
    # This function performs ONE GraphSAGE mean-aggregation layer.
    # ------------------------------------------------------------------

    n, d = features.shape  # number of nodes, feature dimension

    # Create an empty tensor to store the aggregated features
    agg = torch.zeros_like(features)

    # ------------------------------------------------------------------
    # MAIN LOOP: For each node i
    # ------------------------------------------------------------------
    for i in range(n):

        # --------------------------------------------------------------
        # If the node has neighbors
        # --------------------------------------------------------------
        if len(adj[i]) > 0:

            # Gather neighbor feature vectors.
            # adj[i] is a list of neighbor indices â†’ use them to index.
            #
            # Example:
            #   adj[i] = [10, 23, 45]
            #   features[adj[i]] = features[[10, 23, 45]]
            #
            # This automatically moves the neighbor vectors to the GPU
            # because "features" is already on CUDA.
            # ----------------------------------------------------------
            neigh = features[adj[i]]

            # Compute mean of neighbor features â†’ GraphSAGE "mean" aggregator
            agg[i] = neigh.mean(dim=0)

        # --------------------------------------------------------------
        # If the node has NO neighbors (isolated node)
        # --------------------------------------------------------------
        else:
            # Use the node's own features as fallback (avoid zero vector)
            agg[i] = features[i]

    return agg   # return updated node embeddings


# -------------------------------------------------------------
# Run the first GraphSAGE layer
# -------------------------------------------------------------
# h0 contains the initial graph features for each node.
# Passing h0 into sage_layer aggregates information
# from each node's immediate neighbors (1-hop neighbors).
h1 = sage_layer(h0, adj)


# -------------------------------------------------------------
# Run the second GraphSAGE layer
# -------------------------------------------------------------
# h1 now contains features enriched with information from
# 1-hop neighbors. Running GraphSAGE again lets each node
# aggregate information from 2-hop neighbors (neighbors of neighbors).
h2 = sage_layer(h1, adj)


# -------------------------------------------------------------
# Combine embeddings from all 3 stages: h0, h1, h2
# -------------------------------------------------------------
# We concatenate:
#   - h0: original features
#   - h1: 1-hop aggregated features
#   - h2: 2-hop aggregated features
#
# This makes the final representation richer because it includes
# structural information from multiple levels of the graph.
#
# dim=1 means we stack columns side-by-side.
emb = torch.cat([h0, h1, h2], dim=1)


# -------------------------------------------------------------
# Convert from PyTorch tensor â†’ NumPy array
# -------------------------------------------------------------
# emb is a PyTorch tensor, possibly stored on GPU.
# XGBoost training expects a NumPy array.
#
# .cpu() moves the data to CPU memory
# .numpy() converts it to a NumPy ndarray
embedded = emb.cpu().numpy()


# ------------------------------------------------------------
# Create a Linear Layer (Fully Connected Layer)
# ------------------------------------------------------------
# W = Linear(24 â†’ 128)
#
# Meaning:
#   - Input size: 24 (the size of emb: h0+h1+h2)
#   - Output size: 128 (we want a richer, higher-dimensional embedding)
#
# This layer learns a matrix multiplication + bias:
#     output = emb * W^T + b
#
# We move this layer to the GPU (device).
W = torch.nn.Linear(24, 128).to(device)


# ------------------------------------------------------------
# Apply the linear layer to the graph embeddings
# ------------------------------------------------------------
# emb is your (n, 24) tensor containing concatenated graph features.
# Passing it through W projects it into a new 128-dimensional space.
#
# The result is emb128: a (n, 128) tensor.
#
# .detach() removes it from PyTorch's computational graph
# (we are not training a neural network here, just generating features)
#
# .cpu() moves it to CPU memory
#
# .numpy() converts to a NumPy array so it can be merged
# with your train/test data and fed into XGBoost.
emb128 = W(emb).detach().cpu().numpy()


embedding_df = pd.DataFrame(
    emb128,
    index=all_users,
    columns=[f"g_emb_{i}" for i in range(128)]
)

train = train.merge(embedding_df, how="left",
                    left_on="user_hash", right_index=True)
test  = test.merge(embedding_df, how="left",
                    left_on="user_hash", right_index=True)

train.fillna(0, inplace=True)
test.fillna(0, inplace=True)


# ---------------------------------------------------------
# Feature selection (exclude non-features)
# ---------------------------------------------------------

ID_COL = "user_hash"
TARGET = "is_cheating"
HIGH_CONF = "high_conf_clean" # These rows were automatically predicted by the production system as "very unlikely to cheat", it's not a feature
FEATURES = [f"feature_{i:03d}" for i in range(1, 19)]


exclude = [ID_COL, TARGET, HIGH_CONF]
all_features = [c for c in train.columns if c not in exclude]

print("Number of features used:", len(all_features))
all_features[:15]




import xgboost as xgb
import numpy as np

TARGET = "is_cheating"

# Use only labeled rows (drop high_conf_clean == 1 unlabeled)
mask_labeled = ~train[TARGET].isna()

X_train = (
    train.loc[mask_labeled, all_features]
         .fillna(-999)
         .astype("float32")
         .values
)

y_train = (
    train.loc[mask_labeled, TARGET]
         .astype(int)
         .values
)

# Prepare DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)

# XGBoost parameters optimized for tabular + embeddings
params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "gpu_hist",    # ðŸ”¥ USE GPU
    "predictor": "gpu_predictor",
    "max_depth": 7,
    "eta": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "lambda": 1.5,
}

print("ðŸš€ Training XGBoost on GPU...")
model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=800,
)
print("âœ… XGBoost training complete!")



from sklearn.metrics import roc_auc_score, accuracy_score

train_pred_prob = model.predict(dtrain)
train_pred = (train_pred_prob >= 0.5).astype(int)

print("Training AUC:", roc_auc_score(y_train, train_pred_prob))
print("Training Accuracy:", accuracy_score(y_train, train_pred))



X_test = (
    test[all_features]
        .fillna(-999)
        .astype("float32")
        .values
)

dtest = xgb.DMatrix(X_test)
test_preds = model.predict(dtest)

submission = pd.DataFrame({
    "user_hash": test["user_hash"],
    "prediction": test_preds
})

submission.to_csv("submission.csv", index=False)
submission.head()


