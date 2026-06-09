!pip install /kaggle/input/pip-install-pyg/torch_spline_conv-1.2.2+pt25cu124-cp310-cp310-linux_x86_64.whl
!pip install /kaggle/input/pip-install-pyg/torch_sparse-0.6.18+pt25cu124-cp310-cp310-linux_x86_64.whl
!pip install /kaggle/input/pip-install-pyg/pyg_lib-0.4.0+pt25cu124-cp310-cp310-linux_x86_64.whl
!pip install /kaggle/input/pip-install-pyg/torch_cluster-1.6.3+pt25cu124-cp310-cp310-linux_x86_64.whl
!pip install /kaggle/input/pip-install-pyg/torch_geometric-2.6.1-py3-none-any.whl


import numpy as np
import pandas as pd
import networkx as nx

import torch
import pandas as pd
from torch_geometric.nn import LabelPropagation
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from torch_cluster import knn_graph


df = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
df = df.sample(2000).copy()
df.reset_index(inplace=True)
y = df.efs.copy()
num_samples = df.shape[0]
labeled_indices = np.random.choice(
    num_samples, size=int(0.2 * num_samples), replace=False) #20% labels are known


X = df[['age_at_hct', 'donor_age', 'karnofsky_score', 'comorbidity_score']].copy()
X = X.fillna(-1)
y[labeled_indices] = 2 #(unknwon labels)


# Convert DataFrame to PyTorch Geometric Data
x = torch.tensor(np.array(X), dtype=torch.float)  # Features tensor

# Build k-NN Graph (each node connects to its 5 nearest neighbors)
edge_index = knn_graph(x, k=20, loop=True)
edge_index = to_undirected(edge_index)  # Ensure bidirectional edges

# Convert labels to tensor
y_tensor = torch.tensor(y, dtype=torch.long)

# Mask Unknown Labels (-1) Before Label Propagation
mask = y_tensor.clone() != 2  # Clone y tensor

data = Data(x=x, edge_index=edge_index, y=y_tensor)

# Apply Label Propagation
propagation = LabelPropagation(num_layers=5, alpha=1.5)  # Run for 5 iterations
output = propagation(data.y, data.edge_index, mask=mask)

# Assign pseudo-labels (handling -1 values properly)
pseudo_labels = torch.argmax(output, dim=1).numpy()

df['pseudo_label'] = pseudo_labels


df[['efs', 'pseudo_label']][y==2].head(20)


import matplotlib.pyplot as plt


G = nx.Graph()
G.add_edges_from(edge_index.numpy().T)  # Convert edges to NetworkX format

# Draw Graph
plt.figure(figsize=(6, 6))
pos = nx.spring_layout(G, seed=42)

# Draw nodes
nx.draw(G, pos, node_size=10, font_size=12, edge_color="black")




