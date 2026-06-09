!pip install -q torch_geometric rdkit


import torch
import pandas as pd 
import numpy as np 
from tqdm import tqdm
from torch_geometric import utils 
from torch_geometric.data import Data, DataLoader, Batch

train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
train


def generate_graphs(df, is_train=True):
    graphs = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc='Generating Graphs'):
        try:
            # Baseline function for generating the graphs using torch_geometric
            graph = utils.from_smiles(row['SMILES'][1:])
            
            # Adding the labels as graph feature for future predictions
            if is_train:
                graph.labels = row[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].values
            graphs.append(graph)
            
        except:
            print(f"Error on {i} molecule")
            
    return graphs 


# Creating train and test graphs
train_graphs = generate_graphs(train)
test_graphs = generate_graphs(test, is_train=False)


from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader  

# Variables for generating dataloaders 
batch_size = 32
test_size = 0.1
suffle = True

# Data split
train_graphs, val_graphs = train_test_split(
    train_graphs, 
    test_size=test_size, 
    random_state=4564,
    shuffle=True
)

# Torch_geometric dataloaders 
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)



# Iterating the dataloader for extracting dynamic graphs 
batched_graphs = next(iter(train_loader))

# Converting dymacic graphs to separated torch_geometric graphs
graphs_list = Batch.to_data_list(batched_graphs)


import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data

def plot_pyg_graph(data, node_labels=True, edge_labels=True, figsize=(8, 6)):
    
    # Extracting edge data
    edge_index = data.edge_index
    edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

    # Creation of networkX graph
    G = nx.Graph()
    G.add_edges_from(edge_index.t().tolist())

    # Generating node positions
    pos = nx.spring_layout(G, seed=42)

    # Node Drowing
    nx.draw(G, pos, with_labels=node_labels, node_color='skyblue', node_size=500, font_size=10)

    # Generating labels for edge_attr
    edge_attr = edge_attr.cpu().numpy()
    edge_labels_dict = {
        (u, v): f"{attr}" for (u, v), attr in zip(edge_index.t().tolist(), edge_attr)
    }
    # Edge Drowing
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_dict, font_size=8)

    plt.title("Torch Geometric Graph")
    plt.gcf().set_size_inches(figsize)
    plt.axis('off')
    plt.show()
    

# Potting example for our first molecule
plot_pyg_graph(graphs_list[0])

