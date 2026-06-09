!pip install dgl -q


from typing import Tuple, List

import pandas as pd
from dgl import DGLGraph, graph as graph_constructor
from torch.utils.data import Dataset
 

class GraphClassificationDataset(Dataset):
    def __init__(self, csv_path):
        super().__init__()
        
        self.data_path = csv_path
        self._read_data()

    def _read_data(self) -> Tuple[List[DGLGraph], List[int]]:
        self.graph_ids = []
        self.graphs = []
        self.labels = []
        df = pd.read_csv(self.data_path, header=0, index_col='graph_id')
        
        for graph_id, sample in df.iterrows():
            # Remember to split edges_from and edges_to
            graph = self._create_graph(
                num_nodes=sample.num_nodes,
                num_edges=sample.num_edges,
                edges_from=[int(edge) for edge in sample.edges_from.split(' ')],
                edges_to=[int(edge) for edge in sample.edges_to.split(' ')],
            )

            self.graph_ids.append(graph_id)
            self.graphs.append(graph)
            try:
                self.labels.append(sample.label)
            except:
                self.labels.append(graph_id) # just to get it later

    def _create_graph(self, num_nodes: int, num_edges: int, edges_from: List[int], edges_to: List[int]) -> DGLGraph:
        assert len(edges_from) == num_edges, 'Something is wrong with edges_from'
        assert len(edges_to) == num_edges, 'Something is wrong with edges_to'

        graph = graph_constructor((edges_from, edges_to))
        return graph
        
    def __getitem__(self, index) -> Tuple[DGLGraph, int]:
        return self.graphs[index], self.labels[index]
    
    def __len__(self):
        return len(self.labels)


train_set_raw = GraphClassificationDataset('/kaggle/input/pmldl-week-10-gnn-competition-extended/graph_classification_train.csv')
test_dataset = GraphClassificationDataset('/kaggle/input/pmldl-week-10-gnn-competition-extended/graph_classification_test.csv')


from torch.utils.data import random_split

# Set percentage of data to use as a training subset
train_ratio = 0.8

train_size = int(train_ratio * len(train_set_raw))
val_size = len(train_set_raw) - train_size

train_dataset, val_dataset = random_split(train_set_raw, (train_size, val_size))


from torch import Tensor

from dgl import batch as construct_graph_batch

def collate_graph_batch(samples: List[Tuple[DGLGraph, int]]) -> Tuple[Tensor, Tensor]:
    graphs, labels = [graph for graph, _ in samples], [label for _, label in samples]
    batched_graph = construct_graph_batch(graphs)
    return batched_graph, Tensor(labels)


from torch.utils.data import DataLoader

BATCH_SIZE = 64
NUM_CLASSES = 8

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_graph_batch, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_graph_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_graph_batch)


import torch
from torch import nn
import dgl
from dgl.nn.pytorch import GraphConv

class GraphClassifier(nn.Module):
    def __init__(self, hidden_dim, n_classes):
        super().__init__()

        input_dim = 1
        # NOTE: for educational purposes here we use 1 as input dimension. 
        # However, in production feature vector is the information about the node. 
        # For example, in social networks the feature vector could represent the user 
        # (e.g. its choices of movies)

        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()

        ## Add several layers here and classifier head (one Linear works btw)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()
        self.linear = nn.Linear(hidden_dim, n_classes)
#         self.softmax = nn.Softmax()

    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        h = graphs.in_degrees().view(-1, 1).float()
        
        # Perform graph convolution and activation function.

        # Remember that graph convs work like residuals
        # so store hidden vectors somewhere
        h = self.conv1(g, h)
        h = self.relu1(h)
        h = self.conv2(g, h)
        h = self.relu2(h)
        h = self.conv3(g, h)
        h = self.relu3(h)
        h = self.linear(h)

        graphs.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(graphs, 'h')
        return dgl.mean_nodes(graphs, 'h')


HIDDEN_DIM = 256

# Select the where to perform calculations
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Create an instance of the model and pass its weights to the device
model = GraphClassifier(HIDDEN_DIM, NUM_CLASSES).to(device)
# Set the loss function
loss_function = nn.CrossEntropyLoss()
# Set the opimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def calculate_metric(metric_fn, true_y, pred_y):
    if metric_fn != accuracy_score:
        return metric_fn(true_y, pred_y, average="macro")
    else:
        return metric_fn(true_y, pred_y)

def print_scores(p, r, f1, a, batch_size):
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")


import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm

epochs = 600

losses = []
batches = len(train_dataloader)
val_batches = len(val_dataloader)

# loop for every epoch (training + evaluation)
for epoch in range(epochs):
    total_loss = 0

    # progress bar
    progress = tqdm(enumerate(train_dataloader), desc="Loss: ", total=batches)

    # ----------------- TRAINING  --------------------
    # set model to training
    model.train()

    for i, (graphs, labels) in progress:
        pred = model(graphs)
        labels = labels.to(torch.long)
        loss = loss_function(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        total_loss += current_loss * BATCH_SIZE

        # updating progress bar
        progress.set_description("Loss: {:.4f}".format(total_loss/(i+1)))

    # releasing unceseccary memory in GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ----------------- VALIDATION  -----------------
    val_losses = 0
    precision, recall, f1, accuracy = [], [], [], []

    # set model to evaluating (testing)
    model.eval()
    with torch.no_grad():
        for i, (graphs, labels) in enumerate(val_dataloader):
            outputs = model(graphs)
            labels = labels.to(torch.long)
            val_losses += loss_function(outputs, labels).item() * BATCH_SIZE
            predicted_classes = torch.max(outputs, 1)[1]
            
            # calculate P/R/F1/A metrics for batch
            for acc, metric in zip((precision, recall, f1, accuracy),
                                   (precision_score, recall_score, f1_score, accuracy_score)):
                acc.append(
                    calculate_metric(metric, labels.cpu(), predicted_classes.cpu())
                )

    print(f"Epoch {epoch + 1}/{epochs}, training loss: {total_loss/batches}, validation loss: {val_losses/val_batches}")
    print_scores(precision, recall, f1, accuracy, val_batches)
    losses.append(total_loss/batches)


predictions = []

with torch.no_grad():
    model.eval()
    for i, (graphs, graph_ids) in enumerate(test_dataloader):
        outputs = model(graphs)

        predicted = torch.max(outputs, 1)[1]
        predictions.extend(predicted.tolist())


# generate the submission file
submission_df = pd.DataFrame(columns=['graph_id', 'label'])

submission_df['graph_id'] = test_dataset.graph_ids
submission_df['label'] = predictions

submission_df.to_csv('submission3.csv', index=None)




