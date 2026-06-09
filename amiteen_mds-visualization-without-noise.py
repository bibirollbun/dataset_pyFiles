from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
np.random.seed(922)
N = 1000                    # Number of data points
B = 1000                     # Batch size
D = 7                     # Number of clusters
I = 10000                    # Number of training iterations
noise_rate = 0.0            # Probablity of being noise
permutate_indices = np.random.permutation(np.arange(N))
gradient_indicators = list()
data_indices = list()

# Simulate the training process of a learning model on a dataset of 10000 data points. 
for _ in tqdm(range(I)):
    signs = np.random.choice([-1,1],size=(D,),replace=True)
    cluster_indices = np.random.choice(range(N),size=(B,),replace=False)
    is_noise = 1*(np.random.rand(len(cluster_indices))<noise_rate)
    noise = 2*(np.random.rand(len(cluster_indices))>.5)-1 # noising signs
    indices = (permutate_indices[cluster_indices]).reshape(1,-1)
    indicator = ((signs[cluster_indices%D])+is_noise*(noise-(signs[cluster_indices%D]))).reshape(1,-1)
    data_indices.append(indices)
    gradient_indicators.append(indicator)
data_indices = np.concatenate(data_indices,axis=0)
gradient_indicators = np.concatenate(gradient_indicators,axis=0)


np.save("data_indices.npy", data_indices)
np.save("gradient_indicators.npy", gradient_indicators)


device = 'cpu'

cohesive_scores = torch.from_numpy(np.zeros((N,N))).long()
noncohesive_scores = torch.from_numpy(np.zeros((N,N))).long()
for i, (indice, indicator) in tqdm(enumerate(zip(data_indices, gradient_indicators))):
    indices = torch.from_numpy(indice).long().to(device)
    side_indicator = torch.from_numpy(indicator).long().to(device)
    ones = torch.ones(len(side_indicator),len(side_indicator)).long().to(device)
    event_indicator = torch.mul(ones,side_indicator.view(1,-1))
    event_indicator = torch.mul(event_indicator,side_indicator.view(-1,1))
    r_indices_sq = indices.view(-1,1).expand(len(side_indicator),len(side_indicator))
    c_indices_sq = indices.view(1,-1).expand(len(side_indicator),len(side_indicator))
    indices = indices.cpu().reshape(-1)
    side_indicator = side_indicator.cpu().reshape(-1)
    event_indicator = event_indicator.cpu().reshape(-1)
    r_indices_sq = r_indices_sq.cpu().reshape(-1)
    c_indices_sq = c_indices_sq.cpu().reshape(-1)

    cohesive_scores[r_indices_sq,c_indices_sq] = cohesive_scores[r_indices_sq,c_indices_sq]+(event_indicator>0).long()
    noncohesive_scores[r_indices_sq,c_indices_sq] = noncohesive_scores[r_indices_sq,c_indices_sq]+(event_indicator<0).long()


import matplotlib.pyplot as plt
plt.style.use("ggplot")


from sklearn.manifold import MDS
X = 1.-np.abs((cohesive_scores.numpy()/(cohesive_scores.numpy()+noncohesive_scores.numpy()))-.5)
X[range(len(X)),range(len(X))]=0
y = np.ones(N)

print('Original Dimesnion of X = ', X.shape)
# Create an MDS model with the desired number of dimensions
# Number of dimensions for visualization
n_components = 2 
mds = MDS(n_components=n_components)
 
# Fit the MDS model to your data
X_reduced = mds.fit_transform(X)
 
print('Dimesnion of X after MDS = ',X_reduced.shape)
 
# Visualize the reduced data
plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=plt.cm.get_cmap("jet", 2))
plt.colorbar(label='Class Label', ticks=range(2))
plt.title("MDS Visualization of CIFAR-10 Dataset")
plt.xlabel("MDS Dimension 1")
plt.ylabel("MDS Dimension 2")
plt.savefig('/kaggle/working/manifold_visualization.png')
plt.show()




