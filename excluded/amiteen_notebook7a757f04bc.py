from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch

data_path = "/kaggle/input/smallest-representation-of-a-dataset-what-is-it"
data_indices = np.load(Path(data_path)/"data_indices.npy")
gradient_indicators = np.load(Path(data_path)/"gradient_indicators.npy")


device = 'cpu'
N = 10000

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


sample_indices = np.random.choice(range(N), size=(512,), replace=False)
sample_indices


from sklearn.manifold import MDS
sample_cohesive_scores = ((cohesive_scores[sample_indices,:])[:,sample_indices])
sample_noncohesive_scores = ((noncohesive_scores[sample_indices,:])[:,sample_indices])
X = 1.-np.abs((sample_cohesive_scores)/((sample_cohesive_scores)+(sample_noncohesive_scores)+.5)).numpy()
X[range(len(X)),range(len(X))]=0
y = np.ones(512)

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


n_simulations = 1000
rank_score = np.ones((1, N))
edge_ws = np.abs((cohesive_scores)/((cohesive_scores)+(noncohesive_scores)+.5)).numpy()
edge_ws[range(len(edge_ws)),range(len(edge_ws))]=0

for _ in tqdm(range(n_simulations)):
    rank_score = np.matmul(rank_score, edge_ws)
    rank_score = rank_score/rank_score.sum()
rank_score = rank_score.reshape(-1)
rank_score


plt.hist(rank_score*10000)
plt.show()


np.save("rank_score.npy", rank_score)


points = (np.arange(N)[rank_score<np.percentile(rank_score, 1000/N*100, axis=0)])


submission = pd.DataFrame({"id": list(range(len(points))),"indices":points})
submission.to_csv("submission.csv", index=False)


points




