from tqdm import tqdm
import numpy as np
np.random.seed(922)
N = 3000                    # Number of data points
B = 512                     # Batch size
D = 200                     # Number of clusters
I = 1000                    # Number of training iterations
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


def scoring(predicts):
    
    # Scoring codes
    permutate_id2id = np.zeros_like(permutate_indices)
    permutate_id2id[permutate_indices] = np.arange(N)
    
    cluster_counts = np.zeros(D,)
    cluster_ids, counts = np.unique((permutate_id2id[predicts])%D, return_counts=True)
    print(f"Counts for each cluster: {np.unique(np.concatenate([np.arange(D),(permutate_id2id[predicts])], axis=0)%D, return_counts=True)[1]-1}")
    cluster_counts[cluster_ids] = counts
    cluster_scores = np.sum((cluster_counts>0).astype(float))/len(cluster_counts)-(np.mean(cluster_counts)-1)*.5

    return cluster_scores

# Prediction
predicts = permutate_indices
scores = scoring(predicts)

# Score of the prediction (maximum is 1 if and only if, for each data cluster, there is 1 data point that represents all the data points in that cluster). 
scores


# The perfect prediction should be something like this
predicts = (permutate_indices[:(D-0)])

scoring(predicts)




