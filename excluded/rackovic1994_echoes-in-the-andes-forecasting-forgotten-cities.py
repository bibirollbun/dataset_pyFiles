import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sb
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
from sklearn.linear_model import SGDRegressor
import cartopy.crs as ccrs
import cartopy.feature as cfeature


adjecency_matrix = np.load(r'/kaggle/input/perudata3/Peru_adjecency_matrix.npy')
print("\nadjecency_matrix shape : ", adjecency_matrix.shape)
distances_matrix = np.load(r'/kaggle/input/perudata3/Peru_distance_matrix.npy')
print("\ndistances_matrix shape : ", distances_matrix.shape)
print("min value: ", distances_matrix.min())
print("max value: ", distances_matrix.max())
altitude_diff_matrix = np.load(r'/kaggle/input/perudata3/Peru_alt_diff_matrix.npy')
print("\naltitude_diff_matrix shape : ", altitude_diff_matrix.shape)
print("min value: ", altitude_diff_matrix.min())
print("max value: ", altitude_diff_matrix.max())

altitudes = np.load(r'/kaggle/input/perudata3/Peru_altitudes.npy')
print("\naltitudes shape : ", altitudes.shape)
print("min value: ", altitudes.min())
print("max value: ", altitudes.max())
altitudes[altitudes<0]=0

temporal_occurance = np.load(r'/kaggle/input/perudata3/Peru_temporal_occurance_matrix.npy').T
print("\ntemporal_occurance shape : ", temporal_occurance.shape)

river_distances = np.load(r'/kaggle/input/perudata3/Peru_river_distances.npy')
print("\nriver_distances shape : ", river_distances.shape)
print("min value: ", river_distances.min())
print("max value: ", river_distances.max())

river_alt_diff = np.load(r'/kaggle/input/perudata3/Peru_river_alt.npy')
print("\nriver_alt_diff shape : ", river_alt_diff.shape)
print("min value: ", river_alt_diff.min())
print("max value: ", river_alt_diff.max())

locations = np.load(r'/kaggle/input/perudata3/Peru_locations_matrix.npy')
print("\nlocations shape : ", locations.shape)

N = len(adjecency_matrix)
print('\nNumber of settlements: ',N)


dist_min = distances_matrix*1
dist_min[dist_min==0] = 1000
order = np.argsort(dist_min.min(1))
plt.figure(figsize=(17,4))
plt.bar(np.arange(N),distances_matrix.max(1)[order],color='r',label='max distances')
plt.bar(np.arange(N),dist_min.min(1)[order],color='g',label='min distances',alpha=0.5)
plt.ylabel('distance')
plt.legend()
plt.xticks([])
plt.title('Min/Max Distances Between Neighbors')
plt.show()


# These variable values will be used for scaling the respective features later
MAX_D = 150 # max distance
MAX_A = altitudes.max() # max altitude 
MAX_RD = 50 # max distance to river


settlement_graph = nx.Graph(adjecency_matrix)
distances_matrix[adjecency_matrix>0] = (MAX_D-distances_matrix[adjecency_matrix>0])/MAX_D
settlement_graph_weighted =  nx.Graph(distances_matrix)
MAX_DG = adjecency_matrix.sum(1).max()  # max degree
MAX_WDG = distances_matrix.sum(1).max() # max weighted degree


order = np.argsort(adjecency_matrix.sum(1))
plt.figure(figsize=(17,4))
plt.bar(np.arange(N),distances_matrix.sum(1)[order],color='r',label='weighted degree')
plt.bar(np.arange(N),adjecency_matrix.sum(1)[order],color='y',alpha=0.5,label='degree')
plt.legend()
plt.xticks([])
plt.title('Connection Cardinality')
plt.show()
print('Max degree: ', MAX_DG)
print('Max weighted degree: ', MAX_WDG)


order = np.argsort(altitudes)
alt_diff_vec = altitude_diff_matrix.sum(1)/adjecency_matrix.sum(1)
plt.figure(figsize=(17,4))
plt.bar(np.arange(N),altitudes[order],color='r',label='altitude')
plt.bar(np.arange(N),alt_diff_vec[order],color='b',alpha= 0.5,label=' mean altitude change')
plt.ylabel('meters')
plt.xticks([])
plt.legend()
plt.title('Settlement Altitudes and Altitude Changes')
plt.show()
MAX_AD = alt_diff_vec.max() # max altitude difference
print('Max settlement altitude: ', MAX_A)
print('Max average altitude change: ', MAX_AD)


fig, ax = plt.subplots(2,2,figsize=(18,9))

ax[0,0].spines[['right', 'top']].set_visible(False)
ax[0,0].bar(np.arange(N),sorted(river_distances,reverse=True))
ax[0,0].set_title("Distance to river")
ax[0,0].set_ylabel("distance (km)")
ax[0,0].set_xticks([])

river_distances[river_distances>MAX_RD] = MAX_RD
ax[1,0].spines[['right', 'top']].set_visible(False)
ax[1,0].bar(np.arange(N),sorted(river_distances,reverse=True))
ax[1,0].set_title("Distance to river - clipped ")
ax[1,0].set_ylabel("distance (km)")
ax[1,0].set_xticks([])

ax[0,1].spines[['right', 'top']].set_visible(False)
ax[0,1].set_title("Altitude change to river")
ax[0,1].bar(np.arange(N),sorted(river_alt_diff,reverse=True),color='tab:green')
ax[0,1].set_ylabel('meters')
ax[0,1].set_xticks([])

river_alt_diff = np.log(1+river_alt_diff)
MAX_RDD = river_alt_diff.max() # max altitude difference between setllemtn and a river

ax[1,1].spines[['right', 'top']].set_visible(False)
ax[1,1].set_title("Altitude change to river - logarithmed")
ax[1,1].bar(np.arange(N),sorted(river_alt_diff,reverse=True),color='tab:green')
ax[1,1].set_ylabel('log-scale meters')
ax[1,1].set_xticks([])

plt.show()


degrees = adjecency_matrix.sum(1)
order = np.argsort(degrees)
closeness_centrality = np.array(list(nx.closeness_centrality(settlement_graph).values())) # Notice that higher values of closeness indicate higher centrality.
betweenness_centrality = np.array(list(nx.betweenness_centrality(settlement_graph).values())) #  the shortest-path betweenness centrality for nodes.
pagerank = np.array(list(nx.pagerank(settlement_graph).values()))


fig, ax = plt.subplots(2,4,figsize=(18,9))

ax[0,0].spines[['right', 'top']].set_visible(False)
ax[0,0].barh(np.arange(N),degrees[order])
ax[0,0].set_yticks([])
ax[0,0].set_title("Node degree values")
ax[0,0].set_xlabel("degree")

ax[0,1].spines[['right', 'top']].set_visible(False)
ax[0,1].barh(np.arange(N),closeness_centrality[order])
ax[0,1].set_yticks([])
ax[0,1].set_title("Node closeness centrality")
ax[0,1].set_xlabel("centrality")

ax[0,2].spines[['right', 'top']].set_visible(False)
ax[0,2].barh(np.arange(N),betweenness_centrality[order])
ax[0,2].set_yticks([])
ax[0,2].set_title("Node betweenness centrality")
ax[0,2].set_xlabel('centrality')

ax[0,3].spines[['right', 'top']].set_visible(False)
ax[0,3].barh(np.arange(N),pagerank[order])
ax[0,3].set_yticks([])
ax[0,3].set_title("Page rank")
ax[0,3].set_xlabel('rank')

ax[1,0].spines[['right', 'top']].set_visible(False)
ax[1,0].hist(degrees,bins=25,color='g')
ax[1,0].set_title("Histogram of node degrees")
ax[1,0].set_ylabel("count")
ax[1,0].set_xlabel("degree")

ax[1,1].spines[['right', 'top']].set_visible(False)
ax[1,1].hist(closeness_centrality,bins=25,color='g')
ax[1,1].set_title("Histogram of closeness centrality")
ax[1,1].set_ylabel("count")
ax[1,1].set_xlabel("centrality")

ax[1,2].spines[['right', 'top']].set_visible(False)
ax[1,2].hist(betweenness_centrality,bins=25,color='g')
ax[1,2].set_title("Histogram of betweenness")
ax[1,2].set_ylabel("count")
ax[1,2].set_xlabel("centrality")

ax[1,3].spines[['right', 'top']].set_visible(False)
ax[1,3].hist(pagerank,bins=25,color='g')
ax[1,3].set_title("Histogram of page rank")
ax[1,3].set_ylabel("count")
ax[1,3].set_xlabel("rank")

plt.show()




degrees_weighted = distances_matrix.sum(1)
order = np.argsort(degrees_weighted)
closeness_centrality_weighted = np.array(list(nx.closeness_centrality(settlement_graph_weighted).values())) # Notice that higher values of closeness indicate higher centrality.
betweenness_centrality_weighted = np.array(list(nx.betweenness_centrality(settlement_graph_weighted).values())) #  the shortest-path betweenness centrality for nodes.
pagerank_weighted = np.array(list(nx.pagerank(settlement_graph_weighted).values()))


fig, ax = plt.subplots(1,4,figsize=(21,5))

ax[0].spines[['right', 'top']].set_visible(False)
ax[0].scatter(degrees, degrees_weighted,color='k',alpha=0.5)
ax[0].set_title("Node degree values")
ax[0].set_xlabel("degree")
ax[0].set_ylabel("weighted degree")

ax[1].spines[['right', 'top']].set_visible(False)
ax[1].scatter(closeness_centrality, closeness_centrality_weighted,color='k',alpha=0.5)
ax[1].set_title("Node closeness centrality")
ax[1].set_xlabel("closeness centrality")
ax[1].set_ylabel("weighted closeness centrality")

ax[2].spines[['right', 'top']].set_visible(False)
ax[2].scatter(betweenness_centrality, betweenness_centrality_weighted,color='k',alpha=0.5)
ax[2].set_title("Node betweenness centrality")
ax[2].set_xlabel("betweenness centrality")
ax[2].set_ylabel("weighted betweenness centrality")

ax[3].spines[['right', 'top']].set_visible(False)
ax[3].scatter(pagerank, pagerank_weighted,color='k',alpha=0.5)
ax[3].set_title("Page rank")
ax[3].set_xlabel("pagerank")
ax[3].set_ylabel("weighted pagerank")

plt.show()

MAX_PGR = np.max(pagerank)
MAX_WPGR = np.max(pagerank_weighted)
MAX_CC = closeness_centrality.max()
MAX_BC = betweenness_centrality.max()


# ALTITUDES  - CONSTANT
altitudes = altitudes/MAX_A
# RIVER DISTANCES - CONSTANT
river_distances = river_distances/MAX_RD
# RIVER ALT DIFF - CONSTANT
river_alt_diff = river_alt_diff/MAX_RDD


def feature_compilation(graph:nx.Graph, weighted_graph:nx.Graph, temporal_occurance:np.array, adjecency_matrix:np.array, distances_matrix:np.array, altitude_diff_matrix:np.array,  max_dg:int, max_wdg:int, max_cc:int, max_bc:int, max_pgr:int, max_wpgr:int, max_ad:int):

    """
    Compiles features and targets.
    """
    def temp_subgraph(graph:nx.Graph, weighted_graph:nx.Graph, t:int, temporal_occurance:np.array, adjecency_matrix:np.array, distances_matrix:np.array, altitude_diff_matrix:np.array):
        """
        Subsample nodes that should occur in timestamp t, and extract the features of the corresponding subgraph.
    
        -------
        :param graph: Unweighted nonrestricted graph, with all the settlemetns and connections.
        :param weighted_graph: Weighted nonrestricted graph, with all the settlemetns and weighted connections.
        :param t: Observed time stamp.
        :param temporal_occurance: Matrix of temporal occurances - each row contains ones at the poistions of the settlements that should exist at a given time stamp, and zeros otherwise.
        :param adjecency_matrix: Nonrestricted adjacency matrix.
        :param distances_matrix: Nonrestricted ditances/proximities matrix.
        :param altitude_diff_matrix: Matrix of the same shape as adjacency matrix, where entreis are altitude changes on the path from i to j.
        
        -------
        :returns: indices active at time t and features of the subgraph -- node degrees, weighted degrees, closeness and bitweenness centrality, weighted and nonweighted pagerank and altitude differences     
        """
        
        t_indices = np.where(temporal_occurance[t])[0]
        
        t_degrees = adjecency_matrix[t_indices][:,t_indices].sum(0)
        t_weights = distances_matrix[t_indices][:,t_indices].sum(0)
        t_graph = graph.subgraph(t_indices).copy()
        t_graph_weighted = weighted_graph.subgraph(t_indices).copy()
        t_closeness_centrality = np.array(list(nx.closeness_centrality(t_graph).values()))
        t_betweenness_centrality = np.array(list(nx.betweenness_centrality(t_graph).values()))
        t_pagerank = np.array(list(nx.pagerank(t_graph).values()))
        t_pagerank_weighted = np.array(list(nx.pagerank(t_graph_weighted).values()))
        t_degrees1 = 1*t_degrees
        t_degrees1[t_degrees1==0]=1
        t_alt_diff = altitude_diff_matrix[t_indices][:,t_indices].sum(0)/t_degrees1
    
        return t_indices, t_degrees, t_weights, t_closeness_centrality, t_betweenness_centrality, t_pagerank, t_pagerank_weighted, t_alt_diff
    
    def temp_feature_matrix(N:int, t_indices:np.array, t_degrees:np.array, t_weights:np.array, t_closeness_centrality:np.array, t_betweenness_centrality:np.array, t_pagerank:np.array, t_pagerank_weighted:np.array, t_alt_diff:np.array, max_dg:float, max_wdg:float, max_cc:float, max_bc:float, max_pgr:float, max_wpgr:float, max_ad:float) -> np.array:
        """
        Takes the outputs of temp_subgraph, and builds a correpsonding feature matrix. 
    
        -------
        :param N: Number of nodes/settlemnts
        :param t_indices: indices active at time t
        :param t_degrees: node degrees of the subgraph at time t
        :param t_weights: weighted degrees
        :param t_closeness_centrality: Closeness centrality of the subgraph nodes
        :param t_betweenness_centrality: Betweenness centrality of the subgraph nodes
        :param t_pagerank: Pageranks of the subgraph nodes
        :param t_pagerank_weighted: Pageranks centrality of the weighted subgraph nodes
        :param t_alt_diff: Altitude changes between the subgraph nodes
        :param max_dg: Scaling facotr for t_degrees
        :param max_wdg: Scaling facotr for t_weights
        :param max_cc: Scaling facotr for t_closeness_centrality
        :param max_bc: Scaling facotr for t_betweenness_centrality
        :param max_pgr: Scaling facotr for t_pagerank
        :param max_wpgr: Scaling facotr for t_pagerank_weighted
        :param max_ad: Scaling facotr for t_alt_diff
    
        -------
        :retruns: Matrix with scaled features
        """
        t_matrix = np.zeros((7,N))
        t_matrix[0,t_indices] += t_degrees/max_dg
        t_matrix[1,t_indices] += t_weights/max_wdg
        t_matrix[2,t_indices] += t_closeness_centrality/max_cc
        t_matrix[3,t_indices] += t_betweenness_centrality/max_bc
        t_matrix[4,t_indices] += t_pagerank/max_pgr
        t_matrix[5,t_indices] += t_pagerank_weighted/max_wpgr
        t_matrix[6,t_indices] += t_alt_diff/max_ad
        return t_matrix
    
    N = len(adjecency_matrix)
    T = len(temporal_occurance)
    degrees_matrix = np.zeros((T,N))
    features_tensor = np.zeros((T,N,7))
    for t in range(T):
        t_indices, t_degrees, t_weights, t_closeness_centrality, t_betweenness_centrality, t_pagerank, t_pagerank_weighted, t_alt_diff = temp_subgraph(settlement_graph, settlement_graph_weighted, t, temporal_occurance, adjecency_matrix, distances_matrix, altitude_diff_matrix)
        t_matrix = temp_feature_matrix(N, t_indices, t_degrees, t_weights, t_closeness_centrality, t_betweenness_centrality, t_pagerank, t_pagerank_weighted, t_alt_diff, max_dg, max_wdg, max_cc, max_bc, max_pgr, max_wpgr, max_ad)
        degrees_matrix[t,t_indices] += t_degrees
        features_tensor[t] += t_matrix.T
    
    constant_features_tensor = np.stack((np.tile(altitudes,(T,1)).T,np.tile(river_distances,(T,1)).T,np.tile(river_alt_diff,(T,1)).T)).T
    features = np.concatenate((constant_features_tensor, features_tensor),2)
    features = features[:-1].reshape(-1,10)
    features = pd.DataFrame(features,columns=['Degree','Weight','CC','BC','Rank','W Rank', 'Alt Diff', 'Altitude', 'River Dist', 'River Alt Diff'])
    
    targets = (degrees_matrix[1:]-degrees_matrix[:-1]).astype(int)
    targets = targets * (temporal_occurance[:-1] > 0) * (temporal_occurance[1:] > 0)
    targets = targets.reshape(-1)
    features['Target'] = targets
    features['Target Sign'] = np.sign(features['Target'])

    return features, N



features, N = feature_compilation(settlement_graph, settlement_graph_weighted, temporal_occurance, adjecency_matrix, distances_matrix, altitude_diff_matrix,  MAX_DG, MAX_WDG, MAX_CC, MAX_BC, MAX_PGR, MAX_WPGR, MAX_AD)


sb.pairplot(features[::2].drop(columns=['Target','Target Sign']), corner=True)
plt.show()


fig, ax = plt.subplots(figsize=(5,5))
ax.spines[['right', 'top','left']].set_visible(False)
subtargets = features[features['Target']!=0]['Target'].values
print(f'There are {len(features)} data points in total, ')
print(f'but only {len(subtargets)} of them have a non-zero target value,')
print(f'which is {round(100*len(subtargets) /len(features),2)} percent of all the entries.')
print(f'Out of these, {len(subtargets[subtargets>0])} are poistive, and {len(subtargets[subtargets<0])} negative')
bins = np.arange(min(subtargets), max(subtargets) + 2) - 0.45
counts, _, patches = plt.hist(subtargets, bins=bins, width=0.9)
# Annotate each bar with the count
for count, patch in zip(counts, patches):
    if count!=0:
        plt.text(patch.get_x() + patch.get_width()/2, count + 0.05, int(count),
             ha='center', va='bottom', fontsize=9)
# plt.hist(subtargets,bins=bins)
plt.xlabel('target value')
plt.yticks([])
plt.xticks(np.unique(subtargets))
#plt.ylabel('count')
plt.title("Non-zero Targets' Distribution")
plt.show()


sb.pairplot(features[features['Target Sign'] != 0].drop(columns=['Target']),hue='Target Sign', corner=True, palette={-1: 'orange', 1: 'g'})
plt.show()


positive_indices = np.where(features['Target']>0)[0]
negative_indices = np.where(features['Target']<0)[0]
zero_indices     = np.where(features['Target']==0)[0]
PINUM = 60
NINUM = 40
ZINUM = 300
NUM_ITER = 10000
TOLERANCE = 0#.0001


reg = SGDRegressor(max_iter=1, tol=None, learning_rate='invscaling',eta0=0.01)
coef0 = np.zeros(10)
residuals = []
coef_res = []
converged = False
counter = 0
while not converged:
    indices = np.concatenate([np.random.choice(positive_indices,PINUM), np.random.choice(negative_indices,NINUM), np.random.choice(zero_indices,ZINUM)])
    np.random.shuffle(indices)
    X_batch = features.iloc[indices].drop(columns=['Target','Target Sign']).values
    y_batch = features.iloc[indices]['Target'].values
    reg.partial_fit(X_batch, y_batch)
    y_pred = reg.predict(X_batch).round()
    res = np.mean((y_batch - y_pred)**2)
    residuals.append(res)
    coef1 = np.copy(reg.coef_)
    c_res = np.sum(np.abs(coef1-coef0))
    #c_res = np.sum(np.square(coef1-coef0))
    if counter>1:
        coef_res.append(c_res)
    coef0 = np.copy(coef1)
    counter += 1
    if c_res <= TOLERANCE or counter >= NUM_ITER:
        converged = True

coef_df = pd.Series(reg.coef_, index=features.columns.drop(['Target', 'Target Sign']))


fig, ax = plt.subplots(figsize=(15,4))
ax.spines[['right', 'top']].set_visible(False)
plt.plot(residuals,linewidth=0.5)
plt.title('Residuals Convergence')
plt.xlabel('iterations')
plt.show()

fig, ax = plt.subplots(figsize=(15,4))
ax.spines[['right', 'top']].set_visible(False)
plt.plot(coef_res,linewidth=0.5)
plt.title('Coefficients Convergence')
plt.xlabel('iterations')
plt.show()

fig, ax = plt.subplots(figsize=(4,4))
plt.title('LR Coefficients')
ax.spines[['right', 'top']].set_visible(False)
coef_df.plot.bar(color='orange')
plt.show()


y_pred = reg.predict(features.drop(columns=['Target','Target Sign']).values).round()
y_true = features['Target'].values
potential_sites = np.unique(np.where(y_pred - y_true>0)[0]//N)


fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAKES)
ax.add_feature(cfeature.RIVERS)

green_flag = True
red_flag = True
for i in range(len(locations)):
    (lat, lon) = locations[i]
    if red_flag:
        ax.scatter(lon, lat, marker='o',s=10, color='brown', transform=ccrs.Geodetic(),label='Archeological Site')
        red_flag = False
    else:
        ax.scatter(lon, lat, marker='o',s=10, color='brown', transform=ccrs.Geodetic())
    if i in potential_sites:
        if green_flag:
            ax.scatter(lon, lat, marker='o',s=200, color='green',alpha=0.25, transform=ccrs.Geodetic(),label='Region of interest')
            green_flag = False
        else:
            ax.scatter(lon, lat, marker='o',s=200, color='green',alpha=0.25, transform=ccrs.Geodetic())

plt.title('Archeological Sites in Peru')
plt.legend()
plt.show()




