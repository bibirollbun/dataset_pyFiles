# Standard Kaggle imports
import numpy as np
import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objs as go
# Check what files are available in the Kaggle input directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# Define the root path to Meta Kaggle in Kaggle environment
meta_kaggle_path = '/kaggle/input/meta-kaggle/'

# Helper to get full file path
def get_csv(filename):
    return os.path.join(meta_kaggle_path, filename)


users = pd.read_csv(get_csv('Users.csv'))
print(f"Loaded {len(users)} users")

kernels = pd.read_csv(get_csv('Kernels.csv'))
print(f"Loaded {len(kernels)} kernels")

kernel_sources = pd.read_csv(get_csv('KernelVersionKernelSources.csv'))
print(f"Loaded {len(kernel_sources)} kernel sources")

user_followers = pd.read_csv(get_csv('UserFollowers.csv'))
print(f"Loaded {len(user_followers)} user followers")


# Fork edges
forks = kernels[kernels['ForkParentKernelVersionId'].notnull()].copy()
mentor_lookup = kernels[['Id', 'AuthorUserId']].drop_duplicates().rename(
    columns={'Id': 'ForkParentKernelVersionId', 'AuthorUserId': 'MentorUserId'}
)
forks = forks.merge(mentor_lookup, on='ForkParentKernelVersionId', how='left')
fork_edges = forks.dropna(subset=['MentorUserId'])
fork_edges = fork_edges.loc[fork_edges['MentorUserId'] != fork_edges['AuthorUserId'], ['MentorUserId', 'AuthorUserId']]
fork_edges = fork_edges.rename(columns={'MentorUserId': 'source', 'AuthorUserId': 'target'})
fork_edges['weight'] = 2

print(f"Fork edges: {len(fork_edges)}")
# Kernel source reuse edges
kernel_version_map = kernels[['CurrentKernelVersionId', 'AuthorUserId']].drop_duplicates()
kernel_version_map = kernel_version_map.rename(columns={'CurrentKernelVersionId': 'KernelVersionId', 'AuthorUserId': 'TargetUserId'})

kernel_sources = kernel_sources.merge(
    kernel_version_map.rename(columns={'KernelVersionId': 'SourceKernelVersionId', 'TargetUserId': 'SourceUserId'}),
    on='SourceKernelVersionId', how='left'
).merge(
    kernel_version_map.rename(columns={'KernelVersionId': 'KernelVersionId', 'TargetUserId': 'TargetUserId'}),
    on='KernelVersionId', how='left'
)

source_edges = kernel_sources.dropna(subset=['SourceUserId', 'TargetUserId'])
source_edges = source_edges.loc[source_edges['SourceUserId'] != source_edges['TargetUserId'], ['SourceUserId', 'TargetUserId']]
source_edges = source_edges.rename(columns={'SourceUserId': 'source', 'TargetUserId': 'target'})
source_edges['weight'] = 1.5

print(f"Source reuse edges: {len(source_edges)}")
# Follower edges
follower_edges = user_followers.rename(columns={'FollowingUserId': 'source', 'UserId': 'target'})
follower_edges['weight'] = 1

print(f"Follower edges: {len(follower_edges)}")
# Combine all edges
edges = pd.concat([fork_edges, source_edges, follower_edges], ignore_index=True)
edges = edges.groupby(['source', 'target'], as_index=False)['weight'].sum()

print(f"Total combined edges: {len(edges)}")


active_user_ids = set(users[users['PerformanceTier'] > 0]['Id'])
edges = edges[edges['source'].isin(active_user_ids) & edges['target'].isin(active_user_ids)]

print(f"Edges after filtering inactive/bot users: {len(edges)}")


G = nx.from_pandas_edgelist(edges, source='source', target='target', edge_attr='weight', create_using=nx.DiGraph())

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

pagerank = nx.pagerank(G, weight='weight')
betweenness = nx.betweenness_centrality(G, weight='weight', k=500)
in_deg = dict(G.in_degree(weight='weight'))
out_deg = dict(G.out_degree(weight='weight'))

metrics = pd.DataFrame({
    'UserId': list(G.nodes()),
    'PageRank': [pagerank.get(n, 0) for n in G.nodes()],
    'Betweenness': [betweenness.get(n, 0) for n in G.nodes()],
    'InDegree': [in_deg.get(n, 0) for n in G.nodes()],
    'OutDegree': [out_deg.get(n, 0) for n in G.nodes()]
})
metrics = metrics.merge(users, left_on='UserId', right_on='Id', how='left')


top_mentors = metrics.sort_values('PageRank', ascending=False).head(50)
influential_contributors = top_mentors[top_mentors['PerformanceTier'] < 6]

print("Top 50 Mentors:")
display(top_mentors[['DisplayName', 'PageRank', 'PerformanceTier']])

print("Influential Contributors:")
display(influential_contributors[['DisplayName', 'PageRank', 'PerformanceTier']])


top_n_users = top_mentors['UserId']
H = G.subgraph(top_n_users)
pos = nx.spring_layout(H, seed=42, k=0.3)

# Remove layout outliers
distances = {n: np.linalg.norm(pos[n]) for n in H.nodes()}
distance_threshold = 0.5
kept_nodes = [n for n, d in distances.items() if d < distance_threshold]
H_filtered = H.subgraph(kept_nodes)
pos_filtered = {n: pos[n] for n in kept_nodes}

# Edges
edge_x, edge_y = [], []
for src, tgt in H_filtered.edges():
    x0, y0 = pos_filtered[src]
    x1, y1 = pos_filtered[tgt]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines'
)
# Nodes
influential_ids = set(influential_contributors['UserId'])
node_x = [pos_filtered[n][0] for n in H_filtered.nodes()]
node_y = [pos_filtered[n][1] for n in H_filtered.nodes()]
node_sizes = [5 + 40*pagerank.get(n, 0) for n in H_filtered.nodes()]
node_colors = ['orange' if n in influential_ids else 'blue' for n in H_filtered.nodes()]
node_labels = [users[users['Id'] == n]['DisplayName'].values[0] for n in H_filtered.nodes()]

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=node_labels,
    marker=dict(size=node_sizes, color=node_colors, line_width=2),
    hoverinfo='text'
)

# Plotly figure
fig = go.Figure(data=[edge_trace, node_trace])
fig.add_trace(go.Scatter(
    x=[None], y=[None],
    mode='markers',
    marker=dict(size=10, color='orange'),
    name='Influential Contributors'
))
fig.add_trace(go.Scatter(
    x=[None], y=[None],
    mode='markers',
    marker=dict(size=10, color='blue'),
    name='Other Top Mentors'
))
fig.update_layout(
    title='Mentor Network on Kaggle (Top 50 by Influence, Outliers Removed)',
    showlegend=True,
    hovermode='closest',
    width=800,
    height=600,
    plot_bgcolor='white',
    xaxis=dict(
        title='Mentorship Network Space (X)',
        showticklabels=False,
        showgrid=False,
        zeroline=False
    ),
    yaxis=dict(
        title='Mentorship Network Space (Y)',
        showticklabels=False,
        showgrid=False,
        zeroline=False
    )
)
fig.show()


# Label top 5% of PageRank as Top Mentors
threshold = metrics['PageRank'].quantile(0.95)
metrics['IsTopMentor'] = (metrics['PageRank'] >= threshold).astype(int)

print(metrics['IsTopMentor'].value_counts())


# Account age
users['RegisterDate'] = pd.to_datetime(users['RegisterDate'])
snapshot_date = pd.to_datetime('2025-07-01')
users['AccountAgeDays'] = (snapshot_date - users['RegisterDate']).dt.days

# Notebook count
notebook_counts = kernels.groupby('AuthorUserId').size().reset_index(name='NumNotebooks')

# Followers / Following counts
num_followers = user_followers.groupby('FollowingUserId').size().reset_index(name='NumFollowers')
num_following = user_followers.groupby('UserId').size().reset_index(name='NumFollowing')


features = metrics[['UserId', 'IsTopMentor']].merge(
    users[['Id', 'Country', 'PerformanceTier', 'AccountAgeDays']],
    left_on='UserId', right_on='Id', how='left'
).drop(columns=['Id'])

features = features.merge(notebook_counts, left_on='UserId', right_on='AuthorUserId', how='left').drop(columns=['AuthorUserId'])
features = features.merge(num_followers, left_on='UserId', right_on='FollowingUserId', how='left').drop(columns=['FollowingUserId'])
features = features.merge(num_following, left_on='UserId', right_on='UserId', how='left')

# Fill NA for counts
features = features.fillna({'NumNotebooks': 0, 'NumFollowers': 0, 'NumFollowing': 0})


# One-hot encode Country
features['Country'] = features['Country'].fillna('Unknown')
features = pd.get_dummies(features, columns=['Country'], drop_first=True)

# Train/test split
from sklearn.model_selection import train_test_split

X = features.drop(columns=['UserId', 'IsTopMentor'])
y = features['IsTopMentor']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

print(X_train.shape, X_test.shape)


from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report

model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print("AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))


import matplotlib.pyplot as plt

importance = model.feature_importances_
sorted_idx = np.argsort(importance)[::-1]
feature_names = X.columns

# Keep top 15
top_n = 15
top_features = feature_names[sorted_idx][:top_n]
top_importance = importance[sorted_idx][:top_n]

plt.figure(figsize=(8,6))
plt.barh(top_features, top_importance)
plt.gca().invert_yaxis()
plt.title("Top 15 Features for Predicting Emerging Mentors")
plt.xlabel("Importance")
plt.show()

