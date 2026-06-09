import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# ------------------------------
# 1. Charger les données
# ------------------------------
train = pd.read_csv("/kaggle/input/mercor-cheating-detection/train.csv")
test = pd.read_csv("/kaggle/input/mercor-cheating-detection/test.csv")

with open("/kaggle/input/mercor-cheating-detection/feature_metadata.json") as f:
    feature_meta = json.load(f)

# ------------------------------
# 2. Aperçu général
# ------------------------------
print("=== INFO ===")
print(train.info())
print("\n=== HEAD ===")
print(train.head())

# ------------------------------
# 3. Analyse des labels
# ------------------------------
label_counts = train['is_cheating'].value_counts(dropna=False)
print("\n=== Label Counts ===")
print(label_counts)

labeled = train[~train['is_cheating'].isna()]
num_cheaters = (labeled['is_cheating'] == 1).sum()
total_labeled = labeled.shape[0]
perc_cheaters = num_cheaters / total_labeled * 100
print(f"\nNombre de candidats étiquetés : {total_labeled}")
print(f"Nombre de tricheurs : {num_cheaters}")
print(f"Pourcentage de tricheurs : {perc_cheaters:.2f}%")

# ------------------------------
# 4. Analyse high_conf_clean
# ------------------------------
high_conf_counts = train['high_conf_clean'].value_counts(dropna=False)
print("\n=== high_conf_clean counts ===")
print(high_conf_counts)

num_unlabeled_high_conf = train[(train['high_conf_clean']==1) & (train['is_cheating'].isna())].shape[0]
print(f"\nNombre de candidats unlabeled mais high_conf_clean: {num_unlabeled_high_conf}")

# ------------------------------
# 5. Visualisation des labels
# ------------------------------
plt.figure(figsize=(6,4))
sns.countplot(data=labeled, x='is_cheating')
plt.title("Distribution des labels is_cheating (labeled)")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(data=train, x='high_conf_clean')
plt.title("Distribution des high_conf_clean")
plt.show()

# ------------------------------
# 6. Tableau croisé labels x high_conf_clean
# ------------------------------
cross_tab = pd.crosstab(train['is_cheating'].fillna('Unknown'), 
                        train['high_conf_clean'].fillna('Unknown'), margins=True)
print("\n=== Crosstab is_cheating x high_conf_clean ===")
print(cross_tab)

# ------------------------------
# 7. Analyse NaN et types par feature
# ------------------------------
features = [f"feature_{i:03d}" for i in range(1, 19)]  # 001 à 018
nan_stats = train[features].isna().mean() * 100

feature_types = {k:v['type'] for k,v in feature_meta.items() if k in features}
feature_info = pd.DataFrame({
    "feature": features,
    "type": [feature_types[f] for f in features],
    "missing_%": [nan_stats[f] for f in features]
})
print("\n=== Feature types et % de NaN ===")
print(feature_info)

plt.figure(figsize=(10,4))
sns.heatmap(train[features].isna(), cbar=False)
plt.title("Heatmap des NaN dans les features")
plt.show()

# ------------------------------
# 8. Statistiques descriptives features
# ------------------------------
print("\n=== Statistiques descriptives features ===")
print(train[features].describe())

# ------------------------------
# 9. Corrélations
# ------------------------------
corr_matrix = train[features].corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matrice de corrélation entre features")
plt.show()

# ------------------------------
# 10. Bar chart corrélation avec is_cheating
# ------------------------------
corr_target = labeled[features + ['is_cheating']].corr()['is_cheating'].drop('is_cheating')
corr_target.sort_values().plot(kind='barh', figsize=(8,6), color='skyblue')
plt.title("Corrélation de chaque feature avec is_cheating")
plt.xlabel("Corrélation")
plt.show()


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Charger le social graph
graph_df = pd.read_csv("/kaggle/input/mercor-cheating-detection/social_graph.csv")

# Créer le graphe
G = nx.from_pandas_edgelist(graph_df, 'user_a', 'user_b')

# Sélection d'un sous-graphe pour la visualisation (éviter trop de nodes)
sub_nodes = list(G.nodes())[:200]  # par exemple 200 premiers
sub_G = G.subgraph(sub_nodes)

plt.figure(figsize=(10,10))
pos = nx.spring_layout(sub_G, k=0.3)  # disposition “printemps” pour la toile
nx.draw(sub_G, pos, with_labels=False, node_size=50, node_color='skyblue', edge_color='gray', alpha=0.7)
plt.title("Sous-graphe social (toile d'araignée)")
plt.show()



import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# 1. Chargement et Préparation
# ------------------------------
# Assurez-vous d'avoir chargé les données nécessaires
train = pd.read_csv("/kaggle/input/mercor-cheating-detection/train.csv")
graph_df = pd.read_csv("/kaggle/input/mercor-cheating-detection/social_graph.csv")

# Créer le graphe G à partir du social_graph.csv
G = nx.from_pandas_edgelist(graph_df, 'user_a', 'user_b')

# ------------------------------
# 2. Calcul du Degré de Centralité
# ------------------------------
# Calculer le degré de chaque nœud (nombre de connexions)
degree_dict = dict(G.degree())

# Convertir le résultat en DataFrame
degree_df = pd.DataFrame(
    list(degree_dict.items()), 
    columns=['user_hash', 'degree_centrality']
)

# ------------------------------
# 3. Fusion et Nettoyage
# ------------------------------
# Fusionner le degré avec le DataFrame train en utilisant 'user_hash'
train_with_degree = train.merge(degree_df, on='user_hash', how='left')

# Remplacer les NaN dans 'degree_centrality' par 0. 
# Ces NaN correspondent aux utilisateurs qui n'apparaissent pas dans le graphe social (nœuds isolés).
train_with_degree['degree_centrality'] = train_with_degree['degree_centrality'].fillna(0)

# Filtrer uniquement les candidats qui ont un label (is_cheating n'est pas NaN)
labeled_with_degree = train_with_degree[~train_with_degree['is_cheating'].isna()]

print("=== Aperçu des données étiquetées avec le degré ===")
print(labeled_with_degree[['user_hash', 'is_cheating', 'degree_centrality']].head())
print(f"\nNombre de lignes analysées (étiquetées) : {labeled_with_degree.shape[0]}")

# ------------------------------
# 4. Analyse de Corrélation
# ------------------------------

# Calculer la Corrélation de Pearson
correlation = labeled_with_degree['is_cheating'].corr(labeled_with_degree['degree_centrality'])
print(f"\nCorrélation de Pearson entre is_cheating et degree_centrality : {correlation:.4f}")

# ------------------------------
# 5. Visualisation (Box Plot)
# ------------------------------

plt.figure(figsize=(8, 6))

sns.boxplot(
    data=labeled_with_degree, 
    x='is_cheating', 
    y='degree_centrality',
    palette=['skyblue', 'salmon']
)

plt.title("Distribution du Degré de Centralité par Statut de Triche")
plt.xlabel("Est un Tricheurs (0.0 = Non, 1.0 = Oui)")
plt.ylabel("Nombre de Connexions (Degré) - Échelle Logarithmique")
# Utilisation de l'échelle log pour mieux gérer la forte variance
plt.yscale('log') 
plt.grid(axis='y', linestyle='--')
plt.show()

