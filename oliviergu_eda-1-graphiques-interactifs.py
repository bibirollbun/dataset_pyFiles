# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, DBSCAN
from difflib import SequenceMatcher

# ---- ğŸ“Œ VÃ©rification des fichiers disponibles ----
print("\n[INFO] ğŸ“‚ Fichiers disponibles dans le dataset Kaggle :")
print(os.listdir("/kaggle/input/stanford-rna-3d-folding"))

# ---- ğŸ“Œ Fonction pour tester plusieurs encodages ----
def read_csv_flexible(file_path):
    encodings = ['utf-8', 'latin1', 'ISO-8859-1']
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            print(f"[INFO] âœ… Chargement rÃ©ussi avec l'encodage : {enc}")
            return df
        except UnicodeDecodeError:
            print(f"[WARNING] âš ï¸� ProblÃ¨me avec l'encodage : {enc}")
    raise ValueError("[ERROR] â�Œ Aucun encodage valide trouvÃ© pour le fichier !")

# ---- ğŸ“Œ Chargement des fichiers ----
train_labels_path = "/kaggle/input/stanford-rna-3d-folding/train_labels.csv"
train_sequences_path = "/kaggle/input/stanford-rna-3d-folding/train_sequences.csv"

df_labels = read_csv_flexible(train_labels_path)
df_sequences = read_csv_flexible(train_sequences_path)

# ---- ğŸ“Œ Nettoyage des donnÃ©es ----
print("\n[INFO] ğŸ§¹ Nettoyage des donnÃ©es...")
df_clean = df_labels.dropna().copy()
df_clean = df_clean[df_clean['resname'].isin(['A', 'C', 'G', 'U'])]

# ---- ğŸ“Œ Exploration des donnÃ©es ----
print("\n[INFO] ğŸ”� AperÃ§u des donnÃ©es :")
print(df_clean.info())
print(df_clean.head())

# ---- ğŸ“Š Visualisation interactive des nuclÃ©otides ----
fig_nucleotides = px.histogram(df_clean, x="resname", title="Distribution des nuclÃ©otides")
fig_nucleotides.show()

# ---- ğŸ“Š Visualisation interactive des coordonnÃ©es X, Y, Z ----
fig_coords = px.scatter_3d(df_clean, x="x_1", y="y_1", z="z_1", color="resname",
                           title="Visualisation 3D des rÃ©sidus ARN")
fig_coords.show()

# ---- ğŸ“Œ Clustering K-Means ----
print("\n[INFO] ğŸ“Œ Application du clustering K-Means...")
df_cluster = df_clean.sample(n=10000, random_state=42)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_cluster['kmeans_cluster'] = kmeans.fit_predict(df_cluster[['x_1', 'y_1', 'z_1']])

fig_kmeans = px.scatter_3d(df_cluster, x="x_1", y="y_1", z="z_1", color="kmeans_cluster",
                           title="Clustering K-Means des rÃ©sidus ARN")
fig_kmeans.show()

# ---- ğŸ“Œ Clustering DBSCAN ----
print("\n[INFO] ğŸ“Œ Application du clustering DBSCAN...")
dbscan = DBSCAN(eps=20, min_samples=10)
df_cluster['dbscan_cluster'] = dbscan.fit_predict(df_cluster[['x_1', 'y_1', 'z_1']])

fig_dbscan = px.scatter_3d(df_cluster, x="x_1", y="y_1", z="z_1", color="dbscan_cluster",
                           title="Clustering DBSCAN des RÃ©sidus ARN")
fig_dbscan.show()

# ---- ğŸ“Œ Comparaison de plusieurs chaÃ®nes ARN ----
print("\n[INFO] ğŸ“Œ SÃ©lection de 3 chaÃ®nes ARN au hasard...")
random_chains = df_clean['ID'].str.split('_').str[0].drop_duplicates().sample(3, random_state=42).values
df_selected_chains = df_clean[df_clean['ID'].str.startswith(tuple(random_chains))]

fig_chains = px.scatter_3d(df_selected_chains, x="x_1", y="y_1", z="z_1", color="ID",
                           title="Comparaison 3D de plusieurs chaÃ®nes ARN")
fig_chains.show()

# ---- ğŸ“Œ SÃ©lection et analyse dâ€™une sÃ©quence ARN inconnue ----
print("\n[INFO] ğŸ“Œ SÃ©lection d'une sÃ©quence ARN non annotÃ©e...")
existing_ids = df_clean['ID'].str.split('_').str[0].unique()
df_unknown_seq = df_sequences[~df_sequences['target_id'].isin(existing_ids)]
random_unknown_seq = df_unknown_seq.sample(1, random_state=42)

# ---- ğŸ“Œ Recherche des sÃ©quences similaires ----
print("\n[INFO] ğŸ”� Recherche des sÃ©quences similaires...")
def sequence_similarity(seq1, seq2):
    return SequenceMatcher(None, seq1, seq2).ratio()

df_sequences['similarity'] = df_sequences['sequence'].apply(
    lambda x: sequence_similarity(x, random_unknown_seq['sequence'].values[0]) if isinstance(x, str) else 0)

top_similar_sequences = df_sequences.sort_values(by='similarity', ascending=False).head(5)
similar_ids = top_similar_sequences['target_id'].values[1:4]
df_similar_structures = df_clean[df_clean['ID'].str.startswith(tuple(similar_ids))]

# ---- ğŸ“Œ Estimation de la structure 3D de la sÃ©quence inconnue ----
df_estimated_structure = df_similar_structures.groupby('resid')[['x_1', 'y_1', 'z_1']].mean().reset_index()

fig_estimated = px.scatter_3d(df_estimated_structure, x="x_1", y="y_1", z="z_1", color="resid",
                              title="Structure 3D estimÃ©e pour une sÃ©quence ARN inconnue")
fig_estimated.show()

# ---- ğŸ“Œ GÃ©nÃ©ration dâ€™un fichier PDB ----
pdb_filename = "/kaggle/working/4YVJ_C_estimated.pdb"
print(f"\n[INFO] ğŸ“„ GÃ©nÃ©ration du fichier PDB : {pdb_filename}")
with open(pdb_filename, 'w', encoding='utf-8') as f:
    f.write("HEADER    ESTIMATED RNA STRUCTURE 4YVJ_C\n")
    for index, row in df_estimated_structure.iterrows():
        f.write(f"ATOM  {index+1:5d}  C1'  A {int(row['resid']):4d}    {row['x_1']:8.3f} {row['y_1']:8.3f} {row['z_1']:8.3f}\n")
    f.write("END\n")

print(f"[INFO] âœ… Fichier PDB gÃ©nÃ©rÃ© avec succÃ¨s : {pdb_filename}")

