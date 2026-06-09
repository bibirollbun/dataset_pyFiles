!pip install rdkit


import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import tensorflow as tf


train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')


train_df.head()


target = ['Tg','Tc','Density','Rg','FFV']
train_df[target].count()


train_df.isnull().sum()


tc = pd.read_csv('/kaggle/input/tc-smiles/Tc_SMILES.csv')
tg = pd.read_csv('/kaggle/input/tg-smiles-pid-polymer-class/TgSS_enriched_cleaned.csv')


print(tc.head(),'\n',tg.head())


tg.head()


tc.rename(columns={'TC_mean' : 'Tc'}, inplace=True)


tc.head()


tc_tg = pd.merge(tc,tg, on = 'SMILES', how = 'outer').fillna(0)


tc_tg.head()


tc_tg.drop(['PID','Polymer Class'], axis=1, inplace=True)


print(tc_tg.count(),'\n')
print(tc_tg.isnull().sum())


df_merged = train_df.merge(tc_tg, on='SMILES', how='left', suffixes=('', '_new'))


for col in ['Tc', 'Tg']:
    df_merged[col] = df_merged[f'{col}_new'].combine_first(df_merged[col])


train_df = df_merged.drop(columns=[f'{col}_new' for col in ['Tc', 'Tg']])


train_df.fillna(0, inplace=True)


def plot_molecule_examples(train_df, n=10):
    mols = []
    legends = []
    
    for i, row in train_df.head(n).iterrows():
        m = Chem.MolFromSmiles(row['SMILES'])
        if m:
            mols.append(m)
            legends.append(f"ID: {row['id']}")
    
    img = Draw.MolsToGridImage(mols, legends=legends, molsPerRow=3, subImgSize=(300, 200))
    display(img)
plot_molecule_examples(train_df)


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()

    num_atoms = len(atoms)
    features = []
    for atom in atoms:
        features.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetHybridization().real,
        ])
    features = np.array(features, dtype=np.float32)

    adj = np.zeros((num_atoms, num_atoms), dtype=np.float32)
    for bond in bonds:
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    return features, adj



MAX_NODES = 64
smiles_list = train_df['SMILES'].values
labels = train_df[['Tg', 'Rg', 'Tc', 'Density', 'FFV']].values

def pad_graph(features, adj):
    num_nodes = features.shape[0]
    feature_dim = features.shape[1]
    
    padded_features = np.zeros((MAX_NODES, feature_dim), dtype=np.float32)
    padded_adj = np.zeros((MAX_NODES, MAX_NODES), dtype=np.float32)

    padded_features[:num_nodes, :] = features
    padded_adj[:num_nodes, :num_nodes] = adj

    return padded_features, padded_adj

graph_features = []
graph_adjs = []
valid_labels = []

for i, smi in enumerate(smiles_list):
    f, a = smiles_to_graph(smi)
    if f is not None and f.shape[0] <= MAX_NODES:
        pf, pa = pad_graph(f, a)
        graph_features.append(pf)
        graph_adjs.append(pa)
        valid_labels.append(labels[i])

graph_features = np.array(graph_features)
graph_adjs = np.array(graph_adjs)
valid_labels = np.array(valid_labels)



class MPNN(tf.keras.Model):
    def __init__(self, hidden_dim=64, message_passing_steps=2):
        super(MPNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.message_passing_steps = message_passing_steps

        self.node_fc = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.message_fc = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.update_fc = tf.keras.layers.GRUCell(hidden_dim)
        self.output_fc = tf.keras.layers.Dense(5)  # Tg, Rg, Tc, Density, FFV

    def call(self, inputs):
        x, adj = inputs  # x: (B, N, F), adj: (B, N, N)

        h = self.node_fc(x)  # (B, N, H)

        for _ in range(self.message_passing_steps):
            m = tf.matmul(adj, h)  # (B, N, H)
            m = self.message_fc(m)
            h, _ = self.update_fc(tf.reshape(m, [-1, self.hidden_dim]), [tf.reshape(h, [-1, self.hidden_dim])])
            h = tf.reshape(h, tf.shape(m))  # reshape back

        g = tf.reduce_mean(h, axis=1)  # global average pooling
        out = self.output_fc(g)
        return out



from sklearn.model_selection import train_test_split

X_feat_train, X_feat_val, X_adj_train, X_adj_val, y_train, y_val = train_test_split(
    graph_features, graph_adjs, valid_labels, test_size=0.1, random_state=42
)

model = MPNN()
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='mae',  # match wMAE approximation
    metrics=[tf.keras.metrics.MeanAbsoluteError()]
)





model.fit(
    x=(X_feat_train, X_adj_train),
    y=y_train,
    validation_data=((X_feat_val, X_adj_val), y_val),
    batch_size=32,
    epochs=50
)


# Load your test SMILES

test_ids = test_df["id"].values

test_features = []
test_adjs = []

for smi in test_df["SMILES"]:
    f, a = smiles_to_graph(smi)
    pf, pa = pad_graph(f, a)
    test_features.append(pf)
    test_adjs.append(pa)

test_features = np.array(test_features)
test_adjs = np.array(test_adjs)

predictions = model.predict((test_features, test_adjs))

submission = pd.DataFrame({
    "id": test_ids,
    "Tg": predictions[:, 0],
    "FFV": predictions[:, 4],
    "Tc": predictions[:, 2],
    "Density": predictions[:, 3],
    "Rg": predictions[:, 1],
})

submission.to_csv("./submission.csv", index=False)



s = pd.read_csv("submission.csv")


s.head()




