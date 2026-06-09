# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

folder_path = '/kaggle/input/stanford-rna-3d-folding/MSA/'

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_sequences = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")
validation_sequences = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv")
test_sequences = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")

train_labels = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")
validation_labels = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/validation_labels.csv")

sample_submission = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/sample_submission.csv")


# train_sequences
print(f"train_sequences :", train_sequences.head())

# train_labels
print("train_labels :", train_labels.head())

# validation_sequences
print("validation_sequences :", validation_sequences.head())

# validation_labels
print("validation_labels :", validation_labels.head())

#test_sequences
print("test_sequences :", test_sequences.head())



print(train_sequences.shape, train_labels.shape)
print(validation_sequences.shape, validation_labels.shape)
print(test_sequences.shape)



print(train_sequences.columns)
print(train_labels.columns)

print(validation_sequences.columns)
print(validation_labels.columns)

print(test_sequences.columns)


print(train_sequences['target_id'].head())
print(train_labels['ID'].head())

print(validation_sequences['target_id'].head())
print(validation_labels['ID'].head())

print(test_sequences['target_id'].head())


train_sequences['target_id'] = train_sequences['target_id'].str.strip().str.upper()
train_labels['ID'] = train_labels['ID'].str.strip().str.upper()

validation_sequences['target_id'] = validation_sequences['target_id'].str.strip().str.upper()
validation_labels['ID'] = validation_labels['ID'].str.strip().str.upper()

test_sequences['target_id'] = test_sequences['target_id'].str.strip().str.upper()


train_labels['ID'] = train_labels['ID'].str.extract(r'([A-Za-z0-9_]+)_')[0]

validation_labels['ID'] = validation_labels['ID'].str.extract(r'([A-Za-z0-9_]+)_')[0]



print(set(train_labels['ID']).issubset(set(train_sequences['target_id'])))

print(set(validation_labels['ID']).issubset(set(validation_sequences['target_id'])))



train_data = pd.merge(train_sequences, train_labels, left_on='target_id', right_on='ID', how='inner')

validation_data = pd.merge(validation_sequences, validation_labels, left_on='target_id', right_on='ID', how='inner')


print(f" La fusion entre train_sequences et train_labels :{train_data.head()}")

print(f" La fusion entre validation_sequences et validation_labels :{validation_data.head()}")


print(f"Doublons dans train_sequences: {train_data.duplicated().sum()}")

print(f"Doublons dans validation_data: {validation_data.duplicated().sum()}")

print(f"Doublons dans test_sequences: {test_sequences.duplicated().sum()}")


# train_sequences
print("train_data :")
print(train_data.info())

# validation_sequences
print("validation_data :")
print(validation_data.info())

# test_sequences
print("test_sequences :")
print(test_sequences.info())


# train_sequences
print("Nombre total de valeurs manqauntes train_data :")
print(train_data.isnull().sum().sum())
print(train_data.isnull().sum())


# validation_sequences
print("Nombre total de valeurs manqauntes validation_data :")
print(validation_data.isnull().sum().sum())
print(validation_data.isnull().sum())

# test_sequences
print("Nombre total de valeurs manqauntes test_sequences :")
print(test_sequences.isnull().sum().sum())
print(test_sequences.isnull().sum())



# train_data
print("train_data :")
print(train_data.describe())

# validation_data
print("validation_data :")
print(validation_data.describe())

# test_sequences
print("test_sequences :")
print(test_sequences.describe())



print(f"Train sequences shape: {train_data.shape}")

print(f"Validation sequences shape: {validation_data.shape}")

print(f"Test sequences shape: {test_sequences.shape}")


print(train_data.columns)
print(validation_data.columns)


# train_labels
train_data['x_1'] == train_data['x_1'].fillna(train_data['x_1'].median(), inplace=True)
print("Nombre des valeurs manquantes de cahque colonne :",train_data.isnull().sum())

# train_labels
train_data['y_1'] == train_data['y_1'].fillna(train_data['y_1'].median(), inplace=True)
print("Nombre des valeurs manquantes de cahque colonne :",train_data.isnull().sum())

# train_labels
train_data['z_1'] == train_data['z_1'].fillna(train_data['z_1'].median(), inplace=True)
print("Nombre des valeurs manquantes de cahque colonne :",train_data.isnull().sum())


# Train_data
mode_sequences = train_data['all_sequences'].mode()[0]
train_data['all_sequences'] = train_data['all_sequences'].fillna(mode_sequences)
print("Nombre des valeurs manquantes de chaque colonne :",train_data.isnull().sum())
print("Nombre total de valeurs manqauntes :", train_data.isnull().sum().sum())


print(train_data.columns)
print(validation_data.columns)


print(train_data.dtypes)
print(validation_data.dtypes)


train_data


sequences = train_data['sequence']

nucleotides = ['A', 'C', 'G', 'U']
nucleotide_counts = {n: sequences.str.count(n).sum() for n in nucleotides}

sns.barplot(x=list(nucleotide_counts.keys()), y=list(nucleotide_counts.values()))
plt.title('Distribution des nucléotides dans les séquences d\'ARN')
plt.xlabel('Nucléotide')
plt.ylabel('Fréquence')
plt.show()



residue_counts = train_data['resname'].value_counts()

sns.barplot(x=residue_counts.index, y=residue_counts.values)
plt.title('Distribution des résidus dans les labels d\'entraînement')
plt.xlabel('Résidu')
plt.ylabel('Fréquence')
plt.show()



subset_labels = train_data.iloc[:10]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(subset_labels['x_1'], subset_labels['y_1'], subset_labels['z_1'], c='r', marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Coordonnées 3D des atomes')

plt.show()



sequence_lengths = sequences.str.len()

plt.hist(sequence_lengths, bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution des longueurs des séquences d\'ARN')
plt.xlabel('Longueur de la séquence')
plt.ylabel('Fréquence')
plt.show()




validation_data_types = validation_data['sequence']

validation_nucleotide_counts = {n: validation_data_types.str.count(n).sum() for n in nucleotides}

sns.barplot(x=list(validation_nucleotide_counts.keys()), y=list(validation_nucleotide_counts.values()))
plt.title('Distribution des nucléotides dans les séquences de validation')
plt.xlabel('Nucléotide')
plt.ylabel('Fréquence')
plt.show()




subset_validation_labels = validation_data.iloc[:10]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(subset_validation_labels['x_1'], subset_validation_labels['y_1'], subset_validation_labels['z_1'], c='b', marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Coordonnées 3D des atomes dans les labels de validation')

plt.show()



def one_hot_encode(sequence):
    encoding = {'A': [1, 0, 0, 0],
                'C': [0, 1, 0, 0],
                'G': [0, 0, 1, 0],
                'U': [0, 0, 0, 1]}
    
    sequence = sequence.replace('-', '') 

    sequence = [nt for nt in sequence if nt in encoding]
    
    return [encoding[nt] for nt in sequence]

encoded_sequences = [one_hot_encode(seq) for seq in train_data['sequence']]

max_sequence_length = max([len(seq) for seq in encoded_sequences])

padded_sequences = pad_sequences(encoded_sequences, maxlen=max_sequence_length, padding='post', dtype='float32')

train_data['encoded_sequence'] = list(padded_sequences)

print(train_data[['sequence', 'encoded_sequence']].head())


def one_hot_encode(sequence):
    encoding = {'A': [1, 0, 0, 0],
                'C': [0, 1, 0, 0],
                'G': [0, 0, 1, 0],
                'U': [0, 0, 0, 1]}
    
    sequence = sequence.replace('-', '') 

    sequence = [nt for nt in sequence if nt in encoding]
    
    return [encoding[nt] for nt in sequence]

encoded_sequences = [one_hot_encode(seq) for seq in validation_data['sequence']]

max_sequence_length = max([len(seq) for seq in encoded_sequences])

padded_sequences = pad_sequences(encoded_sequences, maxlen=max_sequence_length, padding='post', dtype='float32')

validation_data['encoded_sequence'] = list(padded_sequences)

print(validation_data[['sequence', 'encoded_sequence']].head())


scaler = StandardScaler()
train_data[['x_1', 'y_1', 'z_1']] = scaler.fit_transform(train_data[['x_1', 'y_1', 'z_1']])

print(train_data[['x_1', 'y_1', 'z_1']].head())



scaler = StandardScaler()

cols_to_scale = [f'x_{i}' for i in range(1, 41)] + [f'y_{i}' for i in range(1, 41)] + [f'z_{i}' for i in range(1, 41)]

validation_data[cols_to_scale] = scaler.fit_transform(validation_data[cols_to_scale])

print(validation_data[cols_to_scale].head())




train_data['temporal_cutoff'] = pd.to_datetime(train_data['temporal_cutoff'])

train_data['temporal_cutoff_timestamp'] = train_data['temporal_cutoff'].astype(int) / 10**9



validation_data['temporal_cutoff'] = pd.to_datetime(validation_data['temporal_cutoff'])

validation_data['temporal_cutoff_timestamp'] = validation_data['temporal_cutoff'].astype(int) / 10**9



label_encoder = LabelEncoder()

train_data['resname_encoded'] = label_encoder.fit_transform(train_data['resname'])

print(train_data[['resname', 'resname_encoded']].head())


train_data.drop(['description', 'all_sequences', 'temporal_cutoff', 'resname', 'ID', 'sequence'], axis=1, inplace=True)


train_data.rename(columns={'encoded_sequence': 'sequence', 'temporal_cutoff_timestamp': 'temporal_cutoff', 'resname_encoded': 'resname'}, inplace=True)


label_encoder = LabelEncoder()

validation_data['resname_encoded'] = label_encoder.fit_transform(validation_data['resname'])

print(validation_data[['resname', 'resname_encoded']].head())



validation_data.drop(['sequence', 'description', 'all_sequences', 'temporal_cutoff', 'resname', 'ID'], axis=1, inplace=True)


validation_data.rename(columns={'encoded_sequence': 'sequence', 'temporal_cutoff_timestamp': 'temporal_cutoff', 'resname_encoded': 'resname'}, inplace=True)


def one_hot_encode(sequence):
    encoding = {'A': [1, 0, 0, 0],
                'C': [0, 1, 0, 0],
                'G': [0, 0, 1, 0],
                'U': [0, 0, 0, 1]}
    
    sequence = sequence.replace('-', '') 

    sequence = [nt for nt in sequence if nt in encoding]
    
    return [encoding[nt] for nt in sequence]

encoded_sequences = [one_hot_encode(seq) for seq in test_sequences['sequence']]

max_sequence_length = max([len(seq) for seq in encoded_sequences])

padded_sequences = pad_sequences(encoded_sequences, maxlen=max_sequence_length, padding='post', dtype='float32')

test_sequences['encoded_sequence'] = list(padded_sequences)

print(test_sequences[['sequence', 'encoded_sequence']].head())



test_sequences['temporal_cutoff'] = pd.to_datetime(test_sequences['temporal_cutoff'])

test_sequences['temporal_cutoff_timestamp'] = test_sequences['temporal_cutoff'].astype(int) / 10**9


test_sequences.drop(['sequence', 'description', 'all_sequences', 'temporal_cutoff'], axis=1, inplace=True)    


test_sequences.rename(columns={'encoded_sequence': 'sequence', 'temporal_cutoff_timestamp': 'temporal_cutoff'}, inplace=True)


test_sequences


train_data


validation_data


# from sklearn.tree import DecisionTreeRegressor

# model = DecisionTreeRegressor(random_state=42)
# model.fit(X_train, y_train)

# y_pred = model.predict(X_val)
# mse = mean_squared_error(y_val, y_pred)
# print(f"Erreur quadratique moyenne : {mse}")



# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error

# # Préparer les données
# X = train_data[['sequence']]  # Caractéristiques d'entrée
# y = train_data[['x_1', 'y_1', 'z_1']]  # Coordonnées cibles

# # Diviser en ensembles d'entraînement et de validation
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# # Modèle de régression linéaire
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Prédictions
# y_pred = model.predict(X_val)

# # Évaluation
# mse = mean_squared_error(y_val, y_pred)
# print(f"Erreur quadratique moyenne : {mse}")



# from sklearn.tree import DecisionTreeRegressor

# model = DecisionTreeRegressor(random_state=42)
# model.fit(X_train, y_train)

# y_pred = model.predict(X_val)
# mse = mean_squared_error(y_val, y_pred)
# print(f"Erreur quadratique moyenne : {mse}")



from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = train_data[['sequence']]
y = train_data[['x_1', 'y_1', 'z_1']]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

mse = mean_squared_error(y_val, y_pred)
print(f"Erreur quadratique moyenne : {mse}")



# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM, Dropout
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# import numpy as np

# X = np.array(train_data['sequence'].tolist())
# y = np.array(train_data[['x_1', 'y_1', 'z_1']].values)

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# # Construire un modèle simple de réseau de neurones
# model = Sequential()
# model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(3))  # Trois sorties : x, y et z

# # Compiler le modèle
# model.compile(optimizer='adam', loss='mean_squared_error')

# # Entraîner le modèle
# model.fit(X_train, y_train, epochs=10, batch_size=32)

# y_pred = model.predict(X_val)

# mse = mean_squared_error(y_val, y_pred)
# print(f"Erreur quadratique moyenne : {mse}")



# Préparer les données de test
X_test = np.array(test_sequences['encoded_sequence'].tolist())  # Assurez-vous que c'est un tableau numpy

# Prédire les coordonnées pour l'ensemble de test
y_test_pred = model.predict(X_test)

# Construire le fichier de soumission
submission_data = []

for i, test_seq in test_sequences.iterrows():
    sequence_id = test_seq['target_id']
    resnames = test_seq['sequence']
    
    for j, resname in enumerate(resnames):
        row = {'ID': sequence_id, 'resname': resname, 'resid': j + 1}
        
        # Ajouter les prédictions pour les 5 structures (x, y, z)
        for k in range(5):
            row[f'x_{k + 1}'] = y_test_pred[i][j][0]
            row[f'y_{k + 1}'] = y_test_pred[i][j][1]
            row[f'z_{k + 1}'] = y_test_pred[i][j][2]
        
        submission_data.append(row)

# Créer le DataFrame de soumission
submission_df = pd.DataFrame(submission_data)

# Sauvegarder en fichier CSV
submission_df.to_csv('submission.csv', index=False)




# train_data['sequence_id'] = train_data['all_sequences'].str.split('|').str[0]
# train_data['chain'] = train_data['all_sequences'].str.split('|').str[1]
# train_data['structure'] = train_data['all_sequences'].str.split('|').str[2]
# train_data['species'] = train_data['all_sequences'].str.split('|').str[3]

# train_data[['sequence_id', 'chain', 'structure', 'species']].head()


# train_data


# encoder = LabelEncoder()

# train_data['chain_encoded'] = encoder.fit_transform(train_data['chain'])
# train_data['structure_encoded'] = encoder.fit_transform(train_data['structure'])
# train_data['species_encoded'] = encoder.fit_transform(train_data['species'])

# train_data[['chain', 'chain_encoded', 'structure', 'structure_encoded', 'species', 'species_encoded']].head()



# validation_data['sequence_id'] = validation_data['all_sequences'].str.split('|').str[0]
# validation_data['chain'] = validation_data['all_sequences'].str.split('|').str[1]
# validation_data['structure'] = validation_data['all_sequences'].str.split('|').str[2]
# validation_data['species'] = validation_data['all_sequences'].str.split('|').str[3]

# validation_data[['sequence_id', 'chain', 'structure', 'species']].head()

# validation_data


# encoder = LabelEncoder()

# validation_data['chain_encoded'] = encoder.fit_transform(validation_data['chain'])
# validation_data['structure_encoded'] = encoder.fit_transform(validation_data['structure'])
# validation_data['species_encoded'] = encoder.fit_transform(validation_data['species'])

# validation_data[['chain', 'chain_encoded', 'structure', 'structure_encoded', 'species', 'species_encoded']].head()

# validation_data


# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.optimizers import Adam

# # Définir le modèle LSTM
# model = Sequential()
# model.add(LSTM(128, input_shape=(train_encoded.shape[1], train_encoded.shape[2]), return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(64))
# model.add(Dense(3))

# # Compilation du modèle
# model.compile(optimizer=Adam(), loss='mean_squared_error')

# # Entraînement du modèle
# model.fit(train_encoded, train_labels, epochs=10, batch_size=32, validation_data=(validation_encoded, validation_labels))



# import tensorflow as tf
# from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout

# def transformer_encoder(input_tensor, head_size, num_heads, ff_dim, dropout=0.1):
#     # Attention layer
#     attention = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(input_tensor, input_tensor)
#     attention = Dropout(dropout)(attention)
#     attention = LayerNormalization(epsilon=1e-6)(attention + input_tensor)
    
#     # Feed forward layer
#     ff = Dense(ff_dim, activation="relu")(attention)
#     ff = Dense(input_tensor.shape[-1])(ff)
#     ff = Dropout(dropout)(ff)
#     return LayerNormalization(epsilon=1e-6)(ff + attention)

# input_seq = Input(shape=(train_encoded.shape[1], train_encoded.shape[2]))  # Shape des séquences encodées
# x = transformer_encoder(input_seq, head_size=64, num_heads=4, ff_dim=128)
# x = Dense(3)(x)  # Sortie des coordonnées 3D
# model = tf.keras.Model(inputs=input_seq, outputs=x)

# model.compile(optimizer=Adam(), loss='mean_squared_error')

# # Entraînement du modèle
# model.fit(train_encoded, train_labels, epochs=10, batch_size=32, validation_data=(validation_encoded, validation_labels))



# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# # Modèle simple avec des couches denses
# model = Sequential()
# model.add(Dense(128, input_dim=train_encoded.shape[1], activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(3))  # Prédiction des 3 coordonnées (x, y, z)

# model.compile(optimizer=Adam(), loss='mean_squared_error')

# # Entraînement du modèle
# model.fit(train_encoded, train_labels, epochs=10, batch_size=32, validation_data=(validation_encoded, validation_labels))


