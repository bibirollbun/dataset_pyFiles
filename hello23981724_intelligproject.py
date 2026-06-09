!pip install pyvis
!pip install obonet


!pip install biopython
!pip install progressbar


!pip install tensorflow


import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import progressbar


train_terms=pd.read_csv('/kaggle/input/cafa-5-protein-function-prediction/Train/train_terms.tsv', sep="\t")
train_terms.head()
train_terms


train_terms.shape


import os
import json
from typing import Dict
from collections import Counter

import random
import obonet
import pandas as pd
import numpy as np
from Bio import SeqIO
import re
from transformers import T5Tokenizer, T5EncoderModel
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False) #.to(device)

# Load the model
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)

def get_embeddings(seq):
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", seq)))]

    ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest")

    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # generate embeddings
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids,
                               attention_mask=attention_mask)

    # extract residue embeddings for the first ([0,:]) sequence in the batch and remove padded & special tokens ([0,:7]) 
    emb_0 = embedding_repr.last_hidden_state[0]
    emb_0_per_protein = emb_0.mean(dim=0)
    
    return emb_0_per_protein

file = '/kaggle/input/cafa-5-protein-function-prediction/Train/train_sequences.fasta'
sequences = SeqIO.parse(file, "fasta")
print(sequences)


print(next(iter(sequences)).seq)


sequences = SeqIO.parse(file, "fasta")
sequences



get_embeddings(str(next(iter(sequences)).seq))


train_protein_ids = np.load('/kaggle/input/t5embeds/train_ids.npy')
train_protein_ids_pd=pd.DataFrame(train_protein_ids)
train_protein_ids_pd


train_protein_ids_pd.size


train_embeddings = np.load('/kaggle/input/t5embeds/train_embeds.npy')
column_num = train_embeddings.shape[1]
train_df  = pd.DataFrame(train_embeddings, columns = ["Colomn " + str(i) for i in range(1, column_num+1)])
train_df


#There are more than 50,000 labels in the train_terms.tsv file, so we will choose the most frequent 1500 GO term IDs as labels of the dataset.
#Extracting Go IDs
num_of_labels = 1500
labels = train_terms['term'].value_counts().index
labels=labels[:num_of_labels]
labels


# Extracting the dataset which contains the top 1500(labels) GO terms
train_terms_updated = train_terms.loc[train_terms['term'].isin(labels)]
train_terms_updated=train_terms_updated.reset_index(drop=True)
train_terms_updated


bar_df = train_terms_updated['aspect'].value_counts()

plt.figure(figsize=(8, 5))

sns.barplot(x=bar_df.index, y=bar_df.values, palette="bright")

plt.xlabel("Aspect")
plt.ylabel("Count")
plt.title("Distribution of Aspects")

plt.show()



train_size = train_protein_ids.shape[0]
train_labels = np.zeros((train_size ,num_of_labels))
series_train_protein_ids = pd.Series(train_protein_ids)

bar = progressbar.ProgressBar(maxval=num_of_labels, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()
for i in range(num_of_labels):
    n_train_terms = train_terms_updated[train_terms_updated['term'] ==  labels[i]]
    label_related_proteins = n_train_terms['EntryID'].unique()
    train_labels[:,i] =  series_train_protein_ids.isin(label_related_proteins).astype(float)
    bar.update(i+1)
bar.finish()


!mkdir /kaggle/working/labels
labels_df = pd.DataFrame(data = train_labels, columns = labels)
labels_df.to_csv('/kaggle/working/labels/kaggledata.csv')


labels_df.shape


train_df.shape


import numpy as np
from sklearn.model_selection import train_test_split

# Split the data into train and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(train_df, labels_df, test_size=0.2, random_state=42)

# Define model architecture
INPUT_SHAPE = [train_df.shape[1]]
num_of_labels = labels_df.shape[1]

BATCH_SIZE = 512

model = tf.keras.Sequential([
    tf.keras.layers.BatchNormalization(input_shape=INPUT_SHAPE),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units=num_of_labels, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['binary_accuracy', tf.keras.metrics.AUC()]
)

# Train the model with validation split
history = model.fit(
    train_data,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=20,
    validation_data=(val_data, val_labels)
)





# Get the final training and validation metrics
final_train_metrics = model.evaluate(train_data, train_labels, verbose=0)
final_val_metrics = model.evaluate(val_data, val_labels, verbose=0)

# Display in the desired format
print("\nFinal Training Metrics:")
print("Loss:", final_train_metrics[0])
print("Binary Accuracy:", final_train_metrics[2])
print("AUC:", final_train_metrics[1])

print("\nFinal Validation Metrics:")
print("Loss:", final_val_metrics[0])
print("Binary Accuracy:", final_val_metrics[2])
print("AUC:", final_val_metrics[1])



model.save("/kaggle/working/model1.h5")


model.save_weights("/kaggle/working/model.weights.h5")


model.load_weights("/kaggle/working/model.weights.h5")


custom_input_tensor = np.array([[1 for i in range(1024)]])
print(custom_input_tensor)
print(len(custom_input_tensor[0]))
# Get predictions for custom input tensor
predictions = model.predict(custom_input_tensor)

# 'predictions' will contain the model's output for the custom input tensor
print(predictions)
for i in predictions[0]:
    x=0 if i<0.5 else 1
    #print(x)
# print(len(predictions))



history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss']].plot(title="Cross-entropy")
history_df.loc[:, ['binary_accuracy']].plot(title="Accuracy")


import tqdm
from Bio import SeqIO
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import json
from typing import Dict
from collections import Counter
import random
import obonet
from transformers import T5Tokenizer, T5EncoderModel
import torch
import re

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False) #.to(device)

# Load the model
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)

def get_embeddings(seq):
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", seq)))]

    ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest")

    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # generate embeddings
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids,
                               attention_mask=attention_mask)

    # extract residue embeddings for the first ([0,:]) sequence in the batch and remove padded & special tokens ([0,:7])
    emb_0 = embedding_repr.last_hidden_state[0]
    emb_0_per_protein = emb_0.mean(dim=0)

    return emb_0_per_protein

def predict(fasta_file):
    sequences = SeqIO.parse(fasta_file, "fasta")

    ids = []
    num_sequences=sum(1 for seq in sequences)
    embeds = np.zeros((num_sequences, 1024))
    i = 0
    with open(fasta_file, "r") as fastafile:
      # Iterate over each sequence in the file
      for sequence in SeqIO.parse(fastafile, "fasta"):
        seq_id = sequence.id
        seq_data = str(sequence.seq)
        embeds[i] = get_embeddings(seq_data).detach().cpu().numpy()
        ids.append(seq_id)
        i += 1
        
    INPUT_SHAPE=[1024]
    num_of_labels=1500

    model = tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(input_shape=INPUT_SHAPE),
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(units=num_of_labels, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['binary_accuracy', tf.keras.metrics.AUC()]
    )
    
    model.load_weights('/kaggle/working/model.weights.h5') #load model here
    labels_df=pd.read_csv('/kaggle/input/labels/kaggledata.csv')
    labels_df=labels_df.drop(columns='Unnamed: 0')

    predictions = model.predict(embeds)
    predictions_list1=[]
    predictions_list2=[]

    # 'predictions' will contain the model's output for the custom input tensor
    for prediction in predictions:
        tmp=[]
        t2=[]
        for i in prediction:
            x=0 if i<0.4 else 1
            tmp.append(x)
            t2.append(i)
        predictions_list1.append(tmp.copy())
        predictions_list2.append(t2.copy())

    label_columns = labels_df.columns

    # Convert the predictions into a DataFrame
    predictions_df = pd.DataFrame(predictions_list1, columns=label_columns)
    p21=pd.DataFrame(predictions_list2, columns=label_columns)

    # Save the DataFrame to a CSV file
    predictions_df.to_csv("predictions.csv", index=False)
    p21.to_csv("decimal.csv",index=False)
    print(predictions_df.shape)
    
    



predict('/kaggle/input/fastaexample/example.fasta') 

