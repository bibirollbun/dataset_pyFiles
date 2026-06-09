import os
import glob

import numpy as np
import pandas as pd
import tensorflow as tf

import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity



# TRANSFORM DATASET INTO CSV

def convert_folder_to_csv(base_path: str, label_df: pd.DataFrame = None, split: str = "train") -> pd.DataFrame:
    data = []

    # PATH TO THE TEXT FOLDER
    folders = sorted(glob.glob(os.path.join(base_path, split, "article_*")))

    # FOR EACH TEXT/ARTICLE FOLDER 
    for i, folder in enumerate(folders):
        file_1_path = os.path.join(folder, "file_1.txt")
        file_2_path = os.path.join(folder, "file_2.txt")

        # OPEN 2 TXT FILE IN CURRENT ARTICLE
        with open(file_1_path, "r", encoding="utf-8") as f1, open(file_2_path, "r", encoding="utf-8") as f2:
            text1 = f1.read().strip()
            text2 = f2.read().strip()

        # USING LABEL? (TRAIN CATEGORY)
        if label_df is not None:
            real_id = label_df.iloc[i]["real_text_id"]
            data.append({"id": i, "text1": text1, "text2": text2, "real_text_id": real_id})   # --> APPEND TO DICT WITH LABEL
        
        # IF NO LABEL (TEST CATEGORY)
        else:
            data.append({"id": i, "text1": text1, "text2": text2})    # --> APPEND TO DICT WITHOUT LABEL

    return pd.DataFrame(data)


label_df = pd.read_csv(r'/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv')

train = convert_folder_to_csv('/kaggle/input/fake-or-real-the-impostor-hunt/data', label_df, "train")
test = convert_folder_to_csv('/kaggle/input/fake-or-real-the-impostor-hunt/data', None, "test")

train_path = "/kaggle/working/train_data.csv"
test_path  = "/kaggle/working/test_data.csv"

train.to_csv(train_path, index=False)
test.to_csv(test_path, index=False)


print(f'Train csv saved at {train_path}')
print(f'Test csv saved at {test_path} ')


# DISPLAY DATA

train_df = pd.read_csv(r'/kaggle/working/train_data.csv')
test_df  = pd.read_csv(r'/kaggle/working/test_data.csv')

# NORMALIZE LABEL TO 0 AND 1
label = {
    1 : 0,
    2 : 1
}

train_df.loc[:, 'label'] = train_df['real_text_id'].map(label)

print(f'Train data shape : {train_df.shape}')
print(f'Test data shape  : {test_df.shape}')

train_df


# DATA AUGMENTATION

# SWAP DATA
df_swap = train_df.copy()
df_swap['text1'], df_swap['text2'] = df_swap['text2'], df_swap['text1']
df_swap['label'] = 1 - df_swap['label']

print(f'Before Data Augmentation : ')
display(train_df)

# CONCAT AUGMENTED DATA TO REAL DATA
train_df = pd.concat((train_df, df_swap), axis = 0).reset_index(drop = True)


print(f'\nAfter Data Augmentation : ')
display(train_df)


# SPLIT DATA

x = train_df[['text1', 'text2']]
y = train_df['label']

x.shape, y.shape, type(x), type(y)


# SIAMESE NETWORK

# DEFINE SIAMESE NETWORK
model = SentenceTransformer(model_name_or_path = 'paraphrase-mpnet-base-v2')

# ENCODE TEST DATA OUTSIDE FOLD 
test_text1_emb = model.encode(test_df['text1'], show_progress_bar = True)
test_text2_emb = model.encode(test_df['text2'], show_progress_bar = True)

# ANOTHER FEATURES ON TEST DATA
cos_sim_test = cosine_similarity(test_text1_emb, test_text2_emb).diagonal().reshape(-1, 1)
euclidean_test = np.linalg.norm(test_text1_emb - test_text2_emb, axis=1).reshape(-1, 1)

# CONCAT TEST DATA
x_test_concat = np.concatenate([test_text1_emb,
                                test_text2_emb,
                                test_text1_emb * test_text2_emb,
                                cos_sim_test,
                                euclidean_test], axis = 1)

# STORE LOSS, ACCURACY AND OOF ON EVERY DIFFERENT SEED
score_trains, score_vals, oof_tests = [], [], []
overall_accuracy , overall_val_loss = [], []

training_history = []  # --> STORE NEURAL NETWORK TRAINING HISTORY


# TRAIN CV EVERY DIFFERENT SEED
n_splits = 5
for seed in range(0, 50):  

    # DEFINE SKFOLD
    skfold  = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = seed)

    
    score_train, score_val, val_loss, oof_test = [], [], [], []

    print(f'\nSeed {seed}/19')
    
    # SKFOLD
    for i, (train_index, val_index) in enumerate(skfold.split(x, y)):

        #repeat_num = i // n_splits + 1
        #fold_num   = i % n_splits + 1
    
        # SPLIT DATA
        x_train, x_val = x.iloc[train_index, :], x.iloc[val_index, :]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        x_train = x_train.reset_index(drop=True)
        x_val   = x_val.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_val   = y_val.reset_index(drop=True)

        # ENCODE TRAIN DATA
        train_text1_emb = model.encode(x_train['text1'], show_progress_bar = False)
        train_text2_emb = model.encode(x_train['text2'], show_progress_bar = False)

        # ENCODE VAL DATA
        val_text1_emb = model.encode(x_val['text1'], show_progress_bar = False)
        val_text2_emb = model.encode(x_val['text2'], show_progress_bar = False)

        # EXTRACT FEATURES
        cos_sim_train = cosine_similarity(train_text1_emb, train_text2_emb).diagonal().reshape(-1, 1)
        cos_sim_val   = cosine_similarity(val_text1_emb, val_text2_emb).diagonal().reshape(-1, 1)
        euclidean_train = np.linalg.norm(train_text1_emb - train_text2_emb, axis=1).reshape(-1, 1)
        euclidean_val = np.linalg.norm(val_text1_emb - val_text2_emb, axis=1).reshape(-1, 1)
    
        # CONCAT DATA
        x_train_concat = np.concatenate([train_text1_emb,
                                         train_text2_emb,
                                         train_text1_emb * train_text2_emb,
                                         cos_sim_train,
                                         euclidean_train], axis = 1)
    
        x_val_concat = np.concatenate([val_text1_emb,
                                       val_text2_emb,
                                       val_text1_emb * val_text2_emb,
                                       cos_sim_val,
                                       euclidean_val], axis = 1)
    

        # BUILD NEURAL NETWORK ARCHITECTURE
        model_mlp = tf.keras.Sequential([tf.keras.Input(shape=(x_train_concat.shape[1],)),
                                         tf.keras.layers.Dense(512, activation= "relu", kernel_regularizer=tf.keras.regularizers.L2(0.15)),
                                         tf.keras.layers.Dropout(0.4),
                                         tf.keras.layers.Dense(256, activation= "relu", kernel_regularizer= tf.keras.regularizers.L2(0.1)),
                                         tf.keras.layers.Dropout(0.3),
                                         tf.keras.layers.Dense(64, activation = 'relu'),
                                         tf.keras.layers.Dense(1, activation="sigmoid")
                                        ])
        # COMPILE MODEL
        early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose = True) # EARLY STOPPING
        model_mlp.compile(optimizer='adam', loss="binary_crossentropy", metrics=["accuracy"])
    
        # TRAIN MODEL
        history = model_mlp.fit(x_train_concat, y_train, validation_data=(x_val_concat, y_val), epochs=100, batch_size=16, callbacks = [early_stop], verbose = 0)
        training_history.append(history)
    
        # PREDICTION 
        y_train_predict_proba = model_mlp.predict(x_train_concat, verbose = 0).ravel()
        y_val_predict_proba   = model_mlp.predict(x_val_concat, verbose = 0).ravel()

        y_train_predict = (y_train_predict_proba > 0.5).astype(int)
        y_val_predict   = (y_val_predict_proba > 0.5).astype(int)

        # ACCURACY 
        train_accuracy = accuracy_score(y_train, y_train_predict)
        val_accuracy   = accuracy_score(y_val, y_val_predict)

        score_train.append(train_accuracy) # --> STORE TRAIN ACCURACY
        score_val.append(val_accuracy)     # --> STORE VAL ACCURACY
        val_loss.append(min(history.history['val_loss'])) # --> STORE VAL LOSS

        # DISPLAY LOGS
        print(f'Fold {i+1} ğŸš€: 1ï¸�âƒ£ Train Accuracy = {train_accuracy}, 2ï¸�âƒ£ Val Accuracy = {val_accuracy}, ğŸ’” Best Val Loss = {min(history.history["val_loss"]):.4f}')
    
        # TEST PREDICTION
        y_test_predict_proba = model_mlp.predict(x_test_concat).ravel()
        oof_test.append(y_test_predict_proba)

        # CLEAR USED MODEL
        tf.keras.backend.clear_session()

    
    oof_tests.append(oof_test)

    overall_accuracy.append(np.mean(score_val))
    overall_val_loss.append(np.mean(val_loss))

    print(f'\nOverall OOF on seed {seed}ğŸ�‰: 1ï¸�âƒ£ Train Accuracy = {np.mean(score_train)}, 2ï¸�âƒ£ Val Accuracy = {np.mean(score_val)}, ğŸ’” Best Val Loss = {np.mean(val_loss)}')


# PREPARE SUBMISSION

global_mean = np.mean(oof_tests, axis = 0)
prediction = np.mean(global_mean, axis = 0)

y_test = (prediction > 0.5).astype(int)

inverse_encode = {
    0 : 1,
    1 : 2
}

submission = test_df[['id']].copy()
submission['real_text_id'] = y_test
submission['real_text_id'] = submission['real_text_id'].map(inverse_encode)

submission


submission.to_csv(r'submission-all-cv.csv', index = False)


submission['real_text_id'].value_counts()


# CHECK DISTRIBUTION

sns.countplot(x = submission['real_text_id'])

plt.title("real_text_id Distribution")
plt.xlabel("real_text_id")
plt.ylabel("Frequency")
plt.show()


plt.figure(figsize=(15, 5))

plt.subplot(1,2,1)
plt.plot(overall_val_loss)
plt.xlabel('Seed')
plt.ylabel('Val Loss')
plt.title('Val Loss')

plt.subplot(1,2,2)
plt.plot(overall_accuracy)
plt.xlabel('Seed')
plt.ylabel('Val Accuracy')
plt.title('Val Accuracy')

plt.suptitle('Cross Validation Result on Different Seed')
plt.tight_layout()
plt.show()


# PCA

pca = PCA(n_components = 2)
x_pca = pca.fit_transform(x_test_concat)

plt.figure(figsize=(12, 6))
sns.scatterplot(x = x_pca[:, 0], y = x_pca[:, 1], hue = submission['real_text_id'], palette = 'coolwarm')
plt.title('PCA Cluster Test data')

plt.show()


%%time
# TSNE

tsne = TSNE(n_components = 2, random_state = 2025)
x_tsne = tsne.fit_transform(x_test_concat)

plt.figure(figsize=(12, 6))
sns.scatterplot(x = x_tsne[:, 0], y = x_tsne[:, 1], hue = submission['real_text_id'], palette = 'coolwarm')
plt.title('TSNE Visualization Test data')

plt.show()

