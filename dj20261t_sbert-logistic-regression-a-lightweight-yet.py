from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm



def read_texts_from_dir(dir_path):
    """
    Reads 'file_1.txt' and 'file_2.txt' from each subfolder in the directory
    and returns a DataFrame with columns: id, file_1, file_2
    """
    data = []
    for folder_name in tqdm(sorted(os.listdir(dir_path)), desc=f"Reading from {dir_path}"):
        folder_path = os.path.join(dir_path, folder_name)
        file_1_path = os.path.join(folder_path, 'file_1.txt')
        file_2_path = os.path.join(folder_path, 'file_2.txt')

        try:
            with open(file_1_path, 'r', encoding='utf-8') as f1, \
                 open(file_2_path, 'r', encoding='utf-8') as f2:
                text1 = f1.read().strip()
                text2 = f2.read().strip()

            index = int(folder_name.split('_')[-1])
            data.append((index, text1, text2))

        except (FileNotFoundError, ValueError, OSError) as e:
            print(f"Error in folder '{folder_name}': {e}")

    return pd.DataFrame(data, columns=['id', 'file_1', 'file_2'])

train_path = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train"
test_path = "/kaggle/input/fake-or-real-the-impostor-hunt/data/test"

df_train = read_texts_from_dir(train_path)
df_test = read_texts_from_dir(test_path)

df_train = df_train.merge(pd.read_csv("/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv"), how='inner', on='id')
df_train.head()


# Split the data
train_df, val_df = train_test_split(df_train, test_size=0.1, stratify=df_train['real_text_id'], random_state=42)


model_name = "all-MiniLM-L6-v2"
sbert = SentenceTransformer(model_name)


def embed_pairwise(df):
    """
    Encodes file_1 and file_2 using SBERT.
    Generates 4-way embedding: [A, B, |A-B|, A*B]
    """
    emb_A = sbert.encode(df['file_1'].tolist(), convert_to_numpy=True)
    emb_B = sbert.encode(df['file_2'].tolist(), convert_to_numpy=True)
    
    abs_diff = np.abs(emb_A - emb_B)
    mult = emb_A * emb_B
    features = np.concatenate([emb_A, emb_B, abs_diff, mult], axis=1)
    return features

X_train = embed_pairwise(train_df)
y_train = train_df['real_text_id'].values

X_val = embed_pairwise(val_df)
y_val = val_df['real_text_id'].values

X_test = embed_pairwise(df_test)


clf = LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs')
clf.fit(X_train, y_train)

# Validation F1
val_preds = clf.predict(X_val)
val_f1 = f1_score(y_val, val_preds)
print(f"Validation F1-score: {val_f1:.4f}")


test_preds = clf.predict(X_test)
submission = pd.DataFrame(zip(list(range(df_test.shape[0])), test_preds), columns=['id', 'real_text_id'])
submission.to_csv('submission.csv', index=False)


os.environ['KAGGLE_USERNAME'] = "*************"
os.environ['KAGGLE_KEY'] = "***************"


!kaggle competitions submit -c fake-or-real-the-impostor-hunt -f submission.csv -m "Message"




