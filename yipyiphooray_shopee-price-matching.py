# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install torch
!pip install -U transformers datasets evaluate accelerate timm


from PIL import Image
from IPython.display import display
import torch


train_df = pd.read_csv("/kaggle/input/shopee-product-matching/train.csv")
test_df = pd.read_csv("/kaggle/input/shopee-product-matching/test.csv")


train_df.head()


img = Image.open("/kaggle/input/shopee-product-matching/train_images/001d7f5d9a2fac714f4d5f37b3baffb4.jpg")
display(img)


test_df.head()


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Collect all the titles in a list
documents = [t for t in train_df["title"]]
documents[:5]


# Instantiate vectoriser
vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")

# Fit the vectoriser
X_train = vectorizer.fit_transform(documents)


# Find the most frequent label groups
label_freq = train_df.groupby("label_group", as_index=False).agg(count=('label_group', 'count'))
label_freq.sort_values("count", ascending=False).head(3)


# Look at the title of any 2 same products 
print(train_df[train_df["label_group"] == 994676122]["title"].iloc[1])
print(train_df[train_df["label_group"] == 994676122]["title"].iloc[2])


X = vectorizer.transform(["100Pcs Karet Ikat Rambut Elastis untuk Wanita", "100 Pcs Ikat Rambut Karet Polos Elastis Gaya Korea untuk Wanita"])
similarity_matrix = cosine_similarity(X)
similarity_matrix


k = 10
similarity_matrix = cosine_similarity(X_train, X_train)
topk_indices = np.argsort(-similarity_matrix, axis=1)[:, 1:k+1]


for i in range(5):
    print(f"\nDocument {i}: {documents[i]}")
    similar_docs = [documents[j] for j in topk_indices[i].tolist()]
    print("Top similar:", similar_docs)


# I think this part is optional
from huggingface_hub import notebook_login
notebook_login()


from transformers import ResNetConfig, ResNetModel, AutoImageProcessor
model = ResNetModel.from_pretrained("microsoft/resnet-50")
image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")


# Look at the image of any 2 same products 
print(train_df[train_df["label_group"] == 994676122]["image"].iloc[1])
print(train_df[train_df["label_group"] == 994676122]["image"].iloc[2])


img1_name = "03f94cf522101e71933eb4047c5091fe.jpg"
img2_name = "0d21864616bb5667f86dfade0e7dab89.jpg"
img1 = Image.open(f"/kaggle/input/shopee-product-matching/train_images/{img1_name}")
img2 = Image.open(f"/kaggle/input/shopee-product-matching/train_images/{img2_name}")
inputs = image_processor([img1, img2], return_tensors="pt")


display(img1)
display(img2)


model.to("cpu")
with torch.no_grad():
    outputs = model(**inputs)
    temp_embeddings = outputs.pooler_output  


print(temp_embeddings.shape)
temp_embeddings = temp_embeddings.squeeze()
print(temp_embeddings.shape)


from torch.nn.functional import cosine_similarity
embedding1 = temp_embeddings[0]
embedding2 = temp_embeddings[1]
similarity = cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
print(similarity)


from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

# 1️⃣ Define Dataset
class ShopeeDataset(Dataset):
    def __init__(self, df, image_dir):
        self.df = df
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image']
        img = Image.open(f"{self.image_dir}/{img_name}")
        return img  # return PIL image directly

# 2️⃣ Define collate_fn
def pil_collate(batch):
    return batch  # return list of PIL images as-is

# 3️⃣ Create dataset and dataloader
dataset = ShopeeDataset(train_df, "/kaggle/input/shopee-product-matching/train_images")
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    collate_fn=pil_collate  # use the custom collate function
)

# 4️⃣ Compute embeddings
embeddings_list = []
model = model.to("cuda")

for batch in tqdm(loader, desc="Embedding images"):
    # batch is a list of PIL images
    inputs = image_processor(batch, return_tensors="pt").to("cuda")  # convert batch to tensor and move to GPU
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings_list.append(outputs.pooler_output.cpu())  # move embeddings back to CPU

# 5️⃣ Combine all batches
embeddings = torch.cat(embeddings_list)




"""
embeddings_list = []
batch_size = 32  

model = model.to("cuda")
for i in range(0, len(train_df), batch_size):
    batch_df = train_df.iloc[i:i+batch_size]
    imgs = [Image.open(f"/kaggle/input/shopee-product-matching/train_images/{img_name}") for img_name in batch_df['image']]

    inputs = image_processor(imgs, return_tensors="pt").to("cuda")  # move to GPU
    with torch.no_grad():
        outputs = model(**inputs)
        batch_embeds = outputs.pooler_output.cpu()  # move back to CPU
    embeddings_list.append(batch_embeds)

# Combine all batches into one tensor
embeddings = torch.cat(embeddings_list)
"""


# Cache the embedding
torch.save(embeddings, "train_embeddings.pt")


embeddings.shape


import faiss
import numpy as np

# Flatten to [num_items, embedding_dim]
embeddings_flat = embeddings.view(embeddings.size(0), -1)  # [34250, 2048]

# embeddings_flat: [num_items, dim] as float32 numpy array
embeddings_np = embeddings_flat.cpu().numpy().astype('float32')

index = faiss.IndexFlatIP(embeddings_np.shape[1])  # inner product = cosine if vectors are normalized
faiss.normalize_L2(embeddings_np)  # normalize embeddings to use cosine similarity

index.add(embeddings_np)
top_k = 10
distances, indices = index.search(embeddings_np, top_k + 1)  # +1 to skip self



top_k = 20
normalised = embedding2.unsqueeze(0).cpu().numpy()
faiss.normalize_L2(normalised)
distances, indices = index.search(normalised, top_k + 1)  # +1 to include self


indices


import matplotlib.pyplot as plt

# Make sure query_image_names is a proper list of filenames
query_image_names = train_df['image'].iloc[indices[0]].tolist()  # take first row if 2D

# Path to images
image_dir = "/kaggle/input/shopee-product-matching/train_images"

# Grid settings
top_k = len(query_image_names)
cols = 5  # number of columns
rows = (top_k + cols - 1) // cols  # number of rows needed

plt.figure(figsize=(cols*2, rows*2))  # small images

# Plot images
for i, img_name in enumerate(query_image_names):
    plt.subplot(rows, cols, i+1)
    img = Image.open(f"{image_dir}/{img_name}")
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"{i+1}", fontsize=8)

plt.tight_layout()
plt.show()






embedding1


# Normalise Image Embeddings
embedding1 = embedding1 / np.linalg.norm(embedding1)
embedding2 = embedding2 / np.linalg.norm(embedding2)

# Convert the tf-idf vector to dense (idk what this does)
X0_dense = X[0].toarray().ravel()
X0_dense = torch.from_numpy(X0_dense).float()

X1_dense = X[1].toarray().ravel()
X1_dense = torch.from_numpy(X1_dense).float()
# Concatenate
posting1_embedding = torch.cat([embedding1, X0_dense])
posting2_embedding = torch.cat([embedding2, X1_dense])



similarity = cosine_similarity(posting1_embedding.unsqueeze(0), posting2_embedding.unsqueeze(0))
print(similarity)


# Create a function to calculate the similarity between two postings
def get_similarity(posting1: str, posting2: str, model, processor):
    """
    Obtains image and text embeddings for each posting and concatenates them.
    Cosine similarity is then calculated between the joint embeddings of the
    two postings.

    Args:
        posting1 (str): posting_id of first posting
        posting2 (str): posting_id of second posting
        model (transformers.ResNetModel): Hugging Face ResNet model used to obtain image embeddings.
        processor (transformers.AutoImageProcessor): Processor to prepare images for the ResNet model.


    Returns:
        torch.Tensor: 1D Tensor containing the cosine similarity between the two postings
    """
    
    # Image Embedding
    img1_name = train_df[train_df["posting_id"] == posting1]["image"].iloc[0]
    img2_name = train_df[train_df["posting_id"] == posting2]["image"].iloc[0]
    img1 = Image.open(f"/kaggle/input/shopee-product-matching/train_images/{img1_name}")
    img2 = Image.open(f"/kaggle/input/shopee-product-matching/train_images/{img2_name}")
    inputs = image_processor([img1, img2], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        image_embeddings = outputs.pooler_output
    image_embeddings = image_embeddings.squeeze()
    image_embedding1 = image_embeddings[0]
    image_embedding2 = image_embeddings[1]
    
    # Normalise
    image_embedding1 = image_embedding1 / image_embedding1.norm()
    image_embedding2 = image_embedding2 / image_embedding2.norm()
    
    
    # Text Embedding
    title1 = train_df[train_df["posting_id"] == posting1]["title"].iloc[0]
    title2 = train_df[train_df["posting_id"] == posting2]["title"].iloc[0]
    X = vectorizer.transform([title1, title2])
    X0_dense = X[0].toarray().ravel()
    X0_dense = torch.from_numpy(X0_dense).float()
    X1_dense = X[1].toarray().ravel()
    X1_dense = torch.from_numpy(X1_dense).float()

    # Combine Embeddings
    posting1_embedding = torch.cat([image_embedding1, X0_dense])
    posting2_embedding = torch.cat([image_embedding2, X1_dense])

    return cosine_similarity(posting1_embedding.unsqueeze(0), posting2_embedding.unsqueeze(0))
    
    


train_df[train_df["label_group"] == 994676122].head()


get_similarity("train_1010868925", "train_1561375840", model, image_processor)


import torch
import pandas as pd
from tqdm import tqdm

def compute_similarity_matrix(df, model, processor):
    posting_ids = df['posting_id'].tolist()
    n = len(posting_ids)
    
    # Initialize empty matrix
    sim_matrix = pd.DataFrame(index=posting_ids, columns=posting_ids, dtype=float)
    
    # Compute upper-triangle (including diagonal)
    for i in tqdm(range(n)):
        for j in range(i, n):
            print(j)
            posting1 = posting_ids[i]
            posting2 = posting_ids[j]
            
            sim = get_similarity(posting1, posting2, model, processor).item()
            sim_matrix.at[posting1, posting2] = sim
            sim_matrix.at[posting2, posting1] = sim  # symmetry
    
    return sim_matrix




similarity_matrix = compute_similarity_matrix(train_df, model, image_processor)




