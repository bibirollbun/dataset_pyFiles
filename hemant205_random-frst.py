data_path = '/kaggle/input/fungi-clef-2025/'


!pip install git+https://github.com/mlfoundations/open_clip.git


import os
import json
import yaml
from pathlib import Path
from types import SimpleNamespace
import argparse

import numpy as np
import pandas as pd
import torch
# import faiss

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as tfms
import torchvision.transforms as T
import open_clip

from typing import Sequence, Tuple, Any, Dict, List, Optional, Union
import importlib


class FungiTastic(torch.nn.Module):
    """
    Dataset class for the FewShot subset of the Danish Fungi dataset (size 300, closed-set).

    This dataset loader supports training, validation, and testing splits, and provides
    convenient access to images, class IDs, and file paths. It also supports optional
    image transformations.
    """

    SPLIT2STR = {'train': 'Train', 'val': 'Val', 'test': 'Test'}

    def __init__(self, root: str, split: str = 'val', transform=None):
        """
        Initializes the FungiTastic dataset.

        Args:
            root (str): The root directory of the dataset.
            split (str, optional): The dataset split to use. Must be one of {'train', 'val', 'test'}.
                Defaults to 'val'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__()
        self.split = split
        self.transform = transform
        self.df = self._get_df(root, split)

        assert "image_path" in self.df
        if self.split != 'test':
            assert "category_id" in self.df
            self.n_classes = len(self.df['category_id'].unique())
            self.category_id2label = {
                k: v[0] for k, v in self.df.groupby('category_id')['species'].unique().to_dict().items()
            }
            self.label2category_id = {
                v: k for k, v in self.category_id2label.items()
            }

    def add_embeddings(self, embeddings: pd.DataFrame):
        """
        Updates the dataset instance with new embeddings.

        Args:
            embeddings (pd.DataFrame): A DataFrame containing an 'embedding' column.
                                       It must align with `self.df` in terms of indexing.
        """
        assert isinstance(embeddings, pd.DataFrame), "Embeddings must be a pandas DataFrame."
        assert "embedding" in embeddings.columns, "Embeddings DataFrame must have an 'embedding' column."
        assert len(embeddings) == len(self.df), "Embeddings must match dataset length."

        self.df = pd.merge(self.df, embeddings, on="filename", how="inner")

    def get_embeddings_for_class(self, id):
        # return the embeddings for class class_idx
        class_idxs = self.df[self.df['category_id'] == id].index
        return self.df.iloc[class_idxs]['embedding']
    
    @staticmethod
    def _get_df(data_path: str, split: str) -> pd.DataFrame:
        """
        Loads the dataset metadata as a pandas DataFrame.

        Args:
            data_path (str): The root directory where the dataset is stored.
            split (str): The dataset split to load. Must be one of {'train', 'val', 'test'}.

        Returns:
            pd.DataFrame: A DataFrame containing metadata and file paths for the split.
        """
        df_path = os.path.join(
            data_path,
            "metadata",
            "FungiTastic-FewShot",
            f"FungiTastic-FewShot-{FungiTastic.SPLIT2STR[split]}.csv"
        )
        df = pd.read_csv(df_path)
        df["image_path"] = df.filename.apply(
            lambda x: os.path.join(data_path, "FungiTastic-FewShot", split, '300p', x)
        )
        return df

    def __getitem__(self, idx: int):
        """
        Retrieves a single data sample by index.
    
        Args:
            idx (int): Index of the sample to retrieve.
            ret_image (bool, optional): Whether to explicitly return the image. Defaults to False.
    
        Returns:
            tuple:
                - If embeddings exist: (image?, embedding, category_id, file_path)
                - If no embeddings: (image, category_id, file_path) (original version)
        """
        file_path = self.df["image_path"].iloc[idx].replace('FungiTastic-FewShot', 'images/FungiTastic-FewShot')
    
        if self.split != 'test':
            category_id = self.df["category_id"].iloc[idx]
        else:
            category_id = None
    
        image = Image.open(file_path)
    
        if self.transform:
            image = self.transform(image)
    
        # Check if embeddings exist
        if "embedding" in self.df.columns:
            emb = torch.tensor(self.df.iloc[idx]['embedding'], dtype=torch.float32).squeeze()
        else:
            emb = None  # No embeddings available
    

        return image, category_id, file_path, emb


    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.df)

    def get_class_id(self, idx: int) -> int:
        """
        Returns the class ID of a specific sample.
        """
        return self.df["category_id"].iloc[idx]

    def show_sample(self, idx: int) -> None:
        """
        Displays a sample image along with its class name and index.
        """
        image, category_id, _, _ = self.__getitem__(idx)
        class_name = self.category_id2label[category_id]

        plt.imshow(image)
        plt.title(f"Class: {class_name}; id: {idx}")
        plt.axis('off')
        plt.show()

    def get_category_idxs(self, category_id: int) -> List[int]:
        """
        Retrieves all indexes for a given category ID.
        """
        return self.df[self.df.category_id == category_id].index.tolist()


class BioCLIP(torch.nn.Module):

    def __init__(self, device):
        """
        Initialize the BioCLIP feature extractor.
        """
        super(BioCLIP, self).__init__()
        self.device = device
        self.model = None
        self.processor = None
        self.size = (224, 224)

    def load(self):
        """
        Load the BioCLIP model and its associated image processor.
        
        The model is loaded from the Hugging Face Hub and moved to the specified device.
        """
        self.model, _, self.processor = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
        self.model.to(self.device)

    def extract_features(self, image):
        """
        Extract normalized feature embeddings from a given image.

        Args:
            image (PIL.Image.Image): The input image from which to extract features.

        Returns:
            torch.Tensor: A normalized feature embedding vector for the input image.

        Raises:
            ValueError: If the model has not been loaded prior to calling this method.
        """
        if self.model is None:
            raise ValueError('Model not loaded')
        image = image.resize(self.size, Image.BICUBIC)
        image_tensor_proc = self.processor(image)[None]
        features = self.model.encode_image(image_tensor_proc.to(self.device))
        return self.normalize_embedding(features)

    @staticmethod
    def normalize_embedding(embs):
        """
        Normalize the embedding vectors to have unit length.

        Args:
            embs (torch.Tensor): The raw embedding vectors.

        Returns:
            torch.Tensor: L2-normalized embedding vectors.
        """
        return torch.nn.functional.normalize(embs.float(), dim=1, p=2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BioCLIP(device=device)
model.load()
model.eval()

def generate_embeddings(dataset):

    idxs = np.arange(len(dataset))
    im_names, embs = [], []
    for idx in tqdm(idxs):
        im, label, file_path, _ = dataset[idx]

        with torch.no_grad():
            feat = model.extract_features(im)
            
        # feat_quant = model.quantize_normalized_embedding(feat)

        im_names.append(os.path.basename(file_path))
        embs.append(feat.detach().cpu().numpy())

    embeddings = pd.DataFrame({'filename': im_names, 'embedding': embs})

    return embeddings


### Load the datasets

train_dataset = FungiTastic(root=data_path, split='train', transform=None)
test_dataset = FungiTastic(root=data_path, split='test', transform=None)


embeddings = generate_embeddings(train_dataset)
train_dataset.add_embeddings(embeddings)
embeddings = generate_embeddings(test_dataset)
test_dataset.add_embeddings(embeddings)


train_df=train_dataset.df
test_df=test_dataset.df
test_df = test_df.drop_duplicates(subset=['observationID'], keep='first')


train_df['embedding_1d'] = train_df['embedding'].apply(lambda x: list(np.array(x).flatten()))
test_df['embedding_1d'] = test_df['embedding'].apply(lambda x: list(np.array(x).flatten()))
train_df.drop('embedding',axis=1,inplace=True)
test_df.drop('embedding',axis=1,inplace=True)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_df['category_iden']=le.fit_transform(train_df['category_id'])


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, make_scorer



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



rf_model = RandomForestClassifier(n_estimators=100, random_state=42,verbose=2)
cv_scores = cross_val_score(
    estimator=rf_model,
    X=df_expanded,
    y=target,
    cv=skf,          # Use the StratifiedKFold splitter
    scoring='f1',    # Choose an appropriate metric
    n_jobs=-1        # Use all processors
)


print(f"Individual Fold F1-Scores: {cv_scores}")
print(f"Mean F1-Score: {np.mean(cv_scores):.4f}")
print(f"Standard Deviation of F1-Scores: {np.std(cv_scores):.4f}")





cols=['eventDate', 'year', 'month', 'day', 'habitat', 'countryCode',
       'hasCoordinate', 'substrate', 'latitude', 'longitude', 'coorUncert',
       'observationID', 'region', 'district', 'filename', 'metaSubstrate',
       'elevation', 'landcover', 'biogeographicalRegion', 'image_path',
       'embedding_1d','category_id']


train_df['embedding_1d'].tolist()


X = train_df['embedding_1d'].apply(pd.Series)


y=train_df['category_id']


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
f1_scores = []

for train_idx, val_idx in skf.split(X, y):
    print(f"\n=== Fold {fold} ===")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    rf_model = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
        verbose=2,  # logs for training each tree
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_val)
    f1 = f1_score(y_val, y_pred,average='macro')
    print(f"Fold {fold} F1-Score: {f1:.4f}")
    f1_scores.append(f1)
    fold += 1

print(f"\nMean F1-Score: {np.mean(f1_scores):.4f}")
print(f"Std F1-Score: {np.std(f1_scores):.4f}")



y_pred.shape
y_train.shape











train

