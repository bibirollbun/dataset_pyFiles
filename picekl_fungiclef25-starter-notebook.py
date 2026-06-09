###Install required libraries:
!pip install git+https://github.com/mlfoundations/open_clip.git
!pip install faiss-gpu -qq


import os
import json
import yaml
from pathlib import Path
from types import SimpleNamespace
import argparse

import numpy as np
import pandas as pd
import torch
import faiss

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as tfms
import torchvision.transforms as T
import open_clip

from typing import Sequence, Tuple, Any, Dict, List, Optional, Union
import importlib


# path to fungitatsic dataset
data_path = '/kaggle/input/fungi-clef-2025/'



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


### Load the datasets

train_dataset = FungiTastic(root=data_path, split='train', transform=None)
test_dataset = FungiTastic(root=data_path, split='test', transform=None)

train_dataset.df.head(5)


### Visualize few samples

train_dataset.show_sample(1), train_dataset.show_sample(500), train_dataset.show_sample(1000) 


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


embeddings = generate_embeddings(test_dataset)
test_dataset.add_embeddings(embeddings)


embeddings = generate_embeddings(train_dataset)
train_dataset.add_embeddings(embeddings)


class NNClassifier():
    def __init__(self, train_embeddings, device='cuda'):
        """
        :param cfg: config object, namespace
        :param train_embeddings: list of C torch arrays of shape [N_C, D] where N_C is the number of training samples
        of class C and D is the dimensionality of the embeddings
        """
        
        self.device = device
        self.index, self.idx2cls = self.build_index(train_embeddings)

    def make_prediction(self, embeddings, plot_sim_hist=False, ret_probs=False):
        """
        :param embeddings: torch.Tensor of shape (batch_size, n_channels, height, width)
        :return: probabilities of shape (batch_size, n_classes) computed based on
        the similarity of the embeddings to the class prototypes
        """
        # compute the similarity of each embedding to each prototype
        # embeddings - [N, D], class_prototypes - [C, D]
        similarities, indices = self.index.search(embeddings, 1)
        # get the classes for the indices
        cls = self.idx2cls[indices.squeeze()]
        # get the confidence of the prediction
        conf = similarities
        
        return cls, conf

    def build_index(self, train_dataset):
        idx2cls = train_dataset.df.category_id.values
        # concatenate the embeddings
        embs = np.array(train_dataset.df.embedding.values.tolist(), dtype=np.float32).squeeze()
        # build the index for cosine similarity search
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        return index, idx2cls


classifier = NNClassifier(train_dataset, device='cpu')

cls, conf = classifier.make_prediction(np.array(test_dataset.df.embedding.values.tolist(), dtype=np.float32).squeeze())


test_dataset.df["preds"] = cls
submission = (
    test_dataset.df.groupby("observationID")["preds"]
    .apply(lambda x: " ".join(map(str, sorted(set(x)))))  # Sort and remove duplicates
    .reset_index()
)

submission = submission.rename(columns={"observationID": "observationId", "preds": "predictions"})
submission = submission.drop_duplicates(subset="observationId")
submission.to_csv("baseline-submission-with-nn.csv", index=None)


class PrototypeClassifier(torch.nn.Module):
    def __init__(self, train_dataset, device='cuda'):
        super().__init__()
        self.device = device
        self.train_dataset = train_dataset

        class_embeddings, _ = self._get_classifier_embeddings(train_dataset)
        self.class_prototypes = torch.nn.Parameter(self.get_prototypes(class_embeddings), requires_grad=False)

    def _get_classifier_embeddings(self, dataset_train):
        class_embeddings = []
        empty_classes = []
        n_classes = min(torch.inf, dataset_train.n_classes)
        for cls in range(n_classes):
            cls_embs = dataset_train.get_embeddings_for_class(cls)
            if len(cls_embs) == 0:
                # if no embeddings for class, use zeros
                empty_classes.append(cls)
                class_embeddings.append(torch.zeros(1, dataset_train.emb_dim))
            else:
                class_embeddings.append(torch.tensor(np.vstack(cls_embs.values)))
        return class_embeddings, empty_classes

    def get_prototypes(self, embeddings):
        return torch.stack([class_embs.mean(dim=0) for class_embs in embeddings])

    def make_prediction(self, embeddings):
        similarities = torch.nn.functional.cosine_similarity(embeddings, self.class_prototypes, dim=-1)
        cls = torch.argmax(similarities, dim=1)
        conf = torch.nn.functional.softmax(similarities, dim=1).max(dim=1).values
        return cls, conf


classifier = PrototypeClassifier(train_dataset, device='cpu')
cls, conf = classifier.make_prediction(torch.tensor(np.array(test_dataset.df.embedding.values.tolist(), dtype=np.float32)))


test_dataset.df["preds"] = cls
submission = (
    test_dataset.df.groupby("observationID")["preds"]
    .apply(lambda x: " ".join(map(str, sorted(set(x)))))  # Sort and remove duplicates
    .reset_index()
)

submission = submission.rename(columns={"observationID": "observationId", "preds": "predictions"})
submission = submission.drop_duplicates(subset="observationId")
submission.to_csv("baseline-submission-with-prototypes.csv", index=None)

