# installations
!pip install -q umap-learn rdkit captum git+https://github.com/samoturk/mol2vec;
!pip install -q selfies==2.1.1  simpletransformers==0.63.9 pandarallel==1.6.4 wandb==0.13.10


# Imports

import torch 
import numpy as np
import pandas as pd
import os
import random
import kagglehub
from kaggle_secrets import UserSecretsClient

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'\nUsing {device}')

seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
print('-----Seed Set!-----')

user_secrets = UserSecretsClient()
os.environ["KAGGLE_USERNAME"] = user_secrets.get_secret("Kaggle_Username")
os.environ["KAGGLE_KEY"] = user_secrets.get_secret("Kaggle_Key")
kagglehub.login()


# uncomment to download data
# !kaggle competitions download -c open-problems-single-cell-perturbations
# !unzip -q  open-problems-single-cell-perturbations.zip -d opxmoe_data


from gensim.models import word2vec
model1 = word2vec.Word2Vec.load(f"/kaggle/input/mol2vec/pytorch/default/1/model_300dim.pkl")


df = pd.read_parquet('../input/open-problems-single-cell-perturbations/de_train.parquet')
df.tail()


df.columns


# Function to convert SMILES to SELFIES
from selfies import encoder
def smiles_to_selfies(smiles):
    try:
        return encoder(smiles)
    except Exception as e:
        print(f"Error converting SMILES '{smiles}': {e}") 
        return None  


from abc import ABC, abstractmethod
import json
import pickle
from typing import Dict, Optional, Union, List, Tuple, Literal
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, silhouette_samples

NON_GENE_COLUMNS = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control', 'SELFIES']

class BaseEmbedding(ABC):
    """
    Abstract base class for embeddings with enhanced functionality for gene expression prediction.
    
    This class provides a framework for different types of embeddings (e.g., cell type, small molecule)
    with consistent interfaces for preprocessing, computing, storing, and retrieving embeddings.
    """
    
    def __init__(self, name: str = "base"):
        """
        Initialize the embedding class.
        
        Args:
            name (str): Identifier for the embedding type
        """
        self.embedding_dict: Dict[str, np.ndarray] = {}
        self.metadata: Dict = {}
        self.name = name
        self._is_fitted = False
    
    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the input data before computing embeddings.
        
        Args:
            df (pd.DataFrame): Input DataFrame containing entity information
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        pass
    
    @abstractmethod
    def get_embedding(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Compute embeddings for the input data.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            fit (bool): Whether to fit the embedding model or use pre-fitted model
            
        Returns:
            np.ndarray: Computed embeddings
        """
        pass
    
    @property
    @abstractmethod
    def embedding_size(self) -> int:
        """
        Returns the size of the embedding vector.
        
        Returns:
            int: Size of embedding vector
        """
        pass
    
    def compute_and_store_embeddings(
        self, 
        df: pd.DataFrame, 
        entity_column: str,
        batch_size: Optional[int] = None
    ) -> None:
        """
        Compute and store embeddings for unique entities in the specified column.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            entity_column (str): Column name containing entities (e.g., 'cell_type' or 'sm_name')
            batch_size (Optional[int]): Batch size for processing large datasets
        """
        unique_entities = df[entity_column].unique()
        
        if batch_size:
            # Process in batches
            for i in range(0, len(unique_entities), batch_size):
                batch_entities = unique_entities[i:i + batch_size]
                batch_df = df[df[entity_column].isin(batch_entities)].copy()
                self._process_batch(batch_df, entity_column, batch_entities)
        else:
            self._process_batch(df, entity_column, unique_entities)
        
        self._is_fitted = True
    
    def _process_batch(
        self, 
        df: pd.DataFrame, 
        entity_column: str, 
        entities: np.ndarray
    ) -> None:
        """
        Process a batch of entities and compute their embeddings.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            entity_column (str): Column name containing entities
            entities (np.ndarray): Array of entity names to process
        """
        for entity in entities:
            entity_df = df[df[entity_column] == entity].copy()
            entity_df = self.preprocess(entity_df)
            embedding = self.get_embedding(entity_df, fit=False)
            
            # Handle different embedding return types
            if isinstance(embedding, (torch.Tensor, np.ndarray)):
                self.embedding_dict[entity] = (
                    embedding.mean(axis=0).cpu().numpy() 
                    if isinstance(embedding, torch.Tensor) 
                    else embedding.mean(axis=0)
                )
            else:
                raise ValueError(f"Unsupported embedding type: {type(embedding)}")
    
    def get_entity_embedding(
        self, 
        entity_name: str,
        allow_missing: bool = True
    ) -> np.ndarray:
        """
        Retrieve embedding for a specific entity.
        
        Args:
            entity_name (str): Name of the entity
            allow_missing (bool): If True, return zero vector for missing entities
            
        Returns:
            np.ndarray: Embedding vector for the entity
            
        Raises:
            KeyError: If entity is missing and allow_missing is False
        """
        if entity_name in self.embedding_dict:
            return self.embedding_dict[entity_name]
        elif allow_missing:
            return np.zeros(self.embedding_size)
        else:
            raise KeyError(f"No embedding found for entity: {entity_name}")
    
    def get_multiple_embeddings(
        self, 
        entity_names: List[str]
    ) -> np.ndarray:
        """
        Retrieve embeddings for multiple entities at once.
        
        Args:
            entity_names (List[str]): List of entity names
            
        Returns:
            np.ndarray: Stack of embedding vectors
        """
        return np.stack([self.get_entity_embedding(name) for name in entity_names])
    

    def save_embedding(
        self, 
        filepath: str, 
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Save embeddings and metadata to disk.
        
        Args:
            filepath (str): Path to save the embeddings
            metadata (Optional[Dict]): Additional metadata to save
        """
        if not self._is_fitted:
            raise ValueError("Cannot save embeddings before computing them")
            
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        # Convert numpy arrays to lists for JSON serialization
        serializable_dict = {
            k: v.tolist() if isinstance(v, (np.ndarray, torch.Tensor)) else v 
            for k, v in self.embedding_dict.items()
        }
        
        # Prepare save data
        save_data = {
            'embeddings': serializable_dict,
            'metadata': metadata or self.metadata,
            'embedding_size': self.embedding_size,
            'embedding_type': self.__class__.__name__,
            'name': self.name
        }
        
        # Save as pickle if the filepath ends with .pkl, otherwise save as JSON
        if filepath.endswith('.pkl'):
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
        else:
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2)
                
    def load_embeddings(self, filepath: str) -> bool:
        """
        Load embeddings and metadata from disk.
        
        Args:
            filepath (str): Path to load the embeddings from
            
        Returns:
            bool: True if loading was successful
            
        Raises:
            ValueError: If loaded embedding size doesn't match current size
        """
        try:
            # Load pickle if the filepath ends with .pkl, otherwise load JSON
            if filepath.endswith('.pkl'):
                with open(filepath, 'rb') as f:
                    save_data = pickle.load(f)
            else:
                with open(filepath, 'r') as f:
                    save_data = json.load(f)
            
            # Convert lists back to numpy arrays
            self.embedding_dict = {
                k: np.array(v) if isinstance(v, list) else v 
                for k, v in save_data['embeddings'].items()
            }
            
            self.metadata = save_data.get('metadata', {})
            self.name = save_data.get('name', self.name)
            
            # Verify embedding size matches
            if save_data['embedding_size'] != self.embedding_size:
                raise ValueError(
                    f"Loaded embedding size ({save_data['embedding_size']}) "
                    f"doesn't match current embedding size ({self.embedding_size})"
                )
            
            self._is_fitted = True
            return True
            
        except Exception as e:
            print(f"Error loading embeddings: {str(e)}")
            return False


from sklearn.preprocessing import OneHotEncoder

class OneHotEmbedding(BaseEmbedding):
    """
    One-hot encoding implementation of the BaseEmbedding class.
    Handles both single and multiple column encodings.
    """
    def __init__(self, columns: Union[str, List[str]], name: str = "onehot"):
        """
        Initialize OneHotEmbedding.
        
        Args:
            columns: Column name(s) to encode
            name: Identifier for this embedding
        """
        super().__init__(name=name)
        self.columns = [columns] if isinstance(columns, str) else columns
        self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self._embedding_size = None
        self.feature_names = None
        self._is_fitted = False
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select relevant columns for encoding.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with selected columns
        """
        return df[self.columns]
    
    def get_embedding(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Get one-hot encoded features.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the encoder or use pre-fitted encoder
            
        Returns:
            Array of one-hot encoded features
        """
        input_data = df[self.columns].values
        
        if fit:
            encoded_features = self.encoder.fit_transform(input_data)
            self._is_fitted = True
            self.feature_names = self.encoder.get_feature_names_out(self.columns)
        else:
            if not self._is_fitted:
                raise ValueError("Encoder must be fitted before transform")
            encoded_features = self.encoder.transform(input_data)
        
        self._embedding_size = encoded_features.shape[1]
        
        # Store embeddings for each unique combination
        for idx, row in df[self.columns].iterrows():
            key = tuple(row.values) if len(self.columns) > 1 else row.values[0]
            self.embedding_dict[key] = encoded_features[idx]
        
        return encoded_features
    
    @property
    def embedding_size(self) -> int:
        """
        Get the size of the one-hot encoded vector.
        
        Returns:
            Size of the embedding vector
        """
        if self._embedding_size is None:
            raise ValueError("Embedding size not set. Call get_embedding first.")
        return self._embedding_size
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of the one-hot encoded features.
        
        Returns:
            List of feature names
        """
        if self.feature_names is None:
            raise ValueError("Feature names not available. Call get_embedding with fit=True first.")
        return self.feature_names.tolist()


from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, QED
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcNumRotatableBonds, CalcNumHBA, CalcNumHBD, CalcFractionCSP3
from rdkit.Chem import BRICS, Recap
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

@dataclass
class MoleculeFeatures:
    """Dataclass to store molecular features with their descriptions."""
    name: str
    function: callable
    description: str

class SMILESEmbedding(BaseEmbedding):
    """
    SMILES-based molecular embedding using RDKit descriptors.
    Converts SMILES strings into numerical feature vectors using chemical descriptors.
    """
    
    # Define molecular features as class attribute for easy access and modification
    MOLECULAR_FEATURES = [
        MoleculeFeatures("Molecular Weight", Descriptors.MolWt, "Molecular weight of the compound"),
        MoleculeFeatures("LogP", Descriptors.MolLogP, "Octanol-water partition coefficient"),
        MoleculeFeatures("TPSA", CalcTPSA, "Topological polar surface area"),
        MoleculeFeatures("Number of Atoms", lambda m: m.GetNumAtoms(), "Total number of atoms"),
        MoleculeFeatures("Number of Bonds", lambda m: m.GetNumBonds(), "Total number of bonds"),
        MoleculeFeatures("Number of Rotatable Bonds", CalcNumRotatableBonds, "Number of rotatable bonds"),
        MoleculeFeatures("Number of Hydrogen Bond Acceptors", CalcNumHBA, "Number of H-bond acceptors"),
        MoleculeFeatures("Number of Hydrogen Bond Donors", CalcNumHBD, "Number of H-bond donors"),
        MoleculeFeatures("Number of Rings", Descriptors.RingCount, "Total number of rings"),
        MoleculeFeatures("Number of Aromatic Rings", rdMolDescriptors.CalcNumAromaticRings, "Number of aromatic rings"),
        MoleculeFeatures("Number of Stereocenters", 
                        lambda m: len(Chem.FindMolChiralCenters(m, includeUnassigned=True)),
                        "Number of stereogenic centers"),
        MoleculeFeatures("Fraction of sp3 Carbons", CalcFractionCSP3, "Fraction of sp3 hybridized carbons"),
        MoleculeFeatures("Balaban J Index", Descriptors.BalabanJ, "Topological connectivity index"),
        MoleculeFeatures("Bertz CT", Descriptors.BertzCT, "Complexity index"),
        MoleculeFeatures("QED Score", QED.qed, "Drug-likeness score")
    ]

    def __init__(self, name: str = "smiles", handle_errors: bool = True):
        """
        Initialize SMILESEmbedding.
        
        Args:
            name: Identifier for this embedding
            handle_errors: If True, return null values for invalid SMILES
        """
        super().__init__(name=name)
        self.scaler = StandardScaler()
        self.handle_errors = handle_errors
        self._feature_names = [f.name for f in self.MOLECULAR_FEATURES]
        self._null_embedding = np.zeros(self.embedding_size)
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Verify SMILES column exists and remove invalid entries."""
        if 'SMILES' not in df.columns:
            raise ValueError("DataFrame must contain 'SMILES' column")
        
        # Remove invalid SMILES
        valid_mask = df['SMILES'].apply(lambda x: bool(x) and Chem.MolFromSmiles(x) is not None)
        if not valid_mask.all() and not self.handle_errors:
            invalid_count = (~valid_mask).sum()
            raise ValueError(f"Found {invalid_count} invalid SMILES strings")
        
        return df[valid_mask]
    
    def create_molecule_embedding_dict(self, df: pd.DataFrame) -> None:
        """
        Create a dictionary of molecule embeddings.
        
        Args:
            df: DataFrame containing SMILES column
        """
        df = self.preprocess(df)
        self.compute_and_store_embeddings(df, 'SMILES')
    
    def get_embedding(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Get molecular descriptor-based embeddings for SMILES strings.
        
        Args:
            df: DataFrame containing SMILES column
            fit: Whether to fit the scaler or use pre-fitted scaler
            
        Returns:
            Array of molecular descriptors
        """
        df = self.preprocess(df)
        
        # Extract features for each SMILES
        features_list = []
        for smiles in df['SMILES']:
            try:
                features = self.extract_smiles_info(smiles)
                features_list.append([features[f.name] for f in self.MOLECULAR_FEATURES])
            except Exception as e:
                if self.handle_errors:
                    features_list.append(self._null_embedding)
                else:
                    raise ValueError(f"Error processing SMILES {smiles}: {str(e)}")
        
        # Convert to array and scale
        features_array = np.array(features_list)
        if fit:
            features_scaled = self.scaler.fit_transform(features_array)
        else:
            if not hasattr(self.scaler, 'mean_'):
                raise ValueError("Scaler must be fitted before transform")
            features_scaled = self.scaler.transform(features_array)
        
        # Store embeddings
        for idx, smiles in enumerate(df['SMILES']):
            self.embedding_dict[smiles] = features_scaled[idx]
        
        return features_scaled
    
    @staticmethod
    def extract_smiles_info(smiles: str) -> Dict[str, float]:
        """
        Extract molecular descriptors from SMILES string.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary of molecular descriptors
        """
        if not smiles:
            raise ValueError("Empty SMILES string")
            
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        
        return {
            feature.name: feature.function(mol)
            for feature in SMILESEmbedding.MOLECULAR_FEATURES
        }
    
    @property
    def embedding_size(self) -> int:
        """Size of the molecular descriptor vector."""
        return len(self.MOLECULAR_FEATURES)
    
    def get_feature_importance(self, target_values: np.ndarray) -> pd.DataFrame:
        """
        Calculate correlation between features and target values.
        
        Args:
            target_values: Array of target values
            
        Returns:
            DataFrame with feature importances
        """
        if len(self.embedding_dict) == 0:
            raise ValueError("No embeddings computed yet")
            
        embeddings = np.stack(list(self.embedding_dict.values()))
        correlations = np.corrcoef(embeddings.T, target_values.reshape(1, -1))[-1, :-1]
        
        return pd.DataFrame({
            'Feature': self._feature_names,
            'Correlation': correlations,
            'Absolute Correlation': np.abs(correlations)
        }).sort_values('Absolute Correlation', ascending=False)


import torch.nn as nn
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class Target_autoencoder(nn.Module):
    """Autoencoder module with matching architecture from the previous implementation."""
    
    def __init__(self, input_size: int, hidden_size: int, emb_size: int):
        """
        Initialize the autoencoder with encoder and decoder networks.
        
        Args:
            input_size (int): Input dimension size
            hidden_size (int): Size of hidden layers
            emb_size (int): Size of the latent embedding
        """
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, emb_size)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(emb_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, input_size)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Encoded and decoded tensors
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


import torch.nn as nn
from dataclasses import dataclass
from torch.utils.data import DataLoader, TensorDataset

class TargetEmbedding(BaseEmbedding):
    """
    Target embedding class using dual autoencoders to learn compressed representations 
    of cell type and small molecule medians from gene expression data.
    """
    
    def __init__(
        self,
        emb_size: int = 256,
        hidden_size: int = 1024,
        input_size: int = 18211,
        name: str = "target_embedding"
    ):
        """
        Initialize the target embedding class.
        
        Args:
            emb_size (int): Size of the latent embedding for each component
            hidden_size (int): Size of the hidden layer in encoder/decoder
            input_size (int): Input dimension of gene expression values
            name (str): Identifier for this embedding type
        """
        super().__init__(name=name)
        
        self.emb_size = emb_size
        self._embedding_size = emb_size * 2  # Combined size of both embeddings
        self.input_size = input_size
        
        # Initialize autoencoders
        self.cell_type_autoencoder = Target_autoencoder(input_size, hidden_size, emb_size)
        self.sm_autoencoder = Target_autoencoder(input_size, hidden_size, emb_size)
        
        # Initialize storage for median values
        self.cell_type_medians: Dict = {}
        self.sm_medians: Dict = {}
        self.cell_type_tensors: Dict = {}
        self.sm_tensors: Dict = {}
        
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the input DataFrame and compute medians if needed.
        
        Args:
            df (pd.DataFrame): Input DataFrame with cell_type, sm_name, and gene expression columns
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
            
        Raises:
            ValueError: If required columns are missing
        """
        df = df.copy()
        
        # Validate required columns
        required_cols = ['cell_type', 'sm_name']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Handle missing values in gene expression columns
        gene_cols = [col for col in df.columns if col not in NON_GENE_COLUMNS]
        df[gene_cols] = df[gene_cols].fillna(0)
        
        # Compute medians if not already computed
        if not self.cell_type_medians:
            self._compute_medians(df, gene_cols)
            
        return df
    
    def _compute_medians(self, df: pd.DataFrame, gene_cols: list) -> None:
        """
        Compute median gene expression values for cell types and small molecules.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            gene_cols (list): List of gene expression column names
        """
        # Compute cell type medians
        cell_type_medians = df.groupby('cell_type')[gene_cols].median()
        self.cell_type_medians = {
            ct: values.values for ct, values in cell_type_medians.iterrows()
        }
        self.cell_type_tensors = {
            k: torch.tensor(v, dtype=torch.float32) 
            for k, v in self.cell_type_medians.items()
        }
        
        # Compute small molecule medians
        sm_medians = df.groupby('sm_name')[gene_cols].median()
        self.sm_medians = {
            sm: values.values for sm, values in sm_medians.iterrows()
        }
        self.sm_tensors = {
            k: torch.tensor(v, dtype=torch.float32) 
            for k, v in self.sm_medians.items()
        }
    
    def train_autoencoders(
        self,
        num_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3
    ) -> None:
        """
        Train both autoencoders to learn compressed representations.
        
        Args:
            num_epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for optimizer
        """
        # Prepare data
        cell_type_data = torch.stack(list(self.cell_type_tensors.values()))
        sm_data = torch.stack(list(self.sm_tensors.values()))
        
        # Initialize optimizers
        optimizer = torch.optim.Adam(
            list(self.cell_type_autoencoder.parameters()) + 
            list(self.sm_autoencoder.parameters()),
            lr=learning_rate
        )
        criterion = nn.MSELoss()
        
        for epoch in range(num_epochs):
            # Train cell type autoencoder
            _, ct_decoded = self.cell_type_autoencoder(cell_type_data)
            ct_loss = criterion(ct_decoded, cell_type_data)
            
            # Train small molecule autoencoder
            _, sm_decoded = self.sm_autoencoder(sm_data)
            sm_loss = criterion(sm_decoded, sm_data)
            
            # Combined loss
            total_loss = ct_loss + sm_loss
            
            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Loss: {total_loss.item():.4f}, '
                      f'CT Loss: {ct_loss.item():.4f}, '
                      f'SM Loss: {sm_loss.item():.4f}')
        
        # Store embeddings in the base class dictionary
        self._store_embeddings()
        self._is_fitted = True
    
    def _store_embeddings(self) -> None:
        """Store computed embeddings in the base class dictionary."""
        with torch.no_grad():
            # Store cell type embeddings
            for ct, tensor in self.cell_type_tensors.items():
                encoded = self.cell_type_autoencoder.encoder(tensor.unsqueeze(0))
                self.embedding_dict[f"cell_type_{ct}"] = encoded.squeeze(0).numpy()
            
            # Store small molecule embeddings
            for sm, tensor in self.sm_tensors.items():
                encoded = self.sm_autoencoder.encoder(tensor.unsqueeze(0))
                self.embedding_dict[f"sm_name_{sm}"] = encoded.squeeze(0).numpy()
    
    def get_embedding(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Get embeddings for the input data.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            fit (bool): If True, train the autoencoders before getting embeddings
            
        Returns:
            np.ndarray: Combined embeddings for cell types and small molecules
            
        Raises:
            ValueError: If autoencoders are not trained and fit=False
        """
        df = self.preprocess(df)
        
        if fit and not self._is_fitted:
            self.train_autoencoders()
        elif not self._is_fitted:
            raise ValueError("Autoencoders must be trained before getting embeddings with fit=False")
        
        with torch.no_grad():
            # Get embeddings for each entity
            cell_type_embeddings = []
            sm_embeddings = []
            
            for _, row in df.iterrows():
                # Get cell type embedding
                ct_tensor = self.cell_type_tensors.get(row['cell_type'], 
                    torch.zeros(self.input_size, dtype=torch.float32))
                ct_emb = self.cell_type_autoencoder.encoder(ct_tensor.unsqueeze(0))
                cell_type_embeddings.append(ct_emb.squeeze(0))
                
                # Get small molecule embedding
                sm_tensor = self.sm_tensors.get(row['sm_name'],
                    torch.zeros(self.input_size, dtype=torch.float32))
                sm_emb = self.sm_autoencoder.encoder(sm_tensor.unsqueeze(0))
                sm_embeddings.append(sm_emb.squeeze(0))
            
            # Combine embeddings
            cell_type_tensor = torch.stack(cell_type_embeddings)
            sm_tensor = torch.stack(sm_embeddings)
            combined = torch.cat([cell_type_tensor, sm_tensor], dim=1)
            
        return combined.numpy()
    
    @property
    def embedding_size(self) -> int:
        """
        Returns the size of the combined embedding vector.
        
        Returns:
            int: Size of the combined embedding vector
        """
        return self._embedding_size


from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class MorganFingerPrintEmbedding(BaseEmbedding):
    """
    Morgan Fingerprint embedding class that generates and compresses molecular fingerprints
    using an autoencoder architecture.
    """
    
    def __init__(self, hidden_size: int = 128, name: str = "morgan_fingerprint"):
        """
        Initialize the Morgan Fingerprint embedding.
        
        Args:
            hidden_size (int): Size of the compressed fingerprint representation
            name (str): Identifier for this embedding type
        """
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.autoencoder = None
        self._input_size = 2048  # Default Morgan fingerprint size
        
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the input DataFrame. Verifies SMILES column exists.
        
        Args:
            df (pd.DataFrame): Input DataFrame with SMILES column
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
            
        Raises:
            ValueError: If SMILES column is missing
        """
        if 'SMILES' not in df.columns:
            raise ValueError("DataFrame must contain 'SMILES' column")
        return df
    
    def get_embedding(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Generate embeddings for the molecules in the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame with SMILES column
            fit (bool): Whether to train a new autoencoder or use existing one
            
        Returns:
            np.ndarray: Compressed fingerprint embeddings
            
        Raises:
            ValueError: If autoencoder is not trained and fit=False
        """
        morgan_fp_list = df['SMILES'].apply(self.extract_morgan_fingerprint)
        morgan_fp_array = np.stack([fp for fp in morgan_fp_list if fp is not None])
        
        if fit:
            self.autoencoder = self.train_autoencoder(
                morgan_fp_array, 
                self._input_size, 
                self.hidden_size
            )
        elif self.autoencoder is None:
            raise ValueError("Autoencoder must be trained before getting embeddings with fit=False")
        
        # Convert to tensor and get embeddings
        with torch.no_grad():
            embeddings = self.autoencoder.encoder(
                torch.tensor(morgan_fp_array, dtype=torch.float32)
            ).cpu().numpy()
            
        return embeddings
    
    def extract_morgan_fingerprint(
        self, 
        smiles: str, 
        radius: int = 2, 
        nBits: int = 2048
    ) -> Optional[np.ndarray]:
        """
        Extract Morgan fingerprint from SMILES string.
        
        Args:
            smiles (str): SMILES representation of molecule
            radius (int): Morgan fingerprint radius
            nBits (int): Number of bits in fingerprint
            
        Returns:
            Optional[np.ndarray]: Fingerprint array or None if conversion fails
        """
        if not smiles or not isinstance(smiles, str):
            return None
            
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        morgan_gen = AllChem.GetMorganGenerator(radius=radius, fpSize=nBits)
        fp = morgan_gen.GetFingerprint(mol)
        
        fp_array = np.zeros((nBits,))
        DataStructs.ConvertToNumpyArray(fp, fp_array)
        
        return fp_array
    
    def train_autoencoder(
        self, 
        data: np.ndarray, 
        input_size: int, 
        hidden_size: int,
        num_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ) -> Autoencoder:
        """
        Train the autoencoder model for fingerprint compression.
        
        Args:
            data (np.ndarray): Input fingerprint data
            input_size (int): Size of input fingerprints
            hidden_size (int): Size of compressed representation
            num_epochs (int): Number of training epochs
            batch_size (int): Training batch size
            learning_rate (float): Learning rate for optimization
            
        Returns:
            Autoencoder: Trained autoencoder model
        """
        autoencoder = Autoencoder(input_size=input_size, hidden_size=hidden_size)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
        
        dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(num_epochs):
            for batch in dataloader:
                inputs = batch[0]
                _, reconstructed = autoencoder(inputs)
                loss = criterion(reconstructed, inputs)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        return autoencoder
    
    @property
    def embedding_size(self) -> int:
        """
        Returns the size of the compressed fingerprint representation.
        
        Returns:
            int: Size of embedding vector
        """
        return self.hidden_size


from gensim.models import word2vec
from rdkit import Chem
from mol2vec.features import mol2alt_sentence, MolSentence

class Mol2VecEmbedding(BaseEmbedding):
    """
    Mol2Vec embedding class that generates molecular embeddings using a pre-trained Word2Vec model.
    The embeddings are based on molecular substructures represented as "sentences".
    """
    
    def __init__(
        self, 
        model_path: str,
        name: str = "mol2vec",
        radius: int = 1,
        unseen_vec: Optional[np.ndarray] = None
    ):
        """
        Initialize the Mol2Vec embedding class.
        
        Args:
            model_path (str): Path to the pre-trained Word2Vec model
            name (str): Identifier for this embedding type
            radius (int): Radius for molecular substructure generation
            unseen_vec (Optional[np.ndarray]): Vector to use for unseen substructures
        """
        super().__init__(name=name)
        
        # Load the pre-trained model
        self.model = word2vec.Word2Vec.load(model_path)
        self.keys = set(self.model.wv.key_to_index.keys())
        self.radius = radius
        
        # Initialize unseen vector if not provided
        if unseen_vec is None:
            self.unseen_vec = np.zeros(self.embedding_size)
        else:
            self.unseen_vec = unseen_vec
            
        self.metadata.update({
            'model_path': model_path,
            'radius': radius,
            'vocabulary_size': len(self.keys)
        })
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the input DataFrame by converting SMILES to molecules and generating
        substructure sentences.
        
        Args:
            df (pd.DataFrame): Input DataFrame with SMILES column
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
            
        Raises:
            ValueError: If SMILES column is missing or contains invalid molecules
        """
        if 'SMILES' not in df.columns:
            raise ValueError("DataFrame must contain 'SMILES' column")
        
        df = df.copy()
        
        # Convert SMILES to molecules
        df['mol'] = df['SMILES'].apply(self._smiles_to_mol)
        
        # Check for failed conversions
        failed_smiles = df[df['mol'].isna()]['SMILES'].tolist()
        if failed_smiles:
            raise ValueError(f"Failed to parse {len(failed_smiles)} SMILES strings: {failed_smiles[:5]}")
        
        # Generate substructure sentences
        df['sentence'] = df['mol'].apply(
            lambda x: MolSentence(mol2alt_sentence(x, self.radius))
        )
        
        return df
    
    def _smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """
        Convert SMILES string to RDKit molecule with error handling.
        
        Args:
            smiles (str): SMILES string
            
        Returns:
            Optional[Chem.Mol]: RDKit molecule object or None if conversion fails
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Warning: Could not parse SMILES: {smiles}")
            return mol
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {str(e)}")
            return None
    
    def get_embedding(
        self, 
        df: pd.DataFrame, 
        fit: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for the molecules in the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame with SMILES column
            fit (bool): Not used for Mol2Vec (pre-trained model)
            
        Returns:
            np.ndarray: Molecular embeddings
        """
        df = self.preprocess(df)
        
        # Generate embeddings for each molecule
        vectors = []
        for _, row in df.iterrows():
            vec = self.sentence_to_vector(
                sentence=row['sentence'],
                handle_unseen=True
            )
            vectors.append(vec)
            
            # Store in base class dictionary
            self.embedding_dict[row['SMILES']] = vec
        
        embeddings = np.stack(vectors)
        self._is_fitted = True
        
        return embeddings
    
    def sentence_to_vector(
        self, 
        sentence: MolSentence,
        handle_unseen: bool = True
    ) -> np.ndarray:
        """
        Convert a molecular sentence to a vector by averaging substructure vectors.
        
        Args:
            sentence (MolSentence): Molecular sentence (substructures)
            handle_unseen (bool): Whether to handle unseen substructures with unseen_vec
            
        Returns:
            np.ndarray: Molecular embedding vector
        """
        vectors = []
        for word in sentence:
            if word in self.keys:
                vectors.append(self.model.wv.get_vector(word))
            elif handle_unseen:
                vectors.append(self.unseen_vec)
                
        if not vectors:
            return self.unseen_vec
            
        return np.mean(vectors, axis=0)
    
    def get_entity_embedding(
        self, 
        smiles: str,
        allow_missing: bool = True
    ) -> np.ndarray:
        """
        Get embedding for a specific molecule by SMILES.
        
        Args:
            smiles (str): SMILES string of the molecule
            allow_missing (bool): If True, compute embedding for missing molecules
            
        Returns:
            np.ndarray: Molecular embedding vector
        """
        if smiles in self.embedding_dict:
            return self.embedding_dict[smiles]
        elif allow_missing:
            mol = self._smiles_to_mol(smiles)
            if mol is not None:
                sentence = MolSentence(mol2alt_sentence(mol, self.radius))
                return self.sentence_to_vector(sentence)
            return np.zeros(self.embedding_size)
        else:
            raise KeyError(f"No embedding found for SMILES: {smiles}")
    
    def save_embeddings(
        self, 
        filepath: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Save embeddings and metadata to disk.
        
        Args:
            filepath (str): Path to save the embeddings
            metadata (Optional[Dict]): Additional metadata to save
        """
        if metadata:
            self.metadata.update(metadata)
        super().save_embeddings(filepath, self.metadata)
    
    @property
    def embedding_size(self) -> int:
        """
        Returns the size of the molecular embedding vector.
        
        Returns:
            int: Size of embedding vector
        """
        return len(self.model.wv.get_vector(next(iter(self.keys))))


from transformers import AutoModelForMaskedLM, AutoTokenizer

class ChemBERTaEmbedding(BaseEmbedding):
    """
    ChemBERTa embedding implementation of the BaseEmbedding class.
    Provides molecular embeddings using the ChemBERTa transformer model.
    """
    def __init__(self, 
                 columns: str = 'SMILES',
                 name: str = "chemberta",
                 model_name: str = "DeepChem/ChemBERTa-77M-MTR",
                 embedding_type: str = 'mean_pooling',
                 padding: bool = False):
        """
        Initialize ChemBERTaEmbedding.

        Args:
            columns: Column name containing SMILES strings
            name: Identifier for this embedding
            model_name: Name of the pretrained ChemBERTa model
            embedding_type: Type of embedding ('cls' or 'mean_pooling')
            padding: Whether to use padding in tokenization
        """
        super().__init__(name=name)
        self.columns = columns
        self.model_name = model_name
        self.embedding_type = embedding_type
        self.padding = padding
        
        # Initialize model and tokenizer
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.eval()
        
        self._embedding_size = None
        self.feature_names = None
        self._is_fitted = False

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select relevant columns for encoding.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with selected SMILES column
        """
        if self.columns not in df.columns:
            raise ValueError(f"Column {self.columns} not found in DataFrame")
        return df[[self.columns]]

    def get_embedding(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Get ChemBERTa embeddings for SMILES strings.

        Args:
            df: Input DataFrame
            fit: Whether to store embeddings in dictionary (ignored as model is pre-trained)

        Returns:
            Array of molecular embeddings
        """
        smiles_list = df[self.columns].tolist()
        embeddings_cls, embeddings_mean = self._featurize_batch(smiles_list)

        if self.embedding_type == 'cls':
            embeddings = embeddings_cls
            self.feature_names = [f'ChemBERTa_cls_{i}' for i in range(embeddings.shape[1])]
        elif self.embedding_type == 'mean_pooling':
            embeddings = embeddings_mean
            self.feature_names = [f'ChemBERTa_mean_{i}' for i in range(embeddings.shape[1])]
        else:
            raise ValueError("embedding_type must be 'cls' or 'mean_pooling'")

        self._embedding_size = embeddings.shape[1]
        self._is_fitted = True

        # Store embeddings for each SMILES string
        for idx, smiles in enumerate(smiles_list):
            self.embedding_dict[smiles] = embeddings[idx]

        return embeddings

    def _featurize_batch(self, smiles_list: List[str]) -> tuple:
        """
        Generate embeddings for a batch of SMILES strings.

        Args:
            smiles_list: List of SMILES strings to encode

        Returns:
            Tuple of (CLS embeddings, mean pooled embeddings)
        """
        embeddings_cls = []
        embeddings_mean = []

        with torch.no_grad():
            for smiles in smiles_list:
                encoded_input = self.tokenizer(
                    smiles,
                    return_tensors="pt",
                    padding=self.padding,
                    truncation=True
                )
                model_output = self.model(**encoded_input)

                # Get CLS token embedding
                embedding_cls = model_output[0][:, 0, :]
                embeddings_cls.append(embedding_cls)

                # Get mean pooled embedding
                embedding_mean = torch.mean(model_output[0], 1)
                embeddings_mean.append(embedding_mean)

        # Stack the tensors and convert to numpy
        embeddings_cls = torch.cat(embeddings_cls).numpy()
        embeddings_mean = torch.cat(embeddings_mean).numpy()
        return embeddings_cls, embeddings_mean

    @property
    def embedding_size(self) -> int:
        """
        Get the size of the embedding vector.

        Returns:
            Size of the embedding vector
        """
        if self._embedding_size is None:
            # This will be set correctly during first call to get_embedding
            return self.model.config.hidden_size
        return self._embedding_size

    def get_feature_names(self) -> List[str]:
        """
        Get names of the embedding features.

        Returns:
            List of feature names
        """
        if self.feature_names is None:
            raise ValueError("Feature names not available. Call get_embedding first.")
        return self.feature_names


from transformers import AutoModel, AutoTokenizer

class MolformerEmbedding(BaseEmbedding):
    """
    Molformer embedding implementation of the BaseEmbedding class.
    Provides molecular embeddings using the MolFormer transformer model.
    """
    def __init__(self, 
                 columns: str = 'SMILES',
                 name: str = "molformer",
                 model_path: str = 'ibm/MoLFormer-XL-both-10pct',
                 padding: bool = True):
        """
        Initialize MolformerEmbedding.

        Args:
            columns: Column name containing SMILES strings
            name: Identifier for this embedding
            model_path: Path to the pretrained MolFormer model
            padding: Whether to use padding in tokenization (should be True for this model)
        """
        super().__init__(name=name)
        self.columns = columns
        self.model_path = model_path
        self.padding = padding

        # Initialize model and tokenizer
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()

        self._embedding_size = self.model.config.hidden_size  # Set from model config
        self.feature_names = None
        self._is_fitted = False

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select relevant columns for encoding.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with selected SMILES column
        """
        if self.columns not in df.columns:
            raise ValueError(f"Column {self.columns} not found in DataFrame")
        return df[[self.columns]]

    def get_embedding(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Get MolFormer embeddings for SMILES strings.

        Args:
            df: Input DataFrame
            fit: Whether to store embeddings in dictionary (ignored as model is pre-trained)

        Returns:
            Array of molecular embeddings
        """
        smiles_list = df[self.columns].tolist()
        embeddings = self._featurize_batch(smiles_list)

        # Generate feature names if not already created
        if self.feature_names is None:
            self.feature_names = [f'Molformer_{i}' for i in range(embeddings.shape[1])]

        self._is_fitted = True

        # Store embeddings for each SMILES string
        for idx, smiles in enumerate(smiles_list):
            self.embedding_dict[smiles] = embeddings[idx]

        return embeddings

    def _featurize_batch(self, smiles_list: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of SMILES strings.

        Args:
            smiles_list: List of SMILES strings to encode

        Returns:
            Array of molecular embeddings
        """
        inputs = self.tokenizer(
            smiles_list,
            padding=self.padding,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Use pooler_output for the sentence-level representation
        embeddings = outputs.pooler_output.cpu().numpy()
        return embeddings

    @property
    def embedding_size(self) -> int:
        """
        Get the size of the embedding vector.

        Returns:
            Size of the embedding vector
        """
        return self._embedding_size

    def get_feature_names(self) -> List[str]:
        """
        Get names of the embedding features.

        Returns:
            List of feature names
        """
        if self.feature_names is None:
            raise ValueError("Feature names not available. Call get_embedding first.")
        return self.feature_names


from transformers import AutoModel, AutoTokenizer

class SmoleBartEmbedding(BaseEmbedding):
    """
    SmoleBART embedding implementation of the BaseEmbedding class.
    Provides molecular embeddings using the SmoleBART transformer model.
    """
    def __init__(self, 
                 columns: str = 'SMILES',
                 name: str = "smolebart",
                 model_path: str = "UdS-LSV/smole-bart",
                 embedder: Literal["encoder", "decoder"] = "encoder",
                 padding: bool = True):
        """
        Initialize SmoleBartEmbedding.

        Args:
            columns: Column name containing SMILES strings
            name: Identifier for this embedding
            model_path: Path to the pretrained SmoleBART model
            embedder: Which part of the model to use for embeddings ("encoder" or "decoder")
            padding: Whether to use padding in tokenization
        """
        super().__init__(name=name)
        self.columns = columns
        self.model_path = model_path
        self.embedder = embedder
        self.padding = padding

        # Initialize model and tokenizer
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()

        self._embedding_size = self.model.config.hidden_size  # Set from model config
        self.feature_names = None
        self._is_fitted = False

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select relevant columns for encoding.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with selected SMILES column
        """
        if self.columns not in df.columns:
            raise ValueError(f"Column {self.columns} not found in DataFrame")
        return df[[self.columns]]

    def get_embedding(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Get SmoleBART embeddings for SMILES strings.

        Args:
            df: Input DataFrame
            fit: Whether to store embeddings in dictionary (ignored as model is pre-trained)

        Returns:
            Array of molecular embeddings
        """
        smiles_list = df[self.columns].tolist()
        embeddings = self._featurize_batch(smiles_list)

        # Generate feature names if not already created
        if self.feature_names is None:
            self.feature_names = [f'SmoleBart_{i}' for i in range(embeddings.shape[1])]

        self._is_fitted = True

        # Store embeddings for each SMILES string
        for idx, smiles in enumerate(smiles_list):
            self.embedding_dict[smiles] = embeddings[idx]

        return embeddings

    def _featurize_batch(self, smiles_list: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of SMILES strings.

        Args:
            smiles_list: List of SMILES strings to encode

        Returns:
            Array of molecular embeddings

        Raises:
            ValueError: If embedder type is invalid
        """
        inputs = self.tokenizer(
            smiles_list,
            padding=self.padding,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Select appropriate output based on embedder type
        if self.embedder == "encoder":
            hidden_states = outputs.encoder_last_hidden_state
        elif self.embedder == "decoder":
            hidden_states = outputs.last_hidden_state
        else:
            raise ValueError("embedder must be either 'encoder' or 'decoder'")

        # Apply mean pooling and convert to numpy
        embeddings = hidden_states.mean(dim=1).cpu().numpy()
        return embeddings

    @property
    def embedding_size(self) -> int:
        """
        Get the size of the embedding vector.

        Returns:
            Size of the embedding vector

        Raises:
            ValueError: If embedding size is not yet set
        """
        if self._embedding_size is None:
            raise ValueError("Embedding size not set. Call get_embedding first.")
        return self._embedding_size

    def get_feature_names(self) -> List[str]:
        """
        Get names of the embedding features.

        Returns:
            List of feature names

        Raises:
            ValueError: If feature names are not yet available
        """
        if self.feature_names is None:
            raise ValueError("Feature names not available. Call get_embedding first.")
        return self.feature_names


from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from pandarallel import pandarallel

class SelfFormerEmbedding(BaseEmbedding):
    """
    SELFormer embedding implementation of the BaseEmbedding class.
    Provides molecular embeddings using the SELFormer transformer model with batch processing.
    """
    def __init__(self,
                 columns: str = 'SELFIES',
                 name: str = "selformer",
                 model_name: str = None,
                 tokenizer_path: Optional[str] = None,
                 padding: bool = True,
                 batch_size: int = 32,
                 nb_workers: int = 5,
                 device: Optional[str] = None):
        """
        Initialize SelfFormerEmbedding.

        Args:
            columns: Column name containing SELFIES strings
            name: Identifier for this embedding
            model_name: Path or identifier for the pretrained SELFormer model
            tokenizer_path: Optional path to load tokenizer from
            padding: Whether to use padding in tokenization
            batch_size: Number of samples to process in a single batch
            nb_workers: Number of parallel workers for pandarallel
            device: Device to run the model on ("cuda" or "cpu")
        """
        if model_name is None:
            raise ValueError("model_name must be provided")
            
        super().__init__(name=name)
        
        # Disable parallelism warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["WANDB_DISABLED"] = "true"
        
        self.columns = columns
        self.model_name = model_name
        self.padding = padding
        self.batch_size = batch_size
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model with hidden states enabled
        config = RobertaConfig.from_pretrained(model_name)
        config.output_hidden_states = True
        self.model = RobertaModel.from_pretrained(model_name, config=config).to(self.device)
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path or model_name)
        
        # Initialize parallel processing
        pandarallel.initialize(nb_workers=nb_workers, progress_bar=True)
        
        self._embedding_size = self.model.config.hidden_size
        self.feature_names = None
        self._is_fitted = False

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select relevant columns for encoding.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with selected SELFIES column
        """
        if self.columns not in df.columns:
            raise ValueError(f"Column {self.columns} not found in DataFrame")
        return df[[self.columns]]

    def get_embedding(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Get SELFormer embeddings for SELFIES strings using batch processing.

        Args:
            df: Input DataFrame
            fit: Whether to store embeddings in dictionary (ignored as model is pre-trained)

        Returns:
            Array of molecular embeddings
        """
        selfies_list = df[self.columns].tolist()
        
        try:
            embeddings = self._featurize_batch(selfies_list)
            
            # Verify embedding dimensions
            if embeddings.shape[1] != self._embedding_size:
                raise ValueError(f"Embedding dimension mismatch. Expected {self._embedding_size}, "
                               f"got {embeddings.shape[1]}")

            # Generate feature names if not already created
            if self.feature_names is None:
                self.feature_names = [f'SELFormer_{i}' for i in range(embeddings.shape[1])]

            self._is_fitted = True

            # Store embeddings for each SELFIES string
            for idx, selfie in enumerate(selfies_list):
                self.embedding_dict[selfie] = embeddings[idx]

            return embeddings
            
        except Exception as e:
            print(f"Error during embedding generation: {str(e)}")
            print(f"DataFrame shape: {df.shape}")
            print(f"Expected embedding size: {self._embedding_size}")
            raise

    def _featurize_batch(self, selfies_list: List[str]) -> np.ndarray:
        """
        Generate embeddings for batches of SELFIES strings with explicit shape handling.

        Args:
            selfies_list: List of SELFIES strings to encode

        Returns:
            Array of molecular embeddings
        """
        all_embeddings = []

        for i in range(0, len(selfies_list), self.batch_size):
            batch = selfies_list[i:i + self.batch_size]
            
            try:
                # Tokenize batch with explicit padding
                encoded_input = self.tokenizer(
                    batch,
                    add_special_tokens=True,
                    max_length=512,
                    padding='max_length',  # Force padding to max_length
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

                # Generate embeddings
                with torch.no_grad():
                    output = self.model(**encoded_input)
                    
                # Get the last hidden state
                last_hidden_state = output.last_hidden_state  # Shape: [batch_size, seq_len, hidden_size]
                
                # Create attention mask for proper mean pooling
                attention_mask = encoded_input['attention_mask']  # Shape: [batch_size, seq_len]
                
                # Expand attention mask to 3D
                attention_mask_expanded = attention_mask.unsqueeze(-1)  # Shape: [batch_size, seq_len, 1]
                
                # Apply attention mask before mean pooling
                masked_hidden_states = last_hidden_state * attention_mask_expanded
                
                # Sum tokens and divide by actual sequence length
                sum_hidden_states = torch.sum(masked_hidden_states, dim=1)  # Shape: [batch_size, hidden_size]
                sequence_lengths = torch.sum(attention_mask, dim=1, keepdim=True)  # Shape: [batch_size, 1]
                
                # Compute mean while handling padding
                embeddings = sum_hidden_states / sequence_lengths
                
                all_embeddings.append(embeddings.cpu())

            except Exception as e:
                print(f"Error processing batch {i}-{i+self.batch_size}: {str(e)}")
                print(f"Batch shapes - Input: {encoded_input['input_ids'].shape}, "
                      f"Attention mask: {encoded_input['attention_mask'].shape}, "
                      f"Hidden states: {last_hidden_state.shape}")
                raise

        if not all_embeddings:
            raise ValueError("No embeddings were generated successfully")

        # Concatenate all batches and convert to numpy
        final_embeddings = torch.cat(all_embeddings, dim=0).numpy()
        
        # Verify final shape
        expected_size = (len(selfies_list), self._embedding_size)
        if final_embeddings.shape != expected_size:
            raise ValueError(f"Unexpected embedding shape. Expected {expected_size}, got {final_embeddings.shape}")
            
        return final_embeddings


    @property
    def embedding_size(self) -> int:
        """
        Get the size of the embedding vector.

        Returns:
            Size of the embedding vector
        """
        return self._embedding_size

    def get_feature_names(self) -> List[str]:
        """
        Get names of the embedding features.

        Returns:
            List of feature names

        Raises:
            ValueError: If feature names are not yet available
        """
        if self.feature_names is None:
            raise ValueError("Feature names not available. Call get_embedding first.")
        return self.feature_names


class MultiEmbedding(BaseEmbedding):
    """
    A class that combines multiple embedding types into a single embedding.
    
    This allows for flexible experimentation with different embedding combinations
    while maintaining a consistent interface.
    """
    
    def __init__(
        self,
        embeddings: List[BaseEmbedding],
        name: str = "multi_embedding",
        combination_method: Literal["concat", "sum", "average", "weighted"] = "concat",
        weights: Optional[List[float]] = None
    ):
        """
        Initialize the multi-embedding class.
        
        Args:
            embeddings (List[BaseEmbedding]): List of embedding objects to combine
            name (str): Identifier for this embedding type
            combination_method (str): Method to combine embeddings ("concat", "sum", "average", or "weighted")
            weights (Optional[List[float]]): Weights for weighted combination method
        """
        super().__init__(name=name)
        
        if not embeddings:
            raise ValueError("At least one embedding must be provided")
        
        self.embeddings = embeddings
        self.combination_method = combination_method
        
        # Handle weights for weighted combination
        if combination_method == "weighted":
            if weights is None:
                self.weights = [1.0 / len(embeddings)] * len(embeddings)
            elif len(weights) != len(embeddings):
                raise ValueError(f"Number of weights ({len(weights)}) must match number of embeddings ({len(embeddings)})")
            else:
                # Normalize weights to sum to 1
                total = sum(weights)
                self.weights = [w / total for w in weights]
        else:
            self.weights = None
        
        # Check if embeddings are fitted
        for i, emb in enumerate(embeddings):
            if not emb._is_fitted:
                print(f"Warning: Embedding {i} ({emb.__class__.__name__}) is not fitted yet")
        
        # Update metadata
        self.metadata = {
            'num_embeddings': len(embeddings),
            'embedding_types': [emb.__class__.__name__ for emb in embeddings],
            # 'embedding_sizes': [emb.embedding_size for emb in embeddings],
            'combination_method': combination_method,
            'weights': self.weights if self.weights else None
        }
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the input DataFrame using all embedding models.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        # Preprocess with all embeddings
        result_df = df.copy()
        for emb in self.embeddings:
            result_df = emb.preprocess(result_df.copy())
        return result_df
    
    def get_embedding(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Get combined embeddings from all embedding models.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            fit (bool): Whether to fit the embedding models if not already fitted
            
        Returns:
            np.ndarray: Combined embeddings
        """
        # Get embeddings from all models
        all_embeddings = [emb.get_embedding(df, fit=fit) for emb in self.embeddings]
        
        # Combine embeddings based on the specified method
        if self.combination_method == "concat":
            return np.concatenate(all_embeddings, axis=1)
        
        elif self.combination_method in ["sum", "average", "weighted"]:
            # Check if all embeddings have the same shape
            if not all(emb.shape[1] == all_embeddings[0].shape[1] for emb in all_embeddings):
                raise ValueError(f"Cannot {self.combination_method} embeddings of different dimensions")
            
            if self.combination_method == "sum":
                return sum(all_embeddings)
            
            elif self.combination_method == "average":
                return sum(all_embeddings) / len(all_embeddings)
            
            elif self.combination_method == "weighted":
                weighted_sum = sum(w * emb for w, emb in zip(self.weights, all_embeddings))
                return weighted_sum
        
        else:
            raise ValueError(f"Unsupported combination method: {self.combination_method}")
    
    def compute_and_store_embeddings(
        self,
        df: pd.DataFrame,
        entity_column: str,
        batch_size: Optional[int] = None
    ) -> None:
        """
        Compute and store combined embeddings.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            entity_column (str): Column name containing entities
            batch_size (Optional[int]): Batch size for processing
        """
        # First ensure all underlying embeddings are computed
        for emb in self.embeddings:
            if not emb._is_fitted:
                emb.compute_and_store_embeddings(df, entity_column, batch_size)
        
        # Then compute combined embeddings
        super().compute_and_store_embeddings(df, entity_column, batch_size)
    
    def get_entity_embedding(
        self,
        entity_name: str,
        allow_missing: bool = True
    ) -> np.ndarray:
        """
        Get combined embedding for a specific entity.
        
        Args:
            entity_name (str): Name of the entity
            allow_missing (bool): If True, compute embedding for missing entities
            
        Returns:
            np.ndarray: Combined embedding vector
        """
        if entity_name in self.embedding_dict:
            return self.embedding_dict[entity_name]
        
        # Get embeddings from all models
        all_embeddings = [
            emb.get_entity_embedding(entity_name, allow_missing) 
            for emb in self.embeddings
        ]
        
        # Combine embeddings based on the specified method
        if self.combination_method == "concat":
            combined = np.concatenate(all_embeddings)
        
        elif self.combination_method in ["sum", "average", "weighted"]:
            # Check if all embeddings have the same shape
            if not all(emb.shape == all_embeddings[0].shape for emb in all_embeddings):
                raise ValueError(f"Cannot {self.combination_method} embeddings of different dimensions")
            
            if self.combination_method == "sum":
                combined = sum(all_embeddings)
            
            elif self.combination_method == "average":
                combined = sum(all_embeddings) / len(all_embeddings)
            
            elif self.combination_method == "weighted":
                combined = sum(w * emb for w, emb in zip(self.weights, all_embeddings))
        
        else:
            raise ValueError(f"Unsupported combination method: {self.combination_method}")
        
        # Store for future use
        self.embedding_dict[entity_name] = combined
        return combined
    
    @property
    def embedding_size(self) -> int:
        """
        Returns the size of the combined embedding vector.
        
        Returns:
            int: Size of the combined embedding vector
            
        Raises:
            ValueError: If embedding sizes are incompatible with the combination method
        """
        if self.combination_method == "concat":
            return sum(emb.embedding_size for emb in self.embeddings)
        
        elif self.combination_method in ["sum", "average", "weighted"]:
            # Check if all embeddings have the same size
            sizes = [emb.embedding_size for emb in self.embeddings]
            if not all(size == sizes[0] for size in sizes):
                raise ValueError(
                    f"Cannot {self.combination_method} embeddings of different dimensions: {sizes}"
                )
            return sizes[0]
        
        else:
            raise ValueError(f"Unsupported combination method: {self.combination_method}")
    
    def save_embeddings(
        self,
        filepath: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Save embeddings and metadata to disk.
        
        Args:
            filepath (str): Path to save the embeddings
            metadata (Optional[Dict]): Additional metadata to save
        """
        combined_metadata = self.metadata.copy()
        if metadata:
            combined_metadata.update(metadata)
        super().save_embeddings(filepath, combined_metadata)


def select_top_variable_genes(df, k=256, exclude_controls=True, controls=["Belinostat", "Dabrafenib"]):
    """
    Selects the top k most variable genes based on standard deviation.

    Args:
        df (pd.DataFrame): Gene expression dataframe with multi-index ("cell_type", "sm_name").
        k (int): Number of top variable genes to select.
        exclude_controls (bool): Whether to exclude control drugs.
        controls (list): List of control drugs to exclude.

    Returns:
        List of top k gene names.
    """
    if exclude_controls:
        df = df.loc[~df.index.get_level_values("sm_name").isin(controls)]
    
    # Compute standard deviation per gene
    gene_variability = df.iloc[:, 3:].std(axis=0)  # Avoid non-numeric columns
    
    # Select top k most variable genes
    top_genes = gene_variability.sort_values(ascending=False).head(k).index.tolist()
    
    return top_genes


from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, List, Tuple, Any, Callable, Optional, Union, Type
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import time
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator
from sklearn.metrics import explained_variance_score

def RMSE_rowwise_loss(y_pred, y_true):
    return torch.sqrt(torch.mean((y_pred - y_true)**2, dim=1)).mean()

def RMSE_rowwise_loss_numpy(y_pred, y_true):
    """Calculate row-wise RMSE loss using numpy arrays"""
    # Ensure inputs are numpy arrays
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    
    # Convert to numpy arrays if they're not already
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    # Calculate RMSE
    row_mse = np.mean((y_pred - y_true)**2, axis=1)
    row_rmse = np.sqrt(row_mse)
    return float(np.mean(row_rmse))

def evaluate_evs_score(y_true, y_pred):
    return explained_variance_score(y_true, y_pred, multioutput='variance_weighted')

@dataclass
class BenchmarkResult:
    """Store results for a single model with multiple runs."""
    model_name: str
    embedding_name: str
    mse_mean: float
    mse_std: float
    mrrmse_mean: float
    mrrmse_std: float
    r2_mean: float
    r2_std: float
    evs_mean: float
    evs_std: float
    training_time: float
    inference_time: float
    embedding_size: int
    memory_usage: float


class ModelFactory(ABC):
    """Abstract factory for creating models with specific input sizes."""
    
    @abstractmethod
    def create_model(self, input_size: int) -> Any:
        """Create a new model instance with the specified input size."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model name."""
        pass

class BaseModel(ABC):
    """Abstract base class for all models (both PyTorch and sklearn)."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model name."""
        pass

class PyTorchModelFactory(ModelFactory):
    """Factory for creating PyTorch models with specific architectures."""
    
    def __init__(
        self,
        model_class: Type[nn.Module],
        model_params: Dict[str, Any],
        criterion: Callable,
        optimizer_class: Type[torch.optim.Optimizer],
        optimizer_params: Dict[str, Any],
        batch_size: int = 32,
        num_epochs: int = 50,
        early_stopping: bool = True,
        early_stopping_patience: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        name: str = "PyTorch Model"
    ):
        self.model_class = model_class
        self.model_params = model_params
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self._name = name
    
    def create_model(self, input_size: int) -> 'PyTorchModel':
        """Create a new PyTorch model with the specified input size."""
        model_params = {**self.model_params, 'input_size': input_size}
        model = self.model_class(**model_params)
        
        return PyTorchModel(
            model=model,
            criterion=self.criterion,
            optimizer_class=self.optimizer_class,
            optimizer_params=self.optimizer_params,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            early_stopping = self.early_stopping,
            early_stopping_patience=self.early_stopping_patience,
            device=self.device,
            name=self._name
        )
    
    @property
    def name(self) -> str:
        return self._name

class SklearnModelFactory(ModelFactory):
    """Factory for creating sklearn models."""
    
    def __init__(
        self,
        model_class: Type[BaseEstimator],
        model_params: Dict[str, Any],
        name: str = "Sklearn Model"
    ):
        self.model_class = model_class
        self.model_params = model_params
        self._name = name
    
    def create_model(self, input_size: int) -> 'SklearnModel':
        """Create a new sklearn model (input_size is ignored for most sklearn models)."""
        model = self.model_class(**self.model_params)
        return SklearnModel(model=model, name=self._name)
    
    @property
    def name(self) -> str:
        return self._name

class PyTorchModel(BaseModel):
    """Wrapper for PyTorch models to conform to the BaseModel interface."""
    
    def __init__(
        self,
        model: nn.Module,
        criterion: Callable,
        optimizer_class: torch.optim.Optimizer,
        optimizer_params: Dict[str, Any],
        batch_size: int = 32,
        num_epochs: int = 50,
        early_stopping: bool = True,
        early_stopping_patience: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        name: str = "PyTorch Model"
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self._name = name
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Create data loaders
        train_data = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
        train_loader = DataLoader(
            train_data, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        optimizer = self.optimizer_class(
            self.model.parameters(),
            **self.optimizer_params
        )
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if self.early_stopping and patience_counter >= self.early_stopping_patience:
                    break
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.model(X_tensor).cpu().numpy()
    
    @property
    def name(self) -> str:
        return self._name

class SklearnModel(BaseModel):
    """Wrapper for sklearn models to conform to the BaseModel interface."""
    
    def __init__(self, model: BaseEstimator, name: str = "Sklearn Model"):
        self.model = model
        self._name = name
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    @property
    def name(self) -> str:
        return self._name


from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

class LinearRegressionModel(SklearnModel):
    """Linear Regression wrapper."""
    def __init__(self):
        super().__init__(
            model=LinearRegression(),
            name="Linear Regression"
        )

class DecisionTreeModel(SklearnModel):
    """Decision Tree wrapper."""
    def __init__(
        self,
        max_depth: int = 7,
        min_samples_split: int = 10
    ):
        super().__init__(
            model=DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            ),
            name="Decision Tree"
        )

class KNeighborsModel(SklearnModel):
    """K-Nearest Neighbors wrapper."""
    def __init__(
        self,
        n_neighbors: int = 7,
        weights: str = 'uniform'
    ):
        super().__init__(
            model=KNeighborsRegressor(
                n_neighbors=n_neighbors,
                weights=weights
            ),
            name="KNN"
        )

class MLPRegressorModel(SklearnModel):
    """Neural Network wrapper."""
    def __init__(
        self,
        hidden_layer_sizes: tuple = (64, 32),
        alpha: float = 0.01,
        max_iter: int = 1000
    ):
        super().__init__(
            model=MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                alpha=alpha,
                max_iter=max_iter,
                random_state=42
            ),
            name="MLP"
        )

# Model Factories
class LinearRegressionFactory(ModelFactory):
    """Factory for Linear Regression."""
    def __init__(self):
        self._name = "Linear Regression"
    
    def create_model(self, input_size: int) -> LinearRegressionModel:
        return LinearRegressionModel()
    
    @property
    def name(self) -> str:
        return self._name

class DecisionTreeFactory(ModelFactory):
    """Factory for Decision Tree."""
    def __init__(
        self,
        max_depth: int = 7,
        min_samples_split: int = 10
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self._name = "Decision Tree"
    
    def create_model(self, input_size: int) -> DecisionTreeModel:
        return DecisionTreeModel(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split
        )
    
    @property
    def name(self) -> str:
        return self._name

class KNeighborsFactory(ModelFactory):
    """Factory for KNN."""
    def __init__(
        self,
        n_neighbors: int = 7,
        weights: str = 'uniform'
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._name = "KNN"
    
    def create_model(self, input_size: int) -> KNeighborsModel:
        return KNeighborsModel(
            n_neighbors=self.n_neighbors,
            weights=self.weights
        )
    
    @property
    def name(self) -> str:
        return self._name

class MLPRegressorFactory(ModelFactory):
    """Factory for MLP."""
    def __init__(
        self,
        hidden_layer_sizes: tuple = (64, 32),
        alpha: float = 0.01,
        max_iter: int = 1000
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self._name = "MLP"
    
    def create_model(self, input_size: int) -> MLPRegressorModel:
        return MLPRegressorModel(
            hidden_layer_sizes=self.hidden_layer_sizes,
            alpha=self.alpha,
            max_iter=self.max_iter
        )
    
    @property
    def name(self) -> str:
        return self._name


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import TruncatedSVD

class TSVDBaseModel(BaseModel):
    """Base class for sklearn models with TSVD dimensionality reduction."""
    
    def __init__(
        self,
        model: BaseEstimator,
        n_components: int = 50,
        name: str = "TSVD Base Model"
    ):
        self.model = model
        self.n_components = n_components
        self._name = name
        self.tsvd = TruncatedSVD(n_components=n_components)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit both TSVD and the underlying model."""
        # Apply TSVD to target variables
        y_reduced = self.tsvd.fit_transform(y)
        # Fit the model with reduced targets
        self.model.fit(X, y_reduced)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions and inverse transform them back to original space."""
        # Get predictions in reduced space
        y_pred_reduced = self.model.predict(X)
        # Transform back to original space
        return self.tsvd.inverse_transform(y_pred_reduced)
    
    @property
    def name(self) -> str:
        return self._name

class LinearRegressionTSVD(TSVDBaseModel):
    """Linear Regression with TSVD."""
    def __init__(self, n_components: int = 50):
        super().__init__(
            model=LinearRegression(),
            n_components=n_components,
            name="Linear Regression TSVD"
        )

class DecisionTreeTSVD(TSVDBaseModel):
    """Decision Tree with TSVD."""
    def __init__(
        self,
        n_components: int = 50,
        max_depth: int = 7,
        min_samples_split: int = 10
    ):
        super().__init__(
            model=DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            ),
            n_components=n_components,
            name="Decision Tree TSVD"
        )

class KNeighborsTSVD(TSVDBaseModel):
    """K-Nearest Neighbors with TSVD."""
    def __init__(
        self,
        n_components: int = 50,
        n_neighbors: int = 7,
        weights: str = 'uniform'
    ):
        super().__init__(
            model=KNeighborsRegressor(
                n_neighbors=n_neighbors,
                weights=weights
            ),
            n_components=n_components,
            name="KNN TSVD"
        )

class MLPRegressorTSVD(TSVDBaseModel):
    """Neural Network with TSVD."""
    def __init__(
        self,
        n_components: int = 50,
        hidden_layer_sizes: tuple = (64, 32),
        alpha: float = 0.01,
        max_iter: int = 1000
    ):
        super().__init__(
            model=MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                alpha=alpha,
                max_iter=max_iter,
                random_state=42
            ),
            n_components=n_components,
            name="MLP TSVD"
        )

# Model Factories
class LinearRegressionTSVDFactory(ModelFactory):
    """Factory for Linear Regression with TSVD."""
    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self._name = "Linear Regression TSVD"
    
    def create_model(self, input_size: int) -> LinearRegressionTSVD:
        return LinearRegressionTSVD(n_components=self.n_components)
    
    @property
    def name(self) -> str:
        return self._name

class DecisionTreeTSVDFactory(ModelFactory):
    """Factory for Decision Tree with TSVD."""
    def __init__(
        self,
        n_components: int = 50,
        max_depth: int = 7,
        min_samples_split: int = 10
    ):
        self.n_components = n_components
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self._name = "Decision Tree TSVD"
    
    def create_model(self, input_size: int) -> DecisionTreeTSVD:
        return DecisionTreeTSVD(
            n_components=self.n_components,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split
        )
    
    @property
    def name(self) -> str:
        return self._name

class KNeighborsTSVDFactory(ModelFactory):
    """Factory for KNN with TSVD."""
    def __init__(
        self,
        n_components: int = 50,
        n_neighbors: int = 7,
        weights: str = 'uniform'
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._name = "KNN TSVD"
    
    def create_model(self, input_size: int) -> KNeighborsTSVD:
        return KNeighborsTSVD(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            weights=self.weights
        )
    
    @property
    def name(self) -> str:
        return self._name

class MLPRegressorTSVDFactory(ModelFactory):
    """Factory for MLP with TSVD."""
    def __init__(
        self,
        n_components: int = 50,
        hidden_layer_sizes: tuple = (64, 32),
        alpha: float = 0.01,
        max_iter: int = 1000
    ):
        self.n_components = n_components
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self._name = "MLP TSVD"
    
    def create_model(self, input_size: int) -> MLPRegressorTSVD:
        return MLPRegressorTSVD(
            n_components=self.n_components,
            hidden_layer_sizes=self.hidden_layer_sizes,
            alpha=self.alpha,
            max_iter=self.max_iter
        )
    
    @property
    def name(self) -> str:
        return self._name


class LSTMPredictionModel(nn.Module):
    """LSTM-based prediction model for gene expression."""
    def __init__(self, input_size: int, hidden_size: int = 128, output_size: int = 18211):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.head = nn.Linear(512, output_size)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.linear(out)
        out = self.head(out)
        return out


class EmbeddingBenchmark:
    def __init__(
        self,
        embedding_methods: Dict[str, BaseEmbedding],
        model_factories: List[ModelFactory],
        n_runs: int = 5,
        only_significant = False
    ):
        self.embedding_methods = embedding_methods
        self.model_factories = model_factories
        self.n_runs = n_runs
        self.results: List[BenchmarkResult] = []
        self.gene_cols = self.get_significant_gene_columns(df) if only_significant else self.get_gene_columns(df) 
        self.combined_embedding_size = 0
        self.embedding_ranges = {}
        self.feature_map = {}
        self.only_significant = only_significant

       
        
    def prepare_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Prepare data splits for benchmarking.
        Ensures proper index alignment for all embedding methods.
        """
        df['SELFIES'] = df['SMILES'].apply(smiles_to_selfies)
        
        # Split data
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        
        # Reset indices to ensure proper alignment
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
        # Extract target values
        y_train = train_df[self.gene_cols].values
        y_test = test_df[self.gene_cols].values
        
        return train_df, test_df, y_train, y_test
        
    def get_gene_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of gene columns by excluding known non-gene columns."""
        return [col for col in df.columns if col not in NON_GENE_COLUMNS]

    def get_significant_gene_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of gene columns by excluding non-significant gene columns and non-gene columns."""
        # return [col for col in df.columns if col not in NON_GENE_COLUMNS]
        target_cols = ['sm_lincs_id','SMILES','control',]
        targets = df.copy()
        targets.drop(columns=target_cols, inplace=True)
        targets.set_index(["cell_type", "sm_name"], inplace=True)
        # Select top 256 variable genes
        top_genes = select_top_variable_genes(targets, k=256, exclude_controls=True)
        return top_genes
    
    def evaluate(
        self,
        model_factory: ModelFactory,
        embedding_name: str,
        embedding_method: BaseEmbedding,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> BenchmarkResult:
        """Evaluate a single model with a specific embedding method."""
        run_results = []
        
        # Compute embeddings once
        start_time = time.time()
        train_embeddings = embedding_method.get_embedding(train_df, fit=True)
        test_embeddings = embedding_method.get_embedding(test_df, fit=False)
        embedding_time = time.time() - start_time
        
        # Create model with correct input size
        model = model_factory.create_model(embedding_method.embedding_size)
        
        for run in range(self.n_runs):
            print(f"Run {run + 1}/{self.n_runs}")
            
            # Train model
            start_time = time.time()
            model.fit(train_embeddings, y_train)
            training_time = time.time() - start_time
            
            # Evaluate
            start_time = time.time()
            y_pred = model.predict(test_embeddings)
            inference_time = time.time() - start_time
            
            # Compute metrics
            mse = mean_squared_error(y_test, y_pred)
            mrrmse = RMSE_rowwise_loss_numpy(y_pred, y_test)
            r2 = r2_score(y_test, y_pred)
            evs = evaluate_evs_score(y_test, y_pred)
            
            run_results.append({
                'mse': mse,
                'mrrmse': mrrmse,
                'r2': r2,
                'evs': evs,
                'training_time': training_time,
                'inference_time': inference_time,
                'memory_usage': (train_embeddings.nbytes + test_embeddings.nbytes) / (1024 * 1024)
            })
        
        # Compute statistics
        return BenchmarkResult(
            model_name=model_factory.name,
            embedding_name=embedding_name,
            mse_mean=np.mean([r['mse'] for r in run_results]),
            mse_std=np.std([r['mse'] for r in run_results]),
            mrrmse_mean=np.mean([r['mrrmse'] for r in run_results]),
            mrrmse_std=np.std([r['mrrmse'] for r in run_results]),
            r2_mean=np.mean([r['r2'] for r in run_results]),
            r2_std=np.std([r['r2'] for r in run_results]),
            evs_mean=np.mean([r['evs'] for r in run_results]),
            evs_std=np.std([r['evs'] for r in run_results]),
            training_time=np.mean([r['training_time'] for r in run_results]),
            inference_time=np.mean([r['inference_time'] for r in run_results]),
            embedding_size=embedding_method.embedding_size,
            memory_usage=run_results[0]['memory_usage']
        )
        
    
    def run_benchmark(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run benchmark for all combinations of embeddings and models."""
        df = df.copy()
        train_df, test_df, y_train, y_test = self.prepare_data(df)
        
        for name, method in self.embedding_methods.items():
            print(f"\nEvaluating {name} embedding...")
            
            for model_factory in self.model_factories:
                print(f"Testing with {model_factory.name}...")
                try:
                    result = self.evaluate(
                        model_factory, name, method,
                        train_df, test_df,
                        y_train, y_test
                    )
                    self.results.append(result)
                except Exception as e:
                    print(f"Error evaluating {name} embedding with {model_factory.name}: {str(e)}")
        
        return pd.DataFrame([vars(r) for r in self.results])

    
    def visualize_results(self) -> None:
        """Create detailed visualizations of benchmark results using clear bar charts."""
        if not self.results:
            raise ValueError("No benchmark results available")
        
        results_df = pd.DataFrame([vars(r) for r in self.results])
        
        # Set style
        plt.style.use('seaborn')
        
        # Colors for different models
        model_colors = sns.color_palette("Set2", n_colors=len(results_df['model_name'].unique()))
        
        # 1. Individual Performance Metrics (separate by metric)
        metrics = [
            ('mrrmse_mean', 'mrrmse_std', 'MRRMSE'),
            ('mse_mean', 'mse_std', 'MSE'),
            ('r2_mean', 'r2_std', 'R²'),
            ('evs_mean', 'evs_std', 'Explained Variance Score (EVS)')
        ]
        
        for metric_mean, metric_std, metric_name in metrics:
            plt.figure(figsize=(15, 6))
            
            # Create positions for bars
            embeddings = results_df['embedding_name'].unique()
            models = results_df['model_name'].unique()
            x = np.arange(len(embeddings))
            width = 0.8 / len(models)
            
            # Plot bars for each model
            for i, model in enumerate(models):
                model_data = results_df[results_df['model_name'] == model]
                plt.bar(x + i*width - width*len(models)/2 + width/2, 
                       model_data[metric_mean],
                       width,
                       label=model,
                       color=model_colors[i],
                       yerr=model_data[metric_std],
                       capsize=3)
            
            plt.xlabel('Embedding Method')
            plt.ylabel(metric_name)
            plt.title(f'{metric_name} by Embedding Method and Model')
            plt.xticks(x, embeddings, rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()
        
        # 2. Computational Metrics
        comp_metrics = [
            ('training_time', 'Training Time (s)'),
            ('inference_time', 'Inference Time (s)'),
            ('memory_usage', 'Memory Usage (MB)')
        ]
        
        for metric, metric_label in comp_metrics:
            plt.figure(figsize=(15, 6))
            
            # Create positions for bars
            x = np.arange(len(embeddings))
            width = 0.8 / len(models)
            
            # Plot bars for each model
            for i, model in enumerate(models):
                model_data = results_df[results_df['model_name'] == model]
                plt.bar(x + i*width - width*len(models)/2 + width/2,
                       model_data[metric],
                       width,
                       label=model,
                       color=model_colors[i])
            
            plt.xlabel('Embedding Method')
            plt.ylabel(metric_label)
            plt.title(f'{metric_label} by Embedding Method and Model')
            plt.xticks(x, embeddings, rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()
        
        # 3. Model-centric view
        for metric_mean, metric_std, metric_name in metrics:
            plt.figure(figsize=(15, 6))
            
            # Create positions for bars
            x = np.arange(len(models))
            width = 0.8 / len(embeddings)
            embedding_colors = sns.color_palette("husl", n_colors=len(embeddings))
            
            # Plot bars for each embedding
            for i, embedding in enumerate(embeddings):
                embedding_data = results_df[results_df['embedding_name'] == embedding]
                plt.bar(x + i*width - width*len(embeddings)/2 + width/2,
                       embedding_data[metric_mean],
                       width,
                       label=embedding,
                       color=embedding_colors[i],
                       yerr=embedding_data[metric_std],
                       capsize=3)
            
            plt.xlabel('Model Type')
            plt.ylabel(metric_name)
            plt.title(f'{metric_name} by Model Type and Embedding Method')
            plt.xticks(x, models, rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()
    
    def generate_report(self) -> str:
        """Generate a comprehensive benchmark report with detailed cross-analysis."""
        if not self.results:
            return "No benchmark results available"
            
        results_df = pd.DataFrame([vars(r) for r in self.results])
        
        report = []
        report.append("Comprehensive Benchmark Analysis Report")
        report.append("=" * 50 + "\n")
        
        # 1. Overall Best Performers
        report.append("Overall Best Performers")
        report.append("-" * 30)
        
        metrics = {
            'MRRMSE': ('mrrmse_mean', 'mrrmse_std', 'min'),
            'MSE': ('mse_mean', 'mse_std', 'min'),
            'R²': ('r2_mean', 'r2_std', 'max'),
            'Explained Variance Score (EVS)': ('evs_mean', 'evs_std', 'max'),
        }
        
        for metric_name, (mean_col, std_col, best_func) in metrics.items():
            report.append(f"\nBest {metric_name} Performance:")
            if best_func == 'min':
                idx = results_df[mean_col].idxmin()
            else:
                idx = results_df[mean_col].idxmax()
                
            row = results_df.loc[idx]
            report.append(f"Model: {row['model_name']}")
            report.append(f"Embedding: {row['embedding_name']}")
            report.append(f"Score: {row[mean_col]:.4f} ± {row[std_col]:.4f}")
        
        # 2. Detailed Cross-Analysis
        report.append("\nDetailed Cross-Analysis")
        report.append("-" * 30)
        
        # For each model, analyze performance with different embeddings
        for model in results_df['model_name'].unique():
            report.append(f"\nModel: {model}")
            model_data = results_df[results_df['model_name'] == model]
            
            # Sort embeddings by MRRMSE performance
            embedding_performance = model_data.sort_values('mrrmse_mean')
            
            report.append("\nEmbedding Performance Ranking:")
            for idx, row in embedding_performance.iterrows():
                report.append(f"\n{row['embedding_name']}:")
                report.append(f"  MRRMSE: {row['mrrmse_mean']:.4f} ± {row['mrrmse_std']:.4f}")
                report.append(f"  MSE: {row['mse_mean']:.4f} ± {row['mse_std']:.4f}")
                report.append(f"  R²: {row['r2_mean']:.4f} ± {row['r2_std']:.4f}")
                report.append(f"  Training Time: {row['training_time']:.2f}s")
                report.append(f"  Memory Usage: {row['memory_usage']:.2f}MB")
        
        # 3. Embedding Method Analysis
        report.append("\nEmbedding Method Analysis")
        report.append("-" * 30)
        
        for embedding in results_df['embedding_name'].unique():
            report.append(f"\nEmbedding: {embedding}")
            embedding_data = results_df[results_df['embedding_name'] == embedding]
            
            # Average performance across models
            report.append("\nAverage Performance:")
            report.append(f"MRRMSE: {embedding_data['mrrmse_mean'].mean():.4f} ± {embedding_data['mrrmse_std'].mean():.4f}")
            report.append(f"MSE: {embedding_data['mse_mean'].mean():.4f} ± {embedding_data['mse_std'].mean():.4f}")
            report.append(f"R²: {embedding_data['r2_mean'].mean():.4f} ± {embedding_data['r2_std'].mean():.4f}")
            
            # Best model for this embedding
            best_idx = embedding_data['mrrmse_mean'].idxmin()
            report.append("\nBest Model Performance:")
            report.append(f"Model: {embedding_data.loc[best_idx, 'model_name']}")
            report.append(f"MRRMSE: {embedding_data.loc[best_idx, 'mrrmse_mean']:.4f} ± {embedding_data.loc[best_idx, 'mrrmse_std']:.4f}")
            report.append(f"R²: {embedding_data.loc[best_idx, 'r2_mean']:.4f} ± {embedding_data.loc[best_idx, 'r2_std']:.4f}")
            
            # Technical details
            report.append("\nTechnical Details:")
            report.append(f"Embedding Size: {embedding_data['embedding_size'].iloc[0]}")
            report.append(f"Average Memory Usage: {embedding_data['memory_usage'].mean():.2f}MB")
            report.append(f"Average Training Time: {embedding_data['training_time'].mean():.2f}s")
        
        return "\n".join(report)


    def get_combined_embeddings(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> np.ndarray:
        """
        Get combined embeddings from all embedding methods.
        Also creates a mapping of features to their source embeddings.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the embeddings or use pre-fitted
            
        Returns:
            Combined embedding array
        """
        all_embeddings = []
        current_position = 0
        
        for name, method in self.embedding_methods.items():
            # Get embeddings for this method
            embedding = method.get_embedding(df, fit=fit)
            all_embeddings.append(embedding)
            
            if fit:
                # Store the range for this embedding
                embedding_size = embedding.shape[1]
                end_position = current_position + embedding_size
                self.embedding_ranges[name] = (current_position, end_position)
                
                # Get feature names if available
                try:
                    feature_names = method.get_feature_names()
                except (AttributeError, NotImplementedError):
                    feature_names = [f"{name}_feature_{i}" for i in range(embedding_size)]
                
                # Map each feature to its source embedding
                for i, feature in enumerate(feature_names):
                    self.feature_map[current_position + i] = {
                        'embedding': name,
                        'feature': feature
                    }
                
                current_position = end_position
                self.combined_embedding_size = current_position
        
        return np.hstack(all_embeddings)

    def analyze_combined_feature_importance(
            self,
            df: pd.DataFrame,
            k: int = 10,
            methods: List[str] = ['integrated_gradients', 'shapley']
        ) -> Dict[str, Any]:
            """
            Analyze feature importance using combined embeddings from all methods.
            
            Args:
                df: Input DataFrame
                k: Number of top features to analyze per embedding method
                methods: List of interpretability methods to use
                
            Returns:
                Dictionary containing feature importance analysis results
            """
            import torch
            from captum.attr import (
                IntegratedGradients,
                Saliency,
                DeepLift,
                ShapleyValueSampling
            )
            
            # Prepare data
            train_df, test_df, y_train, y_test = self.prepare_data(df)
            
            # Get combined embeddings
            train_embeddings = self.get_combined_embeddings(train_df, fit=True)
            test_embeddings = self.get_combined_embeddings(test_df, fit=False)
            
            results = {}
            
            for model_factory in self.model_factories:
                print(f"\nAnalyzing with {model_factory.name}...")
                
                # Create and check model
                model = model_factory.create_model(self.combined_embedding_size)
                # if not isinstance(model, torch.nn.Module):
                #     print(f"Skipping {model_factory.name} - not a PyTorch model")
                #     continue
                    
                # Convert data to PyTorch tensors
                X_train = torch.FloatTensor(train_embeddings)
                y_train = torch.FloatTensor(y_train)
                X_test = torch.FloatTensor(test_embeddings)
                
                # Train model
                model.fit(X_train, y_train)
                model.model.eval()
                
                model_results = {
                    'model_name': model_factory.name,
                    'methods': {}
                }
                
                # Create subplot grid based on number of methods
                n_methods = len(methods)
                fig = plt.figure(figsize=(15, 6 * n_methods))
                gs = plt.GridSpec(n_methods, 1, height_ratios=[1] * n_methods)
                
                for idx, method in enumerate(methods):
                    # Compute attributions
                    if method == 'integrated_gradients':
                        explainer = IntegratedGradients(model.model)
                    elif method == 'shapley':
                        explainer = ShapleyValueSampling(model.model)
                    elif method == 'deeplift':
                        explainer = DeepLift(model.model)
                    elif method == 'saliency':
                        explainer = Saliency(model.model)
                    
                    baseline = torch.zeros_like(X_test)
                    attributions = explainer.attribute(X_test, baseline)
                    attr_mean = attributions.mean(dim=0).abs().cpu().detach().numpy()
                    
                    # Process results per embedding method
                    embedding_importance = {}
                    for embedding_name, (start, end) in self.embedding_ranges.items():
                        embedding_attr = attr_mean[start:end]
                        top_k_idx = np.argsort(embedding_attr)[-k:][::-1]
                        
                        embedding_importance[embedding_name] = {
                            'features': [self.feature_map[start + i]['feature'] for i in top_k_idx],
                            'importance': embedding_attr[top_k_idx].tolist(),
                            'all_importance': embedding_attr.tolist()
                        }
                    
                    model_results['methods'][method] = embedding_importance
                    
                    # Create visualization
                    ax = plt.subplot(gs[idx])
                    
                    # Prepare data for plotting
                    plot_data = []
                    for emb_name, importance in embedding_importance.items():
                        for feat, imp in zip(importance['features'], importance['importance']):
                            plot_data.append({
                                'Embedding': emb_name,
                                'Feature': feat,
                                'Importance': imp
                            })
                    
                    plot_df = pd.DataFrame(plot_data)
                    
                    # Create grouped bar plot
                    sns.barplot(
                        data=plot_df,
                        x='Importance',
                        y='Feature',
                        hue='Embedding',
                        palette='Set2',
                        ax=ax
                    )
                    
                    ax.set_title(f'{method.replace("_", " ").title()} - Feature Importance by Embedding')
                    ax.set_xlabel('Absolute Importance')
                    
                    # Adjust legend
                    ax.legend(title='Embedding Type', bbox_to_anchor=(1.05, 1), loc='upper left')
                    
                plt.tight_layout()
                plt.show()
                
                results[model_factory.name] = model_results
            
            return results


from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from tqdm.auto import tqdm

class EnhancedEmbeddingBenchmark(EmbeddingBenchmark):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_results = []  # Store individual run results
        
    def evaluate(
        self,
        model_factory: ModelFactory,
        embedding_name: str,
        embedding_method: BaseEmbedding,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[BenchmarkResult, List[Dict]]:
        """Extended evaluate method to store individual run results"""
        individual_runs = []
        
        # Compute embeddings once
        with tqdm(total=2, desc=f"Computing {embedding_name} embeddings", leave=False) as pbar:
            start_time = time.time()
            train_embeddings = embedding_method.get_embedding(train_df, fit=True)
            pbar.update(1)
            test_embeddings = embedding_method.get_embedding(test_df, fit=False)
            pbar.update(1)
            embedding_time = time.time() - start_time
        
        # Create model with correct input size
        model = model_factory.create_model(embedding_method.embedding_size)

        # Progress bar for runs
        run_pbar = tqdm(
            range(self.n_runs), 
            desc=f"{model_factory.name} with {embedding_name}",
            leave=False
        )
        
        for run in range(self.n_runs):
            print(f"Run {run + 1}/{self.n_runs}")
            
            # Train model
            start_time = time.time()
            model.fit(train_embeddings, y_train)
            training_time = time.time() - start_time
            
            # Evaluate
            start_time = time.time()
            y_pred = model.predict(test_embeddings)
            inference_time = time.time() - start_time
            
            # Compute metrics
            mse = mean_squared_error(y_test, y_pred)
            mrrmse = RMSE_rowwise_loss_numpy(y_pred, y_test)
            r2 = r2_score(y_test, y_pred)
            evs = evaluate_evs_score(y_test, y_pred)
            
            run_result = {
                'run': run,
                'model_name': model_factory.name,
                'embedding_name': embedding_name,
                'mse': mse,
                'mrrmse': mrrmse,
                'r2': r2,
                'evs': evs,
                'training_time': training_time,
                'inference_time': inference_time,
                'memory_usage': (train_embeddings.nbytes + test_embeddings.nbytes) / (1024 * 1024)
            }

            run_pbar.set_postfix({
                'MSE': f"{mse:.4f}",
                'R²': f"{r2:.4f}",
                'MRRMSE': f"{mrrmse:.4f}",
                'EVS': f"{evs:.4f}"
            })
            
            individual_runs.append(run_result)
        
        # Calculate average metrics for backward compatibility
        avg_result = BenchmarkResult(
            model_name=model_factory.name,
            embedding_name=embedding_name,
            mse_mean=np.mean([r['mse'] for r in individual_runs]),
            mse_std=np.std([r['mse'] for r in individual_runs]),
            mrrmse_mean=np.mean([r['mrrmse'] for r in individual_runs]),
            mrrmse_std=np.std([r['mrrmse'] for r in individual_runs]),
            r2_mean=np.mean([r['r2'] for r in individual_runs]),
            r2_std=np.std([r['r2'] for r in individual_runs]),
            evs_mean=np.mean([r['evs'] for r in individual_runs]),
            evs_std=np.std([r['evs'] for r in individual_runs]),
            training_time=np.mean([r['training_time'] for r in individual_runs]),
            inference_time=np.mean([r['inference_time'] for r in individual_runs]),
            embedding_size=embedding_method.embedding_size,
            memory_usage=individual_runs[0]['memory_usage']
        )
        print("Run Compelelted !")
        return avg_result, individual_runs

    def run_benchmark(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced benchmark runner that preserves individual run results"""
        df = df.copy()
        train_df, test_df, y_train, y_test = self.prepare_data(df)
        
        all_individual_runs = []
        aggregated_results = []
        
        for name, method in self.embedding_methods.items():
            print(f"\nEvaluating {name} embedding...")
            
            for model_factory in self.model_factories:
                print(f"Testing with {model_factory.name}...")
                try:
                    avg_result, individual_runs = self.evaluate(
                        model_factory, name, method,
                        train_df, test_df,
                        y_train, y_test
                    )
                    aggregated_results.append(avg_result)
                    all_individual_runs.extend(individual_runs)
                except Exception as e:
                    print(f"Error evaluating {name} embedding with {model_factory.name}: {str(e)}")
        
        self.results = aggregated_results
        self.run_results = all_individual_runs
        
        # Perform statistical analysis
        print("\nPerforming Statistical Analysis...")
        self._analyze_results()
        
        return pd.DataFrame([vars(r) for r in self.results])

    def _analyze_results(self):
        """Perform statistical analysis on the run results"""
        if not self.run_results:
            print("No results available for analysis")
            return
        
        runs_df = pd.DataFrame(self.run_results)
        metrics = ['mrrmse', 'mse', 'r2', 'evs']
        
        for metric in metrics:
            print(f"\nAnalyzing {metric}:")
            
            # Perform one-way ANOVA
            embeddings = runs_df['embedding_name'].unique()
            embedding_groups = [
                runs_df[runs_df['embedding_name'] == emb][metric].values 
                for emb in embeddings
            ]
            
            f_statistic, p_value = stats.f_oneway(*embedding_groups)
            
            print(f"ANOVA Results:")
            print(f"F-statistic: {f_statistic:.4f}")
            print(f"p-value: {p_value:.4f}")
            
            # Visualize distribution of results
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=runs_df, x='embedding_name', y=metric)
            plt.title(f'Distribution of {metric} by Embedding Method\nANOVA p-value: {p_value:.4f}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
            # Perform Tukey's HSD test
            try:
                metrics_df = runs_df.copy()
                metrics_df[metric] = pd.to_numeric(metrics_df[metric], errors='coerce')
                metrics_df = metrics_df.dropna(subset=[metric])
                tukey = pairwise_tukeyhsd(
                    endog=metrics_df[metric],  
                    groups=metrics_df['embedding_name'],  
                    alpha=0.05
                )

                
                # Print Tukey's test summary
                print("\nTukey HSD Test Results:")
                print(tukey)
            
                # Convert summary to DataFrame for better visualization
                tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
                print("\nFormatted Tukey HSD Test Results:")
                print(tukey_df)

            except Exception as e:
                print(f"Could not perform Tukey's HSD test: {str(e)}")


embeddings = {
    "OneHot": OneHotEmbedding(columns=['cell_type', 'sm_name']),
    "SMILES": SMILESEmbedding(),
    "Target": TargetEmbedding(emb_size=256),
    "MorganFP": MorganFingerPrintEmbedding(hidden_size=128),
    "Mol2Vec": Mol2VecEmbedding(model_path=f"/kaggle/input/mol2vec/pytorch/default/1/model_300dim.pkl"),
    "Chemberta": ChemBERTaEmbedding(
        model_name="DeepChem/ChemBERTa-77M-MTR",
        embedding_type='mean_pooling',
    ),
    "Molformer": MolformerEmbedding(
        columns='SMILES',
        padding=True
    ),
    "Smolebart": SmoleBartEmbedding(
        columns='SMILES',
        embedder='encoder',
        padding=True
    ),
    "Selfformer": SelfFormerEmbedding(
        columns='SMILES',
        model_name='/kaggle/input/selfformer/transformers/default/1/SELFormer',
        batch_size=32,
        nb_workers=5
    )
}


def get_vanilla_sklearn():
    return [
        # LinearRegressionFactory(),
        DecisionTreeFactory(
            max_depth=7,
            min_samples_split=10
        ),
        KNeighborsFactory(
            n_neighbors=7,
            weights='uniform'
        ),
        MLPRegressorFactory(
            hidden_layer_sizes=(64, 32),
            alpha=0.01,
            max_iter=1000
        )
    ]

def get_tsvd_sklearn():
    return [
        # LinearRegressionTSVDFactory(n_components=50),
        DecisionTreeTSVDFactory(
            n_components=50,
            max_depth=7,
            min_samples_split=10
        ),
        KNeighborsTSVDFactory(
            n_components=50,
            n_neighbors=7,
            weights='uniform'
        ),
        MLPRegressorTSVDFactory(
            n_components=50,
            hidden_layer_sizes=(64, 32),
            alpha=0.01,
            max_iter=1000
        )
    ]

def get_lstm(output_size: int = 18211):
    return PyTorchModelFactory(
        model_class=LSTMPredictionModel,
        model_params={'hidden_size': 128, 'output_size': output_size},
        num_epochs=50,
        early_stopping=False,
        criterion=RMSE_rowwise_loss,
        optimizer_class=torch.optim.Adam,
        optimizer_params={'lr': 1e-3},
        name="LSTM"
    )


model_factories = get_tsvd_sklearn()
model_factories.append(get_lstm())

# benchmark = EmbeddingBenchmark(embeddings, model_factories = [lstm_factory],  n_runs = 1)
benchmark = EnhancedEmbeddingBenchmark(embeddings, model_factories = model_factories, n_runs = 30)

# Run benchmark
results_df = benchmark.run_benchmark(df)

# Visualize results
benchmark.visualize_results()

# Generate detailed report
report = benchmark.generate_report()
print(report)


combined_embedding = {}

for name, emb in embeddings.items():
    # Skip combining Target with itself
    if name == "Target":
        continue
    
    # Create combined embedding (Target + current embedding)
    combined_name = f"Target+{name}"
    combined_embedding[combined_name] = MultiEmbedding(
        embeddings=[embeddings["Target"], emb],
        name=combined_name,
        combination_method="concat"
    )
    
    # Compute and store the combined embeddings
    # print(f"Computing combined embeddings for {combined_name}...")
    # combinations[combined_name].compute_and_store_embeddings(df, entity_column, batch_size)


model_factories = get_tsvd_sklearn()
model_factories.append(get_lstm())

# Create benchmark instance
benchmark = EnhancedEmbeddingBenchmark(combined_embedding, model_factories = model_factories,  n_runs = 30)

# Run benchmark
results_df = benchmark.run_benchmark(df)

# Visualize results
benchmark.visualize_results()

# Generate detailed report
report = benchmark.generate_report()
print(report)


model_factories = get_vanilla_sklearn()
model_factories.append(get_lstm(256))

# benchmark = EmbeddingBenchmark(embeddings, model_factories = [lstm_factory],  n_runs = 1)
benchmark = EnhancedEmbeddingBenchmark(embeddings, model_factories = model_factories, n_runs = 30, only_significant = True)

# Run benchmark
results_df = benchmark.run_benchmark(df)

# Visualize results
benchmark.visualize_results()

# Generate detailed report
report = benchmark.generate_report()
print(report)


model_factories = get_vanilla_sklearn()
model_factories.append(get_lstm(256))

# Create benchmark instance
benchmark = EnhancedEmbeddingBenchmark(combined_embedding, model_factories = model_factories,  n_runs = 30, only_significant = True)

# Run benchmark
results_df = benchmark.run_benchmark(df)

# Visualize results
benchmark.visualize_results()

# Generate detailed report
report = benchmark.generate_report()
print(report)


class Combined_Benchmarker():
    def __init__(
        self,
        embedding_methods: Dict[str, BaseEmbedding],
        model_factories: List[ModelFactory],
        n_runs: int = 5
    ):
        """
        Initialize with dictionary of embedding methods and model factories.
        Also creates a combined embedding handler.
        """
        self.embedding_methods = embedding_methods
        self.model_factories = model_factories
        self.n_runs = n_runs
        self.results: List[BenchmarkResult] = []
        
        # Add combined embedding functionality
        self.combined_embedding_size = 0
        self.embedding_ranges = {}
        self.feature_map = {}
        
    def get_combined_embeddings(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> np.ndarray:
        """
        Get combined embeddings from all embedding methods.
        Also creates a mapping of features to their source embeddings.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the embeddings or use pre-fitted
            
        Returns:
            Combined embedding array
        """
        all_embeddings = []
        current_position = 0
        
        for name, method in self.embedding_methods.items():
            # Get embeddings for this method
            embedding = method.get_embedding(df, fit=fit)
            all_embeddings.append(embedding)
            
            if fit:
                # Store the range for this embedding
                embedding_size = embedding.shape[1]
                end_position = current_position + embedding_size
                self.embedding_ranges[name] = (current_position, end_position)
                
                # Get feature names if available
                try:
                    feature_names = method.get_feature_names()
                except (AttributeError, NotImplementedError):
                    feature_names = [f"{name}_feature_{i}" for i in range(embedding_size)]
                
                # Map each feature to its source embedding
                for i, feature in enumerate(feature_names):
                    self.feature_map[current_position + i] = {
                        'embedding': name,
                        'feature': feature
                    }
                
                current_position = end_position
                self.combined_embedding_size = current_position
        
        return np.hstack(all_embeddings)
    
    def analyze_combined_feature_importance(
        self,
        df: pd.DataFrame,
        k: int = 10,
        methods: List[str] = ['integrated_gradients', 'shapley']
    ) -> Dict[str, Any]:
        """
        Analyze feature importance using combined embeddings from all methods.
        
        Args:
            df: Input DataFrame
            k: Number of top features to analyze per embedding method
            methods: List of interpretability methods to use
            
        Returns:
            Dictionary containing feature importance analysis results
        """
        import torch
        from captum.attr import (
            IntegratedGradients,
            Saliency,
            DeepLift,
            ShapleyValueSampling
        )
        
        # Prepare data
        train_df, test_df, y_train, y_test = self.prepare_data(df)
        
        # Get combined embeddings
        train_embeddings = self.get_combined_embeddings(train_df, fit=True)
        test_embeddings = self.get_combined_embeddings(test_df, fit=False)
        
        results = {}
        
        for model_factory in self.model_factories:
            print(f"\nAnalyzing with {model_factory.name}...")
            
            # Create and check model
            model = model_factory.create_model(self.combined_embedding_size)
            if not isinstance(model, torch.nn.Module):
                print(f"Skipping {model_factory.name} - not a PyTorch model")
                continue
                
            # Convert data to PyTorch tensors
            X_train = torch.FloatTensor(train_embeddings)
            y_train = torch.FloatTensor(y_train)
            X_test = torch.FloatTensor(test_embeddings)
            
            # Train model
            model.fit(X_train, y_train)
            model.eval()
            
            model_results = {
                'model_name': model_factory.name,
                'methods': {}
            }
            
            # Create subplot grid based on number of methods
            n_methods = len(methods)
            fig = plt.figure(figsize=(15, 6 * n_methods))
            gs = plt.GridSpec(n_methods, 1, height_ratios=[1] * n_methods)
            
            for idx, method in enumerate(methods):
                # Compute attributions
                if method == 'integrated_gradients':
                    explainer = IntegratedGradients(model)
                elif method == 'shapley':
                    explainer = ShapleyValueSampling(model)
                elif method == 'deeplift':
                    explainer = DeepLift(model)
                elif method == 'saliency':
                    explainer = Saliency(model)
                
                baseline = torch.zeros_like(X_test)
                attributions = explainer.attribute(X_test, baseline)
                attr_mean = attributions.mean(dim=0).abs().cpu().detach().numpy()
                
                # Process results per embedding method
                embedding_importance = {}
                for embedding_name, (start, end) in self.embedding_ranges.items():
                    embedding_attr = attr_mean[start:end]
                    top_k_idx = np.argsort(embedding_attr)[-k:][::-1]
                    
                    embedding_importance[embedding_name] = {
                        'features': [self.feature_map[start + i]['feature'] for i in top_k_idx],
                        'importance': embedding_attr[top_k_idx].tolist(),
                        'all_importance': embedding_attr.tolist()
                    }
                
                model_results['methods'][method] = embedding_importance
                
                # Create visualization
                ax = plt.subplot(gs[idx])
                
                # Prepare data for plotting
                plot_data = []
                for emb_name, importance in embedding_importance.items():
                    for feat, imp in zip(importance['features'], importance['importance']):
                        plot_data.append({
                            'Embedding': emb_name,
                            'Feature': feat,
                            'Importance': imp
                        })
                
                plot_df = pd.DataFrame(plot_data)
                
                # Create grouped bar plot
                sns.barplot(
                    data=plot_df,
                    x='Importance',
                    y='Feature',
                    hue='Embedding',
                    palette='Set2',
                    ax=ax
                )
                
                ax.set_title(f'{method.replace("_", " ").title()} - Feature Importance by Embedding')
                ax.set_xlabel('Absolute Importance')
                
                # Adjust legend
                ax.legend(title='Embedding Type', bbox_to_anchor=(1.05, 1), loc='upper left')
                
            plt.tight_layout()
            plt.show()
            
            results[model_factory.name] = model_results
        
        return results


lstm_factory.device = 'cpu'
cb = EmbeddingBenchmark(embeddings, model_factories = [lstm_factory],  n_runs = 1)
# results = cb.analyze_combined_feature_importance(df)

