# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
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




!pip install -q umap-learn rdkit captum git+https://github.com/samoturk/mol2vec selfies==2.1.1  simpletransformers==0.63.9 pandarallel==1.6.4 wandb==0.13.10


from gensim.models import word2vec
model1 = word2vec.Word2Vec.load('/kaggle/input/mol2vec/pytorch/default/1/model_300dim.pkl')


import pandas as pd
import numpy as np
from selfies import encoder

# Function to select top variable genes
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

# Load dataset
df = pd.read_parquet('/kaggle/input/open-problems-single-cell-perturbations/de_train.parquet')

# Apply the function to create the 'SELFIES' column
df["SELFIES"] = df["SMILES"].apply(smiles_to_selfies)

# # Keep SMILES column for later use
# df_smiles = df[["cell_type", "sm_name", "SMILES"]].copy()  

# # Drop unwanted columns before filtering genes
# df.drop(columns=["sm_lincs_id", "control"], inplace=True)

# # Set multi-index
# df.set_index(["cell_type", "sm_name"], inplace=True)

# # Select top 256 variable genes
# top_genes = select_top_variable_genes(df, k=256, exclude_controls=True)
# df_filtered = df[top_genes]  # Keep only selected genes

# # Restore SMILES column
# df_filtered = df_filtered.merge(df_smiles.set_index(["cell_type", "sm_name"]), left_index=True, right_index=True)

# # Function to convert SMILES to SELFIES
# def smiles_to_selfies(smiles):
#     try:
#         return encoder(smiles)
#     except Exception as e:
#         print(f"Error converting SMILES '{smiles}': {e}") 
#         return None  


# # Final dataset
# df_filtered.reset_index(inplace=True)  # Reset index for easier processing
# df_filtered.head()
# df = df_filtered


df


top_genes


from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, MolSentence
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, QED
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcNumRotatableBonds, CalcNumHBA, CalcNumHBD, CalcFractionCSP3
from rdkit.Chem import BRICS, Recap
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import AutoModelForMaskedLM, AutoTokenizer , AutoModel
from tqdm import tqdm
import os
import json
import pickle
from IPython.display import clear_output
from pandarallel import pandarallel
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig

class BaseEmbedding(ABC):
    """Abstract base class for embeddings with save/load functionality."""
    
    def __init__(self):
        self.embedding_dict = {}
        self.metadata = {}
    
    @abstractmethod
    def preprocess(self, df):
        pass
    
    @abstractmethod
    def get_embedding(self, df):
        pass
    
    @property
    @abstractmethod
    def embedding_size(self):
        pass
    
    def compute_and_store_embeddings(self, df, entity_column):
        """
        Compute and store embeddings for unique entities in the specified column.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            entity_column (str): Column name containing entities (e.g., 'cell_type' or 'sm_name')
        """
        unique_entities = df[entity_column].unique()
        for entity in unique_entities:
            entity_df = df[df[entity_column] == entity].copy()
            embedding = self.get_embedding(entity_df, fit=False)
            # Store the mean embedding if there are multiple rows
            self.embedding_dict[entity] = embedding.mean().values
            
    def get_entity_embedding(self, entity_name):
        """
        Retrieve embedding for a specific entity.
        
        Args:
            entity_name (str): Name of the entity
            
        Returns:
            np.ndarray: Embedding vector for the entity
        """
        if entity_name in self.embedding_dict:
            return self.embedding_dict[entity_name]
        else:
            return np.zeros(self.embedding_size)
    
    def save_embeddings(self, filepath, metadata=None):
        """
        Save embeddings and metadata to disk.
        
        Args:
            filepath (str): Path to save the embeddings
            metadata (dict, optional): Additional metadata to save
        """
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
            'embedding_type': self.__class__.__name__
        }
        
        # Save as pickle if the filepath ends with .pkl, otherwise save as JSON
        if filepath.endswith('.pkl'):
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
        else:
            with open(filepath, 'w') as f:
                json.dump(save_data, f)
                
    def load_embeddings(self, filepath):
        """
        Load embeddings and metadata from disk.
        
        Args:
            filepath (str): Path to load the embeddings from
            
        Returns:
            bool: True if loading was successful
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
            
            # Verify embedding size matches
            if save_data['embedding_size'] != self.embedding_size:
                raise ValueError(
                    f"Loaded embedding size ({save_data['embedding_size']}) "
                    f"doesn't match current embedding size ({self.embedding_size})"
                )
            
            return True
            
        except Exception as e:
            print(f"Error loading embeddings: {str(e)}")
            return False


class OneHotEmbedding(BaseEmbedding):
    def __init__(self, columns):
        super().__init__()
        self.columns = columns
        self.encoder = OneHotEncoder(sparse_output=False)
        self._embedding_size = None

    def preprocess(self, df):
        return df[self.columns]

    def get_embedding(self, df, fit=True):
        if fit:
            encoded_features = self.encoder.fit_transform(df[self.columns])
        else:
            encoded_features = self.encoder.transform(df[self.columns])
        
        self._embedding_size = encoded_features.shape[1]
        encoded_df = pd.DataFrame(encoded_features, columns=self.encoder.get_feature_names_out(self.columns))
        
        # Store embeddings using base class functionality
        for idx, row in df[self.columns].iterrows():
            key = tuple(row.values)
            self.embedding_dict[key] = encoded_df.loc[idx].values
            
        return encoded_df

    @property
    def embedding_size(self):
        return self._embedding_size

    
class SMILESEmbedding(BaseEmbedding):
    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()

    def preprocess(self, df):
        return df

    def create_molecule_embedding_dict(self, df):
        """
        Create a dictionary of molecule embeddings.
        
        Args:
            df (pd.DataFrame): DataFrame containing SMILES column
        """
        self.compute_and_store_embeddings(df, 'SMILES')
        
    def get_embedding(self, df, fit=True):
        smiles_info_list = df['SMILES'].apply(self.extract_smiles_info)
        smiles_info_df = pd.DataFrame(smiles_info_list.tolist())
        if fit:
            smiles_info_df = self.scaler.fit_transform(smiles_info_df)
        else:
            smiles_info_df = self.scaler.transform(smiles_info_df)

            
        columns = [
            'Molecular Weight', 'LogP', 'TPSA', 'Number of Atoms', 'Number of Bonds',
            'Number of Rotatable Bonds', 'Number of Hydrogen Bond Acceptors', 'Number of Hydrogen Bond Donors',
            'Number of Rings', 'Number of Aromatic Rings', 'Number of Stereocenters',
            'Fraction of sp3 Carbons', 'Balaban J Index', 'Bertz CT', 'QED Score'
        ]
        
        result_df = pd.DataFrame(smiles_info_df, columns=columns)
        
        # Store embeddings using base class functionality
        for idx, smiles in enumerate(df['SMILES']):
            self.embedding_dict[smiles] = result_df.loc[idx].values
            
        return result_df

    @staticmethod
    def extract_smiles_info(smiles):
        if smiles is None or smiles == '':
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        info = {
            'Molecular Weight': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'TPSA': CalcTPSA(mol),
            'Number of Atoms': mol.GetNumAtoms(),
            'Number of Bonds': mol.GetNumBonds(),
            'Number of Rotatable Bonds': CalcNumRotatableBonds(mol),
            'Number of Hydrogen Bond Acceptors': CalcNumHBA(mol),
            'Number of Hydrogen Bond Donors': CalcNumHBD(mol),
            'Number of Rings': Descriptors.RingCount(mol),
            'Number of Aromatic Rings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'Number of Stereocenters': len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
            'Fraction of sp3 Carbons': CalcFractionCSP3(mol),
            'Balaban J Index': Descriptors.BalabanJ(mol),
            'Bertz CT': Descriptors.BertzCT(mol),
            'QED Score': QED.qed(mol)
        }
        return info

    @property
    def embedding_size(self):
        return 15  

class Mol2VecEmbedding(BaseEmbedding):
    def __init__(self, model_path):
        super().__init__()
        self.model = word2vec.Word2Vec.load(model_path)
        self.keys = set(self.model.wv.key_to_index.keys())

    def preprocess(self, df):
        df['mol'] = df['SMILES'].apply(lambda x: Chem.MolFromSmiles(x))
        df['sentence'] = df.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)
        return df

    def get_embedding(self, df, fit=True):
        df['vector'] = df['sentence'].apply(lambda sentence: self.sentence_to_vector(sentence))
        vector_dim = len(self.model.wv.get_vector(next(iter(self.keys))))
        vector_columns = [f'vector_{i}' for i in range(vector_dim)]
        result_df = pd.DataFrame(df['vector'].tolist(), columns=vector_columns)
        
        # Store embeddings using base class functionality
        for idx, smiles in enumerate(df['SMILES']):
            self.embedding_dict[smiles] = result_df.loc[idx].values
            
        return result_df

    def sentence_to_vector(self, sentence, unseen=False, unseen_vec=np.zeros(300)):
        if unseen:
            vec = sum([self.model.wv.get_vector(word) if word in self.keys else unseen_vec for word in sentence])
        else:
            vec = sum([self.model.wv.get_vector(word) for word in sentence if word in self.keys])
        return vec

    def create_molecule_embedding_dict(self, df):
        """
        Create a dictionary of molecule embeddings.
        
        Args:
            df (pd.DataFrame): DataFrame containing SMILES column
        """
        self.compute_and_store_embeddings(df, 'SMILES')
    
    @property
    def embedding_size(self):
        return len(self.model.wv.get_vector(next(iter(self.keys))))
    

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class TargetEmbedding(BaseEmbedding, nn.Module):
    """Target embedding class using autoencoder to learn compressed representations of medians."""
    
    def __init__(self, df_train, emb_size=256, hidden_size=1024):
        """
        Initialize the autoencoder target embedding class.
        
        Args:
            df_train (pd.DataFrame): Training DataFrame for storing medians
            emb_size (int): Size of the latent embedding for each component
            hidden_size (int): Size of the hidden layer in encoder/decoder
        """
        
        BaseEmbedding.__init__(self)
        nn.Module.__init__(self)
        
        self.emb_size = emb_size
        self._embedding_size = emb_size * 2  # combined size of both embeddings
        self.input_size = 18211  # Original dimension of median values
        
        # Encoder networks for cell type and small molecule embeddings
        self.cell_type_encoder = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, emb_size)
        )
        
        self.sm_encoder = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, emb_size)
        )
        
        # Decoder networks for reconstruction
        self.cell_type_decoder = nn.Sequential(
            nn.Linear(emb_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.input_size)
        )
        
        self.sm_decoder = nn.Sequential(
            nn.Linear(emb_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.input_size)
        )
        
        # Initialize dictionaries
        self.cell_type_dict = {}
        self.sm_dict = {}
        
        # Process training data
        df_train = self.preprocess(df_train)
        
        # Store training data and compute medians
        self.de_cell_type_train = df_train.iloc[:, [0] + list(range(5, df_train.shape[1]))]
        self.de_sm_name_train = df_train.iloc[:, [1] + list(range(5, df_train.shape[1]))]
        
        # Rename columns for consistency
        self.de_cell_type_train.columns = ['cell_type' if i == 0 else col 
                                         for i, col in enumerate(self.de_cell_type_train.columns)]
        self.de_sm_name_train.columns = ['sm_name' if i == 0 else col 
                                         for i, col in enumerate(self.de_sm_name_train.columns)]
        
        # Compute medians from training data
        self.cell_type_medians = self.de_cell_type_train.select_dtypes(include=['number']).groupby(self.de_cell_type_train['cell_type']).median()
        self.sm_name_medians = self.de_sm_name_train.select_dtypes(include=['number']).groupby(self.de_sm_name_train['sm_name']).median()

        
        # Convert medians to tensors
        self.cell_type_tensors = {k: torch.tensor(v).float() 
                                 for k, v in zip(self.cell_type_medians.index, 
                                               self.cell_type_medians.values)}
        self.sm_tensors = {k: torch.tensor(v).float() 
                          for k, v in zip(self.sm_name_medians.index, 
                                        self.sm_name_medians.values)}
        
        self.__class__.name = "TargetEmbedding"
        self.is_fitted = False
    
    def preprocess(self, df):
        """
        Preprocess the input DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        # Copy the DataFrame to avoid modifying the original
        df = df.copy()
        
        # Ensure required columns exist
        required_cols = ['cell_type', 'sm_name']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Handle any missing values in gene expression columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df
    
    def train_autoencoder(self, num_epochs=100, batch_size=32, learning_rate=1e-3):
        """
        Train the autoencoder to learn compressed representations of the medians.
        
        Args:
            num_epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for optimizer
        """
        self.train()  # Set to training mode
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Convert median dictionaries to tensors for training
        cell_type_data = torch.stack(list(self.cell_type_tensors.values()))
        sm_data = torch.stack(list(self.sm_tensors.values()))
        
        for epoch in range(num_epochs):
            # Train cell type autoencoder
            cell_type_encoded = self.cell_type_encoder(cell_type_data)
            cell_type_decoded = self.cell_type_decoder(cell_type_encoded)
            cell_type_loss = criterion(cell_type_decoded, cell_type_data)
            
            # Train small molecule autoencoder
            sm_encoded = self.sm_encoder(sm_data)
            sm_decoded = self.sm_decoder(sm_encoded)
            sm_loss = criterion(sm_decoded, sm_data)
            
            # Combined loss
            total_loss = cell_type_loss + sm_loss
            
            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Loss: {total_loss.item():.4f}, '
                      f'CT Loss: {cell_type_loss.item():.4f}, '
                      f'SM Loss: {sm_loss.item():.4f}')
        
        self.eval()  # Set to evaluation mode
        # After training, update the dictionaries with encoded values
        with torch.no_grad():
            for k, v in self.cell_type_tensors.items():
                self.cell_type_dict[k] = self.cell_type_encoder(v.unsqueeze(0)).squeeze(0)
            
            for k, v in self.sm_tensors.items():
                self.sm_dict[k] = self.sm_encoder(v.unsqueeze(0)).squeeze(0)
        
        self.is_fitted = True
    
    def get_embedding(self, df, fit=False):
        """
        Get embeddings for the input data using the trained autoencoder.
        
        Args:
            df (pd.DataFrame): Input DataFrame with cell_type and sm_name columns
            fit (bool): If True, train the autoencoder before getting embeddings
            
        Returns:
            pd.DataFrame: DataFrame containing the embedded features
        """
        # Preprocess input data
        df = self.preprocess(df)
        
        # Train autoencoder if fit=True and not already fitted
        if fit and not self.is_fitted:
            self.train_autoencoder()
        
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            cell_type_tensors = []
            sm_tensors = []
            
            for _, row in df.iterrows():
                # Get cell type embedding
                ct = row['cell_type']
                if ct in self.cell_type_dict:
                    cell_type_tensors.append(self.cell_type_dict[ct])
                else:
                    cell_type_tensors.append(torch.zeros(self.emb_size))
                
                # Get small molecule embedding
                sm = row['sm_name']
                if sm in self.sm_dict:
                    sm_tensors.append(self.sm_dict[sm])
                else:
                    sm_tensors.append(torch.zeros(self.emb_size))
                
                # Update the parent class embedding dictionary for both entities
                # Store cell type embedding
                if ct in self.cell_type_dict:
                    self.embedding_dict[f"cell_type_{ct}"] = self.cell_type_dict[ct].numpy()
                
                # Store small molecule embedding
                if sm in self.sm_dict:
                    self.embedding_dict[f"sm_name_{sm}"] = self.sm_dict[sm].numpy()
            
            # Convert to tensors
            cell_type_tensor = torch.stack(cell_type_tensors)
            sm_tensor = torch.stack(sm_tensors)
            
            # Combine embeddings
            combined_embedding = torch.cat([cell_type_tensor, sm_tensor], dim=1)
        
        # Create column names for the embedding DataFrame
        ct_cols = [f'target_ct_emb_{i}' for i in range(self.emb_size)]
        sm_cols = [f'target_sm_emb_{i}' for i in range(self.emb_size)]
        all_cols = ct_cols + sm_cols
        
        return pd.DataFrame(combined_embedding.detach().numpy(), columns=all_cols, index=df.index)
        
    @property
    def embedding_size(self):
        """Returns the size of the combined embedding."""
        return self._embedding_size
    
class MorganFingerPrintEmbedding(BaseEmbedding):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.autoencoder = None  # This will store the trained autoencoder

    def preprocess(self, df):
        return df

    def get_embedding(self, df, fit=True):
        morgan_fp_list = df['SMILES'].apply(self.extract_morgan_fingerprint)
        morgan_fp_array = np.stack(morgan_fp_list)

        input_size = morgan_fp_array.shape[1]

        if fit:
            self.autoencoder = self.train_autoencoder(morgan_fp_array, input_size, self.hidden_size)
        
        with torch.no_grad():
            compressed_fp = self.autoencoder.encoder(torch.tensor(morgan_fp_array, dtype=torch.float32)).numpy()

        result_df = pd.DataFrame(compressed_fp, columns=[f'CompressedFP_{i}' for i in range(self.hidden_size)])
        
        # Store embeddings using base class functionality
        for idx, smiles in enumerate(df['SMILES']):
            self.embedding_dict[smiles] = result_df.loc[idx].values
            
        return result_df

    def extract_morgan_fingerprint(self, smiles, radius=2, nBits=2048):
        if smiles is None or smiles == '':
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        morgan_gen = AllChem.GetMorganGenerator(radius=radius, fpSize=nBits)
        fp = morgan_gen.GetFingerprint(mol)
        
        fp_array = np.zeros((nBits,))
        DataStructs.ConvertToNumpyArray(fp, fp_array)
        
        return fp_array

    def train_autoencoder(self, data, input_size, hidden_size, num_epochs=50, batch_size=32, learning_rate=0.001):
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
    def embedding_size(self):
        return self.hidden_size


class ChemBERTaEmbedding(BaseEmbedding):
    def __init__(self, model_name="DeepChem/ChemBERTa-77M-MTR", embedding_type='mean_pooling', padding=False):
        super().__init__()
        self.model_name = model_name
        self.embedding_type = embedding_type  # 'cls' or 'mean_pooling'
        self.padding = padding
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.eval()
        self._embedding_size = None # Dynamically set during get_embedding


    def preprocess(self, df):
         return df # No specific pre-processing

    def get_embedding(self, df, fit=True):
        smiles_list = df['SMILES'].tolist()
        embeddings_cls, embeddings_mean = self.featurize_ChemBERTa(smiles_list, padding=self.padding)

        if self.embedding_type == 'cls':
            embeddings = embeddings_cls
            emb_cols = [f'ChemBERTa_cls_{i}' for i in range(embeddings.shape[1])]
        elif self.embedding_type == 'mean_pooling':
            embeddings = embeddings_mean
            emb_cols = [f'ChemBERTa_mean_{i}' for i in range(embeddings.shape[1])]
        else:
            raise ValueError("embedding_type must be 'cls' or 'mean_pooling'")

        result_df = pd.DataFrame(embeddings, columns=emb_cols)
        self._embedding_size = embeddings.shape[1]
        # Store embeddings
        for idx, smiles in enumerate(df['SMILES']):
            self.embedding_dict[smiles] = result_df.loc[idx].values

        return result_df

    def featurize_ChemBERTa(self, smiles_list, padding=True):

        embeddings_cls = []
        embeddings_mean = []

        with torch.no_grad():
            for smiles in smiles_list: # Removed tqdm for batch processing
                encoded_input = self.tokenizer(smiles, return_tensors="pt", padding=padding, truncation=True)
                model_output = self.model(**encoded_input)

                embedding_cls = model_output[0][:, 0, :]  # CLS token embedding
                embeddings_cls.append(embedding_cls)

                embedding_mean = torch.mean(model_output[0], 1)  # Mean pooling
                embeddings_mean.append(embedding_mean)
        
        # Stack the tensors and convert to numpy
        embeddings_cls = torch.cat(embeddings_cls).numpy()
        embeddings_mean = torch.cat(embeddings_mean).numpy()
        return embeddings_cls, embeddings_mean
    
    @property
    def embedding_size(self):
        if self._embedding_size is None:
          return 600  # Default, but will be correctly set during get_embedding
        return self._embedding_size

class MolformerEmbedding(BaseEmbedding):
    def __init__(self, model_path='ibm/MoLFormer-XL-both-10pct', padding=True):
        super().__init__()
        self.model_path = model_path
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()  # Ensure the model is in evaluation mode
        self._embedding_size = None
        self.padding = padding # padding should be True for this model

    def preprocess(self, df):
        return df  # No specific pre-processing needed

    def get_embedding(self, df, fit=True):
        smiles_list = df['SMILES'].tolist()
        embeddings = self.featurize_Molformer(smiles_list)
        emb_cols = [f'Molformer_{i}' for i in range(embeddings.shape[1])]

        result_df = pd.DataFrame(embeddings, columns=emb_cols, index=df.index) # added index=df.index
        self._embedding_size = embeddings.shape[1]
        # Store embeddings
        for idx, smiles in enumerate(df['SMILES']):
            self.embedding_dict[smiles] = result_df.loc[idx].values

        return result_df

    def featurize_Molformer(self, smiles_list):
        """
        Generates MolFormer embeddings for a list of SMILES strings.
        """
        inputs = self.tokenizer(smiles_list, padding=self.padding, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use pooler_output for the sentence-level representation
        embeddings = outputs.pooler_output.cpu().numpy()  # Move to CPU before converting to NumPy
        return embeddings


    @property
    def embedding_size(self):
        if self._embedding_size is None:
          #  This needs to be determined dynamically after the first call to featurize, or
          #  looked up from the model config if available.  A placeholder won't work well.
          #  The correct embedding size (after inspecting the model) is 768 for
          #  'ibm/MoLFormer-XL-both-10pct'. However, it's best to set this dynamically.
          return 768 # correct for ibm/MoLFormer-XL-both-10pct, but should ideally be dynamic
        return self._embedding_size

class SmoleBartEmbedding(BaseEmbedding):
    def __init__(self, model_path="UdS-LSV/smole-bart", padding=True, embedder="encoder"):
        super().__init__()
        self.model_path = model_path
        self.padding = padding  # whether to pad the SMILES sequences
        self.embedder = embedder  # choose "encoder" or "decoder"
        # load the model and tokenizer using trust_remote_code=True if needed
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()  # set the model to evaluation mode
        self._embedding_size = None

    def preprocess(self, df):
        # No special preprocessing needed; simply return the dataframe.
        return df

    def get_embedding(self, df, fit=True):
        # df is expected to have a column "SMILES"
        smiles_list = df['SMILES'].tolist()
        embeddings = self.featurize_smole_bart(smiles_list, embedder=self.embedder)
        # Create column names based on the embedding dimension
        emb_cols = [f"SmoleBart_{i}" for i in range(embeddings.shape[1])]
        # Create a DataFrame with the embeddings (preserving the index)
        result_df = pd.DataFrame(embeddings, columns=emb_cols, index=df.index)
        # Set the embedding size based on the shape of embeddings
        self._embedding_size = embeddings.shape[1]
        # Store embeddings in the embedding dictionary
        for idx, smiles in enumerate(df['SMILES']):
            self.embedding_dict[smiles] = result_df.loc[idx].values
        return result_df

    def featurize_smole_bart(self, smiles_list, embedder="encoder"):
        # Tokenize the list of SMILES strings.
        inputs = self.tokenizer(smiles_list, padding=self.padding, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        # For a BART model, there is no pooler output by default.
        # Use mean pooling on the appropriate output.
        if embedder == "encoder":
            encoder_out = outputs.encoder_last_hidden_state  # shape: [batch, seq_len, hidden_size]
            embeddings = encoder_out.mean(dim=1)
        elif embedder == "decoder":
            decoder_out = outputs.last_hidden_state  # shape: [batch, seq_len, hidden_size]
            embeddings = decoder_out.mean(dim=1)
        else:
            raise ValueError("embedder must be either 'encoder' or 'decoder'")
        return embeddings.cpu().numpy()

    @property
    def embedding_size(self):
        # Return the computed embedding size; if not computed, default to 0 (or raise an error)
        return self._embedding_size if self._embedding_size is not None else 0




import os
import torch
import pandas as pd
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
from pandarallel import pandarallel

class SelfFormerEmbedding(BaseEmbedding):
    def __init__(self, model_name, tokenizer_path=None, padding=True, batch_size=32, nb_workers=5, device=None):
        """
        Initialize the SelfFormerEmbedding instance.
        
        Args:
            model_name (str): Path or identifier for the pretrained SELFormer model.
            tokenizer_path (str, optional): If provided, load the tokenizer from this path.
            padding (bool): Whether to apply padding when tokenizing.
            batch_size (int): Number of samples to process in a single batch.
            nb_workers (int): Number of parallel workers for pandarallel.
            device (str, optional): Device to run the model on ("cuda" or "cpu"). If None, auto-detect.
        """
        super().__init__()
        
        # Disable parallelism warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["WANDB_DISABLED"] = "true"
        
        # Set device (auto-detect if not provided)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model_name = model_name
        self.padding = padding
        self.batch_size = batch_size
        self._embedding_size = None
        
        # Load model configuration with hidden states enabled
        config = RobertaConfig.from_pretrained(model_name)
        config.output_hidden_states = True
        self.model = RobertaModel.from_pretrained(model_name, config=config).to(self.device)
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path or model_name)
        
        # Initialize parallel processing
        pandarallel.initialize(nb_workers=nb_workers, progress_bar=True)

    def preprocess(self, df):
        """Preprocessing step (if needed). Currently, it returns the dataframe unchanged."""
        return df

    def get_embedding(self, df, fit=True):
        """
        Generate embeddings for all SELFIES in the dataframe using batch processing.
        
        Args:
            df (pd.DataFrame): Dataframe containing a "SELFIES" column.
            fit (bool): Unused compatibility flag.
        
        Returns:
            pd.DataFrame: DataFrame with computed embeddings.
        """
        selfies_list = df["SELFIES"].tolist()
        embeddings = self.get_embeddings_batch(selfies_list, batch_size=self.batch_size)

        if len(embeddings) > 0:
            self._embedding_size = embeddings.shape[1]

        # Create a result DataFrame
        emb_cols = [f'SELFormer_{i}' for i in range(self._embedding_size)]
        result_df = pd.DataFrame(embeddings, columns=emb_cols, index=df.index)

        # Store embeddings in the dictionary
        for idx, selfie in enumerate(df["SELFIES"]):
            self.embedding_dict[selfie] = result_df.loc[idx].values

        # Drop the original SELFIES column
        return result_df

    @property
    def embedding_size(self):
        """Return the embedding size; if not computed yet, return a default value."""
        return self._embedding_size if self._embedding_size is not None else 600

    def get_embeddings_batch(self, selfies_list, batch_size=32):
        """
        Compute embeddings for a batch of SELFIES strings.
        
        Args:
            selfies_list (list of str): List of SELFIES strings.
            batch_size (int): Batch size for processing.
        
        Returns:
            numpy.ndarray: Array of embeddings.
        """
        all_embeddings = []

        for i in range(0, len(selfies_list), batch_size):
            batch = selfies_list[i: i + batch_size]
            encoded_input = self.tokenizer(
                batch,
                add_special_tokens=True,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                output = self.model(**encoded_input)

            # Mean pooling over the sequence dimension
            embeddings = torch.mean(output.last_hidden_state, dim=1)
            all_embeddings.append(embeddings.cpu())

        # Concatenate all batches and convert to NumPy
        return torch.cat(all_embeddings, dim=0).numpy()

        


class Preprocessor:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.embedding_indices = {}
        self.latest_processed_data = None
        self.sm_name_to_smiles = {}
        self.unique_cell_types = set()
        self.unique_sm_names = set()

    def preprocess(self, df, fit=True):
        """
        Process the input DataFrame through all embeddings and combine the results.

        Args:
            df (pd.DataFrame): Input DataFrame
            fit (bool): Whether to fit the embeddings (train encoders, etc.)

        Returns:
            pd.DataFrame: Combined embedded features
        """

        if fit:
            if 'cell_type' in df.columns:
                self.unique_cell_types.update(df['cell_type'].unique())
            if 'sm_name' in df.columns:
                self.unique_sm_names.update(df['sm_name'].unique())

        if fit and 'sm_name' in df.columns and 'SMILES' in df.columns:
            sm_name_smiles_map = df[['sm_name', 'SMILES']].drop_duplicates()
            self.sm_name_to_smiles.update(
                dict(zip(sm_name_smiles_map['sm_name'], sm_name_smiles_map['SMILES']))
            )

        processed_dfs = []
        current_index = 0

        for embedding in self.embeddings:
            embedding_name = embedding.__class__.__name__
            # Handle SMILES-based embeddings, including the new ones
            smiles_based_embeddings = [
                'SMILESEmbedding', 'MorganFingerPrintEmbedding', 'Mol2VecEmbedding',
                'ChemBERTaEmbedding','MolformerEmbedding','SmoleBartEmbedding','SelfFormerEmbedding'
            ]
            if embedding_name in smiles_based_embeddings:
                if 'SMILES' not in df.columns:
                    # Create a copy to avoid modifying the original DataFrame
                    df_with_smiles = df.copy()
                    # Add SMILES column using the stored mapping
                    df_with_smiles['SMILES'] = df_with_smiles['sm_name'].map(self.sm_name_to_smiles)
                    # Check for missing SMILES *after* mapping
                    if df_with_smiles['SMILES'].isnull().any():
                        print(f"Warning: Missing SMILES for some sm_names in {embedding_name}.  Filling with empty string.")
                        df_with_smiles['SMILES'] = df_with_smiles['SMILES'].fillna('')  # Or some other placeholder
                    preprocessed_df = embedding.preprocess(df_with_smiles)
                    embedded_df = embedding.get_embedding(preprocessed_df, fit=fit)
                else:
                     # Check for missing/empty SMILES *before* processing
                    if df['SMILES'].isnull().any() or (df['SMILES'] == '').any():
                        print(f"Warning: Missing/Empty SMILES found in {embedding_name}. Filling with empty string.")
                        df_copy = df.copy() # Work on copy
                        df_copy['SMILES'] = df_copy['SMILES'].fillna('')
                        preprocessed_df = embedding.preprocess(df_copy)
                        embedded_df = embedding.get_embedding(preprocessed_df, fit=fit)
                    else:
                        preprocessed_df = embedding.preprocess(df)
                        embedded_df = embedding.get_embedding(preprocessed_df, fit=fit)
            else:
                preprocessed_df = embedding.preprocess(df)
                embedded_df = embedding.get_embedding(preprocessed_df, fit=fit)


            processed_dfs.append(embedded_df)


            self.embedding_indices[embedding_name] = {
                'start': current_index,
                'end': current_index + embedding.embedding_size,
                'columns': list(embedded_df.columns)  # Store column names
            }
            current_index += embedding.embedding_size

        self.latest_processed_data = pd.concat(processed_dfs, axis=1)
        return self.latest_processed_data
    def get_entity_embeddings(self, cell_type=None, sm_name=None):
        """
        Retrieves combined embeddings for a given cell type and/or small molecule.

        Args:
            cell_type (str, optional): The cell type.
            sm_name (str, optional): The small molecule name.

        Returns:
            pd.DataFrame:  Combined embeddings.
            dict: Individual embeddings from each embedding type.
        """

        if cell_type is None and sm_name is None:
            raise ValueError("At least one of cell_type or sm_name must be provided")

        individual_embeddings = {}
        all_values = []
        all_columns = []

        for embedding in self.embeddings:
            embedding_name = embedding.__class__.__name__
            embedding_info = self.embedding_indices[embedding_name]

            if not hasattr(embedding, 'get_entity_embedding'):
                # Handle embeddings *without* get_entity_embedding
                zeros = np.zeros(embedding.embedding_size)
                all_values.append(zeros)
                all_columns.extend(embedding_info['columns'])
                continue
            
            entity_embedding = None

            # TargetEmbedding (handles both cell_type and sm_name)
            if hasattr(embedding, 'cell_type_dict') and cell_type is not None:
                key = f"cell_type_{cell_type}"
                ct_embedding = embedding.get_entity_embedding(key)
                if ct_embedding is not None:  # Check for None
                   individual_embeddings[f"{embedding_name}_cell_type"] = ct_embedding
                   entity_embedding = ct_embedding

            if hasattr(embedding, 'sm_dict') and sm_name is not None:
                key = f"sm_name_{sm_name}"
                sm_embedding = embedding.get_entity_embedding(key)
                if sm_embedding is not None: # Check for None
                    individual_embeddings[f"{embedding_name}_sm"] = sm_embedding
                    if entity_embedding is None:
                        entity_embedding = sm_embedding
                    else:
                        entity_embedding = np.concatenate([entity_embedding, sm_embedding])
            
            #SMILES Based Embeddings
            smiles_based_embeddings = [
                'SMILESEmbedding', 'MorganFingerPrintEmbedding', 'Mol2VecEmbedding',
                'ChemBERTaEmbedding','MolformerEmbedding','SmoleBartEmbedding','SelfFormerEmbedding'
            ]
            if sm_name is not None and embedding_name in smiles_based_embeddings:
                if sm_name in self.sm_name_to_smiles:
                    smiles = self.sm_name_to_smiles[sm_name]
                    entity_embedding = embedding.get_entity_embedding(smiles)
                    if entity_embedding is not None:  # Check for None
                        individual_embeddings[embedding_name] = entity_embedding
                else:  # Handle missing SMILES
                    entity_embedding = np.zeros(embedding.embedding_size)
                    individual_embeddings[embedding_name] = entity_embedding
                    print(f"Warning:  SMILES not found for {sm_name} in {embedding_name}. Using zero embedding.")
            
            # Fill with zeros if no embedding was found
            if entity_embedding is None:
                entity_embedding = np.zeros(embedding.embedding_size)
                print(f"Using zero embedding for {cell_type}/{sm_name} in {embedding_name}")

            all_values.append(entity_embedding)
            all_columns.extend(embedding_info['columns'])

        combined_vector = np.concatenate(all_values)
        result_df = pd.DataFrame([combined_vector], columns=all_columns)
        return result_df, individual_embeddings
    def generate_expert_config(self, expert_specs):
        """
        Generate expert configuration based on embedding combinations.

        Args:
            expert_specs (dict): Dictionary mapping expert names to lists of embedding names

        Returns:
            dict: Expert configuration with corresponding feature indices
        """
        expert_config = {}
        for expert, embedding_names in expert_specs.items():
            indices = []
            for embedding_name in embedding_names:
                if embedding_name in self.embedding_indices:
                    indices.extend(range(
                        self.embedding_indices[embedding_name]['start'],
                        self.embedding_indices[embedding_name]['end']
                    ))
                else:
                    raise KeyError(f"Embedding '{embedding_name}' not found in preprocessor")
            expert_config[expert] = indices
        return expert_config

    def get_embedding_info(self):
        """
        Get information about available embeddings and their ranges.

        Returns:
            dict: Dictionary containing embedding information including column names
        """
        return {
            embedding_name: {
                'size': self.embedding_indices[embedding_name]['end'] -
                       self.embedding_indices[embedding_name]['start'],
                'start_index': self.embedding_indices[embedding_name]['start'],
                'end_index': self.embedding_indices[embedding_name]['end'],
                'columns': self.embedding_indices[embedding_name]['columns']
            }
            for embedding_name in self.embedding_indices
        }

    def get_smiles_mapping(self):
        """
        Get the dictionary mapping sm_names to SMILES strings.

        Returns:
            dict: Dictionary containing sm_name to SMILES mapping
        """
        return self.sm_name_to_smiles.copy()


# onehot_embedding = OneHotEmbedding(columns=['cell_type', 'sm_name'])
target_embedding = TargetEmbedding(df)
smiles_embedding = SMILESEmbedding()
mol2vec_embedding = Mol2VecEmbedding('/kaggle/input/mol2vec/pytorch/default/1/model_300dim.pkl')
morgan_embedding = MorganFingerPrintEmbedding(hidden_size=256)
chemberta_embedding = ChemBERTaEmbedding(embedding_type='mean_pooling', padding=False)
molformer_embedding = MolformerEmbedding(model_path='ibm/MoLFormer-XL-both-10pct', padding=True)
smolebart_embedding = SmoleBartEmbedding(model_path="UdS-LSV/smole-bart")
model_path = "/kaggle/input/selfformer/transformers/default/1"

selformer_embedding = SelfFormerEmbedding(
    model_name=model_path, 
    device="cuda",  # Ensures GPU usage
    batch_size=16
)




preprocessor_train = Preprocessor([target_embedding, smiles_embedding, mol2vec_embedding, morgan_embedding,
    chemberta_embedding, molformer_embedding,smolebart_embedding,selformer_embedding ])


target_cols = ['cell_type','sm_name','sm_lincs_id','SMILES','control','SELFIES']
targets = df.drop(columns=target_cols)
processed_df_train = preprocessor_train.preprocess(df, fit=True)


df


df


target_cols = ['sm_lincs_id','SMILES','control','SELFIES']
targets = df
targets.drop(columns=target_cols, inplace=True)
targets.set_index(["cell_type", "sm_name"], inplace=True)
# Select top 256 variable genes
top_genes = select_top_variable_genes(targets, k=256, exclude_controls=True)
targets = targets[top_genes]  # Keep only selected genes


processed_df_train.head()


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV

# Prepare data
X = processed_df_train
y = targets.values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dimensionality reduction on the target variable y
tsvd = TruncatedSVD(n_components=50)
y_train_reduced = tsvd.fit_transform(y_train)
y_test_reduced = tsvd.transform(y_test)

 # Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dimensionality reduction on the target variable y 
tsvd = TruncatedSVD(n_components=50)
y_train_reduced = tsvd.fit_transform(y_train)
y_test_reduced = tsvd.transform(y_test)

#best parameters using grid Search 
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor(
        max_depth=7,
        min_samples_split=10
    ),
    'K-Nearest Neighbors Regressor': KNeighborsRegressor(
        n_neighbors=7,
        weights='uniform'
    ),
    'Neural Network Regressor': MLPRegressor(
        hidden_layer_sizes=(64, 32),
        alpha=0.01,
        max_iter=1000
    )
}

# Train and evaluate models
model_results = {}
for name, model in models.items():
    print(f"Training {name}...")
    
    # Fit the model
    model.fit(X_train, y_train_reduced)
    
    # Make predictions
    train_preds_reduced = model.predict(X_train)
    test_preds_reduced = model.predict(X_test)
    
    # Inverse transform the predictions to get the original scale
    train_preds = tsvd.inverse_transform(train_preds_reduced)
    test_preds = tsvd.inverse_transform(test_preds_reduced)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, train_preds)
    test_mse = mean_squared_error(y_test, test_preds)
    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)
    
    model_results[name] = {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'model': model  # store the trained model
    }
    
    print(f"{name} Results:")
    print(f"Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
    print(f"Train R^2: {train_r2:.4f}, Test R^2: {test_r2:.4f}")




