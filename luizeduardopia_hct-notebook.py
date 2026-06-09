import functools
import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import List

from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

from lifelines import KaplanMeierFitter, NelsonAalenFitter
from lifelines.utils import concordance_index

from pytorch_lightning.cli import ReduceLROnPlateau
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar, StochasticWeightAveraging
from pytorch_lightning.utilities import grad_norm

from pytorch_tabular.models.common.layers import ODST
from torch import nn
from torch.utils.data import TensorDataset

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')


def preprocess_data(train, val, transformers):
    """Preprocessing that safely handles missing flags"""
    scaler = StandardScaler()
    
    # Get categorical and numerical columns
    categorical_cols, numerical = get_feature_types(train)
    
    _, X_cat_train = get_X_cat(train, categorical_cols, transformers)
    _, X_cat_val = get_X_cat(val, categorical_cols, transformers)
    
    # Scale numerical features (filling NaNs with mean for scaling)
    X_num_train_filled = train[numerical].fillna(train[numerical].mean())
    X_num_val_filled = val[numerical].fillna(train[numerical].mean())
    
    X_num_train = scaler.fit_transform(X_num_train_filled)
    X_num_val = scaler.transform(X_num_val_filled)
    
    # Get missing flags
    train_missing = train[numerical].isna().astype(float)
    val_missing = val[numerical].isna().astype(float)
    
    # Modify missing flags to prevent all-zeros by adding tiny noise
    for col in train_missing.columns:
        if (train_missing[col] == 0).all():
            train_missing[col] += np.random.uniform(0.001, 0.01, size=len(train_missing))
        if (val_missing[col] == 0).all():
            val_missing[col] += np.random.uniform(0.001, 0.01, size=len(val_missing))
    
    # Create dataloaders
    dl_train = init_dl(X_cat_train, X_num_train, train_missing.values, train, training=True)
    dl_val = init_dl(X_cat_val, X_num_val, val_missing.values, val, training=False)
    
    return X_cat_val, X_num_train, X_num_val, train_missing.values, val_missing.values, dl_train, dl_val, transformers

def init_dl(X_cat, X_num, X_flags, df, training=False):
    """Dataloader that keeps features and missing flags separate"""
    ds_train = TensorDataset(
        torch.tensor(X_cat, dtype=torch.long),
        torch.tensor(X_num, dtype=torch.float32),
        torch.tensor(X_flags, dtype=torch.float32),
        torch.tensor(df.efs_time.values, dtype=torch.float32).log(),
        torch.tensor(df.efs.values, dtype=torch.long)
    )
    bs = 2048
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=bs, pin_memory=True, shuffle=training, num_workers=4, persistent_workers=True)
    return dl_train

@functools.lru_cache(maxsize=128)
def combinations(N):
    ind = torch.arange(N)
    comb = torch.combinations(ind, r=2)
    return comb.cuda()  

def feature_engineering(df):
    # Create additional features
    sex_match = df['sex_match'].str.split("-").str[0] == df['sex_match'].str.split("-").str[1]
    df['sex_match_bool'] = sex_match
    
    tce_imm_match = df['tce_imm_match'].str.split('/').str[0] == df['tce_imm_match'].str.split('/').str[1]
    df['tce_imm_match_bool'] = tce_imm_match
   
    # Expand disease risk index to more descriptive features and label encode
    df['missing_cytogenetics'] = df['dri_score'].str.contains('missing cytogenetics', na=False)
    df['unknown_disease'] = df['dri_score'].str.contains('disease not classifiable', na=False)
    df['dri_n/a'] = df['dri_score'].str.contains('N/A', na=False)
    df['non_malignant'] = df['dri_score'].str.contains('non-malignant indication', na=False)
    df['pediatric'] = df['dri_score'].str.contains('pediatric', na=False)
    df['TBD_cytogenetics'] = df['dri_score'].str.contains('TBD cytogenetics', na=False)
    df['missing_status'] = df['dri_score'].str.contains('Missing disease status', na=False)
    
    def dri_to_ordinal(x):
        if pd.isna(x):
            return np.nan
        elif 'Low' in str(x) and 'Very' not in str(x):
            return 4
        elif 'Intermediate' in str(x):
            return 3
        elif 'High' in str(x) and 'Very' not in str(x):
            return 2
        elif 'Very high' in str(x):
            return 1
        else:
            return 0
    df['dri_ordinal'] = df['dri_score'].apply(dri_to_ordinal)
    
    # Expand conditioning intensity, the level of preparation the patient underwent
    conditioning_mapping = {
            'MAC': 5,  # Highest intensity
            'RIC': 4,  # Moderate intensity
            'NMA': 1,  # Lowest intensity
            'TBD': 2,  # Unknown classification
            'No drugs reported': 2,
            'N/A, F(pre-TED) not submitted': 2
        }
    df['conditioning_intensity_numeric'] = df['conditioning_intensity'].map(conditioning_mapping)
    
    # Label encode cytological score
    cyto_score_mapping = {
            'Favorable': 8,
            'Intermediate': 6,
            'Poor': 1,
            'Normal': 4,
            'Other': 2,
            'TBD': 2,
            'Not tested': 2,
            None: np.nan,  
            float('nan'): np.nan
        }
    df['cyto_score_numeric'] = df['cyto_score'].map(cyto_score_mapping)
    
    # Label encode details in cytological score
    cyto_score_detail_mapping = {
            'Favorable': 6,
            'Intermediate': 4,
            'Poor': 1,
            'TBD': 2,
            'Not tested': 2,
            None: np.nan,  
            float('nan'): np.nan
        }
    df['cyto_score_detail_numeric'] = df['cyto_score_detail'].map(cyto_score_detail_mapping)
    
    # Pulmonary issues
    pulm_severe_mapping = {
            'Yes': 1,
            'No': 4,
            "Not done": 2,
            float('nan'): 2
        }
    df['pulm_severe_numeric'] = df['pulm_severe'].map(pulm_severe_mapping)
    
    # Disease that caused the need for HCT
    primary_disease_mapping = {
            'SAA': 9,  # Severe Aplastic Anemia (often has good transplant outcomes)
            'AI': 8,   # Autoimmune diseases (some have good prognosis post-HCT)
            'HD': 7,   # Hodgkin's Disease (better prognosis)
            'NHL': 6,  # Non-Hodgkin's Lymphoma
            'CML': 5,  # Chronic Myeloid Leukemia (often controlled with TKIs)
            'PCD': 5,  # Plasma Cell Disorders
            'MPN': 5,  # Myeloproliferative Neoplasms
            'IMD': 4,  # Inherited Metabolic Disorders
            'MDS': 4,  # Myelodysplastic Syndromes
            'AML': 3,  # Acute Myeloid Leukemia (worse prognosis)
            'ALL': 3,  # Acute Lymphoblastic Leukemia (depends on risk factors)
            'Other leukemia': 3,  
            'Other acute leukemia': 3,
            'Solid tumor': 2,  # Worse transplant success in solid tumors
            'HIS': 2,   # Histiocytic disorders
            'IPA': 2,   # Inherited Primary Immunodeficiencies
            'IIS': 2,   # Inherited Immune System disorders
            'IEA': 2,   # Inherited Erythroid Aplasia
            'TBD': 3,   # Use a somewhat average value for missing or invalid values
            'Other': 3,  
            None: np.nan,
            float('nan'): np.nan
        }
    df['primary_disease_numeric'] = df['prim_disease_hct'].map(primary_disease_mapping)

    
    # Bin age of patient
    df['newborn'] = df.age_at_hct == 0.044
    df['age_bin'] = pd.cut(df.age_at_hct, [0, 1, 16, 30, 50, 100])
    df['senior'] = df.age_at_hct > 60
    df['age_ts'] = df.age_at_hct / df.donor_age
    df['age_bin'] = df['age_bin'].astype(str)
        
    # Year related features with the rationale that more modern transplants have better odds
    df['years_since_2007'] = df['year_hct'] - 2007
    df['hla_match_total_year']=df['hla_high_res_10']*(df['years_since_2007'])

    # Interaction of other columns
    df['karnofsky_comorbidity']=df['karnofsky_score']*df['comorbidity_score']
    df['age_interaction']=df['age_at_hct']/df['donor_age']
    df['year_total_match']=(df['years_since_2007'])*df['hla_high_res_10']
    
    # Interaction terms to capture the effect of medical advancements on key features
    df['years_x_dri'] = df['years_since_2007'] * df['dri_ordinal']
    df['years_x_conditioning'] = df['years_since_2007'] * df['conditioning_intensity_numeric']
    df['years_x_cyto'] = df['years_since_2007'] * df['cyto_score_numeric']
    df['years_x_primary_disease'] = df['years_since_2007'] * df['primary_disease_numeric']
    
    # Non linear features
    df['log_years_x_dri'] = np.log1p(df['years_x_dri'])
    df['log_years_x_conditioning'] = np.log1p(df['years_x_conditioning'])
    df['log_years_x_cyto'] = np.log1p(df['years_x_cyto'])
    
    # Ratio-based features
    df['dri_to_years_ratio'] = df['dri_ordinal'] / (df['years_since_2007'])
    df['cyto_to_years_ratio'] = df['cyto_score_numeric'] / (df['years_since_2007'])
    df['primary_disease_to_years_ratio'] = df['primary_disease_numeric'] / (df['years_since_2007'])
    
    # Exponential features to capture non-linear trends
    df['exp_years_x_dri'] = np.exp(df['years_x_dri'])
    df['exp_years_x_cyto'] = np.exp(df['years_x_cyto'])
    
    # Label encoding immunosuppressive prophylaxis
    t_cell_depletion = ['TDEPLETION alone', 'TDEPLETION +- other', 'CDselect alone', 'CDselect +- other']
    cyclophosphamide_based = ['Cyclophosphamide alone', 'Cyclophosphamide +- others']
    calcineurin_inhibitors = ['FKalone', 'CSA alone', 'FK+- others(not MMF,MTX)', 'CSA +- others(not FK,MMF,MTX)']
    combo_regimens = ['FK+ MMF +- others', 'CSA + MMF +- others(not FK)', 'CSA + MTX +- others(not MMF,FK)', 'FK+ MTX +- others(not MMF)']
    no_prophylaxis = ['No GvHD Prophylaxis', 'Parent Q = yes, but no agent']

    def categorize_gvhd(prophylaxis):
        if pd.isna(prophylaxis):
            return np.nan
        elif prophylaxis in t_cell_depletion:
            return 5
        elif prophylaxis in cyclophosphamide_based:
            return 4
        elif prophylaxis in calcineurin_inhibitors:
            return 2
        elif prophylaxis in combo_regimens:
            return 3
        elif prophylaxis in no_prophylaxis:
            return 1
        else:
            return 2

    df['GVHD_Risk_Score'] = df['gvhd_proph'].apply(categorize_gvhd)

    # CMV Risk - define categories
    cmv_risk_levels = {
        '+/+': 3,  # Both donor and recipient CMV+
        '+/-': 2,  # Donor CMV+ but recipient CMV-
        '-/+': 2,  # Donor CMV-, recipient CMV+
        '-/-': 1,  # Both donor and recipient CMV-
        None: np.nan # Missing values
    }
    # Create new columns
    df['CMV_Risk_Score'] = df['cmv_status'].map(cmv_risk_levels)

    return df

def is_categorical_column(df, col):
    # Consider a column categorical if it has a small number of unique values (<25) or if it's object type
    return (df[col].dtype == "object") or (2 < df[col].nunique() < 25)

def get_feature_types(train):
    categorical_cols = [col for i, col in enumerate(train.columns) if is_categorical_column(train, col)]
    RMV = ["ID", "efs", "efs_time"]
    FEATURES = [c for c in train.columns if not c in RMV]
    numerical = [i for i in FEATURES if i not in categorical_cols]
    return categorical_cols, numerical

# Class that stores category values 
class IdentityTransformer:
    def __init__(self, classes_):
        self.classes_ = classes_
    def transform(self, x):
        return x.fillna(0).astype(int)  # Missing data that is filled with 0 will still be treated as NaN by the NN thanks to the missing-flags

# Function that prepares categorical data 
def get_X_cat(df, cat_cols, transformers=None):
    if not cat_cols:
        if transformers is None:
            transformers = []
        return transformers, np.zeros((len(df), 0))
    
    if transformers is None:
        transformers = []
        for col in cat_cols:
            # Handle interval columns
            if pd.api.types.is_interval_dtype(df[col]) or isinstance(df[col].iloc[0], pd.Interval):
                df[col] = df[col].astype(str)
            
            # For numeric categorical columns
            if df[col].dtype.kind in 'ifc':
                unique_vals = sorted(df[col].dropna().unique())
                transformers.append(IdentityTransformer(unique_vals))
            else:
                # For categorical columns
                le = LabelEncoder()
                if pd.api.types.is_categorical_dtype(df[col]):
                    all_values = list(df[col].cat.categories) + ['__unknown__', '__missing__']
                else:
                    # Add special missing value category
                    non_missing_values = df[col].dropna().astype(str).unique()
                    all_values = list(non_missing_values) + ['__unknown__', '__missing__']
                
                le.fit(all_values)
                transformers.append(le)
    
    # Transform data with special handling for missingness
    transformed_cols = []
    for i, (col, transformer) in enumerate(zip(cat_cols, transformers)):
        series = df[col].copy()
        missing_mask = series.isna()
        
        if df[col].dtype.kind in 'ifc':
            # For numeric categories, use a special index (e.g. 0) for missing values
            # Missing mask is used to track which values were actually missing
            transformed = np.zeros(len(series))
            transformed[~missing_mask] = series[~missing_mask].astype(int).values
            # Keep 0 for missing - the model will handle them via the mask
        else:
            # For string/object categories, use a special "__missing__" category
            non_missing_series = series[~missing_mask].astype(str)
            if hasattr(transformer, 'classes_') and not isinstance(transformer, IdentityTransformer):
                unknown_mask = ~non_missing_series.isin(transformer.classes_)
                if unknown_mask.any():
                    non_missing_series[unknown_mask] = '__unknown__'
            
            transformed = np.zeros(len(series), dtype=int)  # Initialize with zeros
            transformed[~missing_mask] = transformer.transform(non_missing_series)
            
            # For missing values, find the index of "__missing__" in the classes
            if '__missing__' in transformer.classes_:
                missing_idx = np.where(transformer.classes_ == '__missing__')[0][0]
                transformed[missing_mask] = missing_idx
        
        transformed_cols.append(transformed)
    
    if not transformed_cols:
        return transformers, np.zeros((len(df), 0))
    
    return transformers, np.array(transformed_cols).T


def get_categoricals(train, val):
    categorical_cols, numerical = get_feature_types(train)
    
    # Handle the case where there are no categorical columns
    if not categorical_cols:
        return np.zeros((len(train), 0)), np.zeros((len(val), 0)), numerical, []
    
    remove = []
    for col in categorical_cols:
        # Remove columns with only one value
        if train[col].nunique() == 1:
            remove.append(col)
            
    categorical_cols = [col for col in categorical_cols if col not in remove]
    
    # If no categorical columns remain after filtering
    if not categorical_cols:
        return np.zeros((len(train), 0)), np.zeros((len(val), 0)), numerical, []
    
    transformers, X_cat_train = get_X_cat(train, categorical_cols)
    _, X_cat_val = get_X_cat(val, categorical_cols, transformers)
    
    return X_cat_train, X_cat_val, numerical, transformers

def create_leak_proof_transformers(train_df, test_df, categorical_cols):
    """
    Create transformers that know about all categories in both train and test data
    Function used to ensure the code will run with the private Kaggle dataset even if the code encounter unseen data
    """
    transformers = []
    for col in categorical_cols:
        # Handle interval columns
        if pd.api.types.is_interval_dtype(train_df[col]) or (len(train_df[col]) > 0 and isinstance(train_df[col].iloc[0], pd.Interval)):
            train_df[col] = train_df[col].astype(str)
            if col in test_df.columns:
                test_df[col] = test_df[col].astype(str)
        
        # For numeric categorical columns
        if train_df[col].dtype.kind in 'ifc':
            # Combine unique values from both train and test
            train_unique = set(train_df[col].dropna().unique())
            test_unique = set(test_df[col].dropna().unique()) if col in test_df.columns else set()
            all_unique = sorted(train_unique.union(test_unique))
            transformers.append(IdentityTransformer(all_unique))
        else:
            # For string/categorical columns
            le = LabelEncoder()
            
            # Get unique values from both train and test
            train_values = set(train_df[col].dropna().astype(str).unique())
            test_values = set(test_df[col].dropna().astype(str).unique()) if col in test_df.columns else set()
            all_values = list(train_values.union(test_values)) + ['__unknown__', '__missing__']
            
            le.fit(all_values)
            transformers.append(le)
    
    return transformers


# CatEmbeddings: A module for handling categorical features
class CatEmbeddings(nn.Module):
    def __init__(self, projection_dim: int, categorical_cardinality: List[int], embedding_dim: int):
        super(CatEmbeddings, self).__init__()
        # Create an embedding layer for each categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(max(1, cardinality), embedding_dim)  # Ensure minimum cardinality of 1 to avoid errors
            for cardinality in categorical_cardinality
        ])
        # Project the concatenated embeddings to the desired output dimension
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim * len(categorical_cardinality), projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim)
        )
        self.num_embeddings = len(categorical_cardinality)

    def forward(self, x_cat):
        # Handle the case when there are no categorical features
        if x_cat.shape[1] == 0 or self.num_embeddings == 0:
            batch_size = x_cat.shape[0]
            # Return zeros with appropriate shape when no categorical features exist
            return torch.zeros(batch_size, self.projection[0].out_features, device=x_cat.device)
            
        # Process each categorical feature through its corresponding embedding layer
        embedded_cols = []
        for i, embedding in enumerate(self.embeddings):
            if i < x_cat.shape[1]:  # Only process columns that exist
                # Ensure indices are valid (within embedding range)
                valid_indices = torch.clamp(x_cat[:, i], 0, embedding.num_embeddings - 1)
                embedded_cols.append(embedding(valid_indices))
            else:
                # For missing columns, add zeros with proper dimensions
                embedded_cols.append(torch.zeros(x_cat.shape[0], embedding.embedding_dim, device=x_cat.device))
                
        # Concatenate all embeddings along feature dimension
        x_cat = torch.cat(embedded_cols, dim=1)
        # Project concatenated embeddings to final representation
        return self.projection(x_cat)


# NNWithMissingFlags: Neural network that handles missing values explicitly
class NNWithMissingFlags(nn.Module):
    def __init__(
            self,
            continuous_dim: int,             # Number of continuous features
            categorical_cardinality: List[int],  # Cardinality of each categorical feature
            embedding_dim: int,              # Dimension for categorical embeddings
            projection_dim: int,             # Dimension after projection
            hidden_dim: int,                 # Dimension of hidden layers
            dropout: float = 0.2             # Dropout rate for regularization
    ):
        super().__init__()
        
        # Handle categorical features with embeddings
        if categorical_cardinality:
            self.embeddings = CatEmbeddings(projection_dim, categorical_cardinality, embedding_dim)
            self.has_categorical = True
        else:
            # Create dummy embeddings if no categorical features
            self.embeddings = CatEmbeddings(projection_dim, [1], embedding_dim)
            self.has_categorical = False
        
        # Calculate input dimension: projected categorical + continuous features + missing flags
        # Missing flags are binary indicators for each continuous feature
        total_input_dim = projection_dim + continuous_dim * 2
        
        # Input layer with batch normalization and dropout for regularization
        self.input_layer = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),  # Dense layer
            nn.GELU(),                               # Activation function
            nn.BatchNorm1d(hidden_dim),              # Batch normalization helps training stability
            nn.Dropout(dropout)                      # Dropout for regularization
        )
        
        # ODST (Oblivious Decision Stumps with Trees) layers with residual connections
        # These are specialized layers for tabular data
        self.odst1 = ODST(hidden_dim, hidden_dim)  # First ODST layer
        self.bn1 = nn.BatchNorm1d(hidden_dim)      # Batch normalization
        self.dropout1 = nn.Dropout(dropout)        # Dropout
        
        self.odst2 = ODST(hidden_dim, hidden_dim)  # Second ODST layer
        self.bn2 = nn.BatchNorm1d(hidden_dim)      # Batch normalization
        self.dropout2 = nn.Dropout(dropout)        # Dropout
                
        # Output layer for predictions (single value)
        self.out = nn.Linear(hidden_dim, 1)
        # Global dropout applied to all inputs
        self.global_dropout = nn.Dropout(dropout)

    def forward(self, x_cat, x_cont_with_flags):
        batch_size = x_cont_with_flags.shape[0]
        
        # Process categorical data if available
        if self.has_categorical:
            cat_emb = self.embeddings(x_cat)  # Get embeddings for categorical features
        else:
            # Create empty tensor with correct dimensions if no categorical data
            cat_emb = torch.zeros(batch_size, 
                                 self.input_layer[0].in_features - x_cont_with_flags.shape[1], 
                                 device=x_cont_with_flags.device)
        
        # Concatenate categorical embeddings with numerical features (including missing flags)
        x = torch.cat([cat_emb, x_cont_with_flags], dim=1)
        
        # Apply global dropout to all inputs
        x = self.global_dropout(x)
        
        # Initial processing through input layer
        x = self.input_layer(x)
        
        # Residual block 1 - residual connections help with gradient flow
        res = x
        x = self.odst1(x)       # Apply ODST layer
        x = self.bn1(x)         # Batch normalization
        x = self.dropout1(x)    # Dropout
        x = x + res             # Residual connection (add original input)
        
        # Residual block 2
        res = x
        x = self.odst2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = x + res
        
        # Return both the prediction and the final embeddings
        # Squeeze removes dimension of size 1 (converts [batch_size, 1] to [batch_size])
        return self.out(x).squeeze(1), x


# TabTransformer: Transformer-based model for tabular data
class TabTransformer(nn.Module):
    def __init__(
            self,
            continuous_dim,              # Number of continuous features
            categorical_cardinality,     # Cardinality of each categorical feature
            embedding_dim,               # Dimension for categorical embeddings
            depth=2,                     # Number of transformer layers
            heads=4,                     # Number of attention heads
            dim=128,                     # Internal dimension for transformer
            dropout=0.2                  # Dropout rate
    ):
        super().__init__()
        
        # Handle categorical embeddings
        if categorical_cardinality:
            self.cat_embeddings = CatEmbeddings(dim, categorical_cardinality, embedding_dim)
            self.has_categorical = True
        else:
            self.cat_embeddings = None
            self.has_categorical = False
        
        # Continuous features projection - processes each feature individually
        self.cont_projection = nn.Sequential(
            nn.Linear(1, dim),          # Project single feature to dimension
            nn.LayerNorm(dim),          # Layer normalization
            nn.GELU()                   # Activation function
        )
        
        # Learnable embedding for missing values
        self.missing_embed = nn.Parameter(torch.randn(1, dim))
        
        # Transformer encoder for feature interactions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,                # Model dimension
            nhead=heads,                # Number of attention heads
            dim_feedforward=dim*4,      # Feedforward network dimension
            dropout=dropout,            # Dropout rate
            activation='gelu',          # Activation function
            batch_first=True,           # Batch is first dimension
            norm_first=True             # Apply normalization first
        )
        # Stack multiple transformer encoder layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Output processing
        self.output_norm = nn.LayerNorm(dim)  # Final normalization
        self.output = nn.Linear(dim, 1)       # Output projection to single value
    
    def forward(self, x_cat, x_cont):
        batch_size = x_cont.shape[0]
        
        # Process categorical features if available
        if self.has_categorical:
            cat_embed = self.cat_embeddings(x_cat).unsqueeze(1)  # [B, 1, D] - add token dimension
        else:
            # Empty tensor with correct dimensions if no categorical features
            cat_embed = torch.zeros(batch_size, 0, self.missing_embed.shape[1], device=x_cont.device)
        
        # Handle missing values in continuous features
        cont_mask = torch.isnan(x_cont)  # Create mask for missing values (NaN)
        x_cont_clean = torch.nan_to_num(x_cont)  # Replace NaNs with zeros
        
        # Process each continuous feature individually
        feature_tokens = []
        if x_cont.shape[1] > 0:  # Only process if there are continuous features
            for i in range(x_cont.shape[1]):
                # Extract single feature
                feature = x_cont_clean[:, i:i+1]  # [B, 1]
                
                # Get missing mask for this feature
                missing = cont_mask[:, i:i+1]  # [B, 1]
                
                # Project feature to embedding space
                feat_embed = self.cont_projection(feature)  # [B, D]
                
                # Replace embeddings for missing values with learned embedding
                missing_expanded = missing.expand(-1, self.missing_embed.shape[1])
                feat_embed = torch.where(missing_expanded, self.missing_embed.expand(batch_size, -1), feat_embed)
                feature_tokens.append(feat_embed.unsqueeze(1))  # [B, 1, D]
        
        # Combine all tokens (categorical + continuous)
        all_tokens = torch.cat([cat_embed] + feature_tokens, dim=1)  # [B, N, D]
        
        # Apply transformer to model interactions between features
        trans_out = self.transformer(all_tokens)
        
        # Pool tokens (mean) to get fixed-size representation
        pooled = torch.mean(trans_out, dim=1)  # [B, D]
        
        # Final normalization and output projection
        out = self.output_norm(pooled)
        return self.output(out).squeeze(-1), pooled


# SurvivalHead: Output layer for survival analysis using Cox proportional hazards
class SurvivalHead(nn.Module):
    """Cox proportional hazards model implementation with NaN prevention"""
    def __init__(self, input_dim):
        super().__init__()
        # Linear layer to produce risk scores
        self.risk_score = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        # Return log hazard ratio (risk score)
        risk = self.risk_score(x).squeeze(-1)
        # Replace any NaNs with zeros (neutral risk) for numerical stability
        risk = torch.nan_to_num(risk, nan=0.0)
        return risk


# CombinedModel: Ensemble model that combines NN and Transformer approaches
class CombinedModel(nn.Module):
    def __init__(
            self,
            continuous_dim,
            categorical_cardinality,
            embedding_dim,
            projection_dim,
            hidden_dim,
            transformer_dim,
            transformer_depth,
            transformer_heads,
            dropout=0.2,
            fusion_factor=2,    # Factor to divide hidden_dim by for fusion layer output
            aux_factor=4,      # Factor to divide hidden_dim by for auxiliary layer
    ):
        super().__init__()
        
        # Store key dimensions
        self.hidden_dim = hidden_dim
        self.transformer_dim = transformer_dim
        self.fusion_output_dim = hidden_dim // fusion_factor
        self.aux_output_dim = hidden_dim // aux_factor
        
        # Initialize both models
        self.nn_model = NNWithMissingFlags(
            continuous_dim=continuous_dim,
            categorical_cardinality=categorical_cardinality,
            embedding_dim=embedding_dim,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        self.transformer_model = TabTransformer(
            continuous_dim=continuous_dim,
            categorical_cardinality=categorical_cardinality,
            embedding_dim=embedding_dim,
            dim=transformer_dim,
            depth=transformer_depth,
            heads=transformer_heads,
            dropout=dropout
        )
        
        # Feature fusion layer - calculate dimensions dynamically
        fusion_input_dim = hidden_dim + transformer_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.fusion_output_dim)
        )
        
        # Cox proportional hazards output
        self.output = SurvivalHead(self.fusion_output_dim)
        
        # Auxiliary task: Event prediction
        self.event_predictor = nn.Sequential(
            nn.Linear(self.fusion_output_dim, self.aux_output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.aux_output_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x_cat, x_cont, time_values=None):
        # Forward pass through NN model
        nn_pred, nn_emb = self.nn_model(x_cat, x_cont)
        
        # Forward pass through Transformer model
        trans_pred, trans_emb = self.transformer_model(x_cat, x_cont)
        
        # Simple concatenation with projection
        concat_emb = torch.cat([nn_emb, trans_emb], dim=1)
        
        # Feature fusion with concatenated embeddings
        fusion_emb = self.fusion(concat_emb)
        
        # Component predictions
        risk_pred = self.output(fusion_emb)
        event_prob = self.event_predictor(fusion_emb)
        
        # Ensemble prediction with weighted average
        alpha = 0.7  # Weight for fusion model prediction
        beta = 0.15  # Weight for NN model prediction
        gamma = 0.15  # Weight for transformer model prediction
        
        ensemble_risk_pred = alpha * risk_pred + beta * nn_pred + gamma * trans_pred
        
        return ensemble_risk_pred, event_prob, fusion_emb


# LitNN: PyTorch Lightning wrapper for training and evaluation
class LitNN(pl.LightningModule):
    def __init__(
            self,
            continuous_dim: int,
            categorical_cardinality: List[int],
            embedding_dim: int,
            projection_dim: int,
            hidden_dim: int,
            transformer_depth: int = 2,
            transformer_heads: int = 4,
            transformer_dim: int = 128,
            lr: float = 1e-3,
            dropout: float = 0.2,
            weight_decay: float = 1e-3,
            aux_weight: float = 0.1,
            cindex_weight: float = 0.3,
            event_weight: float = 0.2,
            margin: float = 0.5,
            race_index: int = 0,
            use_fusion: bool = True,
            l2_reg: float = 1e-4,
            fusion_factor: int = 2,
            aux_factor: int = 4
    ):
        super(LitNN, self).__init__()
        self.save_hyperparameters()
        
        # Initialize the improved model with dynamic dimensions
        self.model = CombinedModel(
            continuous_dim=self.hparams.continuous_dim,
            categorical_cardinality=self.hparams.categorical_cardinality,
            embedding_dim=self.hparams.embedding_dim,
            projection_dim=self.hparams.projection_dim,
            hidden_dim=self.hparams.hidden_dim,
            transformer_dim=self.hparams.transformer_dim,
            transformer_depth=self.hparams.transformer_depth,
            transformer_heads=self.hparams.transformer_heads,
            dropout=self.hparams.dropout,
            fusion_factor=self.hparams.fusion_factor,
            aux_factor=self.hparams.aux_factor
        )
        
        self.targets = []
        self.use_fusion = use_fusion
        self.l2_reg = l2_reg
        
        # Dynamic dimensions for auxiliary classifier
        fusion_output_dim = self.hparams.hidden_dim // self.hparams.fusion_factor
        aux_output_dim = fusion_output_dim // self.hparams.aux_factor
        
        # Auxiliary classifier uses the fusion embedding
        self.aux_cls = nn.Sequential(
            nn.Linear(fusion_output_dim, aux_output_dim),
            nn.GELU(),
            nn.Linear(aux_output_dim, 1)
        )
    
    def forward(self, x_cat, x_cont):
        # Pass inputs to the model
        return self.model(x_cat, x_cont)
    
    def training_step(self, batch, batch_idx):
        x_cat, x_cont, x_missing, y_time, efs = batch
        # Combine continuous features and missing flags
        x_cont_with_flags = torch.cat([x_cont, x_missing], dim=1)

        # Forward pass through the model
        risk_pred, event_prob, fusion_emb = self.model(x_cat, x_cont_with_flags, y_time)
        
        # Use risk prediction for survival modeling
        y_hat = risk_pred
        y = y_time
        
        # Calculate loss
        main_loss, race_loss = self.get_full_loss(efs, x_cat, y, y_hat)
        
        # Auxiliary regression loss
        aux_pred = self.aux_cls(fusion_emb).squeeze(1)
        aux_loss = nn.functional.mse_loss(aux_pred, y, reduction='none')
        aux_mask = efs == 1
        aux_loss = (aux_loss * aux_mask).sum() / (aux_mask.sum() + 1e-8)
        
        # Add binary classification loss for event prediction
        # Event probability should predict whether the event happens (efs == 1)
        event_loss = nn.functional.binary_cross_entropy(
            event_prob.squeeze(-1),  # Predicted probabilities
            efs.float(),             # Binary event indicators (0 or 1)
            reduction='mean'
        )
        
        # C-index loss
        races = x_cat[:, self.hparams.race_index]
        cindex_loss = self.cindex_for_training(races, y, y_hat)
        
        # Logging
        self.log("race_loss", race_loss, on_epoch=True, prog_bar=True, logger=True, on_step=False)
        self.log("train_loss", main_loss, on_epoch=True, prog_bar=True)
        self.log("event_loss", event_loss, on_epoch=True, prog_bar=True)
        self.log("cindex_loss", cindex_loss, on_epoch=True, prog_bar=True)
        
        # Combined loss - add event_loss with a weight
        event_weight = 0.2  # You can make this a hyperparameter
        total_loss = (
            main_loss + 
            aux_loss * self.hparams.aux_weight + 
            cindex_loss * self.hparams.cindex_weight +
            event_loss * event_weight
        )
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # Unpack the batch with missing flags
        x_cat, x_cont, x_missing, y_time, efs = batch
        
        # Concatenate numerical features with missing flags
        x_cont_with_flags = torch.cat([x_cont, x_missing], dim=1)
        
        # Forward pass
        risk_pred, event_prob, fusion_emb = self.model(x_cat, x_cont_with_flags, y_time)
        
        # Use risk prediction
        y_hat = risk_pred
        y = y_time
        
        # Calculate loss
        loss, race_loss = self.get_full_loss(efs, x_cat, y, y_hat)
        
        # Store predictions for metric calculation at epoch end
        self.targets.append([y, y_hat.detach(), efs, x_cat[:, self.hparams.race_index]])
        
        # Log validation loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        # Similar to validation_step
        x_cat, x_cont, x_missing, y_time, efs = batch
        x_cont_with_flags = torch.cat([x_cont, x_missing], dim=1)
        risk_pred, event_prob, fusion_emb = self.model(x_cat, x_cont_with_flags, y_time)
        y_hat = risk_pred
        y = y_time
        loss, race_loss = self.get_full_loss(efs, x_cat, y, y_hat)
        self.targets.append([y, y_hat.detach(), efs, x_cat[:, self.hparams.race_index]])
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def get_full_loss(self, efs, x_cat, y, y_hat):
        # Calculate main survival loss
        loss = self.calc_loss(y, y_hat, efs)
        
        # Only calculate race-specific loss during validation or periodically during training
        if not self.training or self.global_step % 5 == 0:
            # Get race-specific losses to ensure fairness across demographic groups
            race_loss = self.get_race_losses(efs, x_cat, y, y_hat)
            loss += 0.1 * race_loss  # Add weighted race loss to total loss
        else:
            race_loss = torch.tensor(0.0, device=y.device)
        
        return loss, race_loss
    
    def get_race_losses(self, efs, x_cat, y, y_hat):
        # Calculate loss separately for each race group
        races = torch.unique(x_cat[:, self.hparams.race_index])
        race_losses = []
        
        # Loop through each race group
        for race in races:
            # Create mask for this race
            ind = x_cat[:, self.hparams.race_index] == race
            # Calculate loss for this race group
            race_losses.append(self.calc_loss(y[ind], y_hat[ind], efs[ind]))
        
        # Calculate mean loss across races
        race_loss = sum(race_losses) / len(race_losses)
        # Calculate standard deviation of losses (for fairness)
        races_loss_std = sum((r - race_loss)**2 for r in race_losses) / len(race_losses)
        
        # Return square root of variance (standard deviation)
        return torch.sqrt(races_loss_std)
    
    def calc_loss(self, y, y_hat, efs):
        # Calculate pairwise ranking loss for survival analysis
        N = y.shape[0]
        device = y.device  # Get the device from input tensor
        
        # Get all possible pairs of indices
        comb = combinations(N)  # This is a helper function to generate all combinations
        if comb.device != device:
            comb = comb.to(device)  # Move to same device as data
        
        # Filter combinations where at least one event has occurred
        event_mask = (efs[comb[:, 0]] == 1) | (efs[comb[:, 1]] == 1)
        comb = comb[event_mask]
        
        # Extract predictions and targets for pairs
        pred_left = y_hat[comb[:, 0]]
        pred_right = y_hat[comb[:, 1]]
        y_left = y[comb[:, 0]]
        y_right = y[comb[:, 1]]
        
        # Calculate pairwise comparison targets (-1 or 1)
        # 1 if left should be ranked higher than right, -1 otherwise
        pair_targets = 2 * (y_left > y_right).int() - 1
        
        # Compute ranking loss with margin
        # This loss enforces correct ordering of risk predictions
        loss = nn.functional.relu(-pair_targets * (pred_left - pred_right) + self.hparams.margin)
        
        # Apply mask for valid comparisons
        mask = self.get_mask(comb, efs, y_left, y_right)
        
        # Calculate final loss
        numerator = (loss.double() * mask.double()).sum()
        denominator = mask.sum()
        
        # Handle edge case with empty mask
        if denominator == 0:
            return torch.tensor(0.0, device=device)
        
        return numerator / denominator
    
    def get_mask(self, comb, efs, y_left, y_right):
        # Create mask for valid comparisons in survival analysis
        # This handles censored data correctly
        
        # Case 1: left outlived right and left had event
        left_outlived = y_left >= y_right
        left_1_right_0 = (efs[comb[:, 0]] == 1) & (efs[comb[:, 1]] == 0)
        mask2 = (left_outlived & left_1_right_0)
        
        # Case 2: right outlived left and right had event
        right_outlived = y_right >= y_left
        right_1_left_0 = (efs[comb[:, 1]] == 1) & (efs[comb[:, 0]] == 0)
        mask2 |= (right_outlived & right_1_left_0)
        
        # Invert mask to get valid comparisons
        mask2 = ~mask2
        mask = mask2
        return mask
    
    def cindex_for_evaluation(self):
        # Gather stored predictions and targets
        y = torch.cat([t[0] for t in self.targets]).cpu().numpy()
        y_hat = torch.cat([t[1] for t in self.targets]).cpu().numpy()
        efs = torch.cat([t[2] for t in self.targets]).cpu().numpy()
        races = torch.cat([t[3] for t in self.targets]).cpu().numpy()
        
        # Handle potential NaN values
        mask = ~(np.isnan(y) | np.isnan(y_hat) | np.isnan(efs))
        if not mask.all():
            print(f"Removed {(~mask).sum()} NaN values from concordance calculation")
            y = y[mask]
            y_hat = y_hat[mask]
            efs = efs[mask]
            races = races[mask]
        
        metric = self._metric(efs, races, y, y_hat)
        cindex = concordance_index(y, y_hat, efs)
        return cindex, metric
    
    def _metric(self, efs, races, y, y_hat):
        metric_list = []
        for race in np.unique(races):
            race_mask = (races == race)
            y_ = y[race_mask]
            y_hat_ = y_hat[race_mask]
            efs_ = efs[race_mask]
            
            # Skip if not enough samples or contains NaNs
            if len(y_) < 2 or np.isnan(y_).any() or np.isnan(y_hat_).any() or np.isnan(efs_).any():
                continue
                
            metric_list.append(concordance_index(y_, y_hat_, efs_))
        
        if not metric_list:
            return 0.5  # Return default value if no valid race groups
            
        metric = float(np.mean(metric_list) - np.sqrt(np.var(metric_list)))
        return metric
    
    def cindex_for_training(self, races, y, pred):
        """Differentiable approximation of C-index with race variance penalty"""
        unique_races = torch.unique(races)
        race_cidx = []
        
        for race in unique_races:
            race_mask = (races == race)
            if race_mask.sum() < 2:  # Need at least 2 samples
                continue
            # Get all pairs of indices for this race
            indices = torch.where(race_mask)[0]
            n = indices.shape[0]
            pairs = torch.combinations(torch.arange(n))
            i, j = pairs[:, 0], pairs[:, 1]
            
            # Get corresponding predictions and targets
            pred_i = pred[indices[i]]
            pred_j = pred[indices[j]]
            y_i = y[indices[i]]
            y_j = y[indices[j]]
            
            # Calculate concordant pairs using sigmoid for smoothness
            concordant = torch.sigmoid(10.0 * (pred_j - pred_i) * torch.sign(y_j - y_i))
            race_cidx.append(torch.mean(concordant))
        
        # Competition metric: mean - sqrt(variance)
        race_scores = torch.stack(race_cidx)
        mean_score = torch.mean(race_scores)
        var_penalty = torch.sqrt(torch.var(race_scores))
        
        return 1.0 - (mean_score - var_penalty)
    
    def on_test_epoch_end(self) -> None:
        cindex, metric = self.cindex_for_evaluation()
        self.log("test_cindex", metric, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_cindex_simple", cindex, on_epoch=True, prog_bar=True, logger=True)
        self.targets.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW( # AdamW handles sparse gradients in tabular data with many categorical features better
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = {
            "scheduler": OneCycleLR( # Helps escape local minima early in training
                optimizer,
                max_lr=self.hparams.lr * 10,  # Peak learning rate
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.3,  # Spend 30% of training in warmup
                div_factor=25,  # Initial LR = max_lr/25
                final_div_factor=1000  # Final LR = max_lr/1000
            ),
            "interval": "step",
            "frequency": 1
        }
    
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

def train_final(X_num_train, dl_train, dl_val, transformers, categorical_cols, missing_flags_train, missing_flags_val, hparams=None):
                
    hparams = {
        "embedding_dim": 32,
        "projection_dim": 128,
        "hidden_dim": 64,
        "lr": 0.005,
        'transformer_dim': 128,
        "dropout": 0.2,
        "aux_weight": 0.3,
        "margin": 0.3,
        "weight_decay": 0.001,
        'transformer_depth':2,
        'transformer_heads':4
    }
    
    race_index = 0
    if categorical_cols and "race_group" in categorical_cols:
        race_index = categorical_cols.index("race_group")
    
    cardinality = []
    if transformers:
        cardinality = [len(t.classes_) for t in transformers]
    
    model = LitNN(
        continuous_dim=X_num_train.shape[1],
        categorical_cardinality=cardinality,
        race_index=race_index,
        **hparams
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="cindex", save_top_k=1)
    trainer = pl.Trainer(
    accelerator='gpu', 
    max_epochs=70,          
    callbacks=[
        checkpoint_callback,
        LearningRateMonitor(logging_interval='step'),
        TQDMProgressBar(),
        StochasticWeightAveraging(swa_lrs=1e-4, swa_epoch_start=55, annealing_epochs=15)
        # SWA is  effective for this survival prediction task because it helps stabilize performance across diverse subgroups
    ],
    gradient_clip_val=1.0,
    accumulate_grad_batches=4,  
    )
    trainer.fit(model, dl_train)
    trainer.test(model, dl_val)
    return model.eval()


train_original = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")

test['efs_time'] = np.nan
test['efs'] = np.nan

# Feature engineering
train_original=feature_engineering(train_original)
test=feature_engineering(test)

#Ensure both datasets have the same columns
train_columns = set(train_original.columns)
test_columns = set(test.columns)
missing_cols = train_columns - test_columns
for col in missing_cols:
    test[col] = np.nan
extra_cols = test_columns - train_columns
test.drop(columns=extra_cols, inplace=True)
test = test[train_original.columns]

train_original.set_index("ID", inplace=True)
test.set_index("ID", inplace=True)
test_pred = np.zeros(test.shape[0])

# Get feature types
categorical_cols, numerical = get_feature_types(train_original)

# Create transformers with knowledge of all categories
global_transformers = create_leak_proof_transformers(train_original, test, categorical_cols)

# Cross-validation
n_splits=5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
for i, (train_index, test_index) in enumerate(kf.split(train_original, train_original.race_group.astype(str))):
    print(f"Training fold {i+1}/{n_splits}")
    
    # Split data
    tt = train_original.copy()
    train = tt.iloc[train_index]
    val = tt.iloc[test_index]
    
    # Preprocess data - pass in global transformers
    X_cat_val, X_num_train, X_num_val, train_missing, val_missing, dl_train, dl_val, _ = preprocess_data(train, val, global_transformers.copy())

    # Train model
    model = train_final(X_num_train, dl_train, dl_val, global_transformers.copy(), categorical_cols=categorical_cols,
                    missing_flags_train=train_missing,
                    missing_flags_val=val_missing)
    
    # For test predictions, also use global transformers
    X_cat_val, X_num_train, X_num_val, train_missing, val_missing, dl_train, dl_val, _ = preprocess_data(train, test, global_transformers.copy())
    
    x_cont_with_flags = torch.cat([
        torch.tensor(X_num_val, dtype=torch.float32),
        torch.tensor(val_missing, dtype=torch.float32)
    ], dim=1).cuda()
    
    pred, _, _ = model.cuda().eval()(
        torch.tensor(X_cat_val, dtype=torch.long).cuda(),
        x_cont_with_flags
    )
    test_pred += pred.detach().cpu().numpy()


# Create submission
subm_data = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
subm_data['prediction'] = -test_pred / n_splits  # Average predictions across folds
subm_data.to_csv('submission.csv', index=False)

print("Submission file created successfully!")
display(subm_data.head())




