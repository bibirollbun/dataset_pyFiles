import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import random

# --- RDKit ---
try:
    from rdkit import Chem
    from rdkit import RDLogger
    # Disable RDKit logging to keep the output clean
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    print("RDKit library is required. Please install it.")
    exit()


# --- General Settings ---
SEED = 42
TARGET_COLUMNS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELS_OUTPUT_DIR = 'trained_models'

# --- Data Paths ---
# Note: Update these paths if your data is located elsewhere.
BASE_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/'
EXTRA_DATA_BASE_PATH = '/kaggle/input/'

TRAIN_CSV_PATH = os.path.join(BASE_PATH, 'train.csv')
TEST_CSV_PATH = os.path.join(BASE_PATH, 'test.csv')
EXTRA_DATA_PATHS = {
    'tc': os.path.join(EXTRA_DATA_BASE_PATH, 'tc-smiles/Tc_SMILES.csv'),
    'tg2': os.path.join(EXTRA_DATA_BASE_PATH, 'smiles-extra-data/JCIM_sup_bigsmiles.csv'),
    'tg3': os.path.join(EXTRA_DATA_BASE_PATH, 'smiles-extra-data/data_tg3.xlsx'),
    'dnst': os.path.join(EXTRA_DATA_BASE_PATH, 'smiles-extra-data/data_dnst1.xlsx')
}
# Path for loading models during inference
INFERENCE_MODELS_PATH = '/kaggle/working/trained_models/'


# --- Model Hyperparameters ---
SMILES_EMBEDDING_DIM = 256

# CNN branch
CNN_FILTERS = [128, 128, 128, 128, 128, 128]
CNN_KERNEL_SIZES = [5, 10, 15, 60, 70, 80]

# LSTM branch
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2

# Fully Connected layers
FC_LAYERS_SIZES = [512, 256]

# --- Training Hyperparameters ---
LEARNING_RATE = 0.001
BATCH_SIZE = 8
EPOCHS = 100
DROPOUT_RATE = 0.4
N_BINS_FOR_STRATIFY = 50 # For stratified splitting
N_TTA = 5 # Number of Test-Time Augmentations


def set_seed(seed_value=42):
    """Sets the seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def make_smile_canonical(s):
    """Converts a SMILES string to its canonical form."""
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True)
    except:
        return None

def add_extra_data(df_train, df_extra, target):
    """Merges external data into the main training dataframe."""
    n_samples_before = len(df_train[df_train[target].notnull()])
    
    df_extra['SMILES'] = df_extra['SMILES'].apply(lambda s: make_smile_canonical(s))
    df_extra = df_extra.dropna(subset=['SMILES', target])
    df_extra = df_extra.groupby('SMILES', as_index=False)[target].mean()
    
    cross_smiles = set(df_extra['SMILES']) & set(df_train['SMILES'])
    unique_smiles_extra = set(df_extra['SMILES']) - set(df_train['SMILES'])
    
    # Impute missing values in train_df with values from extra data
    for smile in cross_smiles:
        if pd.isnull(df_train.loc[df_train['SMILES']==smile, target]).any():
            impute_value = df_extra[df_extra['SMILES']==smile][target].values[0]
            df_train.loc[df_train['SMILES']==smile, target] = impute_value
            
    # Add new, unique SMILES from extra data
    df_train = pd.concat([df_train, df_extra[df_extra['SMILES'].isin(unique_smiles_extra)]], axis=0, ignore_index=True)
    
    n_samples_after = len(df_train[df_train[target].notnull()])
    print(f'For target "{target}": {n_samples_after - n_samples_before} new samples were added.')
    return df_train

# Set the seed for the entire notebook
set_seed(SEED)
print(f"Using device: {DEVICE}")


try:
    train_df = pd.read_csv(TRAIN_CSV_PATH)
except FileNotFoundError:
    print(f"Error: Training file not found at {TRAIN_CSV_PATH}")
    exit()

print("Starting the process of loading and merging additional data...\n")

try:
    # Add Tc data
    data_tc = pd.read_csv(EXTRA_DATA_PATHS['tc']).rename(columns={'TC_mean': 'Tc'})
    train_df = add_extra_data(train_df, data_tc, 'Tc')

    # Add Tg data (Source 2)
    data_tg2 = pd.read_csv(EXTRA_DATA_PATHS['tg2'], usecols=['SMILES', 'Tg (C)']).rename(columns={'Tg (C)': 'Tg'})
    train_df = add_extra_data(train_df, data_tg2, 'Tg')

    # Add Tg data (Source 3)
    data_tg3 = pd.read_excel(EXTRA_DATA_PATHS['tg3']).rename(columns={'Tg [K]': 'Tg'})
    data_tg3['Tg'] = data_tg3['Tg'] - 273.15  # Convert from Kelvin to Celsius
    train_df = add_extra_data(train_df, data_tg3, 'Tg')

    # Add Density data
    data_dnst = pd.read_excel(EXTRA_DATA_PATHS['dnst']).rename(columns={'density(g/cm3)': 'Density'})[['SMILES', 'Density']]
    data_dnst['SMILES'] = data_dnst['SMILES'].apply(lambda s: make_smile_canonical(s))
    data_dnst = data_dnst[(data_dnst['SMILES'].notnull()) & (data_dnst['Density'].notnull()) & (data_dnst['Density'] != 'nylon')]
    data_dnst['Density'] = data_dnst['Density'].astype('float64')
    data_dnst['Density'] -= 0.118 # Adjustment mentioned in problem context
    train_df = add_extra_data(train_df, data_dnst, 'Density')

except FileNotFoundError as e:
    print(f"\nError: Additional data file not found: {e.filename}")
    exit()

print('\n' + '--- Final number of samples for training ---')
for t in TARGET_COLUMNS:
    print(f'"{t}": {len(train_df[train_df[t].notnull()])}')
print('='*45 + '\n')


# Create vocabulary from all SMILES strings in the combined dataset
unique_smiles_chars = sorted(list(set(''.join(train_df['SMILES'].astype(str))) - set(['&', '!'])))

# Character to integer mapping
smiles_char_dict = {char: i + 1 for i, char in enumerate(unique_smiles_chars)}
smiles_char_dict['&'] = 0  # Padding token
smiles_char_dict['!'] = len(smiles_char_dict) # Unknown token

vocab_size = len(smiles_char_dict)

# Determine max length for padding, adding some buffer
max_smiles_len = train_df['SMILES'].astype(str).str.len().max() + 20

print(f"Vocabulary Size: {vocab_size}")
print(f"Max SMILES Length (with padding): {max_smiles_len}")


class PolymerDataset(Dataset):
    def __init__(self, smiles_list, targets, char_dict, max_len, is_train=False):
        self.smiles_list = smiles_list
        self.targets = targets
        self.char_dict = char_dict
        self.max_len = max_len
        self.pad_token_id = char_dict['&']
        self.unknown_token_id = char_dict['!']
        self.is_train = is_train

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        target = self.targets[idx]
        
        # SMILES augmentation for training
        if self.is_train:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                smiles = Chem.MolToSmiles(mol, doRandom=True)
        
        # Tokenize and pad
        tokenized_smiles = [self.char_dict.get(char, self.unknown_token_id) for char in smiles]
        padding_len = self.max_len - len(tokenized_smiles)
        padded_smiles = tokenized_smiles + [self.pad_token_id] * padding_len
        
        return (torch.tensor(padded_smiles[:self.max_len], dtype=torch.long),
                torch.tensor(target, dtype=torch.float))


class PolymerPredictor(nn.Module):
    def __init__(self, smiles_vocab_size, smiles_embedding_dim, cnn_filters, cnn_kernel_sizes,
                 lstm_hidden_size, lstm_num_layers, fc_layers_sizes, dropout_rate=0.25):
        super(PolymerPredictor, self).__init__()

        self.smiles_embedding = nn.Embedding(smiles_vocab_size, smiles_embedding_dim, padding_idx=smiles_char_dict['&'])
        
        # --- CNN Branch ---
        self.parallel_cnns = nn.ModuleList([
            nn.Conv1d(smiles_embedding_dim, out_ch, ks) for out_ch, ks in zip(cnn_filters, cnn_kernel_sizes)
        ])
        self.smi_pool = nn.AdaptiveMaxPool1d(1)
        
        # --- LSTM Branch ---
        self.lstm = nn.LSTM(
            input_size=smiles_embedding_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # --- Fully Connected (FC) Layers ---
        fc_input_size = sum(cnn_filters) + (lstm_hidden_size * 2) # *2 for bidirectional LSTM
        
        self.fc_layers = nn.ModuleList()
        for layer_size in fc_layers_sizes:
            self.fc_layers.append(nn.Linear(fc_input_size, layer_size))
            fc_input_size = layer_size
            
        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(fc_layers_sizes[-1], 1)

    def forward(self, smiles_tensor, return_activations=False):
        # --- Input Processing ---
        # Input for CNN: (batch, embed_dim, seq_len)
        smi_x_embedded_cnn = self.smiles_embedding(smiles_tensor).permute(0, 2, 1)
        # Input for LSTM: (batch, seq_len, embed_dim)
        smi_x_embedded_lstm = smi_x_embedded_cnn.permute(0, 2, 1)

        # --- CNN Branch Processing ---
        cnn_outputs = [self.smi_pool(torch.relu(cnn(smi_x_embedded_cnn))).squeeze(2) for cnn in self.parallel_cnns]
        cnn_features = torch.cat(cnn_outputs, dim=1)
        
        # --- LSTM Branch Processing ---
        _, (h_n, _) = self.lstm(smi_x_embedded_lstm)
        # Concatenate the final hidden states from forward and backward directions
        lstm_features = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)

        # --- Combine Branches and Final Layers ---
        combined_features = torch.cat((cnn_features, lstm_features), dim=1)
        
        x = self.dropout(combined_features)
        for fc_layer in self.fc_layers:
            x = torch.relu(fc_layer(x))
        
        final_prediction = self.output_layer(x).squeeze(1)

        if return_activations:
            return final_prediction, cnn_features, lstm_features
        else:
            return final_prediction


def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs, device):
    """The main training and validation loop."""
    model.to(device)
    best_val_loss = np.inf
    best_model_path = 'best_model_temp.pth'
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        
        for smiles, targets in progress_bar:
            smiles, targets = smiles.to(device), targets.to(device)
            optimizer.zero_grad()
            predictions = model(smiles)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for smiles, targets in val_loader:
                smiles, targets = smiles.to(device), targets.to(device)
                predictions = model(smiles)
                loss = criterion(predictions, targets)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs} -> Train Loss: {avg_train_loss:.8f}, Val Loss: {avg_val_loss:.8f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Better model found! Val Loss: {best_val_loss:.8f}. Model saved.")
            
    print(f"\nTraining finished. Loading best model with Val Loss: {best_val_loss:.8f}")
    model.load_state_dict(torch.load(best_model_path))
    if os.path.exists(best_model_path):
        os.remove(best_model_path) # Clean up temporary file
        
    return model

def analyze_branch_importance(model, val_loader, device):
    """Analyzes the average activation magnitude from CNN and LSTM branches."""
    model.eval()
    cnn_total_magnitude = 0.0
    lstm_total_magnitude = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for smiles, _ in tqdm(val_loader, desc="Analyzing Branch Importance", leave=False):
            smiles = smiles.to(device)
            _, cnn_activations, lstm_activations = model(smiles, return_activations=True)
            
            cnn_total_magnitude += torch.mean(torch.abs(cnn_activations)).item()
            lstm_total_magnitude += torch.mean(torch.abs(lstm_activations)).item()
            num_batches += 1
            
    cnn_avg_mag = cnn_total_magnitude / num_batches
    lstm_avg_mag = lstm_total_magnitude / num_batches
    
    print("\n--- Parallel Branch Importance Analysis (CNN vs. LSTM) ---")
    print(f"Average Activation Magnitude of CNN Branch : {cnn_avg_mag:.6f}")
    print(f"Average Activation Magnitude of LSTM Branch: {lstm_avg_mag:.6f}")
    if cnn_avg_mag > lstm_avg_mag:
        print(">> Conclusion: The CNN branch sends a stronger signal to the final layers.")
    else:
        print(">> Conclusion: The LSTM branch sends a stronger signal to the final layers.")
    print("---------------------------------------------------------")


os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)

for target_col in TARGET_COLUMNS:
    print("\n" + "="*50)
    print(f"Starting process for target column: {target_col}")
    print("="*50)
    
    df_filtered = train_df.dropna(subset=[target_col]).copy().reset_index(drop=True)
    
    if len(df_filtered) < 20:
        print(f"Not enough data for '{target_col}'. Skipping...")
        continue
    
    # Attempt stratified split to maintain target distribution
    try:
        y_bins = pd.qcut(df_filtered[target_col], q=N_BINS_FOR_STRATIFY, labels=False, duplicates='drop')
        stratify_param = y_bins
        print(f"Performing stratified split for '{target_col}'.")
    except ValueError:
        stratify_param = None
        print(f"Performing normal split for '{target_col}'.")

    X_train, X_val, y_train, y_val = train_test_split(
        df_filtered['SMILES'].values,
        df_filtered[target_col].values,
        test_size=0.25,
        random_state=SEED,
        stratify=stratify_param
    )
    
    train_dataset = PolymerDataset(X_train, y_train, smiles_char_dict, max_smiles_len, is_train=True)
    val_dataset = PolymerDataset(X_val, y_val, smiles_char_dict, max_smiles_len, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = PolymerPredictor(
        smiles_vocab_size=vocab_size,
        smiles_embedding_dim=SMILES_EMBEDDING_DIM,
        cnn_filters=CNN_FILTERS,
        cnn_kernel_sizes=CNN_KERNEL_SIZES,
        lstm_hidden_size=LSTM_HIDDEN_SIZE,
        lstm_num_layers=LSTM_NUM_LAYERS,
        fc_layers_sizes=FC_LAYERS_SIZES,
        dropout_rate=DROPOUT_RATE
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.L1Loss() # Mean Absolute Error
    
    print("Starting training of the hybrid CNN-LSTM model...")
    trained_model = train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, EPOCHS, DEVICE)
    
    # Analyze branch importance after training
    analyze_branch_importance(trained_model, val_loader, DEVICE)
    
    model_save_path = os.path.join(MODELS_OUTPUT_DIR, f'model_cnn_lstm_{target_col}.pth')
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Model for {target_col} successfully saved to '{model_save_path}'.")

print("\nTraining process finished for all target columns.")


def smiles_to_tensor(smiles, char_dict, max_len):
    """Converts a single SMILES string to a padded tensor."""
    pad_token_id = char_dict['&']
    unknown_token_id = char_dict['!']
    tokenized = [char_dict.get(c, unknown_token_id) for c in smiles]
    tokenized = tokenized[:max_len]
    padding = [pad_token_id] * (max_len - len(tokenized))
    return torch.tensor(tokenized + padding, dtype=torch.long)


print("\nLoading test data...")
try:
    test_df = pd.read_csv(TEST_CSV_PATH)
    test_df['SMILES'] = test_df['SMILES'].apply(lambda s: make_smile_canonical(s))
    print("Test data loaded successfully.")
except FileNotFoundError:
    print(f"Error: Test file not found at {TEST_CSV_PATH}")
    exit()

submission_df = pd.DataFrame({'id': test_df['id']})

for target_col in TARGET_COLUMNS:
    print(f"\nPredicting for column: {target_col}")
    model_path = os.path.join(INFERENCE_MODELS_PATH, f'model_cnn_lstm_{target_col}.pth')
    
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at '{model_path}'. Using mean value as fallback.")
        # Fallback to mean of the original training data if a model is missing
        original_train_df = pd.read_csv(TRAIN_CSV_PATH)
        submission_df[target_col] = original_train_df[target_col].mean()
        continue

    model = PolymerPredictor(
        smiles_vocab_size=vocab_size,
        smiles_embedding_dim=SMILES_EMBEDDING_DIM,
        cnn_filters=CNN_FILTERS,
        cnn_kernel_sizes=CNN_KERNEL_SIZES,
        lstm_hidden_size=LSTM_HIDDEN_SIZE,
        lstm_num_layers=LSTM_NUM_LAYERS,
        fc_layers_sizes=FC_LAYERS_SIZES,
        dropout_rate=DROPOUT_RATE
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    final_predictions = []
    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Predicting {target_col}"):
            original_smiles = row['SMILES']
            mol = Chem.MolFromSmiles(original_smiles)
            tta_predictions = []
            
            # Prediction on original canonical SMILES
            original_tensor = smiles_to_tensor(original_smiles, smiles_char_dict, max_smiles_len).unsqueeze(0).to(DEVICE)
            pred_original = model(original_tensor).item()
            tta_predictions.append(pred_original)
            
            # Test-Time Augmentation (TTA)
            if mol is not None:
                for _ in range(N_TTA):
                    aug_smiles = Chem.MolToSmiles(mol, doRandom=True)
                    aug_tensor = smiles_to_tensor(aug_smiles, smiles_char_dict, max_smiles_len).unsqueeze(0).to(DEVICE)
                    pred_aug = model(aug_tensor).item()
                    tta_predictions.append(pred_aug)
            
            final_pred = np.mean(tta_predictions)
            final_predictions.append(final_pred)
            
    submission_df[target_col] = final_predictions
    
    # --- Handle Data Leak ---
    # Replace predictions with known values for test SMILES that were in the training set
    leak_df = train_df.dropna(subset=[target_col])
    leak_dict = pd.Series(leak_df[target_col].values, index=leak_df.SMILES).to_dict()
    leaked_values = test_df['SMILES'].map(leak_dict)
    
    submission_df[target_col] = np.where(leaked_values.notna(), leaked_values, submission_df[target_col])

# --- Save Final Submission File ---
submission_path = 'submission.csv'
submission_df.to_csv(submission_path, index=False)

print("\n" + "="*50)
print(f"submission.csv file created successfully.")
print("Sample of the output:")
print(submission_df.head())
print("="*50)

