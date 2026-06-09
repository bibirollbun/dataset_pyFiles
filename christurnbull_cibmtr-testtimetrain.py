import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR
from joblib import dump, load
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, QuantileTransformer
from datetime import datetime
import json
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances

TRAIN_DF_FILEPATH = "/kaggle/input/equity-post-HCT-survival-predictions/train.csv"
TEST_DF_FILEPATH = "/kaggle/input/equity-post-HCT-survival-predictions/test.csv"
SUBMISSION_FILEPATH = "submission.csv"

# Autoencoder configuration - will be updated with actual input dimension
AE_HIDDEN_DIMS = [384]  # Encoder layers
AE_LATENT_DIM = 256  # Bottleneck dimension
AE_DROPOUT = 0.1
AE_TRAIN_BATCH_SIZE = 256
AE_EPOCHS = 1000
AE_LR = 3e-3
AE_LR_SCHEDULER_PATIENCE = 15
AE_LR_SCHEDULER_FACTOR = 0.5
AE_MODEL_FILEPATH = None


RS_MODEL = 'RiskScoreModel'
RS_EFS_LAYERS = [512, 256, 64, 1]
RS_EFS_TIME_LAYERS = [512, 256, 64, 1]
RS_DROPOUT = 0.5
RS_EPOCHS = 1000
RS_LR = 3e-3
RS_WEIGHT_DECAY = 1e-5
RS_EFS_TIME_SIGMA = 10.0
RS_RISK_SCORE_SIGMA = 10.0
RS_MODEL_FILEPATH = None


def df_encode_fit(df):

    df = df.drop(columns=['ID'])
    if 'efs' in df.columns:
        df = df.drop(columns=['efs'])
    if 'efs_time' in df.columns:
        df = df.drop(columns=['efs_time'])

    # handle float columns
    float_col_names = ['donor_age', 'age_at_hct']
    float_df = df[float_col_names].copy()
    float_df.fillna(0, inplace=True)

    hist = np.histogram(float_df['donor_age'].values, bins='auto')
    n_quantiles = len(hist[0])
    donor_age_encoder = KBinsDiscretizer(
        n_bins=n_quantiles, encode='onehot', strategy='quantile', subsample=None)
    donor_age_encoder.fit(float_df['donor_age'].values.reshape(-1, 1))

    age_at_hct_values = float_df['age_at_hct'].values
    hist = np.histogram(age_at_hct_values, bins='auto')
    n_quantiles = len(hist[0])
    age_at_hct_encoder = KBinsDiscretizer(
        n_bins=n_quantiles, encode='onehot', strategy='quantile', subsample=None)
    age_at_hct_encoder.fit(age_at_hct_values.reshape(-1, 1))

    df = df.drop(columns=float_col_names)

    # handle categorical columns
    # make all columns strings for onehot encoding
    df = df.astype(str)

    # move the race group to the first column
    race_group_series = df.pop('race_group')
    df.insert(0, 'race_group', race_group_series)

    # add a row of nans so the encoder can handle them
    df.loc[len(df)] = np.nan

    onehot_encoder = OneHotEncoder(
        handle_unknown='ignore', sparse_output=False)
    onehot_encoder.fit(df)

    # Store feature indices for one-hot constraints
    feature_indices = []
    start_idx = 0

    # Get categorical feature indices
    cat_feature_names = onehot_encoder.feature_names_in_
    for i, feature_name in enumerate(cat_feature_names):
        n_categories = len(onehot_encoder.categories_[i])
        feature_indices.append((start_idx, start_idx + n_categories))
        start_idx += n_categories

    # Add indices for donor_age and age_at_hct
    donor_age_n_bins = donor_age_encoder.n_bins_[0]
    feature_indices.append((start_idx, start_idx + donor_age_n_bins))
    start_idx += donor_age_n_bins

    age_at_hct_n_bins = age_at_hct_encoder.n_bins_[0]
    feature_indices.append((start_idx, start_idx + age_at_hct_n_bins))

    # Create a mapping of feature names to their indices
    feature_names = list(cat_feature_names) + float_col_names
    feature_index_map = {
        feature_names[i]: feature_indices[i] for i in range(len(feature_names))}

    df_encoder = {
        'onehot_encoder': onehot_encoder,
        'donor_age_encoder': donor_age_encoder,
        'age_at_hct_encoder': age_at_hct_encoder,
        'feature_indices': feature_indices,
        'feature_index_map': feature_index_map,
        'total_features': len(feature_indices)
    }
    return df_encoder


def df_encode_transform(df, df_encoder):

    df = df.drop(columns=['ID'])
    if 'efs' in df.columns:
        df = df.drop(columns=['efs'])
    if 'efs_time' in df.columns:
        df = df.drop(columns=['efs_time'])

    # handle float columns
    float_col_names = ['donor_age', 'age_at_hct']
    float_df = df[float_col_names].copy()
    float_df.fillna(0, inplace=True)

    donor_age_encoder = df_encoder['donor_age_encoder']
    donor_age_onehot = donor_age_encoder.transform(
        float_df['donor_age'].values.reshape(-1, 1))
    # Ensure dense array format
    if hasattr(donor_age_onehot, 'toarray'):
        donor_age_onehot = donor_age_onehot.toarray()
    # print(donor_age_onehot.shape)

    age_at_hct_encoder = df_encoder['age_at_hct_encoder']
    age_at_hct_onehot = age_at_hct_encoder.transform(
        float_df['age_at_hct'].values.reshape(-1, 1))
    # Ensure dense array format
    if hasattr(age_at_hct_onehot, 'toarray'):
        age_at_hct_onehot = age_at_hct_onehot.toarray()
    # print(age_at_hct_onehot.shape)

    df = df.drop(columns=float_col_names)

    # handle categorical columns
    df = df.astype(str)
    race_group_series = df.pop('race_group')
    df.insert(0, 'race_group', race_group_series)

    onehot_encoder = df_encoder['onehot_encoder']
    cat_onehot = onehot_encoder.transform(df)
    # Ensure dense array format
    if hasattr(cat_onehot, 'toarray'):
        cat_onehot = cat_onehot.toarray()

    # Concatenate all one-hot encoded features
    transformed = np.hstack([cat_onehot, donor_age_onehot, age_at_hct_onehot])
    # print(transformed.shape)

    return transformed


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(ResidualBlock, self).__init__()

        # Main path
        self.main_path = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features)
        )

        # Skip connection (if dimensions don't match)
        self.skip_connection = None
        if in_features != out_features:
            self.skip_connection = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )

        # Final activation
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        # Main path
        main = self.main_path(x)

        # Skip connection
        skip = x
        if self.skip_connection is not None:
            skip = self.skip_connection(x)

        # Combine and activate
        return self.activation(main + skip)


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout):
        super(Autoencoder, self).__init__()

        # Encoder with residual connections
        self.encoder_input = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )

        # Encoder residual blocks
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.encoder_blocks.append(
                ResidualBlock(
                    hidden_dims[i],
                    hidden_dims[i+1],
                    dropout
                )
            )

        # Bottleneck layer
        self.bottleneck = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder input
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[-1]),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )

        # Decoder residual blocks
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1, 0, -1):
            self.decoder_blocks.append(
                ResidualBlock(
                    hidden_dims[i],
                    hidden_dims[i-1],
                    dropout
                )
            )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[0], input_dim),
            nn.Sigmoid()  # Sigmoid for binary data
        )

    def forward(self, x):
        # Encode
        x = self.encoder_input(x)
        for block in self.encoder_blocks:
            x = block(x)
        encoded = self.bottleneck(x)

        # Decode
        x = self.decoder_input(encoded)
        for block in self.decoder_blocks:
            x = block(x)
        decoded = self.output_layer(x)

        return decoded

    def encode(self, x):
        # Return only the encoded representation
        x = self.encoder_input(x)
        for block in self.encoder_blocks:
            x = block(x)
        return self.bottleneck(x)


class EntropyRegularizedBCELoss(nn.Module):
    """
    BCE Loss with entropy regularization to encourage outputs closer to 0 or 1.
    """

    def __init__(self, penalty_weight=0.1):
        super(EntropyRegularizedBCELoss, self).__init__()
        self.bce = nn.BCELoss()
        self.penalty_weight = penalty_weight

    def forward(self, inputs, targets):
        # Standard BCE loss
        bce_loss = self.bce(inputs, targets)

        # Entropy penalty: -p*log(p) - (1-p)*log(1-p)
        # This term is maximized at p=0.5 and minimized at p=0 or p=1
        # We add a small epsilon to avoid log(0)
        epsilon = 1e-7
        entropy = -inputs * \
            torch.log(inputs + epsilon) - (1 - inputs) * \
            torch.log(1 - inputs + epsilon)

        # Return combined loss
        return bce_loss + self.penalty_weight * entropy.mean()


def prepare_autoencoder_data():
    """Load and prepare the data for training"""
    # Load the encoder
    df_encoder = load('data/df_encoder.joblib')

    # Load the data
    train_df = pd.read_csv(TRAIN_DF_FILEPATH)
    test_df = pd.read_csv(TEST_DF_FILEPATH)

    # Concatenate test data to train data
    train_df = pd.concat([train_df, test_df])

    # Shuffle the data
    train_df = train_df.sample(frac=1).reset_index(drop=True)

    # Transform the data using the encoder
    encoded_data = df_encode_transform(
        train_df, df_encoder)

    # Get the actual input dimension
    input_dim = encoded_data.shape[1]

    # Convert to PyTorch tensors
    data_tensor = torch.tensor(encoded_data, dtype=torch.float32)

    # Create dataset and dataloader
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(
        dataset, batch_size=AE_TRAIN_BATCH_SIZE, shuffle=True)

    return dataloader, input_dim


def train_autoencoder_epoch(model, dataloader, criterion, optimizer, device, epoch, scheduler=None):
    """Train model for one epoch"""
    model.train()
    total_loss = 0.0
    total_batches = 0
    total_correct = 0
    total_elements = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        # Get input data
        inputs = batch[0].to(device)
        batch_size = inputs.size(0)

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, inputs)

        # Calculate accuracy for binary data (treat as binary classification for each feature)
        # Round outputs to 0 or 1 and compare with inputs
        predictions = (outputs > 0.5).float()
        correct = (predictions == inputs).float().sum().item()
        total_elements_batch = inputs.numel()  # Total number of elements in the batch
        total_correct += correct
        total_elements += total_elements_batch

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Step the scheduler if it's OneCycleLR
        if scheduler is not None:
            scheduler.step()

        # Update metrics
        total_loss += loss.item() * batch_size
        total_batches += batch_size

        # Update progress bar
        batch_accuracy = correct / total_elements_batch
        pbar.set_postfix({'loss': loss.item(), 'acc': batch_accuracy})

    epoch_loss = total_loss / total_batches
    epoch_accuracy = total_correct / total_elements if total_elements > 0 else 0

    return epoch_loss, epoch_accuracy


def train_autoencoder():
    global AE_MODEL_FILEPATH
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, input_dim = prepare_autoencoder_data()

    # Create model with the actual input dimension
    model = Autoencoder(
        input_dim=input_dim,
        hidden_dims=AE_HIDDEN_DIMS,
        latent_dim=AE_LATENT_DIM,
        dropout=AE_DROPOUT
    ).to(device)

    # Create optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=AE_LR, weight_decay=1e-5)

    # Use custom loss function with penalty for values close to 0.5
    # You can choose between BinaryFocalLoss or EntropyRegularizedBCELoss
    # Uncomment the one you prefer:

    # Option 1: Focal Loss with confidence penalty
    # criterion = BinaryFocalLoss(penalty_weight=0.1)

    # Option 2: BCE with entropy regularization
    criterion = EntropyRegularizedBCELoss(penalty_weight=0.1)

    # Original BCE loss (for reference)
    # criterion = nn.BCELoss()

    # Calculate total steps for OneCycleLR
    total_steps = AE_EPOCHS * len(train_loader)

    # Use OneCycleLR scheduler for better convergence
    scheduler = OneCycleLR(
        optimizer,
        max_lr=AE_LR,
        total_steps=total_steps,
        pct_start=0.3,  # Spend 30% of training time warming up
        div_factor=25,  # Initial learning rate will be max_lr/25
        final_div_factor=1000  # Final learning rate will be max_lr/1000
    )

    # Log to tensorboard
    log_dir = datetime.now().strftime(f'autoencoder_enhanced_%Y-%m-%dT%H:%M:%S')
    tensorboard_log_folder = os.path.join("runs", log_dir)
    os.makedirs(tensorboard_log_folder, exist_ok=True)
    writer = SummaryWriter(tensorboard_log_folder)

    # Save hyperparameters
    hparam_dict = {
        "model": "EnhancedAutoencoder",
        "input_dim": input_dim,
        "hidden_dims": str(AE_HIDDEN_DIMS),
        "latent_dim": AE_LATENT_DIM,
        "dropout": AE_DROPOUT,
        "epochs": AE_EPOCHS,
        "learning_rate": AE_LR,
        "batch_size": AE_TRAIN_BATCH_SIZE,
        "scheduler": "OneCycleLR",
        "weight_decay": 1e-5
    }

    # Save hyperparameters to a JSON file for reference
    with open(os.path.join(tensorboard_log_folder, 'config.json'), 'w') as f:
        json.dump(hparam_dict, f, indent=4)

    # Training loop
    best_loss = float('inf')
    for epoch in range(1, AE_EPOCHS + 1):
        # Train for one epoch
        train_loss, train_accuracy = train_autoencoder_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scheduler)

        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)

        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        writer.flush()  # Force write to disk

        # Save model if it's the best so far
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), os.path.join(
                tensorboard_log_folder, 'best_model.pt'))
            print(
                f"Epoch {epoch}: New best model saved with loss {best_loss:.6f}, accuracy {train_accuracy:.6f}")

    # Add hyperparameters after training is complete
    metric_dict = {
        "hparam/best_loss": best_loss,
    }
    writer.add_hparams(hparam_dict, metric_dict, run_name=".")

    AE_MODEL_FILEPATH = os.path.join(tensorboard_log_folder, 'best_model.pt')
    print(f"Autoencoder model saved to: {AE_MODEL_FILEPATH}")
    return AE_MODEL_FILEPATH


def fit_quantile_scaler(target, output_distribution='uniform', random_state=42):

    hist = np.histogram(target, bins='auto')
    n_quantiles = len(hist[0])

    scaler = QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution=output_distribution,
        random_state=random_state
    )

    scaler.fit(target.reshape(-1, 1))

    config = {
        "n_quantiles": scaler.n_quantiles_,
        "output_distribution": scaler.output_distribution,
        "random_state": scaler.random_state
    }
    config["quantiles"] = scaler.quantiles_.tolist() if hasattr(scaler,
                                                                'quantiles_') else None
    return scaler, config


def find_val_data():
    print("Loading data and encoder...")
    df_encoder = load('data/df_encoder.joblib')

    # Load original DataFrames to keep track of IDs
    train_df = pd.read_csv(TRAIN_DF_FILEPATH)
    test_df = pd.read_csv(TEST_DF_FILEPATH)

    # Get test and train IDs
    train_ids = train_df['ID'].values
    test_ids = test_df['ID'].values

    # Extract target values before they get dropped during transformation
    print("Extracting and processing target values...")

    # Create a DataFrame to store target values with IDs
    target_df = pd.DataFrame({
        'ID': train_df['ID'],
        'efs': train_df['efs'],
        'efs_time': train_df['efs_time']
    })

    # Fit quantile scaler on efs_time
    efs_time_values = train_df['efs_time'].values
    efs_time_scaler, efs_time_config = fit_quantile_scaler(
        efs_time_values,
        output_distribution='normal'
    )

    # Transform efs_time using the scaler
    efs_time_scaled = efs_time_scaler.transform(
        efs_time_values.reshape(-1, 1)).flatten()

    # Add scaled efs_time to the target DataFrame
    target_df['efs_time_scaled'] = efs_time_scaled

    # Save the scaler for future use
    # dump(efs_time_scaler, 'data/efs_time_scaler.joblib')

    # Create a dictionary mapping ID to target values for quick lookup
    target_dict = {
        row['ID']: {
            'efs': row['efs'],
            'efs_time': row['efs_time'],
            'efs_time_scaled': row['efs_time_scaled']
        } for _, row in target_df.iterrows()
    }

    # Transform data for encoding
    test_data = df_encode_transform(test_df, df_encoder)
    train_data = df_encode_transform(train_df, df_encoder)
    print(
        f"Test data shape: {test_data.shape}, Train data shape: {train_data.shape}")

    # Load the autoencoder model
    autoencoder_model_filepath = AE_MODEL_FILEPATH

    # Get input dimension from the data
    input_dim = test_data.shape[1]

    # Initialize the model with the same architecture as during training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Autoencoder(
        input_dim=input_dim,
        hidden_dims=AE_HIDDEN_DIMS,
        latent_dim=AE_LATENT_DIM,
        dropout=AE_DROPOUT
    ).to(device)
    model.load_state_dict(torch.load(
        autoencoder_model_filepath, map_location=device, weights_only=True))

    model.eval()  # Set to evaluation mode

    print("Encoding data using the autoencoder...")
    # Convert data to PyTorch tensors
    test_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
    train_tensor = torch.tensor(train_data, dtype=torch.float32).to(device)

    # Encode the data in batches to avoid memory issues
    batch_size = 1024

    # Encode test data
    test_encoded = []
    with torch.no_grad():
        for i in range(0, len(test_tensor), batch_size):
            batch = test_tensor[i:i+batch_size]
            encoded_batch = model.encode(batch).cpu().numpy()
            test_encoded.append(encoded_batch)
    test_encoded = np.vstack(test_encoded)

    # Encode train data
    train_encoded = []
    with torch.no_grad():
        for i in range(0, len(train_tensor), batch_size):
            batch = train_tensor[i:i+batch_size]
            encoded_batch = model.encode(batch).cpu().numpy()
            train_encoded.append(encoded_batch)
    train_encoded = np.vstack(train_encoded)

    print(
        f"Encoded shapes - Test: {test_encoded.shape}, Train: {train_encoded.shape}")

    # Find the closest train sample for each test sample
    print("Finding validation set based on shortest distances...")
    val_indices = []

    # Using a progress bar to track the process
    for i in tqdm(range(len(test_encoded))):
        # Calculate distances from this test sample to all train samples
        distances = euclidean_distances(
            test_encoded[i].reshape(1, -1),
            train_encoded
        ).flatten()

        # Find the index of the closest train sample
        closest_idx = np.argmin(distances)
        val_indices.append(closest_idx)

    # Convert to numpy array for easier manipulation
    val_indices = np.array(val_indices)

    # Create validation set from the selected train samples
    val_ids = train_ids[val_indices]

    # Get the one-hot encoded data for validation set
    val_data_onehot = train_data[val_indices]

    # Extract target values for validation set
    val_efs = np.array([target_dict[id]['efs'] for id in val_ids])
    val_efs_time = np.array([target_dict[id]['efs_time'] for id in val_ids])
    val_efs_time_scaled = np.array(
        [target_dict[id]['efs_time_scaled'] for id in val_ids])

    # Remove validation samples from train set
    # Create a mask of samples to keep in the train set
    train_mask = np.ones(len(train_data), dtype=bool)
    train_mask[val_indices] = False

    # Filter train set
    filtered_train_data_onehot = train_data[train_mask]
    filtered_train_ids = train_ids[train_mask]

    # Extract target values for filtered train set
    filtered_train_efs = np.array(
        [target_dict[id]['efs'] for id in filtered_train_ids])
    filtered_train_efs_time = np.array(
        [target_dict[id]['efs_time'] for id in filtered_train_ids])
    filtered_train_efs_time_scaled = np.array(
        [target_dict[id]['efs_time_scaled'] for id in filtered_train_ids])

    # Create y arrays (combining all target variables)
    # For train set
    y_train = np.column_stack((
        filtered_train_efs,
        filtered_train_efs_time,
        filtered_train_efs_time_scaled
    ))

    # For validation set
    y_val = np.column_stack((
        val_efs,
        val_efs_time,
        val_efs_time_scaled
    ))

    # For test set (if targets are available, otherwise will be empty)
    y_test = np.array([])  # Empty array as test set doesn't have targets

    print(f"Final dataset sizes:")
    print(f"Test: {len(test_data)} samples")
    print(f"Validation: {len(val_data_onehot)} samples")
    print(f"Filtered Train: {len(filtered_train_data_onehot)} samples")
    print(f"Target shapes - Train: {y_train.shape}, Val: {y_val.shape}")

    # Save the datasets with clear X and y separation
    print("Saving datasets with features (X) and targets (y)...")
    ml_data = {
        # X data (features)
        'X_train': filtered_train_data_onehot,
        'X_val': val_data_onehot,
        'X_test': test_data,

        # y data (targets)
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,

        # IDs for reference
        'train_ids': filtered_train_ids,
        'val_ids': val_ids,
        'test_ids': test_ids,

        # Target column information
        'y_columns': ['efs', 'efs_time', 'efs_time_scaled'],

        # Scaler for future transformations
        'efs_time_scaler': efs_time_scaler,
        'efs_time_config': efs_time_config
    }
    dump(ml_data, 'data/ml_data.joblib')

    # Also save the validation indices for reference
    val_indices_data = {
        'val_indices': val_indices,
        'val_ids': val_ids
    }
    dump(val_indices_data, 'data/val_indices.joblib')

    print("Done! Machine learning data saved to 'data/ml_data.joblib'")
    return ml_data


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable layer sizes
    """

    def __init__(self, layer_sizes, dropout=0.6):
        """
        Args:
            layer_sizes (list): List of layer sizes, including input and output dimensions
            dropout (float): Dropout probability
        """
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()

        # Create layers based on the provided sizes
        for i in range(len(layer_sizes) - 1):
            # Add linear layer
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

            # Add batch norm, activation, and dropout for all but the last layer
            if i < len(layer_sizes) - 2:
                self.layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(dropout))

    def forward(self, x):
        """Forward pass through the MLP"""
        for layer in self.layers:
            x = layer(x)
        return x


class RiskScoreModel(nn.Module):
    """Neural network for CIBMTR risk score prediction with configurable MLPs"""

    def __init__(self,
                 input_dim=183,
                 efs_layers=None,
                 efs_time_layers=None,
                 dropout=0.6,
                 device='cpu'):
        """
        Args:
            input_dim (int): Input dimension
            efs_layers (list): List of layer sizes for EFS MLP (excluding input and including output)
            efs_time_layers (list): List of layer sizes for EFS time MLP (excluding input and including output)
            dropout (float): Dropout probability
            device (str): Device to run the model on
        """
        super(RiskScoreModel, self).__init__()
        self.device = device

        # Default layer configurations if none provided
        if efs_layers is None:
            efs_layers = [input_dim, input_dim, 57, 1]
        else:
            efs_layers = [input_dim] + efs_layers

        if efs_time_layers is None:
            efs_time_layers = [input_dim, input_dim, 57, 1]
        else:
            efs_time_layers = [input_dim] + efs_time_layers

        # Create MLPs for EFS and EFS time
        self.efs_mlp = MLP(efs_layers, dropout)
        self.efs_time_mlp = MLP(efs_time_layers, dropout)

        # Initialize trainable weight parameters
        self.w_efs = nn.Parameter(torch.tensor(1.0))
        self.w_efs_time = nn.Parameter(torch.tensor(1.0))

        # Activation functions
        self.sigmoid = nn.Sigmoid()

        # Move model to specified device
        self.to(device)

    def _risk_score(self, efs, efs_time):
        """
        Calculate a risk score for HCT survival based on event-free survival (efs) and efs_time.

        This is a PyTorch implementation of the calculate_risk_score function from scale.py.

        Args:
            efs (torch.Tensor): Binary indicator where 1 means an event occurred (death or relapse) 
                              and 0 means the patient is event-free.
            efs_time (torch.Tensor): Time in days until event or last follow-up, normalized to [0,1] range.
                                   Higher values indicate longer survival time.

        Returns:
            torch.Tensor: Risk score between 0 and 1, where higher values indicate higher risk.
        """
        # Use trainable weight parameters
        # When efs=1 (event occurred), risk is higher
        # When efs_time is higher (longer survival), risk is lower
        # - For efs=1 (event occurred): (1 + w_efs) * (1 - w_efs_time * efs_time) = higher risk
        # - For efs=0 (no event): (1) * (1 - w_efs_time * efs_time) = lower risk
        risk_score = (1 + self.w_efs * efs) * (1 - self.w_efs_time * efs_time)

        # Sigmoid function to map to [0,1] range
        risk_score = self.sigmoid(risk_score)

        return risk_score

    def forward(self, features):
        """
        Forward pass through the model

        Args:
            features (torch.Tensor): Input features

        Returns:
            torch.Tensor: Predicted risk score
        """
        # Predict EFS (event occurrence)
        efs = self.sigmoid(self.efs_mlp(features)).squeeze(-1)

        # Predict EFS time (scaled)
        efs_time_scaled = self.sigmoid(self.efs_time_mlp(features)).squeeze(-1)

        # Convert efs and efs_time to a risk score
        risk_score = self._risk_score(efs, efs_time_scaled)

        return risk_score


class CIBMTRDataset(Dataset):
    """Dataset handler for CIBMTR risk score prediction"""

    def __init__(self,
                 X,
                 y,
                 device='cpu'):

        self.features = torch.FloatTensor(X).to(device)

        self.true_efs = torch.FloatTensor(y[:, 0]).to(device)
        self.true_efs_time_unscaled = torch.FloatTensor(y[:, 1]).to(device)
        self.true_efs_time_scaled = torch.FloatTensor(y[:, 2]).to(device)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features[idx]
        race_onehot = features[0:6]

        return \
            features, \
            race_onehot, \
            self.true_efs[idx], \
            self.true_efs_time_unscaled[idx], \
            self.true_efs_time_scaled[idx]


class StratifiedConcordanceIndexLoss(nn.Module):
    """
    Differentiable Stratified Concordance Index Loss Module for PyTorch.

    This module provides a differentiable approximation of the Stratified Concordance Index
    for use as a loss function in PyTorch models. It uses sigmoid functions to create
    smooth approximations of the discrete comparisons in the original metric.

    The loss converges to 0 as the mean concordance index approaches 1 and the standard deviation approaches 0.

    Args:
        efs_time_sigma (float, optional): Controls the steepness of the sigmoid for EFS time.
                                          Higher values make the approximation closer to the step function.
                                          Defaults to 5.0
        risk_score_sigma (float, optional): Controls the steepness of the sigmoid for risk score.
                                            Higher values make the approximation closer to the step function.
                                            Defaults to 10.0
    """

    def __init__(self, efs_time_sigma=5.0, risk_score_sigma=10.0):
        super().__init__()
        self.efs_time_sigma = efs_time_sigma
        self.risk_score_sigma = risk_score_sigma

    def forward(self, predictions, event_times, events, race_onehot):
        """
        Calculate the Stratified Concordance Index Loss.

        Args:
            predictions (torch.Tensor): Predicted risk scores
            event_times (torch.Tensor): Event times
            events (torch.Tensor): Event indicators (1 if event occurred, 0 if censored)
            race_onehot (torch.Tensor): One-hot encoded race groups

        Returns:
            torch.Tensor: The stratified concordance index loss
        """
        # Ensure inputs are the right shape
        predictions = predictions.squeeze()
        event_times = event_times.squeeze()
        events = events.squeeze()

        num_races = race_onehot.shape[1]
        group_c_indices = []

        # For each race group
        for race_idx in range(num_races):
            # Get indices for this race group
            race_mask = race_onehot[:, race_idx].bool()
            if torch.sum(race_mask) < 2:
                continue

            group_pred = predictions[race_mask]
            group_times = event_times[race_mask]
            group_events = events[race_mask]

            # Expand tensors for pairwise comparisons
            event_preds = group_pred.unsqueeze(1)  # Shape: (n, 1)
            all_preds = group_pred.unsqueeze(0)    # Shape: (1, n)

            group_event_times = group_times.unsqueeze(1)  # Shape: (n, 1)
            group_all_times = group_times.unsqueeze(0)    # Shape: (1, n)

            group_events = group_events.unsqueeze(1)      # Shape: (n, 1)

            # Calculate valid pairs using sigmoid
            # A pair is valid if:
            # 1. The first case has an event (not censored)
            # 2. The first case's time is less than the second case's time
            valid_time_pairs = torch.sigmoid(
                self.efs_time_sigma * (group_all_times - group_event_times))
            valid_pairs = valid_time_pairs * group_events

            # Calculate concordant pairs using sigmoid
            # A pair is concordant if the predicted risk is higher for the case
            # that failed earlier
            concordant = torch.sigmoid(
                self.risk_score_sigma * (event_preds - all_preds))

            # Combine concordant and valid pairs
            concordant_valid = concordant * valid_pairs

            # Calculate c-index for this group
            n_valid = torch.sum(valid_pairs)
            if n_valid > 0:
                group_c_index = torch.sum(concordant_valid) / n_valid
                group_c_indices.append(group_c_index)
            else:
                group_c_indices.append(torch.tensor(
                    0.0, device=event_times.device))

        # Calculate mean and std of c-indices across groups
        if len(group_c_indices) > 1:
            group_c_indices = torch.stack(group_c_indices)
            mean_c_index = torch.mean(group_c_indices)
            std_c_index = torch.std(group_c_indices)
        else:
            # If only one group, use its c-index and zero std
            mean_c_index = group_c_indices[0]
            std_c_index = torch.tensor(0.0, device=predictions.device)

        # Return loss (1 - mean_c_index + std_penalty)
        return 1 - (mean_c_index - std_c_index)


def stratified_concordance_index_metric(
    pred_risk_scores: np.ndarray,
    true_efs_time: np.ndarray,
    true_efs_event: np.ndarray,
    X_race_indices: np.ndarray,
    n_races: int = 6,
) -> float:
    """
    Optimized version of the Stratified Concordance Index calculation using vectorized operations.
    """
    # Initialize counters
    group_c_indices = []

    # print(pred_risk_scores[:10])
    # print(true_efs_time[:10])
    # print(true_efs_event[:10])
    # print(X_race_indices[:10])

    for race in range(n_races):
        indices = np.where(X_race_indices == race)[0]
        if len(indices) < 2:
            continue

        # Get group-specific data
        group_risk_scores = pred_risk_scores[indices]
        group_time = true_efs_time[indices]
        group_event = true_efs_event[indices]

        # Find events
        event_mask = group_event == 1
        event_indices = np.where(event_mask)[0]

        if len(event_indices) == 0:
            continue  # Skip instead of adding 0.5

        # Create comparison matrices
        event_times = group_time[event_indices][:, np.newaxis]
        all_times = group_time[np.newaxis, :]

        # Time comparison matrix
        valid_pairs = (event_times < all_times)

        # Prediction comparison matrix
        event_preds = group_risk_scores[event_indices]
        if len(event_preds.shape) > 1:
            event_preds = event_preds.squeeze()
        event_preds = event_preds[:, np.newaxis]

        all_preds = group_risk_scores
        if len(all_preds.shape) > 1:
            all_preds = all_preds.squeeze()
        all_preds = all_preds[np.newaxis, :]

        concordant = (event_preds > all_preds)  # Higher score = lower risk now

        # Count valid and concordant pairs
        n_valid = np.sum(valid_pairs)
        n_concordant = np.sum(concordant & valid_pairs)

        # Calculate c-index for this group
        if n_valid > 0:
            group_c_indices.append(n_concordant / n_valid)

    if not group_c_indices:
        return 0.5  # Return 0.5 if no valid groups found

    # Calculate final stratified concordance index
    group_c_indices_np = np.array(group_c_indices)
    mean_c_index = np.mean(group_c_indices_np)
    std_c_index = np.sqrt(np.var(group_c_indices_np))

    return mean_c_index - std_c_index


def train_risk_score_epoch(model, train_loader, criterion, optimizer, epoch, scheduler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_samples = 0

    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        # Unpack features and targets
        features, race_onehot, true_efs, true_efs_time_unscaled, true_efs_time_scaled = batch
        batch_size = features.size(0)

        # Forward pass
        optimizer.zero_grad()

        pred_risk_score = model(features)

        loss = criterion(pred_risk_score, true_efs_time_scaled,
                         true_efs, race_onehot)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Backward pass
        loss.backward()
        optimizer.step()

        # Step the scheduler if it's provided and is batch-based
        if scheduler is not None and isinstance(scheduler, OneCycleLR):
            scheduler.step()

        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'epoch': epoch,
        })

    return total_loss / total_samples


def validate_risk_score(model, val_loader, criterion):
    """
    Validate the model on the validation set.
    """
    model.eval()
    total_loss = 0
    total_samples = 0
    all_pred_risk_score = []
    all_true_efs = []
    all_true_efs_time_unscaled = []
    all_X_race_indices = []

    with torch.no_grad():
        for batch in val_loader:
            features, race_onehot, true_efs, true_efs_time_unscaled, true_efs_time_scaled = batch
            batch_size = features.size(0)

            pred_risk_score = model(features)

            loss = criterion(pred_risk_score, true_efs_time_scaled,
                             true_efs, race_onehot)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Get race indices
            X_race_indices = torch.argmax(race_onehot, dim=1)

            # Collect predictions and ground truth
            all_pred_risk_score.extend(pred_risk_score.cpu().numpy())
            all_true_efs.extend(true_efs.cpu().numpy())
            all_true_efs_time_unscaled.extend(
                true_efs_time_unscaled.cpu().numpy())
            all_X_race_indices.extend(X_race_indices.cpu().numpy())

    sci = stratified_concordance_index_metric(
        pred_risk_scores=np.asarray(all_pred_risk_score),
        true_efs_time=np.asarray(all_true_efs_time_unscaled),
        true_efs_event=np.asarray(all_true_efs),
        X_race_indices=np.asarray(all_X_race_indices),
    )

    # Calculate average loss per sample
    val_loss = total_loss / total_samples
    return val_loss, sci


def train_risk_score_model():

    global RS_MODEL_FILEPATH

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Using device: {device}")

    # Load the data
    ml_data = load('data/ml_data.joblib')

    # Extract features and targets
    X_train = ml_data['X_train']
    y_train = ml_data['y_train']
    X_val = ml_data['X_val']
    y_val = ml_data['y_val']

    train_dataset = CIBMTRDataset(
        X_train, y_train,  device=device)
    val_dataset = CIBMTRDataset(
        X_val, y_val,  device=device)

    INPUT_DIM = X_train.shape[1]
    TRAIN_BATCH_SIZE = X_train.shape[0]
    VAL_BATCH_SIZE = X_val.shape[0]

    train_loader = DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)

    # log to tensorboard
    log_dir = datetime.now().strftime(f'{RS_MODEL}_%Y-%m-%dT%H:%M:%S')
    tensorboard_log_folder = os.path.join("runs", log_dir)
    os.makedirs("runs", exist_ok=True)
    writer = SummaryWriter(tensorboard_log_folder)

    hparam_dict = {
        "model": RS_MODEL,
        "input_dim": INPUT_DIM,
        "dropout": RS_DROPOUT,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "val_batch_size": VAL_BATCH_SIZE,
        "epochs": RS_EPOCHS,
        "learning_rate": RS_LR,
        "weight_decay": RS_WEIGHT_DECAY,
        "efs_time_sigma": RS_EFS_TIME_SIGMA,
        "risk_score_sigma": RS_RISK_SCORE_SIGMA,
        "initial_w_efs": 1.0,
        "initial_w_efs_time": 1.0
    }
    metric_dict = {
        "hparam/dummy": 0,
    }
    writer.add_hparams(hparam_dict, metric_dict, run_name=".")
    with open(os.path.join(tensorboard_log_folder, 'config.json'), 'w') as f:
        json.dump(hparam_dict, f, indent=4)

    model = RiskScoreModel(
        input_dim=INPUT_DIM,
        efs_layers=RS_EFS_LAYERS,
        efs_time_layers=RS_EFS_TIME_LAYERS,
        dropout=RS_DROPOUT,
        device=device
    )

    criterion = StratifiedConcordanceIndexLoss(
        efs_time_sigma=RS_EFS_TIME_SIGMA, risk_score_sigma=RS_RISK_SCORE_SIGMA)

    # Create optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=RS_LR,
                            weight_decay=RS_WEIGHT_DECAY)

    # Calculate total steps for OneCycleLR
    total_steps = RS_EPOCHS * len(train_loader)

    # Use OneCycleLR scheduler for better convergence
    scheduler = OneCycleLR(
        optimizer,
        max_lr=RS_LR,
        total_steps=total_steps,
        pct_start=0.3,  # Spend 30% of training time warming up
        div_factor=25,  # Initial learning rate will be max_lr/25
        final_div_factor=1000  # Final learning rate will be max_lr/1000
    )

    # Training loop
    best_vl = {'e': 0, 'vl': float('inf'), 'sci': -float('inf')}
    best_sci = {'e': 0, 'vl': float('inf'), 'sci': -float('inf')}
    for epoch in range(RS_EPOCHS):

        # Train
        train_loss = train_risk_score_epoch(
            model, train_loader, criterion, optimizer, epoch, scheduler)
        writer.add_scalar('Loss/Training', train_loss, epoch)

        # Validate
        val_loss, sci = validate_risk_score(model, val_loader, criterion)

        # Log metrics
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Metrics/Stratified Concordance Index', sci, epoch)

        # Log learning rate and trainable weights
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        writer.add_scalar('Weights/w_efs', model.w_efs.item(), epoch)
        writer.add_scalar('Weights/w_efs_time', model.w_efs_time.item(), epoch)

        if val_loss < best_vl['vl']:
            best_vl['e'] = epoch
            best_vl['vl'] = val_loss
            best_vl['sci'] = sci
            model_save_path = os.path.join(
                tensorboard_log_folder, f'best_vl.pt')
            torch.save(model.state_dict(), model_save_path)
            writer.add_scalar(
                'Best/Validation Loss', val_loss, epoch)

        if sci > best_sci['sci']:
            best_sci['e'] = epoch
            best_sci['vl'] = val_loss
            best_sci['sci'] = sci
            model_save_path = os.path.join(
                tensorboard_log_folder, f'best_sci.pt')
            torch.save(model.state_dict(), model_save_path)
            writer.add_scalar(
                'Best/Best Stratified Concordance Index', sci, epoch)

    os.rename(
        os.path.join(tensorboard_log_folder, 'best_vl.pt'),
        os.path.join(tensorboard_log_folder,
                     f"{RS_MODEL}_e{best_vl['e']}_vl{best_vl['vl']:.5f}_sci{best_vl['sci']:.5f}_vl.pt")
    )

    os.rename(
        os.path.join(tensorboard_log_folder, 'best_sci.pt'),
        os.path.join(tensorboard_log_folder,
                     f"{RS_MODEL}_e{best_sci['e']}_vl{best_sci['vl']:.5f}_sci{best_sci['sci']:.5f}_sci.pt")
    )

    RS_MODEL_FILEPATH = os.path.join(
        tensorboard_log_folder, f"{RS_MODEL}_e{best_sci['e']}_vl{best_sci['vl']:.5f}_sci{best_sci['sci']:.5f}_sci.pt")

    print(f"Best stratified concordance index: {best_sci['sci']:.5f}")
    print(f"Best validation loss: {best_vl['vl']:.5f}")
    print(
        f"Final learned weights - w_efs: {model.w_efs.item():.5f}, w_efs_time: {model.w_efs_time.item():.5f}")
    writer.close()


def predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df_encoder = load('data/df_encoder.joblib')
    test_df = pd.read_csv(TEST_DF_FILEPATH)
    test_data = df_encode_transform(test_df, df_encoder)
    input_dim = test_data.shape[1]

    model = RiskScoreModel(
        input_dim=input_dim,
        efs_layers=RS_EFS_LAYERS,
        efs_time_layers=RS_EFS_TIME_LAYERS,
        dropout=RS_DROPOUT,
        device=device
    )
    model.load_state_dict(torch.load(
        RS_MODEL_FILEPATH, map_location=device, weights_only=True))
    model.eval()
    X = torch.FloatTensor(test_data).to(device)
    y = model(X)

    predictions = y.detach().cpu().numpy()
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'prediction': predictions.flatten()
    })
    submission = submission.sort_values(by='ID')
    submission.to_csv(SUBMISSION_FILEPATH, index=False)
    print(f"Submission dataframe shape: {submission.shape}")
    print(submission.head())


def main():

    if not os.path.exists('data'):
        os.makedirs('data')

    train_df = pd.read_csv(TRAIN_DF_FILEPATH)
    test_df = pd.read_csv(TEST_DF_FILEPATH)
    concat_df = pd.concat([train_df, test_df])

    df_enoder = df_encode_fit(concat_df)
    dump(df_enoder, 'data/df_encoder.joblib')

    model_path = train_autoencoder()
    print(f"Using autoencoder model from: {model_path}")

    find_val_data()

    train_risk_score_model()

    predict()


main()


