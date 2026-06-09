import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import glob
import os
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
pl.seed_everything(42)

# Configuration
class Config:
    # Data paths
    train_data_path = "/kaggle/input/ariel-data-challenge-2025/train"
    test_data_path = "/kaggle/input/ariel-data-challenge-2025/test"
    metadata_path = "/kaggle/input/ariel-data-challenge-2025"
    
    # Model parameters
    batch_size = 2
    learning_rate = 1e-4
    num_epochs = 3
    hidden_dim = 32
    num_layers = 1
    dropout = 0.1
    
    # Instrument parameters
    airs_shape = (32, 356)
    fgs1_shape = (32, 32)
    
    # Training parameters
    num_workers = 1
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    devices = 1

config = Config()

# Load metadata
def load_metadata():
    train_star_info = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/train_star_info.csv")
    train_spectra = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/train.csv")
    wavelengths = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/wavelengths.csv")
    axis_info = pd.read_parquet("/kaggle/input/ariel-data-challenge-2025/axis_info.parquet")
    adc_info = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/adc_info.csv")
    
    return train_star_info, train_spectra, wavelengths, axis_info, adc_info

# Data preprocessing functions
def preprocess_images(data, gain, offset, instrument):
    data = data.astype(np.float64) / gain + offset
    return data

def extract_light_curve(data, instrument):
    light_curve = np.nansum(data, axis=(1, 2))
    return light_curve

# Dataset class
class ArielDataset(Dataset):
    def __init__(self, data_path, star_info, spectra, adc_info, is_train=True, max_samples=2):
        self.data_path = data_path
        self.star_info = star_info
        self.spectra = spectra
        self.adc_info = adc_info
        self.is_train = is_train
        self.planet_ids = star_info['planet_id'].values
        
        self.file_paths = []
        for planet_id in self.planet_ids:
            planet_id_str = str(planet_id)
            planet_path = os.path.join(data_path, planet_id_str)
            
            if not os.path.exists(planet_path):
                continue
                
            airs_files = glob.glob(os.path.join(planet_path, "AIRS-CH0_signal_*.parquet"))
            fgs1_files = glob.glob(os.path.join(planet_path, "FGS1_signal_*.parquet"))
            
            for airs_file in airs_files:
                self.file_paths.append((planet_id_str, 'AIRS-CH0', airs_file))
            
            for fgs1_file in fgs1_files:
                self.file_paths.append((planet_id_str, 'FGS1', fgs1_file))
            
            if len(self.file_paths) >= max_samples:
                break
        
        print(f"Found {len(self.file_paths)} observation files")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        planet_id, instrument, file_path = self.file_paths[idx]
        
        try:
            signal_data = pd.read_parquet(file_path).values
            
            if instrument == 'AIRS-CH0':
                gain = self.adc_info['AIRS-CH0_adc_gain'].values[0]
                offset = self.adc_info['AIRS-CH0_adc_offset'].values[0]
            else:
                gain = self.adc_info['FGS1_adc_gain'].values[0]
                offset = self.adc_info['FGS1_adc_offset'].values[0]
            
            signal_data = preprocess_images(signal_data, gain, offset, instrument)
            
            if instrument == 'AIRS-CH0':
                signal_data = signal_data.reshape(-1, config.airs_shape[0], config.airs_shape[1])
            else:
                signal_data = signal_data.reshape(-1, config.fgs1_shape[0], config.fgs1_shape[1])
            
            light_curve = extract_light_curve(signal_data, instrument)
            
            # Downsample to fixed length
            target_length = 256  # Fixed length for all sequences
            if len(light_curve) > target_length:
                step = len(light_curve) // target_length
                light_curve = light_curve[::step]
                if len(light_curve) > target_length:
                    light_curve = light_curve[:target_length]
            elif len(light_curve) < target_length:
                # Pad with zeros if shorter
                pad_length = target_length - len(light_curve)
                light_curve = np.concatenate([light_curve, np.zeros(pad_length)])
            
            star_params = self.star_info[self.star_info['planet_id'] == int(planet_id)].drop('planet_id', axis=1).values[0]
            
            if self.is_train:
                target = self.spectra[self.spectra['planet_id'] == int(planet_id)].drop('planet_id', axis=1).values[0]
            else:
                target = np.zeros(283)
                
        except Exception as e:
            print(f"Error loading data for {planet_id}: {e}")
            light_curve = np.zeros(256)
            star_params = np.zeros(8)
            target = np.zeros(283)
        
        light_curve = torch.FloatTensor(light_curve)
        star_params = torch.FloatTensor(star_params)
        target = torch.FloatTensor(target)
        
        return {
            'light_curve': light_curve,
            'star_params': star_params,
            'target': target,
            'planet_id': planet_id,
            'instrument': instrument
        }

# Custom collate function
def collate_fn(batch):
    light_curves = [item['light_curve'] for item in batch]
    star_params = torch.stack([item['star_params'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    planet_ids = [item['planet_id'] for item in batch]
    instruments = [item['instrument'] for item in batch]
    
    # All sequences should be same length now (256)
    light_curves = torch.stack(light_curves)
    attention_masks = torch.ones_like(light_curves)
    
    return {
        'light_curves': light_curves,
        'attention_masks': attention_masks,
        'star_params': star_params,
        'targets': targets,
        'planet_ids': planet_ids,
        'instruments': instruments
    }

# Simplified Model architecture
class ArielModel(nn.Module):
    def __init__(self, input_dim, star_param_dim, output_dim, hidden_dim=32, dropout=0.1):
        super(ArielModel, self).__init__()
        
        # Light curve encoder (1D CNN)
        self.light_curve_encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.AdaptiveAvgPool1d(16)  # Fixed size output
        )
        
        # Star parameter encoder
        self.star_encoder = nn.Sequential(
            nn.Linear(star_param_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Combined feature processing
        self.combined_processor = nn.Sequential(
            nn.Linear(32 * 16 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(32 * 16 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()
        )
    
    def forward(self, light_curves, attention_masks, star_params):
        batch_size, seq_len = light_curves.shape
        
        # Reshape for CNN
        light_curves = light_curves.unsqueeze(1)
        
        # Encode light curves with CNN
        cnn_features = self.light_curve_encoder(light_curves)
        cnn_features = cnn_features.view(batch_size, -1)  # Flatten
        
        # Encode star parameters
        star_features = self.star_encoder(star_params)
        
        # Combine features
        combined_features = torch.cat([cnn_features, star_features], dim=1)
        
        # Predict spectrum
        spectrum_pred = self.combined_processor(combined_features)
        
        # Predict uncertainty
        uncertainty = self.uncertainty_head(combined_features)
        
        return spectrum_pred, uncertainty

# PyTorch Lightning Module
class ArielLightningModule(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-4):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = self.gaussian_log_likelihood_loss
    
    def gaussian_log_likelihood_loss(self, pred_mean, pred_std, target):
        variance = pred_std ** 2 + 1e-6
        log_likelihood = -0.5 * (torch.log(2 * torch.tensor(np.pi)) + torch.log(variance) + (target - pred_mean) ** 2 / variance)
        return -log_likelihood.mean()
    
    def training_step(self, batch, batch_idx):
        light_curves = batch['light_curves']
        attention_masks = batch['attention_masks']
        star_params = batch['star_params']
        targets = batch['targets']
        
        pred_mean, pred_std = self.model(light_curves, attention_masks, star_params)
        loss = self.loss_fn(pred_mean, pred_std, targets)
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        light_curves = batch['light_curves']
        attention_masks = batch['attention_masks']
        star_params = batch['star_params']
        targets = batch['targets']
        
        pred_mean, pred_std = self.model(light_curves, attention_masks, star_params)
        loss = self.loss_fn(pred_mean, pred_std, targets)
        
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

# Main training function
def train_model():
    train_star_info, train_spectra, wavelengths, axis_info, adc_info = load_metadata()
    
    train_dataset = ArielDataset(config.train_data_path, train_star_info, train_spectra, adc_info, is_train=True, max_samples=2)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers,
        collate_fn=collate_fn
    )
    
    input_dim = 1
    star_param_dim = train_star_info.drop('planet_id', axis=1).shape[1]
    output_dim = train_spectra.drop('planet_id', axis=1).shape[1]
    
    model = ArielModel(input_dim, star_param_dim, output_dim, 
                      hidden_dim=config.hidden_dim, 
                      dropout=config.dropout)
    
    lightning_model = ArielLightningModule(model, learning_rate=config.learning_rate)
    
    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        accelerator=config.accelerator,
        devices=config.devices,
        log_every_n_steps=1,
        enable_progress_bar=True
    )
    
    trainer.fit(lightning_model, train_loader, val_loader)
    
    return lightning_model, trainer

# Prediction function
def predict(model, data_loader):
    model.eval()
    all_preds = []
    all_uncertainties = []
    all_planet_ids = []
    
    with torch.no_grad():
        for batch in data_loader:
            light_curves = batch['light_curves'].to(model.device)
            attention_masks = batch['attention_masks'].to(model.device)
            star_params = batch['star_params'].to(model.device)
            
            pred_mean, pred_std = model(light_curves, attention_masks, star_params)
            
            all_preds.append(pred_mean.cpu().numpy())
            all_uncertainties.append(pred_std.cpu().numpy())
            all_planet_ids.extend(batch['planet_ids'])
    
    unique_planet_ids = list(set(all_planet_ids))
    predictions = {pid: {'mean': [], 'std': []} for pid in unique_planet_ids}
    
    for i, pid in enumerate(all_planet_ids):
        predictions[pid]['mean'].append(all_preds[i])
        predictions[pid]['std'].append(all_uncertainties[i])
    
    final_predictions = {}
    for pid in unique_planet_ids:
        mean_vals = np.mean(predictions[pid]['mean'], axis=0)
        std_vals = np.mean(predictions[pid]['std'], axis=0)
        final_predictions[pid] = {'mean': mean_vals, 'std': std_vals}
    
    return final_predictions

# Create submission file
def create_submission(predictions, sample_submission_path, output_path='/kaggle/working/submission.csv'):
    sample_submission = pd.read_csv(sample_submission_path)
    
    submission_data = []
    for planet_id in sample_submission['planet_id']:
        planet_id_str = str(planet_id)
        if planet_id_str in predictions:
            pred = predictions[planet_id_str]
            row = [planet_id] + list(pred['mean']) + list(pred['std'])
            submission_data.append(row)
        else:
            row = [planet_id] + [1e-6] * (len(sample_submission.columns) - 1)
            submission_data.append(row)
    
    submission_df = pd.DataFrame(submission_data, columns=sample_submission.columns)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    
    return submission_df

# Main execution
if __name__ == "__main__":
    print("Starting NeurIPS - Ariel Data Challenge 2025 solution...")
    
    print("Training model...")
    try:
        lightning_model, trainer = train_model()
        model = lightning_model.model
    except Exception as e:
        print(f"Training failed: {e}")
        sample_submission = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/sample_submission.csv")
        dummy_preds = {str(pid): {'mean': np.ones(283) * 1e-6, 'std': np.ones(283) * 1e-6} 
                      for pid in sample_submission['planet_id']}
        create_submission(dummy_preds, "/kaggle/input/ariel-data-challenge-2025/sample_submission.csv")
        exit()
    
    print("Loading test data...")
    test_star_info = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/test_star_info.csv")
    adc_info = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/adc_info.csv")
    
    test_spectra = pd.DataFrame({'planet_id': test_star_info['planet_id']})
    for i in range(283):
        test_spectra[f'wavelength_{i}'] = 0.0
    
    test_dataset = ArielDataset(config.test_data_path, test_star_info, test_spectra, adc_info, is_train=False, max_samples=5)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers,
        collate_fn=collate_fn
    )
    
    print("Making predictions...")
    predictions = predict(model, test_loader)
    
    print("Creating submission...")
    sample_submission_path = "/kaggle/input/ariel-data-challenge-2025/sample_submission.csv"
    submission = create_submission(predictions, sample_submission_path)
    
    print("Done!")

