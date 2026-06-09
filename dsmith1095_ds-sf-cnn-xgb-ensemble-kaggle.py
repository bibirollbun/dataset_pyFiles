# Imports and Configuration

import os                                                             # OS interaction
import math
import numpy as np                                                    # arrays, functions
import pandas as pd                                                   # dataframes, functions
from PIL import Image                                                 # processing images

from sklearn.model_selection import train_test_split, KFold           # Splitting data for training, testing, k-fold cross validation
from sklearn.preprocessing import LabelEncoder, StandardScaler        # Encoding and scaling data for normalization
from sklearn.ensemble import RandomForestClassifier                   # Imputation model for categorical data -- Needed afer XGB??
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import torch
import torch.nn as nn                                                 # neural network via pytorch
from torch.utils.data import Dataset, DataLoader, Subset              # creating/loading datasets for nn
import torchvision.transforms as transforms                           # transform for image augmentation
import torchvision.models as models                                   # use of model functions

# Global settings
np.random.seed(42)                                                    # Set random seeds for both torch and numpy for reproducability
torch.manual_seed(42)

SAMPLE_PERCENT = 0.1                                                  
NUM_EPOCHS = 100
BATCH_SIZE = 32




# Set paths for data
data_dir = os.path.dirname('../input/applications-of-deep-learning-wustl-summer-2025/train.csv')
train_csv = os.path.join(data_dir, 'train.csv')
test_csv = os.path.join(data_dir, 'test.csv')

# Load data to csvs
df_train_full = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def encode_categoricals(df_train, df_test, cat_cols):
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_train[col] = df_train[col].astype(str)
        le.fit(df_train[col])
        df_train[col] = le.transform(df_train[col])

        if col in df_test.columns:
            df_test[col] = df_test[col].astype(str).apply(lambda x: x if x in le.classes_ else 'unknown')
            le.classes_ = np.append(le.classes_, 'unknown')
            df_test[col] = le.transform(df_test[col])

        encoders[col] = le
    return encoders



# Imputation Function for categorical, numeric, feature columns

def impute_missing_values(df, categorical_cols, numeric_cols, feature_cols):
    """ For each column with nulls, train a model to predict null values based on other features in the dataset.
        Using Random Forest Classifier for categorical values and median for numeric"""
    
    df = df.copy()
    df_encoded = pd.get_dummies(df[feature_cols], drop_first=True, dummy_na=True)    # Use one hot encoding on feature columns
    print("Starting imputation...")                                                  # log process start
    
    # Categorical - Classification
    for col in categorical_cols:
        if df[col].isnull().sum() == 0:                                              # skip columns w/o nulls
            continue
        mask = df[col].notnull()                                                     # create mask filter for non-nulls
        X_train = df_encoded.loc[mask].dropna()                                      # use encoded cols (X) with non-null target col to train
        y_train = df.loc[mask, col][X_train.index]                                   # select non-null target (y) for classification
        model = RandomForestClassifier(n_estimators=100, random_state=42)            # classification model for categorical values
        model.fit(X_train, y_train)

        X_pred = df_encoded.loc[df[col].isnull()].dropna()                           # make prediction on x values
        df.loc[X_pred.index, col] = model.predict(X_pred)                            # update dataframe with results

    # Numeric - Median
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    print("Imputation completed.")                                                   # log process stop
    return df



# Preprocessing

def preprocess_tabular(df_train, df_test, target_col, id_col):
    # ignores target and id columns, assigns the rest to tabular_cols variable
    ignore_cols = [id_col, target_col]
    tabular_cols = [c for c in df_train.columns if c not in ignore_cols]

    # Assigns numeric columns and categorical columns to respective variables per datatype for training data
    numeric_cols = df_train[tabular_cols].select_dtypes(include='number').columns.tolist()
    cat_cols = df_train[tabular_cols].select_dtypes(include=['object', 'category']).columns.tolist()

    # Refines numeric/categorical cols to be those in testing data
    numeric_cols = [c for c in numeric_cols if c in df_test.columns]
    cat_cols = [c for c in cat_cols if c in df_test.columns]

    # Scales data in numeric columns to standardize # Not necessary to scale for XGBoost or CatBoost but retained for models (NN/linear) that might benefit
    scaler = StandardScaler()
    if numeric_cols:
        df_train[numeric_cols] = scaler.fit_transform(df_train[numeric_cols])
        df_test[numeric_cols] = scaler.transform(df_test[numeric_cols])

    # Call encoding function
    encoders = encode_categoricals(df_train, df_test, cat_cols)

    # Assigns full list of updated features to full list of features
    tabular_feats = numeric_cols + cat_cols
    return tabular_feats, encoders, len(tabular_feats)


# Dataset Class

class MultiModalDataset(Dataset):
    def __init__(self, df, data_dir, id_col, tabular_feats, target_col=None, transform=None, is_train=True):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.id_col = id_col
        self.tabular_feats = tabular_feats
        self.target_col = target_col
        self.transform = transform
        self.is_train = is_train

        self.tab_data = torch.tensor(df[tabular_feats].fillna(0).values, dtype=torch.float32)                  # Extract features, fillna with 0, 
        if is_train:
            self.targets = torch.tensor(df[target_col].values, dtype=torch.float32)                            # Will this always be true since we're setting default?

    def __len__(self):
        """ Mark the size of the dataset for later use (i.e., batching)"""
        return len(self.df)

    def __getitem__(self, idx):
        """Iterate through dataset rows. 
            For imgs, build path to img, use predefined img to transform (if available).
            For tabular, get data as a tensor (see self.tab_data).
            Selective return results for training vs. testing."""
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, f"{int(row[self.id_col])}.jpg")
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        tab = self.tab_data[idx]
        return (img, tab, self.targets[idx]) if self.is_train else (img, tab, row[self.id_col])

# Get image only dataset

class ImageOnlyDataset(Dataset):
    def __init__(self, df, data_dir, id_col, transform):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.id_col = id_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, f"{int(row[self.id_col])}.jpg")
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, row[self.id_col]



# Class for adding Gaussian Noise (more img transformation)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


# Image Transformations

    # """Transformations on the image data to improve generalization,
    #     simulate real-world variability to improve predictions """

transform = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),                  # combine RandomCrop + Resize (zoom/framing variability)
    transforms.RandomRotation(degrees=20),                                            # for possibility of real-world change in orientation
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),                        # add some perspective warp
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),                      # play with sharpness in image
    transforms.RandomAutocontrast(p=0.3),                                             # play with contrast
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),                    # random blur applied
    transforms.RandomHorizontalFlip(p=0.5),                                           # common, covers generalization per horiz symmetry
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),   # for lighting/color imbalance
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),                       # for real-world, slight scale/rotation changes
    transforms.ToTensor(),                                                            # Must come after all PIL-based transforms except noise
    AddGaussianNoise(mean=0.0, std=0.03),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
])



# Learning Models

class MultiModalModel(nn.Module):
    """Merges models and passes through final set of layers for prediction"""
    def __init__(self, num_tab_feats, backbone='resnet18', dropout_prob=0.3):
        super().__init__()
        cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)              # use resnet18 for image feature extraction
        
        for param in cnn.parameters():                                              # set parameters to customize resnet
            param.requires_grad = False                                             # start with freezing layers then gradually unfreeze them

        for param in cnn.layer4.parameters():
            param.requires_grad = True                                              # calculate gradients for parameters so they're updated during backpropgagation
        
        feat_dim = cnn.fc.in_features                                               # process image data, needs done to use with self.head later
        cnn.fc = nn.Identity()                                                      
        self.cnn = cnn

        self.tab_net = nn.Sequential(                                               # process tabular data with feed forward network
            nn.Linear(num_tab_feats, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        self.head = nn.Sequential(                                                 # concatenates img (feat_dim) and tab features (128) into a vector
            nn.Linear(feat_dim + 128, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 1)
        )

    def forward(self, img, tab):
        img_feat = self.cnn(img)                                                  # process image data
        tab_feat = self.tab_net(tab)                                              # process tabular data
        x = torch.cat([img_feat, tab_feat], dim=1)                                # combine image/tab data
        return self.head(x)                                                       # return final prediction

class ImageFeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.cnn = model.cnn  # Only keep image model
    def forward(self, img):
        return self.cnn(img)



# Image feature extraction

def extract_image_features(model, df, data_dir, transform, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImageFeatureExtractor(model).to(device)
    model.eval()

    dataset = ImageOnlyDataset(df, data_dir, id_col='id', transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_features, all_ids = [], []

    with torch.no_grad():
        for imgs, ids in loader:
            imgs = imgs.to(device)
            feats = model(imgs)
            all_features.append(feats.cpu().numpy())
            all_ids.extend(ids.tolist())

    features_array = np.vstack(all_features)
    return pd.DataFrame(features_array, index=all_ids).sort_index()


# Training Model

def train_model(model, train_loader, val_loader, device, num_epochs):
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)  # updated after unfreezing to pick up new trainable parameters
    criterion = nn.MSELoss()                                                                      # loss function
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(                                       # scheduler for image data
        optimizer, mode='min', factor=0.5, patience=1, verbose=True
    )
    model.to(device)
    
    best_rmse = float('inf')                                                                  # early stopping setup
    patience, patience_counter = 30, 0                                                         # early stopping params

    for epoch in range(num_epochs):
        """ For each epoch, train the model, gather running loss, predictions, targets"""
        model.train()
        running_loss, all_preds, all_targets = 0, [], []

        for imgs, tabs, ys in train_loader:
            # Move data to device
            imgs, tabs, ys = imgs.to(device), tabs.to(device), ys.unsqueeze(1).to(device)    #for img, tab, target move to CPU/GPU, reshape target for model

            # Ensure data is on GPU
            # assert imgs.device.type == 'cuda', "Images not on GPU"
            # assert tabs.device.type == 'cuda', "Tabular data not on GPU"
            # assert ys.device.type == 'cuda', "Targets not on GPU"            
            
            # Forward pass 
            preds = model(imgs, tabs)                                                        #pass img, tabular data through model for predictions
            
            # Ensure outputs are on GPU
            # assert preds.device.type == 'cuda', "Model outputs not on GPU"            
            
            loss = criterion(preds, ys)                                                      #calculate loss for predictions/targets

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)                                       #accumulate total loss per epoch by batch
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(ys.detach().cpu().numpy())                                    #prep for calculating RMSE


        
        train_rmse = mean_squared_error(np.vstack(all_targets), np.vstack(all_preds), squared=False)
        print(f"Epoch {epoch+1} | Loss: {running_loss/len(train_loader.dataset):.4f} | Train RMSE: {train_rmse:.4f}")

        # Validation
        if val_loader is not None:
            model.eval()
            val_preds, val_targets = [], []
            
            with torch.no_grad():
                for imgs, tabs, ys in val_loader:
                    imgs, tabs, ys = imgs.to(device), tabs.to(device), ys.unsqueeze(1).to(device)
                    
                    # GPU checks
                    # assert imgs.device.type == 'cuda'
                    # assert tabs.device.type == 'cuda'
                    # assert ys.device.type == 'cuda'
                    
                    preds = model(imgs, tabs)
    
                    # GPU check
                    # assert preds.device.type == 'cuda'
        
                    val_preds.append(preds.cpu().numpy())
                    val_targets.append(ys.cpu().numpy())
    
            val_rmse = mean_squared_error(np.vstack(val_targets), np.vstack(val_preds), squared=False) # End of epochs
            print(f"Epoch {epoch+1} | Val RMSE: {val_rmse:.4f}")
    
            scheduler.step(val_rmse)                                                                   # Use scheduler, prevent overfitting on images
            
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                patience_counter = 0
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break




# Submission Generation

def generate_submission(model, test_loader, device, id_col, target_col, filename='submission.csv'):
    model.eval()
    ids, preds = [], []
    with torch.no_grad():
        for imgs, tabs, img_ids in test_loader:
            imgs, tabs = imgs.to(device), tabs.to(device)
            outputs = model(imgs, tabs).squeeze(1).cpu().numpy()
            ids.extend(img_ids.tolist())
            preds.extend(outputs.tolist())

    df_submission = pd.DataFrame({id_col: ids, target_col: preds})
    df_submission.to_csv(filename, index=False)
    print(f"Saved submission file: {filename}")


# Load training data (use imputed if available)
imputed_path = 'train_imputed.csv'
if os.path.exists(imputed_path):
    print("Loading pre-imputed training data...")
    df_train_full = pd.read_csv(imputed_path)
else:
    print("Loading raw training data from Kaggle input directory...")
    data_dir = os.path.dirname('../input/applications-of-deep-learning-wustl-summer-2025/train.csv')
    train_csv = os.path.join(data_dir, 'train.csv')
    df_train_full = pd.read_csv(train_csv)

    # Define imputation columns (redefine here if needed)
    categorical_cols = ['gem_type', 'primary_color', 'origin_culture', 'inscription_script',
                        'pattern_type', 'environment_type', 'historical_significance', 'craftsmanship_level']
    numeric_cols = df_train_full.select_dtypes(include=['int64', 'float64']).columns.tolist()
    feature_cols = ['item_type', 'primary_material', 'secondary_material', 'surface_texture',
                    'rarity_level', 'oxidation_level', 'restoration_status', 'authentication_status',
                    'shine_factor', 'description']

    print("Imputing missing values...")
    df_train_full = impute_missing_values(df_train_full, categorical_cols, numeric_cols, feature_cols)
    df_train_full.to_csv(imputed_path, index=False)
    print("Saved imputed training data.")


# Usage outline

# 2. Sample subset for fast iteration
# df_train = df_train_full.sample(frac=SAMPLE_PERCENT, random_state=42).reset_index(drop=True)

df_train = df_train_full.copy().reset_index(drop=True)

# 3. Preprocess tabular data (scaling, encoding)
tabular_feats, encoders, num_tab_feats = preprocess_tabular(df_train, df_test, target_col='preservation_score', id_col='id')

# 4. Add in XGBoost Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
kf = KFold(n_splits=5, shuffle=True, random_state=42)

val_scores, blend_preds_all, targets_all = [], [], []

if not os.path.exists('best_model.pth'):
    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train)):
        print(f"=== Fold {fold + 1} ===")
        df_train_fold = df_train.iloc[train_idx].reset_index(drop=True)
        df_val_fold   = df_train.iloc[val_idx].reset_index(drop=True)
    
        # XGBoost on tabular
        X_train_tab = df_train_fold[tabular_feats]
        y_train_tab = df_train_fold['preservation_score']
        X_val_tab   = df_val_fold[tabular_feats]
        y_val       = df_val_fold['preservation_score']
    
        xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, random_state=42)
        xgb_model.fit(X_train_tab, y_train_tab)
        xgb_preds = xgb_model.predict(X_val_tab)
    
        # CNN
        train_dataset = MultiModalDataset(df_train_fold, data_dir, 'id', tabular_feats, 'preservation_score', transform, is_train=True)
        val_dataset   = MultiModalDataset(df_val_fold, data_dir, 'id', tabular_feats, 'preservation_score', transform, is_train=True)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
        model = MultiModalModel(num_tab_feats).to(device)
        train_model(model, train_loader, val_loader, device, NUM_EPOCHS)
    
        model.eval()
        cnn_preds = []
        with torch.no_grad():
            for imgs, tabs, _ in val_loader:
                imgs, tabs = imgs.to(device), tabs.to(device)
                pred = model(imgs, tabs).squeeze(1).cpu().numpy()
                cnn_preds.extend(pred.tolist())
        
        cnn_preds = np.array(cnn_preds)
        xgb_preds = np.array(xgb_preds)
        y_val_np  = y_val.to_numpy()
    
        blended = (cnn_preds + xgb_preds) / 2
        rmse = mean_squared_error(y_val_np, blended, squared=False)
        val_scores.append(rmse)
        blend_preds_all.extend(blended)
        targets_all.extend(y_val_np)
    
        print(f"Fold {fold+1} RMSE: {rmse:.4f}")

    print(f"Avg K-Fold Blended RMSE: {np.mean(val_scores):.4f}")

else:
    print("Skipping training — found saved model: best_model.pth")
    model = MultiModalModel(num_tab_feats).to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()


# === Fit stacking model on full tabular training data ===
cat_model = CatBoostRegressor(verbose=0, random_state=42)
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, random_state=42)

stack_model = StackingRegressor(
    estimators=[('xgb', xgb_model), ('cat', cat_model)],
    final_estimator=LinearRegression()
)
stack_model.fit(df_train[tabular_feats], df_train['preservation_score'])

# === Fit CNN model on full data ===

if os.path.exists('best_model.pth'):
    model = MultiModalModel(num_tab_feats).to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()
    print("Loaded trained CNN model.")
else:
    model = MultiModalModel(num_tab_feats).to(device)
    full_dataset = MultiModalDataset(df_train, data_dir, 'id', tabular_feats, 'preservation_score', transform, is_train=True)
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_model(model, full_loader, None, device, NUM_EPOCHS)
    
    # === Predict CNN on test set ===
    cnn_test_preds, test_ids = [], []
    test_dataset = MultiModalDataset(df_test, data_dir, 'id', tabular_feats, transform=transform, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model.eval()
    with torch.no_grad():
        for imgs, tabs, ids in test_loader:
            imgs, tabs = imgs.to(device), tabs.to(device)
            preds = model(imgs, tabs).squeeze(1).cpu().numpy()
            cnn_test_preds.extend(preds.tolist())
            test_ids.extend(ids.tolist())
    
    # === Predict stacking model on test set ===
    stack_test_preds = stack_model.predict(df_test[tabular_feats])
    
    # === Blend predictions ===
    blended_test_preds = 0.7 * stack_test_preds + 0.3 * np.array(cnn_test_preds)
    
    # === Save submission ===
    submission = pd.DataFrame({'id': test_ids, 'preservation_score': blended_test_preds})
    submission['id'] = submission['id'].astype(int)
    submission.to_csv('submission.csv', index=False)
    print("Submission saved.")


# === Re-run inference without retraining ===

# Load imputed and preprocessed data
df_train = pd.read_csv('train_imputed.csv')
df_test = pd.read_csv(test_csv)

# Sample the full data (no sampling for full predictions)
# Preprocess again to get tabular features
df_train, df_test, tabular_feats, encoders, num_tab_feats = preprocess_tabular(df_train, df_test, target_col='preservation_score', id_col='id')

# Load XGBoost model (if not saved, you’ll need to retrain or pickle it)
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, random_state=42)
xgb_model.fit(df_train[tabular_feats], df_train['preservation_score'])

# Recreate and load trained CNN model
model = MultiModalModel(num_tab_feats).to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Prepare test loader
test_dataset = MultiModalDataset(df_test, data_dir, 'id', tabular_feats, transform=transform, is_train=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Generate predictions with CNN model
cnn_preds, ids = [], []
with torch.no_grad():
    for imgs, tabs, id_batch in test_loader:
        imgs, tabs = imgs.to(device), tabs.to(device)
        preds = model(imgs, tabs).squeeze(1).cpu().numpy()
        cnn_preds.extend(preds)
        ids.extend(id_batch)

# Predict with XGBoost
xgb_preds = xgb_model.predict(df_test[tabular_feats])

# Blend and create new submission
final_preds = (np.array(cnn_preds) + xgb_preds) / 2
submission = pd.DataFrame({'id': ids, 'preservation_score': final_preds})
submission.to_csv('submission_rerun.csv', index=False)

print("New submission file saved as submission_rerun.csv")




