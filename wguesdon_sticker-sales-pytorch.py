import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import holidays
import warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    n_folds = 5
    epochs = 100
    batch_size = 4096
    lr = 1e-3
    weight_decay = 1e-4
    patience = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42
    emb_dim = 8

torch.manual_seed(Config.seed)
np.random.seed(Config.seed)

# ----------------- Data Loading & Preprocessing -----------------
def load_data():
    train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv", parse_dates=['date'])
    test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv", parse_dates=['date'])
    
    # Create combined dataset
    train['is_train'] = 1
    test['is_train'] = 0
    combined = pd.concat([train, test]).sort_values(['country', 'store', 'product', 'date'])
    
    # Create temporal features first
    combined['year'] = combined['date'].dt.year
    combined['month'] = combined['date'].dt.month
    combined['day_of_week'] = combined['date'].dt.dayofweek
    combined['day_of_year'] = combined['date'].dt.dayofyear
    
    # GDP Feature Engineering
    gdp_df = pd.read_csv("/kaggle/input/world-gdpgdp-gdp-per-capita-and-annual-growths/gdp_per_capita.csv")
    years_gdp = [str(y) for y in range(2010, 2021)]
    countries = combined['country'].unique()
    
    gdp_filtered = gdp_df[gdp_df['Country Name'].isin(countries)]
    gdp_filtered = gdp_filtered.set_index('Country Name')[years_gdp]
    
    gdp_ratios = gdp_filtered.stack().reset_index()
    gdp_ratios.columns = ['country', 'year', 'gdp_ratio']
    gdp_ratios['year'] = gdp_ratios['year'].astype(int)
    
    # Merge GDP data
    combined = combined.merge(gdp_ratios, on=['country', 'year'], how='left')
    combined['gdp_ratio'] = combined.groupby('country')['gdp_ratio'].ffill()
    
    # Lag features
    combined = combined.sort_values(['country', 'store', 'product', 'date'])
    for lag in [7, 14, 21, 28]:
        combined[f'lag_{lag}'] = combined.groupby(['country', 'store', 'product'])['num_sold'].shift(lag)
    
    # Split back to train/test
    train = combined[combined['is_train'] == 1].drop(columns=['is_train'])
    test = combined[combined['is_train'] == 0].drop(columns=['is_train', 'num_sold'])
    
    # Handle missing values
    train.dropna(subset=['num_sold'], inplace=True)
    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)
    
    return train, test

# ----------------- Feature Engineering -----------------
def add_features(df):
    # Cyclical features
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Holiday features
    country_map = {
        'Norway': 'NO', 'Finland': 'FI', 'Sweden': 'SE',
        'Denmark': 'DK', 'Netherlands': 'NL', 'Belgium': 'BE',
        'Canada': 'CA', 'Italy': 'IT', 'Kenya': 'KE', 'Singapore': 'SG'
    }
    
    df['is_holiday'] = 0
    for country in df['country'].unique():
        if country not in country_map: 
            continue
        try:
            holidays_list = holidays.CountryHoliday(country_map[country])
            mask = (df['date'].isin(holidays_list)) & (df['country'] == country)
            df.loc[mask, 'is_holiday'] = 1
        except:
            continue
    
    return df.drop(columns=['date'])

# ----------------- Neural Network -----------------
class SalesModel(nn.Module):
    def __init__(self, emb_sizes, n_cont):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(ni, nd) for ni, nd in emb_sizes])
        self.emb_drop = nn.Dropout(0.1)
        n_emb = sum(nd for _, nd in emb_sizes)
        
        self.main = nn.Sequential(
            nn.Linear(n_emb + n_cont, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x_cat, x_cont):
        embeddings = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)
        x = torch.cat([x, x_cont], 1)
        return self.main(x).squeeze()

# ----------------- Data Pipeline -----------------
class SalesDataset(Dataset):
    def __init__(self, X, y=None):
        self.cat = X[cat_cols].values.astype(np.int64)
        self.cont = X[cont_cols].values.astype(np.float32)
        self.y = y.values.astype(np.float32) if y is not None else None

    def __len__(self):
        return len(self.cat)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return (torch.tensor(self.cat[idx], dtype=torch.long),
                    torch.tensor(self.cont[idx], dtype=torch.float),
                    torch.tensor(self.y[idx], dtype=torch.float))
        return (torch.tensor(self.cat[idx], dtype=torch.long),
                torch.tensor(self.cont[idx], dtype=torch.float))

# ----------------- Training -----------------
def train_model():
    train, test = load_data()
    train = add_features(train)
    test = add_features(test)
    
    # Prepare features
    global cat_cols, cont_cols
    cat_cols = ['country', 'store', 'product', 'day_of_week', 'is_holiday']
    cont_cols = [c for c in train.columns if c not in cat_cols + ['num_sold', 'id']]
    
    # Encode categoricals
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]])
        le.fit(combined)
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
        encoders[col] = le
    
    # Scale features
    scaler = StandardScaler()
    train[cont_cols] = scaler.fit_transform(train[cont_cols])
    test[cont_cols] = scaler.transform(test[cont_cols])
    
    # Prepare data
    X, y = train.drop(columns=['num_sold', 'id']), np.log1p(train['num_sold'])
    X_test = test[X.columns]
    
    # Cross-validation
    gkf = GroupKFold(Config.n_folds)
    test_preds = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, groups=train['year'])):
        print(f"\nFold {fold+1}/{Config.n_folds}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Data loaders
        train_ds = SalesDataset(X_train, y_train)
        val_ds = SalesDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=Config.batch_size*2, shuffle=False)
        
        # Model setup
        emb_sizes = [(len(encoders[col].classes_), Config.emb_dim) for col in cat_cols]
        model = SalesModel(emb_sizes, len(cont_cols)).to(Config.device)
        opt = optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=3)
        criterion = nn.MSELoss()
        
        best_loss = float('inf')
        for epoch in range(Config.epochs):
            model.train()
            total_loss = 0
            for x_cat, x_cont, y_batch in train_loader:
                x_cat, x_cont, y_batch = x_cat.to(Config.device), x_cont.to(Config.device), y_batch.to(Config.device)
                opt.zero_grad()
                preds = model(x_cat, x_cont)
                loss = criterion(preds, y_batch)
                loss.backward()
                opt.step()
                total_loss += loss.item() * x_cat.size(0)
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x_cat, x_cont, y_batch in val_loader:
                    x_cat, x_cont, y_batch = x_cat.to(Config.device), x_cont.to(Config.device), y_batch.to(Config.device)
                    preds = model(x_cat, x_cont)
                    val_loss += criterion(preds, y_batch).item() * x_cat.size(0)
            
            val_loss /= len(val_ds)
            scheduler.step(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), f'best_model_fold{fold}.pth')
            
            print(f"Epoch {epoch+1:02d} | Val Loss: {val_loss:.4f}")
        
        # Predict test
        model.load_state_dict(torch.load(f'best_model_fold{fold}.pth'))
        test_ds = SalesDataset(X_test)
        test_loader = DataLoader(test_ds, batch_size=Config.batch_size*2, shuffle=False)
        
        fold_preds = []
        with torch.no_grad():
            for x_cat, x_cont in test_loader:
                x_cat, x_cont = x_cat.to(Config.device), x_cont.to(Config.device)
                preds = model(x_cat, x_cont).cpu().numpy()
                fold_preds.append(preds)
        
        test_preds += np.concatenate(fold_preds) / Config.n_folds
    
    # Generate submission
    submission = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")
    submission['num_sold'] = np.expm1(test_preds)
    submission['num_sold'] = submission['num_sold'].clip(lower=1).round().astype(int)
    submission.to_csv('submission.csv', index=False)
    print("Submission saved!")

if __name__ == "__main__":
    train_model()

