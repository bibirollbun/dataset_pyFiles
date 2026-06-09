import numpy as np
import pandas as pd
#from edazer import Edazer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch import optim

torch.manual_seed(42)




df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
df.set_index("id", inplace=True)

actual_test_df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'


#Edazer(df).summarize_df()


df.columns


df.shape


X, y = df.drop(columns=['BeatsPerMinute']), df['BeatsPerMinute']


other_features = [col for col in X.columns if col != 'TrackDurationMs']

track_duration_pipeline = make_pipeline(
    StandardScaler(),
    MinMaxScaler()
)

scaler = ColumnTransformer([
    ("Trackduration", track_duration_pipeline, ['TrackDurationMs']),
    ("others", StandardScaler(), other_features)
])

def scale_features(X, fit=True) -> pd.DataFrame:
    if fit:
        return pd.DataFrame(scaler.fit_transform(X), columns=scaler.feature_names_in_)
    return pd.DataFrame(scaler.transform(X), columns=scaler.feature_names_in_)


# train-val-test : 70-10-20

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = scale_features(X_train)
X_test = scale_features(X_test, fit=False)

target_scaler = StandardScaler()
y_train = target_scaler.fit_transform(y_train.to_frame()).flatten()
y_test = target_scaler.transform(y_test.to_frame())

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/8, random_state=42)

input_dim = X_train.shape[1]


class BMPSDatatset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        self.feature_matrix = torch.from_numpy(X.values).to(torch.float32)
        self.targets = torch.from_numpy(y).to(torch.float32)
    
    def __len__(self):
        return len(self.feature_matrix)
    
    def __getitem__(self, index):
        return self.feature_matrix[index], self.targets[index]

train_dataset = BMPSDatatset(X_train, y_train)
val_dataset = BMPSDatatset(X_val, y_val)
test_dataset = BMPSDatatset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size= 64, shuffle= True)
val_loader = DataLoader(val_dataset, batch_size= 64, shuffle= False)
test_loader = DataLoader(test_dataset, batch_size= 64, shuffle= False)



class BmpsNet(nn.Module):
    def __init__(self, input_dim):
        super(BmpsNet, self).__init__()
        
        self.input_dim = input_dim
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(0.01),
            
            nn.Linear(32, 16),
            nn.LeakyReLU(0.01),

            nn.Linear(16, 8),
            nn.LeakyReLU(0.01),

            nn.Linear(8, 1)            
            )
    def forward(self, X):
        return self.model(X)

bmps_nn = BmpsNet(input_dim)


def train(train_loader):
    bmps_nn.train()

    #losses = []
    N_EPOCHS = 30
    lr = 1e-2

    criterion = nn.MSELoss()
    optimizer = optim.SGD(bmps_nn.parameters(), lr=lr)

    for epoch in range(N_EPOCHS):
        epoch_loss = 0
        n_batches = 0
        
        for batch_features, batch_targets in train_loader:
            batch_targets = batch_targets.view(-1, 1)
            
            outputs = bmps_nn(batch_features)
            loss = criterion(outputs, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                 # Convert predictions and targets back to original scale
                outputs_unscaled = target_scaler.inverse_transform(outputs.numpy())
                targets_unscaled = target_scaler.inverse_transform(batch_targets.numpy())

                # Compute RMSE in original scale
                batch_rmse = np.sqrt(np.mean((outputs_unscaled - targets_unscaled) ** 2))
                epoch_loss += batch_rmse
                n_batches += 1
        
            #epoch_loss += loss.item()
            #n_batches += 1
        avg_rmse = epoch_loss / n_batches
        #losses.append(avg_rmse)
        print(f"epoch: {epoch+1} -> {avg_rmse:.4f}")
        

train(train_loader)


@torch.no_grad()
def test(test_loader):
    bmps_nn.eval()
    
    sum_sqerr = 0.0
    total_samples = 0
    
    for batch_features, batch_targets in test_loader:
        preds = bmps_nn(batch_features)

        batch_sqerr = (preds - batch_targets) ** 2

        sum_sqerr += torch.sum(batch_sqerr).item()
        total_samples += batch_sqerr.numel()  

    overall_rmse = (sum_sqerr / total_samples) ** 0.5
    return overall_rmse

overall_rmse = test(test_loader)


overall_rmse


actual_test_df = scale_features(actual_test_df, fit=False)


with torch.no_grad():
    bmps_nn.eval()
    final_preds = bmps_nn(torch.from_numpy(actual_test_df.values).to(torch.float32)).numpy()


idx = range(524164, 524164 + len(actual_test_df))


final_preds = np.c_[idx, target_scaler.inverse_transform(final_preds).ravel()]


final_preds


final_preds_df = pd.DataFrame(final_preds)


final_preds_df[0] = final_preds_df[0].astype(np.int32)
final_preds_df.rename(columns={0:'id', 1:'BeatsPerMinute'}, inplace=True)


final_preds_df.to_csv("subs.csv", index=False)




