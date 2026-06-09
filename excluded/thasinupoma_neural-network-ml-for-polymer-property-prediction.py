# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#  Install networkx 
!pip install /kaggle/input/mordred-offline-use/networkx-2.8.8-py3-none-any.whl

# Install mordred (prebuilt wheel - no build required)
!pip install /kaggle/input/mordred-offline-use/mordred-1.2.0-py3-none-any.whl
# Install rdkit
!pip install /kaggle/input/mordred-offline-use/rdkit_pypi-2022.9.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl



# Import libraries
import torch
import joblib
import torch.nn as nn
from rdkit import Chem
import warnings
from rdkit.Chem import Descriptors
from mordred import Calculator,descriptors
from mordred.error import Missing, Error
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import torch.optim as optim


# Skip Run time warning
warnings.filterwarnings("ignore", category=RuntimeWarning)
# Check the directory of dataset
os.listdir('/kaggle/input/neurips-open-polymer-prediction-2025')


# Read train data and check 
train_data = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")

# Print data info
print(f'Number of Rows in train data : {train_data.shape[0]}, Number of columns in test data: {train_data.shape[1]}')
print(f'Number of null values in each column of train data:\n{train_data.isnull().sum()}')
# Check training data
train_data.head()


# Read test data and check 
test_data = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")

# Print data info
print(f'Number of Rows in test data : {test_data.shape[0]}, Number of columns in test data: {test_data.shape[1]}')
test_data


# Target values
target_values = ['Tg','FFV','Tc','Density','Rg']
print(f"Total number of rows : {train_data.shape[0]}")
print(f"Number of missing values in each target columns :")

# Check number of null values in each target columns
for target in target_values:
    print(f"{target} : {train_data[target].isnull().sum()}")


def get_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    calc = calc = Calculator([descriptors.CPSA,descriptors.BCUT,descriptors.HydrogenBond], ignore_3D=True)
    
    if mol is None:
        return
        
    # RDkit basic descriptors
    rdkit_desc = {
    'MolWt': Descriptors.MolWt(mol) if mol is not None else None,
    'LogP': Descriptors.MolLogP(mol) if mol is not None else None,
    'TPSA': Descriptors.TPSA(mol) if mol is not None else None,
    'HDonors': Descriptors.NumHDonors(mol) if mol is not None else None,
    'HAcceptors': Descriptors.NumHAcceptors(mol)if mol is not None else None,
    'RotBonds': Descriptors.NumRotatableBonds(mol)if mol is not None else None
    }

    # Mordred descriptors 
    mordred_series = calc(mol)
    mordred_clean = {
        str(k): v for k, v in mordred_series.items()
        if v is not None and not isinstance(v, (Missing, Error))
    }

    # Combine RDKit and Mordred descriptors
    combined_desc = {**rdkit_desc, **mordred_clean}
    return combined_desc
    

# Example output of get_descriptors function
get_descriptors('*CC(*)c1ccccc1C(=O)OCCCCCC')


# Storing test data in initial format before feature extraction
test_data_temp = test_data
# Extract features from smiles and merge with original train and test data
train_data = train_data.join(train_data['SMILES'].apply(get_descriptors).apply(pd.Series))
test_data = test_data.join(test_data['SMILES'].apply(get_descriptors).apply(pd.Series))


# Remove SMILES column as already extracted feature and id is not a feature
train_data= train_data.drop(columns=['SMILES','id'])
test_data= test_data.drop(columns=['SMILES','id'])



# Function for train and evaluate ML models 
def MLModels(model_name,target_value,X_train_scaled,Y_train,X_val_scaled,Y_val):
    if model_name == 'Linear Regression':
        model = LinearRegression()
    elif model_name == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else :return None
    
    model.fit(X_train_scaled,Y_train)
    
    # Predict on scaled test data
    Y_pred = model.predict(X_val_scaled)
    
    # Compute performance metrics
    mse = mean_squared_error(Y_val, Y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y_val, Y_pred)
    r2 = r2_score(Y_val, Y_pred)

       
    # Print results
    print(f"\nPerformance Evaluation of {model_name} for {target_value} Prediction")
    print(f"RMSE: {rmse :.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"RÂ² Score: {r2:.2f}")


# Define Custom model
class Predictor_Model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Output value
        )

    def forward(self, x):
        return self.model(x)



# define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the directory if it doesn't exist
os.makedirs("/kaggle/working/saved_models", exist_ok=True)


def Train_and_Validate_NN(X_train_scaled,X_val_scaled,Y_train,Y_val,target_value):
    
    # Convert inputs and targets to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32).view(-1, 1)
    Y_val_tensor = torch.tensor(Y_val.values, dtype=torch.float32).view(-1, 1)
    
    # Combine inputs and outputs into a dtaset
    train_dataset = TensorDataset(X_train_tensor,Y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor,Y_val_tensor)
    
    # Create DataLoader with batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Create validation DataLoader with batching and without shuffling
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    """
    Train Model
    """
    # Lists to store losses
    train_losses= []
    val_losses = []
    # Define model, optimizer and num_epochs
    model_ann = Predictor_Model(input_size = X_train_scaled.shape[1]).to(device) 
    optimizer_ann = optim.Adam(model_ann.parameters(), lr=0.001)
    best_loss_ann = float("inf")
    num_epochs = 50
    criterion = nn.MSELoss()
    for epoch in range(num_epochs):
        model_ann.train()
        epoch_train_loss = 0.0
    
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer_ann.zero_grad()
            pred = model_ann(X_batch)
            loss = criterion(pred, Y_batch)
            loss.backward()
            optimizer_ann.step()
            epoch_train_loss += loss.item()
    
        avg_train = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train)

                # Validation
        model_ann.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for X_val, Y_val in val_loader:
                X_val, Y_val = X_val.to(device), Y_val.to(device)
                val_pred = model_ann(X_val)
                val_loss = criterion(val_pred, Y_val)
                epoch_val_loss += val_loss.item()
    
        avg_val = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val)
    
        if avg_val < best_loss_ann:
            best_loss_ann = avg_val
            torch.save(model_ann.state_dict(), f"/kaggle/working/saved_models/{target_value}_Model.pth")
    
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Train Loss: {avg_train:.6f}, Val Loss: {avg_val:.6f}")



# Input features and Target values
input_features = test_data.columns.to_list()
target_values = ['Tg','FFV','Tc','Density','Rg']


# Check performance of each ML models to predict each target values
for target_value in target_values:
    #Handle missing data in target columns
    train_target =  train_data.dropna(subset=[target_value])
    # Split input and output data
    X_train = train_target[input_features]
    Y_train = train_target[target_value]
    # Handle null values of input feature
    X_train = X_train.fillna(X_train.median())
    # Split data for train and validation
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
    
    # Test data
    test_target = test_data
    X_test = test_target[input_features]
    X_test = X_test.fillna(X_train.median())
    # Apply scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)          

    
    """
    # Linear Regression Model
    # """
    
    MLModels('Linear Regression',target_value,X_train_scaled,Y_train,X_val_scaled,Y_val)
    
    
    """
    Random Forest Model
    """
    MLModels('Random Forest',target_value,X_train_scaled,Y_train,X_val_scaled,Y_val)




# To save scaling to apply same in test data
os.makedirs("/kaggle/working/scalers", exist_ok=True)

# Check performance of each NN model to predict each target values
for target_value in target_values:
    #Handle missing data in target columns
    train_target =  train_data.dropna(subset=[target_value])
    # Split input and output data
    X_train = train_target[input_features]
    Y_train = train_target[target_value]
    # Handle null values of input feature
    X_train = X_train.fillna(X_train.median())
    # Split data for train and validation
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
    
    # Apply scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)          
 
    # Saving scale to appy in test data also
    joblib.dump(scaler, f'/kaggle/working/scalers/{target_value}_scaler.pkl')
    """
    Custom NN Model
    """
    print(f"Training for Predicting {target_value}:")
    Train_and_Validate_NN(X_train_scaled,X_val_scaled,Y_train,Y_val,target_value)


# Define dictionary to store all prediction
all_predictions = {}

for target_value in target_values:
    # Load Scaler 
    scaler = joblib.load(f'/kaggle/working/scalers/{target_value}_scaler.pkl')

    X_test = test_data 
    X_test_scaled = scaler.transform(X_test)

   # Convert to PyTorch tensor
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

    #Initialize model and load the model
    model = Predictor_Model(input_size=len(input_features)).to(device)
    model_path = f"/kaggle/working/saved_models/{target_value}_Model.pth"
    model.load_state_dict(torch.load(model_path))

    model.eval()

    # Predict 
    with torch.no_grad():
        pred = model(X_test_tensor).cpu().numpy().flatten()

    # Save predictions
    all_predictions[target_value] = pred

    print(f"{target_value} prediction complete.")
        


# Read submission sample and check 
submission_sample = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')
# print(f"Sample Submission\n{submission_sample}")

# Convert prediction dictionary into DataFrame
pred_df = pd.DataFrame(all_predictions)

# Merge test input with model output
merged_df= test_data_temp.join(pred_df)

# Drop SMILES columns 
merged_df = merged_df.drop(columns=['SMILES'])

# Print Final output
print(merged_df)



# Save final output 
merged_df.to_csv('submission.csv', index=False)

