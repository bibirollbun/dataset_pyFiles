import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# -----------------------
# Device Configuration
# -----------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=10)
# -----------------------
# Problem Parameters
# -----------------------
eps = 10e-4
b_x = 2.0
b_y = 3.0

# -----------------------
# Define the Heat Source Function f(x, y)
# -----------------------
def f_func(x, y):
    # x and y are expected to be torch tensors of shape (N, 1)
    term1 = 2 * eps * (-x + torch.exp(2 * (x - 1) * eps))
    term2 = x * y**2
    term3 = 6 * x * y
    term4 = - x * torch.exp((3 * (y - 1)) / eps)
    term5 = - y**2 * torch.exp((2 * (x - 1)) / eps)
    term6 = 2 * y**2
    term7 = -6 * y * torch.exp((2 * (x - 1)) / eps)
    term8 = -2 * torch.exp((3 * (y - 1)) / eps)
    term9 = torch.exp((2 * x + 3 * y - 5 )/ eps)
    return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9

# -----------------------
# Define a More Expressive Neural Network Model (PINN)
# -----------------------
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.activation = nn.Tanh()
        layer_list = []
        for i in range(len(layers) - 1):
            linear_layer = nn.Linear(layers[i], layers[i+1])
            # Xavier initialization for better convergence
            nn.init.xavier_normal_(linear_layer.weight)
            nn.init.zeros_(linear_layer.bias)
            layer_list.append(linear_layer)
            layer_list.append(nn.Dropout(0.3))
        self.layers = nn.ModuleList(layer_list)
        
    def forward(self, x):
        # x is of shape (N, 2) containing [x, y] coordinates
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        # Last layer (no activation)
        x = self.layers[-1](x)
        return x

# -----------------------
# Compute the PDE Residual
# -----------------------
def pde_residual(model, x, y):
    # Concatenate x and y to form input coordinates and move to the proper device
    X = torch.cat([x, y], dim=1).to(device)
    X.requires_grad_(True)
    
    # Predicted solution
    u = model(X)
    
    # First derivatives
    grad_u = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u),
                                 retain_graph=True, create_graph=True)[0]
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    
    # Second derivatives
    u_xx = torch.autograd.grad(u_x, X, grad_outputs=torch.ones_like(u_x),
                               retain_graph=True, create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, X, grad_outputs=torch.ones_like(u_y),
                               retain_graph=True, create_graph=True)[0][:, 1:2]
    
    # PDE residual: -eps*(u_xx+u_yy) + b_x*u_x + b_y*u_y - f(x,y)
    f_val = f_func(x.to(device), y.to(device))
    residual = -eps * (u_xx + u_yy) + b_x * u_x + b_y * u_y - f_val
    return residual

# -----------------------
# Sampling Functions for Collocation and Boundary Points
# -----------------------
def sample_points(N_int, N_bnd):
    # Sample interior collocation points uniformly in the domain [0,1]x[0,1]
    x_int = torch.rand(N_int, 1, device=device,dtype=torch.float64)
    y_int = torch.rand(N_int, 1, device=device, dtype=torch.float64)
    
    # Sample boundary points from the four edges of the unit square
    N_side = N_bnd // 4
    # Left edge: x = 0
    x_left = torch.zeros(N_side, 1, device=device)
    y_left = torch.rand(N_side, 1, device=device, dtype=torch.float64)
    # Right edge: x = 1
    x_right = torch.ones(N_side, 1, device=device)
    y_right = torch.rand(N_side, 1, device=device, dtype=torch.float64)
    # Bottom edge: y = 0
    x_bottom = torch.rand(N_side, 1, device=device, dtype=torch.float64)
    y_bottom = torch.zeros(N_side, 1, device=device)
    # Top edge: y = 1
    x_top = torch.rand(N_side, 1, device=device, dtype=torch.float64)
    y_top = torch.ones(N_side, 1, device=device)
    
    x_bnd = torch.cat([x_left, x_right, x_bottom, x_top], dim=0)
    y_bnd = torch.cat([y_left, y_right, y_bottom, y_top], dim=0)
    
    return x_int, y_int, x_bnd, y_bnd

# -----------------------
# Training Loop with Early Stopping using Adam
# -----------------------
def train_adam(model, optimizer, epochs, N_int=200000, N_bnd=1600, lambda_bc=1.0, patience=2000):
    best_loss = float('inf')
    counter = 0
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Sample interior and boundary points
        x_int, y_int, x_bnd, y_bnd = sample_points(N_int, N_bnd)
        
        # PDE residual loss over interior points
        residual = pde_residual(model, x_int, y_int)
        loss_pde = torch.mean(residual**2)
        
        # Boundary condition loss enforcing u(x,y)=0
        X_bnd = torch.cat([x_bnd, y_bnd], dim=1)
        u_bnd = model(X_bnd)
        loss_bc = torch.mean(u_bnd**2)
        
        # Total loss
        loss = loss_pde + lambda_bc * loss_bc
        loss.backward()
        optimizer.step()
        
        # Early Stopping Check
        if loss.item() < best_loss:
            best_loss = loss.item()
            counter = 0  # reset if improvement is seen
        else:
            counter += 1
        
        if epoch % 100 == 0:
            print(f"Adam Epoch {epoch}: Total Loss {loss.item():.6e} | PDE Loss {loss_pde.item():.6e} | BC Loss {loss_bc.item():.6e}")
        
        if counter >= patience:
            print(f"Early stopping triggered at Adam epoch {epoch} with best loss {best_loss:.6e}")
            break

# -----------------------
# Second Stage: Fine-Tuning with LBFGS
# -----------------------
# def train_lbfgs(model, optimizer, epochs, N_int=1000, N_bnd=400, lambda_bc=1.0):
#     # LBFGS requires a closure function that re-computes the loss and gradients.
#     def closure():
#         optimizer.zero_grad()
#         x_int, y_int, x_bnd, y_bnd = sample_points(N_int, N_bnd)
#         residual = pde_residual(model, x_int, y_int)
#         loss_pde = torch.mean(residual**2)
#         X_bnd = torch.cat([x_bnd, y_bnd], dim=1)
#         u_bnd = model(X_bnd)
#         loss_bc = torch.mean(u_bnd**2)
#         loss = loss_pde + lambda_bc * loss_bc
#         loss.backward()
#         return loss

#     for i in range(epochs):
#         loss = optimizer.step(closure)
#         if i % 50 == 0:
#             print(f"LBFGS Iteration {i}: Loss {loss.item():.6e}")

# -----------------------
# Main Execution: Model Setup and Training
# -----------------------
if __name__ == "__main__":
    # Define a more expressive network: Increase depth and width
    layers = [2, 128,128,128,128, 1]  # Input:2, 5 hidden layers with 100 neurons each, Output:1
    model = PINN(layers).to(device)
    
    # Stage 1: Train with Adam + Early Stopping
    adam_optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print("Starting Adam training...")
    train_adam(model, adam_optimizer, epochs=20000, patience=2000)
    
    # Stage 2: Fine-tune with LBFGS
    # lbfgs_optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=500, history_size=50)
    # print("Starting LBFGS fine-tuning...")
    # train_lbfgs(model, lbfgs_optimizer, epochs=500)
    
    # -----------------------
    # Prediction on Test Data and Save Submission
    # -----------------------
    def predict_and_save(model, test_csv='/kaggle/input/ziq-sciml-challenge/test.csv', submission_csv='/kaggle/working/submission.csv'):
        # Load test points from test.csv (expected columns: x, y)
        test_data = pd.read_csv(test_csv, dtype={'x': np.float64, 'y': np.float64})
        x_test = torch.tensor(test_data['x'].values, dtype=torch.float64, device=device).unsqueeze(1)
        y_test = torch.tensor(test_data['y'].values, dtype=torch.float64, device=device).unsqueeze(1)
        X_test = torch.cat([x_test, y_test], dim=1)
        
        # Switch to evaluation mode and disable gradient calculation
        model.eval()
        with torch.no_grad():
            u_pred = model(X_test).cpu().detach().numpy().flatten()  # move predictions back to CPU
        
        # Create submission DataFrame in the required format
        submission = pd.DataFrame({
            'ID': range(1, len(u_pred) + 1),
            'x': test_data['x'],
            'y': test_data['y'],
            'u': u_pred
        })
        submission.to_csv(submission_csv, index=False, float_format='%.17f')
        print(f"Submission saved to {submission_csv}")
    
    # Generate submission after training
    predict_and_save(model)


import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Set default dtype to double precision and choose device
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=17)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -----------------------
# Problem Parameters
# -----------------------
eps = 10e-4
b_x = 2.0
b_y = 3.0

# -----------------------
# Define the Heat Source Function f(x, y)
# -----------------------
def f_func(x, y):
    # x and y are expected to be torch tensors of shape (N, 1)
    term1 = 2 * eps * (-x + torch.exp(2 * (x - 1) * eps))
    term2 = x * y**2
    term3 = 6 * x * y
    term4 = - x * torch.exp((3 * (y - 1)) / eps)
    term5 = - y**2 * torch.exp((2 * (x - 1)) / eps)
    term6 = 2 * y**2
    term7 = -6 * y * torch.exp((2 * (x - 1)) / eps)
    term8 = -2 * torch.exp((3 * (y - 1)) / eps)
    term9 = torch.exp((2 * x + 3 * y - 5 )/ eps)
    return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9

# -----------------------
# Define the PINN Model
# -----------------------
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.activation = nn.Tanh()
        layer_list = []
        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i+1])
            # Xavier initialization for better convergence
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            layer_list.append(layer)
            # layer_list.append(nn.Dropout(0.1))
        self.layers = nn.ModuleList(layer_list)
        
    def forward(self, x):
        # x has shape (N, 2)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

# -----------------------
# Load Training Points from CSV
# -----------------------
def load_training_points(csv_file):
    # Force the x and y columns to be read in double precision
    df = pd.read_csv(csv_file, dtype={'x': np.float64, 'y': np.float64})
    x = torch.tensor(df['x'].values, dtype=torch.float64, device=device).unsqueeze(1)
    y = torch.tensor(df['y'].values, dtype=torch.float64, device=device).unsqueeze(1)
    return x, y

# -----------------------
# Partition Points into Interior and Boundary
# -----------------------
def partition_points(x, y, tol=1e-5):
    # Points are considered on the boundary if x or y is nearly 0 or 1.
    boundary_mask = (torch.abs(x - 0) < tol) | (torch.abs(x - 1) < tol) | \
                    (torch.abs(y - 0) < tol) | (torch.abs(y - 1) < tol)
    interior_mask = ~boundary_mask
    x_int = x[interior_mask.squeeze()]
    y_int = y[interior_mask.squeeze()]
    x_bnd = x[boundary_mask.squeeze()]
    y_bnd = y[boundary_mask.squeeze()]
    return x_int, y_int, x_bnd, y_bnd

# -----------------------
# Compute the PDE Residual
# -----------------------
def pde_residual(model, x, y):
    X = torch.cat([x, y], dim=1)
    X.requires_grad_(True)
    u = model(X)
    # First derivatives
    grad_u = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u),
                                 retain_graph=True, create_graph=True)[0]
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    # Second derivatives
    u_xx = torch.autograd.grad(u_x, X, grad_outputs=torch.ones_like(u_x),
                               retain_graph=True, create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, X, grad_outputs=torch.ones_like(u_y),
                               retain_graph=True, create_graph=True)[0][:, 1:2]
    f_val = f_func(x, y)
    residual = -eps * (u_xx + u_yy) + b_x * u_x + b_y * u_y - f_val
    return residual

# -----------------------
# Training Loop Using CSV Points
# -----------------------
def train_with_csv(model, csv_file, epochs, lambda_bc=1.0):
    # Load the training points from the CSV file
    x_all, y_all = load_training_points(csv_file)
    # Partition the points into interior and boundary sets
    x_int, y_int, x_bnd, y_bnd = partition_points(x_all, y_all)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=100, min_lr=1e-6)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Compute PDE residual loss for interior points
        if x_int.numel() > 0:
            res = pde_residual(model, x_int, y_int)
            loss_pde = torch.mean(res**2)
        else:
            loss_pde = torch.tensor(0.0, dtype=torch.float64, device=device)
        
        # Compute boundary loss enforcing u(x,y) = 0 for boundary points
        if x_bnd.numel() > 0:
            X_bnd = torch.cat([x_bnd, y_bnd], dim=1)
            u_bnd = model(X_bnd)
            loss_bc = torch.mean(u_bnd**2)
        else:
            loss_bc = torch.tensor(0.0, dtype=torch.float64, device=device)
        
        loss = loss_pde + lambda_bc * loss_bc
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # scheduler.step(loss.item())
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Total Loss = {loss.item():.6e}, PDE Loss = {loss_pde.item():.6e}, BC Loss = {loss_bc.item():.6e}")
    print("Training complete.")

# -----------------------
# Prediction and Submission
# -----------------------
def predict_and_save(model, csv_file, submission_csv='submission.csv'):
    # Load test points from CSV
    df = pd.read_csv(csv_file, dtype={'x': np.float64, 'y': np.float64})
    x = torch.tensor(df['x'].values, dtype=torch.float64, device=device).unsqueeze(1)
    y = torch.tensor(df['y'].values, dtype=torch.float64, device=device).unsqueeze(1)
    X = torch.cat([x, y], dim=1)
    
    model.eval()
    with torch.no_grad():
        u_pred = model(X).cpu().numpy().flatten()
    
    # Create submission DataFrame; preserve the "ID" column if it exists
    if 'ID' in df.columns:
        submission = pd.DataFrame({
            'ID': df['ID'],
            'x': df['x'],
            'y': df['y'],
            'u': u_pred
        })
    else:
        submission = pd.DataFrame({
            'ID': range(1, len(u_pred) + 1),
            'x': df['x'],
            'y': df['y'],
            'u': u_pred
        })
    submission.to_csv(submission_csv, index=False, float_format='%.17f')
    print(f"Submission saved to {submission_csv}")

# -----------------------
# Main Execution
# -----------------------
if __name__ == "__main__":
    # Define model architecture (example: 4 hidden layers with 100 neurons each)
    layers = [2, 250,250,250, 1]
    model = PINN(layers).to(device)
    
    # Use the test CSV file as training inputs
    csv_file = '/kaggle/input/ziq-sciml-challenge/test.csv'
    train_with_csv(model, csv_file, epochs=10000, lambda_bc=1.0)
    
    # After training, predict on the same CSV points and save the submission
    predict_and_save(model, csv_file, submission_csv='/kaggle/working/submission.csv')



import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Set default dtype to double precision and choose device
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=17)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -----------------------
# Problem Parameters
# -----------------------
eps = 10e-4
b_x = 2.0
b_y = 3.0

# -----------------------
# Define the Heat Source Function f(x, y)
# -----------------------
def f_func(x, y):
    term1 = 2 * eps * (-x + torch.exp(2 * (x - 1) * eps))
    term2 = x * y**2
    term3 = 6 * x * y
    term4 = - x * torch.exp((3 * (y - 1)) / eps)
    term5 = - y**2 * torch.exp((2 * (x - 1)) / eps)
    term6 = 2 * y**2
    term7 = -6 * y * torch.exp((2 * (x - 1)) / eps)
    term8 = -2 * torch.exp((3 * (y - 1)) / eps)
    term9 = torch.exp((2 * x + 3 * y - 5 )/ eps)
    return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9

# -----------------------
# Define the PINN Model
# -----------------------
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.activation = nn.Tanh()
        layer_list = []
        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i+1])
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            layer_list.append(layer)
        self.layers = nn.ModuleList(layer_list)
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

# -----------------------
# Load Training Points from CSV
# -----------------------
def load_training_points(csv_file):
    df = pd.read_csv(csv_file, dtype={'x': np.float64, 'y': np.float64})
    x = torch.tensor(df['x'].values, dtype=torch.float64, device=device).unsqueeze(1)
    y = torch.tensor(df['y'].values, dtype=torch.float64, device=device).unsqueeze(1)
    return x, y

# -----------------------
# Partition Points into Interior and Boundary
# -----------------------
def partition_points(x, y, tol=1e-5):
    boundary_mask = (torch.abs(x - 0) < tol) | (torch.abs(x - 1) < tol) | \
                    (torch.abs(y - 0) < tol) | (torch.abs(y - 1) < tol)
    interior_mask = ~boundary_mask
    x_int = x[interior_mask.squeeze()]
    y_int = y[interior_mask.squeeze()]
    x_bnd = x[boundary_mask.squeeze()]
    y_bnd = y[boundary_mask.squeeze()]
    return x_int, y_int, x_bnd, y_bnd

# -----------------------
# Compute the PDE Residual
# -----------------------
def pde_residual(model, x, y):
    X = torch.cat([x, y], dim=1)
    X.requires_grad_(True)
    u = model(X)
    grad_u = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), 
                                 retain_graph=True, create_graph=True)[0]
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    u_xx = torch.autograd.grad(u_x, X, grad_outputs=torch.ones_like(u_x), 
                               retain_graph=True, create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, X, grad_outputs=torch.ones_like(u_y), 
                               retain_graph=True, create_graph=True)[0][:, 1:2]
    f_val = f_func(x, y)
    residual = -eps * (u_xx + u_yy) + b_x * u_x + b_y * u_y - f_val
    return residual

# -----------------------
# Training Loop with Batch Processing and LBFGS Fine-Tuning
# -----------------------
def train_with_csv(model, csv_file, epochs, batch_size=5000, lambda_bc=1.0):
    x_all, y_all = load_training_points(csv_file)
    x_int, y_int, x_bnd, y_bnd = partition_points(x_all, y_all)
    
    dataset = TensorDataset(x_int, y_int)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    loss_history = []
    
    for epoch in range(epochs):
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            res = pde_residual(model, x_batch, y_batch)
            loss_pde = torch.mean(res**2)
            X_bnd = torch.cat([x_bnd, y_bnd], dim=1)
            loss_bc = torch.mean(model(X_bnd)**2)
            loss = loss_pde + lambda_bc * loss_bc
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        loss_history.append(loss.item())
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6e}")
    
    # LBFGS Fine-Tuning
    def closure():
        optimizer.zero_grad()
        res = pde_residual(model, x_int, y_int)
        loss_pde = torch.mean(res**2)
        X_bnd = torch.cat([x_bnd, y_bnd], dim=1)
        loss_bc = torch.mean(model(X_bnd)**2)
        loss = loss_pde + lambda_bc * loss_bc
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        loss.backward()
        return loss
    
    optimizer = optim.LBFGS(model.parameters(), lr=1e-6, max_iter=500, history_size=50)
    optimizer.step(closure)
    
    plt.plot(loss_history)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.show()
    print("Training complete.")

# -----------------------
# Prediction and Submission
# -----------------------
def predict_and_save(model, csv_file, submission_csv='submission.csv'):
    df = pd.read_csv(csv_file, dtype={'x': np.float64, 'y': np.float64})
    x = torch.tensor(df['x'].values, dtype=torch.float64, device=device).unsqueeze(1)
    y = torch.tensor(df['y'].values, dtype=torch.float64, device=device).unsqueeze(1)
    X = torch.cat([x, y], dim=1)
    
    model.eval()
    with torch.no_grad():
        u_pred = model(X).cpu().numpy().flatten()
    
    if 'ID' in df.columns:
        submission = pd.DataFrame({
            'ID': df['ID'],
            'x': df['x'],
            'y': df['y'],
            'u': u_pred
        })
    else:
        submission = pd.DataFrame({
            'ID': range(1, len(u_pred) + 1),
            'x': df['x'],
            'y': df['y'],
            'u': u_pred
        })
    submission.to_csv(submission_csv, index=False, float_format='%.17f')
    print(f"Submission saved to {submission_csv}")

# -----------------------
# Main Execution
# -----------------------
if __name__ == "__main__":
    layers = [2, 250, 250, 1]
    model = PINN(layers).to(device)
    csv_file = '/kaggle/input/ziq-sciml-challenge/test.csv'
    train_with_csv(model, csv_file, epochs=1000, lambda_bc=1.0)
    predict_and_save(model, csv_file, submission_csv='/kaggle/working/submission.csv')



#LBFGS
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# -----------------------
# Device and Precision Setup
# -----------------------
torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -----------------------
# Problem Parameters
# -----------------------
eps = 10e-4
b_x = 2.0
b_y = 3.0

# -----------------------
# Define the Heat Source Function f(x, y)
# -----------------------
def f_func(x, y):
    # x and y are expected to be torch tensors of shape (N, 1)
    term1 = 2 * eps * (-x + torch.exp(2 * (x - 1) * eps))
    term2 = x * y**2
    term3 = 6 * x * y
    term4 = - x * torch.exp((3 * (y - 1)) / eps)
    term5 = - y**2 * torch.exp((2 * (x - 1)) / eps)
    term6 = 2 * y**2
    term7 = -6 * y * torch.exp((2 * (x - 1)) / eps)
    term8 = -2 * torch.exp((3 * (y - 1)) / eps)
    term9 = torch.exp((2 * x + 3 * y - 5 )/ eps)
    return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9

# -----------------------
# Define the PINN Model
# -----------------------
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.activation = nn.Tanh()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            nn.init.xavier_normal_(self.layers[-1].weight)
            nn.init.zeros_(self.layers[-1].bias)
            if i < len(layers) - 2:
                self.norms.append(nn.BatchNorm1d(layers[i+1]))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
            if i < len(self.norms):
                x = self.norms[i](x)
        return self.layers[-1](x)

# -----------------------
# Loading and Partitioning CSV Data
# -----------------------
def load_training_points(csv_file):
    # Force x and y columns to be float64 (17-digit precision)
    df = pd.read_csv(csv_file, dtype={'x': np.float64, 'y': np.float64})
    x = torch.tensor(df['x'].values, dtype=torch.float64, device=device).unsqueeze(1)
    y = torch.tensor(df['y'].values, dtype=torch.float64, device=device).unsqueeze(1)
    return x, y

def partition_points(x, y, tol=1e-6):
    # Points are on the boundary if x or y is within tol of 0 or 1.
    boundary_mask = (torch.abs(x - 0) < tol) | (torch.abs(x - 1) < tol) | \
                    (torch.abs(y - 0) < tol) | (torch.abs(y - 1) < tol)
    interior_mask = ~boundary_mask
    x_int = x[interior_mask.squeeze()]
    y_int = y[interior_mask.squeeze()]
    x_bnd = x[boundary_mask.squeeze()]
    y_bnd = y[boundary_mask.squeeze()]
    return x_int, y_int, x_bnd, y_bnd

# -----------------------
# PDE Residual Computation
# -----------------------
def pde_residual(model, x, y):
    X = torch.cat([x, y], dim=1)
    X.requires_grad_(True)
    u = model(X)
    grad_u = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u),
                                 retain_graph=True, create_graph=True)[0]
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    u_xx = torch.autograd.grad(u_x, X, grad_outputs=torch.ones_like(u_x),
                               retain_graph=True, create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, X, grad_outputs=torch.ones_like(u_y),
                               retain_graph=True, create_graph=True)[0][:, 1:2]
    f_val = f_func(x, y)
    residual = -eps * (u_xx + u_yy) + b_x * u_x + b_y * u_y - f_val
    return residual

# -----------------------
# Training with Adam
# -----------------------
def train_with_csv_adam(model, csv_file, epochs, lambda_bc=1.0, patience=2000):
    x_all, y_all = load_training_points(csv_file)
    x_int, y_int, x_bnd, y_bnd = partition_points(x_all, y_all)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    best_loss = float('inf')
    counter = 0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # PDE loss on interior points
        if x_int.numel() > 0:
            res = pde_residual(model, x_int, y_int)
            loss_pde = torch.mean(res**2)
        else:
            loss_pde = torch.tensor(0.0, dtype=torch.float64, device=device)
        
        # Boundary loss: enforce u(x,y)=0 on boundary points
        if x_bnd.numel() > 0:
            X_bnd = torch.cat([x_bnd, y_bnd], dim=1)
            u_bnd = model(X_bnd)
            loss_bc = torch.mean(u_bnd**2)
        else:
            loss_bc = torch.tensor(0.0, dtype=torch.float64, device=device)
        
        loss = loss_pde + lambda_bc * loss_bc
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            counter = 0
        else:
            counter += 1
        
        if epoch % 1000 == 0:
            print(f"[Adam] Epoch {epoch}: Total Loss = {loss.item():.6e}, PDE Loss = {loss_pde.item():.6e}, BC Loss = {loss_bc.item():.6e}")
        if counter >= patience:
            print(f"[Adam] Early stopping at epoch {epoch} with best loss {best_loss:.6e}")
            break

# -----------------------
# Fine-Tuning with LBFGS
# -----------------------
def train_with_csv_lbfgs(model, csv_file, lbfgs_epochs, lambda_bc=1.0):
    # Load and partition training data once (we assume it's constant)
    x_all, y_all = load_training_points(csv_file)
    x_int, y_int, x_bnd, y_bnd = partition_points(x_all, y_all)
    
    # LBFGS optimizer requires a closure that re-computes the loss
    optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=0.1, max_iter=500, history_size=50)
    
    def closure():
        optimizer_lbfgs.zero_grad()
        # PDE loss for interior
        if x_int.numel() > 0:
            res = pde_residual(model, x_int, y_int)
            loss_pde = torch.mean(res**2)
        else:
            loss_pde = torch.tensor(0.0, dtype=torch.float64, device=device)
        
        # Boundary loss for boundary points
        if x_bnd.numel() > 0:
            X_bnd = torch.cat([x_bnd, y_bnd], dim=1)
            u_bnd = model(X_bnd)
            loss_bc = torch.mean(u_bnd**2)
        else:
            loss_bc = torch.tensor(0.0, dtype=torch.float64, device=device)
        
        loss = loss_pde + lambda_bc * loss_bc
        loss.backward()
        return loss
    
    for i in range(lbfgs_epochs):
        loss = optimizer_lbfgs.step(closure)
        if i % 50 == 0:
            print(f"[LBFGS] Iteration {i}: Loss = {loss.item():.6e}")

# -----------------------
# Prediction and Saving Submission
# -----------------------
def predict_and_save(model, csv_file, submission_csv='/kaggle/working/submission.csv'):
    # Read test points from CSV file with full precision
    df = pd.read_csv(csv_file, dtype={'x': np.float64, 'y': np.float64})
    x = torch.tensor(df['x'].values, dtype=torch.float64, device=device).unsqueeze(1)
    y = torch.tensor(df['y'].values, dtype=torch.float64, device=device).unsqueeze(1)
    X = torch.cat([x, y], dim=1)
    
    model.eval()
    with torch.no_grad():
        u_pred = model(X).cpu().numpy().flatten()
    
    # Create submission DataFrame; preserve "ID" column if it exists.
    if 'ID' in df.columns:
        submission = pd.DataFrame({
            'ID': df['ID'],
            'x': df['x'],
            'y': df['y'],
            'u': u_pred
        })
    else:
        submission = pd.DataFrame({
            'ID': range(1, len(u_pred) + 1),
            'x': df['x'],
            'y': df['y'],
            'u': u_pred
        })
    submission.to_csv(submission_csv, index=False, float_format='%.17f')
    print(f"Submission saved to {submission_csv}")

# -----------------------
# Main Execution
# -----------------------
if __name__ == "__main__":
    # Define the model architecture (example: 4 hidden layers with 100 neurons each)
    layers = [2, 250,250, 1]
    model = PINN(layers).to(device)
    
    csv_file = '/kaggle/input/ziq-sciml-challenge/test.csv'  # Use your test CSV file as training data
    
    print("Starting Adam training...")
    train_with_csv_adam(model, csv_file, epochs=5000, lambda_bc=1.0)
    
    print("Starting LBFGS fine-tuning...")
    train_with_csv_lbfgs(model, csv_file, lbfgs_epochs=1000, lambda_bc=1.0)
    
    print("Generating predictions and saving submission...")
    predict_and_save(model, csv_file, submission_csv='/kaggle/working/submission.csv')



#Heatmap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_heatmap_from_csv(csv_file):
    # Load the CSV file (assumes it has columns: 'x', 'y', 'u')
    df = pd.read_csv(csv_file, dtype={'x': np.float64, 'y': np.float64, 'u': np.float64})
    
    # Pivot the DataFrame into a 2D grid. 
    # This works well if your points lie on a regular grid.
    grid = df.pivot(index='y', columns='x', values='u')
    
    # For visualization, sort the y-index so that the lower values are at the bottom
    grid = grid.sort_index(ascending=True)
    
    # Get the x and y extents from the grid
    x_min, x_max = grid.columns.min(), grid.columns.max()
    y_min, y_max = grid.index.min(), grid.index.max()
    
    plt.figure(figsize=(12, 10))
    # Use imshow to display the heatmap. 'origin=lower' puts the lowest y-value at the bottom.
    plt.imshow(grid.values, extent=[x_min, x_max, y_min, y_max], origin='lower', 
               aspect='auto', cmap='jet')
    plt.colorbar(label='u')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Heatmap of u values in the 1×1 Unit Domain')
    plt.show()

# Example usage:
plot_heatmap_from_csv('/kaggle/working/submission.csv')



#Laplace Transform of Inputs
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import torch.fft  # For computing inverse Laplace Transform via Bromwich Integral

# -----------------------
# Device and Precision Setup
# -----------------------
torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Problem Parameters
# -----------------------
eps = 10e-4
b_x = 2.0
b_y = 3.0

# -----------------------
# Define the Heat Source Function f(x, y)
# -----------------------
def f_func(x, y):
    # x and y are expected to be torch tensors of shape (N, 1)
    term1 = 2 * eps * (-x + torch.exp(2 * (x - 1) * eps))
    term2 = x * y**2
    term3 = 6 * x * y
    term4 = - x * torch.exp((3 * (y - 1)) / eps)
    term5 = - y**2 * torch.exp((2 * (x - 1)) / eps)
    term6 = 2 * y**2
    term7 = -6 * y * torch.exp((2 * (x - 1)) / eps)
    term8 = -2 * torch.exp((3 * (y - 1)) / eps)
    term9 = torch.exp((2 * x + 3 * y - 5 )/ eps)
    return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9
    
# -----------------------
# Laplace Transform (Forward & Inverse)
# -----------------------
def laplace_transform(model, x, y, s_vals):
    """Compute Laplace-transformed solution U(s)."""
    N = x.shape[0]  # Number of training points
    M = s_vals.shape[0]  # Number of Laplace domain points
    
    # Expand x and y to match s_vals
    x_exp = x.repeat_interleave(M, dim=0)  # [N*M, 1]
    y_exp = y.repeat_interleave(M, dim=0)  # [N*M, 1]
    
    # Expand s_vals to match x and y
    s_exp = s_vals.repeat(N, 1)  # [N*M, 1]

    X = torch.cat([x_exp, y_exp, s_exp], dim=1)  # Shape [N*M, 3]
    return model(X).reshape(N, M)  # Return as [N, M] to match expected output


def inverse_laplace_transform(U_s, s_vals):
    """Compute inverse Laplace Transform using numerical Bromwich integral approximation."""
    exp_term = torch.exp(s_vals * U_s)
    return torch.trapz(exp_term, s_vals) / (2 * torch.pi)

def load_training_points(csv_file):
    # Force x and y columns to be float64 (17-digit precision)
    df = pd.read_csv(csv_file, dtype={'x': np.float64, 'y': np.float64})
    x = torch.tensor(df['x'].values, dtype=torch.float64, device=device).unsqueeze(1)
    y = torch.tensor(df['y'].values, dtype=torch.float64, device=device).unsqueeze(1)
    return x, y

def partition_points(x, y, tol=1e-6):
    # Points are on the boundary if x or y is within tol of 0 or 1.
    boundary_mask = (torch.abs(x - 0) < tol) | (torch.abs(x - 1) < tol) | \
                    (torch.abs(y - 0) < tol) | (torch.abs(y - 1) < tol)
    interior_mask = ~boundary_mask
    x_int = x[interior_mask.squeeze()]
    y_int = y[interior_mask.squeeze()]
    x_bnd = x[boundary_mask.squeeze()]
    y_bnd = y[boundary_mask.squeeze()]
    return x_int, y_int, x_bnd, y_bnd
    
# -----------------------
# Define the PINN Model (Laplace Domain)
# -----------------------
class LaplacePINN(nn.Module):
    def __init__(self, layers):
        super(LaplacePINN, self).__init__()
        self.activation = nn.Tanh()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            nn.init.xavier_normal_(self.layers[-1].weight)
            nn.init.zeros_(self.layers[-1].bias)
            if i < len(layers) - 2:
                self.norms.append(nn.BatchNorm1d(layers[i+1]))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
            if i < len(self.norms):
                x = self.norms[i](x)
        return self.layers[-1](x)

# -----------------------
# Laplace Transformed PDE Residual Computation
# -----------------------
def pde_residual_laplace(model, x, y, s):
    """Computes the residual in Laplace space."""
    U_s = laplace_transform(model, x, y, s)
    
    grad_U = torch.autograd.grad(U_s, x, grad_outputs=torch.ones_like(U_s), 
                                 retain_graph=True, create_graph=True)[0]
    U_x, U_y = grad_U[:, 0:1], grad_U[:, 1:2]

    U_xx = torch.autograd.grad(U_x, x, grad_outputs=torch.ones_like(U_x), 
                               retain_graph=True, create_graph=True)[0][:, 0:1]
    U_yy = torch.autograd.grad(U_y, x, grad_outputs=torch.ones_like(U_y), 
                               retain_graph=True, create_graph=True)[0][:, 1:2]
    
    f_s = torch.fft.fft(f_func(x, y))  # Laplace transform of source function

    residual = -eps * (U_xx + U_yy) + b_x * U_x + b_y * U_y - f_s
    return residual

# -----------------------
# Training Function (Laplace PINN)
# -----------------------
def train_laplace_pinn(model, csv_file, epochs, lambda_bc=1.0):
    x_all, y_all = load_training_points(csv_file)
    x_int, y_int, x_bnd, y_bnd = partition_points(x_all, y_all)

    s_vals = torch.linspace(1, 10, steps=50, dtype=torch.float64, device=device).unsqueeze(1)  # Laplace variable range
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5000, factor=0.5, verbose=True)

    best_loss = float('inf')
    patience = 2000
    counter = 0

    for epoch in range(epochs):
        optimizer.zero_grad()

        if x_int.numel() > 0:
            res = pde_residual_laplace(model, x_int, y_int, s_vals)
            loss_pde = torch.mean(res**2)
        else:
            loss_pde = torch.tensor(0.0, dtype=torch.float64, device=device)

        if x_bnd.numel() > 0:
            X_bnd = torch.cat([x_bnd, y_bnd], dim=1)
            u_bnd = model(X_bnd)
            indicator = adaptive_indicator(x_bnd, y_bnd, alpha=30, beta=1e9)
            loss_bc = torch.mean((indicator * u_bnd)**2)
        else:
            loss_bc = torch.tensor(0.0, dtype=torch.float64, device=device)

        loss = loss_pde + lambda_bc * loss_bc
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if loss.item() < best_loss:
            best_loss = loss.item()
            counter = 0
        else:
            counter += 1

        if epoch % 1000 == 0:
            print(f"[LaplacePINN] Epoch {epoch}: Loss = {loss.item():.6e}, PDE = {loss_pde.item():.6e}, BC = {loss_bc.item():.6e}")

        if counter >= patience:
            print(f"[LaplacePINN] Early stopping at epoch {epoch} with best loss {best_loss:.6e}")
            break

# -----------------------
# Execution
# -----------------------
if __name__ == "__main__":
    model = LaplacePINN([3, 64, 64, 1]).to(device)  # Added 's' as input
    csv_file = '/kaggle/input/ziq-sciml-challenge/test.csv'

    print("Starting Laplace-PINN training...")
    train_laplace_pinn(model, csv_file, epochs=5000, lambda_bc=3.0)



#Analytical Solution RMSE
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# -----------------------
# Device and Precision Setup
# -----------------------
torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -----------------------
# Problem Parameters
# -----------------------
eps = 10e-4
b_x = 2.0
b_y = 3.0

# -----------------------
# Define the Heat Source Function f(x, y)
# -----------------------
def f_func(x, y):
    # x and y are expected to be torch tensors of shape (N, 1)
    term1 = 2 * eps * (-x + torch.exp(2 * (x - 1) * eps))
    term2 = x * y**2
    term3 = 6 * x * y
    term4 = - x * torch.exp((3 * (y - 1)) / eps)
    term5 = - y**2 * torch.exp((2 * (x - 1)) / eps)
    term6 = 2 * y**2
    term7 = -6 * y * torch.exp((2 * (x - 1)) / eps)
    term8 = -2 * torch.exp((3 * (y - 1)) / eps)
    term9 = torch.exp((2 * x + 3 * y - 5) / eps)
    return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9

# -----------------------
# Define the Analytical (Manufactured) Solution
# -----------------------
def analytical_solution(x, y):
    # Manufactured solution: u(x,y) = sin(pi*x) * sin(pi*y)
    return torch.sin(np.pi * x) * torch.sin(np.pi * y)

# -----------------------
# Compute RMSE between model predictions and analytical solution
# -----------------------
def compute_rmse(model, x, y):
    X = torch.cat([x, y], dim=1)
    with torch.no_grad():
        u_pred = model(X)
    u_true = analytical_solution(x, y)
    rmse = torch.sqrt(torch.mean((u_pred - u_true)**2))
    return rmse

# -----------------------
# Define the PINN Model
# -----------------------
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.activation = nn.Tanh()
        layer_list = []
        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i+1])
            # Xavier initialization for better convergence
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            layer_list.append(layer)
            # layer_list.append(nn.Dropout(0.1))
        self.layers = nn.ModuleList(layer_list)
        
    def forward(self, x):
        # x has shape (N, 2)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)
# -----------------------
# Loading and Partitioning CSV Data
# -----------------------
def load_training_points(csv_file):
    # Force x and y columns to be float64 (17-digit precision)
    df = pd.read_csv(csv_file, dtype={'x': np.float64, 'y': np.float64})
    x = torch.tensor(df['x'].values, dtype=torch.float64, device=device).unsqueeze(1)
    y = torch.tensor(df['y'].values, dtype=torch.float64, device=device).unsqueeze(1)
    return x, y

def partition_points(x, y, tol=1e-6):
    # Points are on the boundary if x or y is within tol of 0 or 1.
    boundary_mask = (torch.abs(x - 0) < tol) | (torch.abs(x - 1) < tol) | \
                    (torch.abs(y - 0) < tol) | (torch.abs(y - 1) < tol)
    interior_mask = ~boundary_mask
    x_int = x[interior_mask.squeeze()]
    y_int = y[interior_mask.squeeze()]
    x_bnd = x[boundary_mask.squeeze()]
    y_bnd = y[boundary_mask.squeeze()]
    return x_int, y_int, x_bnd, y_bnd

# -----------------------
# PDE Residual Computation
# -----------------------
def pde_residual(model, x, y):
    X = torch.cat([x, y], dim=1)
    X.requires_grad_(True)
    u = model(X)
    grad_u = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u),
                                 retain_graph=True, create_graph=True)[0]
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    u_xx = torch.autograd.grad(u_x, X, grad_outputs=torch.ones_like(u_x),
                               retain_graph=True, create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, X, grad_outputs=torch.ones_like(u_y),
                               retain_graph=True, create_graph=True)[0][:, 1:2]
    f_val = f_func(x, y)
    residual = -eps * (u_xx + u_yy) + b_x * u_x + b_y * u_y - f_val
    return residual

# -----------------------
# Training with Adam including RMSE minimization
# -----------------------
def train_with_csv_adam_rmse(model, csv_file, epochs, lambda_bc=1.0, lambda_rmse=0.1, patience=2000):
    x_all, y_all = load_training_points(csv_file)
    x_int, y_int, x_bnd, y_bnd = partition_points(x_all, y_all)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_loss = float('inf')
    counter = 0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # PDE loss on interior points
        if x_int.numel() > 0:
            res = pde_residual(model, x_int, y_int)
            loss_pde = torch.mean(res**2)
        else:
            loss_pde = torch.tensor(0.0, dtype=torch.float64, device=device)
        
        # Boundary loss: enforce u(x,y)=0 on boundary points
        if x_bnd.numel() > 0:
            X_bnd = torch.cat([x_bnd, y_bnd], dim=1)
            u_bnd = model(X_bnd)
            loss_bc = torch.mean(u_bnd**2)
        else:
            loss_bc = torch.tensor(0.0, dtype=torch.float64, device=device)
        
        # RMSE loss: difference between predicted and analytical solution over all points
        rmse_loss = compute_rmse(model, x_all, y_all)
        
        # Total loss: combine PDE loss, boundary condition loss, and RMSE loss
        loss = loss_pde + lambda_bc * loss_bc + lambda_rmse * rmse_loss
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            counter = 0
        else:
            counter += 1
        
        if epoch % 1000 == 0:
            print(f"[Adam] Epoch {epoch}: Total Loss = {loss.item():.6e}, PDE Loss = {loss_pde.item():.6e}, "
                  f"BC Loss = {loss_bc.item():.6e}, RMSE = {rmse_loss.item():.6e}")
        if counter >= patience:
            print(f"[Adam] Early stopping at epoch {epoch} with best loss {best_loss:.6e}")
            break

# -----------------------
# Fine-Tuning with LBFGS
# -----------------------
def train_with_csv_lbfgs(model, csv_file, lbfgs_epochs, lambda_bc=1.0):
    x_all, y_all = load_training_points(csv_file)
    x_int, y_int, x_bnd, y_bnd = partition_points(x_all, y_all)
    
    optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=0.1, max_iter=500, history_size=50)
    
    def closure():
        optimizer_lbfgs.zero_grad()
        if x_int.numel() > 0:
            res = pde_residual(model, x_int, y_int)
            loss_pde = torch.mean(res**2)
        else:
            loss_pde = torch.tensor(0.0, dtype=torch.float64, device=device)
        
        if x_bnd.numel() > 0:
            X_bnd = torch.cat([x_bnd, y_bnd], dim=1)
            u_bnd = model(X_bnd)
            loss_bc = torch.mean(u_bnd**2)
        else:
            loss_bc = torch.tensor(0.0, dtype=torch.float64, device=device)
        
        loss = loss_pde + lambda_bc * loss_bc
        loss.backward()
        return loss
    
    for i in range(lbfgs_epochs):
        loss = optimizer_lbfgs.step(closure)
        if i % 50 == 0:
            print(f"[LBFGS] Iteration {i}: Loss = {loss.item():.6e}")

# -----------------------
# Prediction and Saving Submission
# -----------------------
def predict_and_save(model, csv_file, submission_csv='/kaggle/working/submission.csv'):
    # Read test points from CSV file with full precision
    df = pd.read_csv(csv_file, dtype={'x': np.float64, 'y': np.float64})
    x = torch.tensor(df['x'].values, dtype=torch.float64, device=device).unsqueeze(1)
    y = torch.tensor(df['y'].values, dtype=torch.float64, device=device).unsqueeze(1)
    X = torch.cat([x, y], dim=1)
    
    model.eval()
    with torch.no_grad():
        u_pred = model(X).cpu().numpy().flatten()
    
    # Create submission DataFrame; preserve "ID" column if it exists.
    if 'ID' in df.columns:
        submission = pd.DataFrame({
            'ID': df['ID'],
            'x': df['x'],
            'y': df['y'],
            'u': u_pred
        })
    else:
        submission = pd.DataFrame({
            'ID': range(1, len(u_pred) + 1),
            'x': df['x'],
            'y': df['y'],
            'u': u_pred
        })
    submission.to_csv(submission_csv, index=False, float_format='%.17f')
    print(f"Submission saved to {submission_csv}")

# -----------------------
# Main Execution
# -----------------------
if __name__ == "__main__":
    # Define the model architecture (example: 4 hidden layers with 50, 100, 50 neurons)
    layers = [2, 50, 50, 1]
    model = PINN(layers).to(device)
    
    csv_file = '/kaggle/input/ziq-sciml-challenge/test.csv'  # Use your test CSV file as training data
    
    print("Starting Adam training with RMSE minimization...")
    train_with_csv_adam_rmse(model, csv_file, epochs=20000, lambda_bc=3.0, lambda_rmse=1)
    
    print("Starting LBFGS fine-tuning...")
    # Uncomment the following line to run LBFGS fine-tuning:
    # train_with_csv_lbfgs(model, csv_file, lbfgs_epochs=500, lambda_bc=1.0)
    
    print("Generating predictions and saving submission...")
    predict_and_save(model, csv_file, submission_csv='/kaggle/working/submission.csv')



#Model of Experts
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# -----------------------
# Device and Precision Setup
# -----------------------
torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -----------------------
# Problem Parameters
# -----------------------
eps = 10e-4
b_x = 2.0
b_y = 3.0

# -----------------------
# Define the Heat Source Function f(x, y)
# -----------------------
def f_func(x, y):
    term1 = 2 * eps * (-x + torch.exp(2 * (x - 1) * eps))
    term2 = x * y**2
    term3 = 6 * x * y
    term4 = - x * torch.exp((3 * (y - 1)) / eps)
    term5 = - y**2 * torch.exp((2 * (x - 1)) / eps)
    term6 = 2 * y**2
    term7 = -6 * y * torch.exp((2 * (x - 1)) / eps)
    term8 = -2 * torch.exp((3 * (y - 1)) / eps)
    term9 = torch.exp((2 * x + 3 * y - 5) / eps)
    return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9

# -----------------------
# Define the Analytical (Manufactured) Solution
# -----------------------
def analytical_solution(x, y):
    # Manufactured solution: u(x,y) = sin(pi*x) * sin(pi*y)
    return torch.sin(np.pi * x) * torch.sin(np.pi * y)

# -----------------------
# Compute RMSE between model predictions and analytical solution
# -----------------------
def compute_rmse(model, x, y):
    X = torch.cat([x, y], dim=1)
    with torch.no_grad():
        u_pred = model(X)
    u_true = analytical_solution(x, y)
    rmse = torch.sqrt(torch.mean((u_pred - u_true)**2))
    return rmse

# -----------------------
# Define the PINN Expert Model (without BatchNorm, configurable activation)
# -----------------------
class PINN(nn.Module):
    def __init__(self, layers, activation):
        super(PINN, self).__init__()
        self.activation = activation
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i+1])
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)
                
    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
        return self.layers[-1](x)

# -----------------------
# Define the Gating Network for MoE
# -----------------------
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_dims=[20, 20]):
        super(GatingNetwork, self).__init__()
        dims = [input_dim] + hidden_dims
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            layer = nn.Linear(dims[i], dims[i+1])
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)
        # Final layer to produce logits for each expert
        self.output_layer = nn.Linear(dims[-1], num_experts)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        
    def forward(self, x):
        for layer in self.layers:
            x = torch.tanh(layer(x))  # using tanh in gating network
        logits = self.output_layer(x)
        weights = torch.softmax(logits, dim=1)  # normalized weights for experts
        return weights

# -----------------------
# Define the Mixture-of-Experts (MoE) PINN Model
# -----------------------
class MoEPINN(nn.Module):
    def __init__(self, experts, gating_net):
        super(MoEPINN, self).__init__()
        # experts: list of PINN models
        self.experts = nn.ModuleList(experts)
        self.gating_net = gating_net
        
    def forward(self, x):
        # Compute weights from the gating network
        gate_weights = self.gating_net(x)  # shape: (N, num_experts)
        # Compute each expert's output; each output shape: (N, 1)
        expert_outputs = [expert(x) for expert in self.experts]
        # Stack expert outputs: shape (num_experts, N, 1)
        outputs = torch.stack(expert_outputs, dim=0)
        # Permute to shape (N, num_experts, 1)
        outputs = outputs.permute(1, 0, 2)
        # Multiply each expert's output by its corresponding weight and sum along experts
        gate_weights = gate_weights.unsqueeze(2)  # shape: (N, num_experts, 1)
        output = torch.sum(gate_weights * outputs, dim=1)  # shape: (N, 1)
        return output

# -----------------------
# Loading and Partitioning CSV Data
# -----------------------
def load_training_points(csv_file):
    df = pd.read_csv(csv_file, dtype={'x': np.float64, 'y': np.float64})
    x = torch.tensor(df['x'].values, dtype=torch.float64, device=device).unsqueeze(1)
    y = torch.tensor(df['y'].values, dtype=torch.float64, device=device).unsqueeze(1)
    return x, y

def partition_points(x, y, tol=1e-6):
    boundary_mask = (torch.abs(x - 0) < tol) | (torch.abs(x - 1) < tol) | \
                    (torch.abs(y - 0) < tol) | (torch.abs(y - 1) < tol)
    interior_mask = ~boundary_mask
    x_int = x[interior_mask.squeeze()]
    y_int = y[interior_mask.squeeze()]
    x_bnd = x[boundary_mask.squeeze()]
    y_bnd = y[boundary_mask.squeeze()]
    return x_int, y_int, x_bnd, y_bnd

# -----------------------
# PDE Residual Computation
# -----------------------
def pde_residual(model, x, y):
    X = torch.cat([x, y], dim=1)
    X.requires_grad_(True)
    u = model(X)
    grad_u = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u),
                                 retain_graph=True, create_graph=True)[0]
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    u_xx = torch.autograd.grad(u_x, X, grad_outputs=torch.ones_like(u_x),
                               retain_graph=True, create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, X, grad_outputs=torch.ones_like(u_y),
                               retain_graph=True, create_graph=True)[0][:, 1:2]
    f_val = f_func(x, y)
    residual = -eps * (u_xx + u_yy) + b_x * u_x + b_y * u_y - f_val
    return residual

# -----------------------
# Training with Adam including RMSE minimization and ReLoBRaLo adaptive weighting (for MoE)
# -----------------------
def train_with_csv_adam_rmse_relobralo_moe(model, csv_file, epochs, lambda_bc=1.0, lambda_rmse=0.1, patience=9000):
    x_all, y_all = load_training_points(csv_file)
    x_int, y_int, x_bnd, y_bnd = partition_points(x_all, y_all)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_loss = float('inf')
    counter = 0
    
    # Initialize EMAs for PDE and BC losses
    ema_pde = 1.0
    ema_bc = 1.0
    alpha = 0.01
    eps_stability = 1e-8
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # PDE loss on interior points
        if x_int.numel() > 0:
            res = pde_residual(model, x_int, y_int)
            loss_pde = torch.mean(res**2)
        else:
            loss_pde = torch.tensor(0.0, dtype=torch.float64, device=device)
        
        # Boundary loss on boundary points
        if x_bnd.numel() > 0:
            X_bnd = torch.cat([x_bnd, y_bnd], dim=1)
            u_bnd = model(X_bnd)
            loss_bc = torch.mean(u_bnd**2)
        else:
            loss_bc = torch.tensor(0.0, dtype=torch.float64, device=device)
        
        # Update EMAs
        ema_pde = (1 - alpha) * ema_pde + alpha * loss_pde.item()
        ema_bc = (1 - alpha) * ema_bc + alpha * loss_bc.item()
        
        # Compute adaptive weights
        w_pde = ema_bc / (ema_pde + ema_bc + eps_stability)
        w_bc = ema_pde / (ema_pde + ema_bc + eps_stability)
        
        # RMSE loss over all points
        rmse_loss = compute_rmse(model, x_all, y_all)
        
        # Total loss
        loss = w_pde * loss_pde + lambda_bc * w_bc * loss_bc + lambda_rmse * rmse_loss
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            counter = 0
        else:
            counter += 1
        
        if epoch % 1000 == 0:
            print(f"[Adam MoE] Epoch {epoch}: Total Loss = {loss.item():.6e}, PDE Loss = {loss_pde.item():.6e}, "
                  f"BC Loss = {loss_bc.item():.6e}, RMSE = {rmse_loss.item():.6e}, "
                  f"w_pde = {w_pde:.4f}, w_bc = {w_bc:.4f}")
        if counter >= patience:
            print(f"[Adam MoE] Early stopping at epoch {epoch} with best loss {best_loss:.6e}")
            break

# -----------------------
# Fine-Tuning with LBFGS (Optional)
# -----------------------
def train_with_csv_lbfgs(model, csv_file, lbfgs_epochs, lambda_bc=1.0):
    x_all, y_all = load_training_points(csv_file)
    x_int, y_int, x_bnd, y_bnd = partition_points(x_all, y_all)
    
    optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=0.1, max_iter=500, history_size=50)
    
    def closure():
        optimizer_lbfgs.zero_grad()
        if x_int.numel() > 0:
            res = pde_residual(model, x_int, y_int)
            loss_pde = torch.mean(res**2)
        else:
            loss_pde = torch.tensor(0.0, dtype=torch.float64, device=device)
        if x_bnd.numel() > 0:
            X_bnd = torch.cat([x_bnd, y_bnd], dim=1)
            u_bnd = model(X_bnd)
            loss_bc = torch.mean(u_bnd**2)
        else:
            loss_bc = torch.tensor(0.0, dtype=torch.float64, device=device)
        loss = loss_pde + lambda_bc * loss_bc
        loss.backward()
        return loss
    
    for i in range(lbfgs_epochs):
        loss = optimizer_lbfgs.step(closure)
        if i % 50 == 0:
            print(f"[LBFGS] Iteration {i}: Loss = {loss.item():.6e}")

# -----------------------
# Prediction and Saving Submission
# -----------------------
def predict_and_save(model, csv_file, submission_csv='/kaggle/working/submission.csv'):
    df = pd.read_csv(csv_file, dtype={'x': np.float64, 'y': np.float64})
    x = torch.tensor(df['x'].values, dtype=torch.float64, device=device).unsqueeze(1)
    y = torch.tensor(df['y'].values, dtype=torch.float64, device=device).unsqueeze(1)
    X = torch.cat([x, y], dim=1)
    model.eval()
    with torch.no_grad():
        u_pred = model(X).cpu().numpy().flatten()
    if 'ID' in df.columns:
        submission = pd.DataFrame({'ID': df['ID'], 'x': df['x'], 'y': df['y'], 'u': u_pred})
    else:
        submission = pd.DataFrame({'ID': range(1, len(u_pred)+1), 'x': df['x'], 'y': df['y'], 'u': u_pred})
    submission.to_csv(submission_csv, index=False, float_format='%.17f')
    print(f"Submission saved to {submission_csv}")

# -----------------------
# Main Execution
# -----------------------
if __name__ == "__main__":
    # Define network architecture
    layers = [2, 50,100,50, 1]
    
    # Create three PINN experts (for example, all with tanh activation, but they can be configured differently)
    expert1 = PINN(layers, activation=torch.tanh).to(device)
    expert2 = PINN(layers, activation=torch.sigmoid).to(device)
    expert3 = PINN(layers, activation=torch.sin).to(device)
    experts = [expert1, expert2, expert3]
    
    # Create the gating network; input_dim=2 (x,y), output dimension equals number of experts (3)
    gating_net = GatingNetwork(input_dim=2, num_experts=3).to(device)
    
    # Build the MoE PINN model
    moe_model = MoEPINN(experts, gating_net)
    
    csv_file = '/kaggle/input/ziq-sciml-challenge/test.csv'  # Use your test CSV file as training data
    
    print("Starting Adam training with RMSE minimization, ReLoBRaLo loss, and MoE PINNs...")
    train_with_csv_adam_rmse_relobralo_moe(moe_model, csv_file, epochs=20000, lambda_bc=1.0, lambda_rmse=3)
    
    print("Starting LBFGS fine-tuning...")
    # Uncomment the following line to run LBFGS fine-tuning:
    # train_with_csv_lbfgs(moe_model, csv_file, lbfgs_epochs=1000, lambda_bc=1.0)
    
    print("Generating predictions and saving submission...")
    predict_and_save(moe_model, csv_file, submission_csv='/kaggle/working/submission.csv')



#ReLOBRaLO
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# -----------------------
# Device and Precision Setup
# -----------------------
torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -----------------------
# Problem Parameters
# -----------------------
eps = 10e-4
b_x = 2.0
b_y = 3.0

# -----------------------
# Define the Heat Source Function f(x, y)
# -----------------------
def f_func(x, y):
    # x and y are expected to be torch tensors of shape (N, 1)
    term1 = 2 * eps * (-x + torch.exp(2 * (x - 1) * eps))
    term2 = x * y**2
    term3 = 6 * x * y
    term4 = - x * torch.exp((3 * (y - 1)) / eps)
    term5 = - y**2 * torch.exp((2 * (x - 1)) / eps)
    term6 = 2 * y**2
    term7 = -6 * y * torch.exp((2 * (x - 1)) / eps)
    term8 = -2 * torch.exp((3 * (y - 1)) / eps)
    term9 = torch.exp((2 * x + 3 * y - 5) / eps)
    return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9

# -----------------------
# Define the Analytical (Manufactured) Solution
# -----------------------
def analytical_solution(x, y):
    # Manufactured solution: u(x,y) = sin(pi*x) * sin(pi*y)
    return torch.sin(np.pi * x) * torch.sin(np.pi * y)

# -----------------------
# Compute RMSE between model predictions and analytical solution
# -----------------------
def compute_rmse(model, x, y):
    X = torch.cat([x, y], dim=1)
    with torch.no_grad():
        u_pred = model(X)
    u_true = analytical_solution(x, y)
    rmse = torch.sqrt(torch.mean((u_pred - u_true)**2))
    return rmse

# -----------------------
# Define the PINN Model
# -----------------------
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.activation = nn.Tanh()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            nn.init.xavier_normal_(self.layers[-1].weight)
            nn.init.zeros_(self.layers[-1].bias)
            if i < len(layers) - 2:
                self.norms.append(nn.BatchNorm1d(layers[i+1]))
                
    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
            if i < len(self.norms):
                x = self.norms[i](x)
        return self.layers[-1](x)

# -----------------------
# Loading and Partitioning CSV Data
# -----------------------
def load_training_points(csv_file):
    # Force x and y columns to be float64 (17-digit precision)
    df = pd.read_csv(csv_file, dtype={'x': np.float64, 'y': np.float64})
    x = torch.tensor(df['x'].values, dtype=torch.float64, device=device).unsqueeze(1)
    y = torch.tensor(df['y'].values, dtype=torch.float64, device=device).unsqueeze(1)
    return x, y

def partition_points(x, y, tol=1e-6):
    # Points are on the boundary if x or y is within tol of 0 or 1.
    boundary_mask = (torch.abs(x - 0) < tol) | (torch.abs(x - 1) < tol) | \
                    (torch.abs(y - 0) < tol) | (torch.abs(y - 1) < tol)
    interior_mask = ~boundary_mask
    x_int = x[interior_mask.squeeze()]
    y_int = y[interior_mask.squeeze()]
    x_bnd = x[boundary_mask.squeeze()]
    y_bnd = y[boundary_mask.squeeze()]
    return x_int, y_int, x_bnd, y_bnd

# -----------------------
# PDE Residual Computation
# -----------------------
def pde_residual(model, x, y):
    X = torch.cat([x, y], dim=1)
    X.requires_grad_(True)
    u = model(X)
    grad_u = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u),
                                 retain_graph=True, create_graph=True)[0]
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    u_xx = torch.autograd.grad(u_x, X, grad_outputs=torch.ones_like(u_x),
                               retain_graph=True, create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, X, grad_outputs=torch.ones_like(u_y),
                               retain_graph=True, create_graph=True)[0][:, 1:2]
    f_val = f_func(x, y)
    residual = -eps * (u_xx + u_yy) + b_x * u_x + b_y * u_y - f_val
    return residual

# -----------------------
# Training with Adam including RMSE minimization and ReLoBRaLo adaptive weighting
# -----------------------
def train_with_csv_adam_rmse_relobralo(model, csv_file, epochs, lambda_bc=1.0, lambda_rmse=0.1, patience=2000):
    x_all, y_all = load_training_points(csv_file)
    x_int, y_int, x_bnd, y_bnd = partition_points(x_all, y_all)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_loss = float('inf')
    counter = 0
    
    # Initialize exponential moving averages for PDE and BC losses
    ema_pde = 1.0
    ema_bc = 1.0
    alpha = 0.01  # smoothing factor for EMA
    eps_stability = 1e-8  # small constant for stability
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Compute PDE loss on interior points
        if x_int.numel() > 0:
            res = pde_residual(model, x_int, y_int)
            loss_pde = torch.mean(res**2)
        else:
            loss_pde = torch.tensor(0.0, dtype=torch.float64, device=device)
        
        # Compute Boundary loss: enforce u(x,y)=0 on boundary points
        if x_bnd.numel() > 0:
            X_bnd = torch.cat([x_bnd, y_bnd], dim=1)
            u_bnd = model(X_bnd)
            loss_bc = torch.mean(u_bnd**2)
        else:
            loss_bc = torch.tensor(0.0, dtype=torch.float64, device=device)
        
        # Update exponential moving averages for adaptive weighting (ReLoBRaLo)
        ema_pde = (1 - alpha) * ema_pde + alpha * loss_pde.item()
        ema_bc = (1 - alpha) * ema_bc + alpha * loss_bc.item()
        
        # Compute adaptive weights based on the EMAs
        w_pde = ema_bc / (ema_pde + ema_bc + eps_stability)
        w_bc = ema_pde / (ema_pde + ema_bc + eps_stability)
        
        # Compute RMSE loss: difference between predicted and analytical solution over all points
        rmse_loss = compute_rmse(model, x_all, y_all)
        
        # Total loss: combine weighted PDE loss, weighted BC loss, and RMSE loss
        loss = w_pde * loss_pde + lambda_bc * w_bc * loss_bc + lambda_rmse * rmse_loss
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            counter = 0
        else:
            counter += 1
        
        if epoch % 1000 == 0:
            print(f"[Adam] Epoch {epoch}: Total Loss = {loss.item():.6e}, PDE Loss = {loss_pde.item():.6e}, "
                  f"BC Loss = {loss_bc.item():.6e}, RMSE = {rmse_loss.item():.6e}, "
                  f"w_pde = {w_pde:.4f}, w_bc = {w_bc:.4f}")
        if counter >= patience:
            print(f"[Adam] Early stopping at epoch {epoch} with best loss {best_loss:.6e}")
            break

# -----------------------
# Fine-Tuning with LBFGS
# -----------------------
def train_with_csv_lbfgs(model, csv_file, lbfgs_epochs, lambda_bc=1.0):
    x_all, y_all = load_training_points(csv_file)
    x_int, y_int, x_bnd, y_bnd = partition_points(x_all, y_all)
    
    optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=0.1, max_iter=500, history_size=50)
    
    def closure():
        optimizer_lbfgs.zero_grad()
        if x_int.numel() > 0:
            res = pde_residual(model, x_int, y_int)
            loss_pde = torch.mean(res**2)
        else:
            loss_pde = torch.tensor(0.0, dtype=torch.float64, device=device)
        
        if x_bnd.numel() > 0:
            X_bnd = torch.cat([x_bnd, y_bnd], dim=1)
            u_bnd = model(X_bnd)
            loss_bc = torch.mean(u_bnd**2)
        else:
            loss_bc = torch.tensor(0.0, dtype=torch.float64, device=device)
        
        loss = loss_pde + lambda_bc * loss_bc
        loss.backward()
        return loss
    
    for i in range(lbfgs_epochs):
        loss = optimizer_lbfgs.step(closure)
        if i % 50 == 0:
            print(f"[LBFGS] Iteration {i}: Loss = {loss.item():.6e}")

# -----------------------
# Prediction and Saving Submission
# -----------------------
def predict_and_save(model, csv_file, submission_csv='/kaggle/working/submission.csv'):
    # Read test points from CSV file with full precision
    df = pd.read_csv(csv_file, dtype={'x': np.float64, 'y': np.float64})
    x = torch.tensor(df['x'].values, dtype=torch.float64, device=device).unsqueeze(1)
    y = torch.tensor(df['y'].values, dtype=torch.float64, device=device).unsqueeze(1)
    X = torch.cat([x, y], dim=1)
    
    model.eval()
    with torch.no_grad():
        u_pred = model(X).cpu().numpy().flatten()
    
    # Create submission DataFrame; preserve "ID" column if it exists.
    if 'ID' in df.columns:
        submission = pd.DataFrame({
            'ID': df['ID'],
            'x': df['x'],
            'y': df['y'],
            'u': u_pred
        })
    else:
        submission = pd.DataFrame({
            'ID': range(1, len(u_pred) + 1),
            'x': df['x'],
            'y': df['y'],
            'u': u_pred
        })
    submission.to_csv(submission_csv, index=False, float_format='%.17f')
    print(f"Submission saved to {submission_csv}")

# -----------------------
# Main Execution
# -----------------------
if __name__ == "__main__":
    # Define the model architecture (example: 4 hidden layers with 50, 100, 50 neurons)
    layers = [2, 100,100, 1]
    model = PINN(layers).to(device)
    
    csv_file = '/kaggle/input/ziq-sciml-challenge/test.csv'  # Use your test CSV file as training data
    
    print("Starting Adam training with RMSE minimization and ReLoBRaLo loss...")
    train_with_csv_adam_rmse_relobralo(model, csv_file, epochs=5000, lambda_bc=1.0, lambda_rmse=1)
    
    print("Starting LBFGS fine-tuning...")
    # Uncomment the following line to run LBFGS fine-tuning:
    train_with_csv_lbfgs(model, csv_file, lbfgs_epochs=2000, lambda_bc=1.0)
    
    print("Generating predictions and saving submission...")
    predict_and_save(model, csv_file, submission_csv='/kaggle/working/submission.csv')



#MoE Learning Rate Annealing
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import math

# -----------------------
# Device and Precision Setup
# -----------------------
torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -----------------------
# Problem Parameters
# -----------------------
eps = 1e-4  # Corrected scientific notation
b_x = 2.0
b_y = 3.0

# -----------------------
# Define the Heat Source Function f(x, y)
# -----------------------
def f_func(x, y):
    term1 = 2 * eps * (-x + torch.exp(2 * (x - 1) / eps))
    term2 = x * y**2
    term3 = 6 * x * y
    term4 = -x * torch.exp(3 * (y - 1) / eps)
    term5 = -y**2 * torch.exp(2 * (x - 1) / eps)
    term6 = 2 * y**2
    term7 = -6 * y * torch.exp(2 * (x - 1) / eps)
    term8 = -2 * torch.exp(3 * (y - 1) / eps)
    term9 = torch.exp((2 * x + 3 * y - 5) / eps)
    return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9

# -----------------------
# Modified Activation Functions for Boundary Conditions
# -----------------------
class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class SwishActivation(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ModifiedTanh(nn.Module):
    def forward(self, x):
        return torch.tanh(x)**2

# -----------------------
# Boundary-Aware Neural Network
# -----------------------
class PINN(nn.Module):
    def __init__(self, layers, activation, boundary_aware=True):
        super(PINN, self).__init__()
        
        # Activation function selection
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sin':
            self.activation = torch.sin
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'swish':
            self.activation = SwishActivation()
        elif activation == 'modified_tanh':
            self.activation = ModifiedTanh()
        else:
            self.activation = torch.tanh
            
        self.boundary_aware = boundary_aware
        
        # Network layers
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            
        # Special initialization for better convergence
        for i in range(len(self.layers)):
            nn.init.xavier_normal_(self.layers[i].weight, gain=1.5)
            nn.init.zeros_(self.layers[i].bias)
                
    def forward(self, x):
        inputs = x.clone()  # Store original inputs for boundary enforcement
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
        
        # Final layer without activation
        output = self.layers[-1](x)
        
        # Enforce zero boundary conditions if boundary_aware is True
        if self.boundary_aware:
            # Get x and y coordinates
            x_coord = inputs[:, 0:1]
            y_coord = inputs[:, 1:2]
            
            # Compute distance to boundaries
            dist_left = x_coord
            dist_right = 1.0 - x_coord
            dist_bottom = y_coord
            dist_top = 1.0 - y_coord
            
            # Compute boundary factor (approaches 0 near boundaries)
            boundary_factor = dist_left * dist_right * dist_bottom * dist_top
            
            # Scale the output to satisfy boundary conditions
            return output * boundary_factor
            
        return output

# -----------------------
# FeedForward Network with Residual Connections
# -----------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim, activation):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sin':
            self.activation = torch.sin
        elif activation == 'swish':
            self.activation = SwishActivation()
        else:
            self.activation = torch.tanh
            
        # Initialize with small weights for stability
        nn.init.xavier_normal_(self.linear1.weight, gain=0.5)
        nn.init.xavier_normal_(self.linear2.weight, gain=0.5)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
        
    def forward(self, x):
        identity = x
        out = self.activation(self.linear1(x))
        out = self.linear2(out)
        return self.activation(out + identity)

class ResidualPINN(nn.Module):
    def __init__(self, layers, activation='tanh', num_res_blocks=2, boundary_aware=True):
        super(ResidualPINN, self).__init__()
        
        # Input layer
        model_layers = [nn.Linear(layers[0], layers[1])]
        
        # Activation
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sin':
            self.activation = torch.sin
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'swish':
            self.activation = SwishActivation()
        else:
            self.activation = torch.tanh
            
        # Add residual blocks
        self.res_blocks = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.res_blocks.append(ResidualBlock(layers[1], activation))
            
        # Output layer
        self.output_layer = nn.Linear(layers[1], layers[-1])
        
        # Initialize input and output layers
        nn.init.xavier_normal_(model_layers[0].weight, gain=1.0)
        nn.init.zeros_(model_layers[0].bias)
        nn.init.xavier_normal_(self.output_layer.weight, gain=1.0)
        nn.init.zeros_(self.output_layer.bias)
        
        self.model = nn.ModuleList(model_layers)
        self.boundary_aware = boundary_aware
        
    def forward(self, x):
        inputs = x.clone()  # Store original inputs
        
        # Input layer with activation
        out = self.activation(self.model[0](x))
        
        # Residual blocks
        for res_block in self.res_blocks:
            out = res_block(out)
            
        # Output layer
        out = self.output_layer(out)
        
        # Apply boundary condition enforcement
        if self.boundary_aware:
            # Get x and y coordinates
            x_coord = inputs[:, 0:1]
            y_coord = inputs[:, 1:2]
            
            # Compute distance to boundaries
            dist_left = x_coord
            dist_right = 1.0 - x_coord
            dist_bottom = y_coord
            dist_top = 1.0 - y_coord
            
            # More sophisticated boundary factor that creates sharper transitions
            boundary_factor = (dist_left * dist_right * dist_bottom * dist_top)
            
            # Scale the output to satisfy boundary conditions
            return out * boundary_factor
            
        return out

# -----------------------
# Define the Gating Network with Better Architecture
# -----------------------
class ImprovedGatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_dims=[40, 20]):
        super(ImprovedGatingNetwork, self).__init__()
        
        dims = [input_dim] + hidden_dims
        self.layers = nn.ModuleList()
        
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            
        # Final layer to produce logits for each expert
        self.output_layer = nn.Linear(dims[-1], num_experts)
        
        # Special initialization for better convergence
        for i in range(len(self.layers)):
            nn.init.xavier_normal_(self.layers[i].weight, gain=1.0)
            nn.init.zeros_(self.layers[i].bias)
            
        nn.init.xavier_normal_(self.output_layer.weight, gain=0.1)  # Small weights to start with mild gating
        nn.init.zeros_(self.output_layer.bias)
        
        # Add temperature parameter for softmax
        self.temperature = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        
    def forward(self, x):
        for layer in self.layers:
            x = torch.tanh(layer(x))
            
        logits = self.output_layer(x)
        
        # Apply temperature scaling for sharper or smoother expert selection
        weights = torch.softmax(logits / torch.clamp(self.temperature, min=0.1), dim=1)
        
        return weights

# -----------------------
# Improved Mixture-of-Experts (MoE) PINN Model
# -----------------------
class ImprovedMoEPINN(nn.Module):
    def __init__(self, experts, gating_net):
        super(ImprovedMoEPINN, self).__init__()
        self.experts = nn.ModuleList(experts)
        self.gating_net = gating_net
        
    def forward(self, x):
        # Compute weights from the gating network
        gate_weights = self.gating_net(x)  # shape: (N, num_experts)
        
        # Compute each expert's output
        expert_outputs = [expert(x) for expert in self.experts]
        
        # Stack expert outputs: shape (num_experts, N, 1)
        outputs = torch.stack(expert_outputs, dim=0)
        
        # Permute to shape (N, num_experts, 1)
        outputs = outputs.permute(1, 0, 2)
        
        # Multiply each expert's output by its corresponding weight and sum
        gate_weights = gate_weights.unsqueeze(2)  # shape: (N, num_experts, 1)
        output = torch.sum(gate_weights * outputs, dim=1)  # shape: (N, 1)
        
        return output
        
    def get_expert_weights(self, x):
        """Return the weights assigned to each expert for visualization/debugging"""
        return self.gating_net(x)

# -----------------------
# Loading and Partitioning CSV Data
# -----------------------
def load_training_points(csv_file):
    df = pd.read_csv(csv_file, dtype={'x': np.float64, 'y': np.float64})
    
    # Round coordinates to 6 decimal places
    df['x'] = df['x'].round(2)
    df['y'] = df['y'].round(2)
    
    # Remove duplicates after rounding
    # We need to check for duplicates in the combined (x,y) pairs, not individually
    df = df.drop_duplicates(subset=['x', 'y'])
    
    # Print statistics about deduplication
    print(f"After rounding to 6 decimal places and removing duplicates:")
    print(f"  - Original size: {len(pd.read_csv(csv_file))}")
    print(f"  - New size: {len(df)}")
    print(f"  - Removed {len(pd.read_csv(csv_file)) - len(df)} duplicate points")
    
    x = torch.tensor(df['x'].values, dtype=torch.float64, device=device).unsqueeze(1)
    y = torch.tensor(df['y'].values, dtype=torch.float64, device=device).unsqueeze(1)
    return x, y

def partition_points(x, y, tol=1e-10):
    """Partition points into boundary and interior with improved precision"""
    boundary_mask = (torch.abs(x) < tol) | (torch.abs(x - 1.0) < tol) | \
                    (torch.abs(y) < tol) | (torch.abs(y - 1.0) < tol)
    interior_mask = ~boundary_mask
    
    x_int = x[interior_mask.squeeze()]
    y_int = y[interior_mask.squeeze()]
    x_bnd = x[boundary_mask.squeeze()]
    y_bnd = y[boundary_mask.squeeze()]
    
    return x_int, y_int, x_bnd, y_bnd

# -----------------------
# PDE Residual Computation with Improved Numerical Stability
# -----------------------
def pde_residual(model, x, y):
    X = torch.cat([x, y], dim=1)
    X.requires_grad_(True)
    u = model(X)
    
    # Compute first derivatives
    grad_u = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u),
                                 retain_graph=True, create_graph=True)[0]
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    
    # Compute second derivatives with better numerical stability
    u_xx = torch.autograd.grad(u_x, X, grad_outputs=torch.ones_like(u_x),
                               retain_graph=True, create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, X, grad_outputs=torch.ones_like(u_y),
                               retain_graph=True, create_graph=True)[0][:, 1:2]
    
    # Compute PDE residual
    f_val = f_func(x, y)
    residual = -eps * (u_xx + u_yy) + b_x * u_x + b_y * u_y - f_val
    
    return residual

# -----------------------
# Improved Training Function with Robust Adaptive Weighting
# -----------------------
def train_improved_moe(model, csv_file, epochs, lambda_bc=10.0, patience=2000, lr=1e-3, weight_decay=1e-8):
    x_all, y_all = load_training_points(csv_file)
    x_int, y_int, x_bnd, y_bnd = partition_points(x_all, y_all)
    
    # Add additional points near boundaries to capture steep gradients
    n_boundary_points = 1000
    additional_x = torch.zeros((n_boundary_points, 1), device=device, dtype=torch.float64)
    additional_y = torch.zeros((n_boundary_points, 1), device=device, dtype=torch.float64)
    
    # Points close to x=0 boundary
    idx = 0
    for i in range(250):
        x_val = 0.001 + i * 0.001  # Points starting from 0.001 up to 0.25
        for j in range(4):
            y_val = 0.2 + j * 0.2  # Evenly spaced in y-direction
            additional_x[idx] = x_val
            additional_y[idx] = y_val
            idx += 1
            
    # Add additional interior points if needed
    if x_int.numel() > 0:
        x_int = torch.cat([x_int, additional_x], dim=0)
        y_int = torch.cat([y_int, additional_y], dim=0)
    
    # Adam optimizer with weight decay to prevent overfitting
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience//10, 
                                                    factor=0.5, min_lr=1e-6, verbose=True)
    
    # Initialize tracking variables
    best_loss = float('inf')
    counter = 0
    
    # Initialize EMAs for loss components and adaptive weights
    ema_pde = 1.0
    ema_bc = 1.0
    alpha = 0.01  # EMA update rate
    eps_stability = 1e-12  # Numerical stability
    
    # Weight annealing parameters
    lambda_bc_start = lambda_bc
    lambda_bc_end = lambda_bc * 10  # Gradually increase BC importance
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Compute PDE loss on interior points
        if x_int.numel() > 0:
            res = pde_residual(model, x_int, y_int)
            loss_pde = torch.mean(res**2)
        else:
            loss_pde = torch.tensor(0.0, dtype=torch.float64, device=device)
        
        # Compute boundary condition loss
        if x_bnd.numel() > 0:
            X_bnd = torch.cat([x_bnd, y_bnd], dim=1)
            u_bnd = model(X_bnd)
            loss_bc = torch.mean(u_bnd**2)
        else:
            loss_bc = torch.tensor(0.0, dtype=torch.float64, device=device)
        
        # Update EMAs for adaptive weighting
        ema_pde = (1 - alpha) * ema_pde + alpha * loss_pde.item()
        ema_bc = (1 - alpha) * ema_bc + alpha * loss_bc.item()
        
        # Calculate adaptive weights
        w_pde = ema_bc / (ema_pde + ema_bc + eps_stability)
        w_bc = ema_pde / (ema_pde + ema_bc + eps_stability)
        
        # Anneal boundary condition weight
        progress = min(epoch / (epochs * 0.5), 1.0)  # First half of training
        current_lambda_bc = lambda_bc_start + progress * (lambda_bc_end - lambda_bc_start)
        
        # Compute total loss with adaptive weighting
        loss = w_pde * loss_pde + current_lambda_bc * w_bc * loss_bc
        
        # Backward pass and optimization step
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        
        optimizer.step()
        
        # Update learning rate scheduler
        scheduler.step(loss)
        
        # Early stopping logic
        if loss.item() < best_loss:
            best_loss = loss.item()
            counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            counter += 1
        
        # Logging
        if epoch % 100 == 0:
            print(f"[Adam] Epoch {epoch}: Loss = {loss.item():.6e}, PDE Loss = {loss_pde.item():.6e}, "
                  f"BC Loss = {loss_bc.item():.6e}, λ_BC = {current_lambda_bc:.2f}, "
                  f"w_pde = {w_pde:.4f}, w_bc = {w_bc:.4f}, LR = {optimizer.param_groups[0]['lr']:.2e}")
            
        # Early stopping
        if counter >= patience:
            print(f"[Adam] Early stopping at epoch {epoch} with best loss {best_loss:.6e}")
            # Load best model
            model.load_state_dict(torch.load('best_model.pt'))
            break
            
    return model

# -----------------------
# L-BFGS Fine-Tuning with Heavy Boundary Enforcement
# -----------------------
def fine_tune_with_lbfgs(model, csv_file, lbfgs_epochs=200, lambda_bc=100.0):
    print("Starting L-BFGS fine-tuning...")
    x_all, y_all = load_training_points(csv_file)
    x_int, y_int, x_bnd, y_bnd = partition_points(x_all, y_all)
    
    # Optimizer with appropriate parameters for the problem
    optimizer_lbfgs = optim.LBFGS(model.parameters(), 
                                  lr=0.5, 
                                  max_iter=20,
                                  history_size=20,
                                  line_search_fn="strong_wolfe")
    
    def closure():
        optimizer_lbfgs.zero_grad()
        
        # PDE residual
        if x_int.numel() > 0:
            res = pde_residual(model, x_int, y_int)
            loss_pde = torch.mean(res**2)
        else:
            loss_pde = torch.tensor(0.0, dtype=torch.float64, device=device)
        
        # Boundary conditions
        if x_bnd.numel() > 0:
            X_bnd = torch.cat([x_bnd, y_bnd], dim=1)
            u_bnd = model(X_bnd)
            loss_bc = torch.mean(u_bnd**2)
        else:
            loss_bc = torch.tensor(0.0, dtype=torch.float64, device=device)
        
        # Emphasize boundary conditions strongly
        loss = loss_pde + lambda_bc * loss_bc
        
        # Logger
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: Loss is NaN or Inf in L-BFGS, skipping update")
            loss = torch.tensor(1e10, dtype=torch.float64, device=device, requires_grad=True)
        else:
            loss.backward()
            
        return loss
    
    # Main training loop
    for i in range(lbfgs_epochs):
        try:
            loss = optimizer_lbfgs.step(closure)
            
            if i % 10 == 0:
                # Check boundary conditions
                if x_bnd.numel() > 0:
                    X_bnd = torch.cat([x_bnd, y_bnd], dim=1)
                    with torch.no_grad():
                        u_bnd = model(X_bnd)
                    bc_error = torch.mean(u_bnd**2).item()
                    print(f"[LBFGS] Iteration {i}: Loss = {loss.item():.6e}, BC Error = {bc_error:.6e}")
                else:
                    print(f"[LBFGS] Iteration {i}: Loss = {loss.item():.6e}")
        except Exception as e:
            print(f"Error in L-BFGS iteration {i}: {e}")
            break
            
    return model

# -----------------------
# Prediction and Saving Submission
# -----------------------
def predict_and_save(model, csv_file, submission_csv='/kaggle/working/submission.csv'):
    df = pd.read_csv(csv_file, dtype={'x': np.float64, 'y': np.float64})
    x = torch.tensor(df['x'].values, dtype=torch.float64, device=device).unsqueeze(1)
    y = torch.tensor(df['y'].values, dtype=torch.float64, device=device).unsqueeze(1)
    X = torch.cat([x, y], dim=1)
    
    model.eval()
    with torch.no_grad():
        u_pred = model(X).cpu().numpy().flatten()
        
    # Ensure predictions exactly satisfy the boundary conditions
    for i in range(len(df)):
        x_val, y_val = df.iloc[i]['x'], df.iloc[i]['y']
        # Check if point is on boundary (with small tolerance)
        if (abs(x_val) < 1e-10 or abs(x_val - 1.0) < 1e-10 or 
            abs(y_val) < 1e-10 or abs(y_val - 1.0) < 1e-10):
            u_pred[i] = 0.0
    
    if 'ID' in df.columns:
        submission = pd.DataFrame({'ID': df['ID'], 'x': df['x'], 'y': df['y'], 'u': u_pred})
    else:
        submission = pd.DataFrame({'ID': range(1, len(u_pred)+1), 'x': df['x'], 'y': df['y'], 'u': u_pred})
        
    submission.to_csv(submission_csv, index=False, float_format='%.17f')
    print(f"Submission saved to {submission_csv}")

# -----------------------
# Main Execution
# -----------------------
if __name__ == "__main__":
    print("Initializing improved PINN solution for convection-diffusion problem...")
    
    # Fix errors in heat source function
    print("Verifying heat source function...")
    x_test = torch.tensor([[0.5]], dtype=torch.float64, device=device)
    y_test = torch.tensor([[0.5]], dtype=torch.float64, device=device)
    f_value = f_func(x_test, y_test)
    print(f"f(0.5, 0.5) = {f_value.item()}")
    
    # Create a diverse set of experts with different architectures and activations
    print("Creating expert models...")
    
    # Expert 1: Standard PINN with tanh activation
    expert1 = PINN([2, 80, 80, 80, 1], 'tanh', boundary_aware=True).to(device)
    
    # Expert 2: PINN with sine activation (good for oscillatory phenomena)
    expert2 = PINN([2, 60, 120, 60, 1], 'sin', boundary_aware=True).to(device)
    
    # Expert 3: PINN with swish activation (often performs well)
    expert3 = PINN([2, 100, 100, 1], 'swish', boundary_aware=True).to(device)
    
    # Expert 4: Residual network with tanh (good for complex functions)
    expert4 = ResidualPINN([2, 64, 1], 'tanh', num_res_blocks=4, boundary_aware=True).to(device)
    
    # Expert 5: Very wide network for capturing fine details
    expert5 = PINN([2, 120, 120, 120, 1], 'modified_tanh', boundary_aware=True).to(device)
    
    # Combine experts
    experts = [expert1, expert2, expert3, expert4, expert5]
    
    # Create improved gating network
    gating_net = ImprovedGatingNetwork(input_dim=2, num_experts=len(experts), hidden_dims=[40, 40]).to(device)
    
    # Build the MoE PINN model
    moe_model = ImprovedMoEPINN(experts, gating_net).to(device)
    print(f"Created MoE model with {len(experts)} experts")
    
    # Path to the CSV file
    csv_file = '/kaggle/input/ziq-sciml-challenge/test.csv'
    
    # First phase: Train with Adam and adaptive weighting
    print("Starting improved Adam training...")
    moe_model = train_improved_moe(
        moe_model, 
        csv_file, 
        epochs=10000,  # More epochs but with better early stopping
        lambda_bc=20.0,  # Stronger boundary condition enforcement 
        patience=500,    # More responsive early stopping
        lr=1e-6,
        weight_decay=1e-8
    )
    
    # Second phase: Fine-tune with L-BFGS
    moe_model = fine_tune_with_lbfgs(
        moe_model,
        csv_file,
        lbfgs_epochs=200,
        lambda_bc=100.0  # Even stronger boundary emphasis
    )
    
    # Generate predictions and save submission
    print("Generating predictions and saving submission...")
    predict_and_save(moe_model, csv_file, submission_csv='/kaggle/working/submission.csv')
    
    print("PINN training and prediction complete!")


#Ensemble
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time
import math

# -----------------------
# Device and Precision Setup
# -----------------------
torch.set_default_dtype(torch.float64)  # Changed from float64 to float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -----------------------
# Problem Parameters
# -----------------------
eps = 10e-4
b_x = 2.0
b_y = 3.0

# -----------------------
# Define the Heat Source Function f(x, y)
# -----------------------
def f_func(x, y):
    term1 = 2 * eps * (-x + torch.exp(2 * (x - 1) / eps))
    term2 = x * y**2
    term3 = 6 * x * y
    term4 = - x * torch.exp((3 * (y - 1)) / eps)
    term5 = - y**2 * torch.exp((2 * (x - 1)) / eps)
    term6 = 2 * y**2
    term7 = -6 * y * torch.exp((2 * (x - 1)) / eps)
    term8 = -2 * torch.exp((3 * (y - 1)) / eps)
    term9 = torch.exp((2 * x + 3 * y - 5) / eps)
    return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9

# -----------------------
# Define the Analytical Solution
# -----------------------
def analytical_solution(x, y):
    return x * (1 - x) * y * (1 - y) * torch.exp((2 * x + 3 * y - 5) / eps)

# -----------------------
# Compute RMSE between model predictions and analytical solution
# -----------------------
def compute_rmse(model, x, y):
    X = torch.cat([x, y], dim=1)
    with torch.no_grad():
        u_pred = model(X)
    u_true = analytical_solution(x, y)
    rmse = torch.sqrt(torch.mean((u_pred - u_true)**2))
    return rmse, u_true, u_pred

# -----------------------
# Simplified Activation Function
# -----------------------
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# -----------------------
# Define the Smaller PINN Model
# -----------------------
class OptimizedPINN(nn.Module):
    def __init__(self, layers, activation=None):
        super(OptimizedPINN, self).__init__()
        
        # Select activation function
        self.activation = activation if activation is not None else Swish()
        
        # Network layers
        self.layers = nn.ModuleList()
        
        # Simpler initialization
        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i+1])
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)
            
    def forward(self, x):
        # Forward pass
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation(x)
        
        # Final layer
        x = self.layers[-1](x)
        return x

# -----------------------
# PDE Residual Computation with Auto-Differentiation
# -----------------------
def pde_residual(model, x, y):
    X = torch.cat([x, y], dim=1)
    X.requires_grad_(True)
    u = model(X)
    
    # Compute first derivatives
    grad_u = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u),
                                 retain_graph=True, create_graph=True)[0]
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    
    # Compute second derivatives
    u_xx = torch.autograd.grad(u_x, X, grad_outputs=torch.ones_like(u_x),
                               retain_graph=True, create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, X, grad_outputs=torch.ones_like(u_y),
                               retain_graph=True, create_graph=True)[0][:, 1:2]
    
    # Compute PDE residual
    f_val = f_func(x, y)
    residual = -eps * (u_xx + u_yy) + b_x * u_x + b_y * u_y - f_val
    
    return residual

# -----------------------
# Explicitly Create Boundary Points
# -----------------------
def create_boundary_points(num_points_per_edge=400):
    # Create points along each edge of the unit square
    t = torch.linspace(0, 1, num_points_per_edge, device=device)
    
    # Bottom edge: y=0, x∈[0,1]
    x_bottom = t
    y_bottom = torch.zeros_like(t)
    
    # Top edge: y=1, x∈[0,1]
    x_top = t
    y_top = torch.ones_like(t)
    
    # Left edge: x=0, y∈[0,1]
    x_left = torch.zeros_like(t)
    y_left = t
    
    # Right edge: x=1, y∈[0,1]
    x_right = torch.ones_like(t)
    y_right = t
    
    # Combine all edges
    x_bnd = torch.cat([x_bottom, x_top, x_left, x_right])
    y_bnd = torch.cat([y_bottom, y_top, y_left, y_right])
    
    # Reshape to column vectors
    x_bnd = x_bnd.reshape(-1, 1)
    y_bnd = y_bnd.reshape(-1, 1)
    
    return x_bnd, y_bnd

# -----------------------
# Loading and Partitioning CSV Data
# -----------------------
def load_training_points(csv_file):
    df = pd.read_csv(csv_file)
    x = torch.tensor(df['x'].values, dtype=torch.float64, device=device).unsqueeze(1)
    y = torch.tensor(df['y'].values, dtype=torch.float64, device=device).unsqueeze(1)
    return x, y

def partition_points(x, y, tol=1e-5):
    boundary_mask = (torch.abs(x) < tol) | (torch.abs(x - 1) < tol) | \
                    (torch.abs(y) < tol) | (torch.abs(y - 1) < tol)
    interior_mask = ~boundary_mask
    x_int = x[interior_mask.squeeze()]
    y_int = y[interior_mask.squeeze()]
    x_bnd = x[boundary_mask.squeeze()]
    y_bnd = y[boundary_mask.squeeze()]
    return x_int, y_int, x_bnd, y_bnd

# -----------------------
# Generate Collocation Points
# -----------------------
def generate_interior_points(num_points):
    # Generate random points in the interior
    x_int = torch.rand(num_points, 1, dtype=torch.float64, device=device)
    y_int = torch.rand(num_points, 1, dtype=torch.float64, device=device)
    return x_int, y_int

# -----------------------
# Batch Processing for Training
# -----------------------
def get_batch(x, y, batch_size):
    if x.shape[0] <= batch_size:
        return x, y
    
    idx = torch.randperm(x.shape[0])[:batch_size]
    return x[idx], y[idx]

# -----------------------
# Streamlined Training Function with Fixed BC Loss
# -----------------------
def train_model(model, csv_file, epochs=5000, batch_size=1024, lambda_bc=100.0, lr=1e-6):
    # Load data from CSV
    x_all, y_all = load_training_points(csv_file)
    
    # Generate interior points
    x_int, y_int = generate_interior_points(50000)
    
    # Generate explicit boundary points
    x_bnd, y_bnd = create_boundary_points(10000)
    
    print(f"Training with {x_int.shape[0]} interior points and {x_bnd.shape[0]} boundary points")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
    
    best_loss = float('inf')
    best_model_state = None
    
    # For monitoring loss changes
    loss_history = []
    bc_loss_history = []
    pde_loss_history = []
    
    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Use batching for interior points to save memory
        x_int_batch, y_int_batch = get_batch(x_int, y_int, batch_size)
            
        # Use batching for boundary points
        x_bnd_batch, y_bnd_batch = get_batch(x_bnd, y_bnd, batch_size)
        
        # PDE loss on interior points
        res = pde_residual(model, x_int_batch, y_int_batch)
        loss_pde = torch.mean(res**2)
        
        # Boundary loss on boundary points
        X_bnd = torch.cat([x_bnd_batch, y_bnd_batch], dim=1)
        u_bnd = model(X_bnd)
        u_bnd_true = torch.zeros_like(u_bnd)  # Dirichlet BC: u=0 on boundary
        loss_bc = torch.mean((u_bnd - u_bnd_true)**2)
        
        # Total loss
        loss = loss_pde + lambda_bc * loss_bc
        
        # Record loss history
        loss_history.append(loss.item())
        bc_loss_history.append(loss_bc.item())
        pde_loss_history.append(loss_pde.item())
        
        # Backpropagation
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Save best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = {key: val.cpu() for key, val in model.state_dict().items()}
        
        # Print progress
        if epoch % 100 == 0:
            elapsed = time.time() - start_time
            # Calculate RMSE using a small batch
            test_idx = torch.randperm(x_all.shape[0])[:min(1000, x_all.shape[0])]
            x_test, y_test = x_all[test_idx], y_all[test_idx]
            rmse, _, _ = compute_rmse(model, x_test, y_test)
            
            print(f"Epoch {epoch}/{epochs} [{elapsed:.2f}s]")
            print(f"  Loss: {loss.item():.4e}, PDE: {loss_pde.item():.4e}, BC: {loss_bc.item():.4e}, RMSE: {rmse.item():.4e}")
            
            # Check if boundary loss is improving
            if epoch > 0 and epoch % 10000 == 0:
                avg_bc_loss_start = sum(bc_loss_history[max(0, epoch-1000):max(0, epoch-900)]) / 100
                avg_bc_loss_end = sum(bc_loss_history[epoch-100:epoch]) / 100
                
                print(f"  BC Loss change: {avg_bc_loss_start:.6e} -> {avg_bc_loss_end:.6e}")
                
                # If boundary loss isn't improving, increase lambda_bc
                if avg_bc_loss_end > 0.9 * avg_bc_loss_start and epoch > 1000:
                    lambda_bc *= 2.0
                    print(f"  Increasing boundary weight to {lambda_bc}")
            
            # Save checkpoint occasionally
            if epoch % 10000 == 0 and epoch > 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'loss': loss.item(),
                    'rmse': rmse.item()
                }, f'checkpoint_epoch_{epoch}.pt')
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict({key: val.to(device) for key, val in best_model_state.items()})
    
    # Calculate final RMSE on smaller batches
    batch_size = 500
    num_batches = (x_all.shape[0] + batch_size - 1) // batch_size
    rmse_total = 0.0
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, x_all.shape[0])
        x_batch = x_all[start_idx:end_idx]
        y_batch = y_all[start_idx:end_idx]
        rmse, _, _ = compute_rmse(model, x_batch, y_batch)
        rmse_total += rmse.item() * (end_idx - start_idx)
    
    final_rmse = rmse_total / x_all.shape[0]
    print(f"Final RMSE: {final_rmse:.17e}")
    
    return model

# -----------------------
# Prediction and Saving Submission
# -----------------------
def predict_and_save(model, csv_file, submission_csv='submission.csv'):
    df = pd.read_csv(csv_file)
    
    # Process in batches to save memory
    batch_size = 500
    num_samples = len(df)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    u_pred_list = []
    model.eval()
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        
        batch_df = df.iloc[start_idx:end_idx]
        x = torch.tensor(batch_df['x'].values, dtype=torch.float64, device=device).unsqueeze(1)
        y = torch.tensor(batch_df['y'].values, dtype=torch.float64, device=device).unsqueeze(1)
        X = torch.cat([x, y], dim=1)
        
        with torch.no_grad():
            u_pred_batch = model(X).cpu().numpy().flatten()
            u_pred_list.append(u_pred_batch)
    
    u_pred = np.concatenate(u_pred_list)
    
    if 'ID' in df.columns:
        submission = pd.DataFrame({'ID': df['ID'], 'u': u_pred})
    else:
        submission = pd.DataFrame({'ID': range(1, len(u_pred)+1), 'u': u_pred})
    
    submission.to_csv(submission_csv, index=False)
    print(f"Submission saved to {submission_csv}")

# -----------------------
# Run a Quick Test to Verify Boundary Conditions
# -----------------------
def test_boundary_conditions(model):
    # Create boundary points
    num_test = 100
    x_bnd, y_bnd = create_boundary_points(num_test)
    X_bnd = torch.cat([x_bnd, y_bnd], dim=1)
    
    # Get model predictions
    with torch.no_grad():
        u_bnd = model(X_bnd)
    
    # Check if boundary values are close to zero
    avg_bc_value = torch.mean(torch.abs(u_bnd)).item()
    max_bc_value = torch.max(torch.abs(u_bnd)).item()
    
    print(f"Boundary condition test:")
    print(f"  Average |u| on boundary: {avg_bc_value:.17e}")
    print(f"  Maximum |u| on boundary: {max_bc_value:.17e}")
    
    return avg_bc_value, max_bc_value

# -----------------------
# Main Execution
# -----------------------
if __name__ == "__main__":
    # Define smaller network architecture
    layers = [2, 250,250,250, 1]  # Slightly increased width for better capacity
    
    # Create a more memory-efficient PINN model
    model = OptimizedPINN(
        layers=layers,
        activation=Swish()
    ).to(device)
    
    csv_file = '/kaggle/input/ziq-sciml-challenge/test.csv'  # Adjust path as needed
    
    # Verify the model can handle boundary conditions
    print("Testing initial boundary condition handling...")
    test_boundary_conditions(model)
    
    print("Starting training with memory-optimized model...")
    model = train_model(
        model=model,
        csv_file=csv_file,
        epochs=20000,
        batch_size=1000,
        lambda_bc=3,  # Increased boundary condition weight
        lr=5e-7
    )
    
    # Verify boundary conditions are being enforced
    print("Testing final boundary condition handling...")
    test_boundary_conditions(model)
    
    print("Generating predictions and saving submission...")
    predict_and_save(model, csv_file, submission_csv='submission.csv')


import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time
import math
import matplotlib.pyplot as plt

# -----------------------
# Device and Precision Setup
# -----------------------
torch.set_default_dtype(torch.float64)  # Use double precision
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -----------------------
# Problem Parameters
# -----------------------
eps = 10e-4
b_x = 2.0
b_y = 3.0

# -----------------------
# Define the Heat Source Function f(x, y)
# -----------------------
def f_func(x, y):
    term1 = 2 * eps * (-x + torch.exp(2 * (x - 1) / eps))
    term2 = x * y**2
    term3 = 6 * x * y
    term4 = - x * torch.exp((3 * (y - 1)) / eps)
    term5 = - y**2 * torch.exp((2 * (x - 1)) / eps)
    term6 = 2 * y**2
    term7 = -6 * y * torch.exp((2 * (x - 1)) / eps)
    term8 = -2 * torch.exp((3 * (y - 1)) / eps)
    term9 = torch.exp((2 * x + 3 * y - 5) / eps)
    return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9

# -----------------------
# Define the Analytical Solution
# -----------------------
def analytical_solution(x, y):
    return x * (1 - x) * y * (1 - y) * torch.exp((2 * x + 3 * y - 5) / eps)

# -----------------------
# Compute RMSE between model predictions and analytical solution
# -----------------------
def compute_rmse(model, x, y):
    X = torch.cat([x, y], dim=1)
    with torch.no_grad():
        u_pred = model(X)
    u_true = analytical_solution(x, y)
    rmse = torch.sqrt(torch.mean((u_pred - u_true)**2))
    return rmse, u_true, u_pred

# -----------------------
# Simplified Activation Function (Swish)
# -----------------------
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# -----------------------
# Define the Optimized PINN Model
# -----------------------
class OptimizedPINN(nn.Module):
    def __init__(self, layers, activation=None):
        super(OptimizedPINN, self).__init__()
        self.activation = activation if activation is not None else Swish()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i+1])
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)
            
    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x

# -----------------------
# PDE Residual Computation with Auto-Differentiation
# -----------------------
def pde_residual(model, x, y):
    X = torch.cat([x, y], dim=1)
    X.requires_grad_(True)
    u = model(X)
    grad_u = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u),
                                 retain_graph=True, create_graph=True)[0]
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    u_xx = torch.autograd.grad(u_x, X, grad_outputs=torch.ones_like(u_x),
                               retain_graph=True, create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, X, grad_outputs=torch.ones_like(u_y),
                               retain_graph=True, create_graph=True)[0][:, 1:2]
    f_val = f_func(x, y)
    residual = -eps * (u_xx + u_yy) + b_x * u_x + b_y * u_y - f_val
    return residual

# -----------------------
# Explicitly Create Boundary Points
# -----------------------
def create_boundary_points(num_points_per_edge=400):
    t = torch.linspace(0, 1, num_points_per_edge, device=device)
    x_bottom = t
    y_bottom = torch.zeros_like(t)
    x_top = t
    y_top = torch.ones_like(t)
    x_left = torch.zeros_like(t)
    y_left = t
    x_right = torch.ones_like(t)
    y_right = t
    x_bnd = torch.cat([x_bottom, x_top, x_left, x_right])
    y_bnd = torch.cat([y_bottom, y_top, y_left, y_right])
    x_bnd = x_bnd.reshape(-1, 1)
    y_bnd = y_bnd.reshape(-1, 1)
    return x_bnd, y_bnd

# -----------------------
# Loading and Partitioning CSV Data
# -----------------------
def load_training_points(csv_file):
    df = pd.read_csv(csv_file)
    x = torch.tensor(df['x'].values, dtype=torch.float64, device=device).unsqueeze(1)
    y = torch.tensor(df['y'].values, dtype=torch.float64, device=device).unsqueeze(1)
    return x, y

def partition_points(x, y, tol=1e-5):
    boundary_mask = (torch.abs(x) < tol) | (torch.abs(x - 1) < tol) | \
                    (torch.abs(y) < tol) | (torch.abs(y - 1) < tol)
    interior_mask = ~boundary_mask
    x_int = x[interior_mask.squeeze()]
    y_int = y[interior_mask.squeeze()]
    x_bnd = x[boundary_mask.squeeze()]
    y_bnd = y[boundary_mask.squeeze()]
    return x_int, y_int, x_bnd, y_bnd

# -----------------------
# Generate Collocation Points (Interior Points)
# -----------------------
def generate_interior_points(num_points):
    x_int = torch.rand(num_points, 1, dtype=torch.float64, device=device)
    y_int = torch.rand(num_points, 1, dtype=torch.float64, device=device)
    return x_int, y_int

# -----------------------
# Batch Processing Utility
# -----------------------
def get_batch(x, y, batch_size):
    if x.shape[0] <= batch_size:
        return x, y
    idx = torch.randperm(x.shape[0])[:batch_size]
    return x[idx], y[idx]

# -----------------------
# Training Function with Adaptive Sampling and LBFGS Fine-Tuning
# -----------------------
def train_model(model, csv_file, epochs=5000, batch_size=1024, lambda_bc=100.0, lr=1e-6, adaptive_threshold=1e-2):
    # Load data from CSV
    x_all, y_all = load_training_points(csv_file)
    
    # Generate interior points
    x_int, y_int = generate_interior_points(40000)
    
    # Generate explicit boundary points
    x_bnd, y_bnd = create_boundary_points(100)
    
    print(f"Training with {x_int.shape[0]} interior points and {x_bnd.shape[0]} boundary points")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.95)
    
    best_loss = float('inf')
    best_model_state = None
    
    loss_history = []
    bc_loss_history = []
    pde_loss_history = []
    
    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # ----- Adaptive Sampling -----
        # Get a uniform batch from interior points
        x_uniform, y_uniform = get_batch(x_int, y_int, batch_size // 2)
        
        # Compute residuals on all interior points for adaptive sampling
        # Compute with grad then detach the result
        residuals_full = pde_residual(model, x_int, y_int).detach()
        high_error_mask = (residuals_full.abs() > adaptive_threshold).squeeze()
        if high_error_mask.sum() > 0:
            x_high = x_int[high_error_mask]
            y_high = y_int[high_error_mask]
            x_adapt, y_adapt = get_batch(x_high, y_high, batch_size // 2)
            # Combine uniform and adaptive samples
            x_batch = torch.cat([x_uniform, x_adapt], dim=0)
            y_batch = torch.cat([y_uniform, y_adapt], dim=0)
        else:
            x_batch, y_batch = get_batch(x_int, y_int, batch_size)
        # ------------------------------
        
        # PDE loss on interior batch
        res = pde_residual(model, x_batch, y_batch)
        loss_pde = torch.mean(res**2)
        
        # Boundary loss on a batch of boundary points
        x_bnd_batch, y_bnd_batch = get_batch(x_bnd, y_bnd, batch_size)
        X_bnd = torch.cat([x_bnd_batch, y_bnd_batch], dim=1)
        u_bnd = model(X_bnd)
        u_bnd_true = torch.zeros_like(u_bnd)  # Dirichlet BC: u=0 on boundary
        loss_bc = torch.mean((u_bnd - u_bnd_true)**2)
        
        loss = loss_pde + lambda_bc * loss_bc
        
        loss_history.append(loss.item())
        bc_loss_history.append(loss_bc.item())
        pde_loss_history.append(loss_pde.item())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = {key: val.cpu() for key, val in model.state_dict().items()}
        
        if epoch % 100 == 0:
            elapsed = time.time() - start_time
            test_idx = torch.randperm(x_all.shape[0])[:min(1000, x_all.shape[0])]
            x_test, y_test = x_all[test_idx], y_all[test_idx]
            rmse, _, _ = compute_rmse(model, x_test, y_test)
            print(f"Epoch {epoch}/{epochs} [{elapsed:.2f}s]")
            print(f"  Loss: {loss.item():.4e}, PDE: {loss_pde.item():.4e}, BC: {loss_bc.item():.4e}, RMSE: {rmse.item():.4e}")
            
            # Save checkpoint occasionally
            if epoch % 10000 == 0 and epoch > 0:
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'loss': loss.item(),
                            'rmse': rmse.item()},
                           f'checkpoint_epoch_{epoch}.pt')
    
    # LBFGS Fine-Tuning
    def closure():
        optimizer_lbfgs.zero_grad()
        res = pde_residual(model, x_int, y_int)
        loss_pde_lbfgs = torch.mean(res**2)
        X_bnd_full = torch.cat([x_bnd, y_bnd], dim=1)
        loss_bc_lbfgs = torch.mean((model(X_bnd_full))**2)
        loss_lbfgs = loss_pde_lbfgs + lambda_bc * loss_bc_lbfgs
        loss_lbfgs.backward()
        return loss_lbfgs
    
    optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=1e-7, max_iter=500, history_size=50)
    optimizer_lbfgs.step(closure)
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict({key: val.to(device) for key, val in best_model_state.items()})
    
    # Final RMSE calculation over the whole dataset
    batch_size_rmse = 500
    num_batches = (x_all.shape[0] + batch_size_rmse - 1) // batch_size_rmse
    rmse_total = 0.0
    for i in range(num_batches):
        start_idx = i * batch_size_rmse
        end_idx = min((i + 1) * batch_size_rmse, x_all.shape[0])
        x_batch = x_all[start_idx:end_idx]
        y_batch = y_all[start_idx:end_idx]
        rmse, _, _ = compute_rmse(model, x_batch, y_batch)
        rmse_total += rmse.item() * (end_idx - start_idx)
    final_rmse = rmse_total / x_all.shape[0]
    print(f"Final RMSE: {final_rmse:.17e}")
    
    # Plot loss history
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label="Total Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.show()
    
    return model

# -----------------------
# Prediction and Saving Submission
# -----------------------
def predict_and_save(model, csv_file, submission_csv='submission.csv'):
    df = pd.read_csv(csv_file)
    batch_size = 500
    num_samples = len(df)
    num_batches = (num_samples + batch_size - 1) // batch_size
    u_pred_list = []
    model.eval()
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        batch_df = df.iloc[start_idx:end_idx]
        x = torch.tensor(batch_df['x'].values, dtype=torch.float64, device=device).unsqueeze(1)
        y = torch.tensor(batch_df['y'].values, dtype=torch.float64, device=device).unsqueeze(1)
        X = torch.cat([x, y], dim=1)
        with torch.no_grad():
            u_pred_batch = model(X).cpu().numpy().flatten()
            u_pred_list.append(u_pred_batch)
    u_pred = np.concatenate(u_pred_list)
    if 'ID' in df.columns:
        submission = pd.DataFrame({'ID': df['ID'], 'u': u_pred})
    else:
        submission = pd.DataFrame({'ID': range(1, len(u_pred)+1), 'u': u_pred})
    submission.to_csv(submission_csv, index=False)
    print(f"Submission saved to {submission_csv}")

# -----------------------
# Run a Quick Test to Verify Boundary Conditions
# -----------------------
def test_boundary_conditions(model):
    num_test = 100
    x_bnd, y_bnd = create_boundary_points(num_test)
    X_bnd = torch.cat([x_bnd, y_bnd], dim=1)
    with torch.no_grad():
        u_bnd = model(X_bnd)
    avg_bc_value = torch.mean(torch.abs(u_bnd)).item()
    max_bc_value = torch.max(torch.abs(u_bnd)).item()
    print("Boundary condition test:")
    print(f"  Average |u| on boundary: {avg_bc_value:.17e}")
    print(f"  Maximum |u| on boundary: {max_bc_value:.17e}")
    return avg_bc_value, max_bc_value

# -----------------------
# Main Execution
# -----------------------
if __name__ == "__main__":
    layers = [2, 500,500,500,500, 1]
    model = OptimizedPINN(layers=layers, activation=Swish()).to(device)
    
    csv_file = '/kaggle/input/ziq-sciml-challenge/test.csv'
    
    print("Testing initial boundary condition handling...")
    test_boundary_conditions(model)
    
    print("Starting training with adaptive sampling, ensemble strategy, and LBFGS fine-tuning...")
    model = train_model(model=model,
                        csv_file=csv_file,
                        epochs=10000,
                        batch_size=1000,
                        lambda_bc=100,
                        lr=7e-7,
                        adaptive_threshold=1e-3)
    
    print("Testing final boundary condition handling...")
    test_boundary_conditions(model)
    
    print("Generating predictions and saving submission...")
    predict_and_save(model, csv_file, submission_csv='submission.csv')



import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time
import math
import matplotlib.pyplot as plt

# -----------------------
# Device and Precision Setup
# -----------------------
torch.set_default_dtype(torch.float64)  # Use double precision
torch.set_printoptions(precision=17)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -----------------------
# Problem Parameters
# -----------------------
eps = 10e-4
b_x = 2.0
b_y = 3.0

# -----------------------
# Define the Heat Source Function f(x, y)
# -----------------------
def f_func(x, y):
    term1 = 2 * eps * (-x + torch.exp(2 * (x - 1) / eps))
    term2 = x * y**2
    term3 = 6 * x * y
    term4 = - x * torch.exp((3 * (y - 1)) / eps)
    term5 = - y**2 * torch.exp((2 * (x - 1)) / eps)
    term6 = 2 * y**2
    term7 = -6 * y * torch.exp((2 * (x - 1)) / eps)
    term8 = -2 * torch.exp((3 * (y - 1)) / eps)
    term9 = torch.exp((2 * x + 3 * y - 5) / eps)
    return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9

# -----------------------
# Define the Analytical Solution
# -----------------------
def analytical_solution(x, y):
    return x * (1 - x) * y * (1 - y) * torch.exp((2 * x + 3 * y - 5) / eps)

# -----------------------
# Compute RMSE between model predictions and analytical solution
# -----------------------
def compute_rmse(model, x, y):
    X = torch.cat([x, y], dim=1)
    with torch.no_grad():
        u_pred = model(X)
    u_true = analytical_solution(x, y)
    rmse = torch.sqrt(torch.mean((u_pred - u_true)**2))
    return rmse, u_true, u_pred

# -----------------------
# Simplified Activation Function (Swish)
# -----------------------
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# -----------------------
# Define the Optimized PINN Model
# -----------------------
class OptimizedPINN(nn.Module):
    def __init__(self, layers, activation=None):
        super(OptimizedPINN, self).__init__()
        self.activation = activation if activation is not None else Swish()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i+1])
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)
            
    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x

# -----------------------
# PDE Residual Computation with Auto-Differentiation
# -----------------------
def pde_residual(model, x, y):
    X = torch.cat([x, y], dim=1)
    X.requires_grad_(True)
    u = model(X)
    grad_u = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u),
                                 retain_graph=True, create_graph=True)[0]
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    u_xx = torch.autograd.grad(u_x, X, grad_outputs=torch.ones_like(u_x),
                               retain_graph=True, create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, X, grad_outputs=torch.ones_like(u_y),
                               retain_graph=True, create_graph=True)[0][:, 1:2]
    f_val = f_func(x, y)
    residual = -eps * (u_xx + u_yy) + b_x * u_x + b_y * u_y - f_val
    return residual

# -----------------------
# Explicitly Create Boundary Points
# -----------------------
def create_boundary_points(num_points_per_edge=400):
    t = torch.linspace(0, 1, num_points_per_edge, device=device)
    x_bottom = t
    y_bottom = torch.zeros_like(t)
    x_top = t
    y_top = torch.ones_like(t)
    x_left = torch.zeros_like(t)
    y_left = t
    x_right = torch.ones_like(t)
    y_right = t
    x_bnd = torch.cat([x_bottom, x_top, x_left, x_right])
    y_bnd = torch.cat([y_bottom, y_top, y_left, y_right])
    x_bnd = x_bnd.reshape(-1, 1)
    y_bnd = y_bnd.reshape(-1, 1)
    return x_bnd, y_bnd

# -----------------------
# Loading and Partitioning CSV Data
# -----------------------
def load_training_points(csv_file):
    df = pd.read_csv(csv_file)
    x = torch.tensor(df['x'].values, dtype=torch.float64, device=device).unsqueeze(1)
    y = torch.tensor(df['y'].values, dtype=torch.float64, device=device).unsqueeze(1)
    return x, y

def partition_points(x, y, tol=1e-5):
    boundary_mask = (torch.abs(x) < tol) | (torch.abs(x - 1) < tol) | \
                    (torch.abs(y) < tol) | (torch.abs(y - 1) < tol)
    interior_mask = ~boundary_mask
    x_int = x[interior_mask.squeeze()]
    y_int = y[interior_mask.squeeze()]
    x_bnd = x[boundary_mask.squeeze()]
    y_bnd = y[boundary_mask.squeeze()]
    return x_int, y_int, x_bnd, y_bnd

# -----------------------
# Generate Collocation Points (Interior Points)
# -----------------------
def generate_interior_points(num_points):
    x_int = torch.rand(num_points, 1, dtype=torch.float64, device=device)
    y_int = torch.rand(num_points, 1, dtype=torch.float64, device=device)
    return x_int, y_int

# -----------------------
# Batch Processing Utility
# -----------------------
def get_batch(x, y, batch_size):
    if x.shape[0] <= batch_size:
        return x, y
    idx = torch.randperm(x.shape[0])[:batch_size]
    return x[idx], y[idx]

# -----------------------
# Training Function with Adaptive Sampling and LBFGS Fine-Tuning
# -----------------------
def train_model(model, csv_file, epochs=5000, batch_size=1024, lambda_bc=100.0, lr=1e-6, adaptive_threshold=1e-2):
    # Load data from CSV
    x_all, y_all = load_training_points(csv_file)
    
    # Generate interior points
    x_int, y_int = generate_interior_points(150000)
    
    # Generate explicit boundary points
    x_bnd, y_bnd = create_boundary_points(1000)
    
    print(f"Training with {x_int.shape[0]} interior points and {x_bnd.shape[0]} boundary points")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.75)
    
    best_loss = float('inf')
    best_model_state = None
    
    loss_history = []
    bc_loss_history = []
    pde_loss_history = []
    
    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # ----- Adaptive Sampling -----
        # Get a uniform batch from interior points
        x_uniform, y_uniform = get_batch(x_int, y_int, batch_size // 2)
        
        # Compute residuals on all interior points for adaptive sampling
        # Compute with grad then detach the result
        residuals_full = pde_residual(model, x_int, y_int).detach()
        high_error_mask = (residuals_full.abs() > adaptive_threshold).squeeze()
        if high_error_mask.sum() > 0:
            x_high = x_int[high_error_mask]
            y_high = y_int[high_error_mask]
            x_adapt, y_adapt = get_batch(x_high, y_high, batch_size // 2)
            # Combine uniform and adaptive samples
            x_batch = torch.cat([x_uniform, x_adapt], dim=0)
            y_batch = torch.cat([y_uniform, y_adapt], dim=0)
        else:
            x_batch, y_batch = get_batch(x_int, y_int, batch_size)
        # ------------------------------
        
        # PDE loss on interior batch
        res = pde_residual(model, x_batch, y_batch)
        loss_pde = torch.mean(res**2)
        
        # Boundary loss on a batch of boundary points
        x_bnd_batch, y_bnd_batch = get_batch(x_bnd, y_bnd, batch_size)
        X_bnd = torch.cat([x_bnd_batch, y_bnd_batch], dim=1)
        u_bnd = model(X_bnd)
        u_bnd_true = torch.zeros_like(u_bnd)  # Dirichlet BC: u=0 on boundary
        loss_bc = torch.mean((u_bnd - u_bnd_true)**2)
        
        loss = loss_pde + lambda_bc * loss_bc
        
        loss_history.append(loss.item())
        bc_loss_history.append(loss_bc.item())
        pde_loss_history.append(loss_pde.item())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = {key: val.cpu() for key, val in model.state_dict().items()}
        
        if epoch % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - start_time
            test_idx = torch.randperm(x_all.shape[0])[:min(1000, x_all.shape[0])]
            x_test, y_test = x_all[test_idx], y_all[test_idx]
            rmse, _, _ = compute_rmse(model, x_test, y_test)
            print(f"Epoch {epoch}/{epochs} [{elapsed:.2f}s]")
            print(f"  Loss: {loss.item():.4e}, PDE: {loss_pde.item():.4e}, BC: {loss_bc.item():.4e}, LR: {current_lr:.4e},RMSE:{rmse.item():.6e}")
            
            # Save checkpoint occasionally
            if epoch % 10000 == 0 and epoch > 0:
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'loss': loss.item(),
                            'rmse': rmse.item()},
                           f'checkpoint_epoch_{epoch}.pt')
    
    # LBFGS Fine-Tuning
    def closure():
        optimizer_lbfgs.zero_grad()
        res = pde_residual(model, x_int, y_int)
        loss_pde_lbfgs = torch.mean(res**2)
        X_bnd_full = torch.cat([x_bnd, y_bnd], dim=1)
        loss_bc_lbfgs = torch.mean((model(X_bnd_full))**2)
        loss_lbfgs = loss_pde_lbfgs + lambda_bc * loss_bc_lbfgs
        loss_lbfgs.backward()
        return loss_lbfgs
    
    optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=1e-8, max_iter=5000, history_size=50)
    optimizer_lbfgs.step(closure)
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict({key: val.to(device) for key, val in best_model_state.items()})
    
    # Final RMSE calculation over the whole dataset
    batch_size_rmse = 500
    num_batches = (x_all.shape[0] + batch_size_rmse - 1) // batch_size_rmse
    rmse_total = 0.0
    for i in range(num_batches):
        start_idx = i * batch_size_rmse
        end_idx = min((i + 1) * batch_size_rmse, x_all.shape[0])
        x_batch = x_all[start_idx:end_idx]
        y_batch = y_all[start_idx:end_idx]
        rmse, _, _ = compute_rmse(model, x_batch, y_batch)
        rmse_total += rmse.item() * (end_idx - start_idx)
    final_rmse = rmse_total / x_all.shape[0]
    print(f"Final RMSE: {final_rmse:.17e}")
    
    # Plot loss history
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label="Total Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.show()
    
    return model

# -----------------------
# Prediction and Saving Submission
# -----------------------
def predict_and_save(model, csv_file, submission_csv='submission.csv'):
    df = pd.read_csv(csv_file)
    batch_size = 500
    num_samples = len(df)
    num_batches = (num_samples + batch_size - 1) // batch_size
    u_pred_list = []
    model.eval()
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        batch_df = df.iloc[start_idx:end_idx]
        x = torch.tensor(batch_df['x'].values, dtype=torch.float64, device=device).unsqueeze(1)
        y = torch.tensor(batch_df['y'].values, dtype=torch.float64, device=device).unsqueeze(1)
        X = torch.cat([x, y], dim=1)
        with torch.no_grad():
            u_pred_batch = model(X).cpu().numpy().flatten()
            u_pred_list.append(u_pred_batch)
    u_pred = np.concatenate(u_pred_list)
    if 'ID' in df.columns:
        submission = pd.DataFrame({'ID': df['ID'], 'u': u_pred})
    else:
        submission = pd.DataFrame({'ID': range(1, len(u_pred)+1), 'u': u_pred})
    submission.to_csv(submission_csv, index=False, float_format='%.17f')
    print(f"Submission saved to {submission_csv}")

# -----------------------
# Run a Quick Test to Verify Boundary Conditions
# -----------------------
def test_boundary_conditions(model):
    num_test = 100
    x_bnd, y_bnd = create_boundary_points(num_test)
    X_bnd = torch.cat([x_bnd, y_bnd], dim=1)
    with torch.no_grad():
        u_bnd = model(X_bnd)
    avg_bc_value = torch.mean(torch.abs(u_bnd)).item()
    max_bc_value = torch.max(torch.abs(u_bnd)).item()
    print("Boundary condition test:")
    print(f"  Average |u| on boundary: {avg_bc_value:.17e}")
    print(f"  Maximum |u| on boundary: {max_bc_value:.17e}")
    return avg_bc_value, max_bc_value

# -----------------------
# Main Execution
# -----------------------
if __name__ == "__main__":
    layers = [2, 250,250, 1]
    model = OptimizedPINN(layers=layers, activation=nn.Tanh()).to(device)
    
    csv_file = '/kaggle/input/ziq-sciml-challenge/test.csv'
    
    print("Testing initial boundary condition handling...")
    test_boundary_conditions(model)
    
    print("Starting training with adaptive sampling, ensemble strategy, and LBFGS fine-tuning...")
    model = train_model(model=model,
                        csv_file=csv_file,
                        epochs=2500,
                        batch_size=10000,
                        lambda_bc=0.5,
                        lr=1e-5)
    
    print("Testing final boundary condition handling...")
    test_boundary_conditions(model)
    
    print("Generating predictions and saving submission...")
    predict_and_save(model, csv_file, submission_csv='submission.csv')





