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


import pandas as pd
import numpy as np
from collections import OrderedDict
import os#interact with operation system
from sklearn.model_selection import KFold
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from torch.utils.data import DataLoader, Dataset,TensorDataset

import warnings#avoid some negligible errors
#The filterwarnings () method is used to set warning filters, which can control the output method and level of warning information.
warnings.filterwarnings('ignore')
import random#provide some function to generate random_seed.
#set random seed,to make sure model can be recurrented.
def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=42)


train=pd.read_csv("/kaggle/input/geology-forecast-challenge-open/data/train.csv")
test=pd.read_csv("/kaggle/input/geology-forecast-challenge-open/data/test.csv")
sub=pd.read_csv('/kaggle/input/geology-forecast-challenge-open/data/sample_submission.csv')
sub.head()


# --- Delete top 100 samples with highest number of NaNs from train ---
# Calculate the number of NaNs for each row in the training data
nan_counts_per_row = train.isnull().sum(axis=1)

# Sort rows by NaN count in descending order to find the highest
sorted_nan_counts = nan_counts_per_row.sort_values(ascending=False)

# Get the indices of the top 100 samples with the most NaNs
top_100_nan_indices = sorted_nan_counts.head(100).index

# Print debug information about the rows being dropped
print(f"Original train DataFrame shape: {train.shape}")
print(f"Number of rows to drop (top 100 NaNs): {len(top_100_nan_indices)}")
print(f"NaN counts for top 5 rows to be dropped: \n{sorted_nan_counts.head(5)}")

# Drop these rows from the train DataFrame
train = train.drop(index=top_100_nan_indices)

# Print debug information about the new shape of the train DataFrame
print(f"Train DataFrame shape after dropping top 100 NaN rows: {train.shape}")

# --- Impute missing values with the mean of each column ---
# Calculate means from the training data (for numeric columns only).
# This is crucial to prevent data leakage from the test set into the training process.
train_means = train.mean(numeric_only=True)

# Apply mean imputation to both train and test sets.
# Any column in 'train' that has NaNs will be filled with its mean from 'train_means'.
# Any column in 'test' that has NaNs will be filled with its mean from 'train_means' (if the column exists in train).
train = train.fillna(train_means)
test = test.fillna(train_means)

# Check for any remaining NaNs after imputation (e.g., if a column was all NaNs).
# If a column was entirely NaNs, its mean would be NaN, and fillna(train_means) wouldn't fill it.
# In such rare cases, fill remaining NaNs with 0 as a final fallback.
# However, this is usually not necessary if your columns are well-populated.
if train.isnull().any().any():
    print("Warning: NaNs still present in train after mean imputation. Filling remaining with 0.")
    train = train.fillna(0)
if test.isnull().any().any():
    print("Warning: NaNs still present in test after mean imputation. Filling remaining with 0.")
    test = test.fillna(0)


print(train.shape, test.shape)


FEATURES=[c for c in test.columns if c!='geology_id']
TARGETS=[c for c in sub.columns if c!='geology_id']

train = train.reset_index(drop=True)
train_sub = train[['geology_id'] + TARGETS].copy()
solution = train[['geology_id'] + TARGETS].copy()
train.head()


import torch
import torch.nn as nn
import numpy as np

class TransformerUNet1D(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_layers, d_model, d_ff, dropout, num_realizations):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        self.num_realizations = num_realizations

        # Initial projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Encoder Convs (downsampling)
        self.encoder_conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.encoder_conv2 = nn.Conv1d(d_model, d_model * 2, kernel_size=3, stride=2, padding=1)

        # Transformer at bottleneck
        self.positional_encoding = self._generate_positional_encoding(max_len=300, d_model=d_model * 2)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model * 2, nhead=num_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder Convs (upsampling)
        self.decoder_conv1 = nn.ConvTranspose1d(d_model * 2, d_model, kernel_size=4, stride=2, padding=1)
        self.decoder_conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)

        # Final output
        self.output_proj = nn.Linear(d_model, output_dim * num_realizations)

    def _generate_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, src):
        batch_size, seq_len = src.size(0), src.size(1)

        # Input projection
        x = self.input_proj(src.unsqueeze(-1))  # (B, S, d_model)
        x = x.permute(0, 2, 1)  # (B, d_model, S)

        # Encoder
        x1 = torch.relu(self.encoder_conv1(x))     # (B, d_model, S)
        x2 = torch.relu(self.encoder_conv2(x1))     # (B, d_model*2, S/2)

        # Transformer bottleneck
        x2 = x2.permute(0, 2, 1)                    # (B, S/2, d_model*2)
        pe = self.positional_encoding[:, :x2.size(1), :].to(x2.device)
        x2 = x2 + pe
        x2 = self.transformer(x2)                   # (B, S/2, d_model*2)
        x2 = x2.permute(0, 2, 1)                    # (B, d_model*2, S/2)

        # Decoder
        x = self.decoder_conv1(x2)                  # (B, d_model, S)
        x = x + x1                                   # Skip connection
        x = torch.relu(self.decoder_conv2(x))       # (B, d_model, S)

        # Output projection
        x = x.permute(0, 2, 1)                       # (B, S, d_model)
        out = self.output_proj(x)                   # (B, S, output_dim * num_realizations)
        out = out[:, -1, :]                         # Take the last time step
        return out.view(batch_size, self.num_realizations, self.output_dim)



folds = 5
epochs = 50
loss_fn = nn.HuberLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device:{device}")


from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import pandas as pd

# Assume these variables are defined: train, test, FEATURES, TARGETS, folds, epochs, loss_fn, device

scaler = StandardScaler()
scaler2 = StandardScaler()
kf = KFold(n_splits=folds, random_state=42, shuffle=True)
X_num, y = np.log(31 + train[FEATURES].values), train[TARGETS].values

scaler.fit(X_num)
X_num = scaler.transform(X_num)
X_num_test = np.log(31 + test[FEATURES].values)
X_num_test = scaler.transform(X_num_test)

scaler2.fit(y)
y = scaler2.transform(y)

# Create test dataset and loader
test_dataset = TensorDataset(torch.tensor(X_num_test, dtype=torch.float32).to(device))
test_dl = DataLoader(test_dataset, batch_size=1024, shuffle=False)
test_tabm = np.zeros((folds, len(test), len(TARGETS)))

# Track losses
train_losses = []
val_losses = []

# OOF prediction container
oof_preds = np.zeros((len(train), len(TARGETS)))

for i, (train_index, val_index) in enumerate(kf.split(train[FEATURES])):
    X_num_train, X_num_val = X_num[train_index], X_num[val_index]
    y_train, y_val = y[train_index], y[val_index]

    train_dataset = TensorDataset(
        torch.tensor(X_num_train, dtype=torch.float32).to(device),
        torch.tensor(y_train, dtype=torch.float32).to(device),
    )
    valid_dataset = TensorDataset(
        torch.tensor(X_num_val, dtype=torch.float32).to(device),
        torch.tensor(y_val, dtype=torch.float32).to(device),
    )

    train_dl = DataLoader(train_dataset, batch_size=128, shuffle=True)
    valid_dl = DataLoader(valid_dataset, batch_size=128, shuffle=False)

    model = TransformerUNet1D(
        input_dim=1,
        output_dim=300,
        num_realizations=10,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=512,
        dropout=0.1
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-2,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    print(f"< Fold {i+1} Training >")
    fold_train_losses = []
    fold_val_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        with tqdm(train_dl, total=len(train_dl), leave=True) as phar:
            for train_tensor in phar:
                optimizer.zero_grad()
                X_num_train, y_train = train_tensor
                output = model(X_num_train)
                reshaped_y_train = y_train.view(y_train.size(0), 10, 300)
                loss = loss_fn(
                    (output - reshaped_y_train.mean(dim=0)) / reshaped_y_train.std(dim=0),
                    (reshaped_y_train - reshaped_y_train.mean(dim=0)) / reshaped_y_train.std(dim=0),
                )
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item() * X_num_train.size(0)
                phar.set_postfix(OrderedDict(epoch=f"{epoch+1}/{epochs}", loss=f"{loss.item():.6f}"))
        avg_train_loss = epoch_train_loss / len(train_dataset)
        fold_train_losses.append(avg_train_loss)

        # Validation Loss
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for valid_tensor in valid_dl:
                X_num_val, y_val = valid_tensor
                output_val = model(X_num_val)
                reshaped_y_val = y_val.view(y_val.size(0), 10, 300)
                loss_val = loss_fn(
                    (output_val - reshaped_y_val.mean(dim=0)) / reshaped_y_val.std(dim=0),
                    (reshaped_y_val - reshaped_y_val.mean(dim=0)) / reshaped_y_val.std(dim=0),
                )
                epoch_val_loss += loss_val.item() * X_num_val.size(0)
        avg_valid_loss = epoch_val_loss / len(valid_dataset)
        fold_val_losses.append(avg_valid_loss)

        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_valid_loss:.6f}")
        scheduler.step()

    # OOF predictions
    model.eval()
    valid_preds = []
    for valid_tensor in valid_dl:
        X_num_val, _ = valid_tensor
        with torch.no_grad():
            output = model(X_num_val)
        valid_preds.append(output.cpu().numpy())

    valid_preds = np.concatenate(valid_preds)
    valid_preds = valid_preds.reshape(valid_preds.shape[0], -1)
    oof_preds[val_index] = valid_preds  # Store OOF predictions

    train_losses.append(fold_train_losses)
    val_losses.append(fold_val_losses)

    # Test set prediction for this fold
    print("< model prediction >")
    model.eval()
    test_preds = []
    with torch.no_grad():
        for test_tensor in test_dl:
            X_num_test = test_tensor[0]
            output = model(X_num_test)
            test_preds.append(output.cpu().numpy())

    all_test_preds = np.concatenate(test_preds, axis=0)
    reshaped_test_preds = all_test_preds.reshape(all_test_preds.shape[0], -1)
    test_tabm[i] = reshaped_test_preds

# Plot training and validation loss for each fold
for i in range(folds):
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, epochs + 1)
    plt.plot(epochs_range, train_losses[i], label=f'Fold {i+1} Training Loss')
    plt.plot(epochs_range, val_losses[i], label=f'Fold {i+1} Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss - Fold {i+1}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot average losses
avg_train_loss_across_folds = np.mean(train_losses, axis=0)
avg_val_loss_across_folds = np.mean(val_losses, axis=0)

plt.figure(figsize=(10, 6))
epochs_range = range(1, epochs + 1)
plt.plot(epochs_range, avg_train_loss_across_folds, label='Average Training Loss')
plt.plot(epochs_range, avg_val_loss_across_folds, label='Average Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Average Training and Validation Loss Across All Folds')
plt.legend()
plt.grid(True)
plt.show()

print(f"\nFinal Average Training Loss across all folds: {avg_train_loss_across_folds[-1]:.6f}")
print(f"Final Average Validation Loss across all folds: {avg_val_loss_across_folds[-1]:.6f}")

# Save OOF predictions
oof_preds_unscaled = scaler2.inverse_transform(oof_preds)
oof_df = pd.DataFrame(oof_preds_unscaled, columns=TARGETS)
oof_df.to_csv("transformer_oof_predictions.csv", index=False)

# Save test predictions (averaged across folds)
final_test_preds = scaler2.inverse_transform(np.mean(test_tabm, axis=0))
submission_df = pd.DataFrame(final_test_preds, columns=TARGETS)
submission_df.to_csv("submission.csv", index=False)



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import pandas.api.types
import numpy as np


class ParticipantVisibleError(Exception):
    # If you want an error message to be shown to participants, you must raise the error as a ParticipantVisibleError
    # All other errors will only be shown to the competition host. This helps prevent unintentional leakage of solution data.
    pass

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    NLL metric for the Geology Forecast Challenge
    Detailed desctiption is the competition file
    www.kaggle.com/competitions/geology-forecast-challenge-open/overview/evaluation
    """
    # TODO: You likely want to delete the row ID column, which Kaggle's system uses to align
    # the solution and submission before passing these dataframes to score().
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    # TODO: adapt or remove this check depending on what data types make sense for your metric
    for col in submission.columns:
        if not pd.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')

    # TODO: add additional checks appropriate for your metric. Common things to check include:
    # Non-negative inputs, the wrong number of columns in the submission, values that should be restricted to a range, etc.
    # The more errors you tag as participant visible, the easier it will be for participants to debug their submissions.

    # calculate your metric here. For the template, we'll just calculate a simple mean absolute error over all non-id columns.
    # By default, the metric doesn't have information about what columns to expect. You probably want to add the names of other
    # columns as arguments to score, like row_id_column_name, if you won't be processing all columns the same way.

    NEGATIVE_PART = -299
    LARGEST_CHUNK = 600
    SMALLEST_CHUNK = 350
    TOTAL_REALIZATIONS = 10
    INFLATION_SIGMA = 600

    # exponent_1 = 0.9879100831042902
    # mult_1 = 0.7557775492195064
    #
    # exponent_2 = 1.9758201662085804
    # mult_2 = 0.5711997039042433

    sigma_2 = np.ones((LARGEST_CHUNK + NEGATIVE_PART - 1))
    from_ranges = [1, 61, 245]
    to_ranges_excl = [61, 245, 301]
    log_slopes = [1.0406028049510443, 0.0, 7.835345062351012]
    log_offsets = [-6.430669850650689, -2.1617411566043896, -45.24876794412965]

    for growth_mode in range(len(from_ranges)):
        for i in range(from_ranges[growth_mode], to_ranges_excl[growth_mode]):
            sigma_2[i - 1] = np.exp(np.log(i) * log_slopes[growth_mode] + log_offsets[growth_mode])

    # trying to inflate sigma not to get errors
    sigma_2 *= INFLATION_SIGMA
    # plt.plot(sigma_2)
    # print(sigma_2)
    # plt.savefig('sigma_2.png', bbox_inches='tight')
    # plt.show()

    # sigma_2 = np.array(list(mult_2*np.power(i, exponent_2) for i in range(1, LARGEST_CHUNK + NEGATIVE_PART)))

    # Compute the inverse of the diagonal covariance matrix (element-wise inverse)
    cov_matrix_inv_diag = 1. / sigma_2  # Inverse of the diagonal elements

    # formula for multivariate gaussian
    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    # https://en.wikipedia.org/wiki/Covariance_matrix#Covariance_matrix_as_a_parameter_of_a_distribution
    # formula for mixture density
    # https://en.wikipedia.org/wiki/Mixture_model#Multivariate_Gaussian_mixture_model

    num_rows = solution.shape[0]

    # TODO add probabilities
    # now we use all probs equal to:
    p = 1. / TOTAL_REALIZATIONS
    ps = np.full((num_rows, TOTAL_REALIZATIONS), p)

    log_p = -np.log(TOTAL_REALIZATIONS)
    inner_product_matrix = np.zeros((num_rows, TOTAL_REALIZATIONS))

    # ps = np.full((num_rows, TOTAL_REALIZATIONS), p)

    # exp_misfit = np.zeros((num_rows, TOTAL_REALIZATIONS))

    num_columns = LARGEST_CHUNK + NEGATIVE_PART - 1

    # collecting solution and submission in numpy arrays
    full_submission = np.zeros((num_rows, TOTAL_REALIZATIONS, num_columns))
    full_solution = np.zeros((num_rows, TOTAL_REALIZATIONS, num_columns))

    # Iterating through column names
    for k in range(TOTAL_REALIZATIONS):
        # compute misfits in individual columns
        # Create an array filled with zeros
        misfit = np.zeros((num_rows, num_columns))
        for i in range(num_columns):
            if k == 0:
                column_name = str(i + 1)
            else:
                column_name = f"r_{k}_pos_{i + 1}"
            # this needs to be different cov matrix
            misfit[:, i] = solution[column_name].values - submission[column_name].values
            full_submission[:, k, i] = submission[column_name].values
            full_solution[:, k, i] = solution[column_name].values
        misfit_scaled = misfit * cov_matrix_inv_diag
        inner_product = np.sum(misfit_scaled * misfit, axis=1)
        # exp_misfit_cur = np.exp(inner_product)
        # exp_misfit[:, k] = exp_misfit_cur
        inner_product_matrix[:, k] = inner_product #+ log_p

    # Find normalization constant
    # max_vals = 100*np.ones_like(inner_product_matrix[:,0]) # np.zeros_like(inner_product_matrix[:,0]) # np.max(inner_product_matrix, axis=1)
    max_vals = np.max(inner_product_matrix, axis=1)
    # add and subtract max_vals to avoid numerical instability
    product = np.exp(inner_product_matrix + log_p - max_vals[:, None])
    log_sum_exp = max_vals + np.log(np.sum(product, axis=1))

    nll = -log_sum_exp
    computed_score = nll.mean()

    # Check for infinite scores
    if np.isinf(computed_score):
        raise ParticipantVisibleError(f"Your score is {computed_score}, which means there is room for improvement.")

    return computed_score
print(f"final CV:{score(solution,train_sub,'geology_id')}")


sub[TARGETS]=scaler2.inverse_transform(test_tabm.mean(axis=0))

sub.to_csv("submission.csv",index=None)
sub.head()


import base64

file_path = "/kaggle/working/submission.csv"
with open(file_path, "rb") as f:
    data = f.read()

b64 = base64.b64encode(data).decode()
href = f'<a href="data:file/csv;base64,{b64}" download="submission.csv">Download submission.csv</a>'
from IPython.display import display, HTML
display(HTML(href))

