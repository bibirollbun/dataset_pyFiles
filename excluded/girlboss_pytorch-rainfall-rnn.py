# Core libraries
import numpy as np  # Numerical operations
import pandas as pd  # Data manipulation
import matplotlib.pyplot as plt  # Data visualization
import seaborn as sns  # Advanced visualizations

# Machine learning (if needed)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier  # Example model
from sklearn.metrics import accuracy_score

# Ignore warnings (optional)
import warnings
warnings.filterwarnings("ignore")

# Display settings
pd.set_option("display.max_columns", None)  # Show all columns
plt.style.use("ggplot")  # Improve visualization aesthetics

# Interactive visualizations (optional)
%matplotlib inline



train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


def dataframe_summary(df):
    
    print("ðŸ”¹ DataFrame Overview ðŸ”¹\n")

    print(f"ðŸ“Œ Shape: {df.shape}\n")

    print(f"ðŸ“Œ Columns Names: {df.columns}\n")

    print("ðŸ“Œ Column Information:\n")
    print(df.info(), "\n")

    print("ðŸ“Œ Missing Values:\n")
    print(df.isnull().sum(), "\n")

    print("ðŸ“Œ Summary Statistics (Numerical Columns):\n")
    print(df.describe(), "\n")

    # Checking if categorical columns exist before calling describe
    cat_cols = df.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        print("ðŸ“Œ Summary Statistics (Categorical Columns):\n")
        print(df.describe(include=["object"]), "\n")
    else:
        print("ðŸ“Œ No categorical columns found.\n")

    print("ðŸ“Œ Unique Values per Column:\n")
    print(df.nunique(), "\n")



dataframe_summary(train_df)


dataframe_summary(test_df)


def replace_nan_with_median(df):
    """
    Replaces NaN values in each column with the column's median.

    Parameters:
    df (pd.DataFrame): The DataFrame in which NaNs will be replaced.

    Returns:
    pd.DataFrame: A new DataFrame with NaNs replaced by median values.
    """
    return df.apply(lambda col: col.fillna(col.median()) if col.dtype in ['int64', 'float64'] else col)


train_df = replace_nan_with_median(train_df)  # although train data don't have nan values
test_df = replace_nan_with_median(test_df)


selected_columns = [col for col in train_df.columns if col not in ["id", "rainfall"]]
X_train = train_df[selected_columns]


def select_features(df, exclude_columns):
    selected_columns = [col for col in df.columns if col not in exclude_columns]
    return df[selected_columns]


X_train = select_features(train_df , ["id" , "rainfall"])
X_test =  select_features(test_df , ["id" , "rainfall"])


X_train.shape


from sklearn.preprocessing import StandardScaler

def standardize_train_test(X_train, X_test):

    scaler = StandardScaler()
    
    # Fitting on training data, transform both train & test
    X_train_scaled = X_train.copy()
    X_train_scaled.iloc[:, :] = scaler.fit_transform(X_train)

    X_test_scaled = X_test.copy()
    X_test_scaled.iloc[:, :] = scaler.transform(X_test)  # Use same scaler (no fit)
    
    return X_train_scaled, X_test_scaled
    

X_train_scaled, X_test_scaled = standardize_train_test(X_train, X_test)


y_train = train_df["rainfall"]
y_train.head(3)


from sklearn.model_selection import train_test_split

# Spliting X_train_scaled and y_train into train and validation sets
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Checking the shapes
print("X_train_final shape:", X_train_final.shape)
print("X_val shape:", X_val.shape)
print("y_train_final shape:", y_train_final.shape)
print("y_val shape:", y_val.shape)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import catboost as cb

# Function to train and evaluate models
def train_and_evaluate(X_train, y_train, X_val, y_val):
    models = {
        "Logistic Regression": LogisticRegression(solver='liblinear', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "CatBoost": cb.CatBoostClassifier(verbose=0, random_state=42)  # Silent mode for clean output
    }
    
    results = {}
    
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Evaluate performance
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc_roc = roc_auc_score(y_val, y_pred_proba)
        
        results[name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "AUC-ROC": auc_roc
        }
        
        print(f"\nðŸ”¹ Model: {name}")
        print(classification_report(y_val, y_pred))
        print(f"AUC-ROC Score: {auc_roc:.4f}")
    
    return pd.DataFrame(results)

# ðŸš€ Train & evaluate models
results_df = train_and_evaluate(X_train_final, y_train_final, X_val, y_val)

# ðŸ“Š Visualize performance
plt.figure(figsize=(12,6))
sns.heatmap(results_df, annot=True, cmap="Blues", fmt=".4f")
plt.title("Model Performance Comparison")
plt.show()


!pip install lightning --q


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------------------------------------- #
# 1. Split Data (train_df is already scaled before splitting)
# ----------------------------------------------------------- #
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_scaled,  # <-- Already scaled data
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

# âœ… Minimal fix: if train_df was already scaled, just rename X_val
X_val_scaled = X_val  # This ensures X_val_scaled is defined and used below

print("X_train_final shape:", X_train_final.shape)
print("X_val shape:", X_val.shape)
print("y_train_final shape:", y_train_final.shape)
print("y_val shape:", y_val.shape)

# ----------------------------------------------------------- #
# 2. Custom Dataset Class
# ----------------------------------------------------------- #
class RainfallDataset(Dataset):
    def __init__(self, X, y=None, train=True):
        self.X = torch.tensor(np.array(X), dtype=torch.float32)  # Convert to tensor
        self.train = train
        if train:
            self.y = torch.tensor(np.array(y), dtype=torch.float32)  # Convert labels to tensor

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.train:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]

# ----------------------------------------------------------- #
# 3. Define Neural Network Model
# ----------------------------------------------------------- #
class RainfallModel(L.LightningModule):
    def __init__(self, input_dim):
        super(RainfallModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Since we're predicting rainfall probability
        )
        self.loss_fn = nn.BCELoss()  # Binary Cross-Entropy Loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X).squeeze()
        loss = self.loss_fn(y_hat, y)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

# ----------------------------------------------------------- #
# 4. Load & Prepare Data for Training
# ----------------------------------------------------------- #
train_dataset = RainfallDataset(X_train_final, y_train_final)
val_dataset = RainfallDataset(X_val_scaled, y_val)
# For test set, assume you have already created X_test_scaled
test_dataset = RainfallDataset(X_test_scaled, train=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ----------------------------------------------------------- #
# 5. Model Training
# ----------------------------------------------------------- #
input_dim = X_train_final.shape[1]  # Number of features
model = RainfallModel(input_dim)

trainer = L.Trainer(max_epochs=20, accelerator="auto")
trainer.fit(model, train_loader, val_loader)

# ----------------------------------------------------------- #
# 6. Evaluation
# ----------------------------------------------------------- #
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        y_hat = model(X_batch).squeeze().cpu().numpy()
        y_pred.extend(y_hat)
        y_true.extend(y_batch.cpu().numpy())

# Convert predictions to binary (threshold = 0.5)
y_pred_binary = (np.array(y_pred) > 0.5).astype(int)

accuracy = accuracy_score(y_true, y_pred_binary)
print(f"âœ… Model Accuracy: {accuracy:.4f}")
print(classification_report(y_true, y_pred_binary))

# ----------------------------------------------------------- #
# 7. Generate Submission File
# ----------------------------------------------------------- #
model.eval()
test_predictions = []

with torch.no_grad():
    for X_batch in test_loader:
        preds = model(X_batch).squeeze().cpu().numpy()
        test_predictions.extend(preds)

# Convert probabilities to dataframe
submission = pd.DataFrame({
    "id": test_df["id"],  # Ensure `test_df` has the "id" column
    "rainfall": test_predictions
})

submission.to_csv("submission.csv", index=False)
print("ðŸŽ¯ Submission file saved as 'submission.csv'!")



import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------------------------------------- #
# 1. NO RANDOM SPLIT!! But temporal [TODO]
# ----------------------------------------------------------- #
split_idx = int(0.8*len(X_train_scaled))
X_train_final = X_train_scaled[:split_idx]
X_val = X_train_scaled[split_idx:]
y_train_final = y_train[:split_idx]
y_val = y_train[split_idx:]

X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_scaled,  # <-- Already scaled data
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

# âœ… Minimal fix: if train_df was already scaled, just rename X_val
X_val_scaled = X_val  # This ensures X_val_scaled is defined and used below

print("X_train_final shape:", X_train_final.shape)
print("X_val shape:", X_val.shape)
print("y_train_final shape:", y_train_final.shape)
print("y_val shape:", y_val.shape)

# ----------------------------------------------------------- #
# 2. Custom Dataset Class
# ----------------------------------------------------------- #
class RainfallSequenceDataset(Dataset):
    def __init__(self, data, labels=None, seq_length=10, train=True):
        """
        data: NumPy array or DataFrame of shape (num_samples, num_features)
        labels: corresponding labels (if available)
        seq_length: number of timesteps per sample
        train: whether labels are available (True for training/validation)
        """
        self.data = torch.tensor(np.array(data), dtype=torch.float32)
        self.train = train
        self.seq_length = seq_length
        
        if self.train:
            # Ensure labels are in the same order and format as data.
            self.labels = torch.tensor(np.array(labels), dtype=torch.float32)

    def __len__(self):
        # Each sample is a sequence, so we lose (seq_length) rows from the end.
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        # Return a sequence of features
        x_seq = self.data[idx: idx + self.seq_length]
        
        if self.train:
            # For example, predict rainfall on the day right after the sequence
            y = self.labels[idx + self.seq_length]
            return x_seq, y
        else:
            return x_seq

# ----------------------------------------------------------- #
# 3. Define Recurrent Neural Network Model
# ----------------------------------------------------------- #
class RainfallRNN(L.LightningModule):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, seq_length=10):
        super(RainfallRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_length = seq_length
        
        # The RNN (here LSTM) processes the sequence.
        # Set batch_first=True so input shape is (batch, seq_length, features)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # A fully connected layer to map the hidden state to a single output
        self.fc = nn.Linear(hidden_dim, 1)
        
        # Binary Cross-Entropy for a binary classification problem
        self.loss_fn = nn.BCELoss()

    def forward(self, x):
        # x shape: (batch, seq_length, input_dim)
        # LSTM returns output for all timesteps and hidden state
        lstm_out, _ = self.lstm(x)
        # Use the last timestep's output as summary for prediction
        final_out = lstm_out[:, -1, :]
        out = self.fc(final_out)
        return torch.sigmoid(out)

    def training_step(self, batch, batch_idx):
        x_seq, y = batch
        y_hat = self(x_seq).squeeze()
        loss = self.loss_fn(y_hat, y)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

# ----------------------------------------------------------- #
# 4. Load & Prepare Data for Training
# ----------------------------------------------------------- #
train_dataset = RainfallSequenceDataset(X_train_final, y_train_final)
val_dataset = RainfallSequenceDataset(X_val_scaled, y_val)
# For test set, assume you have already created X_test_scaled
seq_length = 10
# Assume `pad_value` is a reasonable default or you have historical data
pad_value = np.zeros((seq_length, X_test_scaled.shape[1]))
X_test_padded = np.concatenate([pad_value, X_test_scaled], axis=0)
test_dataset = RainfallSequenceDataset(X_test_padded, train=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ----------------------------------------------------------- #
# 5. Model Training
# ----------------------------------------------------------- #
input_dim = X_train_final.shape[1]  # Number of features
model = RainfallRNN(input_dim)

trainer = L.Trainer(max_epochs=20, accelerator="auto")
trainer.fit(model, train_loader, val_loader)

# ----------------------------------------------------------- #
# 6. Evaluation
# ----------------------------------------------------------- #
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        y_hat = model(X_batch).squeeze().cpu().numpy()
        y_pred.extend(y_hat)
        y_true.extend(y_batch.cpu().numpy())

# Convert predictions to binary (threshold = 0.5)
y_pred_binary = (np.array(y_pred) > 0.5).astype(int)

accuracy = accuracy_score(y_true, y_pred_binary)
print(f"âœ… Model Accuracy: {accuracy:.4f}")
print(classification_report(y_true, y_pred_binary))

# ----------------------------------------------------------- #
# 7. Generate Submission File
# ----------------------------------------------------------- #
model.eval()
test_predictions = []



with torch.no_grad():
    for X_batch in test_loader:
        preds = model(X_batch).squeeze().cpu().numpy()
        test_predictions.extend(preds)

# Convert probabilities to dataframe
submission = pd.DataFrame({
    "id": test_df["id"],  # Ensure `test_df` has the "id" column
    "rainfall": test_predictions
})

submission.to_csv("submission.csv", index=False)
print("ðŸŽ¯ Submission file saved as 'submission.csv'!")











