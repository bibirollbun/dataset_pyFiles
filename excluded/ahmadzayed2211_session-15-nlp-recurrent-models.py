import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import pandas as pd
import string
import re
import unicodedata
from collections import Counter

# For plotting and progress
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Load the text data
with open('/kaggle/input/shakespeare-plays/alllines.txt', 'r') as f:
    text = f.read()

# Create character vocabulary
chars = tuple(sorted(set(text)))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}
vocab_size = len(chars)

print(f"The text has {len(text)} characters.")
print(f"The vocabulary has {vocab_size} unique characters.")

# Encode the entire text
encoded_text = np.array([char2int[ch] for ch in text])


def create_sequences(text_array, seq_length):
    x, y = [], []
    for i in range(len(text_array) - seq_length):
        # Input sequence
        x.append(text_array[i : i+seq_length])
        # Target sequence (shifted by one)
        y.append(text_array[i+1 : i+seq_length+1])
    return torch.LongTensor(x), torch.LongTensor(y)

# Hyperparameters for data creation
SEQ_LENGTH = 100
BATCH_SIZE = 128

# Create the sequences and data loader
# We use a smaller subset of the data for faster training in this demo
subset_text = encoded_text[:500000]
inputs, targets = create_sequences(subset_text, SEQ_LENGTH)
dataset = TensorDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

print("Input shape:", inputs.shape)
print("Target shape:", targets.shape)


class CharRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=1):
        super(CharRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        out, hidden = self.rnn(embedded, hidden)
        # We want to predict a character for each char in the input sequence
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        # Initialize hidden state with zeros
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)


# Model Hyperparameters
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
N_LAYERS = 2
LR = 0.001
EPOCHS = 3 # Keep low for demo

# --- Training Function ---
def train_model(model, dataloader, epochs, lr):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        hidden = model.init_hidden(BATCH_SIZE)
        model.train()
        
        for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Detach hidden state to prevent backprop through entire history
            if isinstance(hidden, tuple): # For LSTM
                hidden = tuple([h.detach() for h in hidden])
            else: # For RNN/GRU
                hidden = hidden.detach()

            optimizer.zero_grad()
            
            output, hidden = model(inputs, hidden)
            
            # Reshape targets to match output
            loss = criterion(output, targets.view(-1))
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")

# --- Generation Function ---
def generate_text(model, start_string, size=100):
    model.eval()
    chars_to_gen = [ch for ch in start_string]
    # Convert start string to integers
    input_seq = torch.LongTensor([char2int[ch] for ch in start_string]).unsqueeze(0).to(device)
    
    # Initialize hidden state
    hidden = model.init_hidden(1)

    # "Warm up" the model with the start string
    for i in range(len(start_string)):
        output, hidden = model(input_seq[:, i].unsqueeze(1), hidden)

    # Predict the next characters
    for _ in range(size):
        output = torch.softmax(output.squeeze(), dim=0)
        # Get the character with the highest probability
        char_index = torch.multinomial(output, 1).item()
        
        chars_to_gen.append(int2char[char_index])
        
        # Use the predicted character as the next input
        input_seq = torch.LongTensor([[char_index]]).to(device)
        output, hidden = model(input_seq, hidden)
        
    return "".join(chars_to_gen)


# Instantiate and train the RNN
rnn_model = CharRNN(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS)
print("--- Training RNN Model ---")
train_model(rnn_model, dataloader, EPOCHS, LR)

# Generate text with the trained RNN
print("\n--- RNN Generated Text ---")
print(generate_text(rnn_model, start_string="KING:", size=200))


class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=1):
        super(CharLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # The key difference is here: nn.LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        out, hidden = self.lstm(embedded, hidden)
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        # LSTM hidden state is a TUPLE (hidden_state, cell_state)
        hidden_state = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        cell_state = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        return (hidden_state, cell_state)

# Instantiate and train the LSTM
lstm_model = CharLSTM(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS)
print("\n--- Training LSTM Model ---")
train_model(lstm_model, dataloader, EPOCHS, LR)

# Generate text with the trained LSTM
print("\n--- LSTM Generated Text ---")
print(generate_text(lstm_model, start_string="JULIET:", size=200))


class CharGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=1):
        super(CharGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # The key difference is here: nn.GRU
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        out, hidden = self.gru(embedded, hidden)
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        # GRU hidden state is a single tensor, like RNN
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)

# Instantiate and train the GRU
gru_model = CharGRU(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS)
print("\n--- Training GRU Model ---")
train_model(gru_model, dataloader, EPOCHS, LR)

# Generate text with the trained GRU
print("\n--- GRU Generated Text ---")
print(generate_text(gru_model, start_string="ROMEO:", size=200))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"RNN Parameters:  {count_parameters(rnn_model):,}")
print(f"GRU Parameters:  {count_parameters(gru_model):,}")
print(f"LSTM Parameters: {count_parameters(lstm_model):,}")


# --- 1. Installation and Setup ---
# First, we need to install spaCy and download its small English language model.
# The '!' command runs a shell command from within the notebook.
!pip install -q spacy
!python -m spacy download en_core_web_sm

import spacy
import pandas as pd
from spacy import displacy

# Load the pre-trained English model
# This 'nlp' object contains the entire processing pipeline (tokenizer, tagger, parser, NER, etc.)
nlp = spacy.load("en_core_web_sm")

print("spaCy setup complete and model loaded.")




# --- 2. Performing POS Tagging ---
# Here's our sample sentence
text = "Apple is looking at buying a U.K. startup for $1 billion."

# Process the text with the spaCy pipeline
doc = nlp(text)

# We'll collect the POS information into a list of dictionaries
pos_data = []
for token in doc:
    # Skip punctuation and spaces for cleaner output
    if not token.is_punct and not token.is_space:
        pos_data.append({
            'Token': token.text,
            'Coarse-Grained POS': token.pos_,
            'Fine-Grained TAG': token.tag_,
            'Explanation': spacy.explain(token.tag_)
        })

# Display the results in a clean table using pandas
pos_df = pd.DataFrame(pos_data)
print("--- Part-of-Speech Tags ---")
display(pos_df)



# --- 3. Performing Named Entity Recognition ---
# We'll reuse the 'doc' object from the previous step,
# as the NER pipeline has already run.
print(f"Original Text: \"{text}\"")

# We can access the identified entities directly from the doc.ents attribute
ner_data = []
for ent in doc.ents:
    ner_data.append({
        'Entity Text': ent.text,
        'Entity Label': ent.label_,
        'Explanation': spacy.explain(ent.label_)
    })

# Display the extracted entities in a table
ner_df = pd.DataFrame(ner_data)
print("\n--- Named Entities Found ---")
display(ner_df)


# --- 4. Visualizing Entities with displacy ---
# 
# The 'ent' style highlights named entities.
# jupyter=True tells displacy to render the output directly in the notebook.
print("--- NER Visualization ---")
displacy.render(doc, style="ent", jupyter=True)


# Core data manipulation and analysis
import numpy as np
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# PyTorch deep learning framework
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F

# Preprocessing and evaluation
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Configure plotting style
plt.style.use('default')
sns.set_palette("husl")

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"ğŸ”§ Using device: {device}")


# Load the dataset
# Note: Replace this path with your actual dataset path
df = pd.read_csv("/kaggle/input/playground-series-s3e19/train.csv", 
                 parse_dates=["date"], 
                 index_col="date")

# Sort by date to ensure chronological order
df = df.sort_index()

print("ğŸ“Š Dataset Overview:")
print(f"Shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Number of unique dates: {len(df.index.unique())}")
print(f"Number of unique products: {df['product'].nunique()}")
print(f"Number of unique stores: {df['store'].nunique()}")
print(f"Number of unique countries: {df['country'].nunique()}")


# Initialize the scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit and transform the data
# We only scale the target variable 'num_sold'
sales_data = df[['num_sold']].values
scaled_data = scaler.fit_transform(sales_data)

def create_sequences(data, sequence_length):
    """
    Create sequences for time series prediction.
    
    Parameters:
    -----------
    data : array-like
        The time series data
    sequence_length : int
        Number of time steps to look back
        
    Returns:
    --------
    X : torch.Tensor
        Input sequences of shape (samples, sequence_length, features)
    y : torch.Tensor
        Target values of shape (samples, features)
    """
    X, y = [], []
    
    for i in range(len(data) - sequence_length):
        # Create input sequence
        sequence = data[i:(i + sequence_length)]
        X.append(sequence)
        
        # Create target (next value)
        target = data[i + sequence_length]
        y.append(target)
    
    # Convert to PyTorch tensors
    X = torch.FloatTensor(np.array(X))
    y = torch.FloatTensor(np.array(y))
    
    return X, y

# Set sequence length (number of days to look back)
SEQUENCE_LENGTH = 30  # Using past 30 days to predict the next day

# Create sequences
X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)


# Split data chronologically (no shuffling for time series!)
test_size = 0.2
split_index = int(len(X) * (1 - test_size))

X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

# Move tensors to device (GPU if available)
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

print("ğŸ“Š Train-Test Split Results:")
print(f"Training sequences: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Testing sequences: {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
print()
print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
print(f"Testing data shape: X={X_test.shape}, y={y_test.shape}")
print(f"Device: {X_train.device}")
print()
print("ğŸ’¡ Note: Data is split chronologically to maintain temporal order!")
print(f"ğŸ”§ All tensors moved to: {device}")


# Create PyTorch DataLoaders for efficient batch processing
BATCH_SIZE = 32

# Create datasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)  # Don't shuffle time series!
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("ğŸ”„ DataLoader Creation:")
print(f"Batch size: {BATCH_SIZE}")
print(f"Training batches: {len(train_loader)}")
print(f"Testing batches: {len(test_loader)}")
print(f"Shuffle: False (important for time series!)")

# Test the data loader
for batch_X, batch_y in train_loader:
    print(f"\nğŸ“¦ Sample batch:")
    print(f"Batch X shape: {batch_X.shape}")
    print(f"Batch y shape: {batch_y.shape}")
    print(f"Batch device: {batch_X.device}")
    break


class SalesLSTM(nn.Module):
    """
    PyTorch LSTM model for sales forecasting.
    
    This class inherits from nn.Module, which is the base class for all neural
    network modules in PyTorch. It provides automatic gradient computation
    and parameter management.
    """
    
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, dropout=0.2):
        """
        Initialize the LSTM model.
        
        Parameters:
        -----------
        input_size : int
            Number of input features (1 for univariate time series)
        hidden_size : int
            Number of LSTM hidden units
        num_layers : int
            Number of LSTM layers
        dropout : float
            Dropout probability
        """
        super(SalesLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True  # Input shape: (batch, seq, feature)
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Dense layers
        self.fc1 = nn.Linear(hidden_size, 25)
        self.fc2 = nn.Linear(25, 1)
        
        # Activation function
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
        --------
        torch.Tensor
            Output predictions of shape (batch_size, 1)
        """
        batch_size = x.size(0)
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x, (h0, c0))
        
        # Take the output of the last time step
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(last_output)
        
        # Dense layers with ReLU activation
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        
        return out
    
    def init_weights(self):
        """
        Initialize weights using Xavier/Glorot initialization.
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

# Create the model
model = SalesLSTM(
    input_size=1,
    hidden_size=50,
    num_layers=2,
    dropout=0.2
).to(device)

# Initialize weights
model.init_weights()

print("ğŸ§  PyTorch LSTM Model Architecture:")
print(model)
print()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"ğŸ“Š Model Statistics:")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model device: {next(model.parameters()).device}")
print(f"Model precision: {next(model.parameters()).dtype}")


# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Training parameters
EPOCHS = 50
PRINT_EVERY = 5

print("ğŸš€ Training Setup:")
print(f"Loss function: {criterion.__class__.__name__}")
print(f"Optimizer: {optimizer.__class__.__name__}")
print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
print(f"Epochs: {EPOCHS}")
print(f"Device: {device}")
print()

# Lists to store training history
train_losses = []
val_losses = []
learning_rates = []


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Returns:
    --------
    float: Average training loss for the epoch
    """
    model.train()  # Set model to training mode
    total_loss = 0.0
    num_batches = 0
    
    for batch_X, batch_y in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(batch_X)
        
        # Compute loss
        loss = criterion(predictions, batch_y)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update parameters
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def validate_epoch(model, val_loader, criterion, device):
    """
    Validate the model for one epoch.
    
    Returns:
    --------
    float: Average validation loss for the epoch
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():  # Disable gradient computation for efficiency
        for batch_X, batch_y in val_loader:
            # Forward pass
            predictions = model(batch_X)
            
            # Compute loss
            loss = criterion(predictions, batch_y)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


# Split training data for validation
val_size = 0.1
val_split = int(len(train_dataset) * (1 - val_size))

train_subset = torch.utils.data.Subset(train_dataset, range(val_split))
val_subset = torch.utils.data.Subset(train_dataset, range(val_split, len(train_dataset)))

train_loader_subset = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

print(f"ğŸ“Š Data Split:")
print(f"Training samples: {len(train_subset)}")
print(f"Validation samples: {len(val_subset)}")
print(f"Testing samples: {len(test_dataset)}")
print()

# Training loop
print("ğŸš€ Starting training...")
best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(EPOCHS):
    # Train for one epoch
    train_loss = train_epoch(model, train_loader_subset, criterion, optimizer, device)
    
    # Validate
    val_loss = validate_epoch(model, val_loader, criterion, device)
    
    # Store history
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    learning_rates.append(optimizer.param_groups[0]['lr'])
    
    # Update learning rate scheduler
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), 'best_sales_lstm.pth')
    else:
        patience_counter += 1
    
    # Print progress
    if (epoch + 1) % PRINT_EVERY == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:3d}/{EPOCHS}] | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    # Early stopping
    if patience_counter >= patience:
        print(f"\nâ�¹ï¸�  Early stopping at epoch {epoch+1}")
        break

# Load best model
model.load_state_dict(torch.load('best_sales_lstm.pth'))
print(f"\nâœ… Training completed!")
print(f"ğŸ“Š Best validation loss: {best_val_loss:.6f}")
print(f"ğŸ“Š Total epochs trained: {len(train_losses)}")


# Plot training history
def plot_training_history(train_losses, val_losses, learning_rates):
    """Plot training and validation loss over epochs."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, label='Training Loss', linewidth=2, color='blue')
    ax1.plot(epochs, val_losses, label='Validation Loss', linewidth=2, color='red', linestyle='--')
    ax1.set_title('ğŸ“‰ Model Loss Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot learning rate
    ax2.plot(epochs, learning_rates, label='Learning Rate', linewidth=2, color='green')
    ax2.set_title('ğŸ“Š Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.show()

plot_training_history(train_losses, val_losses, learning_rates)

# Print training summary
print(f"ğŸ“Š Training Summary:")
print(f"Total epochs trained: {len(train_losses)}")
print(f"Final training loss: {train_losses[-1]:.6f}")
print(f"Final validation loss: {val_losses[-1]:.6f}")
print(f"Best validation loss: {min(val_losses):.6f}")
print(f"Final learning rate: {learning_rates[-1]:.2e}")
print(f"ğŸ’¡ PyTorch gives us full control over the training process!")


def make_predictions(model, test_loader, device, scaler):
    """
    Make predictions using the trained model.
    
    Returns:
    --------
    y_pred : numpy.ndarray
        Predictions in original scale
    y_true : numpy.ndarray
        True values in original scale
    """
    model.eval()
    predictions = []
    targets = []
    
    print("ğŸ”® Making predictions...")
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            # Forward pass
            pred = model(batch_X)
            
            # Move to CPU and convert to numpy
            predictions.extend(pred.cpu().numpy())
            targets.extend(batch_y.cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Transform back to original scale
    y_pred = scaler.inverse_transform(predictions)
    y_true = scaler.inverse_transform(targets.reshape(-1, 1))
    
    return y_pred, y_true

# Make predictions
y_pred, y_test_original = make_predictions(model, test_loader, device, scaler)

print("âœ… Predictions completed!")
print(f"Predictions shape: {y_pred.shape}")
print(f"Test targets shape: {y_test_original.shape}")
print(f"Data type: {type(y_pred)} (converted from PyTorch tensors)")


# Calculate evaluation metrics
mse = mean_squared_error(y_test_original, y_pred)
mae = mean_absolute_error(y_test_original, y_pred)
rmse = np.sqrt(mse)

# Calculate percentage errors (avoid division by zero)
mask = y_test_original != 0
mape = np.mean(np.abs((y_test_original[mask] - y_pred[mask]) / y_test_original[mask])) * 100

# Calculate R-squared
ss_res = np.sum((y_test_original - y_pred) ** 2)
ss_tot = np.sum((y_test_original - np.mean(y_test_original)) ** 2)
r2 = 1 - (ss_res / ss_tot)

print("ğŸ“Š PyTorch Model Performance Metrics:")
print("=" * 55)
print(f"Mean Squared Error (MSE): {mse:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
print(f"Mean Absolute Error (MAE): {mae:,.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"R-squared (RÂ²): {r2:.4f}")
print()


print(f"\nğŸ’¡ All computations performed on: {device}")
print(f"ğŸ”¥ PyTorch model successfully trained and evaluated!")

