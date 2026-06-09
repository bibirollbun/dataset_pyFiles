import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
print("Loading data...")
df = pd.read_csv('/kaggle/input/movie-recomendation-fall-2020/train.txt', sep='\t', header=None, names=['user_id', 'movie_id', 'rating'])

print(f"Dataset shape: {df.shape}")
print(f"Number of unique users: {df['user_id'].nunique()}")
print(f"Number of unique movies: {df['movie_id'].nunique()}")
print(f"Rating range: {df['rating'].min()} - {df['rating'].max()}")

# Preprocessing
user_ids = df['user_id'].unique()
item_ids = df['movie_id'].unique()

user_encoder = {user: idx for idx, user in enumerate(user_ids)}
item_encoder = {item: idx for idx, item in enumerate(item_ids)}

df['user_idx'] = df['user_id'].map(user_encoder)
df['item_idx'] = df['movie_id'].map(item_encoder)

num_users = len(user_encoder)
num_items = len(item_encoder)

print(f"Encoded users: {num_users}, Encoded items: {num_items}")

# Prepare data
users = df['user_idx'].values
items = df['item_idx'].values
ratings = df['rating'].values.astype(np.float32)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    np.column_stack([users, items]), ratings, test_size=0.2, random_state=42, shuffle=True
)

print(f"Training set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")

# Create Dataset class
class RatingDataset(Dataset):
    def __init__(self, users, items, ratings):
        self.users = torch.LongTensor(users)
        self.items = torch.LongTensor(items)
        self.ratings = torch.FloatTensor(ratings)
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

# Create data loaders
train_dataset = RatingDataset(X_train[:, 0], X_train[:, 1], y_train)
val_dataset = RatingDataset(X_val[:, 0], X_val[:, 1], y_val)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
# NeuMF Model
# Training function
# Evaluation function


#NeuMF
class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, mf_embedding=32, mlp_embedding=32, hidden_layers=[64, 32, 16]):
        super(NeuMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.mf_embedding = mf_embedding
        self.mlp_embedding = mlp_embedding
        
        # GMF Embeddings
        self.mf_user_embedding = nn.Embedding(num_users, mf_embedding)
        self.mf_item_embedding = nn.Embedding(num_items, mf_embedding)
        
        # MLP Embeddings
        self.mlp_user_embedding = nn.Embedding(num_users, mlp_embedding)
        self.mlp_item_embedding = nn.Embedding(num_items, mlp_embedding)
        
        # MLP Layers
        self.mlp_layers = nn.Sequential()
        input_size = mlp_embedding * 2
        
        for i, hidden_size in enumerate(hidden_layers):
            self.mlp_layers.add_module(f"fc_{i}", nn.Linear(input_size, hidden_size))
            self.mlp_layers.add_module(f"relu_{i}", nn.ReLU())
            self.mlp_layers.add_module(f"dropout_{i}", nn.Dropout(0.2))
            input_size = hidden_size
        
        # Final output layer
        self.output_layer = nn.Linear(mf_embedding + hidden_layers[-1], 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Initialize embedding weights
        nn.init.normal_(self.mf_user_embedding.weight, std=0.01)
        nn.init.normal_(self.mf_item_embedding.weight, std=0.01)
        nn.init.normal_(self.mlp_user_embedding.weight, std=0.01)
        nn.init.normal_(self.mlp_item_embedding.weight, std=0.01)
        
        # Initialize MLP layers
        for layer in self.mlp_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
        
        # Initialize output layer
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0.0)
    
    def forward(self, user_indices, item_indices):
        # GMF path
        mf_user_embedded = self.mf_user_embedding(user_indices)
        mf_item_embedded = self.mf_item_embedding(item_indices)
        mf_vector = mf_user_embedded * mf_item_embedded  # Element-wise product
        
        # MLP path
        mlp_user_embedded = self.mlp_user_embedding(user_indices)
        mlp_item_embedded = self.mlp_item_embedding(item_indices)
        mlp_vector = torch.cat([mlp_user_embedded, mlp_item_embedded], dim=-1)
        mlp_vector = self.mlp_layers(mlp_vector)
        
        # Concatenate GMF and MLP
        concat_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        
        # Output
        output = self.output_layer(concat_vector)
        return output.squeeze()


# Training function
def train_model(model, train_loader, val_loader, epochs=30, learning_rate=0.001):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, (users, items, ratings) in enumerate(train_loader):
            users = users.to(device)
            items = items.to(device)
            ratings = ratings.to(device)
            
            optimizer.zero_grad()
            predictions = model(users, items)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for users, items, ratings in val_loader:
                users = users.to(device)
                items = items.to(device)
                ratings = ratings.to(device)
                
                predictions = model(users, items)
                loss = criterion(predictions, ratings)
                val_loss += loss.item()
                
                val_predictions.extend(predictions.cpu().numpy())
                val_targets.extend(ratings.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Calculate RMSE
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_predictions))
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_neumf_model.pth')
        else:
            patience_counter += 1
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val RMSE: {val_rmse:.4f}')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}, Patience: {patience_counter}/{patience}')
        
         # if patience_counter >= patience:
         #    print("Early stopping triggered!")
         #    break
    
    # Load best model
    model.load_state_dict(torch.load('best_neumf_model.pth'))
    return model, train_losses, val_losses


# Evaluation function
def evaluate_model(model, data_loader):
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for users, items, ratings in data_loader:
            users = users.to(device)
            items = items.to(device)
            ratings = ratings.to(device)
            
            preds = model(users, items)
            predictions.extend(preds.cpu().numpy())
            targets.extend(ratings.cpu().numpy())
    
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = np.mean(np.abs(np.array(targets) - np.array(predictions)))
    
    return rmse, mae, predictions, targets


# Build and train model
print("Building NeuMF model...")
model = NeuMF(
    num_users=num_users,
    num_items=num_items,
    mf_embedding=32,
    mlp_embedding=32,
    hidden_layers=[64, 32, 16]
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Train model
print("Training model...")
trained_model, train_losses, val_losses = train_model(
    model, train_loader, val_loader, epochs=30, learning_rate=0.001
)

# Final evaluation
print("\nFinal Evaluation:")
train_rmse, train_mae, _, _ = evaluate_model(trained_model, train_loader)
val_rmse, val_mae, val_predictions, val_targets = evaluate_model(trained_model, val_loader)

print(f"{'='*50}")
print(f"Final Results:")
print(f"{'='*50}")
print(f"Train RMSE: {train_rmse:.4f}, Train MAE: {train_mae:.4f}")
print(f"Val RMSE: {val_rmse:.4f}, Val MAE: {val_mae:.4f}")
print(f"{'='*50}")

# Plot training history
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Training History')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(val_targets, val_predictions, alpha=0.5)
plt.plot([min(val_targets), max(val_targets)], [min(val_targets), max(val_targets)], 'r--', lw=2)
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Predictions vs Actual')

plt.subplot(1, 3, 3)
errors = np.array(val_targets) - np.array(val_predictions)
plt.hist(errors, bins=50, alpha=0.7)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Error Distribution')

plt.tight_layout()
plt.show()

# Sample predictions
print("\nSample Predictions:")
print(f"{'User':>6} {'Item':>6} {'Actual':>8} {'Predicted':>10} {'Error':>8}")
print("-" * 50)

sample_indices = np.random.choice(len(X_val), 10, replace=False)
trained_model.eval()
with torch.no_grad():
    for idx in sample_indices:
        user = torch.LongTensor([X_val[idx, 0]]).to(device)
        item = torch.LongTensor([X_val[idx, 1]]).to(device)
        actual = y_val[idx]
        predicted = trained_model(user, item).cpu().item()
        error = abs(actual - predicted)
        
        print(f"{X_val[idx, 0]:6d} {X_val[idx, 1]:6d} {actual:8.2f} {predicted:10.4f} {error:8.4f}")

# Save final model
torch.save({
    'model_state_dict': trained_model.state_dict(),
    'user_encoder': user_encoder,
    'item_encoder': item_encoder,
    'num_users': num_users,
    'num_items': num_items,
}, 'neumf_final_model.pth')

print(f"\nModel saved as 'neumf_final_model.pth'")

# Model info
print(f"\nModel Information:")
print(f"Number of users: {num_users}")
print(f"Number of items: {num_items}")
print(f"Total parameters: {sum(p.numel() for p in trained_model.parameters()):,}")
print(f"Embedding dimensions: MF={model.mf_embedding}, MLP={model.mlp_embedding}")


# =============================================================================
# TEST TRÊN TẬP TEST VÀ TẠO FILE SUBMISSION THEO ĐÚNG FORMAT KAGGLE
# =============================================================================

# Load test data
print("LOADING TEST DATA AND MAKING PREDICTIONS")
test_data = pd.read_csv('/kaggle/input/movie-recomendation-fall-2020/test.txt', 
                       sep='\t', names=['user_id', 'item_id'])

print(f"Test data shape: {test_data.shape}")
print(f"Test data preview:")
print(test_data.head())

# Kiểm tra distribution của test data
print(f"\nTest data statistics:")
print(f"Unique users in test: {test_data['user_id'].nunique()}")
print(f"Unique items in test: {test_data['item_id'].nunique()}")
print(f"User ID range in test: {test_data['user_id'].min()} - {test_data['user_id'].max()}")
print(f"Item ID range in test: {test_data['item_id'].min()} - {test_data['item_id'].max()}")

# Kiểm tra overlap với training data
train_users = set(user_encoder.keys())
train_items = set(item_encoder.keys())
test_users = set(test_data['user_id'].unique())
test_items = set(test_data['item_id'].unique())

print(f"\nData overlap analysis:")
print(f"Users in both train and test: {len(train_users.intersection(test_users))}")
print(f"Items in both train and test: {len(train_items.intersection(test_items))}")
print(f"New users in test (cold start): {len(test_users - train_users)}")
print(f"New items in test (cold start): {len(test_items - train_items)}")

# Xử lý test data
print("\nProcessing test data for prediction...")

def safe_map_to_index(original_id, encoder, id_type):
    """Map original ID to encoded index, handle cold start problems"""
    if original_id in encoder:
        return encoder[original_id]
    else:
        # Xử lý cold start: sử dụng index 0 (có thể thay bằng các strategies khác)
        print(f"Warning: {id_type} {original_id} not found in training data, using default index 0")
        return 0

# Map test data to encoded indices
test_data['user_idx'] = test_data['user_id'].apply(
    lambda x: safe_map_to_index(x, user_encoder, 'user'))
test_data['item_idx'] = test_data['item_id'].apply(
    lambda x: safe_map_to_index(x, item_encoder, 'item'))

print(f"Test data after mapping:")
print(test_data.head())

# Dự đoán ratings
print("\nMaking predictions on test data...")
test_predictions = []

trained_model.eval()
with torch.no_grad():
    for i in range(len(test_data)):
        user_idx = test_data.iloc[i]['user_idx']
        item_idx = test_data.iloc[i]['item_idx']
        
        user_tensor = torch.LongTensor([user_idx]).to(device)
        item_tensor = torch.LongTensor([item_idx]).to(device)
        
        prediction = trained_model(user_tensor, item_tensor).cpu().item()
        test_predictions.append(prediction)

# Thêm predictions vào test_data
test_data['rating'] = test_predictions

# Clip predictions to valid rating range (1-5)
test_data['rating'] = test_data['rating'].clip(1.0, 5.0)

# TẠO FILE SUBMISSION THEO ĐÚNG FORMAT KAGGLE: Id, Score
print("\n" + "="*60)
print("CREATING SUBMISSION FILE WITH CORRECT KAGGLE FORMAT")
print("="*60)

# Tạo submission theo format Kaggle: Id, Score
# Id là số thứ tự từ 1 đến len(test_data)
submission = pd.DataFrame({
    'Id': range(1, len(test_data) + 1),  # Id bắt đầu từ 1
    'Score': test_data['rating'].values  # Score là predicted rating
})

# Kiểm tra format submission
print(f"Submission data shape: {submission.shape}")
print(f"Submission preview:")
print(submission.head(10))

print(f"\nSubmission format verification:")
print(f"Columns: {submission.columns.tolist()}")
print(f"Id range: {submission['Id'].min()} - {submission['Id'].max()}")
print(f"Score range: {submission['Score'].min():.3f} - {submission['Score'].max():.3f}")
print(f"Total predictions: {len(submission)}")

# Lưu file submission
submission_file = 'submission.csv'
submission.to_csv(submission_file, index=False, header=True)

print(f"\n✓ Submission file saved as: {submission_file}")

# Kiểm tra file đã lưu
import os
if os.path.exists(submission_file):
    check_df = pd.read_csv(submission_file)
    print(f"✓ File verification successful:")
    print(f"  File size: {os.path.getsize(submission_file)} bytes")
    print(f"  Columns: {check_df.columns.tolist()}")
    print(f"  First 5 rows:")
    print(check_df.head())
else:
    print(f"❌ Error: File {submission_file} was not created!")

# Validation thêm với training data để đảm bảo model hoạt động tốt
print(f"\n" + "="*60)
print("FINAL VALIDATION ON TRAINING DATA")
print("="*60)

# Tính final metrics trên validation set
final_val_rmse, final_val_mae, final_val_pred, final_val_true = evaluate_model(trained_model, val_loader)

print(f"Final Validation Metrics:")
print(f"RMSE: {final_val_rmse:.4f}")
print(f"MAE: {final_val_mae:.4f}")

# Plot final predictions vs actual
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(final_val_true, final_val_pred, alpha=0.5)
plt.plot([1, 5], [1, 5], 'r--', lw=2)
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Final Validation: Predictions vs Actual')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
residuals = np.array(final_val_true) - np.array(final_val_pred)
plt.hist(residuals, bins=50, alpha=0.7, color='skyblue')
plt.axvline(x=0, color='red', linestyle='--')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Error Distribution')

plt.subplot(1, 3, 3)
# Distribution of test predictions
plt.hist(test_data['rating'], bins=30, alpha=0.7, color='lightgreen')
plt.xlabel('Predicted Ratings')
plt.ylabel('Frequency')
plt.title('Test Predictions Distribution')

plt.tight_layout()
plt.savefig('final_validation_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Summary report
print(f"\n" + "="*60)
print("SUMMARY REPORT")
print("="*60)
print(f"Model: NeuMF")
print(f"Training users: {num_users}")
print(f"Training items: {num_items}")
print(f"Training samples: {len(df)}")
print(f"Test samples: {len(test_data)}")
print(f"Final Validation RMSE: {final_val_rmse:.4f}")
print(f"Final Validation MAE: {final_val_mae:.4f}")
print(f"Test predictions range: {test_data['rating'].min():.3f} - {test_data['rating'].max():.3f}")
print(f"Submission file: {submission_file}")
print(f"Submission format: Id, Score")
print("="*60)

# Lưu thêm model info và results (ĐÃ SỬA LỖI JSON)
results_summary = {
    'model': 'NeuMF',
    'num_users': int(num_users),
    'num_items': int(num_items),
    'train_samples': int(len(df)),
    'test_samples': int(len(test_data)),
    'val_rmse': float(final_val_rmse),
    'val_mae': float(final_val_mae),
    'test_min_rating': float(test_data['rating'].min()),
    'test_max_rating': float(test_data['rating'].max()),
    'test_mean_rating': float(test_data['rating'].mean()),
    'submission_file': submission_file,
    'submission_format': 'Id,Score'
}

import json
with open('training_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)



pip install torchsummary

