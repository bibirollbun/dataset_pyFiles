import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import gc

device = 'cuda' if torch.cuda.is_available() else 'cpu'


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv", index_col="id")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv", index_col="id")
original_data=pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")


#train = train[:100000]
#original_data = original_data[:10000]


# Splitting the training data into training and validation sets
train_, val = train_test_split(train, test_size=0.2, random_state=42)


class FertilizerDataset(Dataset):
    def __init__(self, data, original_data=None, target_col=None):

        self.data = data  
        if original_data is not None:
            for _ in range(4):
                self.data = pd.concat([self.data, original_data], ignore_index=True)
        else: 
            self.data = data

        if target_col is not None and target_col in self.data.columns:
            self.target = self.data[target_col].values
            self.data = self.data.drop(columns=[target_col])
        else:
            self.target = None

        # self.num_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = self.data.columns

        # Numerical features
        # self.scaler = StandardScaler()
        # self.numerical_data = self.scaler.fit_transform(self.data[self.num_cols])

        # Categorical features
        self.categorical_data = {}
        self.label_encoders = {}
        for col in self.cat_cols:
            le = LabelEncoder()
            self.categorical_data[col] = le.fit_transform(self.data[col])
            self.label_encoders[col] = le

        if self.target is not None:
            self.target_encoder = LabelEncoder()
            self.target = self.target_encoder.fit_transform(self.target)

        self.categorical_tensors = {}
        for col in self.cat_cols:
            self.categorical_tensors[col] = torch.tensor(
                self.categorical_data[col], dtype=torch.long
            )
        
        if self.target is not None:
            self.target_tensor = torch.tensor(self.target, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # numerical_features = torch.tensor(self.numerical_data[idx], dtype=torch.float32)

        categorical_features = {}
        for col in self.cat_cols:
            categorical_features[col] = self.categorical_tensors[col][idx]

        if self.target is not None:
            return categorical_features, self.target_tensor[idx]
        else:
            return categorical_features
    
train_dataset = FertilizerDataset(train, original_data, target_col='Fertilizer Name')
test_dataset = FertilizerDataset(test, target_col='Fertilizer Name')

test_dataloader = DataLoader(test_dataset, batch_size=1024*8, shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True)


next(iter(test_dataloader))  # Check if the dataloader works


embedding_configs = dict(zip(train_dataset.cat_cols, [train_dataset.label_encoders[train_dataset.cat_cols[i]].classes_.shape[0] for i in range(len(train_dataset.cat_cols))]))
embedding_configs


class SpatialDropout1D(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
            
        mask = torch.rand(x.size(0), x.size(1), device=x.device) > self.p
        return x * mask.float() / (1 - self.p)
        
class FertilizerClassifier(nn.Module):
    def __init__(self, n_classes, embedding_configs, embedding_dim_max=64, hidden_dim=64, dropout=0.2):
        super().__init__()

        self.dropout = dropout

        self.embeddings = nn.ModuleDict()
        self.dropouts = nn.ModuleDict()
        
        embeddings_size = 0

        for feature_name, vocab_size in embedding_configs.items():
            embedding_dim = int(min(np.ceil(vocab_size * 2), embedding_dim_max))
            embeddings_size += embedding_dim
            self.embeddings[feature_name] = nn.Embedding(vocab_size, embedding_dim)
            if embedding_dim > 25:
                self.dropouts[feature_name] = nn.Dropout1d(self.dropout)


        self.normalization = nn.BatchNorm1d(embeddings_size)

        self.ll1 = nn.Linear(embeddings_size, hidden_dim)
        self.normalization1 = nn.BatchNorm1d(hidden_dim)

        self.ll2 = nn.Linear(hidden_dim, 128)
        self.normalization2 = nn.BatchNorm1d(128)

        self.output_layer = nn.Linear(128, n_classes)

    def forward(self, inputs):

        embeddings = []

        for feature_name, values in inputs.items():
            emb = self.embeddings[feature_name](values)
            
            if feature_name in self.dropouts:
                emb = emb.unsqueeze(1) 
                emb = self.dropouts[feature_name](emb)
                emb = emb.squeeze(1) 
            
            embeddings.append(emb)
        
        x = torch.cat(embeddings, dim=1)
        x = F.dropout(x, p=self.dropout)
        x = self.normalization(x)

        x = F.relu(self.ll1(x))
        x = F.dropout(x, p=self.dropout)
        x = self.normalization1(x)

        x = F.relu(self.ll2(x))
        x = F.dropout(x, p=self.dropout)
        x = self.normalization2(x)

        output = self.output_layer(x)
        return output

N_CLASSES       = train_dataset.target_encoder.classes_.shape[0]
EMBEDDING_DIM   = 64*2
HIDDEN_DIM      = 1024*2
DROPOUT         = 0.2
model = FertilizerClassifier(n_classes=N_CLASSES, 
                             embedding_configs=embedding_configs, 
                             embedding_dim_max=EMBEDDING_DIM, 
                             hidden_dim=HIDDEN_DIM,
                             dropout=DROPOUT
                             ).to(device)

model = FertilizerClassifier(n_classes=N_CLASSES, 
                             embedding_configs=embedding_configs, 
                             embedding_dim_max=EMBEDDING_DIM, 
                             hidden_dim=HIDDEN_DIM,
                             dropout=DROPOUT).to(device)
    
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)


from torchinfo import summary

summary(model)


def map3(preds, targets):
    map_score = 0.0
    for j in range(len(targets)):
        # Get top 3 predictions for this sample
        top_3_preds_j = preds[j]
        correct = 0
        precision = 0.0
        
        for k, pred in enumerate(top_3_preds_j):
            if pred == targets[j]:
                correct += 1
                precision += correct / (k + 1)
        
        # Average precision for this sample
        if correct > 0:
            map_score += precision / min(1, correct)

    return (map_score / len(targets))


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-4)


K = 10
PATIENCE = 5
EPOCHS = 100

skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

train_oof = np.zeros((len(train_dataset), N_CLASSES))
test_pred= np.zeros((len(test), N_CLASSES))

y_ori = train_dataset.target

lr_scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max',        
        factor=0.5,          
        patience=1,           
        min_lr=1e-6
    )

for i, (train_idx, val_idx) in enumerate(skf.split(train_dataset, y_ori)):
    print(f"Fold {i}\n {10*'-'}")

    val_idx = val_idx[val_idx < train.shape[0]]
    
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)

    train_dataloader = DataLoader(train_subset, 
                                  batch_size=1024*2, 
                                  shuffle=True,
                                  pin_memory=True,
                                  persistent_workers=True,
                                  num_workers=4)
    
    val_dataloader = DataLoader(val_subset, 
                                batch_size=1024*8, 
                                shuffle=False,
                                num_workers=4,
                                pin_memory=True,
                                persistent_workers=True)

    # Early stopping variables
    best_val_map3 = 0.0
    patience_counter = 0
    best_model_state = None

    torch.cuda.empty_cache()
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 20)

        model.train()
        train_loss = 0
        train_batches = 0
        for batch, (data) in enumerate(train_dataloader):
            cat_data, y = data
            cat_data = {key: value.cuda(non_blocking=True) for key, value in cat_data.items()}
            y = y.cuda(non_blocking=True)

            y_pred_logits = model(cat_data)

            loss = loss_fn(y_pred_logits, y)
            train_loss += loss
            train_batches += 1

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # if batch % 30 == 0:
            #     print(f"Batch {batch}, Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / train_batches

        ### Validation
        model.eval()
        val_loss = 0
        val_map3_total = 0
        val_batches = 0
        oof_preds_list = []
        with torch.no_grad():
            for batch, (data_val) in enumerate(val_dataloader):
                cat_data_val, y_val = data_val
                cat_data_val = {key: value.cuda(non_blocking=True) for key, value in cat_data_val.items()}
                y_val = y_val.cuda(non_blocking=True)

                y_val_pred = model(cat_data_val)
                # y_val_pred_probs = torch.softmax(y_val_pred, dim=1)

                oof_preds_list.append(y_val_pred.cpu().numpy())
                
                val_loss += loss_fn(y_val_pred, y_val)

                top_3_values, top_3_indices = torch.topk(y_val_pred, k=3, dim=1, largest=True)
                batch_map3 = map3(top_3_indices, y_val)
                val_map3_total += batch_map3
                val_batches += 1

                # if batch % 10 == 0:
                    # print(f"Val Batch {batch} | Val MAP@3: {map3(top_3_indices, y_val):.4f}")

        avg_val_loss = val_loss / val_batches
        avg_val_map3 = val_map3_total / val_batches

        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val MAP@3: {avg_val_map3:.4f}")

        lr_scheduler.step(avg_val_map3)

        # Early stopping logic
        if avg_val_map3 > best_val_map3:
            best_val_map3 = avg_val_map3
            patience_counter = 0
            # Save best model state
            best_model_state = model.state_dict().copy()
            print(f"New best validation MAP@3: {best_val_map3:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")
            
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Restore best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with MAP@3: {best_val_map3:.4f}")
    
    # Final validation predictions with best model
    model.eval()
    final_oof_preds_list = []
    with torch.no_grad():
        for batch, data_val in enumerate(val_dataloader):
            cat_data_val, y_val = data_val
            cat_data_val = {key: value.cuda(non_blocking=True) for key, value in cat_data_val.items()}

            y_val_pred = model(cat_data_val)
            final_oof_preds_list.append(y_val_pred.cpu().numpy())
    
    oof_preds = np.vstack(final_oof_preds_list)
    train_oof[val_idx] = oof_preds

    # Generate test predictions with best model
    test_preds_list = []
    model.eval()
    with torch.no_grad():
        for batch, data_test in enumerate(test_dataloader):
            cat_data_test = data_test
            cat_data_test = {key: value.to(device) for key, value in cat_data_test.items()}
            
            y_test_pred = model(cat_data_test)
            test_preds_list.append(y_test_pred.cpu().numpy())
    
    test_pred += np.vstack(test_preds_list)
    
    # Calculate final fold MAP@3 score
    y_val_true = y_ori[val_idx]
    oof_top3 = np.argsort(oof_preds, axis=1)[:, -3:][:, ::-1].copy()
    oof_map = map3(torch.tensor(oof_top3), torch.tensor(y_val_true))
    print(f'Fold_{i} final map3 score: {oof_map:.4f}')
    
    # Memory cleanup (equivalent to tf.keras.backend.clear_session())
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    print(f"Fold {i} completed with best MAP@3: {best_val_map3:.4f}")
    print("=" * 50)

# Average test predictions across folds
test_pred /= K

# Calculate overall OOF score
oof_top3_overall = np.argsort(train_oof, axis=1)[:, -3:][:, ::-1].copy()
overall_oof_map = map3(torch.tensor(oof_top3_overall), torch.tensor(y_ori))
print(f'Overall OOF MAP@3 score: {overall_oof_map:.4f}')
            
            



test_ids = test.index
test_pred_np = test_pred
top3_idx = np.argsort(-test_pred_np, axis=1)[:, :3]
labels = train_dataset.target_encoder.inverse_transform(top3_idx.ravel())
pred_names = labels.reshape(top3_idx.shape)
submission_format = [' '.join(row) for row in pred_names]

submission = pd.DataFrame({'id': test_ids, 'Fertilizer Name': submission_format})
submission.to_csv('submission.csv', index=False)
submission.head()

