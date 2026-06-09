import pandas as pd
from IPython.display import display

train1 = pd.read_csv("train.csv")
train2 = pd.read_csv("training_extra.csv")
train = pd.concat([train1, train2], ignore_index = True)
test = pd.read_csv("test.csv")
display(train.head())
display(train.describe())
display(train.isna().sum())
for col in train:
    print(train[col].unique())
    print(train[col].dtype)


import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler

class BackpackPriceDataset(Dataset):
    def __init__(self, csv_files, test_mode=False) -> None:
        self.device = torch.device('cpu')
        self.test_mode = test_mode

        if isinstance(csv_files, str):
            self.data = pd.read_csv(csv_files)
        else:
            dfs = [pd.read_csv(file) for file in csv_files]
            self.data = pd.concat(dfs, ignore_index=True)


        cols_to_scale = ['Weight Capacity (kg)']
        self.numerical_cols = ['Size', 'Compartments', 'Laptop Compartment', 'Waterproof', 'Weight Capacity (kg)']
        self.categorical_cols = ['Brand', 'Material', 'Style', 'Color']

        self._handle_missing_values()

        self.label_encoders = {}
        self.scaler = StandardScaler()

        for col in self.categorical_cols:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col])
            self.label_encoders[col] = le

        self.data[cols_to_scale] = self.scaler.fit_transform(self.data[cols_to_scale])

        self.num_categories = {
            col: len(self.data[col].unique()) for col in self.categorical_cols
        }

        drop_cols = ['id']
        if not test_mode:
            drop_cols.append('Price')
            self.target = self.data['Price']

        self.features = self.data.drop(drop_cols, axis=1)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # return categorical and numerical seperately due to embedding of categorical
        categorical = torch.tensor([
            self.data[col].iloc[idx] for col in self.categorical_cols
        ], dtype=torch.long)#.to(self.device)

        numerical = torch.tensor([
            self.data[col].iloc[idx] for col in self.numerical_cols
        ], dtype=torch.float32)#.to(self.device)

        if self.test_mode:
            return categorical, numerical
        else:
            target = torch.tensor([self.target.iloc[idx]], dtype=torch.float32)#.to(self.device)
            return categorical, numerical, target

    def _handle_missing_values(self):
        # https://medium.com/@felipecaballero/deciphering-the-cryptic-futurewarning-for-fillna-in-pandas-2-01deb4e411a1
        with pd.option_context('future.no_silent_downcasting', True):
            for col in self.categorical_cols:
                self.data[col] = self.data[col].fillna("Missing")
        
            self.data['Size'] = self.data['Size'].fillna("Missing")
            self.data['Size'] = self.data['Size'].replace({
                'Small': -1,
                'Medium': 0,
                'Large': 1,
                'Missing': 0  # Assume missing sizes are Medium
            }).infer_objects()
            
            # Compartments (whole numbers)
            median_compartments = round(self.data['Compartments'].median())
            self.data['Compartments'] = self.data['Compartments'].fillna(median_compartments).infer_objects()
            
            # Weight Capacity (continuous)
            self.data['Weight Capacity (kg)'] = self.data['Weight Capacity (kg)'].fillna(
                self.data['Weight Capacity (kg)'].median()
            )
            
            # Binary features (assume missing means "No")
            for col in ['Laptop Compartment', 'Waterproof']:
                self.data[col] = self.data[col].fillna("No")
                self.data[col] = self.data[col].replace({'No': 0, 'Yes': 1}).infer_objects()


train_dataset = BackpackPriceDataset(["train.csv", "training_extra.csv"], test_mode=False)
test_dataset = BackpackPriceDataset("test.csv", test_mode=True)
random_seed = 42


from sklearn.ensemble import RandomForestRegressor
import numpy as np 

train2 = BackpackPriceDataset("train.csv", test_mode=False)
X, y = train2.features, train2.target

rf = RandomForestRegressor(n_estimators=100, random_state=random_seed, max_depth=10)
rf.fit(X,y)

importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(importance)


for col, count in train_dataset.num_categories.items():
    rec_dim = min(50, count//2)
    print(f"{col}: {count}. Recommended embedding_dim: {rec_dim}")


import torch
import torch.nn as nn

class BackpackPriceNet(nn.Module):
    def __init__(self, num_categories_dict) -> None:
        super().__init__()

        self.embeddings = nn.ModuleDict({
            'Brand': nn.Embedding(num_categories_dict['Brand'], 3),
            'Material': nn.Embedding(num_categories_dict['Material'], 2),
            'Style': nn.Embedding(num_categories_dict['Style'], 2),
            'Color': nn.Embedding(num_categories_dict['Color'], 3)
        })

        embedding_dim = 3 + 2 + 2 + 3
        numerical_dim = 5
        input_dim = embedding_dim + numerical_dim


        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),


            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),


            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),

            nn.Linear(64, 1) 
        )

    def forward(self, categorical_inputs, numerical_inputs):
        embeddings = []
        for i, (_, embedding_layer) in enumerate(self.embeddings.items()):
            embedding = embedding_layer(categorical_inputs[:, i])
            embeddings.append(embedding)
        
        x_cat = torch.cat(embeddings, dim=1)
        x = torch.cat([x_cat, numerical_inputs], dim=1)

        return self.model(x)


from torch.utils.data import DataLoader
def train_loop(dataloader: DataLoader, model: BackpackPriceNet, loss_fn, optimizer):
    num_batches = len(dataloader)
    model.train()
    total_loss = 0
    
    for batch, (cat_features, num_features, target) in enumerate(dataloader):
        pred = model(cat_features, num_features)
        loss = loss_fn(pred, target) 
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            print(f"Training batch loss: {loss.item():>7f} [{batch:>5d}/{num_batches:>5d}]")

    return total_loss / num_batches

def val_loop(dataloader: DataLoader, model: BackpackPriceNet, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch, (cat_features, num_features, target) in enumerate(dataloader):
            pred = model(cat_features, num_features)
            val_loss += loss_fn(pred, target).item()

    val_loss /= num_batches
    print(f"\nValdidation average loss: {val_loss:>8f}\n")
    return val_loss

def predict(dataloader: DataLoader, model: BackpackPriceNet):
    model.eval()
    predictions = []

    with torch.no_grad():
        for cat_features, num_features in dataloader:
            outputs = model(cat_features, num_features)
            predictions.extend(outputs.cpu().numpy())

    return np.array(predictions)


from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np
from collections import defaultdict

torch.manual_seed(random_seed)
np.random.seed(random_seed)

indicies = np.arange(len(train_dataset))
train_indicies, val_indicies = train_test_split(indicies, test_size=0.2, random_state=random_seed)

train_sampler = SubsetRandomSampler(train_indicies)
val_sampler = SubsetRandomSampler(val_indicies)

batch_size = 128
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=val_sampler)

my_model = BackpackPriceNet(train_dataset.num_categories)
device = torch.device('cpu')
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(my_model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.2, patience=2)
history = defaultdict(list)

best_val_loss = float('inf')
best_model_path = "best_model.pth"

patience = 3
early_stopping_counter = 0

num_epochs = 20
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}\n----------------------")
    train_loss = train_loop(train_loader, my_model, loss_fn, optimizer)
    val_loss = val_loop(val_loader, my_model, loss_fn)
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)


    if val_loss < best_val_loss:
        early_stopping_counter = 0
        best_val_loss = val_loss
        torch.save(my_model.state_dict(), best_model_path)
        print(f"Saved best new model with val_loss: {val_loss:.4f}")
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

    scheduler.step(val_loss)


my_model.load_state_dict(torch.load(best_model_path))
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
predictions = predict(test_loader, my_model)
test_predictions_df = pd.DataFrame({
    'id': test_dataset.data['id'],
    'Price': predictions.flatten()
})
test_predictions_df.to_csv('predictions.csv', index=False)

