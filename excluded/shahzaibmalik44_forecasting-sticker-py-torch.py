import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


train_df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


def remove_non(df):
    return df.dropna()

train_df, test_df = map(remove_non, [train_df, test_df])


cat_cols = list(train_df.select_dtypes(include=['object']).columns)
for col in cat_cols:
    print(f"{col}, ", end="")
    train_df[col], _ = train_df[col].factorize()
    train_df[col] -= train_df[col].min()
    test_df[col], _ = test_df[col].factorize()
    test_df[col] -= test_df[col].min()


X = train_df.drop(['num_sold', 'date'], axis=1)
y = train_df['num_sold']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1) 
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1) 

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)



class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 1)  

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
        
input_dim = X_train.shape[1]
model = SimpleNN(input_dim)
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    num_samples = 0
    for data, target in train_loader:
        outputs = model(data)
        loss = criterion(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)  
        num_samples += data.size(0)  
    avg_loss = total_loss / num_samples  
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}")



model.eval()
with torch.no_grad():
    total_val_loss = 0
    num_val_samples = 0
    for data, target in val_loader:
        outputs = model(data)
        loss = criterion(outputs, target)
        total_val_loss += loss.item() * data.size(0)
        num_val_samples += data.size(0)
    avg_val_loss = total_val_loss / num_val_samples
    print(f"Validation Loss: {avg_val_loss:.4f}")



# Make Predictions on Test Data
model.eval()
with torch.no_grad():
    predictions = model(torch.tensor(test_df.drop(['id'], axis=1).values, dtype=torch.float32))
    predictions = predictions.numpy()  

test_df['target'] = predictions  
test_df[['id', 'target']].to_csv('submission.csv', index=False)
print("Submission file saved!")


