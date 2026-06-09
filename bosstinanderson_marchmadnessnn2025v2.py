import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class SimpleNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train_network(model, train_loader, criterion, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device) #move data to device
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

def evaluate_network(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1) #get prediction
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total:.2f}%')

def custom_cross_entropy(input, target):
    """
    Custom cross-entropy loss function.

    Args:
        input (torch.Tensor): Logits from the model (batch_size, num_classes).
        target (torch.Tensor): Ground truth labels (batch_size,) or (batch_size, num_classes).

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Your custom loss calculation here...
    # Ensure the output is a scalar tensor.
    # example:
    # log_probs = torch.log_softmax(input, dim=1)
    # loss = -torch.mean(torch.sum(target * log_probs, dim=1))

    return loss

# Example Usage
if __name__ == "__main__":
    # Hyperparameters
    input_size = 64*181
    hidden_size = 181
    output_size = 64  # Example: 5 classes for classification
    learning_rate = 0.001
    epochs = 50
    batch_size = 64

    # Sample Data (replace with your actual data)
    import numpy as np
    #X_train = np.random.rand(1000, input_size).astype(np.float32)
    X_train = []
    years = list(range(2003,2025)) 
    years.remove(2020)
    print(years)
    for y in years:
        X_trainyear = []
        for i in ids[y]:
            X_trainyear += [big[y][i][s] for s in syss]
        X_train.append(X_trainyear)
        print(f'year is {y} X_trainyearlen is {len(X_trainyear)}')
    y_train = np.random.randint(0, output_size, len(X_train)).astype(np.int64)
    X_test = np.random.rand(200, input_size).astype(np.float32)
    y_test = np.random.randint(0, output_size, 200).astype(np.int64)
    #print(f"Xtrain is {X_train} \n ytrain is {y_train}")

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train)
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the network
    model = SimpleNeuralNet(input_size, hidden_size, output_size).to(device)

    criterion = nn.CrossEntropyLoss() #for multiclass classification. Use nn.BCELoss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the network
    train_network(model, train_loader, criterion, optimizer, epochs, device)

    # Evaluate the network
    evaluate_network(model, test_loader, device)

    


print(len(X_train))
print([len(y) for y in X_train])


import pandas as pd
import numpy as np
dfres = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv")
dfres.loc[-1] = [2021, 137, 1332, 71,1433,70,"N",0]  # adding a row, covid game
dfres.index = dfres.index + 1  # shifting index
dfres = dfres.sort_index()  # sorting by index
dfres = dfres[np.where(dfres["Season"] == 2021, dfres["DayNum"] >= 137, dfres["DayNum"] >= 136)]
#dfres = dfres[dfres["DayNum"] >= 136]
dfres = dfres[["Season","WTeamID","LTeamID"]]
df = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MMasseyOrdinals.csv")
#print(df)

df = df[df["RankingDayNum"] >= 133]

syss = df["SystemName"].unique()

print(syss)
    #X_train = np.random.rand(1000, input_size).astype(np.float32)
    #y_train = np.random.randint(0, output_size, 1000).astype(np.int64)
    #X_test = np.random.rand(200, input_size).astype(np.float32)
    #y_test = np.random.randint(0, output_size, 200).astype(np.int64)


big = dict()
ids = dict()
years = list(range(2003,2025)) 
years.remove(2020)
print(years)
for y in years:
    big[y] = dict()
    dfy = dfres[dfres['Season'] == y] #only do teams in tourney
    teamGameList[y] = []
    dfy = dfy.reset_index()
    for index, row in dfy.iterrows():
        #print(row)
        teamGameList[y].append(int(row['WTeamID']))
        teamGameList[y].append(int(row['LTeamID']))
    ids[y] = list(set(teamGameList[y]))
    for i in ids[y]:
        big[y][i] = dict()
        dfi = df[df['TeamID'] == i]
        for s in syss:
            dfs = dfi[dfi['SystemName'] == s]
            if not dfs[dfs['Season'] == y].empty:
                big[y][i][s] = np.float32(dfs[dfs['Season'] == y].iat[0,4])
            else:
                big[y][i][s] = np.float32(999)


teamGameList = dict() #feeds team_ratings, gets fed into loss fxn
years = list(range(2003,2025)) 
years.remove(2020)
for s in years:
    teamGameList[s] = []
    dfs = dfy[dfy['Season'] == s]
    dfs = dfs.reset_index()
    for index, row in dfs.iterrows():
        #print(row)
        teamGameList[s].append(int(row['WTeamID']))
        teamGameList[s].append(int(row['LTeamID']))

#teamList[s]
#print(teamGameList)
num_games = 63
brier_scores = []

for i in range(num_games):
    team1_ratings = team_ratings[i * 2] # team ratings are in pairs, winner first
    team2_ratings = team_ratings[i * 2 + 1] # team ratings are in pairs.
    predicted_probability = prediction_function(team1_ratings, team2_ratings)
    brier_score = (predicted_probability - 1) ** 2
    brier_scores.append(brier_score)

return torch.mean(torch.stack(brier_scores))

#y_train is built into teamGameList since the odd entries are winners


#print(big[2023])
print(list(set(teamGameList[2023])))
print(big[2023][1281])


print(len(syss))


def predMaker(team1score,team2score):
    return (0.5 + 0.5*(team1score-team2score)/(team1score+team2score))
    


import pandas as pd
dfres = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv")
dtest = dfres[dfres["Season"] == 2021]
dtest.loc[-1] = [2021, 137, 1332, 71,1433,70,"N",0]  # adding a row, covid game
dtest.index = dtest.index + 1  # shifting index
dtest = dtest.sort_index()  # sorting by index
for index,row in dtest.iterrows():
    print(row)


