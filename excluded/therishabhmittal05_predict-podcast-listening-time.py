import torch
from torch import nn, optim
import numpy as np
import pandas as pd


dataset = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")


dataset.head(5)


dataset.info()
dataset.isna().sum()


from sklearn.model_selection import train_test_split
X = dataset.drop(["id","Listening_Time_minutes"], axis=1)
y = dataset["Listening_Time_minutes"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train.shape, y_train.shape


from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(sparse_output=False,handle_unknown='ignore'))
])

numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

preprocessor = ColumnTransformer([
    ('categorical', categorical_pipeline, categorical_cols),
    ('numeric', numeric_pipeline, numeric_cols)
], remainder='passthrough')

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


type(X_train), type(X_test), type(y_train), type(y_test)


X_train.shape, y_train.shape


import seaborn as sb
import matplotlib.pyplot as plt
numeric_X = dataset.select_dtypes(include=['number'])
c_mat = numeric_X.corr()
sb.heatmap(c_mat, vmax = .8, square = True)
plt.show()


device = "cuda" if torch.cuda.is_available() else "cpu"
device


X_train = torch.from_numpy(X_train).type(torch.float).to(device)
X_test = torch.from_numpy(X_test).type(torch.float).to(device)
y_train = torch.from_numpy(y_train.to_numpy()).type(torch.float).to(device)
y_test = torch.from_numpy(y_test.to_numpy()).type(torch.float).to(device)


type(X_train), type(X_test), type(y_train), type(y_test), X_train.shape, y_train.shape


class Listening_time_model(nn.Module):
  def __init__(self):
    super().__init__()
    self.regressor = nn.Sequential(
        nn.Linear(in_features=X_train.shape[1], out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=1)
    )

  def forward(self, x):
    return self.regressor(x)


model = Listening_time_model().to(device)
model


from torchsummary import summary
summary(model, input_size=(X_train.shape[1],))


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


y_log = model(X_test)[:5]
y_log


epochs = 5000

for epoch in range(epochs):
  model.train()
  y_pred = model(X_train).squeeze()
  loss = torch.sqrt(criterion(y_pred, y_train))
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  model.eval()
  with torch.inference_mode():
    test_pred = model(X_test).squeeze()
    test_loss = torch.sqrt(criterion(test_pred, y_test))
  if epoch%100 == 0:
    print(f"Epoch: {epoch} | Train loss: {loss} | Test Loss: {test_loss}")


sub_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
sub_df.head(5)


sub_df.info(), sub_df.isna().sum()


sub_test = preprocessor.transform(sub_df)


sub_test = torch.from_numpy(sub_test).type(torch.float).to(device)


sub_test.shape


model.eval()
with torch.no_grad():
  pred = model(sub_test).squeeze()


submission = pd.DataFrame(
    {
        "id": sub_df["id"],
        "Listening_Time_minutes": pred.cpu().numpy()
    }
)
submission.to_csv("submission01.csv", index=False)


submission.head()




