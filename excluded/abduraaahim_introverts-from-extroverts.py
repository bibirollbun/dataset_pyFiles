import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv',index_col=0)
test=pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv',index_col=0)
sub = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv',index_col=0)

print(train.shape)
train.head(10)


train.info()


train.describe()


train.describe(include='O')


print(train.isna().sum()[train.isna().sum()>0])


nan_columns = train.isna().sum()[train.isna().sum()>0].index.to_list()
print(nan_columns)


num_nan_cols = train[nan_columns].select_dtypes(include='number').columns.to_list()
cat_nan_cols = train[nan_columns].select_dtypes(include='object').columns.to_list()


# Column indexes (based on num_features)
# ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
time_alone_ix = 0
event_attend_ix = 1
going_outside_ix = 2
friends_ix = 3
post_freq_ix = 4

class SocialFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_extroversion_score=True):
        self.add_extroversion_score = add_extroversion_score

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        alone_vs_social = X[:, time_alone_ix] / (X[:, event_attend_ix] + 1)
        active_friend_ratio = X[:, post_freq_ix] / (X[:, friends_ix] + 1)
        
        if self.add_extroversion_score:
            extro_score = (
                X[:, event_attend_ix] +
                X[:, going_outside_ix] +
                X[:, friends_ix] -
                X[:, time_alone_ix]
            )
            return np.c_[X, alone_vs_social, active_friend_ratio, extro_score]
        else:
            return np.c_[X, alone_vs_social, active_friend_ratio]


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('feature-engineering', SocialFeatureAdder(add_extroversion_score=True)),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, num_nan_cols),
    ('cat', categorical_pipeline, cat_nan_cols)
])


X = train.drop('Personality',axis=1)
y = train['Personality']


label_encoder = LabelEncoder()

X_prepared = preprocessor.fit_transform(X)
y_prepared = label_encoder.fit_transform(y)


X_train,X_test,y_train,y_test = train_test_split(
    X_prepared,
    y_prepared,
    test_size=0.2,
    random_state=42
)


from sklearn.ensemble import RandomForestClassifier

forest_model = RandomForestClassifier(
    n_estimators=100, 
    random_state=42
)

forest_model.fit(X_train,y_train)
y_hat = forest_model.predict(X_test)
print(f"accuracy score: {accuracy_score(y_test,y_hat)}")


from catboost import CatBoostClassifier
cat_features = ['Stage_fear', 'Drained_after_socializing']

cat_model = CatBoostClassifier(verbose=100)
cat_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = cat_model.predict(X_test)
print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")


import torch
from torch import nn 
from torch.utils.data import Dataset, DataLoader


class PersonalityDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = PersonalityDataset(X_train,y_train)
test_dataset = PersonalityDataset(X_test,y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32,  shuffle=True)


import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)


model = Model(input_dim=12)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# Training loop
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch, y_batch

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
    
    acc = correct / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")


test_prepared = preprocessor.transform(test)
test_prepared_transform = torch.tensor(test_prepared, dtype=torch.float32)


model.eval()
with torch.no_grad():
    logits = model(test_prepared_transform)
    preds = torch.argmax(logits, dim=1)
    preds_np = preds.cpu().numpy()


yhat = cat_model.predict(test_prepared)
sub.drop('Personality',axis=1,inplace=True)


sub['Personality_torch'] = preds_np
sub['Personality_catboost'] = yhat
sub.head(50)


label_map = {0: "Extrovert", 1: "Introvert"}
preds_labels = [label_map[p] for p in preds_np]


submission = pd.DataFrame({
    'id':test.index,
    'Personality':preds_labels
})


submission.to_csv('sub.csv',index=False)

