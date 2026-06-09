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


df = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv")
test_df = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv")


from gensim.models import Word2Vec


sentences  =  [sentence.split() for sentence in df['target'] ] \
            + [sent.split() for sent in df["anchor"] ] \
            +[sentence.split() for sentence in test_df['target'] ] \
            + [sentence.split() for sentence in test_df['anchor'] ]            



model = Word2Vec(
    sentences=sentences,
    vector_size=100,  # размер вектора
    window=5,         # окно контекста
    min_count=5,      # минимальная частота слова
    workers=1,        # количество потоков
    sg=1,             # 1 для skip-gram, 0 для CBOW
    epochs=10         # количество эпох обучения
)


def get_text_vector(text, model, vector_size=100):
    if not isinstance(text, str):
        return np.zeros(vector_size)
    
    words = text.split()
    vectors = []
    for word in words:
        try:
            vectors.append(model.wv[word])
        except KeyError:
            vectors.append(np.zeros(vector_size))
    
    if len(vectors) == 0:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

y = df["score"].values
unique_values =  pd.concat([df["context"], test_df["context"]]).unique().tolist()
encoding_dict = {v: i for i, v in enumerate(unique_values)}


df["context"] = df['context'].apply(lambda x: encoding_dict[x] )

df['anchor_vector'] = df['anchor'].apply(
    lambda x: get_text_vector(x,model) )

df['target_vector'] = df['target'].apply(
    lambda x: get_text_vector(x,model) )

new_df = df.drop(columns = [ "anchor", "target", "id","score" ])


idies = test_df["id"]
test_df["context"] = test_df['context'].apply(lambda x: encoding_dict[x] )

test_df['anchor_vector'] = test_df['anchor'].apply(
    lambda x: get_text_vector(x,model) )

test_df['target_vector'] = test_df['target'].apply(
    lambda x: get_text_vector(x,model) )

new_test_df = test_df.drop(columns = [ "anchor", "target", "id" ])




X_anchor = np.vstack(new_df['anchor_vector'].values)
X_target = np.vstack(new_df['target_vector'].values)
X_context = np.vstack(new_df['context'].values)
# Объединяем признаки
X = np.hstack([X_anchor, X_target,X_context])

X_test_anchor = np.vstack(new_test_df['anchor_vector'].values)
X_test_target = np.vstack(new_test_df['target_vector'].values)
X_test_context = np.vstack(new_test_df['context'].values)
X_test_df = np.hstack([X_test_anchor, X_test_target, X_test_context])


from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
# 1. Создаем кастомный Dataset
class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels) if len(labels.shape) == 1 else torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Разделяем на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем Dataset и DataLoader
train_dataset = MyDataset(X_train, y_train)
test_dataset = MyDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3. Модель (улучшенная версия)
class MyModel(nn.Module):
    def __init__(self, input_size):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 5)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 4. Инициализация модели
input_size = X_train.shape[1]  # автоматическое определение размера входа
output_size = len(torch.unique(torch.Tensor(y))) if len(y.shape) == 1 else y.shape[1]
model = MyModel(input_size)

# 5. Определяем loss и optimizer
criterion = nn.CrossEntropyLoss() if output_size > 1 else nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)




from tqdm import tqdm


# 6. Цикл обучения
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Валидация
    model.eval()
    test_loss = 0.0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            
            if output_size > 1:  # для классификации
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
    
    # Печатаем статистику
    train_loss /= len(train_loader)
    test_loss /= len(test_loader)
    
    print(f'Epoch {epoch+1}/{num_epochs}')
    print(f'Train Loss: {train_loss:.4f}')
    if output_size > 1:
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
    else:
        print(f'Test Loss: {test_loss:.4f} (Regression)')


from torch import vmap
unique_classes = torch.tensor(list(set(y))).tolist()
a =   torch.argmax( torch.softmax( (model(torch.FloatTensor(X_test_df))) , dim=0)  , dim = 1).tolist()
answer = list( map(lambda x: unique_classes[x], a) )


ans = pd.DataFrame({
    'id': idies,
    'score': answer
})


ans.to_csv("submission.csv", index = False )

