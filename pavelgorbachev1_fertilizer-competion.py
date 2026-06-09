# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")


import seaborn as sns

# Задаем желаемый порядок категорий
crop_order = sorted(df['Crop Type'].unique())  # или вручную: ['Wheat', 'Rice', ...]

fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(12, 30))
plt.subplots_adjust(hspace=0.7)

colors = [
    "#1f77b4",  # синий
    "#ff7f0e",  # оранжевый
    "#2ca02c",  # зеленый
    "#d62728",  # красный
    "#9467bd",  # фиолетовый
    "#8c564b",  # коричневый
    "#e377c2",  # розовый
]
for ind, fert_name in enumerate(df["Fertilizer Name"].unique()):
    subset = df[df["Fertilizer Name"] == fert_name]
    sns.histplot(
        data=subset,
        x="Crop Type",
        ax=axes[ind],
        color=colors[ind],
        discrete=True,  # Для категориальных данных
        shrink=0.8,
        stat='percent'# Уменьшает ширину столбцов  # Фиксируем порядок категорий
    )
    axes[ind].set_title(f"Crop types for {fert_name}")
    axes[ind].tick_params(axis='x', rotation=45)
    
    for p in axes[ind].patches:
        height = p.get_height()
        axes[ind].text(
            p.get_x() + p.get_width()/2.,  # x-позиция
            height + 0.5,                  # y-позиция (немного выше столбца)
            f'{height:.1f}%',              # текст с процентом
            ha='center',                   # выравнивание по центру
            va='bottom',                   # выравнивание по низу текста
            fontsize=9
        )



def add_feautures(X):
    X["quadrat_temp_moist_plus"] = X["Temparature"]**2 + X["Moisture"]**2
    X["Nitr_potas_quad"] = X["Nitrogen"]**2 + X["Potassium"]**2
    X["Nitr_phosthorus_quad"] = X["Phosphorous"]**2 + X["Nitrogen"]**2
    X["Potas_phosphorus_quad"] = X["Potassium"]**2 + X["Phosphorous"]**2
    X["temp_hum_moist"] = X["Temparature"] * X["Humidity"] * X["Moisture"] 
    X["Pot_phosph_nitr"] = X["Potassium"] * X["Phosphorous"] * X["Nitrogen"] 
    
def drop_cat_feautures(X):
    categorial = [col for col in X.columns if X[col].dtype == "object" ]  
    return X.drop( columns = categorial )
    


target = df["Fertilizer Name"]
X = df.drop( columns = ["Fertilizer Name","id"])


# Находим только категориальные колонки
categorical_cols = [col for col in X.columns if X[col].dtype == "object"]

# Создаем OneHotEncoder только для категориальных колонок
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Применяем кодирование только к категориальным колонкам
X_encoded = encoder.fit_transform(X[categorical_cols])

# Создаем DataFrame из закодированных данных
X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_cols))

# Объединяем с числовыми колонками
X_numeric = X.drop(columns=categorical_cols)

X = pd.concat([X_numeric.reset_index(drop=True), X_encoded_df.reset_index(drop=True)], axis=1)

name2ind = {name : ind for ind,name in enumerate(target.unique()) }
ind2name = { ind : name for name,ind in name2ind.items() } 
# enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
# y_f = enc.fit_transform(y.values.reshape(-1,1))
# y_final  = pd.DataFrame(y_f, columns=enc.get_feature_names_out())
y = target.apply(lambda x: name2ind[x])


target = df["Fertilizer Name"]
data = df.drop( columns = ["Fertilizer Name","id"])

name2ind = {name : ind for ind,name in enumerate(target.unique()) }
ind2name = { ind : name for name,ind in name2ind.items() } 



y = target.apply(lambda x: name2ind[x])
X = drop_cat_feautures(data)

add_feautures(X)


tree = RandomForestClassifier()
tree.fit(X,y)


feature_names = [f"{col}" for col in X.columns ]
forest_importances = pd.Series(tree.feature_importances_, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar()
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()


X.shape
target.unique()


import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

# 1. Определение класса Dataset
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)  # Для классификации
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 2. Модель (модифицированная версия с BatchNorm и Dropout)
class DenseModel(nn.Module):
    def __init__(self, len_input):
        super().__init__()
        self.L1 = nn.Sequential(
            nn.Linear(len_input, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.L2 = nn.Sequential(
            nn.Linear(512 + len_input, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.L3 = nn.Sequential(
            nn.Linear(512 + 512 + len_input, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.L4 = nn.Sequential(
            nn.Linear(512 + 512 + 512 + len_input, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.L5 = nn.Linear(512 * 4 + len_input, 7)  # Учитываем все skip-connections
        

    def forward(self, x):
        
        out1 = self.L1(x)
        
        concatenated1 = torch.cat([x, out1], dim=1)
       
        out2 = self.L2(concatenated1)

        concatenated2 = torch.cat([concatenated1, out2], dim=1)
        
        out3 = self.L3(concatenated2)
        
        concatenated3 = torch.cat([concatenated2, out3], dim=1)
        
        out4 = self.L4(concatenated3)
        
        concatenated = torch.cat([concatenated3,out4], dim=1)
        
        return self.L5(concatenated)


from sklearn.model_selection import train_test_split


features_train, features_val, labels_train , labels_val = train_test_split(X.values,y.values,train_size = 0.9)


train_dataset = CustomDataset(features_train, labels_train)
val_dataset = CustomDataset(features_val, labels_val)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# 4. Инициализация модели, оптимизатора и функции потерь
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DenseModel(len_input=features_train.shape[1]).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()


print(f"Features shape: {features_train.shape}")
print(f"Labels shape: {labels_train.shape}")
print(f"Features shape: {features_val.shape}")
print(f"Labels shape: {labels_val.shape}")


num_epochs = 10
best_val_acc = 0

for epoch in  range(num_epochs) :
    # Тренировка
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        total_train += target.size(0)
        correct_train += predicted.eq(target).sum().item()
    
    train_acc = 100. * correct_train / total_train
    
    # Валидация
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            
            loss = criterion(output, target)
            
            val_loss += loss.item()
            _, predicted = output.max(1)
            total_val += target.size(0)
            correct_val += predicted.eq(target).sum().item()
    
    val_acc = 100. * correct_val / total_val
    
    # Сохранение лучшей модели
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
    
    print(f'Epoch {epoch+1}/{num_epochs} | '
          f'Train Loss: {train_loss/len(train_loader):.4f} | '
          f'Train Acc: {train_acc:.2f}% | '
          f'Val Loss: {val_loss/len(val_loader):.4f} | '
          f'Val Acc: {val_acc:.2f}%')

print(f'Best Validation Accuracy: {best_val_acc:.2f}%')


!pip install optuna


!pip install pytorch-tabnet



from sklearn.model_selection import train_test_split
import torch


DEVICE = torch.cuda.is_available()
DEVICE


features_train, features_val, labels_train , labels_val = train_test_split(X.values,y.values,train_size = 0.9)
# features_train = torch.FloatTensor(features_train).to('cuda')
# features_val = torch.FloatTensor(features_val).to('cuda')
# labels_train = torch.FloatTensor(labels_train).to('cuda')
# labels_val = torch.FloatTensor(labels_val).to('cuda')


import optuna
from optuna.pruners import MedianPruner
import optuna.visualization as vis

from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score

def objective(trial):
    # Параметры для оптимизации
    params = {
        "n_d": trial.suggest_int("n_d", 8, 12),
        "n_a": trial.suggest_int("n_a", 8, 12),
        "n_steps": trial.suggest_int("n_steps", 8, 10),
        "gamma": trial.suggest_float("gamma", 1.5, 2.0),
        "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-5, 1e-1, log=True),
        "optimizer_params": {"lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True)},
        "mask_type": trial.suggest_categorical("mask_type", ["sparsemax", "entmax"])
    }

    model = TabNetClassifier(**params, device_name='cuda')
    model.fit(
        features_train, labels_train,
        eval_set=[(features_val, labels_val)],
        max_epochs=10,
        patience = 5,
        batch_size = 1000
    )

    preds = model.predict(features_val)
    accuracy = accuracy_score(labels_val, preds)
    return accuracy


study = optuna.create_study(
    direction="maximize",
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
)
study.optimize(objective, n_trials=10)  # 10 попыток подбора параметров

# Лучшие параметры
print("Best trial:")
trial = study.best_trial
print(f"  Accuracy: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")


# График оптимизации
vis.plot_optimization_history(study)

# Важность гиперпараметров
vis.plot_param_importances(study)

# Зависимость accuracy от параметров
vis.plot_slice(study, params=["n_d", "n_a", "lr"])


best_params = study.best_params
best_params 



best_params = study.best_params

model = TabNetClassifier(**best_params)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    max_epochs=10,
    patience=20,
    verbose=1,
)


test_data = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
  
id_pred = test_data["id"]

t = test_data.drop( columns = ["id"])

# t = drop_cat_feautures(t)
# add_feautures(t)


# Применяем кодирование только к категориальным колонкам
X_encoded = encoder.fit_transform(t[categorical_cols])

# Создаем DataFrame из закодированных данных
X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_cols))

# Объединяем с числовыми колонками
X_numeric = t.drop(columns=categorical_cols)

t = pd.concat( [ X_numeric.reset_index(drop=True), X_encoded_df.reset_index(drop=True) ], axis=1)

#add_feautures(t)



predictions = tree.predict(t)



t.shape


torch.cuda.empty_cache()
#t =  torch.FloatTensor(t.values).to(device)
ans = torch.tensor([], device=device) 
model.eval()
chunk_size = 100
for i in range(0, len(test_data), chunk_size):
    chunk = t[i:i+chunk_size]
    with torch.no_grad():
        pred = model(chunk.to(device))
        ans = torch.cat([ans, pred], dim=0)
        


predictions = torch.argmax(ans, dim = 1)
predictions[9].item()


predictions = [ind2name[x.item()] for x in predictions]


with open("submission.csv" , "w") as f:
    f.write(f"id,Fertilizer Name")
    for i in range(len(t)):
        f.write("\n")
        f.write( f"{ id_pred.iloc[i] },{  predictions[i]  }" )


from catboost import CatBoostClassifier, Pool

cat_features = X.columns[X.dtypes == 'object'].tolist()

for col in cat_features:
    X[col] = X[col].astype('category')
model = CatBoostClassifier(iterations=1, learning_rate=0.1, depth=8,cat_features=cat_features)
model.fit(X, y , cat_features=cat_features )

test_data = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")

for col in cat_features:
    test_data[col] = test_data[col].astype('category')
    
id_pred = test_data["id"]
t = test_data.drop( columns = ["id"])
prediction_pool = Pool(t, cat_features=cat_features)
predictions = model.predict( prediction_pool )


with open("submission.csv" , "w") as f:
    f.write(f"id,Fertilizer Name")
    for i in range(len(t)):
        f.write("\n")
        f.write( f"{ id_pred.iloc[i] },{ predictions[i][0] }" )
        
    


