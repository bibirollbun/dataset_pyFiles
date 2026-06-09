import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import sklearn as sl

mpl.rcParams.update({
    "figure.figsize": (7, 5), 
    "figure.dpi": 128, 
    "font.size":14
})


# Сразу сделаем случайную подвыборку 0.8|0.2

path = "/kaggle/input/playground-series-s5e10/train.csv"
test_path = "/kaggle/input/playground-series-s5e10/test.csv"
df = pd.read_csv(path)

from sklearn.model_selection import train_test_split
df_train, df_val, y_train, y_val = train_test_split(
        df, df['accident_risk'], test_size=0.2)

df_train.reset_index(inplace=True)
df_val.reset_index(inplace=True)

df_train = df_train.drop(columns='index')
df_val = df_val.drop(columns='index')


df_train.info()


df_train.head()


# Описание данных
describe_df = df_train.describe().drop(columns=['id'])
# Числовые колонки
cols_num = describe_df.columns
# Категориальные колонки
cols_cat = list(set(df_train.columns).difference(set(cols_num)).difference(['id']))

# Выведем табличку
describe_df


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_validate

class LinRegClip(LinearRegression):
    def predict(self, X):
        return np.clip(LinearRegression.predict(self, X), 0, 1)

base_reg_model = LinRegClip()

X = df_train[cols_num].drop(columns='accident_risk')
y = df_train['accident_risk']

validator = cross_validate(base_reg_model, X, y, scoring=['r2', 'neg_root_mean_squared_error'], cv=5)

print(f"Среднее время обучения {validator.get('fit_time').mean(): .4f} c.")
print(f"Среднее время инференса {validator.get('score_time').mean(): .4f} c.")
print(f"Среднее качество R^2 {validator.get('test_r2').mean(): .4f}")
print(f"Среднее качество NRMSE {validator.get('test_neg_root_mean_squared_error').mean(): .4f}")


# Фит_предикт тренировки
base_reg_model.fit(X, y)
preds = base_reg_model.predict(X)


sns.histplot(data=df_train, x='accident_risk', 
             stat='density', bins='sturges', alpha=0.5)
sns.kdeplot(data=df_train, x='accident_risk', label='данные')
sns.histplot(x=preds, stat='density', 
             bins='sturges', alpha=0.5)
sns.kdeplot(x=preds, label='оценка л.м.')
plt.title('Распределения для данных/предсказаний')
plt.ylabel('Плотность распределения')
plt.xlabel('Риск инцидента')
plt.legend()
plt.show()


from scipy.stats import kstest, normaltest

errors = df_train['accident_risk'] - preds
sns.histplot(x = errors, 
             stat='density', 
             bins='sturges')
plt.show()

print(normaltest(errors))
print(kstest((errors - np.mean(errors)) / np.std(errors), 'norm'))


X_val = df_val[cols_num].drop(columns='accident_risk')
y_val = df_val['accident_risk']

# Фит_предикт тренировки
base_reg_model.fit(X, y)
preds_val = base_reg_model.predict(X_val)

print(f"Оценка R^2 метрики: {r2_score(y_val, preds_val)}")
print(f"Оценка MAE метрики: {mean_absolute_error(y_val, preds_val)}")


plt.scatter(preds_val, y_val, s=1, alpha=0.6)
plt.plot(preds_val, preds_val, c='red')
plt.title("График разброса данных относительно модели")
plt.xlabel("accident_risk")
plt.ylabel("Оценка модели")
plt.grid()
plt.tight_layout()
plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold

base_logreg_model = LogisticRegression()

X = df_train[cols_num].drop(columns='accident_risk')
y = df_train['accident_risk']
y_cat = (y > 0.3).replace({True: 1, False: 0})
kf = KFold(n_splits=5)
kf.get_n_splits(X)

scores = []
for i, (train_index, test_index) in enumerate(kf.split(X)):
    base_logreg_model.fit(X.loc[train_index], y_cat[train_index])
    preds = base_logreg_model.predict_proba(X.loc[test_index])
    scores.append(r2_score(y[test_index], preds[:, 1]))

print(f"Среднее качество R^2 {np.mean(scores): .4f}")


# Фит_предикт тренировки
base_logreg_model.fit(X, y_cat)
preds_log = base_logreg_model.predict_proba(X)[:, 1]


sns.histplot(data=df_train, x='accident_risk', 
             stat='density', bins='sturges', alpha=0.5)
sns.kdeplot(data=df_train, x='accident_risk', label='данные')
sns.histplot(x=preds_log, stat='density', 
             bins='sturges', alpha=0.5)
sns.kdeplot(x=preds_log, label='оценка лог.м.')
plt.title('Распределения для данных/предсказаний')
plt.ylabel('Плотность распределения')
plt.xlabel('Риск инцидента')
plt.legend()
plt.show()


fig, ax = plt.subplots(len(cols_num), 2, figsize=(10, 4 * len(cols_num)))
for idx, col in enumerate(cols_num):
    sns.histplot(data=df_train, x=col, stat='density', ax=ax[idx, 0])
    sns.boxplot(data=df_train, x=col, ax=ax[idx, 1])
plt.tight_layout()
plt.show()


for idx, col in enumerate(cols_cat):
    plt.figure(figsize=(5,3))
    sns.countplot(data=df_train, x=col)
    plt.tight_layout()
    plt.show()


for idx, col in enumerate(cols_cat):
    plt.figure(figsize=(5,3))
    sns.histplot(data=df_train, x='accident_risk', hue=col)
    plt.tight_layout()
    plt.show()


x = np.linspace(0, 1, 1000)
normal = lambda x, m, s: 1/(np.sqrt(2 * np.pi) * s) * np.exp(-(x - m)**2/(2 * s**2))
mean_risk = df_train['accident_risk'].mean()
std_risk = df_train['accident_risk'].std()

sns.histplot(data=df_train, x='accident_risk', stat='density', bins='sturges')
sns.kdeplot(data=df_train, x='accident_risk', label="Ядерная оценка")
sns.lineplot(x=x, 
             y=normal(x, mean_risk, std_risk),
             label="Оценка плотности")
plt.title("Плотность распределения риска")
plt.xlabel("Риск инцидента")
plt.ylabel("Вероятность возникновения")
plt.legend()
plt.grid()
plt.show()


corr = df_train[cols_num].corr('spearman')


mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


dummies_train = pd.get_dummies(df_train)

pd.concat(
    [
    df_train["lighting"], 
    dummies_train[dummies_train.columns[pd.Series(dummies_train.columns).str.startswith('light')]]
    ], 
    axis=1)


import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Создаем экземпляр OneHotEncoder
encoder = OneHotEncoder(sparse=False,
                        handle_unknown='ignore',
                        drop='first')

# Применяем кодирование к категориальным столбцам
encoded_data = encoder.fit_transform(df_train[cols_cat])

# Создаем новый DataFrame с закодированными значениями
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(cols_cat))

# Объединяем закодированные данные с исходным DataFrame
result = pd.concat([df_train, encoded_df], axis=1).drop(columns=cols_cat)

print(result)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_validate

class LinRegClip(LinearRegression):
    def predict(self, X):
        return np.clip(LinearRegression.predict(self, X), 0, 1)

reg_model = LinRegClip()

X = result.drop(columns=['id', 'accident_risk'])
y = result['accident_risk']

validator = cross_validate(reg_model, X, y, scoring=['r2', 'neg_root_mean_squared_error'], cv=5)

print(f"Среднее время обучения {validator.get('fit_time').mean(): .4f} c.")
print(f"Среднее время инференса {validator.get('score_time').mean(): .4f} c.")
print(f"Среднее качество R^2 {validator.get('test_r2').mean(): .4f}")
print(f"Среднее качество NRMSE {validator.get('test_neg_root_mean_squared_error').mean(): .4f}")


# Фит_предикт тренировки
reg_model.fit(X, y)
preds = reg_model.predict(X)


sns.histplot(data=df_train, x='accident_risk', 
             stat='density', bins='sturges', alpha=0.5)
sns.kdeplot(data=df_train, x='accident_risk', label='данные')
sns.histplot(x=preds, stat='density', 
             bins='sturges', alpha=0.5)
sns.kdeplot(x=preds, label='оценка л.м.')
plt.title('Распределения для данных/предсказаний')
plt.ylabel('Плотность распределения')
plt.xlabel('Риск инцидента')
plt.legend()
plt.show()


from scipy.stats import kstest, normaltest

errors = df_train['accident_risk'] - preds
sns.histplot(x = errors, 
             stat='density', 
             bins='sturges')
plt.show()

# Статистические тесты
print("Тест на нормальность:")
print(normaltest(errors))

print("\nТест Колмогорова-Смирнова:")
print(kstest((errors - np.mean(errors)) / np.std(errors), 'norm'))


from statsmodels.graphics.gofplots import qqplot

# Q-Q plot с использованием statsmodels
qqplot((errors-np.mean(errors))/np.std(errors), line='s', fit=True, 
       loc=np.mean(errors), scale=np.std(errors))
plt.title('Q-Q plot для ошибок')
plt.xlabel('Теоретические квантили')
plt.ylabel('Выборочные квантили')
plt.show()


# Применяем кодирование к категориальным столбцам
encoded_val = encoder.transform(df_val[cols_cat])

# Создаем новый DataFrame с закодированными значениями
encoded_df_val = pd.DataFrame(encoded_val, columns=encoder.get_feature_names_out(cols_cat))

# Объединяем закодированные данные с исходным DataFrame
result_val = pd.concat([df_val, encoded_df_val], axis=1).drop(columns=cols_cat)

print(result_val)


X_val = result_val.drop(columns=['id', 'accident_risk'])
y_val = result_val['accident_risk']

# Фит_предикт тренировки
reg_model.fit(X, y)
preds_val = reg_model.predict(X_val)

print(f"Оценка R^2 метрики: {r2_score(y_val, preds_val)}")
print(f"Оценка MAE метрики: {mean_absolute_error(y_val, preds_val)}")


plt.scatter(preds_val, y_val, s=1, alpha=0.6)
plt.plot(preds_val, preds_val, c='red')
plt.title("График разброса данных относительно модели")
plt.xlabel("accident_risk")
plt.ylabel("Оценка модели")
plt.grid()
plt.tight_layout()
plt.show()


coeffs_table = dict(zip(reg_model.feature_names_in_, reg_model.coef_))
coeffs_table['intercept'] = reg_model.intercept_
coeffs_table


# Создадим экземпляр данных для того, чтобы проверить работу модели
one_row = dict(zip(
    reg_model.feature_names_in_, 
    [
     [2],                 # 
     [0.1], 
     [60], 
     [1], 
     [False], 
     [False], 
     [False], 
     [False], 
     [False], 
     [False], 
     [False], 
     [False], 
     [False], 
     [False], 
     [False], 
     [False]
    ]   
))
one_row


reg_model.predict(pd.DataFrame(one_row))


from catboost import CatBoostRegressor

model_cb = CatBoostRegressor(iterations=200,
                             depth=3,
                             learning_rate=1,
                             l2_leaf_reg=1.2,
                             loss_function='RMSE',
                             silent=True)

model_cb.fit(X, y)
preds_cb = model_cb.predict(X_val)


print("RMSE", model_cb.best_score_)
print("R^2:", model_cb.score(X_val, y_val))
print("MAE:", mean_absolute_error(y_val, preds_cb))


dict(zip(model_cb.feature_names_, model_cb.feature_importances_))


plt.scatter(preds_cb, y_val, s=1, alpha=0.6)
plt.plot(preds_cb, preds_cb, c='red')
plt.title("График разброса данных относительно catboost")
plt.xlabel("accident_risk")
plt.ylabel("Оценка модели")
plt.grid()
plt.tight_layout()
plt.show()


# Фит_предикт тренировки
preds_cb_train = model_cb.predict(X)


sns.histplot(data=df_train, x='accident_risk', 
             stat='density', bins='sturges', alpha=0.5)
sns.kdeplot(data=df_train, x='accident_risk', label='данные')
sns.histplot(x=preds_cb_train, stat='density', 
             bins='sturges', alpha=0.5)
sns.kdeplot(x=preds_cb_train, label='оценка л.м.')
plt.title('Распределения для данных/предсказаний')
plt.ylabel('Плотность распределения')
plt.xlabel('Риск инцидента')
plt.legend()
plt.show()


sns.histplot(data=df_val, x='accident_risk', 
             stat='density', bins='sturges', alpha=0.5)
sns.kdeplot(data=df_val, x='accident_risk', label='данные')
sns.histplot(x=preds_cb, stat='density', 
             bins='sturges', alpha=0.5)
sns.kdeplot(x=preds_cb, label='оценка л.м.')
plt.title('Распределения для данных/предсказаний')
plt.ylabel('Плотность распределения')
plt.xlabel('Риск инцидента')
plt.legend()
plt.show()


# Создадим экземпляр данных для того, чтобы проверить работу модели
one_row = dict(zip(
    model_cb.feature_names_, 
    [
     [2],                 # 
     [0.1], 
     [60], 
     [1], 
     [False], 
     [False], 
     [False], 
     [False], 
     [False], 
     [False], 
     [False], 
     [False], 
     [False], 
     [False], 
     [False], 
     [False]
    ]   
))
one_row


model_cb.predict(pd.DataFrame(one_row))


from catboost import CatBoostRegressor


encoded_full = encoder.transform(df[cols_cat])
encoded_df_full = pd.DataFrame(encoded_full, 
                               columns=encoder.get_feature_names_out(cols_cat))
result_full = pd.concat([df, encoded_df_full], axis=1).drop(columns=cols_cat)

X_full = result_full.drop(columns=['id', 'accident_risk'])
y_full = result_full['accident_risk']

reg_model = LinRegClip()

model_cb = CatBoostRegressor(iterations=200,
                             depth=3,
                             learning_rate=1,
                             l2_leaf_reg=1.2,
                             loss_function='RMSE',
                             silent=True)

reg_model.fit(X_full, y_full)
model_cb.fit(X_full, y_full)


test_path = "/kaggle/input/playground-series-s5e10/test.csv"
df_test = pd.read_csv(test_path)
df_test.isna().sum()


encoded_test = encoder.transform(df_test[cols_cat])
encoded_df_test = pd.DataFrame(encoded_test, 
                               columns=encoder.get_feature_names_out(cols_cat))
result_test = pd.concat([df_test, encoded_df_test], axis=1).drop(columns=cols_cat)

X_test = result_test.drop(columns=['id'])

preds_cb = model_cb.predict(X_test)
preds_lr = reg_model.predict(X_test)


submission_cb = pd.DataFrame({'id': df_test.id, 'accident_risk': preds_cb})
submission_cb.to_csv("submission_cb.csv", index=False)


submission_lr = pd.DataFrame({'id': df_test.id, 'accident_risk': preds_lr})
submission_lr.to_csv("submission_lr.csv", index=False)


submission_ = pd.DataFrame({'id': df_test.id, 'accident_risk': preds_lr*0.15 + preds_cb*0.85})
submission_.to_csv("submission.csv", index=False)


# ====== Импорт ======
import os, math, warnings, json, pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

import matplotlib.pyplot as plt
from IPython.display import clear_output


# Константы
RANDOM_SEED = 42
VAL_SIZE    = 0.1
BATCH_SIZE  = 512
EPOCHS      = 100
LR          = 1e-3
WEIGHT_DECAY= 0.0
PATIENCE    = 10      # ранняя остановка
MIN_DELTA   = 1e-6
PLOT_EVERY  = 5       # график каждые N эпох

cols_num = list(cols_num)
if 'accident_risk' in cols_num:
    cols_num.remove('accident_risk')

# ====== Утилиты ======
def set_seed(seed=RANDOM_SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NumpyDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = None if y is None else torch.from_numpy(y.astype(np.float32)).reshape(-1, 1)
        
    def __len__(self): 
        return self.X.shape[0]
    
    def __getitem__(self, i):
        if self.y is None: 
            return self.X[i]
        return self.X[i], self.y[i]


class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=MIN_DELTA):
        self.patience = patience 
        self.min_delta = min_delta
        self.best_loss = math.inf 
        self.counter = 0
        self.best_state = None 
        self.best_epoch = -1
    
    def step(self, val_loss, model, epoch):
        improved = (self.best_loss - val_loss) > self.min_delta
        if improved:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.best_epoch = epoch
            return False
        self.counter += 1
        return self.counter >= self.patience


class MLP(nn.Module):
    # простая полносвязная сеть
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x): 
        return self.net(x)


train = df
test  = df_test

X_train_df = train.drop(columns=['accident_risk'])
y_train = train['accident_risk']
X_test_df  = test.copy()

# One-hot категориальные
full_cat = pd.get_dummies(X_train_df[cols_cat], drop_first=False)

# Масштабируем числовые
scaler = StandardScaler()
full_num = pd.DataFrame(
    scaler.fit_transform(X_train_df[cols_num]),
    columns=cols_num, index=X_train_df.index
)

# Финальная матрица признаков
X_train = pd.concat([full_num, full_cat], axis=1)
print(X_train.head())


# ====== Train/Val разделение ======
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_SEED
)
X_tr = np.array(X_tr)
X_val = np.array(X_val)
y_tr = np.array(y_tr)
y_val = np.array(y_val)

train_loader = DataLoader(NumpyDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(NumpyDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)


# ====== Модель и обучение ======
model = MLP(in_dim=X_tr.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

train_losses, val_losses = [], []
es = EarlyStopping()


for epoch in range(1, EPOCHS+1):
    # --- train ---
    model.train()
    tr_sum = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        tr_sum += loss.item() * xb.size(0)
    tr_loss = tr_sum / len(train_loader.dataset)

    # --- valid ---
    model.eval()
    val_sum = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            val_sum += loss.item() * xb.size(0)
    v_loss = val_sum / len(val_loader.dataset)

    train_losses.append(tr_loss); val_losses.append(v_loss)

    if epoch == 1 or epoch % PLOT_EVERY == 0:
        clear_output(wait=True)
        print(f"Эпоха {epoch}/{EPOCHS} | train: {tr_loss:.6f} | valid: {v_loss:.6f}")
        plt.figure(figsize=(7,4))
        plt.plot(range(1, len(train_losses)+1), train_losses, label="train")
        plt.plot(range(1, len(val_losses)+1),   val_losses,   label="valid")
        plt.xlabel("Эпоха")
        plt.ylabel("Loss")
        plt.yscale('log')
        plt.title("Кривые обучения")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

    if es.step(v_loss, model, epoch):
        print(f"Ранняя остановка на эпохе {epoch}. Лучшая: {es.best_epoch+1}, val_loss={es.best_loss:.6f}")
        model.load_state_dict(es.best_state)
        break

print(f"Финальная valid loss: {val_losses[-1]:.6f} (лучшая: {es.best_loss:.6f} на эпохе {es.best_epoch+1})")


# ====== Простая оценка на валидации ======
model.eval()
with torch.no_grad():
    val_logits = model(torch.from_numpy(X_val.astype(np.float32)).to(device)).cpu().numpy().ravel()


val_preds = val_logits
mae  = metrics.mean_absolute_error(y_val, val_preds)
rmse = metrics.mean_squared_error(y_val, val_preds, squared=False)
r2   = metrics.r2_score(y_val, val_preds)
print(f"Валидация (regression): MAE={mae:.4f} | RMSE={rmse:.4f} | R2={r2:.4f}")


test_cat = pd.get_dummies(X_test_df[cols_cat], drop_first=False)
test_num = pd.DataFrame(
    scaler.transform(X_test_df[cols_num]),
    columns=cols_num, index=X_test_df.index
)

X_test = np.array(pd.concat([test_num, test_cat], axis=1))

# ====== Предсказания на test ======
with torch.no_grad():
    test_logits = model(torch.from_numpy(X_test.astype(np.float32)).to(device)).cpu().numpy().ravel()

out = pd.DataFrame({"id":df_test.id, "accident_risk": test_logits})
out.to_csv("predictions.csv", index=False)




