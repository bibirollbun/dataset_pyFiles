import os

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

from tqdm import tqdm

from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    f1_score
)



# Путь к train/test таблицам
train_path = "/kaggle/input/petfinder-pawpularity-score/train.csv"
test_path = "/kaggle/input/petfinder-pawpularity-score/test.csv"

# Название колонки с ID картинки
IMAGE_ID_COL = "Id"     # поменяй, если у тебя другой столбец

# Название колонки с таргетом
TARGET_COL = "Pawpularity"         # сюда поставь имя своей целевой переменной

# Тип задачи: "regression" или "classification"
TASK_TYPE = "regression"      # если классификация, поставь "classification"

# Папки с картинками
train_images_dir = "/kaggle/input/petfinder-pawpularity-score/train"
test_images_dir = "/kaggle/input/petfinder-pawpularity-score/test"

# Расширение картинок
# Если в таблице image_id уже содержит ".jpg", поставь IMAGE_EXT = ""
IMAGE_EXT = ".jpg"

# Имя файла сабмита
submission_path = "submission.csv"



train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Train columns:", train.columns.tolist())
print("Test columns:", test.columns.tolist())
# Если в IMAGE_ID_COL уже лежит something.jpg, можно сделать IMAGE_EXT = "" и не дописывать расширение.


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Using device:", device)



import torchvision.models as models
import torch
import torch.nn as nn

# Путь к сохранённым весам (проверь в браузере Kaggle, как называется твой датасет и файл)
WEIGHTS_PATH = "/kaggle/input/resnet50/resnet50_imagenet.pth"

# 1. Создаём архитектуру ResNet50 БЕЗ предобученных весов
resnet = models.resnet50(weights=None)  # для старой версии: pretrained=False

# 2. Загружаем state_dict из файла
state_dict = torch.load(WEIGHTS_PATH, map_location=device)

# 3. Кладём веса в модель
# Если структура совпадает — этого достаточно:
resnet.load_state_dict(state_dict)

# Если вдруг будет ошибка по ключам — можно так:
# resnet.load_state_dict(state_dict, strict=False)

# 4. Обрезаем последний полносвязный слой,
#    чтобы модель возвращала эмбеддинг (2048 чисел)
resnet.fc = nn.Identity()

# 5. Отправляем на GPU/CPU
resnet.to(device)

# 6. Включаем eval-режим (инференс, без обучения)
resnet.eval()



image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])



def get_normalized_embedding(image_path):
    """
    1. Загружает картинку по пути image_path.
    2. Применяет стандартные трансформации под ResNet.
    3. Прогоняет через ResNet на GPU/CPU.
    4. Возвращает L2-нормализованный вектор (numpy, shape = (2048,)).
    """
    # Загружаем изображение и приводим к RGB
    image = Image.open(image_path).convert("RGB")
    
    # Применяем трансформации
    image = image_transform(image)
    
    # Добавляем batch dimension: [C, H, W] -> [1, C, H, W]
    image = image.unsqueeze(0)
    
    # Переносим на устройство
    image = image.to(device)
    
    # Прогоняем через модель
    with torch.no_grad():
        emb_tensor = resnet(image)   # shape: [1, 2048]
    
    # Переводим в numpy и убираем размер батча
    emb = emb_tensor.cpu().numpy().reshape(-1)  # shape: (2048,)
    
    # L2-нормализация
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    
    return emb



train_embeddings = []

for image_id in tqdm(train[IMAGE_ID_COL], desc="Train embeddings"):
    # Если в image_id уже есть ".jpg", используем IMAGE_EXT = ""
    img_name = str(image_id) + IMAGE_EXT
    img_path = os.path.join(train_images_dir, img_name)
    
    emb = get_normalized_embedding(img_path)
    train_embeddings.append(emb)

train_embeddings = np.array(train_embeddings)

print("Train embeddings shape:", train_embeddings.shape)



test_embeddings = []

for image_id in tqdm(test[IMAGE_ID_COL], desc="Test embeddings"):
    img_name = str(image_id) + IMAGE_EXT
    img_path = os.path.join(test_images_dir, img_name)
    
    emb = get_normalized_embedding(img_path)
    test_embeddings.append(emb)

test_embeddings = np.array(test_embeddings)

print("Test embeddings shape:", test_embeddings.shape)



emb_dim = train_embeddings.shape[1]
emb_cols = [f"f_{i}" for i in range(emb_dim)]

train_emb_df = pd.DataFrame(train_embeddings, columns=emb_cols)
test_emb_df = pd.DataFrame(test_embeddings, columns=emb_cols)

train_full = pd.concat([train.reset_index(drop=True), train_emb_df], axis=1)
test_full = pd.concat([test.reset_index(drop=True), test_emb_df], axis=1)

print("Train full shape:", train_full.shape)
print("Test full shape:", test_full.shape)



# Считаем средний эмбеддинг по train
mean_emb = train_embeddings.mean(axis=0)

# L2-нормализуем центр
mean_norm = np.linalg.norm(mean_emb)
if mean_norm > 0:
    mean_emb = mean_emb / mean_norm

def cosine_similarity(a, b):
    """
    Косинусная похожесть между двумя векторами a и b.
    a и b могут быть уже нормализованы, но мы перестрахуемся.
    """
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    value = np.dot(a, b) / denom
    return value

# Добавляем фичу для train
train_cos_sim = []
for emb in train_embeddings:
    sim = cosine_similarity(emb, mean_emb)
    train_cos_sim.append(sim)

train_full["cos_sim_to_mean"] = train_cos_sim

# Добавляем фичу для test
test_cos_sim = []
for emb in test_embeddings:
    sim = cosine_similarity(emb, mean_emb)
    test_cos_sim.append(sim)

test_full["cos_sim_to_mean"] = test_cos_sim

print("Added feature cos_sim_to_mean")



#Тут важное место: как раз про категориальные фичи — их оставляем строками и отмечаем индексы колонок, а не значения.
# Эмбеддинги (все колонки, начинающиеся с "f_")
embedding_features = []
for col in train_full.columns:
    if col.startswith("f_"):
        embedding_features.append(col)

# Числовые фичи (пример — добавь свои)
numeric_features = [
    # "price",
    # "age",
]

# Бинарные фичи (0/1)
binary_features = [
    # "is_new",
    # "has_discount",
    'Subject Focus',
    'Eyes',
    'Face',
    'Near',
    'Action',
    'Accessory',
    'Group',
    'Collage',
    'Human',
    'Occlusion',
    'Info',
    'Blur'
]

# Фичи расстояний (если добавил cos_sim_to_mean — он числовой)
distance_features = [
    "cos_sim_to_mean"  # убери, если не использовал блок с косинусом
]

# Категориальные фичи (ОБЯЗАТЕЛЬНО ОСТАВИТЬ ИХ СТРОКАМИ)
categorical_features = [
    # "color",
    # "store_type",
]

# Собираем общий список фич в правильном порядке
feature_cols = []
feature_cols.extend(embedding_features)
feature_cols.extend(numeric_features)
feature_cols.extend(binary_features)
feature_cols.extend(distance_features)
feature_cols.extend(categorical_features)

print("Всего фич:", len(feature_cols))
print("Первые 10 фич:", feature_cols[:10])



cat_feature_indices = []

for cat_col in categorical_features:
    if cat_col in feature_cols:
        idx = feature_cols.index(cat_col)
        cat_feature_indices.append(idx)

print("Категориальные столбцы:", categorical_features)
print("Индексы категориальных столбцов:", cat_feature_indices)
#Здесь мы как раз делаем то, о чём говорили: передаём CatBoost индексы колонок, а сами значения оставляем строками.


X = train_full[feature_cols]
y = train_full[TARGET_COL]

X_test = test_full[feature_cols]


from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error
import optuna
import gc
import numpy as np

# Берём подвыборку для Optuna (чтобы быстрее и безопаснее)
X_sample, _, y_sample, _ = train_test_split(
    X,
    y,
    train_size=3000,      # 3k строк вполне достаточно для подбора
    random_state=42,
    shuffle=True
)

X_train_opt, X_valid_opt, y_train_opt, y_valid_opt = train_test_split(
    X_sample,
    y_sample,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

train_pool_opt = Pool(
    data=X_train_opt,
    label=y_train_opt,
    cat_features=cat_feature_indices
)

valid_pool_opt = Pool(
    data=X_valid_opt,
    label=y_valid_opt,
    cat_features=cat_feature_indices
)

print("Optuna train shape:", X_train_opt.shape)
print("Optuna valid shape:", X_valid_opt.shape)

optuna.logging.set_verbosity(optuna.logging.INFO)



def objective(trial):
    depth = trial.suggest_int("depth", 4, 9)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True)
    random_strength = trial.suggest_float("random_strength", 0.1, 2.0)
    bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 1.0)

    model = CatBoostRegressor(
        iterations=400,               # на CPU + маленькая выборка это нормально
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        random_strength=random_strength,
        bagging_temperature=bagging_temperature,
        loss_function="RMSE",
        eval_metric="RMSE",
        task_type="CPU",             # <<< ВАЖНО: CPU, НЕ GPU
        verbose=100,
        od_type="Iter",
        od_wait=40
    )

    model.fit(train_pool_opt, eval_set=valid_pool_opt)

    y_pred_valid = model.predict(valid_pool_opt)
    rmse = mean_squared_error(y_valid_opt, y_pred_valid, squared=False)

    print(f"[Trial {trial.number}] RMSE = {rmse:.4f}")

    del model
    gc.collect()

    return rmse



study = optuna.create_study(direction="minimize")

study.optimize(
    objective,
    n_trials=15,        # начни с 10–15, потом можно поднять
    timeout=1200        # максимум ~20 минут
)

print("Лучшее значение RMSE:", study.best_value)
print("Лучшие параметры:", study.best_trial.params)

best_params = study.best_trial.params



from sklearn.model_selection import KFold

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_full))
test_preds = np.zeros(len(test_full))

fold = 0

for train_idx, valid_idx in kf.split(X, y):
    fold += 1
    print(f"\n===== Fold {fold}/{n_splits} =====")

    X_tr = X.iloc[train_idx]
    y_tr = y.iloc[train_idx]
    X_val = X.iloc[valid_idx]
    y_val = y.iloc[valid_idx]

    train_pool_fold = Pool(
        data=X_tr,
        label=y_tr,
        cat_features=cat_feature_indices
    )

    valid_pool_fold = Pool(
        data=X_val,
        label=y_val,
        cat_features=cat_feature_indices
    )

    model_fold = CatBoostRegressor(
        iterations=1000,                             # побольше, финальная модель
        depth=best_params["depth"],
        learning_rate=best_params["learning_rate"],
        l2_leaf_reg=best_params["l2_leaf_reg"],
        random_strength=best_params["random_strength"],
        bagging_temperature=best_params["bagging_temperature"],
        loss_function="RMSE",
        eval_metric="RMSE",
        task_type="GPU",                             # <<< здесь уже GPU
        devices="0",
        verbose=100,
        od_type="Iter",
        od_wait=60
    )

    model_fold.fit(train_pool_fold, eval_set=valid_pool_fold)

    oof_fold = model_fold.predict(valid_pool_fold)
    oof_preds[valid_idx] = oof_fold

    test_pool = Pool(X_test, cat_features=cat_feature_indices)
    test_fold = model_fold.predict(test_pool)
    test_preds += test_fold / n_splits

    del model_fold
    gc.collect()



from sklearn.metrics import mean_squared_error

rmse_oof = mean_squared_error(y, oof_preds, squared=False)
print(f"\nOOF RMSE (по {n_splits} фолдам): {rmse_oof:.4f}")

submission = pd.DataFrame()
submission["Id"] = test_full["Id"]      # проверь имя колонки
submission["Pawpularity"] = test_preds

submission.to_csv("submission.csv", index=False)


