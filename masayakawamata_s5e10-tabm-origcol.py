!pip install -qq tabm


import warnings
warnings.simplefilter('ignore')


import pandas as pd, numpy as np

train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
orig = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv')
print('Train Shape:', train.shape)
print('Test Shape:', test.shape)
print('Orig Shape:', orig.shape)

train.head(3)


TARGET = 'accident_risk'
BASE = [col for col in train.columns if col not in ['id', TARGET]]
CATS = ['road_type', 'lighting', 'weather', 'road_signs_present', 'public_road', 'time_of_day', 'holiday', 'school_season']

print(f'{len(BASE)} Base Features:{BASE}')


ORIG = []

for col in BASE:
    tmp = orig.groupby(col)[TARGET].mean()
    new_col_name = f"orig_{col}"
    tmp.name = new_col_name
    train = train.merge(tmp, on=col, how='left')
    test = test.merge(tmp, on=col, how='left')
    ORIG.append(new_col_name)

print(len(ORIG), 'Orig Features Created!!')


META = []

for df in [train, test, orig]:
    base_risk = (
        0.3 * df["curvature"] + 
        0.2 * (df["lighting"] == "night").astype(int) + 
        0.1 * (df["weather"] != "clear").astype(int) + 
        0.2 * (df["speed_limit"] >= 60).astype(int) + 
        0.1 * (np.array(df["num_reported_accidents"]) > 2).astype(int)
    )
    df['Meta'] = base_risk

META.append('Meta')


train['orig_curvature'] = train['orig_curvature'].fillna(orig[TARGET].mean())
test['orig_curvature'] = test['orig_curvature'].fillna(orig[TARGET].mean())


road_type_map = {'urban': 0, 'rural': 1, 'highway': 2}
lighting_map = {'daylight': 0, 'dim': 1, 'night': 2}
weather_map = {'clear': 0, 'rainy': 1, 'foggy': 2} 
time_of_day_map = {'morning': 0, 'afternoon': 1, 'evening': 2}

boolean_map = {False: 0, True: 1}

master_mapping = {
    'road_type': road_type_map,
    'lighting': lighting_map,
    'weather': weather_map,
    'road_signs_present': boolean_map,
    'public_road': boolean_map,
    'time_of_day': time_of_day_map,
    'holiday': boolean_map,
    'school_season': boolean_map
}

for column, mapping in master_mapping.items():
    train[column] = train[column].map(mapping)
    test[column] = test[column].map(mapping)

train.head(3)


FEATURES = BASE + ORIG + META
print(len(FEATURES), 'Features.')


X = train[FEATURES]
y = train[TARGET]


from sklearn.model_selection import KFold

N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)


import torch
from torch.utils.data import TensorDataset, DataLoader
from tabm import TabM
from rtdl_num_embeddings import LinearReLUEmbeddings

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


NUMS = [col for col in FEATURES if col not in CATS]

N_SPLITS = 5
EPOCHS = 200 
PATIENCE = 20 
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 512


oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))

n_num_features = len(NUMS)
cat_cardinalities = [len(train[col].unique()) for col in CATS]
d_out = 1

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f'--- Fold {fold+1}/{N_SPLITS} ---')
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train[NUMS])
    X_val_num_scaled = scaler.transform(X_val[NUMS])
    test_num_scaled = scaler.transform(test[NUMS])

    X_train_num_tensor = torch.tensor(X_train_num_scaled, dtype=torch.float32)
    X_train_cat_tensor = torch.tensor(X_train[CATS].values, dtype=torch.int64)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    
    X_val_num_tensor = torch.tensor(X_val_num_scaled, dtype=torch.float32)
    X_val_cat_tensor = torch.tensor(X_val[CATS].values, dtype=torch.int64)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
    
    train_dataset = TensorDataset(X_train_num_tensor, X_train_cat_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_dataset = TensorDataset(X_val_num_tensor, X_val_cat_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TabM.make(
        n_num_features=n_num_features,
        num_embeddings=LinearReLUEmbeddings(n_num_features),
        cat_cardinalities=cat_cardinalities,
        n_blocks=2,
        d_block=128,
        k=32,
        arch_type='tabm',
        d_out=d_out,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_num, batch_cat, batch_y in train_loader:
            batch_num, batch_cat, batch_y = batch_num.to(device), batch_cat.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            y_pred_ensemble = model(batch_num, batch_cat)
            y_pred = y_pred_ensemble.mean(1)
            loss = loss_fn(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_num, batch_cat, batch_y in val_loader:
                batch_num, batch_cat, batch_y = batch_num.to(device), batch_cat.to(device), batch_y.to(device)
                val_pred_ensemble = model(batch_num, batch_cat)
                val_pred = val_pred_ensemble.mean(1)
                val_loss += loss_fn(val_pred, batch_y).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'best_model_fold_{fold}.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print(f"Epoch {epoch+1}: EarlyStopping... Best Val Loss: {best_val_loss:.4f}")
            break
            
    model.load_state_dict(torch.load(f'best_model_fold_{fold}.pth'))
    model.eval()
    
    val_preds_list = []
    with torch.no_grad():
        for batch_num, batch_cat, _ in val_loader:
            batch_num, batch_cat = batch_num.to(device), batch_cat.to(device)
            preds_tensor = model(batch_num, batch_cat).mean(1)
            val_preds_list.append(preds_tensor.cpu().numpy())
    val_preds = np.concatenate(val_preds_list).squeeze()
    oof_preds[val_idx] = val_preds

    test_num_tensor = torch.tensor(test_num_scaled, dtype=torch.float32)
    test_cat_tensor = torch.tensor(test[CATS].values, dtype=torch.int64)
    test_dataset = TensorDataset(test_num_tensor, test_cat_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    test_preds_list = []
    with torch.no_grad():
        for batch_num, batch_cat in test_loader:
            batch_num, batch_cat = batch_num.to(device), batch_cat.to(device)
            preds_tensor = model(batch_num, batch_cat).mean(1)
            test_preds_list.append(preds_tensor.cpu().numpy())
    fold_test_preds = np.concatenate(test_preds_list).squeeze()
    test_preds += fold_test_preds

    print(f"Fold {fold+1} RMSE: {mean_squared_error(y_val, val_preds, squared=False):.5f}")

test_preds /= N_SPLITS

print(f"Overall OOF RMSE: {mean_squared_error(y, oof_preds, squared=False):.5f}")


pd.DataFrame({'id': train.id, TARGET: oof_preds}).to_csv('oof_tabm_plus_origcol.csv', index=False)
pd.DataFrame({'id': test.id, TARGET: test_preds}).to_csv('test_tabm_plus_origcol.csv', index=False)

