!ls /kaggle/input/favorita-grocery-sales-forecasting


import os
import subprocess

# ===== Cáº¥u hÃ¬nh thÆ° má»¥c =====
input_dir = "/kaggle/input/favorita-grocery-sales-forecasting"
output_dir = "/kaggle/working/favorita"
os.makedirs(output_dir, exist_ok=True)

print("ğŸ“‚ ThÆ° má»¥c nguá»“n:", input_dir)
print("ğŸ“‚ ThÆ° má»¥c giáº£i nÃ©n:", output_dir)

# ===== Ä�áº£m báº£o cÃ³ sáºµn cÃ´ng cá»¥ 7z =====
try:
    subprocess.run(["7z"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except FileNotFoundError:
    print("âš™ï¸� CÃ i Ä‘áº·t p7zip-full...")
    os.system("apt-get install -y p7zip-full")

# ===== Giáº£i nÃ©n toÃ n bá»™ file .7z =====
files = [f for f in os.listdir(input_dir) if f.endswith(".7z")]
if not files:
    print("âš ï¸� KhÃ´ng tÃ¬m tháº¥y file .7z nÃ o trong thÆ° má»¥c input.")
else:
    for file in files:
        src_path = os.path.join(input_dir, file)
        print(f"ğŸ—œï¸� Ä�ang giáº£i nÃ©n: {file}")
        # DÃ¹ng subprocess Ä‘á»ƒ gá»�i 7z
        cmd = ["7z", "x", src_path, f"-o{output_dir}", "-y"]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print(f"âœ… Ä�Ã£ giáº£i nÃ©n: {file}")
        else:
            print(f"â�Œ Lá»—i khi giáº£i nÃ©n {file}: {result.stderr.decode()}")

print("\nğŸ“� CÃ¡c file sau khi giáº£i nÃ©n:")
print(os.listdir(output_dir))



import pandas as pd


# ===== 1ï¸�âƒ£ TRAIN =====
train_path = os.path.join(output_dir, "train.csv")
train = pd.read_csv(train_path, parse_dates=["date"])
train = train[(train["date"] >= "2017-06-01") & (train["date"] < "2017-07-01")]
print(f"ğŸ“¦ train.csv : {train.shape[0]} hÃ ng Ã— {train.shape[1]} cá»™t\n")

# ===== 2ï¸�âƒ£ TEST =====
test_path = os.path.join(output_dir, "test.csv")
test = pd.read_csv(test_path, parse_dates=["date"])
print(f"ğŸ“¦ test.csv: {test.shape[0]} hÃ ng Ã— {test.shape[1]} cá»™t\n")

# ===== 3ï¸�âƒ£ TRANSACTIONS =====
transactions_path = os.path.join(output_dir, "transactions.csv")
transactions = pd.read_csv(transactions_path, parse_dates=["date"])
print(f"ğŸ“¦ transactions.csv: {transactions.shape[0]} hÃ ng Ã— {transactions.shape[1]} cá»™t\n")

# ===== 4ï¸�âƒ£ STORES =====
stores_path = os.path.join(output_dir, "stores.csv")
stores = pd.read_csv(stores_path)
print(f"ğŸ“¦ stores.csv: {stores.shape[0]} hÃ ng Ã— {stores.shape[1]} cá»™t\n")

# ===== 5ï¸�âƒ£ ITEMS =====
items_path = os.path.join(output_dir, "items.csv")
items = pd.read_csv(items_path)
print(f"ğŸ“¦ items.csv: {items.shape[0]} hÃ ng Ã— {items.shape[1]} cá»™t\n")

# ===== 6ï¸�âƒ£ HOLIDAYS =====
holidays_path = os.path.join(output_dir, "holidays_events.csv")
holidays = pd.read_csv(holidays_path, parse_dates=["date"])
print(f"ğŸ“¦ holidays_events.csv: {holidays.shape[0]} hÃ ng Ã— {holidays.shape[1]} cá»™t\n")

# ===== 7ï¸�âƒ£ OIL =====
oil_path = os.path.join(output_dir, "oil.csv")
oil = pd.read_csv(oil_path, parse_dates=["date"])
print(f"ğŸ“¦ oil.csv: {oil.shape[0]} hÃ ng Ã— {oil.shape[1]} cá»™t\n")

# ===== Tá»•ng káº¿t =====
print("âœ… Táº¥t cáº£ dá»¯ liá»‡u Ä‘Ã£ Ä‘á»�c xong:")
print({
    "train": train.shape,
    "test": test.shape,
    "transactions": transactions.shape,
    "stores": stores.shape,
    "items": items.shape,
    "holidays": holidays.shape,
    "oil": oil.shape
})


print("=== OIL ===")
print("Missing values before:\n", oil.isna().sum())

# Ä�iá»�n ná»‘t giÃ¡ trá»‹ NaN Ä‘áº§u tiÃªn
if oil["dcoilwtico"].isna().sum() > 0:
    oil["dcoilwtico"].fillna(method="bfill", inplace=True)

# Ä�iá»�n giÃ¡ trá»‹ thiáº¿u báº±ng ná»™i suy tuyáº¿n tÃ­nh
oil["dcoilwtico"] = oil["dcoilwtico"].interpolate(method="linear")

# TÃ¡ch cá»™t ngÃ y
oil["year"] = oil["date"].dt.year
oil["month"] = oil["date"].dt.month
oil["day"] = oil["date"].dt.day

print("\nâœ… Missing values after:\n", oil.isna().sum())
print(oil.head(), "\n")


print("=== HOLIDAYS ===")
holidays.fillna({
    "locale": "None",
    "locale_name": "None",
    "description": "None",
    "transferred": False
}, inplace=True)

# Chuáº©n kiá»ƒu boolean
holidays["transferred"] = holidays["transferred"].astype(bool)

# TÃ¡ch ngÃ y
holidays["year"] = holidays["date"].dt.year
holidays["month"] = holidays["date"].dt.month
holidays["day"] = holidays["date"].dt.day

print("âœ… Holidays cleaned:")
print(holidays.isna().sum())
print(holidays.head(), "\n")


# Chuáº©n hÃ³a cá»™t date trong táº¥t cáº£ cÃ¡c báº£ng cÃ³ cá»™t 'date'
for df in [train, test, transactions]:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

print("âœ… Chuáº©n hÃ³a date hoÃ n táº¥t trong train, test, transactions.\n")

# ===== 4ï¸�âƒ£ TÃ¡ch Ä‘áº·c trÆ°ng thá»�i gian trong train =====
train["year"] = train["date"].dt.year
train["month"] = train["date"].dt.month
train["day"] = train["date"].dt.day
train["dayofweek"] = train["date"].dt.dayofweek

print("ğŸ“… train sau khi tÃ¡ch cá»™t thá»�i gian:")
print(train.head())


# Sao chÃ©p train Ä‘á»ƒ trÃ¡nh Ä‘á»¥ng dá»¯ liá»‡u gá»‘c
train_merged = train.copy()

print("=== MERGING DATASETS ===")
train_merged = pd.merge(train_merged, stores, on="store_nbr", how="left")
train_merged = pd.merge(train_merged, items, on="item_nbr", how="left")
train_merged = pd.merge(train_merged, transactions, on=["date", "store_nbr"], how="left")
train_merged = pd.merge(train_merged, oil[["date", "dcoilwtico"]], on="date", how="left")
train_merged = pd.merge(train_merged, holidays[["date", "type", "locale", "locale_name", "description", "transferred"]], 
                        on="date", how="left")

print("\nâœ… Merge hoÃ n táº¥t.")
print("ğŸ“Š train_merged:", train_merged.shape)

# Kiá»ƒm tra sá»‘ lÆ°á»£ng NA
print("\nğŸ”� Missing values (top 10 cá»™t cÃ³ NA):")
print(train_merged.isna().sum().sort_values(ascending=False).head(10))

# Xem vÃ i dÃ²ng Ä‘áº§u
print("\nğŸ“„ Preview:")
print(train_merged.head())



# ===== Xá»­ lÃ½ giÃ¡ dáº§u (oil) =====
train_merged["dcoilwtico"] = train_merged["dcoilwtico"].fillna(method="ffill")

# ===== Xá»­ lÃ½ transactions =====
# Ä�iá»�n theo trung bÃ¬nh cá»­a hÃ ng (store_nbr)
transactions_mean = train_merged.groupby("store_nbr")["transactions"].transform("mean")
train_merged["transactions"] = train_merged["transactions"].fillna(transactions_mean)

# ===== Xá»­ lÃ½ dá»¯ liá»‡u ngÃ y lá»… (holidays) =====
train_merged["type_y"].fillna("Work Day", inplace=True)
train_merged["locale"].fillna("None", inplace=True)
train_merged["locale_name"].fillna("None", inplace=True)
train_merged["description"].fillna("Normal Day", inplace=True)
train_merged["transferred"].fillna(False, inplace=True)

print("âœ… Ä�Ã£ xá»­ lÃ½ toÃ n bá»™ missing values.")
print(train_merged.isna().sum().sort_values(ascending=False).head(10))



train_merged.head()


import numpy as np

# ===== 1ï¸�âƒ£ Encode dá»¯ liá»‡u Boolean vÃ  Category =====
train_fe = train_merged.copy()

# onpromotion: True/False â†’ 1/0
train_fe["onpromotion"] = train_fe["onpromotion"].astype(str).map({"True": 1, "False": 0})

# transferred: True/False â†’ 1/0
train_fe["transferred"] = train_fe["transferred"].astype(bool).astype(int)

# Chuyá»ƒn cÃ¡c cá»™t categorical sang mÃ£ sá»‘ (Label Encoding)
categorical_cols = ["family", "city", "state", "type", "type_y", "locale", "locale_name"]
for col in categorical_cols:
    if col in train_fe.columns:
        train_fe[col] = train_fe[col].astype("category").cat.codes

print("âœ… Ä�Ã£ encode xong dá»¯ liá»‡u categorical & boolean.")


# ===== 2ï¸�âƒ£ Feature thá»�i gian =====
train_fe["weekofyear"] = train_fe["date"].dt.isocalendar().week.astype(int)
train_fe["is_weekend"] = train_fe["dayofweek"].isin([5, 6]).astype(int)
train_fe["quarter"] = train_fe["date"].dt.quarter

# Ä�Ã¡nh dáº¥u Ä‘áº§u/thÃ¡ng/cuá»‘i thÃ¡ng
train_fe["is_month_start"] = train_fe["date"].dt.is_month_start.astype(int)
train_fe["is_month_end"] = train_fe["date"].dt.is_month_end.astype(int)


# ===== 3ï¸�âƒ£ Feature vá»� mÃ¹a (Season) =====
def season_from_month(m):
    if m in [12, 1, 2]: return 0  # Winter
    elif m in [3, 4, 5]: return 1  # Spring
    elif m in [6, 7, 8]: return 2  # Summer
    else: return 3  # Fall
train_fe["season"] = train_fe["month"].apply(season_from_month)


# ===== 4ï¸�âƒ£ Feature tÆ°Æ¡ng tÃ¡c há»¯u Ã­ch =====
# Trung bÃ¬nh sales theo cá»­a hÃ ng vÃ  nhÃ³m hÃ ng
store_avg_sales = train_fe.groupby("store_nbr")["unit_sales"].transform("mean")
family_avg_sales = train_fe.groupby("family")["unit_sales"].transform("mean")

train_fe["store_avg_sales"] = store_avg_sales
train_fe["family_avg_sales"] = family_avg_sales

# ===== 5ï¸�âƒ£ Xá»­ lÃ½ target (unit_sales) dáº¡ng log =====
# Má»™t sá»‘ dÃ²ng cÃ³ unit_sales <= 0 â†’ thay báº±ng 0.1 Ä‘á»ƒ trÃ¡nh log(0)
train_fe["unit_sales"] = train_fe["unit_sales"].clip(lower=0.1)
train_fe["log_sales"] = np.log1p(train_fe["unit_sales"])

print("âœ… HoÃ n táº¥t feature engineering & log-transform target.")


# ===== LÆ°u train_fe vÃ o working directory =====
fe_path = "/kaggle/working/train_fe.csv"
train_fe.to_csv(fe_path, index=False)
print(f"âœ… Ä�Ã£ lÆ°u file feature engineered táº¡i: {fe_path}")


# ===== Chá»�n top-15 feature importance =====
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1ï¸�âƒ£ Chuáº©n hÃ³a tÃªn cá»™t
train_fe.rename(columns={"type_x": "store_type", "type_y": "holiday_type"}, inplace=True)

# 2ï¸�âƒ£ Danh sÃ¡ch feature há»£p lá»‡
selected_features = [
    "store_nbr", "item_nbr", "onpromotion", "transactions", "dcoilwtico",
    "family", "class", "perishable", "city", "state",
    "store_type", "holiday_type",
    "year", "month", "day", "dayofweek", "weekofyear", "is_weekend",
    "quarter", "is_month_start", "is_month_end", "season",
    "store_avg_sales", "family_avg_sales"
]

# 3ï¸�âƒ£ Láº¥y máº«u nhá»� Ä‘á»ƒ train
sample_df = train_fe.sample(500_000, random_state=42)
X = sample_df[selected_features].copy()
y = sample_df["log_sales"].copy()

# Encode object â†’ category â†’ int
for col in X.select_dtypes(include="object").columns:
    X[col] = X[col].astype("category").cat.codes

# 4ï¸�âƒ£ Train XGBoost nháº¹ Ä‘á»ƒ phÃ¢n tÃ­ch feature importance
xgb = XGBRegressor(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)
xgb.fit(X, y)

# 5ï¸�âƒ£ PhÃ¢n tÃ­ch Ä‘á»™ quan trá»�ng
importance_df = (
    pd.DataFrame({
        "Feature": selected_features,
        "Importance": xgb.feature_importances_
    })
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)

# 6ï¸�âƒ£ Lá»�c top-15 feature
top_features = importance_df["Feature"].head(15).tolist()
top_features.append("log_sales")
print("ğŸ”¥ Top features Ä‘Æ°á»£c giá»¯ láº¡i:", top_features)

# 7ï¸�âƒ£ Táº¡o vÃ  lÆ°u file final
train_final = train_fe[top_features].copy()
final_path = "/kaggle/working/train_final.csv"
train_final.to_csv(final_path, index=False)

print(f"âœ… Ä�Ã£ lÆ°u file train_final táº¡i: {final_path}")
print("ğŸ“¦ KÃ­ch thÆ°á»›c:", train_final.shape)

# 8ï¸�âƒ£ Hiá»ƒn thá»‹ biá»ƒu Ä‘á»“ top-15 feature
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"].head(15)[::-1],
          importance_df["Importance"].head(15)[::-1])
plt.xlabel("Importance")
plt.title("Top 15 Feature Importance (XGBoost)")
plt.tight_layout()
plt.show()

# 9ï¸�âƒ£ In báº£ng importance
print("\nğŸ“Š Top 15 Feature quan trá»�ng nháº¥t:")
print(importance_df.head(15))



# ===== Giáº£i phÃ³ng bá»™ nhá»› Ä‘á»ƒ trÃ¡nh trÃ n =====
del train_final, train_fe, train_merged, train, X, y, xgb
import gc; gc.collect()
print("ğŸ§¹ Ä�Ã£ xoÃ¡ toÃ n bá»™ biáº¿n náº·ng trong RAM.")


import pandas as pd


train_final = pd.read_csv("/kaggle/working/train_final.csv")


# Encode store_type (object -> category codes)
train_final["store_type"] = train_final["store_type"].astype("category").cat.codes
print("âœ… store_type Ä‘Ã£ encode:", train_final["store_type"].unique()[:10])

train_final.info()
train_final.head()


for col in train_final.select_dtypes(include="int64").columns:
    train_final[col] = pd.to_numeric(train_final[col], downcast="integer")

for col in train_final.select_dtypes(include="float64").columns:
    train_final[col] = pd.to_numeric(train_final[col], downcast="float")

print("âœ… Ä�Ã£ giáº£m dung lÆ°á»£ng. KÃ­ch thÆ°á»›c má»›i:")
train_final.info(memory_usage="deep")
train_final.head()


from sklearn.model_selection import train_test_split

X = train_final.drop(columns=["log_sales"])
y = train_final["log_sales"]

# Láº¥y máº«u nháº¹ hÆ¡n náº¿u cáº§n
# sample_frac = 0.2  # chá»‰ láº¥y 20% náº¿u muá»‘n thá»­ nhanh
# X = X.sample(frac=sample_frac, random_state=42)
# y = y.loc[X.index]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("ğŸ“¦ Train:", X_train.shape, " | Test:", X_test.shape)



import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# ===== HÃ m tiá»‡n Ã­ch Ä‘Ã¡nh giÃ¡ =====
def evaluate_model(name, y_true, y_pred, start_time):
    """
    Ä�Ã¡nh giÃ¡ mÃ´ hÃ¬nh vÃ  tráº£ vá»� (rmse, mae, r2)
    """
    elapsed = (time.time() - start_time) / 60  # phÃºt
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\nğŸ“Š {name}")
    print(f"â�±ï¸� Thá»�i gian train + predict: {elapsed:.2f} phÃºt")
    print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | RÂ²: {r2:.4f}")

    return rmse, mae, r2

# ===== 1ï¸�âƒ£ Linear Regression =====
start = time.time()
lr = LinearRegression(n_jobs=-1)
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
rmse_lr, mae_lr, r2_lr = evaluate_model("Linear Regression", y_test, pred_lr, start)

# ===== 2ï¸�âƒ£ XGBoost Regressor =====
start = time.time()
xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train, y_train)
pred_xgb = xgb.predict(X_test)
rmse_xgb, mae_xgb, r2_xgb = evaluate_model("XGBoost Regressor", y_test, pred_xgb, start)

results_df = pd.DataFrame([
    {"Model": "Linear Regression", "RMSE": rmse_lr, "MAE": mae_lr, "R2": r2_lr},
    {"Model": "XGBoost Regressor", "RMSE": rmse_xgb, "MAE": mae_xgb, "R2": r2_xgb}
]).sort_values("RMSE")

print("\nâœ… So sÃ¡nh káº¿t quáº£:")
print(results_df)



import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np

# ===== 1ï¸�âƒ£ Chuyá»ƒn dá»¯ liá»‡u sang tensor =====
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train_np = y_train.to_numpy().astype(np.float32)
y_test_np = y_test.to_numpy().astype(np.float32)

# ===== 2ï¸�âƒ£ Ä�á»‹nh nghÄ©a Dataset tuá»³ chá»‰nh =====
class SalesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = SalesDataset(X_train_scaled, y_train_np)
test_ds  = SalesDataset(X_test_scaled, y_test_np)

# ===== 3ï¸�âƒ£ Táº¡o DataLoader =====
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

print(f"âœ… DataLoader sáºµn sÃ ng: {len(train_loader)} batch train | {len(test_loader)} batch test")



import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ===== 1ï¸�âƒ£ Ä�á»‹nh nghÄ©a mÃ´ hÃ¬nh MLP =====
class MLPModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

mlp = MLPModel(X_train_scaled.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(mlp.parameters(), lr=1e-3)

# ===== 2ï¸�âƒ£ Train loop =====
epochs = 30
mlp.train()
for epoch in range(epochs):
    total_loss = 0
    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        xb, yb = xb.to(device), yb.to(device).view(-1, 1)
        pred = mlp(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.5f}")



from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlp.eval()
preds, trues = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        pred = mlp(xb).cpu().numpy().flatten()
        preds.append(pred)
        trues.append(yb.numpy())

preds = np.concatenate(preds)
trues = np.concatenate(trues)

rmse_mlp = np.sqrt(mean_squared_error(trues, preds))
mae_mlp = mean_absolute_error(trues, preds)
r2_mlp = r2_score(trues, preds)

print("\nğŸ“Š MLP Evaluation:")
print(f"RMSE: {rmse_mlp:.4f} | MAE: {mae_mlp:.4f} | RÂ²: {r2_mlp:.4f}")



# ===== Chuáº©n bá»‹ dá»¯ liá»‡u 3D cho LSTM =====
X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_lstm  = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

train_ds_lstm = SalesDataset(X_train_lstm, y_train_np)
test_ds_lstm  = SalesDataset(X_test_lstm, y_test_np)

train_loader_lstm = DataLoader(train_ds_lstm, batch_size=512, shuffle=True, num_workers=2, pin_memory=True)
test_loader_lstm  = DataLoader(test_ds_lstm, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

# ===== Ä�á»‹nh nghÄ©a LSTM model =====
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

lstm = LSTMModel(X_train_scaled.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm.parameters(), lr=1e-3)

# ===== Train loop =====
epochs = 8
lstm.train()
for epoch in range(epochs):
    total_loss = 0
    for xb, yb in tqdm(train_loader_lstm, desc=f"LSTM Epoch {epoch+1}/{epochs}"):
        xb, yb = xb.to(device), yb.to(device).view(-1, 1)
        pred = lstm(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader_lstm):.5f}")



lstm.eval()
preds, trues = [], []
with torch.no_grad():
    for xb, yb in test_loader_lstm:
        xb = xb.to(device)
        pred = lstm(xb).cpu().numpy().flatten()
        preds.append(pred)
        trues.append(yb.numpy())

preds = np.concatenate(preds)
trues = np.concatenate(trues)

rmse_lstm = np.sqrt(mean_squared_error(trues, preds))
mae_lstm = mean_absolute_error(trues, preds)
r2_lstm = r2_score(trues, preds)

print("\nğŸ“Š LSTM Evaluation:")
print(f"RMSE: {rmse_lstm:.4f} | MAE: {mae_lstm:.4f} | RÂ²: {r2_lstm:.4f}")



results_all = pd.DataFrame([
    {"Model": "Linear Regression", "RMSE": rmse_lr, "MAE": mae_lr, "R2": r2_lr},
    {"Model": "XGBoost", "RMSE": rmse_xgb, "MAE": mae_xgb, "R2": r2_xgb},
    {"Model": "MLP (PyTorch)", "RMSE": rmse_mlp, "MAE": mae_mlp, "R2": r2_mlp},
    {"Model": "LSTM (PyTorch)", "RMSE": rmse_lstm, "MAE": mae_lstm, "R2": r2_lstm}
]).sort_values("RMSE")

print("\nâœ… So sÃ¡nh toÃ n bá»™ mÃ´ hÃ¬nh:")
print(results_all)


