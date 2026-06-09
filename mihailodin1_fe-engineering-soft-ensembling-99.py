import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import random
from sklearn.preprocessing import RobustScaler
import category_encoders as ce

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour, plot_slice
from sklearn.model_selection import cross_val_score, train_test_split

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from catboost.utils import get_gpu_device_count
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

sns.set_style("darkgrid")
plt.rcParams.update({
    "figure.figsize": (16, 8),    
    "figure.facecolor": "white",    
    "figure.autolayout": True,     
})

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
if device:
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


print(f"GPU: {get_gpu_device_count()}")


train = pd.read_csv("//kaggle/input/hotel-booking-demand-3/train_final.csv")
test = pd.read_csv("/kaggle/input/hotel-booking-demand-3/test_final.csv")


train.head()


train.info()


train.describe()


train.columns


train["hotel"] = train["hotel"].str.replace(" Hotel", "")
test["hotel"] = test["hotel"].str.replace(" Hotel", "")


month_mapping = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}


def process_reservation_status_date(df):
    df_processed = df.copy()
    
    df_processed["reservation_status_date"] = pd.to_datetime(df_processed["reservation_status_date"])
    
    df_processed["arrival_date"] = pd.to_datetime(
        dict(year=df_processed["arrival_date_year"],
             month=df_processed["arrival_date_month"].map(month_mapping),
             day=df_processed["arrival_date_day_of_month"])
    )
    
    df_processed["days_between_status_and_arrival"] = (
        df_processed["arrival_date"] - df_processed["reservation_status_date"]
    ).dt.days
    
    df_processed["booking_timing"] = pd.cut(
        df_processed["days_between_status_and_arrival"],
        bins=[-float("inf"), 0, 1, 7, 30, float("inf")],
        labels=["last_minute", "same_day", "few_days", "weeks_ahead", "months_ahead"]
    )
    
    df_processed["status_relative_to_arrival"] = np.where(
        df_processed["days_between_status_and_arrival"] < 0,
        "after_arrival",
        np.where(df_processed["days_between_status_and_arrival"] == 0,
                "on_arrival_day",
                "before_arrival")
    )
    
    df_processed["arrival_season"] = df_processed["arrival_date"].dt.month.map({
        1: "winter", 2: "winter", 3: "spring", 
        4: "spring", 5: "spring", 6: "summer",
        7: "summer", 8: "summer", 9: "autumn",
        10: "autumn", 11: "autumn", 12: "winter"
    })
    
    df_processed["arrival_weekend"] = (df_processed["arrival_date"].dt.dayofweek >= 5).map(lambda x: "Yes" if x  else "No")
    
    return df_processed.drop(columns=["arrival_date", "reservation_status_date"])

train = process_reservation_status_date(train)
test = process_reservation_status_date(test)


def cast_month_cycle(df):
    df["month_num"] = df["arrival_date_month"].map(month_mapping)
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)

    df = df.drop(columns=["arrival_date_month", "month_num"], axis=1)

    return df


train = cast_month_cycle(train)
test = cast_month_cycle(test)


def cust_week_cycle(df):
    df["week_sin"] = np.sin(2 * np.pi * df["arrival_date_week_number"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["arrival_date_week_number"] / 52)

    df = df.drop(columns=["arrival_date_week_number"])

    return df


train = cust_week_cycle(train)
test = cust_week_cycle(test)


def day_position(day):
    if day <= 10:
        return "beginning"
    elif day <= 20:
        return "middle"
    else:
        return "end"

train["arrival_date_day_of_month"] = train["arrival_date_day_of_month"].apply(day_position)
test["arrival_date_day_of_month"] = test["arrival_date_day_of_month"].apply(day_position)


train["meal"] = train["meal"].str.replace(
    "Undefined",
    train.meal.value_counts().index[0]
)

test["meal"] = test["meal"].str.replace(
    "Undefined",
    test.meal.value_counts().index[0]
)


train["room_transition"] = (
    train["reserved_room_type"].astype(str) + "_to_" + train["assigned_room_type"].astype(str)
)

test["room_transition"] = (
    test["reserved_room_type"].astype(str) + "_to_" + test["assigned_room_type"].astype(str)
)


num_cols = [
    "lead_time", "adr", "days_between_status_and_arrival", "month_sin", "month_cos", "week_sin", "week_cos"
]

discrete_columns = [
    "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies", "previous_cancellations", "previous_bookings_not_canceled",
    "booking_changes", "days_in_waiting_list", "required_car_parking_spaces", "total_of_special_requests", 
]

cat_columns = [
    "hotel", "arrival_date_year", "arrival_date_day_of_month", "meal", "country", "market_segment",
    "distribution_channel", "is_repeated_guest", "reserved_room_type", "assigned_room_type", "room_transition", "deposit_type", 
    "customer_type", "booking_timing", "status_relative_to_arrival", "arrival_season", "arrival_weekend"
]


sns.countplot(data=train, x="is_canceled", palette="coolwarm")
plt.title("Target: is_canceled", fontsize=16)
for i, count in enumerate(train["is_canceled"].value_counts().values):
    plt.text(i, count + 100, str(count), ha="center", fontsize=12)
plt.show()


fig, axes = plt.subplots(1, len(num_cols[:-4]), figsize=(18, 5))

for i, col in enumerate(num_cols[:-4]):
    sns.histplot(data=train, x=col, bins=50, kde=True, ax=axes[i], color="skyblue")

plt.show()


fig, axes = plt.subplots(1, len(num_cols[:-4]), figsize=(18, 6))

for i, col in enumerate(num_cols[:-4]):
    sns.boxplot(data=train, x="is_canceled", y=col, ax=axes[i], palette="Set2")
    axes[i].set_title(f"{col} vs Canceled")
  
plt.show()


fig, axes = plt.subplots(int(np.ceil(len(discrete_columns) / 2)), 2, figsize=(16, 20))
axes = axes.ravel()

for i, col in enumerate(discrete_columns):
    top_cats = train[col].value_counts().head(10).index
    data_plot = train[train[col].isin(top_cats)].copy()
    
    sns.countplot(data=data_plot, x=col, hue="is_canceled", ax=axes[i], palette="Set1")
    axes[i].set_title(f"{col} — Distribution")
    axes[i].tick_params(axis="x", rotation=0)

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.show()


top_countries = train["country"].value_counts().head(10).index
train["country_top"] = train["country"].where(train["country"].isin(top_countries), "Other")

plot_columns = [col for col in cat_columns if col != "country"] + ["country_top"]
plot_columns.remove("arrival_date_day_of_month") 

fig, axes = plt.subplots(6, 3, figsize=(20, 24))
axes = axes.ravel()

for i, col in enumerate(plot_columns):
    order = train.groupby(col)["is_canceled"].mean().sort_values(ascending=False).index
    
    sns.barplot(data=train, x=col, y="is_canceled", estimator=np.mean, order=order,
                ax=axes[i], palette="viridis", errorbar=None)
    axes[i].tick_params(axis="x", rotation=45)

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])
train.drop(columns=["country_top"], inplace=True)

plt.show()


corr_cols = num_cols + [
    "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies",
    "previous_cancellations", "booking_changes", "total_of_special_requests"
]

plt.figure(figsize=(14, 10))
correlation_matrix = train[corr_cols + ["is_canceled"]].corr()

sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
            square=True, linewidths=0.5)
plt.show()


sample_df = train.sample(1000, random_state=42)  # чтобы не тормозить
pair_cols = ["lead_time", "adr", "stays_in_week_nights", "is_canceled"]
pair_data = sample_df[pair_cols].copy()
pair_data["is_canceled"] = pair_data["is_canceled"].map({0: "Not Canceled", 1: "Canceled"})

sns.pairplot(pair_data, hue="is_canceled", palette="Set1", plot_kws={"alpha": 0.6})
plt.show()


X = train.drop(columns=["is_canceled"])
target = train["is_canceled"]
test = test.loc[:, X.columns]


TE = ce.TargetEncoder(smoothing=7, cols=cat_columns)

train_encoded = TE.fit_transform(X[cat_columns], target)

test_encoded = TE.transform(test[cat_columns])
te_cols = []

for col in cat_columns:
    te_col_name = f"TE_{col}"
    X[te_col_name] = train_encoded[col]
    test[te_col_name] = test_encoded[col]
    te_cols.append(te_col_name)


X_xgb = X.copy()
test_xgb = test.copy()

intera_cols = []
for cat_col in cat_columns:
    for num_col in num_cols + discrete_columns:
        intera_col = f"{cat_col}_x_{num_col}"
        X[intera_col] = X[cat_col].astype(str) + "_" + X[num_col].astype(str)
        test[intera_col] = test[cat_col].astype(str) + "_" + test[num_col].astype(str)
        intera_cols.append(intera_col)
        
        X_xgb[intera_col] = (X[cat_col].astype(str) + "_" + X[num_col].astype(str)).astype("category").cat.codes
        test_xgb[intera_col] = (test[cat_col].astype(str) + "_" + test[num_col].astype(str)).astype("category").cat.codes
    
    X_xgb[cat_col] = X[cat_col].astype("category").cat.codes
    test_xgb[cat_col] = test[cat_col].astype("category").cat.codes

cat_columns += intera_cols


num_cols_for_scaling = num_cols + discrete_columns + te_cols

X_cat_df = X_xgb[cat_columns].astype("int64").copy()
test_cat_df = test_xgb[cat_columns].astype("int64").copy()

scaler = RobustScaler().set_output(transform="pandas")
scaler.fit(X_xgb[num_cols_for_scaling])

X_num_df = scaler.transform(X_xgb[num_cols_for_scaling])
test_num_df = scaler.transform(test_xgb[num_cols_for_scaling])


X_xgb = pd.concat([X_num_df, X_cat_df], axis=1)
test_xgb = pd.concat([test_num_df, test_cat_df], axis=1)


category_maps = {}

for col in cat_columns:
    X_xgb[col] = X_xgb[col].astype(str)
    test_xgb[col] = test_xgb[col].astype(str)
    
    unique_train = X_xgb[col].unique()
    categories = list(unique_train) + ["__OOV__"]
    category_map = {cat: idx for idx, cat in enumerate(categories)}
    category_maps[col] = category_map

for col in cat_columns:
    X_xgb[col] = X_xgb[col].map(category_maps[col]).astype(int)

for col in cat_columns:
    def map_with_oov(x):
        return category_maps[col].get(x, category_maps[col]["__OOV__"])
    
    test_xgb[col] = test_xgb[col].apply(map_with_oov).astype(int)


class TabularDataset(Dataset):
    def __init__(self, df, target, cat_cols, num_cols):
        self.df = df
        self.target = target.values.astype(np.float32)
        self.cat_cols = cat_cols
        self.num_cols = num_cols
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x_cat = torch.tensor(row[self.cat_cols].values, dtype=torch.long)
        x_num = torch.tensor(row[self.num_cols].values, dtype=torch.float32)
        y = torch.tensor(self.target[idx], dtype=torch.float32)
        return x_cat, x_num, y


class EmbeddingNN(nn.Module):
    def __init__(self, cat_cardinalities, num_features, embedding_dim_rule=lambda x: min(50, (x // 2) + 1), hidden_sizes=[256, 128, 64], dropout=0.3):
        super().__init__()
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories, embedding_dim_rule(num_categories))
            for num_categories in cat_cardinalities
        ])
        
        emb_size_total = sum(embedding_dim_rule(n_cat) for n_cat in cat_cardinalities)
        input_size = emb_size_total + num_features
        
        layers = []
        for h in hidden_sizes:
            layers.append(nn.Linear(input_size, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_size = h
        layers.append(nn.Linear(input_size, 1))
        self.fc = nn.Sequential(*layers)
    
    def forward(self, x_cat, x_num):
        embedded = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embeddings)]
        embedded = torch.cat(embedded, dim=1)
        x_all = torch.cat([embedded, x_num], dim=1)
        return self.fc(x_all)


X_train, X_val, y_train, y_val = train_test_split(X_xgb, target, test_size=0.1, random_state=42)

train_dataset = TabularDataset(X_train, y_train, cat_columns, num_cols_for_scaling)
val_dataset = TabularDataset(X_val, y_val, cat_columns, num_cols_for_scaling)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)


cat_cardinalities = [len(category_maps[col]) for col in cat_columns]
num_features = len(num_cols_for_scaling)

model = EmbeddingNN(cat_cardinalities, num_features)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss() 


for epoch in range(15):
    model.train()
    train_loss, train_acc = 0, 0
    for x_cat, x_num, y in tqdm(train_loader):
        x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x_cat, x_num).squeeze()
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * len(y)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        train_acc += (preds == y).float().sum().item()
    
    model.eval()
    val_loss, val_acc = 0, 0
    with torch.no_grad():
        for x_cat, x_num, y in val_loader:
            x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)
            logits = model(x_cat, x_num).squeeze()
            loss = criterion(logits, y)
            val_loss += loss.item() * len(y)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            val_acc += (preds == y).float().sum().item()
    
    print(f"Epoch {epoch+1}: "
          f"Train Loss={train_loss/len(train_dataset):.4f}, Train Acc={train_acc/len(train_dataset):.4f} | "
          f"Val Loss={val_loss/len(val_dataset):.4f}, Val Acc={val_acc/len(val_dataset):.4f}")


test_dataset = TabularDataset(
    test_xgb, 
    pd.Series(np.zeros(len(test_xgb))),
    cat_columns, 
    num_cols_for_scaling
)

test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)


model.eval()
all_probs = []

with torch.no_grad():
    for x_cat, x_num, _ in tqdm(test_loader):
        x_cat, x_num = x_cat.to(device), x_num.to(device)
        logits = model(x_cat, x_num).squeeze()  
        prob_class_1 = torch.sigmoid(logits)  
        prob_class_0 = 1 - prob_class_1       
        
        probs = torch.stack([prob_class_0, prob_class_1], dim=1)
        all_probs.append(probs.cpu().numpy())

all_probs = np.concatenate(all_probs, axis=0)


def objective_catboost(trial):
    max_depth = trial.suggest_int("max_depth", 3, 6)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
    n_estimators = trial.suggest_int("n_estimators", 200, 1500)

    model = CatBoostClassifier(
        task_type="GPU",
        devices="0:1", 
        cat_features=cat_columns,
        random_state=SEED,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        silent=True
    )

    score = cross_val_score(model, X, target, cv=3, scoring="accuracy").mean()
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective_catboost, n_trials=40)
params_catboost = study.best_params


def objective_xgb(trial):
    max_depth=trial.suggest_int("max_depth", 3, 7)
    n_estimators=trial.suggest_int("n_estimators", 100, 1000)
    # learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True)

    model = XGBClassifier(
        tree_method="gpu_hist",
        random_state=SEED,
        max_depth=max_depth,
        n_estimators=n_estimators,
        # learning_rate=learning_rate,
    )

    scores = cross_val_score(
        model, 
        X_xgb, 
        target, 
        cv=3,
        scoring="accuracy"  
    )

    return scores.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective_xgb, n_trials=65)
params_xgb = study.best_params


catboost_model = CatBoostClassifier(
    task_type="GPU",
    devices="0:1",
    cat_features=cat_columns,
    random_state=SEED,
    **params_catboost
)

catboost_model.fit(X, target)


xgb_model = XGBClassifier(
    tree_method="gpu_hist",
    random_state=SEED,
    **params_xgb
)
xgb_model.fit(X_xgb, target)


lgbm_model = LGBMClassifier(
    device="gpu",
    random_state=SEED,
)
lgbm_model.fit(X_xgb, target)


pred_xgb = xgb_model.predict_proba(test_xgb)
pred_catboost = catboost_model.predict_proba(test)
pred_lgbm = lgbm_model.predict_proba(test_xgb)


pred_avg = (pred_xgb + pred_catboost + pred_lgbm + all_probs) / 4
final_preds = np.argmax(pred_avg, axis=1)


pd.DataFrame({
    "index": test.index,
    "is_canceled": final_preds
}).to_csv("submission.csv", index=False)


pd.read_csv("submission.csv")

