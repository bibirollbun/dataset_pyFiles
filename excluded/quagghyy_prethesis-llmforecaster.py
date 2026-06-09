import os
import pandas as pd

print(os.listdir("/kaggle/input"))



DATA_DIR = "/kaggle/input/favorita-unzipped-csv"


train = pd.read_csv(f"{DATA_DIR}/train.csv", parse_dates=["date"])
holidays = pd.read_csv(f"{DATA_DIR}/holidays_events.csv", parse_dates=["date"])
oil = pd.read_csv(f"{DATA_DIR}/oil.csv", parse_dates=["date"])
stores = pd.read_csv(f"{DATA_DIR}/stores.csv")



# sort by store, item, date
df = train.sort_values(["store_nbr", "item_nbr", "date"]).reset_index(drop=True)

# optional: chọn một subset nhỏ để phát triển prototype
sample_stores = df["store_nbr"].unique()[:20]
sample_items = df["item_nbr"].unique()[:200]

df = df[df["store_nbr"].isin(sample_stores) & df["item_nbr"].isin(sample_items)].copy()
df = df.sort_values(["store_nbr", "item_nbr", "date"]).reset_index(drop=True)

# xử lý unit_sales âm (do return) → đơn giản nhất: clamp về 0
df["unit_sales"] = df["unit_sales"].clip(lower=0)

df.head()



DATA_DIR = "/kaggle/input/favorita-unzipped-csv"

# giả sử bạn đã có df được tạo từ train, ví dụ:
# df = train.sort_values(["store_nbr", "item_nbr", "date"]).reset_index(drop=True)

# 1) đọc lại oil từ file gốc cho chắc
oil = pd.read_csv(os.path.join(DATA_DIR, "oil.csv"), parse_dates=["date"])
print("oil columns:", oil.columns)
print(oil.head())

# 2) merge trực tiếp oil vào df
# sau merge, df sẽ có cột 'dcoilwtico'
df = df.merge(oil, on="date", how="left")

# 3) đổi tên 'dcoilwtico' → 'oil_price'
df = df.rename(columns={"dcoilwtico": "oil_price"})

# 4) kiểm tra đã có oil_price chưa
print(df[["date", "oil_price"]].head())

# 5) xử lý missing bằng interpolate + bfill + ffill
s = df["oil_price"].interpolate()
s = s.bfill()
s = s.ffill()
df["oil_price"] = s

# 6) tạo time features
df["dow"] = df["date"].dt.dayofweek
df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year

df.head()



# đảm bảo sort đúng
df = df.sort_values(["store_nbr", "item_nbr", "date"]).reset_index(drop=True)

# optional: giới hạn bớt để chạy nhanh lúc dev
sample_stores = df["store_nbr"].unique()[:5]
sample_items = df["item_nbr"].unique()[:50]

df = df[df["store_nbr"].isin(sample_stores) & df["item_nbr"].isin(sample_items)].copy()
df = df.sort_values(["store_nbr", "item_nbr", "date"]).reset_index(drop=True)

# unit_sales không âm
df["unit_sales"] = df["unit_sales"].clip(lower=0)

# tạo lag features
for lag in [1, 7, 14]:
    df[f"lag_{lag}"] = df.groupby(["store_nbr", "item_nbr"])["unit_sales"].shift(lag)

# bỏ các dòng đầu tiên bị NaN do lag
df = df.dropna(subset=["lag_1", "lag_7", "lag_14"]).reset_index(drop=True)

df.head()



# bỏ bớt cột dầu dư thừa nếu còn
for col in ["oil_price_x", "oil_price_y"]:
    if col in df.columns:
        df = df.drop(columns=col)

# đảm bảo kiểu dữ liệu onpromotion là số, không phải string
df["onpromotion"] = df["onpromotion"].fillna(0).astype("int8")

# kiểm tra nhanh
print(df[["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion", "oil_price", "lag_1", "lag_7", "lag_14"]].head())



from sklearn.metrics import mean_squared_error
import numpy as np
import lightgbm as lgb

# đảm bảo sort đúng
df = df.sort_values(["store_nbr", "item_nbr", "date"]).reset_index(drop=True)

# Chia train / validation theo thời gian (120 ngày cuối làm validation)
VALID_DAYS = 120  # bạn có thể dùng 60, 90, 120 tùy sức máy

cutoff_date = df["date"].max() - pd.Timedelta(days=VALID_DAYS)
print("Cutoff date:", cutoff_date)

train_df = df[df["date"] <= cutoff_date].copy()
valid_df = df[df["date"] > cutoff_date].copy()

print("Train size:", train_df.shape)
print("Valid size:", valid_df.shape)


# Chọn features cho baseline
features = [
    "dow", "weekofyear", "month", "year",
    "lag_1", "lag_7", "lag_14",
    "onpromotion", "oil_price"
]

X_train = train_df[features]
y_train = train_df["unit_sales"]
X_valid = valid_df[features]
y_valid = valid_df["unit_sales"]

# Khởi tạo và train LightGBM
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    objective="regression",
    random_state=42
)

model.fit(X_train, y_train)

# Dự báo cho validation
valid_pred = model.predict(X_valid)

rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))
print("Baseline validation RMSE:", rmse)



valid_df = valid_df.copy()
valid_df["baseline_p50"] = valid_pred
valid_df["baseline_p90"] = valid_df["baseline_p50"] * 1.2  # tạm thời nhân 1.2 làm p90

valid_df[["date", "store_nbr", "item_nbr", "unit_sales", "baseline_p50", "baseline_p90"]].head()



# 4.1 Holiday_text: mô tả khoảng cách tới ngày lễ gần nhất

keep_types = ["Holiday", "Additional", "Transfer"]
holidays_small = holidays[holidays["type"].isin(keep_types)].copy()
holidays_small["description_clean"] = holidays_small["description"].str.replace("_", " ")

def nearest_holiday_text(current_date, hol_df):
    diffs = hol_df["date"] - current_date
    if diffs.isna().all():
        return ""
    idx = diffs.abs().idxmin()
    h_row = hol_df.loc[idx]
    
    diff_days = (h_row["date"] - current_date).days
    diff_weeks = diff_days // 7
    holiday_name = h_row["description_clean"]
    holiday_day_str = h_row["date"].date().isoformat()
    
    if abs(diff_weeks) > 8:
        return ""
    
    if diff_weeks > 0:
        return f"The holiday {holiday_name} on {holiday_day_str} is {diff_weeks} weeks after this date."
    elif diff_weeks < 0:
        return f"The holiday {holiday_name} on {holiday_day_str} was {-diff_weeks} weeks before this date."
    else:
        return f"This date is exactly the holiday {holiday_name} on {holiday_day_str}."

valid_df["holiday_text"] = valid_df["date"].apply(lambda d: nearest_holiday_text(d, holidays_small))


# 4.2 Weather_text: thời tiết synthetic theo tháng

def weather_summary(date):
    m = date.month
    if m in [12, 1, 2]:
        return "Weather is hot and humid during this period."
    if m in [6, 7, 8]:
        return "Cooler season with increased rainfall."
    return "Temperatures are mild with no extreme conditions expected."

valid_df["weather_text"] = valid_df["date"].apply(weather_summary)


# 4.3 Promo_text: mô tả onpromotion

def promo_summary(onpromo):
    if onpromo > 0:
        return "A promotion is active for this product on this date."
    return "No promotion activity is observed for this product on this date."

valid_df["promo_text"] = valid_df["onpromotion"].fillna(0).astype(int).apply(promo_summary)


# 4.4 Product_text: mô tả synthetic cho item

def product_summary(row):
    return f"This item with ID {row['item_nbr']} is a common grocery product frequently purchased in this store."

valid_df["product_text"] = valid_df.apply(product_summary, axis=1)

valid_df[["date", "store_nbr", "item_nbr", "holiday_text", "weather_text", "promo_text", "product_text"]].head()



final_cols = [
    "date",
    "store_nbr",
    "item_nbr",
    "unit_sales",
    "onpromotion",
    "oil_price",
    "baseline_p50",
    "baseline_p90",
    "holiday_text",
    "weather_text",
    "promo_text",
    "product_text"
]

final_df = valid_df[final_cols].sort_values(["store_nbr", "item_nbr", "date"]).reset_index(drop=True)

print(final_df.head())
print(final_df.info())

final_path = "/kaggle/working/final_favorita_llm_dataset.csv"
final_df.to_csv(final_path, index=False)
print("Saved final dataset to:", final_path)



import pandas as pd
import numpy as np

final_path = "/kaggle/working/final_favorita_llm_dataset.csv"

df = pd.read_csv(final_path, parse_dates=["date"])
print(df)
print(df.info())
print(df["weather_text"].unique())




# chỉ dùng những quan sát có baseline và sales dương
df = df[(df["baseline_p50"] > 0) & (df["unit_sales"] > 0)].copy()

# lambda = log(y / y_hat_base)
df["lambda"] = np.log(df["unit_sales"] / df["baseline_p50"])

df[["date", "store_nbr", "item_nbr", "unit_sales", "baseline_p50", "lambda"]].head()



def build_prompt(row):
    return (
        "You are a forecasting assistant that corrects retail demand forecasts.\n"
        f"Baseline forecast: {row['baseline_p50']:.3f} units.\n"
        f"Holiday info: {row['holiday_text']}\n"
        f"Weather info: {row['weather_text']}\n"
        f"Promotion info: {row['promo_text']}\n"
        f"Product description: {row['product_text']}\n"
        "Predict the log scaling factor lambda that corrects the baseline forecast towards the true sales."
    )

df["prompt"] = df.apply(build_prompt, axis=1)

df[["prompt", "lambda"]].head(2)



print(df.head(1))
print(df.info())



from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

print("Train size:", train_df.shape, "Test size:", test_df.shape)



!pip install -q "transformers>=4.30.0" --no-deps



import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)



MODEL_NAME = "distilbert-base-uncased"  # hoặc roberta-base, deberta-v3-base, ...

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class LLMForecastDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.prompts = df["prompt"].tolist()
        self.labels = df["lambda"].astype("float32").tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        text = self.prompts[idx]
        label = self.labels[idx]

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.float32)
        return item

train_dataset = LLMForecastDataset(train_df, tokenizer)
test_dataset  = LLMForecastDataset(test_df,  tokenizer)



model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1
)
# báo cho model biết đây là bài toán regression
model.config.problem_type = "regression"



os.environ["WANDB_DISABLED"] = "true"

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds.reshape(-1)
    labels = labels.reshape(-1)
    mse = ((preds - labels) ** 2).mean()
    return {"mse": mse}

training_args = TrainingArguments(
    output_dir="/kaggle/working/llm_forecast_ckpt",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=50,
)


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds.reshape(-1)
    labels = labels.reshape(-1)
    mse = ((preds - labels)**2).mean()
    return {"mse": mse}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()


# 1) Dự đoán lambda_hat cho test set
pred_outputs = trainer.predict(test_dataset)
lambda_hat = pred_outputs.predictions.reshape(-1)

test_df = test_df.copy()
test_df["lambda_hat"] = lambda_hat

# 2) Tạo forecast_final = baseline * exp(lambda_hat)
test_df["forecast_final"] = test_df["baseline_p50"] * np.exp(test_df["lambda_hat"])

test_df[["date","store_nbr","item_nbr","unit_sales","baseline_p50","forecast_final","lambda","lambda_hat"]].head()



from sklearn.metrics import mean_squared_error

y_true = test_df["unit_sales"].values
y_base = test_df["baseline_p50"].values
y_llm  = test_df["forecast_final"].values

rmse_base = np.sqrt(mean_squared_error(y_true, y_base))
rmse_llm  = np.sqrt(mean_squared_error(y_true, y_llm))

mape_base = (np.abs(y_true - y_base) / (y_true + 1e-6)).mean() * 100
mape_llm  = (np.abs(y_true - y_llm)  / (y_true + 1e-6)).mean() * 100

print("RMSE baseline:      ", rmse_base)
print("RMSE LLM-corrected: ", rmse_llm)
print("MAPE baseline:      ", mape_base)
print("MAPE LLM-corrected: ", mape_llm)


