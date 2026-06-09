!pip install git+https://github.com/S-G-mathematics/genuity_os.git --quiet


import pandas as pd
import numpy as np

df = pd.read_csv("/kaggle/input/genuityxethos/real_0.6.csv")
df.head(2)





def remove(df):
    df = df.copy()

    # Replace all null-like values
    df = df.replace([None, "None", np.nan, pd.NaT, "", " ", "nan", "NaN"], 0)

    # Convert numeric columns safely
    for col in df.columns:
        if col not in ["Symbol", "Series"]:  # keep categorical safe
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df



def submit(synthetic_final):
    required_cols = [
        "row_id_column_name", "Symbol", "Prev Close", "Open", "High", "Low",
        "Last", "Close", "VWAP", "Volume", "Turnover", "Trades",
        "Deliverable Volume", "%Deliverble", "t"
    ]

    # 1. Remove Series
    if "Series" in synthetic_final.columns:
        synthetic_final = synthetic_final.drop(columns=["Series"])

    # 2. Keep exactly 3322 rows
    synthetic_final = synthetic_final.head(3322)

    # 3. Sort by t ascending
    synthetic_final = synthetic_final.sort_values(by="t").reset_index(drop=True)

    # 4. Add row_id_column_name
    synthetic_final.insert(0, "row_id_column_name", range(0, len(synthetic_final)))

    # 5. Reorder columns
    synthetic_final = synthetic_final[required_cols]

    # 6. Final null cleaning (important)
    synthetic_final = synthetic_final.replace([None, np.nan, pd.NaT, "None", "", "nan"], 0)

    # 7. Convert numeric columns safely
    for col in synthetic_final.columns:
        if col not in ["Symbol"]:  # categorical remain str
            synthetic_final[col] = pd.to_numeric(synthetic_final[col], errors="coerce").fillna(0)

    # 8. Save CSV
    synthetic_final.to_csv("submission.csv", index=False)
    print("")
    print("")
    print("Submission File Saved Successfully 笨能")

    return synthetic_final



from genuity_os.data_processor.data_preprocess import TabularPreprocessor

pre = TabularPreprocessor(
    scaler_type="standard",
    encoding_strategy="onehot",
)

result = pre.fit_transform(df)
X = result["preprocessed"]


from genuity_os.core_generator.ctgan.ctgan.utils.api import CTGANAPI

ctgan = CTGANAPI()

losses = ctgan.fit(
    data=X.values,
    continuous_cols=list(range(len(result["continuous"].columns))),
    categorical_cols=list(range(
        len(result["continuous"].columns),
        X.shape[1]
    )),
    epochs=400
)



pre.save_preprocessor("prep.joblib")


synthetic = ctgan.generate(3322)
synthetic


import pandas as pd
from genuity_os.data_processor.data_postprocess import TabularPostprocessor

# Load postprocessor
post = TabularPostprocessor(preprocessor_path="/kaggle/working/prep.joblib")

# Build DataFrame for synthetic before decoding
synthetic_df_raw = pd.DataFrame(synthetic, columns=post.feature_names)

# 1) Inverse transforms (scale, PCA, categorical)
synthetic_decoded = post.inverse_transform_modified_data(synthetic_df_raw)

# 2) Force original columns order
original_cols = [
    'Symbol', 'Series', 'Prev Close', 'Open', 'High', 'Low', 'Last', 'Close',
    'VWAP', 'Volume', 'Turnover', 'Trades', 'Deliverable Volume',
    '%Deliverble', 't'
]

synthetic_final = synthetic_decoded[original_cols]

synthetic_final = synthetic_final.fillna(0)

synthetic_final_clean = remove(synthetic_final)
submission_file = submit(synthetic_final_clean)


import torch
import torch.nn as nn
import numpy as np

# Convert X to numpy
data = X.values.astype(np.float32)

# Create dataset
def create_dataset(data, steps=30):
    Xs, ys = [], []
    for i in range(len(data) - steps):
        Xs.append(data[i:i+steps])
        ys.append(data[i+steps])
    return np.array(Xs), np.array(ys)

seqX, seqY = create_dataset(data, 30)

cut = int(len(seqX) * 0.6)
X_train, X_test = seqX[:cut], seqX[cut:]
y_train = seqY[:cut]

# Convert to PyTorch tensors
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
X_test  = torch.tensor(X_test)



input_dim = X.shape[1]
hidden_dim = 128

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



epochs = 30
batch_size = 32

for epoch in range(epochs):
    perm = torch.randperm(X_train.size(0))
    epoch_loss = 0

    for i in range(0, X_train.size(0), batch_size):
        idx = perm[i:i+batch_size]
        batch_x = X_train[idx]
        batch_y = y_train[idx]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if(epoch%10==0):
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
print("Done training for Long Short Term Memory RNN")


def fix_ranges(df):
    df["Prev Close"] = df["Prev Close"].clip(0, 2000)
    df["Open"]       = df["Open"].clip(0, 2000)
    df["High"]       = df["High"].clip(0, 2000)
    df["Low"]        = df["Low"].clip(0, 2000)
    df["Last"]       = df["Last"].clip(0, 2000)
    df["Close"]      = df["Close"].clip(0, 2000)
    df["VWAP"]       = df["VWAP"].clip(0, 2000)

    df["Volume"]     = df["Volume"].clip(10000, 70000000)
    df["Turnover"]   = df["Turnover"].clip(10000, 100000000)     # <= 100 million
    df["Trades"]     = df["Trades"].clip(0, 200000)
    df["Deliverable Volume"] = df["Deliverable Volume"].clip(0, 10000000)
    df["%Deliverble"] = df["%Deliverble"].clip(0, 1)
    df["t"] = df["t"].clip(0, 5000)

    return df



# 1. Predict with PyTorch
model.eval()
with torch.no_grad():
    predictions = model(X_test).numpy()

# 2. Convert to DataFrame
import pandas as pd
synthetic_df_raw = pd.DataFrame(predictions, columns=result["preprocessed"].columns)

# 3. Inverse transform using Genuity postprocessor
from genuity_os.data_processor.data_postprocess import TabularPostprocessor
post = TabularPostprocessor(preprocessor_path="/kaggle/working/prep.joblib")

synthetic_decoded = post.inverse_transform_modified_data(synthetic_df_raw)

# 4. Reorder columns
original_cols = [
    'Symbol', 'Series', 'Prev Close', 'Open', 'High', 'Low', 'Last', 'Close',
    'VWAP', 'Volume', 'Turnover', 'Trades', 'Deliverable Volume',
    '%Deliverble', 't'
]

synthetic_final = synthetic_decoded[original_cols].fillna(0)

# 5. Clean + Submit
synthetic_final_clean = remove(synthetic_final)
synthetic_final = synthetic_decoded[original_cols]
synthetic_fixed = fix_ranges(synthetic_final)
synthetic_fixed = synthetic_fixed.fillna(0)

synthetic_final_clean = remove(synthetic_fixed)
submission_file = submit(synthetic_final_clean)



!pip install ctgan --quiet


from ctgan import TVAE

X_num = X.values   # your preprocessed matrix

tvae = TVAE(
    epochs=500,
    embedding_dim=256,
    compress_dims=(512, 256),
    decompress_dims=(256, 512),
)

tvae.fit(X_num)

synthetic = tvae.sample(len(X_num))


synthetic = tvae.sample(3322)

synthetic_df_raw = pd.DataFrame(
    synthetic,
    columns=result["preprocessed"].columns
)


from genuity_os.data_processor.data_postprocess import TabularPostprocessor
post = TabularPostprocessor(preprocessor_path="/kaggle/working/prep.joblib")

synthetic_decoded = post.inverse_transform_modified_data(synthetic_df_raw)


original_cols = [
    'Symbol', 'Series', 'Prev Close', 'Open', 'High', 'Low', 'Last', 'Close',
    'VWAP', 'Volume', 'Turnover', 'Trades', 'Deliverable Volume',
    '%Deliverble', 't'
]

synthetic_final = synthetic_decoded[original_cols]
def fix_ranges(df):
    df = df.copy()

    df['Prev Close'] = df['Prev Close'].clip(0, 2000)
    df['Open']       = df['Open'].clip(0, 2000)
    df['High']       = df['High'].clip(0, 2000)
    df['Low']        = df['Low'].clip(0, 2000)
    df['Last']       = df['Last'].clip(0, 2000)
    df['Close']      = df['Close'].clip(0, 2000)
    df['VWAP']       = df['VWAP'].clip(0, 2000)

    df['Volume']     = df['Volume'].clip(10000, 100000000)
    df['Turnover']   = df['Turnover'].clip(10000, 200000000)
    df['Trades']     = df['Trades'].clip(0, 200000)
    df['Deliverable Volume'] = df['Deliverable Volume'].clip(0, 20000000)

    df['%Deliverble'] = df['%Deliverble'].clip(0.0, 1.0)
    df['t'] = df['t'].clip(0, 3000)

    return df

synthetic_final = fix_ranges(synthetic_final)
def fix_categories(df):
    df = df.copy()
    df["Symbol"] = "ADANIPORTS"
    df["Series"] = "EQ"
    df["%Deliverble"] = df["%Deliverble"].fillna(0.5)
    return df

synthetic_final = fix_categories(synthetic_final)
synthetic_final = synthetic_final.fillna(0)
import numpy as np
import pandas as pd

def submit(synthetic_final):
    required_cols = [
        "row_id_column_name", "Symbol", "Prev Close", "Open", "High", "Low",
        "Last", "Close", "VWAP", "Volume", "Turnover", "Trades",
        "Deliverable Volume", "%Deliverble", "t"
    ]

    # 1. Remove Series
    if "Series" in synthetic_final.columns:
        synthetic_final = synthetic_final.drop(columns=["Series"])

    # 2. Keep exactly 3322 rows
    synthetic_final = synthetic_final.head(3322)

    # 3. Sort by t ascending (required!)
    synthetic_final = synthetic_final.sort_values(by="t").reset_index(drop=True)

    # 4. Add row_id_column_name
    synthetic_final.insert(0, "row_id_column_name", range(0, len(synthetic_final)))

    # 5. Reorder columns
    synthetic_final = synthetic_final[required_cols]

    # 6. Final cleanup
    synthetic_final = synthetic_final.replace(
        [None, np.nan, pd.NaT, "None", "", "nan"], 0
    )

    # 7. Convert numeric columns safely
    for col in synthetic_final.columns:
        if col not in ["Symbol"]:  # keep Symbol string
            synthetic_final[col] = pd.to_numeric(synthetic_final[col], errors="coerce").fillna(0)

    # 8. Save CSV
    synthetic_final.to_csv("submission.csv", index=False)
    print("\nSubmission File Saved Successfully 笨能")
    return synthetic_final
submission_file = submit(synthetic_final)


submission_file.head(2)

