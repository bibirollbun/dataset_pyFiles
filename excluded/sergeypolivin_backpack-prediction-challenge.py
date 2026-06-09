import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

DATA_PATH = "/kaggle/input/playground-series-s5e2/"


train_df = pd.read_csv(DATA_PATH + "train.csv")
train_df.info()


test_df = pd.read_csv(DATA_PATH + "test.csv")
test_df.info()


sns.set_theme(style="ticks", palette="pastel")


train_df.hist()
plt.tight_layout()
plt.show()


train_df.describe().T


sns.countplot(data=train_df, x="Brand")
plt.tight_layout()
plt.show()


sns.countplot(data=train_df, x="Material")
plt.tight_layout()
plt.show()


sns.countplot(data=train_df, x="Size")
plt.tight_layout()
plt.show()


sns.countplot(data=train_df, x="Compartments")
plt.tight_layout()
plt.show()


sns.countplot(data=train_df, x="Laptop Compartment")
plt.tight_layout()
plt.show()


sns.countplot(data=train_df, x="Waterproof")
plt.tight_layout()
plt.show()


sns.countplot(data=train_df, x="Style")
plt.tight_layout()
plt.show()


sns.countplot(data=train_df, x="Color")
plt.tight_layout()
plt.show()


g = sns.catplot(
    data=train_df, kind="bar",
    x="Material", y="Weight Capacity (kg)", hue="Size",
    errorbar=None, palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("", "Weight Capacity (kg)")
g.legend.set_title("")


sns.boxplot(x="Material", y="Weight Capacity (kg)",
            hue="Laptop Compartment", palette=["m", "g"],
            data=train_df)
sns.despine(offset=10, trim=True)


g = sns.catplot(
    data=train_df, kind="bar",
    x="Size", y="Price", hue="Compartments",
    errorbar=None, palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("", "Price")
g.legend.set_title("")


sns.boxplot(x="Material", y="Price",
            hue="Waterproof", palette=["m", "g"],
            data=train_df)
sns.despine(offset=10, trim=True)


cols_mapping = {
    "Brand": "brand",
    "Material": "material",
    "Size": "size",
    "Compartments": "compartments",
    "Laptop Compartment": "laptop_compartment",
    "Waterproof": "waterproof",
    "Style": "style",
    "Color": "color",
    "Weight Capacity (kg)": "weight_capacity",
}

train_df = train_df.rename(columns=cols_mapping)
train_df = train_df.rename(columns={"Price": "price"})

test_df = test_df.rename(columns=cols_mapping)


train_df.columns, test_df.columns


def identify_missing_values(data):
    """Performs missing values computation.

    Function computes a number and share of missing values
    in DataFrame columns which have NaN-values present
    and displays data types of such columns.

    Parameters
    ----------
    data : DataFrame
        DataFrame which needs to be checked for missing values.

    Returns
    -------
    DataFrame or None
        DataFrame with column names/their data types, number of
        missing values and shares of NaN-values in such columns,
        or None if no missing values have been found.
    """
    # Verifying the presence of missing values
    miss_vals_num = data.isnull().sum()[data.isnull().sum() > 0]
    if miss_vals_num.empty:
        return None

    # Creating a table with numbers of missing values
    cols = {"missing_count": miss_vals_num.values}
    nans_df = pd.DataFrame(data=cols, index=miss_vals_num.index).sort_values(
        by="missing_count", ascending=False
    )

    # Adding shares of missing values
    nans_df["missing_fraction"] = nans_df["missing_count"] / data.shape[0]
    nans_df["missing_fraction"] = nans_df["missing_fraction"].round(4)

    # Adding data types
    nans_df["dtype"] = data[nans_df.index].dtypes
    nans_df = nans_df[["dtype", "missing_count", "missing_fraction"]]

    return nans_df


identify_missing_values(data=train_df)


identify_missing_values(data=test_df)


train_df["compartments"] = train_df["compartments"].astype(int)
test_df["compartments"] = test_df["compartments"].astype(int)


cat_features = train_df.select_dtypes(include="object").columns.tolist()
num_features = ["weight_capacity"]
cat_features, num_features


cat_imputer = SimpleImputer(strategy="most_frequent")
num_imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()
ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)

num_pipe = make_pipeline(num_imputer, scaler)
cat_pipe = make_pipeline(cat_imputer, ohe)


preprocessor = make_column_transformer(
    (cat_pipe, cat_features),
    (num_pipe, num_features),
    verbose_feature_names_out=False,
    remainder="passthrough",
)
preprocessor


train_df = train_df.dropna()


features_prepared = train_df.drop(["id", "price"], axis=1)


features = preprocessor.fit_transform(features_prepared)
features_test = preprocessor.transform(test_df)
target = train_df["price"].values


features_train, features_valid, target_train, target_valid = train_test_split(
    features,
    target,
    train_size=0.8,
    random_state=1,
)


x_train = torch.tensor(features_train).float()
y_train = torch.tensor(target_train).float()

x_valid = torch.tensor(features_valid).float()
y_valid = torch.tensor(target_valid).float()


train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_valid, y_valid)


torch.manual_seed(42)
train_dl = DataLoader(train_ds, 8, shuffle=True)
valid_dl = DataLoader(valid_ds, 8, shuffle=False)


hidden_units = [64, 32]
input_size = x_train.shape[1]
all_layers = []
for hidden_unit in hidden_units:
    layer = nn.Linear(input_size, hidden_unit)
    all_layers.append(layer)
    all_layers.append(nn.LayerNorm(hidden_unit))
    all_layers.append(nn.ReLU())
    input_size = hidden_unit
all_layers.append(nn.Linear(hidden_units[-1], 1))
model = nn.Sequential(*all_layers)
model


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
model.apply(init_weights)


model = model.to("cuda")


loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    "min",
    patience=2,
)


num_epochs = 10
pbar = tqdm(range(num_epochs))
torch.manual_seed(42)
for epoch in pbar:
    loss_hist_train = 0
    loss_hist_valid = 0
    lr = optimizer.param_groups[0]["lr"]
    model.train()
    for i, (x_batch, y_batch) in enumerate(train_dl):
        pbar.set_postfix({"batch": f"{i + 1}/{len(train_dl)}", "lr": lr})
        x_batch, y_batch = x_batch.to("cuda"), y_batch.to("cuda")
        pred = model(x_batch)[:, 0]
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_hist_train += loss.item()

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in valid_dl:
            x_batch, y_batch = x_batch.to("cuda"), y_batch.to("cuda")
            pred = model(x_batch)[:, 0]
            loss = loss_fn(pred, y_batch)
            loss_hist_valid += loss.item()
    
    mse_train = loss_hist_train / len(train_dl)
    mse_valid = loss_hist_valid / len(valid_dl)
    scheduler.step(mse_valid)
    rmse_train = mse_train**(0.5)
    rmse_valid = mse_valid**(0.5)
    print(f"Epoch {epoch + 1}: RMSE=(train={rmse_train:.4f}, valid={rmse_valid:.4f})")


x_test = torch.tensor(features_test).float()


with torch.no_grad():
    predictions_test = model(x_test.to("cuda"))[:, 0]
predictions_test = predictions_test.cpu()
predictions_test


data = {
    "id": test_df.id,
    "Price": predictions_test
}
submission = pd.DataFrame(data)
submission


# Saving the submission
submission.to_csv('submission.csv', index=False)

# Displaying the success message
print("The submission has been successfully saved.")


!head submission.csv

