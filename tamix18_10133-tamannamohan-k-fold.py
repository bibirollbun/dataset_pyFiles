import numpy as np
import pandas as pd
from sklearn import model_selection


def create_folds(data: pd.DataFrame, num_splits: int = 5) -> pd.DataFrame:

    # Initialize kfold column
    data["kfold"] = -1

    # Determine number of bins using Sturges' rule
    num_bins = int(np.floor(1 + np.log2(len(data))))
    print(f"Creating {num_splits} folds with {num_bins} bins.")

    # Create bins for stratification
    data["bins"] = pd.cut(data["Pawpularity"], bins=num_bins, labels=False)

    # Initialize StratifiedKFold
    kf = model_selection.StratifiedKFold(
        n_splits=num_splits, shuffle=True, random_state=42
    )

    # Assign folds
    for fold, (_, val_idx) in enumerate(kf.split(X=data, y=data["bins"].values)):
        data.loc[val_idx, "kfold"] = fold

    # Drop temporary bins column
    data = data.drop("bins", axis=1)

    return data



df = pd.read_csv("../input/petfinder-pawpularity-score/train.csv")
print(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns.")
df.head()



df_5 = create_folds(df.copy(), num_splits=5)
df_10 = create_folds(df.copy(), num_splits=10)



df_5.to_csv("train_5folds.csv", index=False)
df_10.to_csv("train_10folds.csv", index=False)

print("Saved train_5folds.csv and train_10folds.csv successfully.")



df_5.head()





