from pathlib import Path
from typing import Set

import pandas as pd
import typer
from sklearn.model_selection import GroupKFold
from transformers import set_seed


def load_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the input DataFrame to create a structured dataset for misconception analysis.

    Args:
        df: Input DataFrame containing answer and misconception data

    Returns:
        Processed DataFrame with restructured data
    """
    datas: List[Dict[str, Union[str, int]]] = []
    for col in ["A", "B", "C", "D"]:
        answer_col = f"Answer{col}Text"
        misconception_col = f"Misconception{col}Id"

        for _, row in df.iterrows():
            if pd.isna(row[misconception_col]):
                continue

            correct_col = row["CorrectAnswer"]
            correct_answer_col = f"Answer{correct_col}Text"

            data = {
                "Answer": row[answer_col],
                "Correct": row[correct_answer_col],
                "MisconceptionId": int(row[misconception_col]),
            }

            if correct_col == col:
                continue

            row_dict = row.to_dict()
            # Remove unnecessary columns
            for c in ["A", "B", "C", "D"]:
                row_dict.pop(f"Misconception{c}Id")
                row_dict.pop(f"Answer{c}Text")

            data.update(row_dict)
            datas.append(data)

    return pd.DataFrame(datas)


def calculate_misconception_overlap(train_misid: Set[int], val_misid: Set[int]) -> float:
    return 1.0 - len(train_misid & val_misid) / len(val_misid)


def split_dataset(df: pd.DataFrame, n_splits: int, group_col: str) -> pd.DataFrame:
    gkf = GroupKFold(n_splits=n_splits)
    df["fold"] = -1

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(df, groups=df[group_col])):
        if group_col == "QuestionId":
            # Calculate overlap for original data
            train_misid = set(df.loc[train_idx]["MisconceptionId"])
            val_misid = set(df.loc[val_idx]["MisconceptionId"])
            overlap = calculate_misconception_overlap(train_misid, val_misid)
            print(f"Fold {fold_idx} misconception overlap: {overlap:.3f}")

        df.loc[val_idx, "fold"] = fold_idx

    return df



def main() -> None:
    set_seed(42)

    # Load misconception mapping
    mapping = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv")

    # Process original training data
    train_df = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/train.csv")
    df = load_data(train_df).reset_index(drop=True)
    df["original"] = True
    df = split_dataset(df, 5, "QuestionId")

    # Process synthetic data
    synthetic_df = pd.read_csv("/kaggle/input/additional-questions/additional_question.csv").reset_index(drop=True)
    synthetic_df["original"] = False
    synthetic_df = split_dataset(synthetic_df, 5, "SubjectName")

    # Combine datasets
    combined_df = pd.concat([df, synthetic_df], ignore_index=True)
    combined_df["fold"] = combined_df["fold"].astype(int)

    # Validate number of folds
    if combined_df["fold"].nunique() != 5:
        raise ValueError(f"Expected {5} folds, got {combined_df['fold'].nunique()}")

    # Merge with misconception mapping and save
    final_df = combined_df.merge(mapping, on="MisconceptionId")
    output_path = "/kaggle/working/data.csv"
    final_df.to_csv(output_path, index=False)
    print(f"Saved split dataset to {output_path}")


if __name__ == "__main__":
    main()




