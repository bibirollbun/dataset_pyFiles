import pandas as pd

def blend_submissions(weight_dict, output_path):
    dfs = []

    for path, weight in weight_dict.items():
        df = pd.read_csv(path)

        # Validaci贸n defensiva
        assert {"id", "diagnosed_diabetes"}.issubset(df.columns), \
            f"Formato incorrecto en {path}"

        df["weighted_pred"] = df["diagnosed_diabetes"] * weight
        dfs.append(df[["id", "weighted_pred"]])

    # Merge acumulativo
    blended = dfs[0]
    for df in dfs[1:]:
        blended = blended.merge(df, on="id", how="inner", suffixes=("", "_dup"))
        blended["weighted_pred"] += blended["weighted_pred_dup"]
        blended.drop(columns="weighted_pred_dup", inplace=True)

    # Normalizaci贸n
    total_weight = sum(weight_dict.values())
    blended["diagnosed_diabetes"] = blended["weighted_pred"] / total_weight

    blended[["id", "diagnosed_diabetes"]].to_csv(output_path, index=False)
    print(f"Blending final guardado en {output_path}")



weight_dict = {
    "/kaggle/input/diabetes-prediction-vault/submission.csv": 1.000,  # 0.7073
    "/kaggle/input/diabetes-prediction-vault/submission (1).csv": 0.988,  # 0.6989
    "/kaggle/input/test-oof-preds-s5p12/test_xgboost.csv": 0.982,  # 0.6949
    "/kaggle/input/test-oof-preds-s5p12/test_lightgbm.csv": 0.988,  # 0.6988
    "/kaggle/input/test-oof-preds-s5p12/test_catboost.csv": 0.987,  # 0.6981
}

blend_submissions(weight_dict, output_path="submission.csv")


