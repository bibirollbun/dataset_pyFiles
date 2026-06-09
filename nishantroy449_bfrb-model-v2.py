def extract_features(sequence, demographics):
    df = sequence  # Already a pandas DataFrame
    demo = demographics[demographics['sequence_id'] == df['sequence_id'].iloc[0]].iloc[0]
    row = {"sequence_id": df['sequence_id'].iloc[0]}

    for col in ["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"]:
        row[f"{col}_mean"] = df[col].mean()
        row[f"{col}_std"] = df[col].std()
        row[f"{col}_min"] = df[col].min()
        row[f"{col}_max"] = df[col].max()

    for col in [c for c in df.columns if c.startswith("thm_")]:
        row[f"{col}_mean"] = df[col].mean()
        row[f"{col}_std"] = df[col].std()

    for tof_prefix in range(1, 6):
        tof_cols = [f"tof_{tof_prefix}_v{i}" for i in range(64) if f"tof_{tof_prefix}_v{i}" in df.columns]
        if len(tof_cols) > 0:
            tof_data = df[tof_cols].replace(-1, np.nan)
            row[f"tof_{tof_prefix}_mean"] = tof_data.mean(axis=1).mean()
            row[f"tof_{tof_prefix}_std"] = tof_data.std(axis=1).mean()

    for col in ['age', 'sex', 'handedness', 'adult_child', 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']:
        row[col] = demo[col] if col in demo else np.nan

    return pd.DataFrame([row])



print(features_df.shape)
features_df.head()



from sklearn.preprocessing import LabelEncoder

# Load correct sequence-level targets
labels = train_df[["sequence_id", "behavior"]].drop_duplicates(subset="sequence_id")
labels = labels[labels["sequence_id"].isin(features_df["sequence_id"])]

# Ensure alignment
features_df = features_df[features_df["sequence_id"].isin(labels["sequence_id"])].reset_index(drop=True)
labels = labels.set_index("sequence_id").loc[features_df["sequence_id"]].reset_index()

# Encode
le = LabelEncoder()
y = le.fit_transform(labels["behavior"])

# ---------------------------
# Show scores
# ---------------------------
print()
for fold, score in enumerate(scores, 1):
    print(f"Fold {fold} Macro F1: {score:.4f}")
print(f"\nAverage Macro F1: {np.mean(scores):.4f}")



# Save the trained model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)


