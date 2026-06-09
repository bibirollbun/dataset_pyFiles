from huggingface_hub import snapshot_download

# Download the model to /kaggle/working/
model_path = snapshot_download(
    repo_id="openai/gpt-oss-20b",
    local_dir="/kaggle/working",
    local_dir_use_symlinks=False,
    revision="main",  # or a specific branch/tag
    ignore_patterns=["metal/*", "original/*"]
)

print(f"Model downloaded to: {model_path}")

