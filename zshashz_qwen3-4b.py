import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Qwen/Qwen3-4B",local_dir="./")
os.system("rm -r -f .cache")




