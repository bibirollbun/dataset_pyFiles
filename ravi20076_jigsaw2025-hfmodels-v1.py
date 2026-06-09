


%%time 

!pip install huggingface_hub -q
import shutil, os
from gc import collect 
from huggingface_hub import snapshot_download


%%time 

snapshot_download(
    repo_id   = "HuggingFaceTB/SmolLM2-360M-Instruct",
    local_dir = "HuggingFaceTB/SmolLM2-360M-Instruct",
    local_files_only = False,  
)


%%time 

snapshot_download(
    repo_id   = "HuggingFaceTB/SmolLM2-135M-Instruct",
    local_dir = "HuggingFaceTB/SmolLM2-135M-Instruct",
    local_files_only = False,  
)


%%time 

snapshot_download(
    repo_id   = "MiniLLM/MiniPLM-Qwen-200M",
    local_dir = "MiniLLM/MiniPLM-Qwen-200M",
    local_files_only = False,  
)

