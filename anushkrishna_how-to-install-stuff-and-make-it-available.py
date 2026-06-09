requirements = [
    "pip install sentence-transformers --quiet",
    "pip install vllm --quiet",
]

with open("/kaggle/working/input_requirements.txt", "w") as f:
    f.write("\n".join(requirements))

print("input_requirements.txt created")


import subprocess

def download_packages(requirements_file, target_dir="/kaggle/working"):
    with open(requirements_file, "r") as f:
        for line in f:
            cmd = line.strip()
            if not cmd.startswith("pip install"):
                continue

            if "-r" in cmd or "--requirement" in cmd:
                raise ValueError("Nested requirements files not supported.")

            # Convert install -> download
            download_cmd = cmd.replace("install", "download")
            download_cmd = download_cmd.replace("-U", "").replace("--upgrade", "")
            download_cmd += f" -d {target_dir}"

            print(f"Downloading: {download_cmd}")
            subprocess.check_call(download_cmd.split())

download_packages("/kaggle/working/input_requirements.txt")




install_script_path = "/kaggle/working/install_requirements.sh"

install_script = """#!/bin/bash

SCRIPT_DIR=$(dirname "$0")

export PIP_DISABLE_PIP_VERSION_CHECK=true
export PIP_FIND_LINKS=$SCRIPT_DIR
export PIP_NO_INDEX=true

# Install from wheels
for file in "$SCRIPT_DIR"/*.whl; do
    pip install "$file"
done

# GitHub packages last
export PIP_NO_BUILD_ISOLATION=false
pip install git+https://github.com/QwenLM/Qwen.git

echo "All packages installed."
"""

with open(install_script_path, "w") as f:
    f.write(install_script)

print("install_requirements.sh created")



# !chmod +x /kaggle/input/YOUR_DATASET/install_requirements.sh
# !/kaggle/input/YOUR_DATASET/install_requirements.sh


