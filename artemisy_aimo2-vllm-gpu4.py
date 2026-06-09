# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#pip freeze > /kaggle/working/pre_installed_packages.txt


#pip install -r /kaggle/input/required-packages4/installed_packages.txt


#pip freeze > /kaggle/working/post_installed_packages.txt


'''
import pkg_resources

# âœ… Load pre-installed and post-installed package lists
with open("/kaggle/working/pre_installed_packages.txt", "r") as f:
    pre_installed = {pkg.split("==")[0]: pkg for pkg in f.read().splitlines()}  # {package_name: full_name}

with open("/kaggle/working/post_installed_packages.txt", "r") as f:
    post_installed = {pkg.split("==")[0]: pkg for pkg in f.read().splitlines()}  # {package_name: full_name}

# âœ… Find newly installed packages (not in pre-installed)
new_installed = {k: v for k, v in post_installed.items() if k not in pre_installed}

# âœ… Find updated packages (same package but different version)
updated_packages = {k: v for k, v in post_installed.items() if k in pre_installed and v != pre_installed[k]}

# âœ… Combine both into one list
final_installed = sorted(list(new_installed.values()) + list(updated_packages.values()))

# âœ… Save to a new file
with open("/kaggle/working/new_installed_packages.txt", "w") as f:
    f.write("\n".join(final_installed))

print("âœ… new_installed_packages.txt created successfully!")
'''


#!mkdir -p /kaggle/working/packages/nvidia-libs
#!cp -r /usr/lib/x86_64-linux-gnu/libcudnn* /kaggle/working/packages/nvidia-libs/
#!cp -r /usr/local/lib/python3.10/dist-packages/nvidia* /kaggle/working/packages/


'''
import os
import shutil
import importlib.util

# Read package names from file
with open("/kaggle/working/new_installed_packages.txt", "r") as f:
    packages = [line.split("==")[0] for line in f.read().splitlines()]  # Remove versions

# Create destination directory
os.makedirs("/kaggle/working/packages", exist_ok=True)

# Copy Python-importable packages
for package in packages:
    try:
        spec = importlib.util.find_spec(package)
        if spec and spec.origin:
            package_dir = os.path.dirname(spec.origin)
            print(f"ğŸ“¦ Copying Python package: {package} from {package_dir}")
            shutil.copytree(package_dir, f"/kaggle/working/packages/{package}", dirs_exist_ok=True)
        else:
            print(f"âš ï¸� Skipping {package}: Could not find a Python module for it.")
    except Exception as e:
        print(f"â�Œ Error processing {package}: {e}")

# Manually copy NVIDIA system libraries
cuda_paths = [
    "/usr/lib/x86_64-linux-gnu/",
    "/usr/local/cuda/lib64/",
    "/usr/local/lib/python3.10/dist-packages/"
]

nvidia_packages = [pkg for pkg in packages if "nvidia" in pkg]

for nvidia_pkg in nvidia_packages:
    for path in cuda_paths:
        matching_files = [f for f in os.listdir(path) if nvidia_pkg in f]
        if matching_files:
            os.makedirs(f"/kaggle/working/packages/{nvidia_pkg}", exist_ok=True)
            for file in matching_files:
                shutil.copy(os.path.join(path, file), f"/kaggle/working/packages/{nvidia_pkg}/")
            print(f"âœ… Copied NVIDIA package: {nvidia_pkg} from {path}")
        else:
            print(f"âš ï¸� NVIDIA package {nvidia_pkg} not found in {path}")

print("âœ… All packages copied successfully!")
'''


'''
import os
import shutil

# Define source and destination paths
source_path = "/usr/local/lib/python3.10/dist-packages"
dest_path = "/kaggle/working/packages"

# Ensure the destination exists
os.makedirs(dest_path, exist_ok=True)

# NVIDIA packages to copy
nvidia_packages = [
    "nvidia-cublas-cu12",
    "nvidia-cuda-cupti-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-cufft-cu12",
    "nvidia-curand-cu12",
    "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu12",
    "nvidia-ml-py",
    "nvidia-nccl-cu12",
    "nvidia-nvjitlink-cu12",
    "nvidia-nvtx-cu12"
]

# Copy NVIDIA libraries
for package in nvidia_packages:
    package_path = os.path.join(source_path, package)
    
    if os.path.exists(package_path):
        print(f"ğŸ“¦ Copying NVIDIA package: {package} from {package_path}")
        shutil.copytree(package_path, os.path.join(dest_path, package), dirs_exist_ok=True)
    else:
        print(f"âš ï¸� NVIDIA package {package} not found in {source_path}")

print("âœ… NVIDIA packages copying completed.")
'''


#mkdir /kaggle/working/packages


#cp -r /kaggle/input/last-package1/packages/* /kaggle/working/packages/


'''
import os
import shutil
import subprocess
import importlib.util

# Read package names (strip versions)
with open("/kaggle/working/new_installed_packages.txt", "r") as f:
    packages = [line.split("==")[0] for line in f.read().splitlines()]

# Create destination directory
os.makedirs("/kaggle/working/packages", exist_ok=True)

# âœ… First, copy standard Python packages
for package in packages:
    try:
        spec = importlib.util.find_spec(package)
        if spec and spec.origin:
            package_dir = os.path.dirname(spec.origin)
            print(f"ğŸ“¦ Copying Python package: {package} from {package_dir}")
            shutil.copytree(package_dir, f"/kaggle/working/packages/{package}", dirs_exist_ok=True)
    except Exception as e:
        print(f"âš ï¸� Skipping {package}: Could not find as an importable module.")

# âœ… Second, explicitly copy known system-related packages
system_packages = [
    "aiohttp-cors",
    "compressed-tensors",
    "lm-format-enforcer",
    "opencensus-context",
    "partial-json-parser",
    "prometheus-fastapi-instrumentator",
    "py-spy",
    "python-dotenv",
]

for package in system_packages:
    try:
        result = subprocess.run(["pip", "show", package], capture_output=True, text=True)
        for line in result.stdout.split("\n"):
            if line.startswith("Location:"):
                package_dir = line.split(":", 1)[1].strip()
                package_path = os.path.join(package_dir, package.replace("-", "_"))  # Adjust naming

                if os.path.exists(package_path):
                    print(f"ğŸ“¦ Copying system package: {package} from {package_path}")
                    shutil.copytree(package_path, f"/kaggle/working/packages/{package}", dirs_exist_ok=True)
                else:
                    print(f"âš ï¸� {package} was found but could not locate its directory in {package_dir}")
    except Exception as e:
        print(f"âš ï¸� Skipping {package}: Error finding package. {e}")

print("âœ… All packages copied successfully!")
'''


#mkdir -p /kaggle/working/packages
#cp /kaggle/input/package-set-final3/packages/*.whl /kaggle/working/packages/


#pip install vllm #--no-deps --quiet


#pip install --upgrade scikit-learn==1.3.2


#pip install matplotlib==3.8.1


package_path = '/kaggle/input/last-packages2/packages'


import sys
sys.path.append(package_path)


#==1.5.2
#msgspec==0.19.0


#pip install msgspec --quiet


#pip install --no-cache-dir torch==2.1.2+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet


#pip install --no-cache-dir triton==2.1.0


pip install --no-cache-dir vllm==0.6.6 --quiet  # âœ… Use a stable version


#pip install -U pynvml --quiet


#!pip show vllm torch triton pynvml | grep Requires


'''
# List of installed packages
installed_packages = """
torch-2.5.1
torchvision-0.20.1
triton-3.1.0
aiohttp-cors-0.7.0
airportsdata-20241001
astor-0.8.1
blake3-1.0.4
colorful-0.5.6
compressed-tensors-0.8.1
depyf-0.18.0
diskcache-5.6.3
distlib-0.3.9
fastapi-0.115.8
gguf-0.10.0
httptools-0.6.4
interegular-0.3.3
lark-1.2.2
lm-format-enforcer-0.10.9
mistral_common-1.5.3
msgspec-0.19.0
nvidia-cublas-cu12-12.4.5.8
nvidia-cuda-cupti-cu12-12.4.127
nvidia-cuda-nvrtc-cu12-12.4.127
nvidia-cuda-runtime-cu12-12.4.127
nvidia-cudnn-cu12-9.1.0.70
nvidia-cufft-cu12-11.2.1.3
nvidia-curand-cu12-10.3.5.147
nvidia-cusolver-cu12-11.6.1.9
nvidia-cusparse-cu12-12.3.1.170
nvidia-ml-py-12.570.86
nvidia-nccl-cu12-2.21.5
nvidia-nvjitlink-cu12-12.4.127
nvidia-nvtx-cu12-12.4.127
opencensus-0.11.4
opencensus-context-0.1.3
outlines-0.1.11
outlines_core-0.1.26
partial-json-parser-0.2.1.1.post5
prometheus-fastapi-instrumentator-7.0.2
py-spy-0.4.0
pycountry-24.6.1
python-dotenv-1.0.1
starlette-0.45.3
uvicorn-0.34.0
uvloop-0.21.0
virtualenv-20.29.1
vllm-0.6.6
watchfiles-1.0.4
xformers-0.0.28.post3
xgrammar-0.1.11
"""

# Save the list to a file
file_path = "/kaggle/working/installed_packages.txt"
with open(file_path, "w") as f:
    f.write(installed_packages.strip())

'''


import transformers


#!mkdir -p /kaggle/working/packages
#!pip download -r /kaggle/input/required-packages4/installed_packages.txt -d /kaggle/working/packages


#import pynvml
#print("pynvml Path:", os.path.dirname(pynvml.__file__))


#ls -l /usr/local/lib/python3.10/dist-packages | grep pynvml


#pip install autoawq --quiet


#!mkdir -p /kaggle/working/packages
#!cp -r /usr/local/lib/python3.10/dist-packages/triton /kaggle/working/packages/
#!cp -r /usr/local/lib/python3.10/dist-packages/torch /kaggle/working/packages/
#!cp -r /usr/local/lib/python3.10/dist-packages/pynvml* /kaggle/working/packages/
#!cp -r /usr/local/lib/python3.10/dist-packages/vllm /kaggle/working/packages/
#!cp -r /usr/local/lib/python3.10/dist-packages/autoawq /kaggle/working/packages/


#import os

#old_path = "/kaggle/working/packages/compressed-tensors"
#new_path = "/kaggle/working/packages/compressed_tensors"

#if os.path.exists(old_path):
#    os.rename(old_path, new_path)
#    print(f"âœ… Renamed {old_path} -> {new_path}")
#else:
#    print(f"â�Œ Path {old_path} does not exist.")


#!cd /kaggle/working && tar -czf packages.tar.gz packages


#!cd /kaggle/working && zip -r packages.zip packages





#import sys
#sys.exit(0)


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"#,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


#import transformers
#print(transformers.__version__)


import torch

print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("CUDA Version:", torch.version.cuda)
else:
    print("â�Œ No GPU detected! Make sure GPU is enabled in Kaggle settings.")


#!pip install --no-index --find-links=/kaggle/input/package-set-final3/packages -r /kaggle/input/required-packages4/installed_packages.txt


import pynvml


import torch


os.environ["PYTHONPATH"] = package_path + ':' + package_path + 'vllm'
os.environ["PYTHONPATH"] = package_path


from vllm import LLM, SamplingParams


!pip uninstall -y pynvml


import torch
import vllm
import triton
import pynvml
print("Torch version:", torch.__version__)
print("vLLM version:", vllm.__version__)
print("Triton version:", triton.__version__)


sys.path.append(package_path)


import os





import os
import sys
import shutil

package_root = "/kaggle/input/last-package2/packages/"  # Your original package location
writable_path = "/kaggle/working/packages/"  # Writable directory

# Ensure the writable directory exists
os.makedirs(writable_path, exist_ok=True)

# âœ… Copy `compressed-tensors` to `/kaggle/working/packages/` and rename it
source_path = os.path.join(package_root, "compressed-tensors")
destination_path = os.path.join(writable_path, "compressed_tensors")  # Correct naming

if os.path.exists(source_path) and not os.path.exists(destination_path):
    shutil.copytree(source_path, destination_path)
    print(f"ğŸ“¦ Copied and renamed {source_path} -> {destination_path}")


os.environ["PYTHONPATH"] = (
    package_path + "vllm:" +  # Add vLLM
    package_path + "compressed_tensors:" +  # Add compressed_tensors
    os.environ.get("PYTHONPATH", "")  # Keep existing PYTHONPATH
)

# âœ… Also add to sys.path
sys.path.append(package_path)
sys.path.append(package_path + "vllm")
sys.path.append(package_path + "compressed_tensors")


os.environ["LD_LIBRARY_PATH"] = "/kaggle/working/packages:" + os.environ.get("LD_LIBRARY_PATH", "")


# Check if CUDA is available
print("CUDA Available:", torch.cuda.is_available())

# Check cuDNN version
print("cuDNN Version:", torch.backends.cudnn.version())

# Check CUDA device
print("CUDA Device:", torch.cuda.get_device_name(0))


model_path = "/kaggle/input/deepseek-r1/transformers/deepseek-aideepseek-r1-distill-qwen-14b-awq-neody/1"
MAX_NUM_SEQS = 16
MAX_MODEL_LEN = 8192

#llm = LLM(
#    model=model_path,
#    tensor_parallel_size=2,  # âœ… Adjust for 2 GPUs (Kaggle default)
#    gpu_memory_utilization=0.90,  # âœ… Prevents OOM errors
#    dtype="half",  # âœ… FP16 to reduce memory usage
#    trust_remote_code=True,
#    enforce_eager=True,  # âœ… Disable compilation optimizations
#)

#llm = LLM(
#    model_path,
#    #dtype="half",                # The data type for the model weights and activations
#    max_num_seqs=MAX_NUM_SEQS,   # Maximum number of sequences per iteration. Default is 256
#    max_model_len=MAX_MODEL_LEN, # Model context length
#    trust_remote_code=True,      # Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer
#    tensor_parallel_size=2,      # The number of GPUs to use for distributed execution with tensor parallelism
#    gpu_memory_utilization=0.95, # The ratio (between 0 and 1) of GPU memory to reserve for the model
#    seed=2024,
#    dtype="float16",
#    enforce_eager=True,  # âœ… Force disable JIT compilation
#    disable_custom_all_reduce=True,  # âœ… Prevents NCCL issues
#    disable_async_output_proc=True,  # âœ… Disables async processing
#)

llm = LLM(
    model_path,
    enforce_eager=True,
    max_num_seqs=MAX_NUM_SEQS,   # Maximum number of sequences per iteration. Default is 256
    max_model_len=MAX_MODEL_LEN, # Model context length
    trust_remote_code=True,      # Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer
    tensor_parallel_size=2,      # The number of GPUs to use for distributed execution with tensor parallelism
    gpu_memory_utilization=0.95,
    dtype="float16",
    disable_custom_all_reduce=True,
    disable_async_output_proc=True,
    compilation_config={"max_capture_size": 0}  # âœ… Disable CUDA graph captures
)


def generate_response_batch(prompts, max_new_tokens=MAX_MODEL_LEN):
    """ Runs multiple prompts at once and returns responses. """
    
    # âœ… Set sampling parameters for controlled generation
    sampling_params = SamplingParams(
        temperature=0.3,  # Adjusts randomness
        max_tokens=max_new_tokens,
        skip_special_tokens=True
    )
    
    # âœ… Run batched inference
    outputs = llm.generate(prompts, sampling_params)
    
    # âœ… Extract generated text from model responses
    responses = [output.outputs[0].text for output in outputs]
    return responses


import re
def extract_boxed_text(text):
    pattern = r'\\boxed{(\d+)}'  # âœ… Extracts only numbers inside \boxed{}
    matches = re.findall(pattern, text)
    return matches[-1] if matches else "-1"  # âœ… Returns last found answer or "-1"


'''
from tqdm import tqdm

# âœ… Load the dataset
#df = pd.read_csv("/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv")
df = pd.read_csv("/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv")
batch_size = 2  # âœ… Adjust based on GPU memory
results = []
no_question = len(df)
# âœ… Process in batches
for i in tqdm(range(0, no_question, batch_size), total=no_question // batch_size):
    batch = df.iloc[i:i+batch_size]  # Select batch

    # âœ… Format batch prompts
    prompts = [
        f"""Please carefully read the problem statement first to ensure you fully understand its meaning and key points.
        Solve the following math problem through deep reasoning, ensuring all steps are correct. 
        Carefully analyze the problem and perform all necessary calculations.
        Finally, return the answer modulo 1000 and enclose it in \\boxed{{}} like \\boxed{{180}}.
        Problem: {problem}"""
        for problem in batch["problem"]
    ]

    # âœ… Run batch inference with `vLLM`
    responses = generate_response_batch(prompts)

    # âœ… Extract answers
    for j, response in enumerate(responses):
        raw_output = response.strip()  # Remove extra spaces
      #  print(f"ğŸ“� Raw Output {j+1}: {raw_output}")  # âœ… Debugging print

        # âœ… Extract boxed answer
        answer = extract_boxed_text(raw_output)
        
        results.append({"id": batch.iloc[j]["id"], "answer": answer})

# âœ… Convert to DataFrame
results_df = pd.DataFrame(results)

# âœ… Save results for submission

# âœ… Save results
#results_df.to_csv("submission.csv", index=False)
#print("âœ… Submission saved as submission.csv")

# âœ… Convert to DataFrame
results_df = pd.DataFrame(results)

# âœ… Save results for submission
results_df.to_csv("submission.csv", index=False)
'''


#print(results_df)


pd.read_csv(
    '/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv'
).drop('answer', axis=1).to_csv('reference.csv', index=False)


import polars as pl  # Kaggle requires polars DataFrame for submissions

def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    id_ = id_.item(0)
    question = question.item(0)

   # prompts = f"""Please carefully read the problem statement first to ensure you fully understand its meaning and key points.
  #      Solve the following math problem through deep reasoning, ensuring all steps are correct. 
  #      Carefully analyze the problem and perform all necessary calculations.
  #      Finally, return the answer modulo 1000 and enclose it in \\boxed{{}} like \\boxed{{180}}.
  #      Problem: {question}"""
    prompts = f"""
You are a math expert renowned for solving complex problems with clear, logical, and step-by-step reasoning. Your task is to solve the following math problem accurately. Follow these instructions:

1. Read the Problem Carefully: Understand all details of the problem.
2. Plan and Solve: Think through the problem logically, performing all necessary calculations. (Do not output your internal reasoning.)
3. Final Answer Only: After solving, output ONLY the final answer in the exact format: \\boxed{{<number>}}.
4. Modulo Condition: Ensure that the final answer is given modulo 1000.
5. No Extra Text: Do not include any commentary or intermediate steps in your final output.

Example:
Problem: What is 123 + 456?
Work: 123 + 456 = 579.
Final Answer: \\boxed{{579}}

Now solve:
Problem: {question}
"""
    
    # âœ… Generate response from the model
    response = generate_response_batch([prompts])[0]

    # âœ… Extract numeric answer
    answer = extract_boxed_text(response)

    # âœ… Ensure answer is an integer
    try:
        answer = int(answer) % 1000
    except:
        answer = 210  # Default if extraction fails

    return pl.DataFrame({'id': id_, 'answer': answer})


import kaggle_evaluation.aimo_2_inference_server

# âœ… Load the Kaggle evaluation server
inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(predict)

# âœ… If running locally, test on reference dataset
if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.run_local_gateway(
        (
        "/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv",
       )
    )
else:
    inference_server.serve()




