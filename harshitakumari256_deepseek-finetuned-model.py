import io
import os
import shutil
import subprocess

import pandas as pd
import polars as pl


import kaggle_evaluation.konwinski_prize_inference_server


!pip cache purge
!pip install --upgrade transformers torch


! pip install --no-index --find-links /kaggle/input/bitsandbytes bitsandbytes


instance_count = None

def get_number_of_instances(num_instances: int) -> None:
    """ The very first message from the gateway will be the total number of instances to be served.
    You don't need to edit this function.
    """
    global instance_count
    instance_count = num_instances


# first_prediction = True
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# import torch
# model=None
# tokenizer=None

# def predict(problem_statement: str, repo_archive: io.BytesIO, pip_packages_archive: io.BytesIO, env_setup_cmds_templates: list[str]) -> str:
#     """ Replace this function with your inference code.
#     Args:
#         problem_statement: The text of the git issue.
#         repo_path: A BytesIO buffer path with a .tar containing the codebase that must be patched. The gateway will make this directory available immediately before this function runs.
#         pip_packages_archive: A BytesIO buffer path with a .tar containing the wheel files necessary for running unit tests.
#         env_setup_cmds_templates: Commands necessary for installing the pip_packages_archive.
#     """
#     global model,tokenizer,first_prediction
#     if model is None or tokenizer is None:
#         model_name="/kaggle/input/deepseek-github-agent/pytorch/default/1"
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         model = AutoModelForCausalLM.from_pretrained(model_name)

#     inputs = tokenizer(problem_statement,return_tensors="pt",padding=True,truncation=True,max_length=512)
#     inputs = {key: value.to(model.device) for key, value in inputs.items()}
#     outputs = model.generate(
#         inputs["input_ids"],
#         attention_mask=inputs.get("attention_mask"),
#         max_length=2048    
#     )

    # predicted_str=tokenizer.decode(outputs[0],skip_special_tokens=True)
    # if not first_prediction:
    #     return None  

    # with open('repo_archive.tar', 'wb') as f:
    #     f.write(repo_archive.read())
    # repo_path = 'repo'
    # if os.path.exists(repo_path):
    #     shutil.rmtree(repo_path)
    # shutil.unpack_archive('repo_archive.tar', extract_dir=repo_path)
    # os.remove('repo_archive.tar')

    # """
    # Unpack pip_packages if you want to run unit tests on your patch.
    # Note that editing unit tests with your patch -- even to add valid tests -- can cause your submission to be flagged as a failure.
    # Most of the relevant repos use pytest for running tests. You will almost certainly need to run only a subset of the unit tests to avoid running out of inference time.
    # """
    # with open('pip_packages_archive.tar', 'wb') as f:
    #     f.write(pip_packages_archive.read())
    # pip_packages_path = '/path/to/pip_packages'
    # if os.path.exists(pip_packages_path):
    #     shutil.rmtree(pip_packages_path)
    # shutil.unpack_archive('pip_packages_archive.tar', extract_dir=pip_packages_path)
    # os.remove('pip_packages_archive.tar')

    # # Get env setup cmds by setting the pip_packages_path
    # env_setup_cmds = [cmd.format(pip_packages_path=pip_packages_path) for cmd in env_setup_cmds_templates]

    # Run env setup for the repo
   #  subprocess.run(
   #      "\n".join(env_setup_cmds),
   #      shell=True,
   #      executable="/bin/bash",
   #      cwd=repo_path,
   #  )

   #  first_prediction = False
   # # Instead of a valid diff, let's just submit a generic string. This will definitely fail.
   #  return predicted_str


# inference_server = kaggle_evaluation.konwinski_prize_inference_server.KPrizeInferenceServer(
#     get_number_of_instances,   
#     predict
# )

# if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
#     inference_server.serve()
# else:
#     inference_server.run_local_gateway(
#         data_paths=(
#             '/kaggle/input/konwinski-prize/',  # Path to the entire competition dataset
#             '/kaggle/tmp/konwinski-prize/',   # Path to a scratch directory for unpacking data.a_zip.
#         ),
#         use_concurrency=True,  # This can safely be disabled for purposes of local testing if necessary.
#     )


import io
import os
import shutil
import subprocess
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def format_patch(predicted_text: str) -> Optional[str]:
    """Convert model output into a valid patch format."""
    try:
        if not predicted_text.startswith("diff --git"):
            lines = predicted_text.split("\n")
            formatted_lines = []
            in_code_block = False
            
            for line in lines:
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block:
                    formatted_lines.append(line)
            
            if formatted_lines:
                return "\n".join(formatted_lines)
        return predicted_text
    except Exception:
        return None

def cleanup_resources():
    """Clean up temporary files and free memory."""
    if os.path.exists('repo_archive.tar'):
        os.remove('repo_archive.tar')
    if os.path.exists('pip_packages_archive.tar'):
        os.remove('pip_packages_archive.tar')
    torch.cuda.empty_cache()

model = None
tokenizer = None
first_prediction = True

def predict(problem_statement: str, repo_archive: io.BytesIO, 
           pip_packages_archive: io.BytesIO, env_setup_cmds_templates: list[str]) -> Optional[str]:
    """Improved prediction function with proper error handling and resource management."""
    global model, tokenizer, first_prediction
    
    try:
        if model is None or tokenizer is None:
            model_name = "/kaggle/input/deepseek-github-agent/pytorch/default/1"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16
            )
        prompt = f"""Given this GitHub issue, generate a patch to fix it:
{problem_statement}

Generate the patch in unified diff format."""

        inputs = tokenizer(prompt, return_tensors="pt", 
                         padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_length=2048,
            temperature=0.7,
            top_p=0.95,
            num_return_sequences=1
        )

        predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        patch = format_patch(predicted_text)

        if not patch:
            return None

        with open('repo_archive.tar', 'wb') as f:
            f.write(repo_archive.read())
            
        repo_path = 'repo'
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)
        shutil.unpack_archive('repo_archive.tar', extract_dir=repo_path)

        try:

            result = subprocess.run(
                f"cd {repo_path} && git apply --check -",
                input=patch.encode(),
                shell=True,
                capture_output=True
            )
            if result.returncode != 0:
                return None  
        except Exception:
            return None

        return patch

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None
    
    finally:
        cleanup_resources()


inference_server = kaggle_evaluation.konwinski_prize_inference_server.KPrizeInferenceServer(
    get_number_of_instances,   
    predict
)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/konwinski-prize/',
            '/kaggle/tmp/konwinski-prize/',
        ),
        use_concurrency=True,
    )

