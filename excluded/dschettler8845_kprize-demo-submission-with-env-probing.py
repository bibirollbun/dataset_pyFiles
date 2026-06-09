# Install local library just in case...
# !pip install -q /kaggle/input/konwinski-prize/kprize_setup/kprize-1.0.0-py3-none-any.whl --no-index --find-links /kaggle/input/konwinski-prize/kprize_setup/pip_packages

import io
import os
import shutil

import pandas as pd
import polars as pl

import kaggle_evaluation.konwinski_prize_inference_server


instance_count = None

def get_number_of_instances(num_instances: int) -> None:
    """ The very first message from the gateway will be the total number of instances to be served.
    You don't need to edit this function.
    """
    global instance_count
    instance_count = num_instances


import astroid as local_astroid
LOCAL_ASTROID_VERSION = local_astroid.__version__
print(LOCAL_ASTROID_VERSION)
    

def predict(problem_statement: str, repo_archive: io.BytesIO) -> str:
    """ Replace this function with your inference code.
    Args:
        problem_statement: The text of the git issue.
        repo_path: A BytesIO buffer path with a .tar containing the codebase that must be patched. The gateway will make this directory available immediately before this function runs.
    """
    import astroid
    global LOCAL_ASTROID_VERSION

    # If the astroid version in the env is different
    # than this code is executing in the correct environment.
    if LOCAL_ASTROID_VERSION==astroid.__version__:
        return None
    # Basically the more negative the score the more likely we have astroid
    # examples and the env has been configured properly
    else:
        # Everytime we submit this we should get -0.01
        return "Hello World"


inference_server = kaggle_evaluation.konwinski_prize_inference_server.KPrizeInferenceServer(
    get_number_of_instances,   
    predict
)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/konwinski-prize/',  # Path to the entire competition dataset
            '/kaggle/tmp/konwinski-prize/',   # Path to a scratch directory for unpacking data.a_zip.
        )
    )

