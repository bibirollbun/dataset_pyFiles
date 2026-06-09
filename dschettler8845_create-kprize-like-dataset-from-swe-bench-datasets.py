# ############################################################################################### #
# !pip install -q \
#     /kaggle/input/konwinski-prize/kprize_setup/kprize-1.1.0-py3-none-any.whl \
#     --no-index \
#     --find-links /kaggle/input/konwinski-prize/kprize_setup/kprize_setup/pip_packages/kprize
# ############################################################################################### #
# Do this instead of installing because for some reason we can't see the `bundling` part.
# Since we have internet I can always install the dependencies as needed.
import sys; sys.path.insert(0, "/kaggle/input/konwinski-prize/kprize_setup")
# ############################################################################################### #

import io
import os
import shutil
import subprocess
from pathlib import Path
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict

import pandas as pd; pd.options.mode.chained_assignment = None; pd.set_option('display.max_columns', None)
import polars as pl; print(f"\t\t– POLARS VERSION: {pl.__version__}")
import sklearn; print(f"\t\t– SKLEARN VERSION: {sklearn.__version__}")
import numpy as np; print(f"\t\t– NUMPY VERSION: {np.__version__}")

# Built-In Imports (mostly don't worry about these)
from typing import Iterable, Any, Literal, Callable, Generator
from kaggle_datasets import KaggleDatasets
from dataclasses import dataclass
from collections import Counter
from datetime import datetime
from zipfile import ZipFile
from io import StringIO
from glob import glob
import subprocess
import tempfile
import warnings
import requests
import textwrap
import hashlib
import imageio
import IPython
import urllib
import zipfile
import tarfile
import pickle
import random
import shutil
import string
import json
import copy
import math
import time
import gzip
import ast
import sys
import io
import gc
import re
import os

# Rich
from rich import pretty; pretty.install()
from rich.markdown import Markdown
from rich import print as rprint
from rich.console import Console
from rich.style import Style
from rich.live import Live
from rich.text import Text
from rich import inspect
import rich

# --------------------------------------------------------- #
import kaggle_evaluation.konwinski_prize_inference_server
# --------------------------------------------------------- #


# Rudimentary Paths
BASE_DIR = "/kaggle"
TMP_DIR = os.path.join(BASE_DIR, "tmp")
WORKING_DIR = os.path.join(BASE_DIR, "working")
INPUT_DIR = os.path.join(BASE_DIR, "input")
TEMP_DIR = os.path.join("/tmp")

# Basic Competition Paths
COMP_DIR = os.path.join(INPUT_DIR, "konwinski-prize")
COMP_KAGGLE_EVALUATION_DIR = os.path.join(COMP_DIR, "kaggle_evaluation")
COMP_KPRIZE_SETUP_DIR = os.path.join(COMP_DIR, "kprize_setup")

# Dataset Competition Paths
COMP_DATA_ZIP_PATH = os.path.join(COMP_DIR, "data.a_zip")
COMP_TMP_DIR = os.path.join(TMP_DIR, "konwinski-prize-alt")
COMP_TMP_DATA_DIR = os.path.join(COMP_TMP_DIR, "data")
COMP_DATA_PARQUET_PATH = os.path.join(COMP_TMP_DATA_DIR, "data.parquet")
COMP_CONDA_PACKAGES_DIR = os.path.join(COMP_TMP_DATA_DIR, "conda_packages")
COMP_PIP_PACKAGES_DIR = os.path.join(COMP_TMP_DATA_DIR, "pip_packages")
COMP_REPO_CONFIGS_DIR = os.path.join(COMP_TMP_DATA_DIR, "repo_configs")
COMP_REPOS_DIR = os.path.join(COMP_TMP_DATA_DIR, "repos")

# SWE Dataset Paths ... https://huggingface.co/datasets/...
HF_SWE_BENCH_PROVIDER = "princeton-nlp"
HF_SWE_BENCH_PATH = os.path.join(HF_SWE_BENCH_PROVIDER, "SWE-bench")
HF_SWE_BENCH_LITE_PATH = os.path.join(HF_SWE_BENCH_PROVIDER, "SWE-bench_Lite")
HF_SWE_BENCH_VERIFIED_PATH = os.path.join(HF_SWE_BENCH_PROVIDER, "SWE-bench_Verified")

def load_kprize_df(add_local_paths: bool = True) -> pd.DataFrame:
    """Loader function"""
    if not os.path.isfile(COMP_DATA_PARQUET_PATH):    
        # Make the directory to unzip to
        os.makedirs(COMP_TMP_DIR, exist_ok=True)
        
        # Open and extract the zip file
        with ZipFile(COMP_DATA_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(COMP_TMP_DIR)
    _df = pd.read_parquet(COMP_DATA_PARQUET_PATH)

    try:
        if add_local_paths:
            _df.insert(1, "local_pip_packages_path", _df.instance_id.apply(lambda x: os.path.join(COMP_PIP_PACKAGES_DIR , x)))
            _df.insert(1, "local_repo_path", _df.instance_id.apply(lambda x: os.path.join(COMP_REPOS_DIR, f"repo__{x}")))
    except:
        print(f"Could not add local path using {COMP_REPOS_DIR} as root competition directory path.")
    return _df
    

# Competition dataset for comparison
kprize_df = load_kprize_df(add_local_paths=True)

# Load the huggingface datasets (put in order so the biggest one is last)
#   - Test with just the small one
hf_datasets = {
    "swe_bench_lite": load_dataset(HF_SWE_BENCH_LITE_PATH),          # N_EX = 300 + 23 = 323
    #"swe_bench_verified": load_dataset(HF_SWE_BENCH_VERIFIED_PATH),  # N_EX = 500
    #"swe_bench": load_dataset(HF_SWE_BENCH_PATH),                    # N_EX = 19008 + 2294 + 225 = 21527
}

# Let's see 'em
rich.print("\n\nKPRIZE DATASET:\n")
display(kprize_df)

rich.print("\n\n\n\nSWE BENCH DATASETS:\n")
for ds_name, ds in hf_datasets.items(): 
    rich.print(f"\n\n\n\n[bold]{ds_name}[/bold]")
    display(ds)


def read_json_file(
    file_path: str | Path,
    encoding: str = 'utf-8',
    force_jsonl: bool = False
) -> dict | list | list[dict]:
    """
    Read JSON or JSONL files with custom encoding and format options.
    
    Args:
        file_path (str | Path): 
            Path to the JSON/JSONL file
        encoding (str, optional): 
            File encoding (default: utf-8)
        force_jsonl (bool, optional): 
            Force reading as JSONL format
        
    Returns:
        Contents of the file as a dict, list, or list of dicts
    """
    # The file must exist
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding=encoding) as f:

        # If doing JSONL we perform json.loads on each line and return
        if force_jsonl:
            return [json.loads(line.strip()) for line in f if line.strip()]
            
        # Otherwise we load the entire thing as JSON and only fallback to JSONL if that fails.
        try:
            return json.load(f)
        except json.JSONDecodeError:
            # If regular JSON fails, try JSONL
            f.seek(0)
            lines = f.readlines()
            
            try:
                return [json.loads(line.strip()) for line in lines if line.strip()]
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"File is neither valid JSON nor JSONL: {str(e)}", 
                    e.doc, 
                    e.pos
                )


def create_instances_dir_data(
    hf_datasets: dict[str, DatasetDict],
    instances_path: str = "task_instances",
    dataset_specifier_str: str = "{ds_name}-{split_name}",
    jsonl_filename: str = "tasks.jsonl"
) -> None:
    """Convert Hugging Face datasets into JSONL files organized in directories.
    
    This function takes a dictionary of Hugging Face datasets and saves each split
    as a JSONL file in a dedicated directory structure. The directory names are
    formatted using the dataset name and split name.
   
    Args:
        hf_datasets (str, datasets.dataset_dict.DatasetDict): 
            Dictionary mapping dataset names to DatasetDict objects
        instances_path (str, optional): 
            Base directory to store the JSONL files
        dataset_specifier_str (str, optional): 
            Template string for directory names.
            Must use {ds_name} and {split_name} as placeholders.
        jsonl_filename (str, optional): 
            Name of the output JSONL file in each directory
       
    Example:
        >>> datasets = {'squad': squad_dataset, 'conll': conll_dataset}
        >>> create_instances_dir_data(datasets)
        # Creates structure like:
        # task_instances/
        #   squad-train/
        #     tasks.jsonl
        #   squad-validation/
        #     tasks.jsonl
        #   conll-train/
        #     tasks.jsonl
        #   ...
    """
    # Iterate through each dataset in the dictionary
    for ds_name, ds_dict in hf_datasets.items():
        # Process each split (train/validation/test) in the dataset
        for split_name, split_hf_dataset in ds_dict.items():
            # Create the output directory path using the template
            jsonl_output_dir = os.path.join(
                instances_path,
                dataset_specifier_str.format(ds_name=ds_name, split_name=split_name),
            )
            
            # Create the output directory if it doesn't exist
            if not os.path.isdir(jsonl_output_dir):
                os.makedirs(jsonl_output_dir, exist_ok=True)
            
            # Save the dataset split as a JSONL file
            split_hf_dataset.to_json(os.path.join(jsonl_output_dir, jsonl_filename))

            # Print an update message
            print(f"Saved {ds_name} to {os.path.join(jsonl_output_dir, jsonl_filename)} ...")


create_instances_dir_data(hf_datasets)


from kprize.bundling.kprize_bundler import KPrizeBundler


def setup_bundler(
    instances_path: str | Path = "task_instances/",
    output_path: str | Path = "dependencies/",
    compress_bundles: bool = False,
    selected_splits: list[str] = None,
) -> KPrizeBundler:
    """Initialize KPrize bundler with specified configuration.
    
    Args:
        instances_path (str | Path): 
            Directory containing task instances
        output_path (str | Path, optional): 
            Directory for output files
        compress_bundles (bool, optional): 
            Whether to compress dataset bundles
        selected_splits (list[str], optional): 
            List of specific splits to process (e.g., 'dev', 'test', etc.)
    """
    # Create the bundler
    bundler = KPrizeBundler(
        instances_dir=Path(instances_path).absolute(),
        output_dir=Path(output_path).absolute(), 
        bundle_compress=compress_bundles,
        split_filter=selected_splits
    )   
    return bundler

# Initialize bundler with default settings
pip_bundler = setup_bundler()
inspect(pip_bundler, help=True, methods=True)


### Would work if we had docker... but we don't...
# def collect_dependencies(
#     bundler: KPrizeBundler,
#     dataset_id: str = "dschettler8845/swe-kprize-cv-assets",
#     upload_kaggle: bool = True,
#     skip_docker: bool = True,
#     create_dataset: bool = True,
#     force_docker_copy: bool = False,
#     clear_instances: list[str] = None,
# ) -> None:
#     """Run dependency collection process with specified settings.
    
#     Args:
#         bundler (KPrizeBundler): 
#             Initialized KPrizeBundler instance
#         dataset_id (str, optional): 
#             The kaggle dataset identifier
#         upload_kaggle (bool, optional): 
#             Whether to upload results to Kaggle
#         skip_docker (bool, optional): 
#             Skip running in Docker container
#         create_dataset (bool, optional): 
#             Whether to create a new dataset
#         force_docker_copy (bool, optional): 
#             Force copy assets to Docker
#         clear_instances (list[str], optional): 
#             List of instance IDs to clear
#     """
#     bundler.run_dependency_collection(
#         upload_to_kaggle=upload_kaggle,
#         skip_run_in_docker=skip_docker,
#         skip_dataset_creation=not create_dataset,
#         clear_instance_ids=clear_instances,
#         dataset_id=dataset_id,
#         force_copy_assets_to_docker=force_docker_copy,
#     )

# # Run collection with default settings
# collect_dependencies(pip_bundler)


def merge_hf_datasets_to_dataframe(
    hf_datasets: dict[str, DatasetDict],
    instances_path: str = "task_instances",
    dataset_specifier_str: str = "{ds_name}-{split_name}",
    jsonl_filename: str = "tasks.jsonl"
) -> pd.DataFrame:
    """Merge multiple Hugging Face datasets into a single pandas DataFrame.
    
    This function takes a dictionary of Hugging Face DatasetDicts, extracts all
    records, and combines them into a single DataFrame. It also adds:
      - A 'dataset' column in the format '{dataset_name}/{split_name}'.
      - A 'jsonl_path' column with the expected location of the corresponding JSONL file.

    Args:
        hf_datasets (dict[str, datasets.DatasetDict]): 
            Dictionary mapping dataset names to DatasetDict objects.
        instances_path (str, optional): 
            Base directory where JSONL files are expected to be stored.
        dataset_specifier_str (str, optional): 
            Template string for directory names. Must include {ds_name} and {split_name}.
        jsonl_filename (str, optional): 
            Name of the JSONL file expected in each directory.

    Returns:
        pd.DataFrame: A DataFrame containing all dataset records, with additional columns.
    
    Example:
        >>> df = merge_hf_datasets_to_dataframe(hf_datasets)
        >>> df.head()
    
    Example Output:
        | repo         | instance_id | base_commit | ... | dataset               | jsonl_path                          |
        |-------------|------------|-------------|-----|----------------------|-----------------------------------|
        | my_repo_1   | 1234       | abcde123    | ... | swe_bench_lite/dev   | task_instances/swe_bench_lite-dev/tasks.jsonl  |
        | my_repo_2   | 5678       | fghij456    | ... | swe_bench/test       | task_instances/swe_bench-test/tasks.jsonl  |
    """
    all_records = []

    # Iterate through datasets and their splits
    for ds_name, ds_dict in hf_datasets.items():
        for split_name, split_dataset in ds_dict.items():
            # Convert Dataset to pandas DataFrame
            df = split_dataset.to_pandas()

            # Add metadata columns
            df["dataset"] = f"{ds_name}/{split_name}"
            df["jsonl_path"] = os.path.join(
                instances_path,
                dataset_specifier_str.format(ds_name=ds_name, split_name=split_name),
                jsonl_filename
            )

            all_records.append(df)

    # Concatenate all DataFrames into one
    merged_df = pd.concat(all_records, ignore_index=True)

    return merged_df

# All merged together
df = merge_hf_datasets_to_dataframe(hf_datasets)

all_unique_repos = sorted(df.repo.unique())
rich.print(all_unique_repos)

all_instances = []
for x in glob(os.path.join("task_instances", "**", "*.jsonl"), recursive=True):
    all_instances.extend(read_json_file(x))

# Sql fluff is stupid
all_instances_without_sqlfluff = [x for x in all_instances if 'sqlfluff' not in x["repo"]]


from kprize.bundling.repo_collector import RepoCollector

_output_dir = Path(os.path.join(TEMP_DIR, "dependencies"))
_collected_deps_dir = Path(os.path.join(_output_dir, "collected"))
_collected_repos_dir = Path(os.path.join(_collected_deps_dir, "repos"))
_collected_instance_repos_dir = Path(os.path.join(_collected_deps_dir, "instance_repos"))
_collected_pip_packages_dir = Path(os.path.join(_collected_deps_dir, "pip_packages"))
_collected_python_packages_dir = Path(os.path.join(_collected_deps_dir, "python3.11"))
_collected_uv_packages_dir = Path(os.path.join(_collected_deps_dir, "uv"))

repo_collector = RepoCollector(
    collected_repos_dir=_collected_repos_dir,
    collected_instance_repos_dir=_collected_instance_repos_dir,
)

# Collect the repos we need
repo_collector.run_repo_collection(all_unique_repos)
repo_collector.run_instance_repo_collection(all_instances_without_sqlfluff)


# UV
from kprize.bundling.uv_downloader import UvDownloader
os.makedirs(_collected_uv_packages_dir, exist_ok=True)
UvDownloader.download(_collected_uv_packages_dir)

# Python Debs
shutil.copytree("/kaggle/input/python-3-11-debs/python3.11/", _collected_python_packages_dir, dirs_exist_ok=True)


_docker_pip_packages_dir = Path("/tmp/docker/pip_packages")
instances_without_pip_packages = []
instances_with_pip_packages = []
for instance in all_instances_without_sqlfluff:
    instance_pip_dir = _docker_pip_packages_dir / instance["instance_id"]
    if instance_pip_dir.exists():
        instances_with_pip_packages.append(instance)
    else:
        instances_without_pip_packages.append(instance)
print(f"Instances without pip packages: {len(instances_without_pip_packages)}")


from kprize.collection.configs.make_configs import make_configs
make_configs(True, "repo_configs")


# from kprize.collection.configs.repo_config import RepoConfig

# def create_repo_config(repo_path: str, specs_dict: dict) -> RepoConfig:
#     repo_name = repo_path.split("/")[-1]

#     return RepoConfig.from_dict(
#         {
#             "repo_name": repo_name,
#             "repo_path": repo_path,
#             "github_url": f"https://github.com/{repo_path}",
#             "log_parser": "parse_log_pytest",
#             "specs": specs_dict,
#         }
#     )

# SPECS_HUMANEVAL = {
#     k: {"python": "3.9", "test_cmd": "python"} 
#     for k in ["1.0"]
# }
# SPECS_DBT_CORE = {
#     k: {"python": "3.9", "packages": "requirements.txt", "install": "python -m pip install -e .", "test_cmd": "pytest -rA"}
#     for k in ["0.13", "0.14", "0.15", "0.16", "0.17", "0.18", "0.19", "0.20", "0.21", "1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7",]
# }
# _MAP_REPO_VERSION_TO_SPECS_PY = {"humaneval": SPECS_HUMANEVAL, "dbt-core": SPECS_DBT_CORE}

# _output_dir = Path("repo_configs")
# for repo_path, specs_dict in _MAP_REPO_VERSION_TO_SPECS_PY.items():
#     repo_name = repo_path.split("/")[-1]
#     output_file = _output_dir / f"{repo_name}.json"
#     config = create_repo_config(repo_path, specs_dict)
#     config.to_json(output_file)


from kprize.collection.configs.repo_config import RepoConfig
# # !pip install anthropic
# import kprize.collection.validation.validator
# import kprize.collection.collector

# collector = kprize.collection.collector.Collector(
#     Path("dependencies/docker/input/kprize-assets/repos"),
    
# )


def _update_repo_config(instances: list[dict[str, Any]]) -> None:
    if len(instances) == 0:
        return
    for repo in all_unique_repos:
        repo = instances[0]["repo"]
        repo_stem = RepoConfig.map_repo_path_to_repo_stem(repo).rsplit("__", 1)[-1]
        repo_config_path = Path("/kaggle/working/repo_configs") / f"{repo_stem}.json"
        if repo_config_path.exists():
            repo_config=RepoConfig.from_json(repo_config_path)
            print(".", end="")
        else:
            print(repo_stem,)
            raise ValueError
            # repo_config=RepoConfig(
            #     repo_name=repo.split("/")[-1],
            #     repo_path=repo,
            #     github_url=f"https://github.com/{repo}",
            #     log_parser=self.default_config.log_parser,
            #     specs={},
            # )
    
        if len(repo_config.specs) > 0:
            print(".", end="")
            default_version, default_specs = max(repo_config.specs.items(), key=lambda x: x[0])
        else:
            print(repo_stem,)
            raise ValueError
    
        is_updated = False
    
        for instance in instances:
            any_fail_to_pass = len(instance.get("FAIL_TO_PASS", [])) > 0
    
            if any_fail_to_pass and "version" in instance and instance["version"] not in repo_config.specs:
                repo_config.specs[instance["version"]] = default_specs
                is_updated = True
            elif any_fail_to_pass and default_version not in repo_config.specs and len(repo_config.specs) == 0:
                repo_config.specs[default_version] = default_specs
                is_updated = True
    
        if is_updated:
            repo_config.to_json(repo_config_path)


# _update_repo_config(all_instances_without_sqlfluff)

# Update with default
for f_path in glob("repo_configs/*.json"):
    config = read_json_file(f_path)
    config["specs"]["default"] = config["specs"][next(iter(config["specs"]))]
    with open(f_path, 'w') as f:
        json.dump(config, f, indent=4)


import argparse
import tomli as tomllib


def current_ms():
    return round(time.time_ns() / 1000000)

def seconds_since(time_ms):
    return (current_ms() - time_ms) / 1000

def create_dir_path(path: Path|str) -> Path:
    if isinstance(path, str):
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def run_commands(cmds: list[str], log_file=None, console_log=True, log_commands=False):
    if log_file:
        print(f"Writing command logs to {log_file.name}")
    commands = '\n'.join(cmds)
    if log_commands:
        print("Running commands:")
        print(commands)
        print("\n")
    process = subprocess.Popen(
        '/bin/bash',
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True)
    out, err = process.communicate(commands)
    if console_log:
        print("Command stderr:")
        print(err)
        print("Command stdout:")
        print(out)
    if log_file:
        log_file.write(err)
        log_file.write(out)

def get_repo_config_name_from_repo_path(instance_id: str) -> str:
    return instance_id.rsplit("-", maxsplit=1)[0]

def convert_pip_install_to_pip_download(cmd_pip_install: str, download_dir_path: Path) -> str:
    """
    Convert a `pip install` command to a `pip download` command
    """
    cmd_pip_download = cmd_pip_install.replace("install", f"download -d {download_dir_path}")
    cmd_pip_download = cmd_pip_download.replace("-e", "").replace("--verbose", "")
    return cmd_pip_download

def get_build_system_requires(toml_path: Path) -> list[str]:
    requirements = []
    toml_data = tomllib.loads(toml_path.read_text())
    # get build-system requires
    build_system = toml_data.get("build-system", None)
    if build_system:
        requirements.extend(build_system.get("requires", []))
        # get build-system build-backend
        build_backend = build_system.get("build-backend", None)
        if build_backend:
            if build_backend == "setuptools.build_meta":
                requirements.append("wheel")
            elif build_backend == "hatchling.build":
                requirements.append("editables")
    # wrap in double quotes for special cases
    # e.g. "setuptools >= 65.5.1", "setuptools_scm[toml]", "cython>=3.0.0, <4"
    return list(map(lambda r: f'"{r}"', requirements))

def download_pip_packages(
    repos_path: str = "dependencies/docker/input/kprize-assets/repos",
    repo_configs_path: str = "dependencies/docker/input/kprize-assets/repo_configs",
    output_path: str = "dependencies/collected/downloaded_pip_packages",
    instances_json_file: str = "task_instances/test/q3-task-instances-all-new-log-parser-test.jsonl",
    limit: int = None,
    python_exec: str = "python3.11"
) -> tuple[list, list, list, list]:
    """
    Run `pip download` for each repo in a given directory of git repos
    
    Args:
        repos_path: Path to repos directory
        repo_configs_path: Path to repo configs directory
        output_path: Path to output downloaded pip packages
        instances_json_file: Path to task instances json file
        limit: Limit number of instances to process
        python_exec: Python executable to use
    
    Returns:
        tuple containing lists of:
        - skipped_repos: repos that were already processed
        - collected_repos: repos successfully processed
        - missing_install_cmd_repos: repos missing install commands
        - missing_download_cmd_repos: repos where download command creation failed
    """
    repos_path = Path(repos_path)
    repo_configs_path = Path(repo_configs_path)
    output_path = Path(output_path)
    instances_json_file = Path(instances_json_file)

    print(" > repos_path:", repos_path)
    print(" > repo_configs_path:", repo_configs_path)
    print(" > output_path:", output_path)
    print(" > instances_json_file:", instances_json_file)
    print(" > limit:", limit)

    if not repos_path.exists():
        raise ValueError(f"Error: repos path does not exist: {repos_path}")
    if not repo_configs_path.exists():
        raise ValueError(f"Error: repo configs path does not exist: {repo_configs_path}")
    if not instances_json_file.exists():
        raise ValueError(f"Error: instances json file does not exist: {instances_json_file}")
    if not output_path.exists():
        create_dir_path(output_path)

    instance_ids = []
    with open(instances_json_file, "r") as f:
        for line in f:
            instance = json.loads(line)
            instance_ids.append(instance["instance_id"])
    print(" > instance_ids:", instance_ids)

    start_ms = current_ms()
    skipped_repos = []
    missing_install_cmd_repos = []
    missing_download_cmd_repos = []
    collected_repos = []

    if (limit is not None) and (limit > 0):
        print(f"\nLimiting to {limit} instances")
        instance_ids = instance_ids[:limit]

    for instance_id in instance_ids:
        repo_start_ms = current_ms()
        repo_name = f"repo__{instance_id}"
        repo_path = repos_path / repo_name

        repo_pip_packages_path = output_path / instance_id
        if repo_pip_packages_path.exists():
            # skip if already downloaded pip packages for this repo
            skipped_repos.append(repo_name)
            continue

        # get repo config
        repo_config_name = get_repo_config_name_from_repo_path(instance_id).rsplit("__", 1)[-1]
        repo_config_path = repo_configs_path / f"{repo_config_name}.json"
        if not repo_config_path.exists():
            # raise ValueError(f"Error: repo config does not exist: {repo_config_path}")
            print(f'ValueError(f"Error: repo config does not exist: {repo_config_path}") ... SKIPPING')
            continue
            
        repo_config = json.loads(Path(repo_config_path).read_text())

        specs = repo_config["specs"]
        if not specs.get("default", None):
            # error: no default install command in repo config
            missing_install_cmd_repos.append(repo_name)
            continue

        cmd_install = specs["default"]["install"]

        # add extra pip packages
        extra_pip_packages = specs["default"].get("pip_packages", None)
        if extra_pip_packages:
            cmd_install = f"{cmd_install} && pip install {' '.join(extra_pip_packages)}"

        # get build-system requirements
        toml_path = repo_path / "pyproject.toml"
        if toml_path.exists():
            build_requirements = get_build_system_requires(toml_path)
            print(f" > Found build-system requirements in {toml_path.absolute()}:\n > {build_requirements}")
            if build_requirements and len(build_requirements) > 0:
                cmd_install = f"{cmd_install} && pip install {' '.join(build_requirements)}"

        # set env vars
        cmd_env_vars = specs["default"].get("env_vars", None)
        if cmd_env_vars:
            env_vars = ' '.join(list(cmd_env_vars))
            cmd_install = ' && '.join(map(lambda c: f"{env_vars} {c}", cmd_install.split(' && ')))

        # convert `pip install` to `pip download`
        cmd_download = convert_pip_install_to_pip_download(cmd_install, repo_pip_packages_path.absolute())
        if "download" not in cmd_download:
            missing_download_cmd_repos.append(repo_name)
            print(f"Error: no download command in repo config: {repo_config_path}")
            continue

        cmds = [
            f"cd {repo_path}",
            "rm -rf venv",
            f"{python_exec} -m venv venv",
            "source venv/bin/activate",
            cmd_download,
            "deactivate"
        ]
        # run `pip download`
        run_commands(
            cmds,
            log_commands=True
        )
        collected_repos.append(repo_name)
        print(f" > Finished `pip download` for repo: {repo_name} in {seconds_since(repo_start_ms)}s")
    
    print(f"\nFinished `pip download` for {len(instance_ids)} repos in {seconds_since(start_ms)}s")
    print(f"Already collected repos (skipped): {skipped_repos}")
    print(f"Collected repos: {collected_repos}")
    print(f"Missing install command repos: {missing_install_cmd_repos}")
    
    return skipped_repos, collected_repos, missing_install_cmd_repos, missing_download_cmd_repos


!mkdir -p /tmp/dependencies/collected/download_pip_packages

# Dev SWE Bench Lite
download_pip_packages(
    repos_path = "/tmp/dependencies/collected/repos",
    repo_configs_path = "/kaggle/working/repo_configs",
    output_path = "/tmp/dependencies/collected/download_pip_packages",
    instances_json_file = "/kaggle/working/task_instances/swe_bench_lite-dev/tasks.jsonl",
)


read_json_file("/kaggle/working/repo_configs/astropy.json")




