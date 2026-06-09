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


first_prediction = True


def predict(problem_statement: str, repo_archive: io.BytesIO) -> str:
    """ Replace this function with your inference code.
    Args:
        problem_statement: The text of the git issue.
        repo_path: A BytesIO buffer path with a .tar containing the codebase that must be patched. The gateway will make this directory available immediately before this function runs.
    """
    global first_prediction
    if not first_prediction:
        return None  # Skip issue.

    # Unpack
    with open('repo_archive.tar', 'wb') as f:
        f.write(repo_archive.read())
    repo_path = 'repo'
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)
    shutil.unpack_archive('repo_archive.tar', extract_dir=repo_path)
    os.remove('repo_archive.tar')
    first_prediction = False
    # Instead of a valid diff, let's just submit a generic string. This will definitely fail.
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


import os

print("Available datasets:")
print(os.listdir("/kaggle/input"))

if os.path.exists("/kaggle/input/pretrained-models"):
    print("Files in /kaggle/input/pretrained-models:")
    print(os.listdir("/kaggle/input/pretrained-models"))
else:
    print("/kaggle/input/pretrained-models does not exist.")



import json
import os
import re
import urllib.request
from enum import Enum
from functools import cache
from pathlib import Path
from typing import Optional, Tuple

import requests

from kprize.constants import KEY_INSTANCE_ID

WHL_DOWNLOAD_REGEX = re.compile(r"\w*Downloading (.+\.whl)")
WHL_CACHED_REGEX = re.compile(r"\w*Using cached (.+\.whl)")
TAR_GZ_DOWNLOAD_REGEX = re.compile(r"\w*Downloading (.+\.tar\.gz)")

class DownloadStatus(Enum):
    SUCCESS = "SUCCESS"
    SKIPPED = "SKIPPED"
    FAILURE = "FAILURE"

def get_download_pip_dep_from_line(line: str) -> str:
    """
    Extracts the whl file from the download line

    Args:
        line (str): line from the log
    Returns:
        str: whl file name
    """
    match = WHL_DOWNLOAD_REGEX.search(line)
    if match:
        return match.group(1)
    match = TAR_GZ_DOWNLOAD_REGEX.search(line)
    if match:
        return match.group(1)
    match = WHL_CACHED_REGEX.search(line)
    if match:
        return match.group(1)
    return None


def parse_pip_dependencies(log: str) -> list[str]:
    """
    Parser for pip dependencies downloaded as part of the setup (whl files)

    Args:
        log (str): log content of environment and repo setup
    Returns:
        list: list of whls/tar.gz downloaded in the log
    """
    pip_dependency_set = set()
    escapes = "".join(chr(char) for char in range(1, 32))
    translator = str.maketrans("", "", escapes)

    for line in log.split("\n"):
        line = re.sub(r"\[(\d+)m", "", line)
        line = line.translate(translator)
        # print(line)
        dep = get_download_pip_dep_from_line(line)
        if dep:
            pip_dependency_set.add(dep)
    return sorted(pip_dependency_set)


def parse_conda_package_names_from_table(table_output: str) -> list[str]:
    """
    Parses the package names from the given table output.

    Args:
        table_output (str): The table output containing package information.
    Returns:
        list: List of package names.
    """
    package_names = []
    lines = table_output.split('\n')
    for line in lines:
        match = re.match(r"^\s*([^\s]+)\s*\|", line)
        if match:
            package_names.append(match.group(1))
    return package_names


def parse_conda_install_dependencies(output: str) -> dict:
    """
    Parses the given output and returns a dictionary with the first word as the key
    and the second word as the value.

    Args:
        output (str): The output containing package information.
    Returns:
        dict: Dictionary with the first word as the key and the second word as the value.
    """
    result = {}
    lines = output.strip().split('\n')
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            key = parts[0]
            value = parts[1]
            result[key] = value
    return result


def parse_conda_dependencies(log: str) -> set[str]:
    """
    Parser for conda dependencies downloaded as part of the setup

    Args:
        log (str): log content of environment and repo setup
    Returns:
        list: list of conda packages downloaded in the log
    """
    start_conda_dependencies_installed = "The following NEW packages will be INSTALLED:"
    end_conda_dependencies = "Downloading and Extracting Packages:"

    index_start_installed = log.find(start_conda_dependencies_installed)
    if index_start_installed > 0:
        index_start_installed = index_start_installed + len(start_conda_dependencies_installed)
    index_end_installed = log.find(end_conda_dependencies)

    if index_start_installed > 0 and index_end_installed > 0:
        conda_install_output = log[index_start_installed:index_end_installed]
        installed_packages = parse_conda_install_dependencies(conda_install_output)
    else:
        print("Unable to find conda install dependencies in the log")
        installed_packages = {}

    # return set of package urls
    return set(installed_packages.values())


def get_pypi_package_from_whl(whl: str):
    return whl.split("-")[0]


def get_pypi_package_url(package_whl: str):
    return f"https://pypi.debian.net/{get_pypi_package_from_whl(package_whl)}/{package_whl}"


def is_url_valid(url: str) -> bool:
    """ Checks if the given URL is valid."""
    try:
        code = urllib.request.urlopen(url).getcode()
        return code == 200
    except:
        return False


# Url checks are slow, so cache the results
@cache
def get_conda_forge_package_url(package_id: str) -> Optional[str]:
    """ Constructs the download URL for a conda-forge package."""

    package_path = package_id.replace("::", "/")
    url_without_extension = f"https://conda.anaconda.org/{package_path}"
    _conda_url = f"{url_without_extension}.conda"
    if is_url_valid(_conda_url):
        return _conda_url
    _tar_url = f"{url_without_extension}.tar.bz2"
    if is_url_valid(_tar_url):
        return _tar_url
    return None


def get_channels_block_from_environment_yml(yml: str) -> str:
    """ Parses the environment.yml and returns the list of channels."""
    # example
    # channels:
    #   - conda-forge
    # dependencies:
    # get the channels block
    return "channels:" + yml.split("channels:")[1].split("dependencies:")[0]


def get_dependencies_from_setup_logs(setup_log_path: Path) -> Tuple[dict, dict]:
    """
    Parses PIP and Conda dependencies from setup log

    :param setup_log_path:
    :return: [pip_packages_map, conda_packages_map]
    """

    # check if setup log file exists
    if not os.path.exists(setup_log_path):
        print(f"Setup log file does not exist: {setup_log_path}")
        return {}, {}

    # Parse setup logs for dependencies
    with open(setup_log_path, "r") as f:
        log = f.read()
        pip_dependencies = parse_pip_dependencies(log)
        conda_dependencies = parse_conda_dependencies(log)

    # Create package maps { package_name: package_url }
    pip_package_map = {p: get_pypi_package_url(p) for p in pip_dependencies}
    conda_package_map = {}
    for p in conda_dependencies:
        purl = get_conda_forge_package_url(p)
        if purl:
            conda_package_map[purl.split('/')[-1]] = purl
        else:
            print(f"Failed to get conda-forge package url for: {p}")

    return pip_package_map, conda_package_map

class PythonInstallDependencyParser:
    def __init__(
            self,
            install_logs_dir: Path,
            collected_pip_packages_dir: Path,
            collected_conda_packages_dir: Path,
            collected_requirements_log_dir: Path,
            collected_failures_dir: Path,
    ):
        self._install_logs_dir = install_logs_dir
        self._collected_pip_packages_dir = collected_pip_packages_dir
        self._collected_conda_packages_dir= collected_conda_packages_dir
        self._collected_requirements_log_dir = collected_requirements_log_dir
        self._collected_failures_dir = collected_failures_dir

    @staticmethod
    def get_pip_requirements_file_name(instance_id: str) -> str:
        return f"{instance_id}-pip-requirements.txt"

    @staticmethod
    def get_conda_requirements_file_name(instance_id: str) -> str:
        return f"{instance_id}-conda-requirements.txt"

    def get_pip_requirements_path(self, instance_id: str) -> Path:
        return self._collected_requirements_log_dir / self.get_pip_requirements_file_name(instance_id)

    def get_conda_requirements_path(self, instance_id: str) -> Path:
        return self._collected_requirements_log_dir / self.get_conda_requirements_file_name(instance_id)

    def get_pip_requirements_for_instance(self, instance_id: str) -> set[str]:
        """Get PIP requirements for instance"""
        pip_requirements_path = self.get_pip_requirements_path(instance_id)
        pip_requirements = pip_requirements_path.read_text().split("\n") if pip_requirements_path.exists() else []
        for req in pip_requirements:
            if req.endswith(".tar.gz"):
                # locate the corresponding whl file
                matching_whls = list(self._collected_pip_packages_dir.glob(f"{req.replace('.tar.gz', '')}*.whl"))
                if len(matching_whls) > 0:
                    for whl_file in matching_whls:
                        print(f"Adding matching whl file: {whl_file.name} for {req}")
                        pip_requirements.append(whl_file.name)
        return set(pip_requirements)

    def get_conda_requirements_for_instance(self, instance_id: str) -> set[str]:
        """Get CONDA requirements for instance"""
        conda_requirements_path = self.get_conda_requirements_path(instance_id)
        conda_requirements = conda_requirements_path.read_text().split("\n") if conda_requirements_path.exists() else []
        return set(conda_requirements)

    @staticmethod
    def get_requirements_for_instances(instance_ids: list[str], get_requirements_func) -> list[str]:
        """Get PIP requirements for instances"""
        requirements = set()
        for instance_id in instance_ids:
            requirements |= get_requirements_func(instance_id)
        return requirements

    def download_packages(self, package_to_url_map: dict, output_dir: Path):
        """
        Download packages from package_to_url_map to output_dir
        :param package_to_url_map:
        :param output_dir:
        :return:
        """
        failed_downloads_path = self._collected_failures_dir / "failed_downloads.jsonl"
        skipped_packages = []
        for package in package_to_url_map.keys():
            output_file = output_dir / package
            if output_file.exists():
                skipped_packages.append(package)
                continue
            print(f"Downloading package: {package}")
            # download package
            url = package_to_url_map[package]
            response = requests.get(url)
            if response.status_code != 200:
                failed = {"package": package, "url": url}
                print(f"Failed to download package: {json.dumps(failed)}")
                with failed_downloads_path.open("a") as f:
                    f.write(f'{json.dumps(failed)}\n')
                continue
            # save package
            output_file.write_bytes(response.content)
        if len(skipped_packages) > 0:
            print(f"Skipped existing packages: {skipped_packages}")

    def download_packages_from_setup_log(self, instance_id: str, setup_log_path: Path) -> DownloadStatus:
        pip_requirements_file = self.get_pip_requirements_path(instance_id)
        conda_requirements_file = self.get_conda_requirements_path(instance_id)
        # Skip if packages have already been downloaded
        if pip_requirements_file.exists() and conda_requirements_file.exists():
            return DownloadStatus.SKIPPED
        if setup_log_path.exists():
            print(f"\nGetting dependencies for '{instance_id}'")
            # Parse dependencies from logs
            pip_packages, conda_packages = get_dependencies_from_setup_logs(setup_log_path)

            # download dependencies
            self.download_packages(pip_packages, self._collected_pip_packages_dir)
            self.download_packages(conda_packages, self._collected_conda_packages_dir)

            # create whl requirements file for task instance
            conda_requirements_file.write_text("\n".join(conda_packages))
            pip_requirements_file.write_text("\n".join(pip_packages))
            return DownloadStatus.SUCCESS
        else:
            print(f"\nERROR: Setup log not found {setup_log_path}")
            return DownloadStatus.FAILURE

    def download_packages_from_setup_logs(self, instances: list[dict]):
        """
        Download PIP and Conda dependencies from setup logs
        """
        skipped_instances = []
        error_instances = []
        for idx, instance in enumerate(instances):
            instance_id = instance[KEY_INSTANCE_ID]
            setup_log_path = self._install_logs_dir / f"{instance_id}/setup_output.txt"
            status = self.download_packages_from_setup_log(instance_id, setup_log_path)
            if status == DownloadStatus.SKIPPED:
                skipped_instances.append(instance_id)
            elif status == DownloadStatus.FAILURE:
                error_instances.append(instance_id)
        if len(skipped_instances) > 0:
            print(f"Skipped {len(skipped_instances)} instances with existing requirement logs:\n{skipped_instances}")
        if len(error_instances) > 0:
            print(f"Missing setup logs for {len(error_instances)} instances:\n{error_instances}")



import io
import os
import shutil
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from kaggle_evaluation.konwinski_prize_inference_server import KPrizeInferenceServer

# Global variable to store the instance count
instance_count = None

# Define the get_number_of_instances function
def get_number_of_instances(num_instances: int) -> None:
    """
    The very first message from the gateway will be the total number of instances to be served.
    You don't need to return anything, just store the number.
    """
    global instance_count
    instance_count = num_instances

# Define the Example Dataset class
class ExampleDataset(Dataset):
    def __init__(self, problem_statements):
        self.problem_statements = problem_statements

    def __len__(self):
        return len(self.problem_statements)

    def __getitem__(self, idx):
        return self.problem_statements[idx]

# Define the model class
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

# Initialize the model
input_dim = 128  # Placeholder input dimensions
hidden_dim = 64
model = SimpleMLP(input_dim, hidden_dim)

# Optionally load pre-trained weights if available
model_path = "/kaggle/input/pretrained-models/model.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print("Loaded pre-trained model weights.")
else:
    print("No pre-trained model found. Using randomly initialized weights.")

model.eval()

# Define the predict function
def predict(problem_statement: str, repo_archive: io.BytesIO) -> str:
    """
    Predict the resolution for the given problem statement.
    Args:
        problem_statement: The text of the problem statement.
        repo_archive: The repo archive as a binary stream.
    Returns:
        A string representing the prediction (e.g., a patch or resolution).
    """
    # Example heuristic-based feature extraction (placeholder logic)
    problem_length = len(problem_statement)
    features = torch.tensor([problem_length] * 128, dtype=torch.float32)  # Simplistic feature vector
    features = features.unsqueeze(0)  # Add batch dimension

    # Perform prediction
    with torch.no_grad():
        output = model(features)
    prediction = "RESOLVED" if output.item() > 0.5 else "SKIPPED"

    return prediction

# Simulate scoring
def calculate_score(a, b, c):
    """
    Calculate the competition score based on the formula:
    score = (a - b) / (a + b + c)

    :param a: Number of correctly resolved issues
    :param b: Number of failing issues
    :param c: Number of skipped issues
    :return: The calculated score
    """
    if (a + b + c) == 0:
        return 0  # Avoid division by zero
    return (a - b) / (a + b + c)

# Example scoring simulation
def simulate_scores(results):
    """
    Simulate scores for different scenarios.

    :param results: List of dictionaries with 'resolved', 'failed', 'skipped' counts.
    :return: List of simulated scores
    """
    scores = []
    for result in results:
        a = result.get("resolved", 0)
        b = result.get("failed", 0)
        c = result.get("skipped", 0)
        score = calculate_score(a, b, c)
        scores.append({
            "resolved": a,
            "failed": b,
            "skipped": c,
            "score": score
        })
    return scores

# Run scoring simulation
scenarios = [
    {"resolved": 10, "failed": 2, "skipped": 3},
    {"resolved": 15, "failed": 1, "skipped": 4},
    {"resolved": 20, "failed": 5, "skipped": 0},
    {"resolved": 0, "failed": 0, "skipped": 10}
]
simulated_scores = simulate_scores(scenarios)
for score in simulated_scores:
    print(f"Resolved: {score['resolved']}, Failed: {score['failed']}, Skipped: {score['skipped']}, Score: {score['score']:.4f}")

# Optimize dataset handling with Polars
def load_and_process_dataset(file_path: str):
    """
    Load and process a dataset efficiently using Polars.

    :param file_path: Path to the dataset file.
    :return: Processed dataset as a Polars DataFrame.
    """
    df = pl.read_csv(file_path)
    # Example processing: Filter unresolved issues and calculate a new priority column
    df = (
        df.filter(df["status"] == "unresolved")
          .with_columns([
              (df["severity"] * df["urgency"]).alias("priority"),
              (df["reported_date"].str.strptime(pl.Date, "%Y-%m-%d")).alias("parsed_date")
          ])
          .sort("priority", descending=True)
    )
    return df

# Example: Load and preprocess dataset
dataset_path = "/kaggle/input/dataset/issues.csv"
processed_dataset = load_and_process_dataset(dataset_path)
print(processed_dataset.head())

# Initialize the inference server
inference_server = KPrizeInferenceServer(
    get_number_of_instances,
    predict
)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()



import os
print("Available datasets:", os.listdir("/kaggle/input"))



import polars as pl
from io import StringIO

mock_data = StringIO(\"\"\"status,severity,urgency,reported_date
unresolved,3,2,2023-01-01
resolved,2,1,2023-01-02
unresolved,5,3,2023-01-03
\"\"\")

processed_dataset = pl.read_csv(mock_data)
print(processed_dataset)



import os

# Check the contents of each dataset directory
for dataset in ["konwinski-prize", "update"]:
    print(f"Contents of /kaggle/input/{dataset}:")
    print(os.listdir(f"/kaggle/input/{dataset}"))



dataset_path = "/kaggle/input/<dataset-folder-name>/issues.csv"



import polars as pl
from io import StringIO

# Mock dataset
mock_data = StringIO("""
status,severity,urgency,reported_date
unresolved,3,2,2023-01-01
resolved,2,1,2023-01-02
unresolved,5,3,2023-01-03
""")

# Read the mock dataset with Polars
processed_dataset = pl.read_csv(mock_data)
print(processed_dataset)



# Process the dataset
processed_dataset = (
    processed_dataset.filter(processed_dataset["status"] == "unresolved")  # Filter unresolved issues
                     .with_columns([
                         (processed_dataset["severity"] * processed_dataset["urgency"]).alias("priority"),  # Add priority
                         (processed_dataset["reported_date"].str.strptime(pl.Date, "%Y-%m-%d")).alias("parsed_date")  # Parse date
                     ])
                     .sort("priority", descending=True)  # Sort by priority
)

print("Processed Dataset:")
print(processed_dataset)



resolved_count = 1  # From mock data
failed_count = 1    # Assume one processing error
skipped_count = 1   # Assume one unresolved

score = calculate_score(resolved_count, failed_count, skipped_count)
print(f"Resolved: {resolved_count}, Failed: {failed_count}, Skipped: {skipped_count}, Score: {score:.4f}")



import io
import os
import shutil
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from kaggle_evaluation.konwinski_prize_inference_server import KPrizeInferenceServer

# Global variable to store the instance count
instance_count = None

# Define the get_number_of_instances function
def get_number_of_instances(num_instances: int) -> None:
    """
    The very first message from the gateway will be the total number of instances to be served.
    You don't need to return anything, just store the number.
    """
    global instance_count
    instance_count = num_instances

# Define the Example Dataset class
class ExampleDataset(Dataset):
    def __init__(self, problem_statements):
        self.problem_statements = problem_statements

    def __len__(self):
        return len(self.problem_statements)

    def __getitem__(self, idx):
        return self.problem_statements[idx]

# Define the model class
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

# Initialize the model
input_dim = 128  # Placeholder input dimensions
hidden_dim = 64
model = SimpleMLP(input_dim, hidden_dim)

# Optionally load pre-trained weights if available
model_path = "/kaggle/input/pretrained-models/model.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print("Loaded pre-trained model weights.")
else:
    print("No pre-trained model found. Using randomly initialized weights.")

model.eval()

# Define the predict function with fine-tuned thresholds
def predict(problem_statement: str, repo_archive: io.BytesIO) -> str:
    """
    Predict the resolution for the given problem statement.
    Args:
        problem_statement: The text of the problem statement.
        repo_archive: The repo archive as a binary stream.
    Returns:
        A string representing the prediction (e.g., a patch or resolution).
    """
    # Example heuristic-based feature extraction (placeholder logic)
    problem_length = len(problem_statement)
    features = torch.tensor([problem_length] * 128, dtype=torch.float32)  # Simplistic feature vector
    features = features.unsqueeze(0)  # Add batch dimension

    # Perform prediction
    with torch.no_grad():
        output = model(features)
    prediction = "RESOLVED" if output.item() > 0.7 else "SKIPPED"  # Adjusted threshold for resolution

    return prediction

# Simulate scoring
def calculate_score(a, b, c):
    """
    Calculate the competition score based on the formula:
    score = (a - b) / (a + b + c)

    :param a: Number of correctly resolved issues
    :param b: Number of failing issues
    :param c: Number of skipped issues
    :return: The calculated score
    """
    if (a + b + c) == 0:
        return 0  # Avoid division by zero
    return (a - b) / (a + b + c)

# Simulate additional scenarios
def simulate_scores(results):
    """
    Simulate scores for different scenarios.

    :param results: List of dictionaries with 'resolved', 'failed', 'skipped' counts.
    :return: List of simulated scores
    """
    scores = []
    for result in results:
        a = result.get("resolved", 0)
        b = result.get("failed", 0)
        c = result.get("skipped", 0)
        score = calculate_score(a, b, c)
        scores.append({
            "resolved": a,
            "failed": b,
            "skipped": c,
            "score": score
        })
    return scores

# Run scoring simulation with additional scenarios
scenarios = [
    {"resolved": 10, "failed": 2, "skipped": 3},
    {"resolved": 15, "failed": 1, "skipped": 4},
    {"resolved": 20, "failed": 5, "skipped": 0},
    {"resolved": 8, "failed": 2, "skipped": 5},
    {"resolved": 12, "failed": 3, "skipped": 2}
]
simulated_scores = simulate_scores(scenarios)
for score in simulated_scores:
    print(f"Resolved: {score['resolved']}, Failed: {score['failed']}, Skipped: {score['skipped']}, Score: {score['score']:.4f}")

# Optimize dataset handling with Polars
import matplotlib.pyplot as plt

def load_and_process_dataset(file_path: str):
    """
    Load and process a dataset efficiently using Polars.

    :param file_path: Path to the dataset file.
    :return: Processed dataset as a Polars DataFrame.
    """
    df = pl.read_csv(file_path)
    # Example processing: Filter unresolved issues and calculate a new priority column
    df = (
        df.filter(df["status"] == "unresolved")
          .with_columns([
              (df["severity"] * df["urgency"]).alias("priority"),
              (df["reported_date"].str.strptime(pl.Date, "%Y-%m-%d")).alias("parsed_date")
          ])
          .sort("priority", descending=True)
    )
    return df

# Visualize the dataset

def visualize_priority_distribution(processed_dataset):
    """
    Plot the priority distribution from the processed dataset.

    :param processed_dataset: Processed dataset with a priority column.
    """
    priorities = processed_dataset["priority"].to_list()
    plt.hist(priorities, bins=10, color="skyblue", edgecolor="black")
    plt.title("Priority Distribution")
    plt.xlabel("Priority")
    plt.ylabel("Frequency")
    plt.show()

# Example: Load, preprocess, and visualize dataset
dataset_path = "/kaggle/input/dataset/issues.csv"
processed_dataset = load_and_process_dataset(dataset_path)
visualize_priority_distribution(processed_dataset)

# Initialize the inference server
inference_server = KPrizeInferenceServer(
    get_number_of_instances,
    predict
)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()



# Example training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

for epoch in range(10):  # Adjust epochs
    for inputs, labels in train_loader:  # Assuming train_loader is defined
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Save the model
torch.save(model.state_dict(), "model.pth")



class ExampleDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)



# Mock data: 100 samples, each with 128 features
import numpy as np

np.random.seed(42)  # For reproducibility
data = np.random.rand(100, 128)  # 100 samples, 128 features each
labels = np.random.randint(0, 2, size=(100, 1))  # Binary labels (0 or 1)

# Create the dataset and DataLoader
train_dataset = ExampleDataset(data, labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)



# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

# Training loop
for epoch in range(10):  # Adjust the number of epochs
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Clear gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), "model.pth")
print("Model training complete and saved as model.pth")



import io
import os
import shutil
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from kaggle_evaluation.konwinski_prize_inference_server import KPrizeInferenceServer
import zipfile
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Extract and process data from data.a_zip
zip_path = "/kaggle/input/konwinski-prize/data.a_zip"
data_folder = "/kaggle/working/data"

if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_folder)
    logger.info("Extracted data from data.a_zip.")
else:
    logger.error("data.a_zip not found. Ensure the file is available.")

# Load data from data.parquet
parquet_path = os.path.join(data_folder, "data/data.parquet")

if os.path.exists(parquet_path):
    df_metadata = pl.read_parquet(parquet_path)
    logger.info(f"Loaded data from {parquet_path}.")
else:
    logger.error(f"Parquet file {parquet_path} not found.")

# Filter and process the training data
if 'patch' in df_metadata.columns and 'problem_statement' in df_metadata.columns:
    df_training = df_metadata.filter(~df_metadata['patch'].is_null())
    logger.info(f"Filtered training data: {len(df_training)} records available.")
else:
    logger.error("Required columns 'patch' or 'problem_statement' not found in metadata.")

# Define the Example Dataset class
class ExampleDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# Define the model class
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

# Initialize the model
input_dim = 128  # Placeholder input dimensions
hidden_dim = 64
model = SimpleMLP(input_dim, hidden_dim)

# Load the trained model weights
model_path = "model.pth"  # Ensure this path points to the trained model
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, weights_only=True))
    logger.info("Loaded the trained model weights.")
else:
    logger.warning("No pre-trained model found. Using randomly initialized weights.")

model.eval()

# Define the predict function with fine-tuned thresholds
def predict(problem_statement: str, repo_archive: io.BytesIO) -> str:
    """
    Predict the resolution for the given problem statement.
    Args:
        problem_statement: The text of the problem statement.
        repo_archive: The repo archive as a binary stream.
    Returns:
        A string representing the prediction (e.g., a patch or resolution).
    """
    try:
        problem_length = len(problem_statement)
        features = torch.tensor([problem_length] * 128, dtype=torch.float32)  # Simplistic feature vector
        features = features.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(features)
        prediction = "RESOLVED" if output.item() > 0.7 else "SKIPPED"  # Adjusted threshold for resolution
        return prediction
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return "SKIPPED"

# Evaluate the model on a test dataset
def evaluate_model(test_loader):
    """
    Evaluate the trained model on a test dataset.

    :param test_loader: DataLoader for the test dataset.
    :return: Accuracy of the model on the test data.
    """
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    logger.info(f"Model Accuracy on Test Data: {accuracy:.2%}")
    return accuracy

# Simulate scoring
def calculate_score(a, b, c):
    """
    Calculate the competition score based on the formula:
    score = (a - b) / (a + b + c)

    :param a: Number of correctly resolved issues
    :param b: Number of failing issues
    :param c: Number of skipped issues
    :return: The calculated score
    """
    if (a + b + c) == 0:
        return 0  # Avoid division by zero
    return (a - b) / (a + b + c)

# Dynamic evaluation handling for unseen instances
def dynamic_evaluation(instance_id: str, problem_statement: str, repo_archive: io.BytesIO):
    """
    Dynamically evaluate an unseen instance.

    :param instance_id: The ID of the instance.
    :param problem_statement: The text describing the issue.
    :param repo_archive: The repo archive as a binary stream.
    :return: Prediction result.
    """
    logger.info(f"Evaluating instance {instance_id}...")
    prediction = predict(problem_statement, repo_archive)
    logger.info(f"Instance {instance_id}: Prediction = {prediction}")
    return prediction

# Initialize the inference server
inference_server = KPrizeInferenceServer(
    get_number_of_instances,
    predict
)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()



import json

# Path to save the notebook
save_path = "/kaggle/working/saved_notebook.ipynb"

# Get the notebook content using the IPython API
try:
    from IPython import get_ipython
    from notebook.notebookapp import list_running_servers

    kernel_id = get_ipython().config["IPKernelApp"]["connection_file"].split("-")[-1].split(".")[0]
    for srv in list_running_servers():
        response = requests.get(srv["url"] + "api/sessions", params={"token": srv.get("token", "")})
        for sess in response.json():
            if sess["kernel"]["id"] == kernel_id:
                with open(save_path, "w") as f:
                    json.dump(sess["notebook"], f)
                print(f"Notebook saved to {save_path}")
except Exception as e:
    print(f"Failed to save notebook: {e}")


