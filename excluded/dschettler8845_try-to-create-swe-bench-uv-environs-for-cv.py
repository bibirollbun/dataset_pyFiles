# ############################################################################################### #
!pip install -q /kaggle/input/konwinski-prize/kprize_setup/kprize-1.1.0-py3-none-any.whl
# ############################################################################################### #
# Do this instead of installing because for some reason we can't see the `bundling` part.
# Since we have internet I can always install the dependencies as needed.
import sys; sys.path.insert(0, "/kaggle/input/konwinski-prize/kprize_setup")
# ############################################################################################### #

import io
import os
import shlex
import logging
import shutil
import subprocess
from enum import Enum
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
import uuid
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
    "swe_bench_verified": load_dataset(HF_SWE_BENCH_VERIFIED_PATH),  # N_EX = 500
    "swe_bench": load_dataset(HF_SWE_BENCH_PATH),                    # N_EX = 19008 + 2294 + 225 = 21527
}

hf_dfs = {}
for k,v in hf_datasets.items():
    hf_dfs[k] = {}
    for _k, _v in v.items():
        _df = pd.DataFrame(_v)
        for col in ["PASS_TO_PASS", "FAIL_TO_PASS"]:
            _df[col] = _df[col].apply(lambda x: ast.literal_eval(x))
        hf_dfs[k][_k] = _df.copy()

# Let's see 'em
rich.print("\n\nKPRIZE DATASET:\n")
display(kprize_df)

rich.print("\n\n\n\nSWE BENCH DATASETS:\n")
for ds_name, ds in hf_datasets.items(): 
    rich.print(f"\n\n\n\n[bold]{ds_name}[/bold]")
    display(ds)


import tomli
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version, parse
from contextlib import contextmanager

@contextmanager
def suppress_logging_below(level):
    logger = logging.getLogger()
    old_level = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(old_level)

class PythonImplementation(Enum):
    """Enumeration of Python implementations."""
    CPYTHON = "cpython"
    PYPY = "pypy"


@dataclass
class PythonVersion:
    """Represents a Python version with its implementation and availability.
    
    Attributes:
        implementation (PythonImplementation): The Python implementation (e.g., CPython, PyPy)
        version (str): The full version string (e.g., '3.10.12')
        is_installed (bool): Whether the version is installed on the system
        path (str | Path | None): The path to the Python executable
        is_freethreaded (bool): Whether the Python version is free-threaded
            Free-threaded Python allows multiple threads to run simultaneously.
    """
    implementation: PythonImplementation
    version: str
    is_installed: bool
    path: str | Path | None = None
    is_freethreaded: bool = False

    @property
    def base_version(self) -> str:
        """Returns the base version (e.g., '3.10' from '3.10.12')."""
        return '.'.join(self.version.split('.')[:2])

    @property
    def parsed_version(self) -> Version:
        """Returns a packaging.version.Version object for comparison."""
        return parse(self.version)

@dataclass
class VersionConstraints:
    """Represents Python version constraints with metadata.
    
    Attributes:
        min_version (str | None): Minimum version constraint
        max_version (str | None): Maximum version constraint
        specifier_set (SpecifierSet): Set of version specifiers
        source_file (str): The file where the constraints were extracted from
        confidence (float): Confidence level of the constraints (0.0 to 1.0)
    """
    min_version: str | None
    max_version: str | None
    specifier_set: SpecifierSet
    source_file: str
    confidence: float  # 0.0 to 1.0

    @classmethod
    def from_specifier_string(cls, spec_str: str, source_file: str, confidence: float = 1.0) -> 'VersionConstraints':
        """Create VersionConstraints from a version specifier string.
        
        Args:
            spec_str (str): The version specifier string
            source_file (str): The file where the constraints were extracted from
            confidence (float): Confidence level of the constraints (0.0 to 1.0)
            
        Returns:
            VersionConstraints: The parsed version constraints
        """
        # (1) Parse the version specifier string
        spec_set = SpecifierSet(spec_str)

        # (2) Extract min and max versions from specifiers
        min_ver = None
        max_ver = None
        # (2a) Iterate over the specifiers
        for spec in spec_set:
            # (2b) Skip non-version specifiers
            ver_str = spec.version
            # (2c) Check for minimum and maximum versions
            if spec.operator in ('>=', '>'):
                # (2d) Update min_version if needed
                if min_ver is None or parse(ver_str) > parse(min_ver):
                    min_ver = ver_str
            # (2e) Check for maximum version
            elif spec.operator in ('<=', '<'):
                # (2f) Update max_version if needed
                if max_ver is None or parse(ver_str) < parse(max_ver):
                    max_ver = ver_str
        # (3) Create and return the VersionConstraints object
        return cls(
            min_version=min_ver,
            max_version=max_ver,
            specifier_set=spec_set,
            source_file=source_file,
            confidence=confidence
        )

@dataclass
class CommandResult:
    """Holds information about a subprocess command result.

    Args:
        command (str): The command that was executed.
        returncode (int): The return code of the command.
        stdout (str): The standard output of the command.
        stderr (str): The standard error of the command.
    """
    command: str
    returncode: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        """Indicates if returncode == 0."""
        return self.returncode == 0

    def __str__(self) -> str:
        """Informal string representation, used for user-facing display."""
        if self.success:
            return "CommandResult(command={!r}, returncode={!r}, success={!r}, stdout={!r})".format(
                self.command, self.returncode, self.success, self.stdout
            )
        else:
            return "CommandResult(command={!r}, returncode={!r}, success={!r}, stdout={!r}, stderr={!r})".format(
                self.command, self.returncode, self.success, self.stdout, self.stderr
            )

    def __repr__(self) -> str:
        """Official string representation, used for debugging."""
        return str(self)

    def raise_for_status(self) -> None:
        """Raise a subprocess.CalledProcessError if the command failed.

        Raises:
            subprocess.CalledProcessError: If the command's return code is non-zero.
        """
        if not self.success:
            raise subprocess.CalledProcessError(
                returncode=self.returncode,
                cmd=self.command,
                output=self.stdout,
                stderr=self.stderr
            )


@dataclass
class EnvironmentConfig:
    """Configuration for UV environment setup.
    
    Args:
        python_version (str, optional): 
            Python version to use (e.g. "3.10")
        base_dir (str, optional): 
            Base directory for environments to be stored.
            NOTE: We default to the /kaggle/tmp directory as it
                  has access to a larger storage volume.
        pytest_options (str, optional): 
            Additional pytest options to pass.
    """
    python_version: str = "3.10"
    base_dir: str = "/kaggle/tmp"
    pytest_options: str = ""


@dataclass
class TestResult:
    """Results from running tests in the environment.
    
    Args:
        success (bool): Whether tests passed
        output (str): Test output
        error (str): Error output if any
        duration (float): Test duration in seconds
    """
    success: bool
    output: str
    error: str
    duration: float


@dataclass
class SWEBenchInstance:
    """Represents a single instance from the SWE-Bench dataset.
    
    Attributes:
        repo (str): Repository URL or identifier (e.g. "owner/repo")
        instance_id (str): Unique identifier for this instance
        base_commit (str): The commit hash where the bug exists
        patch (str): The code changes that fix or modify the bug
        test_patch (str): The test changes associated with the fix
        problem_statement (str): Description of the bug/issue
        hints_text (str, optional): Additional hints or context about the bug
        created_at (datetime): When the instance was created
        version (str): Version identifier for this instance
        fail_to_pass (list[str]): Whether this instance should go from failing to passing
        pass_to_pass (list[str]): Whether this instance should maintain passing status
        environment_setup_commit (str): Commit hash used for environment setup
    """
    repo: str
    instance_id: str
    base_commit: str
    patch: str
    test_patch: str
    problem_statement: str
    hints_text: str | None
    created_at: datetime
    version: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    environment_setup_commit: str

    @classmethod
    def from_df_row(cls, row: Any) -> "SWEBenchInstance":
        """Create an instance from a pandas DataFrame row.
        
        Args:
            row: A single row (pd.Series or dict) from the SWE-Bench-Lite dataset.
        
        Returns:
            A SWEBenchInstance object populated with row data.
        """
        return cls(
            repo=row["repo"],
            instance_id=str(row["instance_id"]),
            base_commit=row["base_commit"],
            patch=row["patch"],
            test_patch=row["test_patch"],
            problem_statement=row["problem_statement"],
            hints_text=row["hints_text"],
            created_at=datetime.fromisoformat(row["created_at"].replace('Z', '+00:00')),
            version=row["version"],
            fail_to_pass=row["FAIL_TO_PASS"],
            pass_to_pass=row["PASS_TO_PASS"],
            environment_setup_commit=row["environment_setup_commit"]
        )

    @property
    def github_repo_url(self) -> str:
        """Constructs the full GitHub URL for the repo.

        Returns:
            str: The GitHub repo URL.
        """
        return os.path.join(f"https://github.com", self.repo)
        
    @property
    def github_pull_url(self) -> str:
        """Constructs the full GitHub URL for the PR that fixes the issue.

        Returns:
            str: The GitHub URL for the PR.
        """
        pull_number = self.instance_id.rsplit("-", 1)[-1]
        return os.path.join(self.github_repo_url, "pull", pull_number)

    @property
    def repo_at_base_commit_url(self) -> str:
        """Constructs the full GitHub URL for the base commit (where the bug exists).

        Returns:
            str: The GitHub URL for the base commit.
        """
        return os.path.join(self.github_repo_url, "tree", self.base_commit)

    @property
    def repo_at_environment_setup_commit_url(self) -> str:
        """Constructs the full GitHub URL for the environment setup commit.

        Returns:
            str: The GitHub URL for the environment setup commit.
        """
        return os.path.join(self.github_repo_url, "tree", self.environment_setup_commit)

demo_df = hf_dfs["swe_bench_verified"]["test"]
demo_row = demo_df.iloc[-1]
demo_instance = SWEBenchInstance.from_df_row(demo_row)

rich.print("[bold]SWE-BENCH-VERIFIED DEMO INSTANCE:[/bold]")
display(demo_instance)
display(demo_instance.github_pull_url)
display(demo_instance.repo_at_base_commit_url)
display(demo_instance.repo_at_environment_setup_commit_url)


class UVManager:
    """Manages a UV virtual environment with persistent shell session support.
    
    This class provides functionality to create, manage, and interact with UV virtual
    environments. It supports both context manager and direct usage patterns, maintains
    a persistent shell session, and provides methods for package installation and
    command execution. This class was designed to be used within a Kaggle notebook
    to allow the creation of small environments for recreating swe-bench 
    github issue environments.
    
    Attributes:
        venv_path (Path): Absolute path to the virtual environment
        python_version (str): Python version being used (e.g. "3.10")
        env_ready (bool): Whether the environment is ready for use
        
    Example:
        >>> # Using as context manager
        >>> with UVManager("./my_venv", python_version="3.10") as uv:
        ...     result = uv.pip_install("requests")
        ...     assert result.success
        
        >>> # Direct usage
        >>> uv = UVManager("./other_venv")
        >>> uv.initialize()
        >>> result = uv.send("python --version")
        >>> print(result.stdout)
        Python 3.10.x
        >>> uv.cleanup()
    """
    
    def __init__(
        self, 
        venv_path: str | Path, 
        python_version: str = "3.10",
        env_vars: dict[str, str] | None = None
    ):
        """Initialize the UV environment manager.
        
        Args:
            venv_path (str | Path): Path where virtual environment should be created
            python_version (str): Python version to use (e.g. "3.10")
            env_vars (dict[str, str] | None): Additional environment variables to set in the shell
            
        Note:
            This doesn't create the environment immediately. Call initialize()
            or use as context manager to create and activate the environment.
        """
        self.venv_path = Path(venv_path).absolute()
        self.python_version = python_version
        self._shell = None
        self._logger = logging.getLogger(self.__class__.__name__)
        self.env_ready = False
        
        # Base environment variables
        self._env_vars = {
            'UV_LINK_MODE': 'copy',  # Prevent hardlink warnings
            'VIRTUAL_ENV': str(self.venv_path),
            'PATH': f"{self.venv_path}/bin:{os.environ.get('PATH', '')}"
        }
        if env_vars:
            self._env_vars.update(env_vars)
            
    def __enter__(self) -> 'UVManager':
        """Initialize environment when used as context manager."""
        self.initialize()
        return self
        
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Cleanup when exiting context manager."""
        self.cleanup()

    def _verify_environment(self) -> bool:
        """Verify that the UV environment is properly set up and functional.
        
        Returns:
            bool: True if environment is ready, False otherwise
        """
        try:
            # Check that $VIRTUAL_ENV matches self.venv_path
            venv_check = self.send("echo $VIRTUAL_ENV", bypass_env_check=True)
            if not venv_check.stdout.strip() == str(self.venv_path):
                self._logger.warning(
                    "VIRTUAL_ENV does not match the expected path. "
                    f"Expected: {self.venv_path}, Got: {venv_check.stdout}"
                )
                return False

            # Verify Python is accessible and correct version substring
            py_check = self.send("python --version", bypass_env_check=True)
            if not py_check.success:
                self._logger.warning("Running 'python --version' failed.")
                return False
            # Check if the declared python_version (e.g. '3.10') is in the output
            if self.python_version not in py_check.stdout:
                self._logger.warning(
                    f"Python version mismatch. Expected string '{self.python_version}' "
                    f"in '{py_check.stdout.strip()}'"
                )
                return False

            # Try importing a basic module and check sys.prefix
            import_check = self.send('python -c "import sys; print(sys.prefix)"', bypass_env_check=True)
            if not import_check.success:
                self._logger.warning("Failed to import and print sys.prefix.")
                return False
            if str(self.venv_path) not in import_check.stdout:
                self._logger.warning(
                    f"sys.prefix does not match the expected path: {import_check.stdout}"
                )
                return False

            return True

        except Exception as e:
            self._logger.warning(f"Environment verification failed: {e}")
            return False

    def _initialize_shell(self) -> None:
        """Initialize a persistent shell session with UV environment setup.
        
        Raises:
            RuntimeError: If shell initialization fails
        """
        if self._shell is not None:
            return

        self._shell = subprocess.Popen(
            ['bash'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, **self._env_vars}
        )
        
        # Setup environment variables
        for name, value in self._env_vars.items():
            self._run_in_shell(f'export {name}="{value}"')
        
        # Activate using UV
        stdout, stderr = self._run_in_shell(f'eval "$(uv venv {self.venv_path})"')
        if stderr:
            self._logger.warning(f"UV environment activation warning: {stderr}")

    def _run_in_shell(self, command: str) -> tuple[str, str]:
        """Execute a command in the persistent shell session.
        
        Args:
            command (str): Shell command to execute
            
        Returns:
            Tuple of (stdout, stderr) from command execution
        """
        if not self._shell:
            self._initialize_shell()

        terminator = f"__CMD_DONE_{id(command)}__"
        full_command = f"{command}; echo {terminator}; echo {terminator} >&2"
        
        self._shell.stdin.write(full_command + '\n')
        self._shell.stdin.flush()

        def read_until_terminator(pipe) -> str:
            output = []
            while True:
                line = pipe.readline()
                if not line or terminator in line:
                    break
                output.append(line)
            return ''.join(output).rstrip()

        stdout = read_until_terminator(self._shell.stdout)
        stderr = read_until_terminator(self._shell.stderr)
        
        return stdout, stderr

    def initialize(self) -> None:
        """Create and initialize the UV virtual environment.
        
        This method:
            (1) Creates the virtual environment directory
            (2) Sets up the UV environment
            (3) Initializes the shell session
            (4) Verifies the environment is working
        
        Raises:
            RuntimeError: If environment creation or verification fails
        """
        self._logger.info(f"Creating UV environment at {self.venv_path}")
        
        try:
            # Create the virtual environment directory
            self.venv_path.mkdir(parents=True, exist_ok=True)
            
            # Create the venv using uv
            result = subprocess.run(
                ["uv", "venv", "--python", self.python_version, str(self.venv_path)],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Initialize shell and verify
            self._initialize_shell()
            if not self._verify_environment():
                raise RuntimeError("Environment verification failed.")
            self.env_ready = True
                
            if not self.env_ready:
                raise RuntimeError("Environment verification failed after waiting")
                
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to create UV environment: {e.stderr}")
        except Exception as e:
            raise RuntimeError(f"Error setting up UV environment: {str(e)}")

    def send(self, command: str, cwd: Path | str | None = None, bypass_env_check: bool = False) -> CommandResult:
        """Sends any arbitrary command to be executed in the UV environment.
        
        Args:
            command (str): The command to execute
            cwd (Path | str | None): Working directory for the command
            bypass_env_check (bool, optional): Whether to bypass the env setup check
                Only really used to allow initial setup commands to pass into the env.
            
        Returns:
            CommandResult containing command output and status
            
        Example:
            >>> uv = UVManager("./my_venv")
            >>> uv.initialize()
            >>> result = uv.send("python --version")
            >>> assert result.success
            >>> print(result.stdout)
            Python 3.10.x
        """
        # (1) We cannot run commands until the env is setup
        if not self.env_ready and not bypass_env_check:
            raise RuntimeError("Environment not ready. Call initialize() first.")

        # (2) Add a change of directory to the command if cwd is passed
        if cwd:
            command = f"cd {cwd} && {command}"

        # Run the command and get the outputs, errors and return code
        stdout, stderr = self._run_in_shell(command)
        retcode_out, _ = self._run_in_shell("echo $?")
        
        try:
            returncode = int(retcode_out.strip())
        except ValueError:
            returncode = -1
            
        return CommandResult(        
            command=command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr
        )

    def run(self, script: str | Path, args: list[str] | None = None) -> CommandResult:
        """Alias to uv_run
        
        Args:
            script (str | Path): Path to the Python script to run
            args (list[str] | None): List of arguments to pass to the script
            
        Returns:
            CommandResult containing script output and status

        Example:
            >>> uv = UVManager("./my_venv")
            >>> uv.initialize()
            >>> result = uv.run("script.py", ["--arg1", "value1"])
        """
        return self.uv_run(script, args)  # Forwarding arguments properly
    
    def uv_run(self, script: str | Path, args: list[str] | None = None) -> CommandResult:
        """Run a Python script using UV's run command.
        
        Args:
            script (str | Path): Path to the Python script to run
            args (list[str] | None): List of arguments to pass to the script
            
        Returns:
            CommandResult containing script output and status
            
        Example:
            >>> uv = UVManager("./my_venv")
            >>> uv.initialize()
            >>> result = uv.uv_run("script.py", ["--arg1", "value1"])
        """
        cmd = ["uv", "run"]
        if args:
            cmd.extend(args)
        cmd.append(str(script))
        
        return self.send(" ".join(cmd))

    def pip_install(
        self, 
        package: str, 
        editable: bool = False, 
        cwd: Path | str | None = None,
        verbose: bool = True
    ) -> CommandResult:
        """Install a package using UV's pip interface.
        
        Args:
            package (str): Package specification (name, path, or requirements file)
            editable (bool): If True, install in editable mode (-e flag)
            verbose (bool): If True, print installation progress
            
        Returns:
            CommandResult containing installation output and status
            
        Example:
            >>> uv = UVManager("./my_venv")
            >>> uv.initialize()
            >>> result = uv.pip_install("requests")
            >>> assert result.success
        """
        if not self.env_ready:
            raise RuntimeError("Environment not ready. Call initialize() first.")

        # If package is a string, split it safely, or handle it as a list
        if isinstance(package, str):
            package_list = shlex.split(package)
        else:
            package_list = [package,]
    
        cmd = ["uv", "pip", "install"]
        if editable:
            cmd.append("-e")
        cmd.extend(package_list)
        
        result = self.send(" ".join(cmd), cwd=cwd)
        
        if verbose:
            if result.stdout:
                self._logger.info(result.stdout)
            if result.stderr:
                if "Installed" in result.stderr or "Resolved" in result.stderr or "Using Python" in result.stderr:
                    self._logger.info(result.stderr)
                else:
                    self._logger.error(result.stderr)
                
        return result

    def cleanup(self) -> None:
        """Clean up resources and terminate the shell session.
        
        This should be called when done using the environment if not using
        the context manager.
        """
        if self._shell:
            self._shell.terminate()
            self._shell = None
        self.env_ready = False


import logging
from rich.logging import RichHandler

# Remove existing handlers (important in Jupyter environments)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Setup logging with RichHandler
logging.basicConfig(
    level=logging.DEBUG,  # Ensure debug messages are shown
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)

logger = logging.getLogger(__name__)

# Test logs
logger.info("This is an info message.")
logger.warning("This is a warning.")
logger.error("This is an error message.")
logger.debug("This is a debug message (should show if level is DEBUG).")


_ENV_NAME = "testbed_venv"

if os.path.isdir(_ENV_NAME):
    shutil.rmtree(_ENV_NAME)

# Context manager
with UVManager(f"./{_ENV_NAME}") as uv:
    uv.pip_install("requests")
    print(uv.send("uv pip list").stdout)
    print(uv.send("pwd").stdout)

# print("\n\n")

# # Direct usage
# uv = UVManager(f"./{_ENV_NAME}")
# uv.initialize()
# uv.pip_install("requests")
# print(uv.send("uv pip list").stdout)
# print(uv.send("pwd").stdout)
# uv.cleanup()


# First create a basic requirements.txt
requirements = """
requests==2.31.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
python-dotenv==1.0.0
pyyaml>=6.0.0
"""

# Write the requirements file
with open("requirements.txt", "w") as f:
    f.write(requirements.strip())

# Now use UV to install from the requirements file
with UVManager(f"./{_ENV_NAME}") as uv:
    # First verify Python is working
    py_ver = uv.send("python --version")
    print(f"Using Python: {py_ver.stdout}")
    
    # Install from requirements file
    print("\nInstalling from requirements.txt...")
    result = uv.pip_install("-r requirements.txt")
    
    if result.returncode != 0:
        print(f"Installation failed: {result.stderr}")
    else:
        # Verify installations
        print("\nVerifying installations...")
        verify = uv.send("uv pip list")
        print("Installed packages:")
        print(verify.stdout)


class GitHubRepo:
    """Handles operations with a GitHub repository (cloning, checkout, patching).

    Attributes:
        instance_id (str): A unique string identifier for the instance (aka GitHub issue).
        repo_name (str): The name of Github repository
        org_name (str): The name of owner of the Github repository
        repo_url (str): The full GitHub repository URL (e.g., "https://github.com/user/repo.git")
        env_setup_commit_hash (str): The commit hash used for environment setup.
        base_commit_hash (str | None): The commit hash for the relevant issue.
        root_dir (Path): Path to the root directory where all temporary directories will be.
        root_repo_path (Path): Path to this repo's specific temporary directory
        _logger (logging.Logger): Internal logger instance
    """
    instance_id: str
    repo_name: str
    org_name: str
    repo_url: str
    env_setup_commit_hash: str
    base_commit_hash: str
    root_dir: Path
    root_repo_path: Path
    _logger: logging.Logger

    def __init__(
            self,
            instance_id: str,
            repo_name: str,
            org_name: str,
            repo_url: str,
            env_setup_commit_hash: str,
            base_commit_hash: str | None = None,
            root_dir: str | Path = "/kaggle/tmp"
    ) -> None:
        """Initializes the GitHubRepo object.

        Args:
            instance_id (str):
                 A unique string identifier for the instance (aka GitHub issue).
            repo_name (str):
                The name of Github repo (repo)
            org_name (str):
                The name of owner of the Github repo (repo)
            repo_url (str):
                The full GitHub repository URL (e.g., "https://github.com/user/repo.git")
            env_setup_commit_hash (str):
                The commit hash used for environment setup.
            base_commit_hash (str, optional):
                The commit hash for the relevant issue.
            root_dir (str | Path, optional):
                The root directory where all temporary directories will be stored.
        """
        # (1) Set attributes related to commit hashes
        self.env_setup_commit_hash = env_setup_commit_hash
        self.base_commit_hash = base_commit_hash or env_setup_commit_hash

        # (2) Set attributes related to the repository name and URL
        self.instance_id = instance_id
        self.repo_name = repo_name
        self.org_name = org_name
        self.repo_url = repo_url

        # (3) Set attributes related to the root directory and paths
        self.root_dir = Path(root_dir)
        self.root_repo_path = self.find_free_directory()

        # (4) Initialize the logger
        self._logger = logging.getLogger(self.__class__.__name__)

    def find_free_directory(self, base_dir: str | Path | None = None) -> Path:
        """Finds a free directory in the base directory, creating one if necessary.
        
        Args:
            base_dir (str | Path, optional): 
                Base directory to search within.
        
        Returns:
            Path:
                The path object pointing to the newly created directory.
        
        Raises:
            RuntimeError: If no free directory could be found after multiple attempts.
        """
        base_path = Path(base_dir or self.root_dir)
        for i in range(100):
            dir_name = f"temp_env_{i}"
            temp_dir = base_path / dir_name
            if not temp_dir.exists():
                temp_dir.mkdir(parents=True, exist_ok=True)
                return temp_dir
        raise RuntimeError("Could not find a free temporary directory")
    
    def clone_and_checkout(self, checkout_commit_hash: str | None = None) -> Path:
        """Clones the repository and checks out the specified commit.

        Args:
            checkout_commit_hash (str, optional):
                The commit hash that we will checkout.
                If not provided we will use the commit hash for the environment setup.

        Returns:
            The local filesystem path to the cloned repo.

        Raises:
            RuntimeError: If clone or checkout fails.
        """
        try:
            # (1) Clone the repository
            self.clone_repo()

            # (2) Checkout the specified commit
            self.checkout_commit(checkout_commit_hash or self.env_setup_commit_hash)

        # (3) Handle any errors
        except subprocess.CalledProcessError as e:
            self._logger.error(f"Git clone/checkout error: {e.stderr}")
            raise RuntimeError(f"Git clone/checkout error: {e.stderr}") from e

        # (4) Return the path to the cloned repository
        return self.root_repo_path

    def clone_repo(self, force_reclone: bool = True) -> "GitHubRepo":
        """Clones the repository into the specified path.

        Args:
            force_reclone (bool, optional):
                Whether to force re-cloning the repository if it already exists.

        Returns:
            GitHubRepo: The instance itself (allows method chaining).
        """
        # (1) If repo (validated by .git) already exists, do nothing
        if (self.root_repo_path / ".git").exists():
            # (1a) If force_reclone is False, log a warning and return
            if not force_reclone:
                self._logger.warning(f"Repository path {self.root_repo_path} already exists. Skipping clone.")
                return self
            # (1b) If force_reclone is True, remove the existing directory and proceed.
            self._logger.warning(f"Repository path {self.root_repo_path} already exists. Recloning...")
            shutil.rmtree(self.root_repo_path)
            self.root_repo_path.mkdir(parents=True, exist_ok=True)

        # (2) Clone a single branch of the repository
        self._logger.info(f"Cloning {self.repo_url} to {self.root_repo_path}...")
        try:
            # Clone with --no-single-branch to get all branches
            subprocess.run(
                ["git", "clone", "--no-single-branch", self.repo_url, str(self.root_repo_path)],
                capture_output=True, text=True, check=True
            )
            
            # Fetch all tags and remote branches
            subprocess.run(
                ["git", "fetch", "--all", "--tags", "--prune"],
                cwd=self.root_repo_path,
                capture_output=True, text=True, check=True
            )

        # (3) Handle any errors
        except subprocess.CalledProcessError as e:
            self._logger.error(f"Git clone error: {e.stderr}")
            raise RuntimeError(f"Git clone error: {e.stderr}") from e

        # (4) Enable method chaining
        return self

    def checkout_commit(self, commit_hash: str) -> "GitHubRepo":
        """Checks out the specified commit hash in the cloned repository.

        Args:
            commit_hash (str): The commit hash to checkout.

        Returns:
            GitHubRepo: The instance itself (allows method chaining).
        """
        # (1) Check if the repository path exists
        if not (self.root_repo_path / ".git").exists():
            raise RuntimeError(f"Repository path {self.root_repo_path} does not exist. Clone the repo first.")

        # (2) Checkout the specified commit
        try:
            subprocess.run(
                ["git", "checkout", commit_hash],
                cwd=self.root_repo_path, capture_output=True, text=True, check=True
            )
            self._logger.info(f"Checked out commit {commit_hash}.")

        # (3) Handle any errors
        except subprocess.CalledProcessError as e:
            self._logger.error(f"Git checkout error: {e.stderr}")
            raise RuntimeError(f"Git checkout error: {e.stderr}") from e

        # (4) Enable method chaining
        return self

    def apply_patch(self, patch_content: str) -> "GitHubRepo":
        """Applies a patch to the cloned repository.

        Args:
            patch_content (str):
                The diff/patch content as a string.

        Raises:
            RuntimeError: If patch application fails.
        """
        # (1) Write the patch content to a file in the temporary directory
        patch_file = self.root_repo_path / "local_changes.patch"
        patch_file.write_text(patch_content)

        # (2) Apply the patch
        self._logger.info(f"Applying patch at {patch_file}...")
        try:
            subprocess.run(
                ["git", "apply", str(patch_file)],
                cwd=self.root_repo_path, capture_output=True, text=True, check=True
            )
            self._logger.info("Patch applied successfully.")
        except subprocess.CalledProcessError as e:
            self._logger.error(f"Patch application error: {e.stderr}")
            raise RuntimeError(f"Patch application error: {e.stderr}") from e

        # Enable method chaining
        return self
        
    @classmethod
    def from_swebench_instance(
            cls,
            instance: SWEBenchInstance,
            root_dir: str | Path = "/kaggle/tmp"
    ) -> "GitHubRepo":
        """Creates a GitHubRepo instance from a SWEBenchInstance object.

        Args:
            instance (SWEBenchInstance):
                The SWE-Bench instance to create the GitHubRepo object from.

        Returns:
            GitHubRepo:
                A GitHubRepo object initialized with the instance's repo URL and commit hash.
        """
        org_name, repo_name = instance.repo.split("/")
        return GitHubRepo(
            instance_id=instance.instance_id,
            repo_name=repo_name.strip(),
            org_name=org_name.strip(),
            repo_url=instance.github_repo_url,
            base_commit_hash=instance.base_commit,
            env_setup_commit_hash=instance.environment_setup_commit,
            root_dir=root_dir
        )

    def cleanup(self) -> None:
        """Cleans up the cloned repository directory."""
        self._logger.info(f"Cleaning up temporary repository directory {self.root_repo_path}...")
        shutil.rmtree(self.root_repo_path, ignore_errors=True)
        self._logger.info("Temporary repository directory cleaned up.")

    def __repr__(self) -> str:
        """String representation of the GitHubRepo object."""
        return (
            f"GitHubRepo("
            f"rood_dir={self.root_dir}, "
            f"root_repo_path={self.root_repo_path}, "
            f"repo_name={self.repo_name}, "
            f"org_name={self.org_name}, "
            f"repo_url={self.repo_url}, "
            f"base_commit_hash={self.base_commit_hash}"
            f"env_setup_commit_hash={self.env_setup_commit_hash})"
        )

rich.print("[bold green]Setup our Github Repo object (clones to a new temporary folder) [/bold green]")
demo_github_repo = GitHubRepo.from_swebench_instance(demo_instance, root_dir="/kaggle/working")
demo_repo_path = demo_github_repo.clone_and_checkout()
# demo_github_repo.cleanup()

# rich.print("[bold red]Show that we can apply the patches and they don't error out.[/bold red]")
# demo_github_repo.apply_patch(demo_instance.patch)
# demo_github_repo.apply_patch(demo_instance.test_patch)


class PythonVersionDetector:
    """Detects appropriate Python versions for a project using modern parsing approaches.

    Attributes:
        python_versions (list[PythonVersion]):
            Available Python versions from the Universal Python Installer
        _logger (logging.Logger): Internal logger instance
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self.python_versions = self._get_available_python_versions()

    def _parse_uv_python_list(self, output: str) -> list[PythonVersion]:
        """Parse the output of 'uv python list' command.

        Args:
            output (str): The output string from 'uv python list'.

        Returns:
            list[PythonVersion]:
                A list of PythonVersion objects.
        """
        # (1) Initialize the list of versions
        versions = []

        # (2) Parse the output line by line
        for line in output.splitlines():

            # (3a) Skip empty lines
            if not line.strip():
                continue

            # (3b) Split the line into parts
            parts = line.split(maxsplit=1)

            # (3c) Skip if no parts
            if not parts:
                continue

            # (3d) Extract version info and availability
            version_info = parts[0]
            availability = parts[1] if len(parts) > 1 else "<download available>"

            # (3e) Match the version info
            match = re.match(
                r'(cpython|pypy)-(\d+\.\d+\.\d+(?:a\d+)?)'
                r'(\+freethreaded)?-linux-x86_64-gnu',
                version_info
            )

            # (3f) Skip if no match
            if not match:
                continue

            # (3g) Extract the implementation, version, and freethreaded info
            impl, version, freethreaded = match.groups()
            implementation = (
                PythonImplementation.CPYTHON if impl == "cpython"
                else PythonImplementation.PYPY
            )

            # (3h) Check if the version is installed
            is_installed = not availability.startswith("<download")
            path = availability if is_installed else None

            # (3i) Append the version to the list if it's installed and available
            if path and " -> " in path:
                path = path.split(" -> ")[0].strip()

            # (3j) Append the version to the list
            versions.append(PythonVersion(
                implementation=implementation,
                version=version,
                is_installed=is_installed,
                path=path,
                is_freethreaded=bool(freethreaded)
            ))

        # (4) Return the list of versions
        return versions

    def _get_available_python_versions(self) -> list[PythonVersion]:
        """Get list of available Python versions from UV.

        Returns:
            list[PythonVersion]:
                A list of PythonVersion objects.
        """
        # (1) Run the 'uv python list' command
        try:
            result = subprocess.run(
                ["uv", "python", "list"],
                capture_output=True,
                text=True,
                check=True
            )
            # (2) Parse the output
            return self._parse_uv_python_list(result.stdout)
        # (3) Handle any errors
        except subprocess.CalledProcessError as e:
            self._logger.warning(f"Failed to list Python versions: {e.stderr}")
            return []

    def parse_pyproject_toml(self, path: Path) -> VersionConstraints | None:
        """Parse Python version constraints from pyproject.toml using proper TOML parser.

        Args:
            path (Path): The path to the pyproject.toml file.

        Returns:
            VersionConstraints | None:
                The parsed version constraints or None if not found.
        """
        # (1) Attempt to parse the pyproject.toml file
        try:
            # (2a) Load the TOML data
            data = tomli.loads(path.read_text())

            # (2b) Check project.requires-python (PEP 621)
            if "project" in data and "requires-python" in data["project"]:
                return VersionConstraints.from_specifier_string(
                    data["project"]["requires-python"],
                    "pyproject.toml",
                    confidence=1.0
                )

            # (2c) Check poetry dependencies (looks for "tool.poetry.dependencies.python")
            if "tool" in data and "poetry" in data:
                if "dependencies" in data["tool"]["poetry"]:
                    if "python" in data["tool"]["poetry"]["dependencies"]:
                        return VersionConstraints.from_specifier_string(
                            data["tool"]["poetry"]["dependencies"]["python"],
                            "pyproject.toml (poetry)",
                            confidence=0.9
                        )

            # (2d) Check PDM dependencies (looks for "tool.pdm.python")
            if "tool" in data and "pdm" in data:
                if "python" in data["tool"]["pdm"]:
                    return VersionConstraints.from_specifier_string(
                        data["tool"]["pdm"]["python"],
                        "pyproject.toml (pdm)",
                        confidence=0.9
                    )
        # (3) Handle any errors
        except Exception as e:
            self._logger.warning(f"Error parsing pyproject.toml: {e}")

        # (4) Return None if no constraints found
        return None

    def parse_setup_py(self, path: Path) -> VersionConstraints | None:
        """Parse Python version constraints from setup.py file.

        Uses multiple strategies to detect version constraints:
            1. Analyzes Python version classifiers
            2. Checks for explicit python_requires in setup arguments
            3. Infers version range from classifier information

        Args:
            path (Path): Path to the setup.py file to analyze

        Returns:
            VersionConstraints | None: Parsed version constraints if found, None otherwise
        """
        # (1) Read and parse the setup.py file
        try:
            content = path.read_text()
            
            # (2a) Look for Python version classifiers with single quotes
            classifiers = re.findall(
                r"'Programming Language :: Python :: (\d+\.\d+)'",
                content
            )
            
            # (2b) If none found, try double quotes
            if not classifiers:
                classifiers = re.findall(
                    r'"Programming Language :: Python :: (\d+\.\d+)"',
                    content
                )
            
            # (3) Process classifier versions if found
            if classifiers:
                # (3a) Filter to only Python 3 versions
                versions = [v for v in classifiers if not v.startswith('2')]
                
                if versions:
                    # (3b) Get min and max supported versions
                    min_ver = min(versions, key=lambda x: parse(x))
                    max_ver = max(versions, key=lambda x: parse(x))
                    
                    # (3c) Create constraint up to next major version
                    major, minor = map(int, max_ver.split('.'))
                    next_minor = f"{major}.{minor + 1}"
                    spec = f">={min_ver},<{next_minor}"
                    
                    return VersionConstraints.from_specifier_string(
                        spec,
                        "setup.py (classifiers)",
                        confidence=0.7
                    )
            
            # (4) Check for explicit python_requires
            requires_match = re.search(
                r'python_requires\s*=\s*[\'"]([^\'"]+)[\'"]',
                content
            )
            if requires_match:
                return VersionConstraints.from_specifier_string(
                    requires_match.group(1),
                    "setup.py (python_requires)",
                    confidence=0.9
                )
                
        # (5) Handle any parsing errors
        except Exception as e:
            self._logger.warning(f"Error parsing setup.py: {e}")
            
        return None
    
    def parse_setup_cfg(self, path: Path) -> VersionConstraints | None:
        """Parse Python version constraints from setup.cfg.

        Args:
            path (Path): The path to the setup.cfg file.

        Returns:
            VersionConstraints | None:
                The parsed version constraints or None if not found.
        """
        # (1) Attempt to parse the setup.cfg file
        try:
            # (2a) Load the setup.cfg data
            from configparser import ConfigParser
            config = ConfigParser()
            config.read(path)

            # (2b) Check for "options.python_requires" in the "metadata" section (PEP 518)
            if "options" in config:
                if "python_requires" in config["options"]:
                    return VersionConstraints.from_specifier_string(
                        config["options"]["python_requires"],
                        "setup.cfg",
                        confidence=0.8
                    )
        # (3) Handle any errors
        except Exception as e:
            self._logger.warning(f"Error parsing setup.cfg: {e}")

        # (4) Return None if no constraints found
        return None

    def parse_requirements_txt(self, path: Path) -> VersionConstraints | None:
        """Parse Python version constraints from requirements.txt using packaging.requirements.

        Args:
            path (Path): The path to the requirements.txt file.

        Returns:
            VersionConstraints | None:
                The parsed version constraints or None if not found.
        """
        # (1) Attempt to parse the requirements.txt file
        try:
            # (2a) Read the requirements.txt file
            content = path.read_text()
            constraints = []

            # (2b) Parse each line in the file (ignoring comments and empty lines)
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # (2c) Attempt to parse the requirement (ignoring any errors and looking for python_version)
                try:
                    req = Requirement(line)
                    if "python_version" in str(req.marker):
                        # Extract the version constraint from the marker
                        marker_str = str(req.marker)
                        version_part = re.search(r'python_version([^"\']+)["\']([^"\']+)["\']', marker_str)
                        if version_part:
                            op, ver = version_part.groups()
                            constraints.append(f"{op}{ver}")
                except Exception:
                    continue

            # (2d) Return the constraints if found
            if constraints:
                return VersionConstraints.from_specifier_string(
                    ",".join(constraints),
                    "requirements.txt",
                    confidence=0.7
                )

        # (3) Handle any errors
        except Exception as e:
            self._logger.warning(f"Error parsing requirements.txt: {e}")

        # (4) Return None if no constraints found
        return None

    def get_project_constraints(self, repo_path: Path | str) -> VersionConstraints | None:
        """Get Python version constraints from all project configuration files.

        Checks multiple config files in order of reliability:
            1. pyproject.toml (PEP 621)
            2. setup.cfg 
            3. setup.py
            4. requirements.txt

        Args:
            repo_path (Path | str): Path to the repository root

        Returns:
            VersionConstraints | None: Version constraints if found, None otherwise
        """
        # (1) Ensure path is a Path object
        repo_path = Path(repo_path)

        # (2) Define config files to check in priority order
        config_files = [
            (repo_path / "pyproject.toml", self.parse_pyproject_toml),
            (repo_path / "setup.cfg", self.parse_setup_cfg),
            (repo_path / "setup.py", self.parse_setup_py),
            (repo_path / "requirements.txt", self.parse_requirements_txt)
        ]

        # (3) Try each config file in order
        for file_path, parser in config_files:
            if file_path.exists():
                if constraints := parser(file_path):
                    self._logger.info(
                        f"Found constraints in {file_path.name}: "
                        f"{constraints.specifier_set} "
                        f"(confidence: {constraints.confidence})"
                    )
                    return constraints

        # (4) No constraints found in any file
        return None

    def select_python_version(
        self,
        repo_path: Path | str,
        python_fallback_version: str = "3.10",
        newest_allowed_is_fallback: bool = True,
    ) -> PythonVersion:
        """Select the most appropriate Python version for a project.

        Uses the following selection process:
            1. Gets project version constraints from config files
            2. Filters available CPython versions to stable releases
            3. Filters versions to those matching constraints
            4. Selects newest compatible version
            5. Falls back to specified version if no match
            6. If specified version is newer than fallback optionally set to fallback

        Args:
            repo_path (Path | str): Path to the repository root
            python_fallback_version (str): Version to use if no compatible version found

        Returns:
            PythonVersion: Selected Python version object
        """
        # (1) Get project constraints
        constraints = self.get_project_constraints(repo_path)
        
        # (2) Filter to stable CPython versions
        cpython_versions = [
            v for v in self.python_versions
            if v.implementation == PythonImplementation.CPYTHON
            and not re.search(r'a|b|rc', v.version)
        ]
        
        # (3) Apply version constraints if found
        if constraints:
            try:
                # (3a) Filter to versions matching constraints
                compatible_versions = []
                for version in cpython_versions:
                    try:
                        if version.parsed_version in constraints.specifier_set:
                            compatible_versions.append(version)
                            self._logger.debug(
                                f"Version {version.version} matches constraints"
                            )
                        else:
                            self._logger.debug(
                                f"Version {version.version} excluded by constraints"
                            )
                    except Exception as e:
                        self._logger.warning(
                            f"Error checking version {version.version}: {e}"
                        )
            except Exception as e:
                self._logger.warning(f"Error applying constraints: {e}")
                compatible_versions = cpython_versions
        else:
            compatible_versions = cpython_versions
            
        # (4) Sort compatible versions newest first
        sorted_versions = sorted(
            compatible_versions,
            key=lambda v: v.parsed_version,
            reverse=True
        )
        
        # (5) Return newest compatible version or fallback
        if not sorted_versions:
            self._logger.warning(
                f"No compatible versions found, using fallback {python_fallback_version}"
            )
            return PythonVersion(
                implementation=PythonImplementation.CPYTHON,
                version=f"{python_fallback_version}.0",
                is_installed=False,
                path=None
            )

        # (6) Return the selected newest version that meets requirements
        selected = sorted_versions[0]
        if selected.base_version>python_fallback_version and newest_allowed_is_fallback:
            self._logger.info(
                f"Selected version {python_fallback_version}.0 "
                f"because actual selected version is newer."
            )
            return PythonVersion(
                implementation=PythonImplementation.CPYTHON,
                version=f"{python_fallback_version}.0",
                is_installed=False,
                path=None
            )
        self._logger.info(
            f"Selected version {selected.version} "
            f"from {len(sorted_versions)} compatible versions"
        )
        return selected

### ######################## ###
### Let's see how to use it. ###
### ######################## ###
# Initialize the detector
detector = PythonVersionDetector()

# Get project constraints
constraints = detector.get_project_constraints(demo_repo_path)

if constraints:
    rich.print(f"Min version: {constraints.min_version}")
    rich.print(f"Max version: {constraints.max_version}")
    rich.print(f"Source: {constraints.source_file}")
    rich.print(f"Confidence: {constraints.confidence}")

# Select the best Python version
selected_version = detector.select_python_version(demo_repo_path)
rich.print(f"\nSelected Python version: {selected_version.version}")
### ######################## ###


# class PythonVersionDetectionMixin:
#     """A mixin that provides methods to:
      
#       (1) Parse installed versions from `uv python list`
#       (2) Extract Python version constraints from project files
#       (3) Pick the newest installed CPython version that satisfies constraints
#     """
#     def __init__(self):
#         self._logger = logging.getLogger(self.__class__.__name__)
#         self.python_versions = self._get_available_python_versions()
        
#     def _parse_uv_python_list(self, output: str) -> list[PythonVersion]:
#         """Parse the output of 'uv python list' command.

#         Args:
#             output (str): Raw output from 'uv python list'

#         Returns:
#             list[PythonVersion]: List of parsed Python versions
#         """
#         versions = []
#         for line in output.splitlines():
#             if not line.strip():
#                 continue

#             # Split into version info and path/availability
#             parts = line.split(maxsplit=1)
#             if not parts:
#                 continue

#             version_info = parts[0]
#             availability = parts[1] if len(parts) > 1 else "<download available>"

#             # Parse the version string
#             # Example: cpython-3.10.12-linux-x86_64-gnu
#             match = re.match(
#                 r'(cpython|pypy)-(\d+\.\d+\.\d+(?:a\d+)?)'
#                 r'(\+freethreaded)?-linux-x86_64-gnu',
#                 version_info
#             )
#             if not match:
#                 continue

#             impl, version, freethreaded = match.groups()
#             implementation = (
#                 PythonImplementation.CPYTHON if impl == "cpython"
#                 else PythonImplementation.PYPY
#             )

#             is_installed = not availability.startswith("<download")
#             path = availability if is_installed else None
#             if path and " -> " in path:  # Handle symlinks
#                 path = path.split(" -> ")[0].strip()

#             versions.append(PythonVersion(
#                 implementation=implementation,
#                 version=version,
#                 is_installed=is_installed,
#                 path=path,
#                 is_freethreaded=bool(freethreaded)
#             ))

#         return versions

#     def _get_available_python_versions(self) -> list[PythonVersion]:
#         """Get list of available Python versions from UV.

#         Returns:
#             list[PythonVersion]: List of available Python versions
#         """
#         try:
#             result = subprocess.run(
#                 ["uv", "python", "list"],
#                 capture_output=True,
#                 text=True,
#                 check=True
#             )
#             return self._parse_uv_python_list(result.stdout)
#         except subprocess.CalledProcessError as e:
#             self._logger.warning(f"Failed to list Python versions: {e.stderr}")
#             return []

#     def _get_pyproject_constraints(self, pyproject_path: Path) -> str | None:
#         """Extract Python version constraints from pyproject.toml file.

#         Args:
#             pyproject_path (Path | str):
#                 The path to the pyproject.toml file.

#         Returns:
#             The version constraints for Python (or None if not found)
#         """
#         try:
#             content = pyproject_path.read_text()

#             # Common patterns in pyproject.toml
#             patterns = [
#                 r'requires-python\s*=\s*[\'"]([^\'"]*)[\'"]',  # PEP 621
#                 r'python\s*=\s*[\'"]([^\'"]*)[\'"]',  # poetry
#                 r'python_version\s*=\s*[\'"]([^\'"]*)[\'"]'  # other tools
#             ]

#             for pattern in patterns:
#                 if match := re.search(pattern, content):
#                     return match.group(1)

#             # If no direct match, look for version patterns near 'python'
#             python_lines = [line for line in content.splitlines()
#                             if 'python' in line.lower()]
#             version_pattern = r'(?:^|\D)(\d+\.\d+(?:\.\d+)?)'

#             versions = []
#             for line in python_lines:
#                 if matches := re.finditer(version_pattern, line):
#                     versions.extend(match.group(1) for match in matches)

#             if versions:
#                 # Convert to constraint format
#                 base_versions = sorted(
#                     set('.'.join(v.split('.')[:2]) for v in versions),
#                     key=lambda v: tuple(map(int, v.split('.')))
#                 )
#                 min_ver = base_versions[0]
#                 max_ver = base_versions[-1]
#                 major, minor = map(int, max_ver.split('.'))
#                 return f">={min_ver},<{major}.{minor + 1}"

#             return None
#         except Exception as e:
#             self._logger.warning(f"Error parsing pyproject.toml: {e}")
#             return None

#     def _get_requirements_constraints(self, requirements_path: Path | str) -> str | None:
#         """Extract Python version constraints from requirements.txt file.

#         Args:
#             requirements_path (Path | str):
#                 The path to the requirements.txt file.

#         Returns:
#             The version constraints for Python (or None if not found)
#         """
#         try:
#             content = Path(requirements_path).read_text()

#             # Look for python_version markers
#             markers = [
#                 r'python_version\s*([><=!~]+)\s*[\'"]([^\'"]+)[\'"]',
#                 r'python_version\s*in\s*[\'"]([^\'"]+)[\'"]'
#             ]

#             constraints = []
#             for line in content.splitlines():
#                 if 'python_version' in line:
#                     for pattern in markers:
#                         if match := re.search(pattern, line):
#                             if len(match.groups()) == 2:  # operator and version
#                                 op, ver = match.groups()
#                                 constraints.append(f"{op}{ver}")
#                             else:  # in operator with version list
#                                 versions = match.group(1).split(',')
#                                 clean_versions = [v.strip(' \'\"') for v in versions]
#                                 return f">={min(clean_versions)},<{max(clean_versions)}"

#             return ','.join(constraints) if constraints else None

#         except Exception as e:
#             self._logger.warning(f"Error parsing requirements.txt: {e}")
#             return None

#     def _get_setup_py_python_constraints(self, setup_path: Path | str) -> str | None:
#         """Extract Python version constraints from setup.py file.

#         Args:
#             setup_path (Path | str):
#                 The path to the setup.py file.

#         Returns:
#             The version constraints for Python (or None if not found)
#         """
#         try:
#             content = Path(setup_path).read_text()

#             # Find all lines containing 'python' case-insensitive
#             python_lines = [line for line in content.splitlines()
#                             if 'python' in line.lower()]

#             version_pattern = r'(?:^|\D)(\d+\.\d+(?:\.\d+)?)'
#             versions = set()

#             for line in python_lines:
#                 # Look for version patterns on python lines
#                 if matches := re.finditer(version_pattern, line):
#                     versions.update(match.group(1) for match in matches)

#             if versions:
#                 # Convert to major.minor only and sort
#                 base_versions = sorted(
#                     set('.'.join(v.split('.')[:2]) for v in versions),
#                     key=lambda v: tuple(map(int, v.split('.')))
#                 )

#                 # Get min/max versions that start with '3'
#                 py3_versions = [v for v in base_versions if v.startswith('3')]
#                 if py3_versions:
#                     min_version = py3_versions[0]
#                     max_version = py3_versions[-1]

#                     # Convert to constraint (e.g., ">=3.5,<3.9")
#                     major, minor = map(int, max_version.split('.'))
#                     next_minor = f"{major}.{minor + 1}"
#                     return f">={min_version},<{next_minor}"

#             return None

#         except Exception as e:
#             self._logger.warning(f"Error parsing setup.py: {e}")
#             return None

#     def _get_project_python_constraints(self, repo_path: str | Path) -> str | None:
#         """Get Python version constraints from project configuration files.

#         Checks pyproject.toml, setup.py, and requirements.txt in order of preference.

#         Args:
#             repo_path (str | Path):
#                 Path to the repo we are finding the best python version for.

#         Returns:
#             str | None: Version constraint string if found, None otherwise
#         """
#         # Check pyproject.toml first (PEP 621)
#         pyproject_path = Path(repo_path) / "pyproject.toml"
#         if pyproject_path.exists():
#             if constraints := self._get_pyproject_constraints(pyproject_path):
#                 return constraints

#         # Then check setup.py
#         setup_path = Path(repo_path) / "setup.py"
#         if setup_path.exists():
#             if constraints := self._get_setup_py_python_constraints(setup_path):
#                 return constraints

#         # Finally check requirements.txt
#         requirements_path = Path(repo_path) / "requirements.txt"
#         if requirements_path.exists():
#             if constraints := self._get_requirements_constraints(requirements_path):
#                 return constraints

#         return None

#     def _version_satisfies_constraint(self, version: PythonVersion, constraint: str) -> bool:
#         """Check if version satisfies the given constraint.

#         Args:
#             version (PythonVersion): Version to check
#             constraint (str): Version constraint (e.g., ">=3.8,<3.11")

#         Returns:
#             bool: True if version satisfies constraint, False otherwise
#         """
#         try:
#             from packaging.specifiers import SpecifierSet
#             from packaging.version import Version

#             spec = SpecifierSet(constraint)
#             # Use base version for comparison (e.g., 3.10 instead of 3.10.12)
#             return Version(version.base_version) in spec
#         except ImportError:
#             self._logger.warning("packaging module not available, falling back to simple version comparison")
#             return version.base_version >= constraint.replace('>=', '').replace('>', '').strip()
#         except Exception as e:
#             self._logger.warning(f"Error checking version constraint: {e}")
#             return False

#     def _select_python_version(self, repo_path: str | Path) -> PythonVersion:
#         """Select the most appropriate Python version based on constraints and availability.

#         Args:
#             repo_path (str | Path):
#                 Path to the repo we are finding the best python version for.
        
#         Returns:
#             PythonVersion: Selected Python version object
#         """
#         constraint = self._get_project_python_constraints(repo_path)

#         # Filter to only CPython versions (excluding alpha/beta)
#         cpython_versions = [
#             v for v in self.python_versions
#             if v.implementation == PythonImplementation.CPYTHON
#                and not re.search(r'a|b|rc', v.version)
#         ]

#         if constraint:
#             # Filter versions that satisfy the constraint
#             compatible_versions = [
#                 v for v in cpython_versions
#                 if self._version_satisfies_constraint(v, constraint)
#             ]
#         else:
#             compatible_versions = cpython_versions

#         # Sort by version number (newest first)
#         sorted_versions = sorted(
#             compatible_versions,
#             key=lambda v: tuple(map(int, v.version.split('.'))),
#             reverse=True
#         )

#         if not sorted_versions:
#             # Create a fallback version object
#             self._logger.warning(f"No compatible versions found, using fallback {self.fallback_python_version}")
#             return PythonVersion(
#                 implementation=PythonImplementation.CPYTHON,
#                 version=f"{self.fallback_python_version}.0",
#                 is_installed=False,
#                 path=None
#             )

#         return sorted_versions[0]


class RepoUVManager(UVManager):
    """A specialized UVManager for tackling SWE-problems.
    
    This class can:
        - clone a GitHub repo
        - check out commits
        - detect Python versions
        - apply patches
        - install dependencies from the local cloned directory

    Inherits:
        UVManager: The base environment manager with persistent shell session.

    Attributes:
        github_repo (GitHubRepo): A reference to the GitHubRepo object managing our cloned repo.
        python_version_detector (PythonVersionDetector): The object responsible for identifying the python version to use.
        repo_path (Path | None): The local filesystem path where the repo is cloned.
        venv_name (str): The name of the virtual environment (usually unique and generated).
        venv_path (Path): The full path to the virtual environment including the name.

    Extended Features:
        - Automatic detection of Python version availability (via `uv python list`).
        - Support for installing dependencies from requirements.txt, pyproject.toml, or setup.py.
        - Ability to run pytest easily via `run_pytest`.
        - Optional patch application through the GitHubRepo reference.
    """
    
    def __init__(
        self,
        venv_dir_path: str | Path,
        github_repo: "GitHubRepo",
        auto_detect_python: bool = True,
        fallback_python_version: str = "3.10",
        venv_name: str | None = None,
        env_vars: dict[str, str] | None = None,
        do_clone_and_checkout: bool = True
    ):
        """Initialize the RepoUVManager.

        Args:
            venv_dir_path (str | Path):
                Path where the UV virtual environment should be created.
            github_repo (GitHubRepo):
                Manages the cloned repository (commit checkout, patching, etc.).
            auto_detect_python (bool, optional):
                Whether to auto-detect a suitable Python version from project constraints.
            fallback_python_version (str, optional):
                Used if no constraints are found or no matching version is available in 'uv python list'.
            venv_name (str, optional):
                The name of the virtual environment to create in venv_dir_path.
                If None, generate a default with generate_venv_name().
            env_vars (dict[str, str] | None, optional):
                Additional environment variables to set in the shell.
            do_clone_and_checkout (bool, optional):
                Whether to clone and checkout the env setup commit during init.
        """
        self._logger = logging.getLogger(self.__class__.__name__)

        # Initialize the github repo (for python versions and other stuff)
        self.github_repo = github_repo
        self.repo_path = github_repo.root_repo_path
        self.clone_and_checkout_repo()
        
        self.venv_name = venv_name or self.generate_venv_name()
        self.venv_path = Path(venv_dir_path) / self.venv_name
        
        # Initialize python detector and get the selected python version
        self.python_version_detector = PythonVersionDetector()
        self.selected_python = self.python_version_detector.select_python_version(
            repo_path=self.repo_path,  
            python_fallback_version=fallback_python_version
        )

        super().__init__(venv_path=self.venv_path, python_version=self.selected_python.version, env_vars=env_vars)

    def generate_venv_name(self, instance_id: str | None = None, commit_hash: str | None = None) -> str:
        """Generates a unique environment name based on the repository, instance ID, and commit hash.

        Any arguments not provided are inferred from self.github_repo, where:
            - instance_id = self.github_repo.instance_id (if defined in GitHubRepo)
            - commit_hash = self.github_repo.env_setup_commit_hash

        Args:
            instance_id (str, optional): 
                The repository identifier string, e.g. 'repo__user-repo-issue'.
            commit_hash (str, optional): 
                The commit hash to reference.

        Returns:
            str: A unique environment name with a random UUID suffix.
        """
        instance_id = instance_id or getattr(self.github_repo, "instance_id", "repo")
        commit_hash = commit_hash or getattr(self.github_repo, "env_setup_commit_hash", "HEAD")

        # In case the repo includes slashes
        sanitized_instance = instance_id.replace("/", "_")
        return f"{sanitized_instance}-{commit_hash}-{uuid.uuid4().hex[:8]}"
    
    def clone_and_checkout_repo(self, commit_hash: str | None = None) -> Path:
        """Clone the GitHub repo (if not already) and check out the given commit hash.

        Args:
            commit_hash (str | None):
                The commit hash to check out. If None, uses the repo's default environment hash.

        Returns:
            Path: The local path to the cloned repository.
        """
        repo_path = self.github_repo.clone_and_checkout(checkout_commit_hash=commit_hash)
        self._logger.info(f"Repo cloned at {repo_path}")
        self.repo_path = repo_path  # This should already be set but we update just in case...
        return repo_path

    def install_repo_dependencies(self) -> CommandResult | None:
        """Install dependencies from the cloned repo by checking:
            (1) pyproject.toml build-system requirements
            (2) requirements.txt
            (3) pyproject.toml [project] table (install local project)
            (4) setup.py (install local project)
            (5) otherwise, do nothing

        We install in editable mode by default, so local code changes are reflected.

        Returns:
            CommandResult | None:
                The result of the pip install command, or None if no install was done.
        """
        if not self.env_ready:
            raise RuntimeError("Environment not ready. Call initialize() first.")

        if not self.repo_path:
            self._logger.warning("No repo is attached or cloned yet. Skipping dependency install.")
            return None

        result: CommandResult | None = None

        # (0) Upgrade pip and setup tools and whatnot
        self.pip_install("--upgrade pip", cwd=self.repo_path, verbose=True)
        self.pip_install("--upgrade setuptools wheel", cwd=self.repo_path, verbose=True)
        
        # (1) Build-System Requires: install them if present in pyproject.toml
        pyproj = self.repo_path / "pyproject.toml"
        if pyproj.exists():
            build_system_requires = self._get_build_system_requires(pyproj)
            if build_system_requires:
                self._logger.info(f"Installing build-system requirements: {build_system_requires}")
                build_cmd = " ".join(build_system_requires)
                build_result = self.pip_install(build_cmd, editable=True, cwd=self.repo_path, verbose=True)
                if not build_result.success:
                    self._logger.error("Failed to install build-system requirements!")
                    return build_result  # Early return if needed
            else:
                self._logger.debug("No build-system.requires found in pyproject.toml")

        # (2) requirements.txt and requirements-dev.txt
        req_file = self.repo_path / "requirements.txt"
        req_dev_file = self.repo_path / "requirements-dev.txt"
        if req_file.exists():
            self._logger.info("Installing dependencies from requirements.txt...")
            result = self.pip_install(["-r", str(req_file)], editable=True, cwd=self.repo_path)
            if req_dev_file.exists():
                self._logger.info("Installing dev dependencies from requirements-dev.txt...")
                dev_result = sself.pip_install(["-r", str(req_dev_file)], editable=True, cwd=self.repo_path)
                if not dev_result.success:
                    self._logger.error("Failed to install dev dependencies!")
                    return dev_result
            return result

        # (3) pyproject.toml [project] 
        if pyproj.exists():
            content = pyproj.read_text()
            if "[project]" in content:
                self._logger.info("Detected [project] table in pyproject.toml; installing with uv pip install . [editable]")
                result = self.pip_install(".", editable=True, cwd=self.repo_path)
                return result
            else:
                # Rename pyproject.toml -> pyproject.toml.bak (effectively 'hiding' it)
                pyproj_backup = pyproj.with_name("pyproject.toml.bak")
                pyproj.rename(pyproj_backup)
                self._logger.info("pyproject.toml found but no [project] table (hiding via rename). Will check setup.py next.")

        # (4) Fallback to setup.py
        setup_py = self.repo_path / "setup.py"
        if setup_py.exists():
            self._logger.info("Installing local code via setup.py with uv pip install . (editable)")
            result = self.pip_install(".", editable=True, cwd=self.repo_path)
            return result

        # (5) No recognized files
        self._logger.info(
            "No recognized dependency file found (requirements.txt, pyproject.toml, or setup.py). Skipping installation."
        )
        return result

    def _get_build_system_requires(self, pyproject_path: Path) -> list[str]:
        """Extract build-system dependencies from pyproject.toml."""
        try:
            import tomllib  # Python 3.11+; use 'tomli' for older versions
        except ImportError:
            import tomli as tomllib

        requirements = []
        toml_data = tomllib.loads(pyproject_path.read_text())

        build_system = toml_data.get("build-system", {})
        if not build_system:
            return requirements  # No build-system table found

        requires_list = build_system.get("requires", [])
        for item in requires_list:
            requirements.append(f'"{item}"')  # Quote items to avoid shell parsing issues

        # Optionally, infer extras based on build-backend
        build_backend = build_system.get("build-backend")
        if build_backend == "setuptools.build_meta" and not any("wheel" in r for r in requirements):
            requirements.append('"wheel"')
        elif build_backend == "hatchling.build" and not any("editables" in r for r in requirements):
            requirements.append('"editables"')

        return requirements

    def run_pytest(self, test_path: str | None = None, extra_args: list[str] | None = None) -> "CommandResult":
        """Convenience method to run pytest in the environment.

        Args:
            test_path (str | None):
                Specific test path or module to run (e.g. "tests/test_file.py::test_func").
                If None, runs all tests in the current repo_path.
            extra_args (list[str] | None):
                Additional command-line arguments (e.g. ["-v", "--pdb"]).

        Returns:
            CommandResult: Contains stdout, stderr, and return code.
        """
        if not self.env_ready:
            raise RuntimeError("Environment not ready. Call initialize() first.")
        if not self.repo_path:
            raise RuntimeError("No repo_path available; cannot run tests in an un-cloned repo.")

        cmd_tokens = ["pytest"]
        if test_path:
            cmd_tokens.append(test_path)
        if extra_args:
            cmd_tokens.extend(extra_args)

        # Build the final command string
        cmd_str = " ".join(cmd_tokens)
        self._logger.info(f"Running pytest: {cmd_str} (cwd={self.repo_path})")
        return self.send(cmd_str, cwd=self.repo_path)

    def apply_patch(self, patch_content: str) -> None:
        """Applies a patch to the cloned repository by delegating to GitHubRepo.

        Args:
            patch_content (str): The diff/patch content as a string.

        Raises:
            RuntimeError: If no GitHubRepo is set or patch application fails.
        """
        self._logger.info("Applying patch content via GitHubRepo...")
        self.github_repo.apply_patch(patch_content)

    def get_git_patch(self) -> str | None:
        """Generate a git patch from the current changes in the repo."""
        try:
            result = self.send("git diff", cwd=self.repo_path)
            if not result.success or not result.stdout.strip():
                return None
            
            # Check if there's at least one line starting with '+'
            lines = result.stdout.splitlines()
            if not any(line.startswith('+') for line in lines):
                return None
            
            # Otherwise return the stdout
            return result.stdout

        # Catch any errors and log and return None
        except Exception as e:
            self._logger.error(f"Failed to generate git patch: {e}")
            return None
    
    def remove_github_actions(self) -> None:
        """Remove the .github directory in the cloned repo, if it exists.
        
        This is done to prevent any unwanted GitHub Actions from interfering locally.
        """
        if not self.repo_path:
            self._logger.warning("No repo path is set, cannot remove .github folder.")
            return

        github_dir = self.repo_path / ".github"
        if github_dir.exists():
            import shutil
            self._logger.info(f"Removing GitHub Actions files at {github_dir}")
            shutil.rmtree(github_dir)
        else:
            self._logger.info("No .github directory found. Skipping removal.")

    def cleanup_repo(self) -> None:
        """Cleans up the cloned repository directory from disk (if desired).
        
        This is separate from environment cleanup, which ends the shell session.
        """
        if not self.repo_path:
            self._logger.warning("No repo path is set. Nothing to remove.")
            return

        import shutil
        if self.repo_path.exists():
            self._logger.info(f"Removing cloned repository at {self.repo_path}...")
            shutil.rmtree(self.repo_path, ignore_errors=True)
            self._logger.info("Repository folder removed.")

    def cleanup(self, remove_venv: bool = True) -> None:
        """Override cleanup() so we can remove the environment directory too, plus the repo.

        This method:
            - Terminates the persistent shell session from UVManager
            - Resets env_ready to False
            - Removes the cloned repo from disk
            - (Optionally, remove self.venv_path if you want the venv folder gone too)

        Args:
            remove_venv (bool): Whether to remove the environment directory
        """
        super().cleanup()  # Kills the shell, sets env_ready=False
        self.cleanup_repo()
        
        if remove_venv:
            if self.venv_path.exists():
                self._logger.info(f"Removing environment directory at {self.venv_path}...")
                shutil.rmtree(self.venv_path, ignore_errors=True)
                self._logger.info("Environment directory removed.")
            else:
                self._logger.info(f"No environment directory found at {self.venv_path}...")


def setup_demo_environment(
    instance: "SWEBenchInstance", 
    root_dir: str | Path = "/kaggle/tmp",
    fallback_python_version: str = "3.10"
) -> RepoUVManager:
    """
    Sets up a demo environment for a given SWE-Bench instance following the steps:
        (1) Generate a name for the environment (instance ID + commit hash).
        (2) Initialize the repository handler and clone the repo at environment_setup_commit.
        (3) Initialize the RepoUVManager with fallback_python_version (or a more advanced detection).
        (4) Create the virtual environment.
        (5) Install dependencies (requirements.txt, pyproject.toml, or setup.py).
        (6) Checkout the base_commit after installation.

    Args:
        instance (SWEBenchInstance):
            Contains information about the environment (repo, commits, etc.).
        root_dir (str | Path):
            The root directory where the repository should be cloned.
        fallback_python_version (str):
            The Python version to use if no advanced detection is done.

    Returns:
        RepoUVManager:
            - The specialized UV environment manager.
    """
    
    # ----------------------------------------------------------
    # (1) Create the GitHub repo object
    # ----------------------------------------------------------
    github_repo = GitHubRepo.from_swebench_instance(
        instance,
        root_dir=root_dir
    )

    # ----------------------------------------------------------
    # (2) Initialize a RepoUVManager with the new environment name
    #     and link to the cloned GitHubRepo
    # ----------------------------------------------------------
    # We store the environment in root_dir/env_name (or any path you like)
    repo_uv = RepoUVManager(
        venv_dir_path=root_dir,
        github_repo=github_repo,
        fallback_python_version=fallback_python_version
    )

    # ----------------------------------------------------------
    # (3) Clone the repo and checkout the correct commit for 
    #     env setup. Done internally now within __init__
    # ----------------------------------------------------------
    # repo_uv.clone_and_checkout_repo()

    # ----------------------------------------------------------
    # (4) Remove github actions
    # ----------------------------------------------------------
    repo_uv.remove_github_actions()
    
    # ----------------------------------------------------------
    # (5) Create the UV virtual environment
    # ----------------------------------------------------------
    repo_uv.initialize()

    # ----------------------------------------------------------
    # (6) Install dependencies, if any
    #     - This might look in requirements.txt or do 'uv pip install .'
    # ----------------------------------------------------------
    repo_uv.install_repo_dependencies()

    # ----------------------------------------------------------
    # (7) Checkout the base_commit if different from environment_setup_commit
    # ----------------------------------------------------------
    if instance.base_commit != instance.environment_setup_commit:
        repo_uv.github_repo.checkout_commit(instance.base_commit)

    # Return both objects so user can interact further
    return repo_uv

demo_repo_uv = setup_demo_environment(demo_instance, "/kaggle/working")
demo_repo_uv


# https://github.com/sympy/sympy/blob/a36caf5c74fe654cedc488e8a8a05fad388f8406/sympy/release.py
demo_repo_uv.send("uv run python -c 'import sympy; print(sympy.__version__)'")


rich.print(demo_instance.problem_statement)


def extract_updated_tests(diff_text: str) -> list[str]:
    """Extracts pytest-compatible test identifiers from a git diff.
    
    This function identifies modified test files and their respective test functions
    using regex, returning them in the pytest node ID format: `path/to/test_file.py::test_function_name`.
    If a test file is modified but no specific test functions are added/modified,
    returns just the file path to run all tests in that file.
    
    Args:
        diff_text (str): The git diff output containing file changes and modifications.
    
    Returns:
        list[str]: A list of pytest test identifiers in the format `file_path::test_function_name`
                   or just `file_path` for modified test files.
    """
    # Patterns for identifying test files and functions
    test_file_pattern = re.compile(r'^[+]{3} b/(.+?test.*?\.py)', re.MULTILINE)
    test_func_pattern = re.compile(r'^\+\s*def (test_[a-zA-Z0-9_]+)', re.MULTILINE)
    
    # Pattern for parameterized tests (they might have different signature)
    param_test_pattern = re.compile(
        r'^\+.*?@pytest\.mark\.parametrize.*?\n.*?\n*?^\+\s*def (test_[a-zA-Z0-9_]+)',
        re.MULTILINE | re.DOTALL
    )
    
    test_identifiers = set()
    
    # Split the diff into file chunks
    diff_files = diff_text.split('diff --git ')
    
    for diff_chunk in diff_files[1:]:  # Skip the first empty chunk
        # Extract the test file path
        file_matches = test_file_pattern.findall(diff_chunk)
        if not file_matches:
            continue
            
        file_path = file_matches[0]
        
        # Skip renamed files without content changes
        if 'similarity index 100%' in diff_chunk:
            continue
            
        # Extract all test functions (both regular and parameterized)
        test_funcs = set(test_func_pattern.findall(diff_chunk))
        param_funcs = set(param_test_pattern.findall(diff_chunk))
        
        # Combine all found test functions
        all_funcs = test_funcs.union(param_funcs)
        
        if all_funcs:
            # If we found specific test functions, add them with the file path
            for func_name in all_funcs:
                test_identifiers.add(f"{file_path}::{func_name}")
        else:
            # If the test file was modified but no specific test functions were found,
            # add the file path to run all tests in that file
            test_identifiers.add(file_path)
    
    return sorted(list(test_identifiers))

# hf_dfs["swe_bench_lite"]["dev"]["test_patch"][10]
test_identifiers = extract_updated_tests(demo_instance.test_patch)
all_updated_tests = extract_updated_tests(demo_instance.test_patch)
rich.print(f"Running the following updated test(s): {all_updated_tests}\n\n")


demo_repo_uv.apply_patch(demo_instance.test_patch)
with suppress_logging_below(logging.WARNING):
    for failing_test in all_updated_tests:
        rich.print("\n\n", Markdown("---"), "\n\n")
        rich.print(demo_repo_uv.run_pytest(failing_test).stdout)
        rich.print("\n\n", Markdown("---"), "\n\n")


demo_repo_uv.apply_patch(demo_instance.patch)
with suppress_logging_below(logging.WARNING):
    for test in all_updated_tests:
        rich.print("\n\n", Markdown("---"), "\n\n")
        rich.print(demo_repo_uv.run_pytest(test).stdout)
        rich.print("\n\n", Markdown("---"), "\n\n")


# class UVEnvironmentManager:
#     """Manages creation and usage of a UV environment, optionally as a context manager.

#     Attributes:
#         env_name (str): Unique name for the environment.
#         python_version (str): Python version to use (e.g., "3.10").
#         repo_path (Path): Path to the cloned repository on disk.
#         env_path (Path): Path to the UV environment directory.
#         _logger (logging.Logger): Internal logger instance.
#         python_versions (list[PythonVersion]): Python versions we have at our disposal
#         selected_python (PythonVersion): The version that will be used within this env.
#     """

#     env_name: str
#     repo_path: Path
#     env_path: Path
#     _logger: logging.Logger
#     python_versions: list[PythonVersion]
#     selected_python = PythonVersion

#     def __init__(self, env_name: str, repo_path: Path, fallback_python_version: str = "3.10") -> None:
#         """Initialize the UVEnvironmentManager.

#         Args:
#             env_name (str): Unique name for the environment.
#             repo_path (str | Path): Path to the cloned repository on disk.
#             fallback_python_version (str): Fallback Python version if no compatible version is found.
#         """
#         self.env_name = env_name
#         self.repo_path = Path(repo_path)
#         self.env_path = repo_path / ".venv"
#         self._logger = logging.getLogger(self.__class__.__name__)
#         self.fallback_python_version = fallback_python_version

        

#     def __enter__(self) -> "UVEnvironmentManager":
#         """Context manager entry: create environment.

#         Returns:
#             The current instance of UVEnvironmentManager.
#         """
#         self.create_environment()
#         return self

#     def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
#         """Context manager exit: cleanup environment.

#         Args:
#             exc_type (Any): Exception type (if any).
#             exc_val (Any): Exception value (if any).
#             exc_tb (Any): Exception traceback (if any).

#         Returns:
#             None; any exceptions are re-raised.
#         """
#         self.cleanup()

#     def __repr__(self) -> str:
#         """Official string representation, used for debugging."""
#         return (f"{self.__class__.__name__}(env_name={self.env_name!r}, "
#                 f"selected_python={self.selected_python!r}, "
#                 f"repo_path={str(self.repo_path)!r}, "
#                 f"env_path={str(self.env_path)!r})")

#     def __str__(self) -> str:
#         """Informal string representation, used for user-facing display."""
#         return str(self.__repr__())

    

#     def create_environment(self) -> None:
#         """Creates a UV virtual environment."""
#         self._logger.info(
#             f"Creating UV environment '{self.env_name}' "
#             f"with Python {self.selected_python.base_version}..."
#         )

#         try:
#             subprocess.run(
#                 ["uv", "venv", "--python", self.selected_python.base_version, str(self.env_path)],
#                 capture_output=True, text=True, check=True
#             )
#             self._logger.info(f"UV environment created successfully at {self.env_path}.")
#         except subprocess.CalledProcessError as e:
#             raise RuntimeError(f"Failed to create UV environment: {e.stderr}") from e

#         # (2) Remove Github Actions if Found
#         self.remove_github_actions()

#     def _uv_pip_install(
#         self,
#         package: str = ".",
#         try_editable_install: bool = True,
#         from_requirements: bool = False
#     ) -> None:
#         """Installs a package using pip within the UV environment.

#         Args:
#             package (str, optional):
#                 The path of the package to install (e.g., "numpy" or "requests").
#             try_editable_install (bool, optional):
#                 Whether to attempt an editable install (e.g., "pip install -e .").
#             from_requirements (bool, optional):
#                 Whether to install from a requirements file.
#         """
#         # (1) Run the command in the UV environment
#         self._logger.info(f"Running pip install within the UV env '{self.env_name}' for package: {package}")

#         # (2) Initialize install commands
#         _install_commands = ["uv", "pip", "install"]

#         # (3) Add the install from requirements file if needed
#         if from_requirements:
#             _install_commands += ["-r", "requirements.txt"]

#         # (4) Add the package to install (editable optional
#         _install_commands += ["-e", package] if try_editable_install else [package]

#         return self._execute_subprocess_run(_install_commands, cwd=self.repo_path)

#     def pip_install(
#             self,
#             package: str = ".",
#             try_editable_install: bool = True,
#             from_requirements: bool = False
#     ) -> CommandResult:
#         """Installs a package using pip within the UV environment.

#         Args:
#             package (str, optional):
#                 The path of the package to install (e.g., "numpy" or "requests").
#             try_editable_install (bool, optional):
#                 Whether to attempt an editable install (e.g., "uv pip install -e .").
#             from_requirements (bool, optional):
#                 Whether to install from a requirements file.

#         Returns:
#             None; raises on failure.
#         """
#         return self._uv_pip_install(package, try_editable_install, from_requirements)

#     def remove_github_actions(self) -> None:
#         """Removes GitHub Actions files from the repository.

#         This is useful when running tests locally to avoid conflicts with the CI/CD workflow.
#         """
#         self._logger.info("Removing GitHub Actions files...")
#         github_dir = self.repo_path / ".github"
#         if github_dir.exists():
#             shutil.rmtree(github_dir)
#             self._logger.info("GitHub Actions files removed.")
#         else:
#             self._logger.info("No GitHub Actions files found; skipping removal.")

#     def force_uv_to_install_from_setup_py(self, pyproject_file: Path):
#         """Forces UV to install dependencies from setup.py instead of pyproject.toml.

#         This is required if the pyproject.toml is invalid in someway as recognized by UV,
#         and we need to fallback to the legacy setup.py method.

#         Args:
#             pyproject_file (str): Path to the pyproject.toml file.
#         """

#         # (1) Backup the original pyproject.toml file
#         backup_path = pyproject_file.with_name("pyproject.toml.bak")

#         # (2) Rename pyproject.toml -> pyproject.toml.bak (effectively 'hiding' it)
#         pyproject_file.rename(backup_path)

#         # (3) Install dependencies with legacy setup.py
#         self.pip_install()

#         ## (4) Rename pyproject.toml.bak back to pyproject.toml ('unhiding' it)
#         # backup_path.rename(pyproject_file)

#     def install_dependencies(self) -> None:
#         """Installs project dependencies using UV within the environment."""
#         self._logger.info("Attempting to install dependencies...")
#         # (1) Define paths to possible install files
#         requirements_file = self.repo_path / "requirements.txt"
#         setup_file = self.repo_path / "setup.py"
#         pyproject_file = self.repo_path / "pyproject.toml"

#         try:
#             # (2a) If there's a requirements.txt, install that first.
#             if requirements_file.exists():
#                 self._logger.info("Installing from requirements.txt...")
#                 self.pip_install(from_requirements=True)

#             # (2b) If there's a pyproject.toml with a [project] table, do a normal 'pip install .'
#             elif pyproject_file.exists():
#                 content = pyproject_file.read_text()

#                 # (2bi) If all is well install from pyproject.toml
#                 if "[project]" in content:
#                     self._logger.info("Detected [project] table in pyproject.toml; installing with uv pip install .")
#                     self.pip_install()
#                 # (2bii) If all is NNOT well fallback to setup.py (with special hiding logic)
#                 elif setup_file.exists():
#                     self._logger.info("No [project] table found; using legacy setup.py install.")
#                     self.force_uv_to_install_from_setup_py(pyproject_file)
#                 # (2biii) If neither exists (nor a requirements) we giveup
#                 else:
#                     self._logger.info("No [project] table or setup.py found; skipping dependency install.")

#             # (2c) Otherwise, fallback to setup.py if it exists
#             elif setup_file.exists():
#                 self._logger.info("Installing via setup.py...")
#                 self.pip_install()

#             else:
#                 self._logger.info("No recognized dependency file found; skipping installation.")

#         except subprocess.CalledProcessError as e:
#             raise RuntimeError(f"Failed to install dependencies: {e.stderr}") from e

#     def _execute_subprocess_run(self, command: list[str], cwd: Path) -> CommandResult:
#         """Executes a command using subprocess.run.

#         Args:
#             command (list[str]): The command to execute as a list of strings.
#             cwd (str): The working directory to run the command from.

#         Returns:
#             CommandResult: The result of the command execution
#         """
#         # (1) Run the subprocess command
#         result = subprocess.run(
#             command,
#             cwd=cwd,
#             capture_output=True,
#             text=True,
#         )

#         # (2) We can choose to raise on non-zero or simply return the result
#         if result.returncode != 0:
#             self._logger.error(f"Command failed (exit code {result.returncode}):\n{result.stderr}")

#         # (3) Return the result
#         return CommandResult(command, result.returncode, result.stdout, result.stderr)

#     def execute_run_command(self, command: str, cwd: Path | str | None = None) -> CommandResult:
#         """Executes a command inside the UV environment.

#         Args:
#             command (str):
#                 The shell command to execute (e.g., "pytest" or "python script.py").
#             cwd (Path, optional):
#                 Optional directory to run the command from. Defaults to the repo path.

#         Returns:
#             A tuple of (returncode, stdout, stderr).
#                 - returncode (int): The exit code of the command.
#                 - stdout (str): The standard output of the command.
#                 - stderr (str): The standard error of the command.

#         Raises:
#             subprocess.CalledProcessError: If the command fails (non-zero return code).
#         """
#         # (1) Set the working directory if provided or default to the repo path
#         cwd = Path(cwd or self.repo_path)

#         # (2) Run the command in the UV environment
#         self._logger.info(f"Running command in UV env '{self.env_name}' (cwd={cwd}): {command}")
#         return self._execute_subprocess_run(["uv", "run", "sh", "-c", command], cwd)


#     def run_pytest(self, test_path: str | Path, extra_args: list[str] | None = None) -> CommandResult:
#         """Runs a specific test (or tests) using pytest inside the UV environment.

#         Args:
#             test_path (str | Path):
#                 The relative path (or module::test_func) to run.
#                     - e.g. "tests/test_something.py::test_something"
#             extra_args (list[str], optional):
#                 Optional extra flags or arguments for pytest
#                     - e.g. ["-v", "--pdb"]

#         Returns:
#             A CommandResult with stdout, stderr, and returncode.
#         """
#         # (0) Parse extra-args if provided otherwise default to empty list.
#         if not extra_args:
#             extra_args = []

#         # (1) Get test path
#         if self.repo_path not in Path(test_path).parents:
#             test_path = str(self.repo_path / test_path)
#         else:
#             test_path = str(test_path)

#         # (2) Build the command, e.g. "pytest test/cli/commands_test.py::test__cli__command_directed -v"
#         cmd_tokens = ["pytest", test_path, *extra_args]

#         # (3) Join into a single shell command
#         cmd_str = " ".join(cmd_tokens)

#         # (4) Run the command and return
#         return self.execute_run_command(cmd_str)

#     def cleanup(self) -> None:
#         """Deletes the UV environment directory from the filesystem."""
#         # (1) Remove the environment directory and its contents
#         self._logger.info(f"Cleaning up environment at {self.env_path}...")
#         if self.env_path.exists():
#             shutil.rmtree(self.env_path)
#             self._logger.info(f"Environment directory removed: {self.env_path}")


def setup_demo_environment(
    instance: SWEBenchInstance, 
    root_dir: str = "/kaggle/tmp",
    fallback_python_version: str = "3.10"
) -> None:
    """Sets up a demo environment for a given SweBench instance.
    
    This setup involves:
        - creating a virtual environment
        - cloning the repository
        - installing dependencies
        - checkout out the correct issue respective commit 
    
    Args:
        instance (SweBenchInstance): The instance containing information about the environment and repository.
        root_dir (str, optional): The root directory where the repository should be cloned.
        fallback_python_version (str, optional): 
            The Python version to use as a fallback for the virtual environment.
            Only applies if the python version cannot be automatically detected and used.
    
    Returns:
        None
    """
    # Step 1: Generate a name for the environment based on instance ID and commit hash.
    _env_name = generate_env_name(
        instance_id=instance.instance_id, 
        commit_hash=instance.environment_setup_commit
    )
    
    # Step 2: Initialize the repository handler and clone the repository.
    _repo_handler = GitHubRepo.from_swebench_instance(demo_instance, root_dir=root_dir)
    _repo_path = _repo_handler.clone_and_checkout()
    
    # Step 3: Initialize the environment manager with the specified Python version.
    _env_manager = UVEnvironmentManager(_env_name, _repo_path, fallback_python_version=fallback_python_version)
    
    # Step 4: Create the virtual environment.
    _env_manager.create_environment()
    
    # Step 5: Install dependencies if a requirements file, setup.py, or pyproject.toml is found.
    _env_manager.install_dependencies()
    
    # Step 6: Checkout the repository at the correct commit after installation.
    _repo_handler.checkout_commit(_repo_handler.base_commit_hash)

    return _env_manager, _repo_handler

demo_env_manager, demo_repo_handler = setup_demo_environment(instance=demo_instance, root_dir="/kaggle/working")


try:
    import pydicom
except ImportError as e:
    rich.print(f"\n[bold red]EXCEPTION FROM KAGGLE: {e}[/bold red]\n\n[bold]... We cannot import pydicom in the kaggle environment but we SHOULD be able to within the UV environment ... [/bold]\n\n")
    rich.print(demo_env_manager.execute_command("python -c 'import pydicom; print(\"\\n\\nThe version of pydicom within the UV environment is:\", pydicom.__version__)'").stdout)
    rich.print("\n\n[bold cyan]For confirmation here is the link to the version file for this commit on Github:[/bold cyan]")
    display("https://github.com/pydicom/pydicom/blob/7241f5d9db0de589b230bb84212fbb643a7c86c3/pydicom/_version.py#L4")


demo_env_manager.run_pytest(str(all_updated_tests[0]), extra_args=["-vv",])


rich.print("\n\n[bold red]... THIS WILL FAIL BECAUSE WE HAVE NOT PATCHED THE TESTS YET ...[/bold red]")
for test_to_show_fail in all_updated_tests:    
    rich.print(demo_env_manager.run_pytest(test_to_show_fail))
display(Markdown("<br><br>---"))
rich.print("\n\n\n\n[bold green]... THESE SHOULD WORK BECAUSE WE HAVE NOW APPLIED THE TEST PATCHES ...[/bold green]")
demo_repo_handler.apply_patch(demo_instance.test_patch)
for test_to_show_fail in all_updated_tests:    
    rich.print(demo_env_manager.run_pytest(test_to_show_fail))


rich.print(demo_env_manager.pip_install("setuptools", False))
demo_env_manager.execute_command("python setup.py install")


from pathlib import Path

# Create the test script
test_script = """
import sqlfluff

def test_sql(sql: str, dialect: str = "tsql") -> None:
    print(f"\\nTesting SQL:\\n{sql}\\n")
    linted = sqlfluff.lint(sql, dialect=dialect)
    if linted:
        print("Violations found:")
        for violation in linted:
            print(f"Line {violation.line_no}: {violation.code} - {violation.description}")
            print(f"  Context: {violation.line_pos}")
    else:
        print("No violations found.")

# Test cases
no_alias_sql = '''
SELECT [hello]
FROM
    mytable
'''

with_alias_sql = '''
SELECT a.[hello]
FROM
    mytable AS a
'''

print("=== Testing SQL without alias ===")
test_sql(no_alias_sql)

print("\\n=== Testing SQL with alias ===")
test_sql(with_alias_sql)
"""

# Write the script to a file
script_path = demo_env_manager.repo_path / "test_sqlfluff.py"
script_path.write_text(test_script)

# Now we can return the path to be executed
print(str(script_path))
demo_env_manager.execute_command(f"python {str(script_path)}")


def run_swe_bench_test(
        instance: SWEBenchInstance,
        python_version: str = "3.10",
        remove_repo: bool = True
) -> None:
    """Orchestrates the full flow: clone repo, create UV env, apply patches, install deps, run tests.

    Args:
        instance (SWEBenchInstance):
            A SWEBenchInstance containing all relevant data.
        python_version (str):
            The Python version to use for UV (e.g. '3.10').
        remove_repo (bool):
            Whether to remove the cloned repository folder after the test.
    """
    # (1) Construct GitHub URL - adapt if your 'repo' field is already a full URL
    repo_url = instance.repo
    if "github.com" not in repo_url:
        repo_url = f"https://github.com/{instance.repo}.git"

    # (2) Determine environment name
    env_name = generate_env_name(instance.instance_id, instance.environment_setup_commit)

    # (3) Clone the repository at environment_setup_commit (or base_commit if desired)
    repo_handler = GitHubRepo(repo_url, commit_hash=instance.base_commit)
    repo_path = repo_handler.clone_and_checkout()

    # (4) Apply patches if present
    if instance.test_patch:
        repo_handler.apply_patch(instance.test_patch)
    if instance.patch:
        repo_handler.apply_patch(instance.patch)

    # (5) Create UV environment and install dependencies
    with UVEnvironmentManager(env_name, repo_path, python_version=python_version) as uv_env:
        uv_env.install_dependencies()

        # (6) Run tests (example: pytest)
        try:
            returncode, stdout, stderr = uv_env.execute_command("pytest")
            logger.info(
                "Tests completed successfully. "
                f"Return code: {returncode}\nOutput:\n{stdout}\nErrors:\n{stderr}"
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Test command failed: {e.stderr}")

    # (7)(Optional)
    #   - Additional cleanup beyond context manager if needed
    #   - For example:
    #       --> if you want to remove the entire cloned repository folder:
    if remove_repo and repo_path.exists():
        shutil.rmtree(repo_path)
        logger.info(f"Repository folder removed: {repo_path}")


run_swe_bench_test(demo_instance)




