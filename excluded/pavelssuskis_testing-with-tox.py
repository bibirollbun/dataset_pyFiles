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


import io
import os
import shutil
import subprocess
import time

start_time = time.time()

import pandas as pd
import polars as pl

import kaggle_evaluation.konwinski_prize_inference_server




import subprocess
import os

# Attempt to simulate pre-load which seems failed



# Run the pip download command and capture output
result = subprocess.run(
    'pip download tox pygit py --dest extra_wheels',
    shell=True,
    executable="/bin/bash",
    capture_output=True,  # Capture stdout and stderr
    text=True
)

# Print stdout and stderr for debugging
print("STDOUT:\n", result.stdout)
print("STDERR:\n", result.stderr)

# Check if the directory exists and list files
if os.path.exists('extra_wheels'):
    files = os.listdir('extra_wheels/')
    print("Contents of the folder:")
    for file in files:
        print(file)
else:
    print("The directory 'extra_wheels' does not exist.")




instance_count = None

def get_number_of_instances(num_instances: int) -> None:
    """ The very first message from the gateway will be the total number of instances to be served.
    You don't need to edit this function.
    """
    global instance_count
    instance_count = num_instances


# a little sugar for installing wheels from pip_packages

import glob
import os
import subprocess

def install_wheels(wheel_dir):
    # Get a list of all .whl files in the directory
    wheel_files = glob.glob(os.path.join(wheel_dir, "*.whl"))

    if wheel_files:
        for wheel_path in wheel_files:
            print(f"Installing: {wheel_path}")

            # Run both commands in a single call using '&&'
            command = f"source .venv/bin/activate && uv pip install {wheel_path}"
            
            result = subprocess.run(
                command,
                shell=True,
                executable="/bin/bash",
                cwd='repo',
                capture_output=True,
                text=True
            )

            # Print the output of the command
            print(result.stdout)
            if result.stderr:
                #print(result.stderr)
                print(f'Error on wheel{wheel_path}')

    else:
        print(f"No .whl files found in {wheel_dir}")

def run_command(command, cwd=None, env=None):
    """Helper function to run shell commands with subprocess."""
    esult = subprocess.run(command, cwd=cwd, env=env, shell=True, text=True, capture_output=True, executable="/bin/bash")
    if result.returncode != 0:
        print(f"Error: {command}\n{result.stderr}")
    else:
        print(result.stdout)
    return result.returncode == 0
    





no_op_patch = """
--- /dev/null
+++ b/docs/changes/table/17048.bugfix.rst
@@ -0,0 +1,2 @@
+Ensure that initializing a ``QTable`` with explicit units` also succeeds if
+one of the units is ``u.one``.
"""

first_prediction = True
number_of_run = 0

import re
import os
import git

def predict(problem_statement: str, repo_archive: io.BytesIO, pip_packages_archive: io.BytesIO, env_setup_cmds_templates: list[str]) -> str:
    """ Replace this function with your inference code.
    Args:
        problem_statement: The text of the git issue.
        repo_path: A BytesIO buffer path with a .tar containing the codebase that must be patched. The gateway will make this directory available immediately before this function runs.
        pip_packages_archive: A BytesIO buffer path with a .tar containing the wheel files necessary for running unit tests.
        env_setup_cmds_templates: Commands necessary for installing the pip_packages_archive.
    """
        #print(problem_statement)
    global number_of_run
    number_of_run += 1
    print(f'Running predict {number_of_run}')
    # !!!!!!!!!!!!!!!!
    # control of runs to focus on speified repo
    # Example if number_of_run !=3 disables everything, but 3rd run
    if number_of_run ==7:
        return None  # Skip issue.
        
    # Unpack the codebase to be patched into a directory that won't be exported when
    # the notebook is saved.
    archive_path = '/tmp/repo_archive.tar'
    with open(archive_path, 'wb') as f:
        f.write(repo_archive.read())
    repo_path = 'repo'
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)
    shutil.unpack_archive(archive_path, extract_dir=repo_path)
    os.remove(archive_path)

    """
    Unpack pip_packages if you want to run unit tests on your patch.
    Note that editing unit tests with your patch -- even to add valid tests -- can cause your submission to be flagged as a failure.
    Most of the relevant repos use pytest for running tests. You will almost certainly need to run only a subset of the unit tests to avoid running out of inference time.
    """
    pip_archive_dir = '/tmp/pip_packages_archive.tar'
    with open(pip_archive_dir, 'wb') as f:
        f.write(pip_packages_archive.read())
    pip_packages_path = 'pip_packages'
    if os.path.exists(pip_packages_path):
        shutil.rmtree(pip_packages_path)
    shutil.unpack_archive(pip_archive_dir, extract_dir=pip_packages_path)
    os.remove(pip_archive_dir)

    # Get env setup cmds by setting the pip_packages_path
    env_setup_cmds = [cmd.format(pip_packages_path=pip_packages_path) for cmd in env_setup_cmds_templates]

  
    subprocess.run(
        "\n".join(env_setup_cmds),
        shell=True,
        executable="/bin/bash",
        cwd=repo_path,
    )


    files = os.listdir(repo_path)
        
    print("Contents of the folder:")
    for file in files:
        print(file)
    if os.path.exists('repo/tox.ini'): # and number_of_run >2:
        print('Testing with tox')
        git_path = os.path.join(repo_path, ".git")
        if not os.path.exists(git_path):
            print('No .GIT ')
            # Initialize Git repository
            repo = git.Repo.init(repo_path)
            print(f"Initialized Git repository in {repo_path}")
            
            # Configure Git user (only if not globally configured)
            with repo.config_writer() as config:
                config.set_value("user", "name", "Your Name")
                config.set_value("user", "email", "your_email@example.com")
            
            # Add all files and commit
            repo.git.add(all=True)
            repo.index.commit("Fake commit")
            print("Committed changes.")
            
            # Create an annotated tag
            repo.create_tag("v1.0.0", message="Set version tag")
            print("Tag v1.0.0 created.")
    
            # print("Installing package using setuptools_scm.")
            
            # Copy environment variables
            env_vars = os.environ.copy()
            env_vars["SETUPTOOLS_SCM_PRETEND_VERSION"] = "1.0.0"
            # Define the old and new project names
            # Define old and new project names
            old_name = "astropy"
            new_name = "my_project"
            
            
            # Rename in pyproject.toml (if it exists)
            pyproject_toml_path = os.path.join(repo_path, "pyproject.toml")
            if os.path.exists(pyproject_toml_path):
                subprocess.run(
                    f"sed -i 's/^name = \"{old_name}\"/name = \"{new_name}\"/' pyproject.toml",
                    shell=True,
                    check=True,
                    executable="/bin/bash",
                    cwd=repo_path,
                )


            # run_command(f"source .venv/bin/activate && uv pip install --no-index --find-links={pip_packages_path} -e {repo_path}", env=env_vars)

            # # Build C extensions
            # run_command(f"source .venv/bin/activate && uv {repo_path}/setup.py build_ext --inplace")
            # # Run setuptools_scm to check version
            # env_vars["GIT_DIR"] = os.path.join(repo_path, ".git")
            # env_vars["GIT_WORK_TREE"] = repo_path
            # run_command(f"source .venv/bin/activate && uv run setuptools_scm", env=env_vars)                  
            run_command(f"source .venv/bin/activate && uv pip install py tox pygit", env=env_vars)
            # run_command(f"source .venv/bin/activate && uv pip install -e astropy[dev-all]", env=env_vars)

            extra_packages_path = 'extra_wheels' 
            install_wheels(extra_packages_path)
            #install_wheels(pip_packages_path)

            #run_command(f"source .venv/bin/activate && uv pip install --no-index --find-links=pip_packages && pip install -e .[test]")
            run_command(f"source .venv/bin/activate && uv pip install --no-index --find-links=pip_packages && pip install . ")
            #run_command(f"source .venv/bin/activate && uv pip uninstall asdf-astropy astropy")


            # ğŸš€ Run Tox Tests with Virtual Environment Activation
            print("\nğŸš€ Running tox tests...")



            
            
            # Verify the change
            subprocess.run("grep 'name=' setup.py", shell=True)

            tox_ini_path = os.path.join(repo_path, "tox.ini")
            #tox_command = f'source .venv/bin/activate && SETUPTOOLS_SCM_PRETEND_VERSION="1.2.3" uv run tox -c {tox_ini_path} -e py311 --parallel auto -vv'
            
            tox_command = f'source .venv/bin/activate && uv pip install tox '

            result = subprocess.run(
                tox_command,
                shell=True,
                executable="/bin/bash",
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            print("STDOUT:\n", result.stdout)
            print("STDERR:\n", result.stderr)

            if os.path.exists(tox_ini_path):
                print("tox.ini exists!")
            tox_command = f'source .venv/bin/activate && uv run tox -c tox.ini -e py311 --parallel auto -vv'

            result = subprocess.run(
                tox_command,
                shell=True,
                executable="/bin/bash",
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            
            print("STDOUT:\n", result.stdout)
            print("STDERR:\n", result.stderr)

        
    else:
        print('tox.ini does not exist or ignored')
        # Run pytest with -rxX to show xfailed and xpassed details
        install_wheels(pip_packages_path)
        result = subprocess.run(
            "source .venv/bin/activate && pytest",
            shell=True,
            executable="/bin/bash",
            cwd='repo',
            capture_output=True,
            text=True
        )
        

      
        # Combine stdout and stderr
        output = result.stdout + result.stderr  
        
        # Save raw output for debugging
        timestamp = str(int(time.time() - start_time)).zfill(5)
        
        with open(f"{timestamp}-out.txt", "w") as f:
            f.write(result.stdout)
        
        with open(f"{timestamp}-err.txt", "w") as f:
            f.write(result.stderr)
        
        # Print full output
        print(len(result.stdout))
        print("\n".join(result.stdout.split("\n")))
        
        print(len(result.stderr))
        print("\n".join(result.stderr.split("\n")))
        
        # ğŸ”� **Extract failed tests using regex**
        failed_tests = re.findall(r"FAILED (tests/[\w/]+\.py::[\w\[\]]+)", output)
        
        # Print and save failed tests
        if failed_tests:
            print("\nâ�Œ Failed Tests List:")
            for test in failed_tests:
                print(test)
            with open(f"{timestamp}-failed-tests.txt", "w") as f:
                f.write("\n".join(failed_tests))
        else:
            print("\nâœ… No failed tests!")


    
    # Cleanup
    
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)
    if os.path.exists(pip_packages_path):
        shutil.rmtree(pip_packages_path)

    return no_op_patch


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
        ),
        use_concurrency=True,  # This can safely be disabled for purposes of local testing if necessary.
    )

