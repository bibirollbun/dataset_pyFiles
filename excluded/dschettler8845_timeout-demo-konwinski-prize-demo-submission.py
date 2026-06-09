import io
import os
import shutil
import subprocess

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


import multiprocessing
from typing import Any, Callable
import time
import os
import signal
from contextlib import contextmanager

class TimeoutError(Exception):
    pass

@contextmanager
def time_limit(seconds):
    """Context manager that raises TimeoutError if the code inside takes longer than specified seconds."""
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")
    
    # Set the signal handler and a alarm
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        # Cancel the alarm
        signal.alarm(0)

def run_with_timeout(
    func: Callable[..., Any], 
    timeout: float | int, 
    *args: Any, 
    max_poll_interval: float = 1.0,  # New parameter to control maximum polling interval
    **kwargs: Any
) -> Any | None:
    """Run a function in a separate process with a guaranteed timeout.
    
    This uses a modified multiprocessing approach that is container-friendly and 
    reliably terminates the function after the timeout period.
    
    Args:
        func (Callable[..., Any]): The target function to execute.
        timeout (float | int): Maximum allowed execution time in seconds.
        *args (Any): Positional arguments to pass to `func`.
        max_poll_interval (float): Maximum time between polling checks (default: 1.0 seconds).
        **kwargs (Any): Keyword arguments to pass to `func`.
    
    Returns:
        Any | None: 
            The result of `func(*args, **kwargs)` if it completes within `timeout` seconds; 
            Otherwise, None.
    """
    t1 = time.time()
    # Create a pipe for communication
    parent_conn, child_conn = multiprocessing.Pipe()
    
    # Use start method that works well in containers
    ctx = multiprocessing.get_context('spawn')
    p = ctx.Process(
        target=_process_worker,
        args=(child_conn, func, args, kwargs)
    )
    
    # Start process
    p.start()
    
    # Set a timeout for receiving data
    start_time = time.time()
    result = None
    received = False
    
    # Start with a short polling interval that gradually increases
    poll_interval = 0.1  # Initial polling interval
    sleep_time = 0.01    # Initial sleep time
    
    while time.time() - start_time < timeout and not received:
        if parent_conn.poll(poll_interval):  # Check with adaptive timeout
            result = parent_conn.recv()
            received = True
        
        # Gradually increase polling interval for efficiency, up to max_poll_interval
        poll_interval = min(poll_interval * 1.5, max_poll_interval)
        
        # Also gradually increase sleep time, up to 1/10th of poll_interval
        sleep_time = min(sleep_time * 1.5, poll_interval / 10)
        
        time.sleep(sleep_time)  # Adaptive sleep to prevent CPU spinning
    
    # Make sure to terminate the process if running
    if p.is_alive():
        p.terminate()
        p.join(1.0)  # Give it a second to clean up
        # Force kill if still alive
        if p.is_alive():
            os.kill(p.pid, signal.SIGKILL)
    
    print(f"TERMINATING USING MANUAL TIMEOUT AFTER: {time.time()-t1:.2f} SECONDS")
    return result

def _process_worker(conn, func, args, kwargs):
    """Worker function that runs in the separate process."""
    try:
        # Additional safety: use signal alarm as a backup timeout mechanism
        with time_limit(60*29):  # 29 minute hard limit as an extra safeguard
            result = func(*args, **kwargs)
            conn.send(result)
    except Exception as e:
        # If any exception occurs, just return None
        pass
    finally:
        conn.close()


import time
first_prediction = True

ACTUAL_TIMEOUT_ERROR_LIMIT = 60*30  # 30 minutes
OUR_TIMEOUT_LIMIT = 60*1  # 1 minute timeout

def inner_predict(random_arg_1: Any, random_arg_2: int = 2):
    """Included some args just for demonstration purposes"""
    print(random_arg_1)
    time.sleep(ACTUAL_TIMEOUT_ERROR_LIMIT*2)  # This function will run for 60 minutes if uninterrupted.
    print(random_arg_2)
    return "**yawn** ... I slept so long ..."
    
def predict(problem_statement: str, repo_archive: io.BytesIO, pip_packages_archive: io.BytesIO, env_setup_cmds_templates: list[str]) -> str:
    """ Replace this function with your inference code.
    Args:
        problem_statement: The text of the git issue.
        repo_path: A BytesIO buffer path with a .tar containing the codebase that must be patched. The gateway will make this directory available immediately before this function runs.
        pip_packages_archive: A BytesIO buffer path with a .tar containing the wheel files necessary for running unit tests.
        env_setup_cmds_templates: Commands necessary for installing the pip_packages_archive.
    """
    global first_prediction
    if not first_prediction:
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
    pip_packages_path = '/path/to/pip_packages'
    if os.path.exists(pip_packages_path):
        shutil.rmtree(pip_packages_path)
    shutil.unpack_archive(pip_archive_dir, extract_dir=pip_packages_path)
    os.remove(pip_archive_dir)

    # Get env setup cmds by setting the pip_packages_path
    env_setup_cmds = [cmd.format(pip_packages_path=pip_packages_path) for cmd in env_setup_cmds_templates]

    # Run env setup for the repo
    subprocess.run(
        "\n".join(env_setup_cmds),
        shell=True,
        executable="/bin/bash",
        cwd=repo_path,
    )

    first_prediction = False
    # Instead of a valid diff, let's just submit a generic string. This will definitely fail.
    prediction_result = run_with_timeout(inner_predict, timeout=OUR_TIMEOUT_LIMIT, random_arg_1="hello", random_arg_2=7)
    return prediction_result


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




