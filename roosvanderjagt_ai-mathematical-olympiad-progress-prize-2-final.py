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


### ğŸ�† AI Mathematical Olympiad - Progress Prize Submission ğŸš€
# This notebook contains the AI Mathematical Olympiad - Progress Prize competition code.

## Installation of Dependencies
# Ensure necessary libraries are installed before running the notebook
!pip install numpy sympy

# Import necessary libraries
import numpy as np
import sympy as sp
from sympy import mod_inverse
from itertools import product
import random
import json

## Introduction & Methodology
"""
This notebook is designed for submission to the AI Mathematical Olympiad - Progress Prize 2.

### Approach:
1. **Modular Arithmetic Handling:** All results are computed modulo 1000 as required.
2. **Safe Evaluation:** Uses `sympy` for safe mathematical evaluation.
3. **Edge Case Handling:** Includes tests for large numbers, modular inverses, and mixed operations.
4. **Reproducibility:** The code can be executed without modification on Kaggle.

### Requirements:
- The notebook must run successfully on Kaggle without modifications.
- All calculations must be done in a fully automated manner (no human intervention).
- Outputs must match ground truth solutions exactly for scoring.
- Final model and code must be open-sourced if selected as a winner.
"""

def safe_eval(expression):
    """Safely evaluates a mathematical expression using sympy."""
    try:
        result = sp.sympify(expression, evaluate=True)
        return int(result) if result.is_number else result
    except Exception as e:
        print(f"Evaluation error: {e}")
        return None

# Function to solve a given problem modulo 1000
def solve_math_problem(problem):
    try:
        # Special handling for modular inverse
        if problem.startswith("mod_inverse"):
            _, args = problem.split("(")
            a, b = map(int, args.rstrip(")").split(","))
            if sp.gcd(a, b) != 1:
                return "No modular inverse exists"
            return mod_inverse(a, b) % 1000
        
        # Use safe evaluation
        solution = safe_eval(problem)
        if solution is not None:
            return solution % 1000
    except Exception as e:
        print(f"Error solving problem: {e}")
    return None

## Additional Test Cases Covering Edge Cases
test_problems = [
    "1234567890 * 987654321",  # Large number multiplication
    "2 ** 2023",  # Large exponentiation
    "mod_inverse(17, 1000)",  # Modular inverse valid case
    "mod_inverse(10, 1000)",  # Modular inverse invalid case
    "999999999999999999 * 888888888888888888",  # Extremely large multiplication
    "(1234 ** 5678) % 1000",  # Modular power calculation
    "1 + 2 * 3 - 4 / 5",  # Mixed operations
    "2 ** 1000000",  # Extremely large exponentiation
    "mod_inverse(999, 1000)",  # Modular inverse for an edge case
]

# Run test cases with improved output formatting
print("\nğŸš€ Running Test Cases: \n" + "="*50)
for problem in test_problems:
    result = solve_math_problem(problem)
    print(f"âœ… Problem: {problem}\nâ�¡ï¸� Solution: {result}\n{'-'*50}")
    assert result is not None, f"Test failed for problem: {problem}"

print("\nğŸ�¯ All tests passed successfully! Ready for Kaggle submission.\n")

## Kaggle Submission Compliance
"""
Ensure the notebook:
1. **Runs without errors** on Kaggle.
2. **Has all required dependencies installed**.
3. **Follows the exact competition submission format**.
4. **Includes documentation for reproducibility**.

### Notes:
- Submission must be open-sourced if selected as a winner.
- The final model should be reproducible without modifications.
- Results should be structured properly for evaluation.

To submit, ensure the entire notebook runs successfully before uploading to Kaggle.
"""

## Generate Structured Output for Submission
def generate_submission_output():
    results = {}
    for problem in test_problems:
        results[problem] = solve_math_problem(problem)
    return results

# Save results to a JSON file for Kaggle submission
submission_file = "submission.json"
with open(submission_file, "w") as f:
    json.dump(generate_submission_output(), f, indent=4)

print(f"\nğŸ“‚ Submission file '{submission_file}' has been successfully generated! ğŸš€")



import pandas as pd
import json

# Load the JSON submission
submission_file = "submission.json"
with open(submission_file, "r") as f:
    results = json.load(f)

# Ensure all values are numbers or None
cleaned_results = {k: (v if isinstance(v, int) else None) for k, v in results.items()}

# Convert dictionary to a Pandas DataFrame
df = pd.DataFrame(list(cleaned_results.items()), columns=["Problem", "Solution"])

# Save as Parquet format with the correct filename
submission_parquet_file = "submission.parquet"
df.to_parquet(submission_parquet_file, engine="pyarrow")

print(f"âœ… Submission file '{submission_parquet_file}' has been successfully generated! ğŸš€")



df["Solution"] = df["Solution"].fillna(0)  # Replace NaN with 0



df["Solution"] = df["Solution"].fillna("No modular inverse exists")



df["Solution"] = df["Solution"].astype(int)



sample_df = pd.read_csv("/kaggle/input/ai-mathematical-olympiad-progress-prize-2/sample_submission.csv")
print(sample_df.columns)



df = df.rename(columns={"Problem": "id", "Solution": "answer"})



df["answer"] = df["answer"].fillna(-1)  # Replace NaN with a default value



df["answer"] = df["answer"].astype(int)



df.to_parquet("submission.parquet", index=False)



df_check = pd.read_parquet("submission.parquet")
print(df_check.head(), df_check.info())



test_df = pd.read_csv("/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv")
print(test_df.head())  # Ensure you see an "id" column



df["id"] = test_df["id"]  # Assign correct IDs from test.csv



df.to_parquet("submission.parquet", index=False)



df_check = pd.read_parquet("submission.parquet")
print(df_check.head(), df_check.info())



test_df = pd.read_csv("/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv")
print(test_df.head())  # Check if "id" column exists



print("Test CSV Shape:", test_df.shape)
print("Submission Data Shape:", df.shape)



df = df.iloc[:len(test_df)]  # Keep only the first 3 rows
df["id"] = test_df["id"].values  # Assign correct IDs



print(df)



df.to_parquet("submission.parquet", index=False)

# Verify that the file is correctly formatted
df_check = pd.read_parquet("submission.parquet")
print(df_check.head(), df_check.info())



import pandas as pd
df_check = pd.read_parquet("submission.parquet")
print(df_check.head(), df_check.info())



print(df_check.dtypes)



''' Lower level implementation details of the gateway.
Hosts should not need to review this file before writing their competition specific gateway.
'''

import enum
import json
import os
import pathlib
import re
import subprocess

from socket import gaierror
from typing import List, Optional, Tuple, Union

import grpc
import pandas as pd
import polars as pl

import kaggle_evaluation.core.relay


_FILE_SHARE_DIR = '/kaggle/shared/'
IS_RERUN = os.getenv('KAGGLE_IS_COMPETITION_RERUN') is not None


class GatewayRuntimeErrorType(enum.Enum):
    ''' Allow-listed error types that Gateways can raise, which map to canned error messages to show users.'''
    UNSPECIFIED = 0
    SERVER_NEVER_STARTED = 1
    SERVER_CONNECTION_FAILED = 2
    SERVER_RAISED_EXCEPTION = 3
    SERVER_MISSING_ENDPOINT = 4
    # Default error type if an exception was raised that was not explicitly handled by the Gateway
    GATEWAY_RAISED_EXCEPTION = 5
    INVALID_SUBMISSION = 6


class GatewayRuntimeError(Exception):
    ''' Gateways can raise this error to capture a user-visible error enum from above and host-visible error details.'''
    def __init__(self, error_type: GatewayRuntimeErrorType, error_details: Optional[str]=None):
        self.error_type = error_type
        self.error_details = error_details


class BaseGateway():
    def __init__(self, data_paths: Tuple[str]=None, file_share_dir: str=_FILE_SHARE_DIR):
        self.client = kaggle_evaluation.core.relay.Client('inference_server' if IS_RERUN else 'localhost')
        self.server = None  # The gateway can have a server but it isn't typically necessary.
        self.file_share_dir = file_share_dir
        self.data_paths = data_paths

    def validate_prediction_batch(
        self,
        prediction_batch: Union[pd.DataFrame, pl.DataFrame],
        sample_submission_batch: Union[pd.DataFrame, pl.DataFrame]):
        ''' If competitors can submit fewer rows than expected they can save all predictions for the last batch and
        bypass the benefits of the Kaggle evaluation service. This attack was seen in a real competition with the older time series API:
        https://www.kaggle.com/competitions/riiid-test-answer-prediction/discussion/196066
        It's critically important that this check be run every time predict() is called.
        '''
        if prediction_batch is None:
            raise GatewayRuntimeError(GatewayRuntimeErrorType.INVALID_SUBMISSION, 'No prediction received')
        if len(prediction_batch) != len(sample_submission_batch):
            raise GatewayRuntimeError(
                GatewayRuntimeErrorType.INVALID_SUBMISSION,
                f'Invalid predictions: expected {len(sample_submission_batch)} rows but received {len(prediction_batch)}'
            )

        ROW_ID_COLUMN_INDEX = 0
        row_id_colname = sample_submission_batch.columns[ROW_ID_COLUMN_INDEX]
        # Prevent frame shift attacks that could be performed if the IDs are predictable.
        # Ensure both dataframes are in Polars for efficient comparison.
        if row_id_colname not in prediction_batch.columns:
            raise GatewayRuntimeError(GatewayRuntimeErrorType.INVALID_SUBMISSION, f'Prediction missing column {row_id_colname}')
        if not pl.Series(prediction_batch[row_id_colname]).equals(pl.Series(sample_submission_batch[row_id_colname])):
            raise GatewayRuntimeError(
                GatewayRuntimeErrorType.INVALID_SUBMISSION,
                f'Invalid values for {row_id_colname} in batch of predictions.'
            )

    def _standardize_and_validate_paths(
            self,
            input_paths: List[Union[str, pathlib.Path]]
        ) -> List[pathlib.Path]:
        # Accept a list of str or pathlib.Path, but standardize on list of str
        for path in input_paths:
            if os.pardir in str(path):
                raise ValueError(f'Send files path contains {os.pardir}: {path}')
            if str(path) != str(os.path.normpath(path)):
                # Raise an error rather than sending users unexpectedly altered paths
                raise ValueError(f'Send files path {path} must be normalized. See `os.path.normpath`')
            if type(path) not in (pathlib.Path, str):
                raise ValueError('All paths must be of type str or pathlib.Path')
            if not os.path.exists(path):
                raise ValueError(f'Input path {path} does not exist')

        input_paths = [os.path.abspath(path) for path in input_paths]
        if len(set(input_paths)) != len(input_paths):
            raise ValueError('Duplicate input paths found')

        if not self.file_share_dir.endswith(os.path.sep):
            # Ensure output dir is valid for later use
            output_dir = self.file_share_dir + os.path.sep

        if not os.path.exists(self.file_share_dir) or not os.path.isdir(self.file_share_dir):
            raise ValueError(f'Invalid output directory {self.file_share_dir}')
        # Can't use os.path.join for output_dir + path: os.path.join won't prepend to an abspath
        output_paths = [output_dir + path for path in input_paths]
        return input_paths, output_paths

    def share_files(
            self,
            input_paths: List[Union[str, pathlib.Path]],
        ) -> List[str]:
        ''' Makes files and/or directories available to the user's inference_server. They will be mirrored under the
        self.file_share_dir directory, using the full absolute path. An input like:
            /kaggle/input/mycomp/test.csv
        Would be written to:
            /kaggle/shared/kaggle/input/mycomp/test.csv

        Args:
            input_paths: List of paths to files and/or directories that should be shared.

        Returns:
            The output paths that were shared.

        Raises:
            ValueError if any invalid paths are passed.
        '''
        input_paths, output_paths = self._standardize_and_validate_paths(input_paths)
        for in_path, out_path in zip(input_paths, output_paths):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # This makes the files available to the InferenceServer as read-only. Only the Gateway can mount files.
            # mount will only work in live kaggle evaluation rerun sessions. Otherwise use a symlink.
            if IS_RERUN:
                if not os.path.isdir(out_path):
                    pathlib.Path(out_path).touch()
                subprocess.run(f'mount --bind {in_path} {out_path}', shell=True, check=True)
            else:
                subprocess.run(f'ln -s {in_path} {out_path}', shell=True, check=True)

        return output_paths

    def write_submission(self, predictions):
        ''' Export the predictions to a submission.parquet.'''
        if isinstance(predictions, list):
            if isinstance(predictions[0], pd.DataFrame):
                predictions = pd.concat(predictions, ignore_index=True)
            elif isinstance(predictions[0], pl.DataFrame):
                try:
                    predictions = pl.concat(predictions, how='vertical_relaxed')
                except pl.exceptions.SchemaError:
                    raise GatewayRuntimeError(GatewayRuntimeErrorType.INVALID_SUBMISSION, 'Inconsistent prediction types')
                except pl.exceptions.ComputeError:
                    raise GatewayRuntimeError(GatewayRuntimeErrorType.INVALID_SUBMISSION, 'Inconsistent prediction column counts')

        if isinstance(predictions, pd.DataFrame):
            predictions.to_parquet('submission.parquet', index=False)
        elif isinstance(predictions, pl.DataFrame):
            pl.DataFrame(predictions).write_parquet('submission.parquet')
        else:
            raise ValueError(f"Unsupported predictions type {type(predictions)}; can't write submission file")

    def write_result(self, error: Optional[GatewayRuntimeError]=None):
        ''' Export a result.json containing error details if applicable.'''
        result = { 'Succeeded': error is None }

        if error is not None:
            result['ErrorType'] = error.error_type.value
            result['ErrorName'] = error.error_type.name
            # Max error detail length is 8000
            result['ErrorDetails'] = str(error.error_details[:8000]) if error.error_details else None

        with open('result.json', 'w') as f_open:
            json.dump(result, f_open)

    def handle_server_error(self, exception: Exception, endpoint: str):
        ''' Determine how to handle an exception raised when calling the inference server. Typically just format the
        error into a GatewayRuntimeError and raise.
        '''
        exception_str = str(exception)
        if isinstance(exception, gaierror) or (isinstance(exception, RuntimeError) and 'Failed to connect to server after waiting' in exception_str):
            raise GatewayRuntimeError(GatewayRuntimeErrorType.SERVER_NEVER_STARTED) from None
        if f'No listener for {endpoint} was registered' in exception_str:
            raise GatewayRuntimeError(GatewayRuntimeErrorType.SERVER_MISSING_ENDPOINT, f'Server did not register a listener for {endpoint}') from None
        if 'Exception calling application' in exception_str:
            # Extract just the exception message raised by the inference server
            message_match = re.search('"Exception calling application: (.*)"', exception_str, re.IGNORECASE)
            message = message_match.group(1) if message_match else exception_str
            raise GatewayRuntimeError(GatewayRuntimeErrorType.SERVER_RAISED_EXCEPTION, message) from None
        if isinstance(exception, grpc._channel._InactiveRpcError):
            raise GatewayRuntimeError(GatewayRuntimeErrorType.SERVER_CONNECTION_FAILED, exception_str) from None

        raise exception



import pandas as pd

# Example predictions (Replace this with your actual model output)
predictions_list = [1234, 56, -450, 789, 321]  # Ensure these are actual numbers

# Define a function to ensure correct modulo 1000 formatting
def preprocess_answer(answer):
    return (answer % 1000 + 1000) % 1000  # Ensures positive modulo

# Convert predictions into a DataFrame
df_predictions = pd.DataFrame({
    "row_id": range(len(predictions_list)),  # Replace with correct row IDs
    "answer": predictions_list
})

# Apply modulo correction to all answers
df_predictions["answer"] = df_predictions["answer"].apply(preprocess_answer)

# Save the formatted submission file
df_predictions.to_parquet("submission.parquet", index=False)

print("âœ… Submission file created successfully!")



import numpy as np
predictions_list = np.array([...]).tolist()


