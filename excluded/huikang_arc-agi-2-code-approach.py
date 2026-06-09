import os
import time
import torch


def is_on_kaggle_commit() -> bool:
    return os.getenv("KAGGLE_KERNEL_RUN_TYPE") == "Batch" and not bool(
        os.getenv("KAGGLE_IS_COMPETITION_RERUN")
    )


def is_on_kaggle_interactive() -> bool:
    return os.getenv("KAGGLE_KERNEL_RUN_TYPE") == "Interactive" and not bool(
        os.getenv("KAGGLE_IS_COMPETITION_RERUN")
    )


cutoff_time: float
if is_on_kaggle_commit():
    cutoff_time = time.time() + 30 * 60  # 30 minutes
    # cutoff_time = time.time() + (9 * 60 - 30) * 60  # 9 hours
else:  # interactive, submission
    cutoff_time = time.time() + (12 * 60 - 30) * 60  # 12 hours

assert torch.cuda.is_available()


import subprocess


def start_vllm_server() -> subprocess.Popen[bytes]:
    """Start vLLM server in the background"""
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    sequence_length = 28672  # 14 * 2048

    command: list[str] = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "/kaggle/input/qwen3/transformers/qwen3-4b-instruct-2507-tune/1",
        "--served-model-name",
        "qwen3",
        "--tensor-parallel-size",
        "4",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--dtype",
        "auto",
        "--max-model-len",
        f"{sequence_length}",
        "--max-seq-len-to-capture",
        f"{sequence_length}",
    ]

    # Start the process in the background
    with open("/kaggle/working/vllm.log", "w") as logfile:
        process: subprocess.Popen[bytes] = subprocess.Popen(
            command, stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True
        )

    print("Logs: /kaggle/working/vllm.log")
    return process


# Start the server
vllm_process: subprocess.Popen[bytes] = start_vllm_server()


from openai import OpenAI
from openai.types.chat import ChatCompletion

# Point the client to your local vLLM server
os.environ["OPENAI_API_BASE"] = "http://127.0.0.1:8000/v1"
os.environ["OPENAI_API_KEY"] = "sk-local"  # any non-empty string

client: OpenAI = OpenAI(
    base_url=os.environ["OPENAI_API_BASE"],
    api_key=os.environ["OPENAI_API_KEY"],
)


import time

for _ in range(15 * 60):
    time.sleep(1)
    try:
        print(client.models.list())
    except Exception:
        continue
    break


resp: ChatCompletion = client.chat.completions.create(
    model="qwen3",  # use your served name; if not set, the model path/name vLLM shows in logs
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about Kaggle GPUs."},
    ],
    max_tokens=128,
    temperature=0.7,
    top_p=0.9,
)

print(resp.choices[0].message.content)


prompt: str = '''
<|im_start|>user
Implement a Python function that maps the input to the outuput.

The function docustring should include
- some key observations
- the general prodcedure

def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Mention the key patterns in the input and output
    2. (other key patterns)
    3. ...
    ...

    Procedure:
    1. Mention key steps involved in solving the problem
    2. (other key patterns)
    3. ...
    ...
    N. Return the result
    """

    <implementation of the procedure here>

    return result

Do not write anything else outside the function.

{problem_string}

<|im_end|>
<|im_start|>assistant
{code_prefix}
'''.strip()


code_prefix: str = '''
def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation'''  # no strip


def generate_code(data_string: str) -> str:
    from openai.types import Completion

    response: Completion = client.completions.create(
        model="qwen3",
        prompt=prompt.format(problem_string=data_string, code_prefix=code_prefix),
        max_tokens=8192,
        top_p=0.9,
    )
    completion: str = response.choices[0].text
    code: str = code_prefix + completion
    return code


driver_code: str = """
{solve_definition}

data = {data_string}

import json

examples: list[dict[str, list[list[int]]]] = data['train']
for example in examples:
    example_input: list[list[int]] = example['input']
    example_output: list[list[int]] = example['output']
    assert solve(example_input) == example_output

testcase_submissions: list[dict[str, list[list[int]]]] = []

testcases: list[dict[str, list[list[int]]]] = data['test']
for testcase in testcases:
    testcase_input: list[list[int]] = testcase['input']
    testcase_output: list[list[int]] = solve(testcase_input)
    assert testcase_output
    assert len(testcase_output) >= 1
    assert len(testcase_output) <= 30
    for row in testcase_output:
        assert len(row) == len(testcase_output[0]) <= 30
    testcase_submissions.append(testcase_output)

with open("results.json", 'w') as f:
    json.dump(testcase_submissions, f)
"""


import os
import json
import tempfile
from json import JSONDecodeError


def execute_code(code: str, timeout: int = 5) -> tuple[bool, str, list[list[list[int]]]]:
    """
    Execute Python code and capture output.
    Returns: (is_success, printout, result)
    """

    # is_successful, stdout, combined_output, valid_prefix
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path: str = os.path.join(temp_dir, "tmp.py")
        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            result: subprocess.CompletedProcess[str] = subprocess.run(
                ["python3", temp_file_path],
                capture_output=True,
                check=False,
                text=True,
                timeout=timeout,
                cwd=temp_dir,
            )
        except subprocess.TimeoutExpired:
            return (
                False,
                f"Execution timed out after {timeout} seconds.",
                [],
            )

        stdout: str = result.stdout.strip()
        stderr: str = result.stderr.strip()

        try:
            with open(os.path.join(temp_dir, "results.json")) as f:
                result_string: str = f.read()
        except FileNotFoundError as e:
            return False, stdout + stderr + str(e), []

        try:
            json_result: list[list[list[int]]] = json.loads(result_string)
        except JSONDecodeError as e:
            return False, stdout + stderr + str(e), []

        return True, stdout + stderr, json_result


def generate_sample_data(seed: int):
    sample_data: dict[str, list[dict[str, list[list[int]]]]] = {
        "train": [
            {
                "input": [[(i + j + x) % 10 for i in range(30)] for j in range(30)],
                "output": [[(i + j + x + seed) % 10 for i in range(30)] for j in range(30)],
            } for x in range(20)
            # a list of "pairs" (typically 3 pairs)
        ],
        "test": [
            {
                "input": [[(i + j - x) % 10 for i in range(30)] for j in range(30)],
            } for x in range(2)
            # although for a small number of tasks, you will be asked to make predictions for two test inputs.
        ],
    }
    return sample_data


def get_matrix_string(matrix: list[list[int]], transpose=False):
    output_string = ""
    if not transpose:
        for row in matrix:
            output_string += "".join(str(cell) for cell in row) + "\n"
    else:
        for col in zip(*matrix):
            output_string += "".join(str(cell) for cell in col) + "\n"
    return output_string


def calculate_islands(matrix, include_corners=False) -> str:
    counter = [0 for _ in range(10)]

    dxy = [(0,1), (1,0), (0,-1), (-1,0)]
    note = "connected by edge only"
    if include_corners:
        dxy += [(1,1), (1,-1), (-1,1), (-1,-1)]
        note = "include connection by corners"

    visited = set()
    for i, row in enumerate(matrix):
        for j, cell in enumerate(row):
            if (i,j) in visited:
                continue
            visited.add((i,j))
            queue = [(i,j)]
            counter[cell] += 1
            while queue:
                x,y = queue.pop()
                for dx, dy in dxy:
                    xx = x + dx
                    yy = y + dy
                    if not 0 <= xx < len(matrix):
                        continue
                    if not 0 <= yy < len(matrix[0]):
                        continue
                    if not matrix[xx][yy] == matrix[x][y]:
                        continue
                    if (xx, yy) in visited:
                        continue
                    visited.add((xx,yy))
                    queue.append((xx,yy))

    output_string = ""
    output_string += f"\nNumber of islands of color ({note}) to count:"
    for color, count in enumerate(counter):
        if count > 0:
            output_string += f"\n{color}: {count}"

    return output_string


def calculate_statistics(matrix: list[list[int]]) -> str:

    output_string = ""
    output_string += f"\nMatrix size: {len(matrix)} x {len(matrix[0])}\n"
    # output_string += calculate_islands(matrix, include_corners=False)
    # output_string += "\n"
    # output_string += calculate_islands(matrix, include_corners=True)
    # output_string += "\n"

    return output_string


def format_input(data: dict[str, list[dict[str, list[list[int]]]]]) -> str:
    output_string = ""
    for example_idx, example in enumerate(data["train"], start=1):
        example_input: list[list[int]] = example['input']
        example_output: list[list[int]] = example['output']
        output_string += f"\n- Example {example_idx} -\n"
        output_string += f"\nExample input {example_idx}:\n{get_matrix_string(example_input)}"
        output_string += f"\nExample input {example_idx} (transposed):\n{get_matrix_string(example_input, transpose=True)}"
        output_string += calculate_statistics(example_input)  
        output_string += f"\nExample output {example_idx}:\n{get_matrix_string(example_output)}"
        output_string += f"\nExample output {example_idx} (transposed):\n{get_matrix_string(example_output, transpose=True)}"
        output_string += calculate_statistics(example_output)  

    train_data_length_limit: int = 15000
    if len(output_string) > train_data_length_limit:
        output_string = output_string[:train_data_length_limit] + " ... truncated"

    output_string += "\n\n"

    for testcase_idx, testcase in enumerate(data["test"], start=1):
        testcase_input: list[list[int]] = testcase['input']
        output_string += f"\n- Testcase {testcase_idx} -\n"
        output_string += f"\nExample input {testcase_idx}:\n{get_matrix_string(testcase_input)}"
        output_string += f"\nExample input {testcase_idx} (transposed):\n{get_matrix_string(testcase_input, transpose=True)}"
        output_string += calculate_statistics(testcase_input)  

    test_data_length_limit: int = 5000
    if len(output_string) > train_data_length_limit + test_data_length_limit:
        output_string = output_string[:train_data_length_limit + test_data_length_limit] + " ... truncated"

    return output_string


sample_data = generate_sample_data(1)
len(str(sample_data)), len(format_input(sample_data))


print(format_input(sample_data))


if is_on_kaggle_interactive():
    for _ in range(2):
        data_string: str = format_input(sample_data)
        solve_definition: str = generate_code(data_string)
        code_to_execute: str = driver_code.format(
            solve_definition=solve_definition, data_string=str(sample_data)
        )
        is_success: bool
        printout: str
        result: list[list[list[int]]]
        is_success, printout, result = execute_code(code_to_execute)
        print(is_success)
        print(printout)
        if is_success:
            print(result)
            break


if is_on_kaggle_interactive():
    print(code_to_execute)


with open("/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json") as f:
    problems_input: dict[str, dict[str, list[dict[str, list[list[int]]]]]] = json.load(
        f
    )


# use toy data for sample runs
if is_on_kaggle_commit() or is_on_kaggle_interactive():
    sample_problems_input: dict[str, dict[str, list[dict[str, list[list[int]]]]]] = {}
    # sample_problems_input["00001111"] = generate_sample_data(1)
    # sample_problems_input["00002222"] = generate_sample_data(2)
    # sample_problems_input["00003333"] = generate_sample_data(3)

    problem_id: str
    for problem_id in list(problems_input.keys())[:12]:
        sample_problems_input[problem_id] = problems_input[problem_id]

    problems_input = sample_problems_input


DEFAULT_VALUE = [[0,0], [0,0]]

# initialize data structure to contain the submissions
all_submissions: dict[str, list[dict[str, list[list[int]]]]] = {}
for problem_id, data in problems_input.items():
    testcases: list[dict[str, list[list[int]]]] = data["test"]
    testcase_submissions: list[dict[str, list[list[int]]]] = []
    testcase: dict[str, list[list[int]]]
    for testcase in testcases:
        testcase_submission: dict[str, list[list[int]]] = {
            "attempt_1": DEFAULT_VALUE,
            "attempt_2": DEFAULT_VALUE,
        }
        testcase_submissions.append(testcase_submission)
    all_submissions[problem_id] = testcase_submissions


def is_problem_attempted(problem_id: str) -> bool:
    current_submission: list[dict[str, list[list[int]]]] = all_submissions[problem_id]
    return (
        current_submission[0]["attempt_1"] != DEFAULT_VALUE
        and current_submission[0]["attempt_2"] != DEFAULT_VALUE
    )


def make_submission(problem_id: str, result: list[list[list[int]]]) -> None:
    testcase_submissions: list[dict[str, list[list[int]]]] = all_submissions[problem_id]
    attempt_key: str
    for attempt_key in ["attempt_1", "attempt_2"]:
        if testcase_submissions[0][attempt_key] == DEFAULT_VALUE:
            testcase_submission: dict[str, list[list[int]]]
            attempt: list[list[int]]
            for testcase_submission, attempt in zip(testcase_submissions, result):
                testcase_submission[attempt_key] = attempt
            return
    return


import os
import threading
from openai import APIConnectionError

# Create solutions directory if it doesn't exist
os.makedirs("solutions", exist_ok=True)

# Lock for thread-safe file writing
file_write_lock = threading.Lock()


def attempt_problem(problem_id: str) -> str:
    if is_problem_attempted(problem_id):
        return "attempted"
    if time.time() > cutoff_time:
        return "cutoff_time"
    data: dict[str, list[dict[str, list[list[int]]]]] = problems_input[problem_id]
    data_string: str = format_input(data)
    try:
        solve_definition: str = generate_code(data_string)
    except APIConnectionError:
        return "api_error"

    if is_problem_attempted(problem_id):
        return "attempted_after_generating"

    code_to_execute = driver_code.format(
        solve_definition=solve_definition, data_string=str(data)
    )

    # Write the generated code to a file
    solution_path: str = f"solutions/{problem_id}.py"
    with file_write_lock:
        with open(solution_path, "w") as f:
            f.write(code_to_execute)

    is_success, _, result = execute_code(code_to_execute)

    if is_problem_attempted(problem_id):
        return "attempted_after_executing"

    if not is_success:
        return "unsuccessful"

    # Write the generated code to a file
    solution_path: str = f"solutions/{problem_id}_submitted.py"
    with file_write_lock:
        if not os.path.exists(solution_path):
            with open(solution_path, "w") as f:
                f.write(code_to_execute)

    make_submission(problem_id, result)

    return "success"


if is_on_kaggle_interactive():
    for _ in range(2):
        sample_problem_id = sorted(problems_input.keys())[0]
        attempt_result = attempt_problem(sample_problem_id)
        print(attempt_result)
        if attempt_result == "success":
            break


from collections import Counter
import concurrent.futures

for _ in range(100):
    if time.time() > cutoff_time:
        print("cutoff_time")
        break
    problem_ids = [problem_id for problem_id in sorted(problems_input.keys()) for _ in range(80)]
    assert problem_ids[0] == problem_ids[1]  # intended to batch, use prompt caching

    with concurrent.futures.ThreadPoolExecutor(max_workers=80) as executor:
        # run in parallel
        attempt_results_iter = executor.map(attempt_problem, problem_ids)
        attempt_results = list(attempt_results_iter)

    print(
        Counter(
            (problem_id, attempt_result)
            for problem_id, attempt_result in zip(problem_ids, attempt_results)
        )
    )

    with open("submission.json", "w") as f:
        f.write(json.dumps(all_submissions))


with open("submission.json", "w") as f:
    f.write(json.dumps(all_submissions))


if is_on_kaggle_interactive() or is_on_kaggle_commit():
    print(str(all_submissions))




