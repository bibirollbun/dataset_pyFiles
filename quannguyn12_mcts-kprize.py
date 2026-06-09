import os

# https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/560682#3113134
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"


import io
import time
import shutil
import subprocess 
import pandas as pd
import polars as pl
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import ThreadPool

import kaggle_evaluation.konwinski_prize_inference_server
from typing import List, Tuple, Dict, Optional

start_time = time.time()


instance_count: Optional[int] = None


def get_number_of_instances(num_instances: int) -> None:
    """The very first message from the gateway will be the total number of instances to be served.
    You don't need to edit this function.
    """
    global instance_count
    instance_count = num_instances


from vllm import LLM, SamplingParams, RequestOutput
import warnings

warnings.simplefilter("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if os.getenv("KAGGLE_KERNEL_RUN_TYPE") or os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    llm_model_pth: str = (
        "/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-32b-awq/1"
    )
else:
    llm_model_pth: str = "/root/volume/KirillR/QwQ-32B-Preview-AWQ"

BATCH_SIZE: int = 6
VALIDATION_COPY_COUNT: int = 1
MAX_TOKENS: int = 4096

MAX_NUM_SEQS: int = 6
MAX_MODEL_LEN: int = 32_768

llm: LLM = LLM(
    llm_model_pth,
    max_num_seqs=MAX_NUM_SEQS,  # Maximum number of sequences per iteration. Default is 256
    max_model_len=MAX_MODEL_LEN,  # Model context length
    trust_remote_code=True,  # Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer
    tensor_parallel_size=4,  # The number of GPUs to use for distributed execution with tensor parallelism
    gpu_memory_utilization=0.95,  # The ratio (between 0 and 1) of GPU memory to reserve for the model
    seed=2024,
)


tokenizer = llm.get_tokenizer()


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


import os


def stringify_directory(directory: str) -> str:
    full_paths: List[str] = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path: str = os.path.join(root, file)
            full_paths.append(full_path)
    return "\n".join(full_paths)


import re


def extract_file_query(xml_content: str) -> Dict[str, List[str]]:
    import xml.etree.ElementTree as ET

    # Prepare a data structure to collect results
    parsed_data: Dict[str, List[str]] = {}
    pattern: str = r"<root>(.*?)</root>"
    matches: List[str] = re.findall(pattern, xml_content, re.DOTALL)

    for match in matches:
        try:
            # Parse the XML
            root = ET.fromstring("<root>" + match + "</root>")

            # Find all <entry> elements
            for entry in root.findall("entry"):
                # Extract the <filepath> text
                filepath = entry.find("filepath")
                filepath_text: Optional[str] = (
                    filepath.text.strip()
                    if filepath is not None and filepath.text is not None
                    else None
                )

                # Locate <strings_to_search> container
                strings_container = entry.find("strings_to_search")

                # Gather each <string_to_search> text
                search_strings: List[str] = []
                if strings_container is not None:
                    for s in strings_container.findall("string_to_search"):
                        if s.text is not None:
                            search_strings.append(s.text.strip())

                # Store in a dictionary: { filepath: [search_strings...] }
                parsed_data[filepath_text] = search_strings  # type: ignore
        except:
            print("Error parsing output")
            print(xml_content)
            return {}

    return parsed_data


import re

reading_prompt: str = (
    """
You will be implementing a git diff patch to solve an issue with the code repository.
You will first need to select files in the file directory.

This is the problem statement.

{problem_statement}

This is the file directory

<directory>
{directory_string}
</directory>

Which files should be inspected so that we can solve the problem?
When we inspect each file, what strings should be searched?

Return the strings to search in this format

(explanation)

<root>
    <entry>
        <filepath>filepath</filepath>
        <strings_to_search>
            <string_to_search>string_to_search</string_to_search>
            ...
            <string_to_search>string_to_search</string_to_search>
        </strings_to_search>
    </entry>
    <entry>
        <filepath>filepath</filepath>
        <strings_to_search>
            <string_to_search>string_to_search</string_to_search>
            ...
            <string_to_search>string_to_search</string_to_search>
        </strings_to_search>
    </entry>
    ...
</root>
...

Notes:
- Make sure to encode each entry between <root> and </root>
- Return the FULL filepath - exactly as specified in <directory> and </directory>
    - Example: <filepath>repo/path/to/directory/file.py</filepath>
- If you are searching for a word instead of a substring, maybe add spaces or brackets before and after the string
    - For example, if you are searching for uses of the function `calculate`, use ` calculate(` as the search string instead of `calculate`
- Prefer searching longer strings
    - Avoid searching for strings that might appear in many parts of the codebase
- Search the test files as well to understand the feature behavior
    - Also search for the relevant function calls in the test files
""".strip()
)


def get_selection_query(
    directory_string: str, problem_statement: str
) -> Tuple[List[str], List[Dict[str, List[str]]]]:
    sampling_params: SamplingParams = SamplingParams(
        temperature=0.6,  # randomness of the sampling
        min_p=0.01,
        skip_special_tokens=True,  # Whether to skip special tokens in the output
        max_tokens=MAX_TOKENS,
    )

    list_of_messages: List[List[Dict[str, str]]] = [
        [
            {
                "role": "user",
                "content": reading_prompt.format(
                    problem_statement=problem_statement[:20_000],
                    directory_string=directory_string[:30_000],
                ),
            },
        ]
        for _ in range(BATCH_SIZE)
    ]

    prompt_texts: List[str] = [
        (
            tokenizer.apply_chat_template(
                conversation=messages, tokenize=False, add_generation_prompt=True
            )  # type: ignore
        )
        + "<think>\n"
        for messages in list_of_messages
    ]
    # print(prompt_texts)

    print("get_selection_query", [count_tokens(text) for text in prompt_texts])
    request_outputs: list[RequestOutput] = llm.generate(
        prompt_texts, sampling_params=sampling_params
    )
    if not request_outputs:
        return [], []
    response_texts: List[str] = [
        request_output.outputs[0].text for request_output in request_outputs
    ]
    print("get_selection_query", [count_tokens(text) for text in response_texts])

    completion_texts = [
        prompt_text + response_text
        for prompt_text, response_text in zip(prompt_texts, response_texts)
    ]
    file_queries: List[Dict[str, List[str]]] = [
        extract_file_query(response_text) for response_text in response_texts
    ]
    return completion_texts, file_queries


REPO_PATH: str = "repo"


def fetch_file_contents(
    files_to_search: Dict[str, List[str]], context_lines: int = 12, max_gap: int = 0
) -> str:
    from io import StringIO
    from typing import Tuple

    def find_lines_in_files_with_context(
        search_map: Dict[str, List[str]], context_lines: int = context_lines
    ) -> List[List[List[Tuple[int, str]]]]:
        """
        Given a dictionary mapping file paths to a list of search terms,
        open each file and gather *snippets* of lines that contain any
        of those search terms, including 'context_lines' before and after.

        Returns a list of lists:
        [
          [  # For file1
             [ (line_number, text), (line_number, text), ... ],
             [ ... ],
          ],
          [  # For file2
             ...
          ],
          ...
        ]
        """
        all_matches_per_file: List[List[List[Tuple[int, str]]]] = []

        for path, terms in search_map.items():
            if not os.path.isfile(path):
                # If the file is not found, record an empty list
                all_matches_per_file.append([])
                continue

            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            file_snippets: List[List[Tuple[int, str]]] = []
            num_lines: int = len(lines)

            for i, line in enumerate(lines, start=1):
                if any(t in line for t in terms):
                    start_idx: int = max(1, i - context_lines)
                    end_idx: int = min(num_lines, i + context_lines)
                    snippet: List[Tuple[int, str]] = []
                    for snippet_no in range(start_idx, end_idx + 1):
                        text_content: str = lines[snippet_no - 1].rstrip("\n")
                        snippet.append((snippet_no, text_content))
                    file_snippets.append(snippet)

            all_matches_per_file.append(file_snippets)

        return all_matches_per_file

    # ---------------------------------------------------------
    # 3. MERGE OVERLAPPING/ADJACENT SNIPPETS
    # ---------------------------------------------------------

    def merge_file_snippets(
        file_snippets: List[List[Tuple[int, str]]], gap: int = 0
    ) -> List[List[Tuple[int, str]]]:
        """
        Merge overlapping or nearly adjacent snippets in a single file’s snippet list.
        """
        intervals: List[Tuple[int, int, List[Tuple[int, str]]]] = []
        for snippet in file_snippets:
            if snippet:
                start_line: int = snippet[0][0]
                end_line: int = snippet[-1][0]
                intervals.append((start_line, end_line, snippet))

        intervals.sort(key=lambda x: x[0])  # sort by start line

        merged: List[Tuple[int, int, List[Tuple[int, str]]]] = []
        for start, end, snippet in intervals:
            if not merged:
                merged.append((start, end, snippet))
                continue

            prev_start, prev_end, prev_snippet = merged[-1]
            if start <= prev_end + gap:
                new_end: int = max(end, prev_end)
                combined_dict: Dict[int, str] = {}
                for ln, txt in prev_snippet:
                    combined_dict[ln] = txt
                for ln, txt in snippet:
                    combined_dict[ln] = txt
                merged_snippet: List[Tuple[int, str]] = [
                    (ln, combined_dict[ln]) for ln in sorted(combined_dict)
                ]
                merged[-1] = (prev_start, new_end, merged_snippet)
            else:
                merged.append((start, end, snippet))

        # Extract just the merged snippet portion
        return [x[2] for x in merged]

    def merge_all_snippets(
        all_files_snips: List[List[List[Tuple[int, str]]]], gap: int = 0
    ) -> List[List[List[Tuple[int, str]]]]:
        """
        Merge snippet blocks within each file.
        all_files_snips is a list-of-lists:
          [
            [ snippetA, snippetB, ... ],  # file 1
            [ snippetC, snippetD, ... ],  # file 2
          ]
        """
        merged: List[List[List[Tuple[int, str]]]] = []
        for snips in all_files_snips:
            merged.append(merge_file_snippets(snips, gap=gap))
        return merged

    # ---------------------------------------------------------
    # 4. RUN LOGIC: generate files, search, merge, and BUILD A STRING
    # ---------------------------------------------------------

    has_any_matches: bool = False

    # 1) Gather snippets around each match
    context_snippets: List[List[List[Tuple[int, str]]]] = (
        find_lines_in_files_with_context(files_to_search, context_lines=context_lines)
    )

    # 2) Merge overlapping snippets
    merged_snips: List[List[List[Tuple[int, str]]]] = merge_all_snippets(
        context_snippets, gap=max_gap
    )

    # 3) Build a string (instead of printing)
    output = StringIO()

    # Header
    output.write("Sample files created successfully.\n\n")
    output.write("Search Results (by file, merging any overlapping context):\n\n")

    # For each file
    for (filepath, terms), snippet_list in zip(files_to_search.items(), merged_snips):
        output.write(f"[file name]: {filepath[len(REPO_PATH) + 1:]}\n")
        terms_searched_as_str = "\n".join(terms)
        output.write(f"[terms searched]:\n{terms_searched_as_str}\n")
        output.write("[file content begin]\n")
        if not snippet_list:
            output.write("  No matches found.\n")
        else:
            has_any_matches = True
            for snippet_idx, snippet in enumerate(snippet_list, start=1):
                snippet_start: int = snippet[0][0]
                snippet_end: int = snippet[-1][0]
                output.write(
                    f"\nMatch #{snippet_idx}, lines {snippet_start} to {snippet_end}:\n"
                )
                for line_no, text in snippet:
                    output.write(f"  {line_no:3d} | {text}\n")
                output.write("\n")
        output.write("[file content end]\n\n")

    file_content_string: str = output.getvalue()

    if has_any_matches:
        return file_content_string
    return ""


import re

# --- Helper Functions - Patch Extraction and Test Outcome ---

def extract_patch_string(text: str) -> Optional[str]:
    pattern: str = r"\n```diff\n(.*?)\n```"
    matches: List[str] = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None
    return matches[-1] + "\n"

def get_unit_test_outcome(repo_path: str) -> List[str]: # Changed return type to List[str]
    """
    Simplified test running logic.  Replace with the *exact* logic
    used in the evaluation (if available).
    """
    # Simulate running unit tests (replace with actual test execution)
    # This is crucial for MCTS to work effectively
    cmd = f"python -m unittest discover -s {repo_path}"
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, timeout=60)
        # If successful, all tests passed (crude approximation)
        return ["PASSED"]  # Return a list to mimic the original logic
    except subprocess.CalledProcessError as e:
        # If there was an error, some tests failed
        # (Parse the output to identify individual failed tests, if possible)
        print(f"Tests failed: {e.stderr}")
        return ["FAILED"]


from pathlib import Path

verifying_prompt: str = (
    """
This is the problem statement.

{problem_statement}

These are the files that is thought to be relevant, which may not be complete.

{file_content_string}

This is the proposed patch to fix the problem.

{patch_string}

Evaluate whether the patch works
- The patch fully fixes the problem described in the problem statement.
- The patch does not cause side effects and make any other tests fail.

End your response with exactly either of
- <label>Yes</label>, this fixes the problem.
- <label>No</label>, this does not fix the problem.

Reminder
- Only evaluate, do not provide suggestion on how to fix.
- Remember to write exactly either of <label>Yes</label> or <label>No</label> in the last line
""".strip()
)


def is_valid_patch_format(patch_string: str) -> bool:
    """
    A quick check to confirm if a patch could be valid.
    """
    if not(isinstance(patch_string, str)):
        return False
    try:
        patch_set = unidiff.PatchSet(patch_string)
        if len(patch_set) == 0:
            return False
    except Exception:
        return False
    return True


def patch_dry_run_succeeds(patch_string: str, repo_path: str = REPO_PATH, timeout: int = 60) -> bool:
    """
    A robust check if the patch will proceed without any errors.
    Should be run after `is_valid_patch_format()`: the patch
    command can hang if the inputs are sufficiently invalid.

    Args:
        patch_path: Path to a file containing the patch.
        repo_path: Path to the directory to be patched.
        timeout: Number of seconds before the dry run will be cancelled.
    """
    with open("patch.txt", "w") as f:
        f.write(patch_string)
    patch_path = "/kaggle/working/patch.txt"

    cmd = f"patch --quiet --dry-run -p1 -i {patch_path} -d {repo_path}"
    try:
        subprocess.run(cmd, shell=True, check=True, timeout=timeout)
        return True
    except subprocess.CalledProcessError:
        return False


def get_verification(
    problem_statement: str,
    file_content_strings: List[str],
    patch_strings: List[Optional[str]],
    repo_path: str,
) -> Tuple[List[List[str]], List[List[bool]]]:
    assert len(file_content_strings) == len(patch_strings)
    sampling_params: SamplingParams = SamplingParams(
        temperature=0.3,  # randomness of the sampling
        min_p=0.01,
        skip_special_tokens=True,  # Whether to skip special tokens in the output
        max_tokens=MAX_TOKENS,
    )

    inference_idx_to_input_idx: list[int] = [
        input_idx
        for _ in range(VALIDATION_COPY_COUNT)
        for input_idx, patch_string in enumerate(patch_strings)
        if patch_string is not None and is_valid_patch_format(patch_string) # and patch_dry_run_succeeds(patch_string, repo_path)
    ]
    print(inference_idx_to_input_idx)

    list_of_messages: List[List[Dict[str, str]]] = [
        [
            {
                "role": "user",
                "content": verifying_prompt.format(
                    problem_statement=problem_statement[:20_000],
                    file_content_string=file_content_strings[input_idx][:30_000],
                    patch_string=patch_strings[input_idx],
                ),
            },
        ]
        for input_idx in inference_idx_to_input_idx
    ]

    prompt_texts: List[str] = [
        (
            tokenizer.apply_chat_template(
                conversation=messages, tokenize=False, add_generation_prompt=True
            )  # type: ignore
        )
        + "<think>\n"
        for messages in list_of_messages
    ]
    # print(prompt_texts)

    print("get_verification", [count_tokens(text) for text in prompt_texts])
    request_outputs: list[RequestOutput] = llm.generate(
        prompt_texts, sampling_params=sampling_params
    )
    response_texts: List[str] = [
        request_output.outputs[0].text for request_output in request_outputs
    ]
    print("get_verification", [count_tokens(text) for text in response_texts])

    completion_texts = [
        prompt_text + response_text
        for prompt_text, response_text in zip(prompt_texts, response_texts)
    ]
    judgments_flattened: List[bool] = [
        "<label>Yes</label>" in response_text for response_text in response_texts
    ]
    print(judgments_flattened)

    judgments_aggregated: List[List[bool]] = [[] for _ in file_content_strings]
    completion_text_aggregated: List[List[str]] = [[] for _ in patch_strings]
    for inference_idx, (completion_text, judgement) in enumerate(
        zip(completion_texts, judgments_flattened)
    ):
        input_idx = inference_idx_to_input_idx[inference_idx]
        completion_text_aggregated[input_idx].append(completion_text)
        judgments_aggregated[input_idx].append(judgement)
    print(judgments_aggregated)

    return completion_text_aggregated, judgments_aggregated


# --- MCTS Node Class ---
import math # Import math for sqrt and log

def get_reward(self, repo_path: str) -> float:
    """More detailed reward function."""
    unit_test_outcomes = get_unit_test_outcome(repo_path) # Assuming this returns a list of test outcomes now
    reward = 0.0
    num_passed_tests = 0
    num_failed_tests = 0
    num_errors = 0

    for outcome in unit_test_outcomes:
        if outcome == "PASSED":
            reward += 2.0
            num_passed_tests += 1
        elif outcome == "FAILED":
            reward -= 3.0
            num_failed_tests += 1
        elif outcome == "ERROR": # Handle ERROR outcomes
            reward -= 1.5
            num_errors += 1
        elif outcome == "SKIPPED": # Handle SKIPPED outcomes
            pass

    if num_passed_tests == 0 and num_failed_tests > 0:
        reward -= 1.0

    if self.patch_history and not patch_dry_run_succeeds(self.patch_history[-1], repo_path):
        reward -= 10.0 # Increased penalty for invalid patch

    return reward


class MCTSNode: # Class definition remains the same as previously optimized
    def __init__(self, repo_state: str, patch_history: List[str], problem_statement: str, parent=None, selected_files_content: str = "", visit_count=0, total_value=0): # Added visit_count and total_value for progressive widening and node reuse
        self.repo_state = repo_state
        self.patch_history = patch_history
        self.problem_statement = problem_statement
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.visits = visit_count # Renamed from visits to visit_count
        self.score = total_value # Renamed from score to total_value
        self.untried_actions: List[str] = []
        self.selected_files_content = selected_files_content

    def generate_actions(self, num_actions=20): # num_actions increased to 20
        patching_prompt = """You are a world-class code patching expert. Your goal is to create the best git diff patch for a given problem... (rest of prompt remains same for CoT)""".strip() # Chain-of-Thought Prompt

        prompt_texts = [patching_prompt.format(problem_statement=self.problem_statement, code=self.selected_files_content) for _ in range(num_actions)]
        sampling_params = SamplingParams(
            temperature=1.0, # Increased temperature for more creative patches
            top_p=0.95,      # High top_p for broader sampling
            min_p=0.005,     # Reduced min_p to allow less common tokens
            skip_special_tokens=True,
            max_tokens=1024, # Reduced max_tokens for faster generation
        ) # Optimized sampling params
        request_outputs: list[RequestOutput] = llm.generate(prompt_texts, sampling_params=sampling_params)
        response_texts: List[str] = [request_output.outputs[0].text for request_output in request_outputs]
        patch_strings: List[str] = [extract_patch_string(text) for text in response_texts if extract_patch_string(text) is not None]

        filtered_patches = []
        for patch_string in patch_strings:
            if patch_string is None:
                continue
            if not is_valid_patch_format(patch_string):
                continue
            if patch_string.count('\n') < 3: # Filter out short patches
                continue
            filtered_patches.append(patch_string)

        import random # Import random inside function for clarity
        if len(filtered_patches) > num_actions:
            self.untried_actions = random.sample(filtered_patches, num_actions)
        else:
            self.untried_actions = filtered_patches


    def apply_patch(self, patch_string: str) -> str: # Apply patch logic remains same
        with open("patch.txt", "w") as f:
            f.write(patch_string)
        patch_path = "/kaggle/working/patch.txt"

        with open("repo_content.txt", "w") as f:
            f.write(self.repo_state)
        content_file_path = "/kaggle/working/repo_content.txt"

        cmd = f"patch < {patch_path} < {content_file_path}"
        try:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, timeout=60)
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Patch application failed: {e.stderr}")
            return self.repo_state


    def get_reward(self, repo_path: str) -> float:
        """More detailed reward function."""
        unit_test_outcomes = get_unit_test_outcome(repo_path) # Assuming this returns a list of test outcomes now
        reward = 0.0
        num_passed_tests = 0
        num_failed_tests = 0
        num_errors = 0

        for outcome in unit_test_outcomes:
            if outcome == "PASSED":
                reward += 2.0
                num_passed_tests += 1
            elif outcome == "FAILED":
                reward -= 3.0
                num_failed_tests += 1
            elif outcome == "ERROR":
                reward -= 1.5
                num_errors += 1
            elif outcome == "SKIPPED":
                pass

        if num_passed_tests == 0 and num_failed_tests > 0:
            reward -= 1.0

        if self.patch_history and not patch_dry_run_succeeds(self.patch_history[-1], repo_path):
            reward -= 10.0 # Increased penalty for invalid patch

        return reward

    def is_terminal(self) -> bool: # Terminal condition remains same
        if len(self.patch_history) >= 5:
            return True
        return False


# --- MCTS Algorithm (mcts function) ---

def mcts(root: MCTSNode, repo_path: str, iterations: int = 100) -> str: # Increased iterations to 100
    C = 2.0 # Exploration constant

    executor = ThreadPoolExecutor(max_workers=8) # Parallel test execution
    
    for _ in range(iterations):
        node = root
        # --- Selection using UCT ---
        while node.untried_actions == [] and node.children != []:
            best_child = None
            best_uct = -float("inf")
            for child in node.children:
                uct = child.score / child.visits + C * (math.log(node.visits) / child.visits) ** 0.5
                if uct > best_uct:
                    best_child = child
                    best_uct = uct
            node = best_child  # type: ignore

        # --- Expansion ---
        if node.untried_actions == [] and not node.is_terminal():
            if node.visits < 10: # Progressive widening: Explore more if node is not visited much
                 node.generate_actions(num_actions=5) # Generate fewer actions initially
            else:
                node.generate_actions(num_actions=20) # Generate more actions for well-visited nodes

        # --- Simulation ---
        if node.untried_actions:
            action = node.untried_actions.pop()

            new_repo_state = node.apply_patch(action)
            new_patch_history = node.patch_history + [action]

            child = MCTSNode(
                repo_state=new_repo_state,
                patch_history=new_patch_history,
                problem_statement=node.problem_statement,
                parent=node,
            )
            node.children.append(child)
            node = child

            # Use parallel execution for get_reward (test running)
            future_reward = executor.submit(child.get_reward, repo_path) # Submit test execution to thread pool
            reward = future_reward.result() # Wait for test execution to complete and get reward
        else:
            #If no untried actions, but not terminal, still do a rollout (exploration)
            if not node.is_terminal() and node.children: # Check if children exist before attempting rollout
              node = random.choice(node.children) # Basic rollout: explore existing children
              reward = node.get_reward(repo_path) # Get reward from rolled-out node
            else:
              reward = node.get_reward(repo_path) # Or get reward directly if terminal or no children


        # --- Backpropagation ---
        while node is not None:
            node.visits += 1
            node.score += reward
            node = node.parent

    executor.shutdown(wait=False) # Shutdown thread pool

    # --- Action Choice ---
    best_child = None
    best_score = -float('inf')
    for child in root.children:
        if child.score > best_score:
            best_child = child
            best_score = child.score

    if best_child:
        return best_child.patch_history[-1]
    return None # Return None if no patch found


# --- Predict Inner Function ---
import shutil
import tempfile

def predict_inner(problem_statement: str, directory: str) -> Optional[str]:

    # 1. Run file selection logic:
    directory_string = stringify_directory(directory)
    selection_completion_texts, file_queries = get_selection_query(directory_string, problem_statement) #Unnecessary for current MCTS implementation.
    # For simplicity, let's just use the first file_query result:
    file_query = file_queries[0] if file_queries else {}

    #2. Get only the selected files content to pass into MCTS:
    selected_files_content = fetch_file_contents(file_query)

    # 3. Create initial repo state as a very long string.
    file_content_repo: str = ""
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path: str = os.path.join(root, file)
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                file_content_repo += f.read() #This entire file content is no longer passed in.


    # 4. Create root node for MCTS with the selected content
    root = MCTSNode(repo_state= file_content_repo,
                       patch_history=[],
                       problem_statement=problem_statement,
                       selected_files_content = selected_files_content) # Pass the selected content


    # 5. Run the MCTS
    best_patch = mcts(root=root, repo_path=directory, iterations=5) # Run MCTS

    return best_patch


# --- Predict Function (Outer Wrapper) ---

def predict(problem_statement: str, repo_archive: io.BytesIO, pip_packages_archive: io.BytesIO, env_setup_cmds_templates: List[str]) -> Optional[str]:
    """Main predict function for inference server."""
    with open("repo_archive.tar", "wb") as f:
        f.write(repo_archive.read())

    repo_path: str = REPO_PATH
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)
    shutil.unpack_archive("repo_archive.tar", extract_dir=repo_path)
    os.remove("repo_archive.tar")

    patch_string: Optional[str] = None
    patch_string = predict_inner(
        problem_statement=problem_statement, directory=repo_path
    )
    shutil.rmtree(repo_path)
    print("submitted patch_string")
    print(patch_string)
    return patch_string


# --- Optional: Data Loading and Demo (for Interactive Testing) ---
import os
import zipfile

os.makedirs("/kaggle/tmp/konwinski-prize-alt", exist_ok=True)

try:
    with zipfile.ZipFile("/kaggle/input/konwinski-prize/data.a_zip", "r") as zip_ref:
        zip_ref.extractall("/kaggle/tmp/konwinski-prize-alt/")
except:
    pass

import pandas as pd

def get_problem(problem_index: int) -> Tuple[str, str, io.BytesIO]:
    df = pd.read_parquet("/kaggle/tmp/konwinski-prize-alt/data/data.parquet")

    problem_statement: str = df["problem_statement"][problem_index]
    repo_path: str = f"/kaggle/tmp/konwinski-prize-alt/data/repos/repo__{df['instance_id'][problem_index]}"

    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.make_archive(os.path.join(tmpdir, "a_repo"), "tar", repo_path)
        with open(os.path.join(tmpdir, "a_repo.tar"), "rb") as f:
            repo_archive = io.BytesIO(f.read())

    return problem_statement, repo_path, repo_archive

demo_problem_index: int = 0

if os.getenv("KAGGLE_KERNEL_RUN_TYPE") == "Interactive" and not os.getenv(
    "KAGGLE_IS_COMPETITION_RERUN"
):
    problem_statement, repo_path, repo_archive = get_problem(
        problem_index=demo_problem_index
    )

    print(repo_path)
    print(problem_statement)
    print(len(list(repo_archive)))
    print(len(list(repo_archive)))


skip_prediction = False


# --- Optional: Run Prediction and Evaluate Locally ---
if os.getenv("KAGGLE_KERNEL_RUN_TYPE") == "Interactive" and not os.getenv(
    "KAGGLE_IS_COMPETITION_RERUN"
):
    skip_prediction = False
    problem_statement, repo_path, repo_archive = get_problem(
        problem_index=demo_problem_index
    )
    patch_string = predict(problem_statement, repo_archive, io.BytesIO(), [])

if (
    os.getenv("KAGGLE_KERNEL_RUN_TYPE") == "Interactive"
    and not os.getenv("KAGGLE_IS_COMPETITION_RERUN")
    and patch_string is not None
):
    import polars as pl
    df = pl.read_parquet("/kaggle/tmp/konwinski-prize-alt/data/data.parquet")
    import kaggle_evaluation.konwinski_prize_gateway

    k_prize_gateway = kaggle_evaluation.konwinski_prize_gateway.KPrizeGateway()
    k_prize_gateway.unpack_data_paths()

    results = k_prize_gateway._evaluate_instance(
        instance=df.row(demo_problem_index, named=True),
        patch=patch_string,
    )

    from collections import Counter
    print(
        demo_problem_index, Counter(result.unit_test_outcome for result in results[1:])
    )

if (
    os.getenv("KAGGLE_KERNEL_RUN_TYPE") == "Interactive"
    and not os.getenv("KAGGLE_IS_COMPETITION_RERUN") and patch_string is not None
):
    from kaggle_evaluation.konwinski_prize_gateway import UnitTestOutcome

    print("\n--- Unit Test Results (if any failed) ---")
    for result in results[1:]: # Skip the first result (instance setup)
        if result.unit_test_outcome != UnitTestOutcome.PASSED:
            print(f"Test Name: {result.test_name}")
            print(f"Fail Description:\n{result.fail_description}\n")


skip_prediction = False
inference_server = (
    kaggle_evaluation.konwinski_prize_inference_server.KPrizeInferenceServer(
        get_number_of_instances, predict
    )
)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            "/kaggle/input/konwinski-prize/",  # Path to the entire competition dataset
            "/kaggle/tmp/konwinski-prize/",  # Path to a scratch directory for unpacking data.a_zip.
        )  # type: ignore
    )

