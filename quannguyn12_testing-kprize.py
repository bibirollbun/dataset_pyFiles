import os
# https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/560682#3113134
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"


import io
import time
import shutil

import pandas as pd
import polars as pl

import kaggle_evaluation.konwinski_prize_inference_server
from typing import List, Tuple, Dict, Optional

start_time = time.time()
allowed_time = [start_time + 60 * 60]


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
        "/kaggle/input/m/huikang/deepseek-r1/transformers/qwen-qwq-32b-awq/1"
    )
else:
    llm_model_pth: str = "/root/volume/KirillR/QwQ-32B-Preview-AWQ"

BATCH_SIZE: int = 7
VALIDATION_COPY_COUNT: int = 3
MAX_TOKENS: int = 4096

MAX_NUM_SEQS: int = 7
MAX_MODEL_LEN: int = 32_768

llm: LLM = LLM(
    llm_model_pth,
    dtype="float16",
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


def setup(
    repo_archive: io.BytesIO,
    pip_packages_archive: io.BytesIO,
    env_setup_cmds_templates: list[str],
    repo_path: str,
) -> None:
    """Replace this function with your inference code.
    Args:
        problem_statement: The text of the git issue.
        repo_path: A BytesIO buffer path with a .tar containing the codebase that must be patched. The gateway will make this directory available immediately before this function runs.
        pip_packages_archive: A BytesIO buffer path with a .tar containing the wheel files necessary for running unit tests.
        env_setup_cmds_templates: Commands necessary for installing the pip_packages_archive.
    """

    # Unpack the codebase to be patched into a directory that won't be exported when
    # the notebook is saved.
    archive_path = "/tmp/repo_archive.tar"
    with open(archive_path, "wb") as f:
        f.write(repo_archive.read())
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)
    shutil.unpack_archive(archive_path, extract_dir=repo_path)
    os.remove(archive_path)

    """
    Unpack pip_packages if you want to run unit tests on your patch.
    Note that editing unit tests with your patch -- even to add valid tests -- can cause your submission to be flagged as a failure.
    Most of the relevant repos use pytest for running tests. You will almost certainly need to run only a subset of the unit tests to avoid running out of inference time.
    """
    pip_archive_dir = "/tmp/pip_packages_archive.tar"
    with open(pip_archive_dir, "wb") as f:
        f.write(pip_packages_archive.read())
    pip_packages_path = "/path/to/pip_packages"
    if os.path.exists(pip_packages_path):
        shutil.rmtree(pip_packages_path)
    shutil.unpack_archive(pip_archive_dir, extract_dir=pip_packages_path)
    os.remove(pip_archive_dir)

    # Get env setup cmds by setting the pip_packages_path
    env_setup_cmds = [
        cmd.format(pip_packages_path=pip_packages_path)
        for cmd in env_setup_cmds_templates
    ]

    # Run env setup for the repo
    subprocess.run(
        "\n".join(env_setup_cmds),
        shell=True,
        executable="/bin/bash",
        cwd=repo_path,
    )


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

def extract_patch_string(text: str) -> Optional[str]:
    pattern: str = r"\n```diff\n(.*?)\n```"
    matches: List[str] = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None
    return matches[-1] + "\n"


patching_prompt: str = (
    """
You will be implementing a git diff patch to solve an issue with the code repository.
This is the problem statement.

{problem_statement}

These are the files that is thought to be relevant

{file_content_string}

Write a git diff within ```diff and ``` that fully fixes the problem.
The git diff should not cause other tests to fail.

Example:

```diff
--- a/first.txt
+++ b/first.txt
@@ -1,3 +1,3 @@
 start
-first change
+new first change
 middle
@@ -7,4 +7,4 @@
 some content
-second change
+new second change
 more content
--- a/second.txt
+++ b/second.txt
@@ -1,3 +1,3 @@
 beginning
-old line
+new line
 end
```

Reminder
- Put your diff within ```diff and ``` and make sure the diff is valid.
- Only the last diff printed will be considered.
""".strip()
)

import re


def get_patch_string(
    problem_statement: str, file_content_strings: List[str]
) -> Tuple[List[str], List[Optional[str]]]:
    sampling_params: SamplingParams = SamplingParams(
        temperature=0.7,  # randomness of the sampling
        min_p=0.01,
        skip_special_tokens=True,  # Whether to skip special tokens in the output
        max_tokens=MAX_TOKENS,
    )

    inference_idx_to_input_idx: list[int] = [
        input_idx
        for input_idx, file_content_string in enumerate(file_content_strings)
        if file_content_string != ""
    ]

    list_of_messages: List[List[Dict[str, str]]] = [
        [
            {
                "role": "user",
                "content": patching_prompt.format(
                    problem_statement=problem_statement[:20_000],
                    file_content_string=file_content_strings[input_idx][:30_000],
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

    print("get_patch_string", [count_tokens(text) for text in prompt_texts])
    request_outputs: list[RequestOutput] = llm.generate(
        prompt_texts, sampling_params=sampling_params
    )
    response_texts_from_inference: List[str] = [
        request_output.outputs[0].text for request_output in request_outputs
    ]
    print(
        "get_patch_string",
        [count_tokens(text) for text in response_texts_from_inference],
    )
    completion_texts_from_inference = [
        prompt_text + response_text
        for prompt_text, response_text in zip(
            prompt_texts, response_texts_from_inference
        )
    ]
    patch_strings_from_inference: List[Optional[str]] = [
        extract_patch_string(response_text)
        for response_text in response_texts_from_inference
    ]

    completion_texts: list[str] = ["" for _ in file_content_strings]
    patch_strings: List[Optional[str]] = [None for _ in file_content_strings]
    for inference_idx, (completion_text, patch_string) in enumerate(
        zip(completion_texts_from_inference, patch_strings_from_inference)
    ):
        input_idx = inference_idx_to_input_idx[inference_idx]
        completion_texts[input_idx] = completion_text
        patch_strings[input_idx] = patch_string

    return completion_texts, patch_strings


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


import unidiff
import subprocess


def calculate_patch_score(
    patch_string: Optional[str],
    judgments: List[bool],
    repo_path: str,
    validation_copy_count: int = VALIDATION_COPY_COUNT
) -> float:
    
    score = 0.0

    if patch_string is None:
        return -3.0  
        
    if not is_valid_patch_format(patch_string):
        return -2.0  

    if not patch_dry_run_succeeds(patch_string, repo_path):
        return -1.0  

    judgment_weight = 2.0
    valid_format_bonus = 1.0
    dry_run_bonus = 2.0
    size_penalty_factor = 0.3
    file_count_penalty = 0.0
    
    if judgments.count(True) == 0: # Heavily penalize no "Yes" votes
        return -100.0
        
    score += judgments.count(True) ** 2 * judgment_weight

    score += valid_format_bonus
    score += dry_run_bonus

    #  Non-linear penalty for patch size - inverse function, penalize very long patches more
    score -= patch_lines_penalty(patch_string, size_penalty_factor) # Using separate function

    patch_set = unidiff.PatchSet(patch_string)
    num_files_changed = len(patch_set)
    score -= num_files_changed * file_count_penalty

    return score

def patch_lines_penalty(patch_string: str, penalty_factor: float) -> float:
    """Calculates a non-linear penalty based on the number of lines in the patch.
    Uses an inverse function to penalize very long patches more heavily.
    """
    patch_lines = patch_string.strip().count('\n') + 1 # Add 1 to avoid division by zero
    if patch_lines <= 15: # Reduced penalty for very short patches
        return patch_lines * penalty_factor * 0.5 # Less penalty for short patches
    return  (100 / patch_lines) * penalty_factor # Inverse penalty - longer patches penalized more sharply
    
def choose_patch_string_optimized(
    patch_strings: list[Optional[str]], judgments_aggregated: List[List[bool]], repo_path: str, correction_threshold: float, validation_copy_count: int = VALIDATION_COPY_COUNT
) -> tuple[list[int], Optional[str]]:
    best_score = -float('inf')
    best_patch_string = None
    scores = []
    
    all_patches_negative_score: bool = True 
    correction_threshold = 0.0 # Threshold for considering a patch as potentially correct
    
    for patch_string, judgments in zip(patch_strings, judgments_aggregated):
        score = calculate_patch_score(patch_string, judgments, repo_path, validation_copy_count)
        scores.append(score)
        
        if score >= correction_threshold: 
            all_patches_negative_score = False 
            if score > best_score:
                best_score = score
                best_patch_string = patch_string

    if all_patches_negative_score or best_score < 1: 
        return scores, None
    else:
        return scores, best_patch_string 


def ensemble_patch_selection(patch_strings_list, judgments_aggregated_list):
    best_patch = None
    max_yes_votes = -1

    for patch_string, judgments in zip(patch_strings_list, judgments_aggregated_list):
        yes_votes = judgments.count(True)
        if yes_votes > max_yes_votes:
            max_yes_votes = yes_votes
            best_patch = patch_string

    return best_patch


# Predict function
def predict_inner(problem_statement: str, directory: str) -> Optional[str]:
    directory_string = stringify_directory(directory)

    selection_completion_texts, file_queries = get_selection_query(
        directory_string, problem_statement
    )

    file_content_strings: List[str] = [
        fetch_file_contents(file_query) for file_query in file_queries
    ]

    patch_completion_texts, patch_strings = get_patch_string(
        problem_statement, file_content_strings
    )

    verification_completion_texts_aggregated, judgments_aggregated = get_verification(
        problem_statement, file_content_strings, patch_strings, directory
    )

    scores, patch_string = choose_patch_string_optimized(
        patch_strings, judgments_aggregated, directory, VALIDATION_COPY_COUNT
    )

    if not os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        data = {
            "problem_statement": [problem_statement] * len(file_queries),
            "selection_completion_text": selection_completion_texts,
            "selection_completion_length": [
                count_tokens(completion_text)
                for completion_text in selection_completion_texts
            ],
            "file_query": file_queries,
            "file_content_string": file_content_strings,
            "patch_completion_text": patch_completion_texts,
            "patch_completion_length": [
                count_tokens(completion_text)
                for completion_text in patch_completion_texts
            ],
            "patch_string": patch_strings, 
        }

        for copy_idx in range(VALIDATION_COPY_COUNT):
            data[f"verification_completion_text_{copy_idx}"] = [
                completion_texts[copy_idx] if completion_texts else None
                for completion_texts in verification_completion_texts_aggregated
            ]
            data[f"verification_completion_length_{copy_idx}"] = [
                count_tokens(completion_texts[copy_idx]) if completion_texts else None
                for completion_texts in verification_completion_texts_aggregated
            ]
            data[f"judgment_{copy_idx}"] = [
                judgments[copy_idx] if judgments else None
                for judgments in judgments_aggregated
            ]

        data["judgment_count_true"] = [judgments.count(True) for judgments in judgments_aggregated] 
        data["score"] = scores

        elapsed_time_int = int(time.time() - start_time)  

        pd.DataFrame(data).to_csv(  
            f"{str(elapsed_time_int).zfill(5)}.csv", index=False  
        )

    return patch_string


import io
from typing import Optional, List

skip_prediction: bool = False


def predict(
    problem_statement: str,
    repo_archive: io.BytesIO,
    pip_packages_archive: io.BytesIO,
    env_setup_cmds_templates: List[str],
) -> Optional[str]:
    """Replace this function with your inference code.
    Args:
        problem_statement: The text of the git issue.
        repo_archive: A BytesIO buffer path with a .tar containing the codebase that must be patched. The gateway will make this directory available immediately before this function runs.
    """
    global skip_prediction
    if skip_prediction:
        return None

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

    if not os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        skip_prediction = True

    print("submitted patch_string")
    print(patch_string)

    if patch_string is None:
        return None

    return patch_string


import os
import zipfile

# !mkdir -p /kaggle/tmp/konwinski-prize-alt
os.makedirs("/kaggle/tmp/konwinski-prize-alt", exist_ok=True)

# !unzip -q -o /kaggle/input/konwinski-prize/data.a_zip -d /kaggle/tmp/konwinski-prize-alt/ 2>/dev/null || true
try:
    with zipfile.ZipFile("/kaggle/input/konwinski-prize/data.a_zip", "r") as zip_ref:
        zip_ref.extractall("/kaggle/tmp/konwinski-prize-alt/")
except:
    pass


import pandas as pd


def get_problem(problem_index: int) -> Tuple[str, str, io.BytesIO]:
    df = pd.read_parquet("/kaggle/tmp/konwinski-prize-alt/data/data.parquet")

    problem_statement: str = df["problem_statement"][problem_index]
    repo_path: str = (
        f"/kaggle/tmp/konwinski-prize-alt/data/repos/repo__{df['instance_id'][problem_index]}"
    )

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
    and not os.getenv("KAGGLE_IS_COMPETITION_RERUN")
    and patch_string is not None
):
    from kaggle_evaluation.konwinski_prize_gateway import UnitTestOutcome

    for result in results[1:]:
        if result.unit_test_outcome != UnitTestOutcome.PASSED:
            print(result.test_name)
            print(result.fail_description)


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



def calculate_score(
    n_correct: int,
    n_wrong: int,
    n_skipped: int,
    incorrect_score=-1,
    skip_score=-10**-4
) -> str:
    score = (n_correct + n_wrong * incorrect_score + n_skipped * skip_score) / (n_correct + n_skipped + n_wrong)
    if n_correct == 0:
        score = incorrect_score
    return f"{score:+.20f}"[:1+1+1+6].lstrip("+")


def calculate_results(score: str) -> list[tuple[int, int, int]]:
    assert type(score) == str
    possible_results = []
    for n_correct in range(72):
        for n_wrong in range(72 - n_correct):
            n_skipped = 71 - n_correct - n_wrong
            if score == calculate_score(n_correct, n_wrong, n_skipped):
                 possible_results.append((n_correct, n_wrong, n_skipped))
    return possible_results


for score in [
    "-0.309916",
    "-0.295836",
    "-0.126839",
    "-0.253590",
    "-0.112753",
    "-0.197254",
    "-0.873249",
    "-0.971830",
    "1.000000",
]:
    print(score, calculate_results(score), "\n")














