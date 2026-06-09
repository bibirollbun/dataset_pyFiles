# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



!pip install -U /kaggle/input/accelerate-0-29-3/accelerate-0.29.3-py3-none-any.whl -qq
!pip install -U /kaggle/input/bitsandbytes-0-43-1/bitsandbytes-0.43.1-py3-none-manylinux_2_24_x86_64.whl -qq


import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import re
import sys
import subprocess
import math
import random
from collections import defaultdict, Counter
import torch
import transformers
import accelerate
import time

# Environment Configuration
COMPETITION_MODE = os.getenv('KAGGLE_IS_COMPETITION_RERUN', False)

# Training Environment Simulator
class TrainingEnvironment:
    """Simulates the competition environment for local testing"""
    
    def __init__(self, shuffle_data=False):
        self.shuffle_enabled = shuffle_data
        self.data_frame = pd.read_csv('/kaggle/input/ai-mathematical-olympiad-prize/train.csv')
        self.data_frame['ground_truth'] = self.data_frame['answer']
        self.data_frame['answer'] = -1
        
        if self.shuffle_enabled:
            self.data_frame = self.data_frame.reset_index().sample(frac=1).reset_index(drop=True)
        
        self.prediction_ready = True
        self.current_index = 0
        self.total_length = len(self.data_frame)
    
    def get_test_iterator(self):
        """Generator that yields test cases"""
        while self.current_index < self.total_length:
            if self.prediction_ready:
                self.prediction_ready = False
                test_data = self.data_frame.loc[[self.current_index]][['id','problem']]
                answer_template = self.data_frame.loc[[self.current_index]][['id','answer']]
                yield test_data, answer_template
            else:
                print("Must call predict() before continuing with test iteration")
                yield None 
                
    def submit_prediction(self, prediction):
        """Submit prediction for current test case"""
        self.data_frame.loc[self.current_index, 'answer'] = prediction['answer'].values[0]
        self.prediction_ready = True
        self.current_index += 1

# Initialize environment based on mode
if not COMPETITION_MODE:
    environment = TrainingEnvironment(shuffle_data=True)
    test_iterator = environment.get_test_iterator()
else:
    import aimo
    environment = aimo.make_env()
    test_iterator = environment.iter_test()

# Model Configuration Parameters
USE_QUANTIZATION = False
ENABLE_PAST_KEYS = True
RANDOM_SEED = 42
MODEL_DIRECTORY = "/kaggle/input/deepseek-math"
COMPUTE_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GENERATION_ATTEMPTS = 19 if COMPETITION_MODE else 4
MAX_TOKEN_GENERATION = 2048 if COMPETITION_MODE else 512
EXECUTION_TIME_LIMIT = 31500 if COMPETITION_MODE else 1

# Set random seeds for reproducibility
transformers.set_seed(RANDOM_SEED)
torch.backends.cuda.enable_mem_efficient_sdp(False)

# GPU Memory Distribution Configuration
GPU_LAYER_MAPPING = {
    'model.embed_tokens': 0,
    **{f'model.layers.{i}': 0 if i < 18 else 1 for i in range(32)},
    'model.norm': 1,
    'lm_head': 1
}

# Generation Parameters
SAMPLING_TEMPERATURE = [0.9, 0.9]  # [general_temp, code_temp]
TOP_P_VALUES = [1.0, 1.0]  # [general_top_p, code_top_p]

# Custom Stopping Criteria Implementation
class CustomStoppingCriteria(transformers.StoppingCriteria):
    """Custom stopping criteria for text generation"""
    
    def __init__(self, termination_sequences=[], min_encounters=1):
        super().__init__()
        self.termination_tokens = [seq.to(COMPUTE_DEVICE) for seq in termination_sequences]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop_sequence in self.termination_tokens:
            recent_tokens = input_ids[0][-len(stop_sequence):]
            if torch.all(torch.eq(stop_sequence, recent_tokens)):
                return True
        return False

# Main Language Model System
class MathematicalProblemSolver:
    """Advanced mathematical problem solver using language models"""
    
    def __init__(self, model_path, device_mapping, temperature_config, top_p_config, prompt_templates):
        # Initialize core components
        self.language_model, self.text_tokenizer = self._setup_language_model(model_path, device_mapping)
        
        # Define stopping sequences for generation
        self.termination_phrases = ["```output", "```python", "```\nOutput", ")\n```", "``````output"]
        self.termination_token_ids = [
            self.text_tokenizer(phrase, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() 
            for phrase in self.termination_phrases
        ]
        self.stopping_criteria = transformers.StoppingCriteriaList([
            CustomStoppingCriteria(termination_sequences=self.termination_token_ids)
        ])
        
        # Store configuration
        self.prompt_templates = prompt_templates
        self.base_temperature = temperature_config[0]
        self.base_top_p = top_p_config[0]
        self.code_temperature = temperature_config[1]
        self.code_top_p = top_p_config[1]
        
        # Initialize tracking variables
        self._reset_tracking_variables()
        
    def _reset_tracking_variables(self):
        """Reset all tracking variables for new session"""
        self.solution_repository = {}
        self.answer_repository = {}
        self.optimal_solutions = {}
        self.generation_outputs = {}
        self.approach_statistics = {}
        self.baseline_counts = (2, 3)
        self.current_problem_number = 0
        self.tokens_generated = 0
        self.execution_error = None
        self.error_occurrence_count = 0
        self.code_execution_result = -1

    def _setup_language_model(self, model_path, device_mapping):
        """Initialize and configure the language model"""
        model_config = transformers.AutoConfig.from_pretrained(model_path)
        model_config.gradient_checkpointing = True
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

        if USE_QUANTIZATION:
            quantization_settings = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="sequential",
                torch_dtype="auto",
                trust_remote_code=True,
                quantization_config=quantization_settings,
                config=model_config
            )
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_mapping,
                torch_dtype="auto",
                trust_remote_code=True,
                config=model_config
            )
            
        return model, tokenizer

    def solve_mathematical_problem(self, problem_statement):
        """Main method to solve a mathematical problem"""
        self.current_problem_number += 1
        execution_start_time = time.time()
        elapsed_time = execution_start_time - NOTEBOOK_START_TIME if 'NOTEBOOK_START_TIME' in globals() else 0
    
        # Check time limit
        if elapsed_time > EXECUTION_TIME_LIMIT:
            return 0

        for attempt in tqdm(range(GENERATION_ATTEMPTS)):
            print(f"\n\nPROBLEM {self.current_problem_number} - ATTEMPT {attempt} - ELAPSED: {elapsed_time:.0f}s")
            
            # Check if we already have a good solution
            current_best, best_frequency = self.optimal_solutions.get(self.current_problem_number, (-1, -1))
            if best_frequency > np.sqrt(attempt):
                print("SKIPPING - OPTIMAL SOLUTION ALREADY FOUND")
                continue

            # Retrieve existing data
            all_outputs = self.generation_outputs.get(self.current_problem_number, [])
            text_solutions, code_solutions = self.approach_statistics.get(
                self.current_problem_number, self.baseline_counts
            )
            
            # Clear memory and reset state
            for _ in range(5):
                self._clear_memory()
                time.sleep(0.2)

            try:
                self._reset_generation_state()
                
                # Select approach based on historical performance
                approach_weights = np.array([text_solutions, code_solutions])
                selected_template = np.random.choice(
                    self.prompt_templates, 1, p=approach_weights/approach_weights.sum()
                )[0]
                
                # Generate solution
                result_from_text, result_from_code = self._generate_solution(
                    problem_statement, selected_template
                )
                
                # Update statistics
                if result_from_code != -1:
                    all_outputs.append(result_from_code)
                    code_solutions += 1

                if result_from_text != -1:
                    all_outputs.append(result_from_text)
                    text_solutions += 1

                # Check for consensus
                if len(all_outputs) > 0:
                    frequency_analysis = Counter(all_outputs).most_common()
                    print(f"Answer frequency: {frequency_analysis}")
                    
                    if frequency_analysis[0][1] > best_frequency:
                        print("IMPROVED SOLUTION FOUND!")
                        current_best = frequency_analysis[0][0]
                        best_frequency = frequency_analysis[0][1]
                        
                    if frequency_analysis[0][1] > 5:
                        print("CONSENSUS REACHED!")
                        break

                # Update repositories
                self._update_repositories(
                    all_outputs, text_solutions, code_solutions, 
                    current_best, best_frequency, result_from_text, result_from_code
                )
                
            except Exception as error:
                print(f"Generation error: {error}")
                result_from_text, result_from_code = -1, -1

        return self.optimal_solutions[self.current_problem_number][0]

    def _reset_generation_state(self):
        """Reset state for new generation attempt"""
        self.tokens_generated = 0
        self.execution_error = None
        self.error_occurrence_count = 0
        self.code_execution_result = -1

    def _generate_solution(self, problem_statement, template):
        """Generate solution using the language model"""
        # Prepare initial prompt
        formatted_prompt = template.format(problem_statement, "{}")
        full_prompt = f"User: {formatted_prompt}"
        original_prompt_length = len(full_prompt)
        
        print(f"Prompt: {full_prompt}\n")

        # Tokenize and generate
        model_input = self.text_tokenizer(full_prompt, return_tensors='pt').to(self.language_model.device)
        prompt_tokens = len(model_input['input_ids'][0])

        generation_result = self.language_model.generate(
            **model_input,
            max_new_tokens=MAX_TOKEN_GENERATION - self.tokens_generated,
            return_dict_in_generate=ENABLE_PAST_KEYS,
            do_sample=True,
            temperature=self.base_temperature,
            top_p=self.base_top_p,
            num_return_sequences=1,
            stopping_criteria=self.stopping_criteria
        )

        # Process generation result
        output_tokens = generation_result.sequences[0] if ENABLE_PAST_KEYS else generation_result[0]
        decoded_response = self.text_tokenizer.decode(output_tokens, skip_special_tokens=True)
        print(f"Initial response: {decoded_response[original_prompt_length:]}\n")
        
        # Handle iterative code execution
        decoded_response, accumulated_code = self._handle_code_execution_loop(
            decoded_response, generation_result, prompt_tokens, original_prompt_length
        )
        
        # Extract final results
        raw_generation = self.text_tokenizer.decode(
            output_tokens[prompt_tokens:], skip_special_tokens=True
        )
        text_result = self._extract_text_answer(raw_generation)
        
        # Process final code result
        try:
            self.code_execution_result = round(float(eval(str(self.code_execution_result)))) % 1000
        except Exception as error:
            print(f"Final evaluation error: {error}")
            self.code_execution_result = -1
            
        return text_result, self.code_execution_result

    def _handle_code_execution_loop(self, response, generation_result, prompt_tokens, original_length):
        """Handle iterative code execution and generation"""
        accumulated_code = ""
        current_response = response
        original_length_tracker = original_length
        
        # Check for stopping conditions
        has_stop_word = any(current_response.endswith(stop) for stop in self.termination_phrases)
        
        while has_stop_word and self.tokens_generated < MAX_TOKEN_GENERATION:
            # Determine generation parameters based on context
            if current_response.endswith("```python"):
                temp, top_p = self.code_temperature, self.code_top_p
                prompt = current_response
            else:
                temp, top_p = self.base_temperature, self.base_top_p
                
                # Execute code if present
                try:
                    code_block = self._extract_code_block(current_response)
                    if code_block:
                        accumulated_code += code_block
                        execution_output, success = self._execute_code_safely(accumulated_code)
                        print(f'Code execution result: {execution_output}')
                        
                        # Handle repeated errors
                        if self.execution_error == execution_output:
                            self.error_occurrence_count += 1
                        else:
                            self.execution_error = execution_output
                            self.error_occurrence_count = 0
                            
                        if not success:
                            accumulated_code = accumulated_code[:-len(code_block)]
                            if self.error_occurrence_count >= 1:
                                print("REPEATED EXECUTION ERRORS - STOPPING")
                                break
                        
                        # Format prompt with execution result
                        if execution_output != -1:
                            if current_response.endswith(")\n```"):
                                prompt = f"{current_response}```output\n{execution_output}\n```\n"
                            else:
                                prompt = f"{current_response}\n{execution_output}\n```\n"
                        else:
                            prompt = current_response
                            accumulated_code = ""
                    else:
                        prompt = current_response
                        
                except Exception as error:
                    print(f'Code parsing error: {error}')
                    self.code_execution_result = -1
                    prompt = current_response
            
            # Continue generation
            current_response, generation_result = self._continue_generation(
                prompt, generation_result, temp, top_p, original_length_tracker
            )
            original_length_tracker += len(current_response[original_length_tracker:])
            
            # Update stopping condition
            has_stop_word = any(current_response.endswith(stop) for stop in self.termination_phrases)
            
        return current_response, accumulated_code

    def _extract_code_block(self, text):
        """Extract Python code block from text"""
        if text.endswith("``````output"):
            return text.split('```python')[-1].split("``````")[0]
        else:
            return text.split('```python')[-1].split("```")[0]

    def _continue_generation(self, prompt, previous_generation, temperature, top_p, original_length):
        """Continue text generation from current state"""
        model_input = self.text_tokenizer(prompt, return_tensors='pt').to(self.language_model.device)
        self.tokens_generated = len(model_input['input_ids'][0]) - len(model_input['input_ids'][0])
        
        past_key_values = previous_generation.past_key_values if ENABLE_PAST_KEYS else None
        
        generation_result = self.language_model.generate(
            **model_input,
            max_new_tokens=MAX_TOKEN_GENERATION - self.tokens_generated,
            return_dict_in_generate=ENABLE_PAST_KEYS,
            past_key_values=past_key_values,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1,
            stopping_criteria=self.stopping_criteria
        )
        
        output_tokens = generation_result.sequences[0] if ENABLE_PAST_KEYS else generation_result[0]
        decoded_output = self.text_tokenizer.decode(output_tokens, skip_special_tokens=True)
        print(f"\nContinued generation:\n{decoded_output[original_length:]}\n")
        
        return decoded_output, generation_result

    def _execute_code_safely(self, code_content):
        """Safely execute Python code with error handling"""
        # Preprocess code for symbolic computation
        processed_code = re.sub(r"symbols\([^)]+\)", self._fix_symbols_syntax, code_content)
        processed_code = processed_code.replace('\n', '\n    ')
        
        # Wrap in try-except block
        safe_code = f"""
try:
    from sympy import *
    {processed_code}
except Exception as e:
    print(e)
    print('EXECUTION_FAILED')
"""
        
        # Write and execute code
        with open('temp_code.py', 'w') as code_file:
            code_file.write(safe_code)

        execution_command = f'timeout 7 {sys.executable} temp_code.py'
        try:
            shell_result = subprocess.check_output(execution_command, shell=True).decode('utf8')
            final_output = self._get_last_line(shell_result, -1)
            print(shell_result)
            
            if final_output == 'EXECUTION_FAILED':
                execution_success = False
                error_message = self._get_last_line(shell_result, -2)
                if "not defined" in error_message:
                    error_message += '\nCheck formatting and imports'
                return error_message, execution_success
            else:
                execution_success = True
                return final_output, execution_success
                
        except Exception as error:
            print(f'Shell execution error: {error}')
            return -1, False

    def _fix_symbols_syntax(self, match):
        """Fix SymPy symbols syntax"""
        if "real" not in match.group():
            return f"{match.group()[:-1]}, real=True)"
        else:
            return f"{match.group()[:-1]})"

    def _get_last_line(self, text, line_index):
        """Get specific line from text output"""
        lines = text.strip().split('\n')
        return lines[line_index] if lines else ""

    def _extract_text_answer(self, text_output):
        """Extract numerical answer from text output"""
        try:
            # Look for boxed answers
            boxed_matches = re.findall(r'\\boxed\{(\d+)\}', text_output)
            print(f'Boxed answers found: {boxed_matches}')
            
            if boxed_matches:
                result = boxed_matches[-1]
            else:
                result = self._parse_number_naively(text_output)
            
            print(f'Final extracted answer: {result}')
            
            if not result or len(str(result)) == 0:
                return -1
            else:
                return round(float(eval(str(result)))) % 1000
                
        except Exception as error:
            print(f'Text parsing error: {error}')
            return -1

    def _parse_number_naively(self, text):
        """Naive number extraction from end of text"""
        digits = []
        started = False
        finished = False
        
        for char in reversed(text):
            if char in '0123456789' and not finished:
                started = True
                digits.append(char)
            elif started:
                finished = True
                
        return ''.join(reversed(digits))

    def _update_repositories(self, outputs, text_count, code_count, best_answer, best_count, text_result, code_result):
        """Update all tracking repositories"""
        self.optimal_solutions[self.current_problem_number] = (best_answer, best_count)
        self.approach_statistics[self.current_problem_number] = (text_count, code_count)
        self.generation_outputs[self.current_problem_number] = outputs
        
        # Update individual result tracking
        if self.current_problem_number not in self.solution_repository:
            self.solution_repository[self.current_problem_number] = []
            self.answer_repository[self.current_problem_number] = []
            
        self.solution_repository[self.current_problem_number].append(text_result)
        self.answer_repository[self.current_problem_number].append(code_result)
        
        print(f"Code solutions: {code_count - self.baseline_counts[1]}, "
              f"Text solutions: {text_count - self.baseline_counts[0]}")

    def _clear_memory(self):
        """Clear GPU memory and run garbage collection"""
        torch.cuda.empty_cache()
        gc.collect()

# Problem-Solving Prompt Templates
STRUCTURED_CODING_PROMPT = """Below is a mathematical problem requiring a positive numerical solution:
"{}"
Create a systematic SymPy-based solution by outlining each computational step and required functions. 
Provide clear instructions and comprehensive code with comments. The final answer must be a positive integer.
Write a complete script implementing all steps and print the result. Output the final answer in \\boxed{{}}.

Solution approach:"""

CHAIN_OF_THOUGHT_PROMPT = """Below is a mathematical problem requiring a positive numerical answer:
"{}"
Analyze this problem systematically and develop a step-by-step programmatic solution. 
Output the final numerical answer in \\boxed{{}}.

Analysis:"""

# Initialize the mathematical problem solver
NOTEBOOK_START_TIME = time.time()
template_options = [STRUCTURED_CODING_PROMPT, CHAIN_OF_THOUGHT_PROMPT]
problem_solver = MathematicalProblemSolver(
    MODEL_DIRECTORY, GPU_LAYER_MAPPING, SAMPLING_TEMPERATURE, TOP_P_VALUES, template_options
)

# Main execution loop
print("Mathematical Problem Solver initialized successfully!")
print("Special tokens loaded and model ready for inference.")

# Process test cases
for test_case, submission_template in test_iterator:
    predicted_answer = problem_solver.solve_mathematical_problem(test_case['problem'].values[0])
    submission_template['answer'] = predicted_answer
    environment.submit_prediction(submission_template)
    print(f"Test case: {test_case}")
    print(f"Submission: {submission_template}\n")


import subprocess
import sys

# Create temporary Python script
script_filename = 'temp_script.py'
test_content = "print('execution completed')"

# Write content to file
with open(script_filename, 'w') as script_file:
    script_file.write(test_content)

# Execute with timeout protection
execution_command = f'timeout 7 {sys.executable} {script_filename}'
try:
    execution_result = subprocess.check_output(execution_command, shell=True).decode('utf8')
    print(execution_result)
except Exception:
    # Silent error handling
    pass




