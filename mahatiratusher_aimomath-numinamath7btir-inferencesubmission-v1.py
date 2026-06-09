# !pip install timeout-decorator


import os
import re
import time
import polars as pl
from transformers import AutoModelForCausalLM, AutoTokenizer
import kaggle_evaluation.aimo_2_inference_server
from io import StringIO
import sys
import torch

# Global variables for lazy model loading
model = None
tokenizer = None
model_path = "/kaggle/input/math_numina_7b_tir/pytorch/default/1/NuminaMath-7B-TIR"

# Verify model files
required_files = ['config.json', 'tokenizer.json', 'model.safetensors.index.json']
weight_files = [f for f in os.listdir(model_path) if f.startswith('model-') and f.endswith('.safetensors')]
if not weight_files:
    raise FileNotFoundError(f"No model weight files (e.g., model-*.safetensors) found in {model_path}")
for file in required_files:
    if not os.path.exists(os.path.join(model_path, file)):
        raise FileNotFoundError(f"Required file {file} not found in {model_path}")
print(f"Found weight files: {weight_files}", flush=True)

# Verify test.csv path
test_csv_path = "/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv"
if not os.path.exists(test_csv_path):
    raise FileNotFoundError(f"Test file not found at {test_csv_path}")
print(f"Test file found at {test_csv_path}", flush=True)

# Inspect test.csv structure
try:
    test_df = pl.read_csv(test_csv_path)
    print("Test.csv preview:", flush=True)
    print(test_df.head(), flush=True)
    required_columns = ['id', 'problem']
    missing_columns = [col for col in required_columns if col not in test_df.columns]
    if missing_columns:
        raise ValueError(f"Test.csv is missing required columns: {missing_columns}")
    print(f"Test.csv contains required columns: {required_columns}", flush=True)
except Exception as e:
    print(f"Error reading test.csv: {e}", flush=True)
    raise

# Optimized prompt with clear instructions and a marker
PROMPT_TEMPLATE = """You are a skilled mathematician tasked with solving a challenging math problem. Provide a clear, step-by-step explanation of your reasoning without repeating the instructions. Avoid generating Python code unless it is necessary for complex calculations. Focus on solving the problem directly and concisely. Your final answer must be a single integer between 0 and 999, presented in a \\boxed{{}} format at the end of your explanation. For example, if the answer is 42, your explanation should end with \\boxed{{42}}.

Solve the following problem:
{}

### Solution:
"""

def load_model():
    """Load the model and tokenizer lazily."""
    global model, tokenizer
    if model is None or tokenizer is None:
        print("Loading model and tokenizer...", flush=True)
        start_time = time.time()
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="float16")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            # Explicitly set pad_token_id to eos_token_id to suppress warning
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                print(f"Set pad_token_id to eos_token_id: {tokenizer.pad_token_id}", flush=True)
            print(f"Model loaded in {time.time() - start_time:.2f} seconds", flush=True)
            # Clear GPU memory caches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"GPU memory after clearing cache: {torch.cuda.memory_reserved() / 1024**2:.2f} MB", flush=True)
        except Exception as e:
            print(f"Model loading failed: {type(e).__name__} - {str(e)}", flush=True)
            raise

def generate_solution_minimal(question, max_retries=3):
    """Minimal test version: Simulate a solution without model inference."""
    print(f"Simulating solution for: {question[:50]}...", flush=True)
    return "The answer is 0"  # Static response

def generate_solution_detailed(question, max_retries=3):
    """Generate a solution with detailed error logging."""
    print("Entering generate_solution_detailed", flush=True)
    load_model()
    print("Model loaded successfully", flush=True)
    prompt = PROMPT_TEMPLATE.format(question)
    print(f"Generated prompt: {prompt[:100]}...", flush=True)

    for attempt in range(max_retries):
        print(f"Attempt {attempt + 1} of {max_retries}", flush=True)
        try:
            print("Tokenizing input...", flush=True)
            inputs = tokenizer(prompt, return_tensors="pt")
            print("Tokenization complete", flush=True)
            print("Moving inputs to device...", flush=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            print("Inputs moved to device", flush=True)
            print("Checking GPU memory before generation...", flush=True)
            if torch.cuda.is_available():
                print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB", flush=True)
                print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB", flush=True)
            print("Generating output...", flush=True)
            output = model.generate(
                **inputs,
                max_length=300,  # Further reduced to avoid timeouts
                do_sample=True,
                temperature=0.5,  # Lowered for more deterministic responses
                top_p=0.9,  # Added for better focus
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id
            )
            print("Generation complete", flush=True)
            print("Decoding output...", flush=True)
            solution = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"Solution generated: {solution}", flush=True)  # Log full solution for debugging
            return solution
        except Exception as e:
            print(f"Generation failed on attempt {attempt + 1}: {type(e).__name__} - {str(e)}", flush=True)
            if attempt == max_retries - 1:
                print("Max retries reached, returning None", flush=True)
                return None
    return None

# Set the detailed version for testing
generate_solution = generate_solution_detailed

def extract_answer(solution):
    """Extract the final answer from the model's output with improved regex."""
    print("Extracting answer...", flush=True)
    if solution is None:
        print("Solution is None, returning default answer 0", flush=True)
        return 0
    
    # Split solution at the marker to ignore the prompt
    if "### Solution:" in solution:
        solution = solution.split("### Solution:")[1]
    
    # Improved regex to handle whitespace inside \boxed{}
    # Find all matches and take the last one
    matches = re.findall(r'\\boxed\{\s*(\d+)\s*\}', solution)
    if matches:
        answer = int(matches[-1])
        if 0 <= answer <= 999:
            print(f"Found boxed answer: {answer}", flush=True)
            return answer
    
    # Fallback: Look for numbers after ### Solution:, not in the problem statement
    numbers = re.findall(r'\d+', solution)
    if numbers:
        answer = int(numbers[-1]) % 1000
        print(f"Found last number as answer: {answer}", flush=True)
        return answer
    print("No valid answer found, returning default 0", flush=True)
    return 0

def solve_problem(question):
    """Solve a single problem and return the answer."""
    print(f"Solving problem: {question[:50]}...", flush=True)
    try:
        solution = generate_solution(question)
        answer = extract_answer(solution)
        print(f"Solved problem, answer: {answer}", flush=True)
        return answer
    except Exception as e:
        print(f"Error in solve_problem: {type(e).__name__} - {str(e)}", flush=True)
        return 0

def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame:
    """API-compatible predict function with robust input handling."""
    print("Received id_:", id_, flush=True)
    print("Received question:", question, flush=True)

    try:
        # Handle id_ as DataFrame or Series
        if isinstance(id_, pl.DataFrame):
            id_value = id_['id'][0] if 'id' in id_.columns else None
        elif isinstance(id_, pl.Series):
            id_value = id_[0] if len(id_) > 0 else None
        else:
            raise ValueError(f"Unexpected id_ type: {type(id_)}")

        # Handle question as DataFrame or Series
        if isinstance(question, pl.DataFrame):
            question_value = question['problem'][0] if 'problem' in question.columns else None
        elif isinstance(question, pl.Series):
            question_value = question[0] if len(question) > 0 else None
        else:
            raise ValueError(f"Unexpected question type: {type(question)}")

        if id_value is None or question_value is None:
            raise ValueError("Failed to extract id_ or question value")

    except Exception as e:
        print(f"Input processing error: {e}", flush=True)
        return pl.DataFrame({'id': ['unknown'], 'answer': [0]}, schema={'id': pl.Utf8, 'answer': pl.Int64})

    # Solve the problem
    answer = solve_problem(question_value)
    # Ensure answer is an integer and id_value is a string
    answer = int(answer)
    id_value = str(id_value)
    result = pl.DataFrame({'id': [id_value], 'answer': [answer]}, schema={'id': pl.Utf8, 'answer': pl.Int64})
    print("Returning result:", result, flush=True)
    print("Result dtypes:", result.dtypes, flush=True)
    return result

# Function to run the test script separately
def run_test():
    try:
        test_question = "What is 1 + 1?"
        print(f"Starting test with question: {test_question}", flush=True)
        solution = generate_solution_detailed(test_question)
        print(f"Test solution: {solution}", flush=True)
        answer = extract_answer(solution)
        print(f"Extracted answer: {answer}", flush=True)
    except Exception as e:
        print(f"Test script failed: {type(e).__name__} - {str(e)}", flush=True)

# Start inference server immediately
inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(predict)
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    try:
        # Process reference problems for validation
        test_df = pl.read_csv('/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv')
        for row in test_df.rows(named=True):
            id_, question = row['id'], row['problem']
            try:
                result = predict(pl.DataFrame({'id': [id_]}), pl.DataFrame({'problem': [question]}))
                print(f"ID: {result['id'][0]}, Predicted Answer: {result['answer'][0]}, True Answer: {row['answer']}", flush=True)
            except Exception as e:
                print(f"Error processing problem {id_}: {type(e).__name__} - {str(e)}", flush=True)
        # Run test.csv problems
        inference_server.run_local_gateway(('/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv',))
    except Exception as e:
        print(f"Gateway error: {e}", flush=True)
        print(f"Exception type: {type(e)}", flush=True)
        print(f"Exception details: {e.__dict__}", flush=True)

# Run the test script only if not in competition mode
if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    run_test()


start_time = time.time()
answer = solve_problem("Solve $4+x=4$ for $x$.")
print(f"Time taken: {time.time() - start_time} seconds")

