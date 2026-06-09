import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")


import os
import pandas as pd
import numpy as np
import polars as pl

import kaggle_evaluation.aimo_2_inference_server


os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"



import time
start_time = time.time()
cutoff_time = start_time + (4 * 60 + 45) * 60
cutoff_times = [int(x) for x in np.linspace(cutoff_time, start_time + 60 * 60, 50 + 1)]


import warnings
warnings.simplefilter('ignore')

os.environ["CUDA_VISIBLE_DEVICES"]   = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"




df=pd.read_csv("/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv")
df_test=pd.read_csv("/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv")
sample_submission=pd.read_csv("/kaggle/input/ai-mathematical-olympiad-progress-prize-2/sample_submission.csv")



df.head(5)


test=df["problem"][0]
test


df_test.head(5)


# Define the model path
model_path = "/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-1.5b/2"


# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda()



MAX_NUM_SEQS = 128
MAX_MODEL_LEN = 8192 * 3 // 2


from vllm import LLM, SamplingParams
# Load the model
llm = LLM(model_path,dtype="float16",max_num_seqs=MAX_NUM_SEQS,   # Maximum number of sequences per iteration. Default is 256
    max_model_len=MAX_MODEL_LEN, # Model context length
    trust_remote_code=True,      # Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer
    tensor_parallel_size=1,      # The number of GPUs to use for distributed execution with tensor parallelism
    gpu_memory_utilization=0.95, # The ratio (between 0 and 1) of GPU memory to reserve for the model
    seed=2024,)


def format_prompt(problem_text):
    return (
        "Solve the following math problem exactly.\n"
        "Return ONLY the final numeric answer inside \\boxed{}.\n"
        "THIS IS NOT A TRICK QUESTION. DON'T OVERTHINK"
        "DO NOT include explanations, reasoning, or intermediate steps.\n"
        "DO NOT repeat or rephrase the question.\n"
        "DO NOT output anything except \\boxed{ANSWER}.\n"
        "Provide the final numeric answer inside \\boxed{} only \n\n"
        "Problem: " + problem_text
    )




 max_tokens = MAX_MODEL_LEN
if time.time() > cutoff_times[-1]:
    print("Speedrun")
    max_tokens = 2 * MAX_MODEL_LEN // 3


sampling_params = SamplingParams(
    temperature=1.0,      # Forces deterministic, precise answers
    top_p=1.0,           # Consider all likely tokens (not limiting)
    top_k=-1,            # No top-k restriction
    max_tokens=max_tokens,       # Small max tokens to prevent long reasoning
    repetition_penalty=1.1,  # Discourage repeated phrases
    ignore_eos=False,    # Let generation stop naturally
)


# def solve_math_problem_full_response(problem_text):
#     # Tokenize input
#     inputs = tokenizer(problem_text, return_tensors="pt").to("cuda") 
    
#     # Generate response
#     with torch.no_grad():
#         output = model.generate(**inputs, max_new_tokens=700,  # Increased max_new_tokens
#                         eos_token_id=tokenizer.eos_token_id,
#                         temperature=0.9,  # Higher temperature
#                         top_p=0.85,       # Adjusted top_p
#                         num_beams=6,       # More beams
#                         repetition_penalty=1.2) # Penalize repetitions     
    
#     # Decode the response
#     response = tokenizer.decode(output[0], skip_special_tokens=True)
    
#     return response


# question = "what is 10+5"
# answer = solve_math_problem_full_response(question)
# print("Generated Answer:", answer)


problem_text = "What is 10 * 5"
formatted_prompt = format_prompt(problem_text)
# Generate response
outputs = llm.generate(formatted_prompt,sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, \nGenerated text: {generated_text!r}")


def compute_final_answer(result):
    try:
        result = int(result)  # Convert to integer
    except ValueError:
        raise ValueError(f"Invalid input: {result} is not a number")

    # Apply modulo 1000 only if result is greater than 999 or negative
    if result > 999 or result < 0:
        final_answer = result % 1000
        # Ensure positive modulo result
        final_answer = final_answer if final_answer >= 0 else final_answer + 1000
    else:
        final_answer = result

    return final_answer  # Returning as an integer



compute_final_answer(1023)


import re

def extract_boxed_text(text: str) -> str:
    pattern = r'boxed{(.*?)}'
    matches = re.findall(pattern, text)
    if not matches:
        return ""
    for match in matches[::-1]:
        if match != "":
            return match
    return ""


final_answer= extract_boxed_text(generated_text)
print("Final Generated Answer:",final_answer )



compute_final_answer(final_answer)


# import re

# def extract_boxed_text(text: str) -> str:
#     pattern = r'boxed{(.*?)}'
#     matches = re.findall(pattern, text)
#     if not matches:
#         return ""
#     for match in matches[::-1]:
#         if match != "":
#             return match
#     return ""

# def solve_math_problem(problem_text):
#     formatted_prompt = (
#     "Solve the following math problem exactly. Do not approximate. "
#     "Return ONLY the correct final numeric answer inside \\boxed{}. "
#     "Do NOT include explanations, reasoning, or intermediate steps. "
#     "If the answer is negative, still use \\boxed{}. "
#     "Ensure the answer is 100% correct before returning. "
#     "Problem: " + problem_text
# )



#     inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

#     with torch.no_grad():
#         output = model.generate(**inputs, max_new_tokens=700,  
#                         eos_token_id=tokenizer.eos_token_id,
#                         temperature=0.9, 
#                         top_p=0.85,      
#                         num_beams=6,     
#                         repetition_penalty=1.2)
#     response = tokenizer.decode(output[0], skip_special_tokens=True).strip()

#     return extract_boxed_text(response)



# question = "what is 10+5"
# answer = solve_math_problem(question)
# print("Generated Answer:", answer)


import re

def solve_math_problem_vllm(llm, problem_text, sampling_params):
    """
    Generates a response using the LLM, extracts the final boxed answer, and returns it.
    
    Args:
        llm: The language model instance.
        problem_text (str): The math problem to solve.
        sampling_params: Parameters for the model generation.

    Returns:
        str: Extracted final numeric answer from \boxed{}.
    """

    # Format the prompt
    formatted_prompt = format_prompt(problem_text)

    # Generate response from the model
    outputs = llm.generate(formatted_prompt, sampling_params)

    # Extract generated text
    generated_text = outputs[0].outputs[0].text if outputs else ""

    # Extract the boxed answer
    final_answer = extract_boxed_text(generated_text)
    # Modulo 
    prediction_modulo=compute_final_answer(final_answer)  #Take the Modulo 

    # Print results
    print(f"Prompt: {formatted_prompt!r}\n")
    print(f"Generated text: {generated_text!r}\n")
    print(f"Final Generated Answer: {prediction_modulo}")
    return prediction_modulo



df_test["problem"][0]


problem_text = df_test["problem"][0]
final_answer = solve_math_problem_vllm(llm, problem_text, sampling_params)




# Function to process all rows and generate predictions using Pandas
def generate_submission(df_test):
    """Generate predictions for all rows in df_test and save to submission.csv."""
    results = []

    for _, row in df_test.iterrows():  # Iterate through Pandas DataFrame rows
        id_ = row["id"]
        question = row["problem"]

        print(f"Processing ID: {id_}, Question: {question}")

        # Generate prediction using LLM
        prediction = solve_math_problem_vllm(llm, question, sampling_params)

        # Append result to list
        results.append({"id": id_, "answer": prediction})

    # Convert results to Pandas DataFrame
    submission_df = pd.DataFrame(results)

    # Save as CSV
    submission_df.to_csv("submission.csv", index=False)
    print("Submission saved as submission.csv")

# Call the function
# generate_submission(df_test)



generate_submission(df_test)


# Replace this function with your inference code.
# The function should return a single integer between 0 and 999, inclusive.
# Each prediction (except the very first) must be returned within 30 minutes of the question being provided.
import polars as pl
import pandas as pd

def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction."""
    # Unpack values
    id_ = id_.item(0)
    print("------")
    print(id_)
    
    question = question.item(0)
    print(question)
    # Generate prediction using the model
    prediction = solve_math_problem_vllm(llm, question, sampling_params)  # Get boxed answer
        
    print("------\n\n\n")

    print("Final Predicted Answer is",prediction)
         
    return pl.DataFrame({'id': [id_], 'answer': prediction})



sample_submission.head(5)


df_test.head(5)


pd.read_csv(
    '/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv'
).drop('answer', axis=1).to_csv('reference.csv', index=False)


inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            "/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv",
            # "reference.csv",
        )
    )




