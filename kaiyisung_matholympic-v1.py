# from datasets import load_dataset

# ds = load_dataset("AI-MO/NuminaMath-CoT")


from vllm import LLM, SamplingParams
import time
import torch
import os


cutoff_time = time.time() + (4 * 60 + 45) * 60
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

llm = LLM(model='/kaggle/input/sky-t1-32b-preview/transformers/default/3',
      trust_remote_code=True,
      tensor_parallel_size=4,
      # dtype=torch.bfloat16,
      max_model_len=32768,#4096*10,  
      gpu_memory_utilization=0.97,
      # max_seq_len_to_capture=9800,
      )

tokenizer = llm.get_tokenizer()


sampling_params = {
    "max_tokens": 23000,
    "top_p":0.6,
    "top_k": 20,
    "temperature": 0.8,
    "repetition_penalty": 1.0,
    "n": 4,
}
sampling_params = SamplingParams(**sampling_params)


def set_prompt(question):
    CoT = ["1.You can write a Python 3 snippet to handle complicated calculation as your solution.",
       "2.The answer to each problem is a non-negative integer.",
          "3.Write a python 3 code for verification if possible. If not, just say not."]
    SYSTEMPROMPT = "You are the smartest mathematician in the world and would solve a Olympiad math question. Please follow the following instructions.\
    1.Python 3 Formalization: Write Python 3 code. Use 'print' statement in the end so that you can get the result of computation.\
    2.Final solution subimission: Please provide a detailed explanation and present the final numeral answer in \\boxed{}.\"
    HUMANPROMPT = f'Answer the following question:\n{question}\nInstructions:\n{CoT[2]}'
    
    prompts = [
        {"role": "system", "content": SYSTEMPROMPT},
        {"role": "user", "content": HUMANPROMPT}
    ]
    processed_prompts = tokenizer.apply_chat_template(
        prompts, tokenize=False, add_generation_prompt=True
    )
    return processed_prompts


def generate_finalSolution(prompt: str):
    import pandas as pd
    import re
    import threading

    data = []

    def extract_answer_in_box(text):
        pattern = r"(?:\\\[ *\\boxed\{(.*?)\} *\\\])|\\boxed\{(.*?)\}"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip() if match.group(1) else match.group(2).strip()
        return None


    def extract_code_blocks(text):
        patterns = [
            r"<code>(.*?)<\/code>",
            r"```python(.*?)```",
        ]

        code_blocks = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            code_blocks.extend([match.strip() for match in matches])

        return code_blocks

    def execute_code(code, timeout=20):
        import io
        import sys

        def run_code():
            nonlocal output, error
            old_stdout = sys.stdout
            new_stdout = io.StringIO()
            sys.stdout = new_stdout

            try:
                exec(code, {})
                output = new_stdout.getvalue().strip()
            except Exception as e:
                error = str(e)
            finally:
                sys.stdout = old_stdout

        output = None
        error = None
        code_thread = threading.Thread(target=run_code)

        code_thread.start()
        code_thread.join(timeout)

        if code_thread.is_alive():
            return "Error: Execution timed out"
        elif error:
            return f"Error: {error}"
        else:
            return output

    def process_text_and_execute(text):
        code_blocks = extract_code_blocks(text)
        result_code = None
        for idx, code in enumerate(code_blocks):
            result_code = execute_code(code)

        return result_code

    def generate_with_timeout(prompt, timeout):
        responses = []

        def run_llm():
            nonlocal responses
            try:
                responses = llm.generate(prompt, sampling_params=sampling_params)
            except Exception as e:
                responses = None

        llm_thread = threading.Thread(target=run_llm)
        llm_thread.start()
        llm_thread.join(timeout)

        if llm_thread.is_alive():
            return None
        return responses

    responses = generate_with_timeout(prompt, 1200)  # 1200-second timeout

    if responses is None:
        data.append({"Final_solution": 369, "Output": 369, "Code_blocks": 369, "Code_result": 369})
        df = pd.DataFrame(data)
        return df

    for idx, response in enumerate(responses):
        for i in range(len(response.outputs)):
            final_solution = extract_answer_in_box(response.outputs[i].text)
            code_blocks = extract_code_blocks(response.outputs[i].text)
            code_result = process_text_and_execute(response.outputs[i].text)
            data.append({"Final_solution": final_solution, "Output": response.outputs[i].text, "Code_blocks": code_blocks, "Code_result": code_result})

    df = pd.DataFrame(data)
    return df



def majority_vote(df):
    from collections import Counter
    import numpy as np
    import re

    def format_number(number) -> int:
        try:
            if number is None:
                return None
            if isinstance(number, str):
                match = re.search(r"[-+]?\d*\.?\d+", number)
                if not match:
                    return None
                number = float(match.group())
            number = float(number)
            if number < 0:
                return None
            number = int(number)
            return number % 1000
        except (ValueError, TypeError):
            return None

    # Apply formatting to the relevant columns
    df['Final_solution'] = df['Final_solution'].apply(format_number)
    df['Code_result'] = df['Code_result'].apply(format_number)

    # Flatten results and exclude None or NaN values
    results = df[['Final_solution', 'Code_result']].values.flatten()
    results = [item for item in results if item is not None and not np.isnan(item)]

    if not results:
        raise ValueError("No valid numbers to compute the majority vote.")

    # Majority vote
    vote_counts = Counter(results)
    majority_vote = vote_counts.most_common(1)[0][0]
    return int(majority_vote)


# prompt = set_prompt("For a positive integer $n$, let $S(n)$ denote the sum of the digits of $n$ in base 10. Compute $S(S(1)+S(2)+\cdots+S(N))$ with $N=10^{100}-2$.")
# result_df = generate_finalSolution(prompt)
# prediction = majority_vote(result_df)
# display(result_df)
# display(prediction)


# for item in result_df["Output"]:
#     display("----------------------------")
#     display(item)


# import pandas as pd
# temp = pd.read_csv("/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv")
# temp = temp.drop(columns = ['answer'])
# temp.to_csv('reference_clean.csv', index = False)


import pandas as pd
import polars as pl
import time
import kaggle_evaluation.aimo_2_inference_server

# Each prediction (except the very first) must be returned within 30 minutes of the question being provided.
def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction."""
    # Unpack values
    id_ = id_.item(0)
    question = question.item(0)

    if question is None or question == "":
        return pl.DataFrame({'id': id_, 'answer': 301})

    if time.time() > cutoff_time:
        display(302)
        return pl.DataFrame({'id': id_, 'answer': 302})

    try:
    # Make a prediction
        display(question)
        prompt = set_prompt(question)
        result_df = generate_finalSolution(prompt)
        prediction = majority_vote(result_df)
        display(result_df)
        display(prediction)
        return pl.DataFrame({'id': id_, 'answer': prediction})
    except Exception as e:
        display("303")
        return pl.DataFrame({'id': id_, 'answer': 303})


inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            '/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv',
            # '/kaggle/working/reference_clean.csv',
        )
    )

