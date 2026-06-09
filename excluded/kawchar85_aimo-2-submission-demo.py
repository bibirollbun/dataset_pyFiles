import os
import re
import random
import pandas as pd
import polars as pl

import kaggle_evaluation.aimo_2_inference_server


def extract_boxed_text(text):
    pattern = r'oxed{(.*?)}'
    matches = re.findall(pattern, text)
    if not matches:
        return ""
    for match in matches[::-1]:
        if match != "":
            return match
    return ""


%%time

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/kaggle/input/qwen2.5/transformers/1.5b-instruct/1"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def predict_for_question(question: str) -> int:
    messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}, after taking modulo 1000."},
        {"role": "user", "content": question}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2048
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("-------")
    print(question)
    answer = extract_boxed_text(response)
    print("Model response:", answer)
    print("--------\n\n\n")
    
    if answer:
        try:
            return int(answer) % 1000
        except:
            return random.randint(0, 999)
    else:
        return random.randint(0, 999)


# The function should return a single integer between 0 and 999, inclusive.
# Each prediction (except the very first) must be returned within 30 minutes of the question being provided.
def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction."""
    # Unpack values
    id_ = id_.item(0)
    question = question.item(0)
    # Make a prediction
    answer = predict_for_question(question)
    return pl.DataFrame({'id': id_, 'answer': answer})


pd.read_csv(
    '/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv'
).drop('answer', axis=1).to_csv('reference.csv', index=False)


%%time

inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            'reference.csv',
        )
    )

