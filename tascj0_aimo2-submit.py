!cp -r /kaggle/input/aimo-packages/fire-0.7.0/fire-0.7.0 fire-0.7.0
!cd fire-0.7.0 && pip install .


!pip install lmdeploy --no-index --find-links='/kaggle/input/aimo-packages'


import os

import pandas as pd
import polars as pl

import kaggle_evaluation.aimo_2_inference_server


import math
import re
import time
from collections import Counter
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig


class Inferencer:
    def __init__(
        self,
        model_path,
        model_format="awq",
        quant_policy=8,
        tp=4,
        max_new_tokens=16384,
        temperature=0.8,
        top_p=0.95,
    ):
        self.model_path = model_path
        self.quant_policy = quant_policy
        self.tp = tp
        self.max_new_tokens = max_new_tokens

        self.backend_config = TurbomindEngineConfig(
            tp=tp,
            model_format=model_format,
            quant_policy=quant_policy,
        )
        self.pipe = pipeline(model_path, backend_config=self.backend_config)
        self.gen_config = GenerationConfig(
            top_p=top_p,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            random_seed=20250401,
        )

        self.base_budget = 60 * 5.5  # 5.5 minutes
        self.budget = 370  # start with N=6

    def get_num_samples(self):
        estimated = (self.budget - 190) / 30
        ret = min(15, math.floor(estimated))
        print(f"Budget: {self.budget} -> N: {ret}")
        return ret

    def format_prompt(self, problem):
        prompt = (
            "<｜begin▁of▁sentence｜>Please reason step by step, take modulo 1000 of the answer and put the result within \\boxed{}.<｜User｜>"
            + problem
            + "<｜Assistant｜>"
        )
        return prompt

    def inference(self, problem):
        start = time.time()
        prompt = self.format_prompt(problem)
        repeats = self.get_num_samples()
        prompts = [prompt] * repeats
        responses = self.pipe(prompts, self.gen_config)
        end = time.time()
        duration = end - start
        print(f"Inference took {duration} seconds")

        budget_left = max(0, self.budget - duration)
        self.budget = self.base_budget + budget_left

        return self.parse_responses(responses)

    def parse_response(self, response):
        answer = re.findall(r"\\boxed{(.*)}", response.text)
        if answer:
            ret = answer[-1]
            if ret.isdigit():
                return int(ret)
            else:
                return None
        else:
            return None

    def parse_responses(self, responses):
        answers = [self.parse_response(response) for response in responses]
        # all None
        if all(answer is None for answer in answers):
            return 0
        counter = Counter(answers)
        print("Answers:", counter)
        counter[None] = -1
        sorted_answers = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        return sorted_answers[0][0] % 1000



inferencer = Inferencer("/kaggle/input/aimo-checkpoints", tp=4)


# Replace this function with your inference code.
# The function should return a single integer between 0 and 999, inclusive.
# Each prediction (except the very first) must be returned within 30 minutes of the question being provided.
def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction."""
    # Unpack values
    id_ = id_.item(0)
    question = question.item(0)
    print("------")
    print(id_)
    print(question)
    print("------\n\n\n")

    # Make a prediction
    answer = inferencer.inference(question)
    return pl.DataFrame({'id': id_, 'answer': answer})



pd.read_csv(
    '/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv'
).drop('answer', axis=1).to_csv('reference.csv', index=False)


inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            '/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv',
            # "/kaggle/working/reference.csv",
        )
    )




