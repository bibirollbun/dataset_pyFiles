# Libraries: Standard
import os

# Libraries: External
import pandas as pd
import polars as pl
import kaggle_evaluation.aimo_2_inference_server

# Packages: Standard
import random


def get_answer(que: str) -> int:
    return random.randint(0, 999)


question = "What is $1-1$?"

answer = get_answer(question)
print(f"The answer to the question: {answer}.")


def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    id_ = id_.item(0)
    question = question.item(0)
    
    prediction = get_answer(question)
    return pl.DataFrame({"id": id_, "answer": prediction})


inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            "/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv",
        )
    )

