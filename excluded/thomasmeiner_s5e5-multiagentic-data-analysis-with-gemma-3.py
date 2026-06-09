!pip install fpdf -q
!pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3 -q --no-cache
#!pip install -U bitsandbytes accelerate -q
!pip install ai_analyst --find-links=file:/kaggle/input/kaggle-ai-analyst/ai_analyst-0.3.3-py3-none-any.whl -q

import numpy as np 
import pandas as pd 
import os

from ai_analyst.analysis_kit.analyse import analyse_data, _validate_pdf_requirements
from ai_analyst.analysis_kit.config import AnalysisConfig

from google import genai 
from kaggle_secrets import UserSecretsClient


DATA_PATH = "/kaggle/input/playground-series-s5e5/train.csv"
df      = pd.read_csv(DATA_PATH)
TARGET  = "Calories"


use_api = False # for free tier it will consume too many tokens probably

if not use_api:
    ai_cfg = AnalysisConfig(
        data_path=DATA_PATH,
        model_path="/kaggle/input/gemma-3/transformers/gemma-3-12b-it/1/",
        data_about="This dataset contains information about rainfall",
        target_column=TARGET,
        max_iterations=10 # how many LLM iterations shall the analysis have
    )

else:
    secret_label = "GEMINI_API_KEY"
    secret_value = UserSecretsClient().get_secret(secret_label)
    api_key = os.getenv("GEMINI_API_KEY", secret_value)
    
    ai_cfg = AnalysisConfig(
        use_api=True,
        api_key=api_key,
        api_model_id="gemma-3-12b-it",  # gemini-2.0-flash
        data_path=DATA_PATH,
        data_about="This dataset contains information about many calories were burned during a workout (target variable is Calories).",
        target_column=TARGET,
        max_iterations=3
    )


analyse_data(df, ai_cfg)

