# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import torch
import pandas as pd
import polars as pl
from transformers import AutoModelForCausalLM, AutoTokenizer
import kaggle_evaluation.aimo_2_inference_server

# ğŸ“Œ Gunakan model yang SUDAH DIUPLOAD di Kaggle Dataset
MODEL_DIR = "/kaggle/input/deepseek-r1/other/transform/1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# âœ… Load model & tokenizer (offline)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Fix pad_token_id warning

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, trust_remote_code=True, torch_dtype=torch.float16
)
model.to(device)
model.eval()

def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    """Melakukan prediksi jawaban soal matematika."""
    id_ = id_.item(0)
    question_text = question.item(0)

    # ğŸ”� Preprocessing input
    inputs = tokenizer(question_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # ğŸ�¯ Model melakukan prediksi
    with torch.no_grad():
        output = model.generate(input_ids, attention_mask=attention_mask, max_length=50, pad_token_id=tokenizer.eos_token_id)

    # ğŸ”¢ Konversi ke angka
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = int(''.join(filter(str.isdigit, answer))[-3:]) if any(char.isdigit() for char in answer) else 0

    return pl.DataFrame({'id': id_, 'answer': answer})

# ğŸ”¥ Jalankan server (tidak butuh internet)
inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        ('/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv',)
    )

# ğŸ”¥ Simpan hasil ke submission.parquet (agar bisa dikirim)
submission = pl.DataFrame({'id': [], 'answer': []})  # Placeholder untuk hasil
submission.write_parquet('submission.parquet')


