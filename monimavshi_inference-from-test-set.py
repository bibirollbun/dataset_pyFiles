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


# IMPORTANT: Replace 'my-map-inference-libs' if you used a different name
!pip install --no-index --find-links=/kaggle/input/my-map-inference-libs \
    bitsandbytes==0.43.1 \
    peft==0.10.0 \
    accelerate==0.30.1 \
    transformers==4.41.2 \
    triton==2.1.0 \
    scikit-learn==1.2.2

print("✅ Libraries installed from your personal offline dataset.")


import torch
import pandas as pd
import numpy as np
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm.auto import tqdm

# Define the paths to your trained model and mappings
# The folder name 'map-finetune-24' comes from the name of your training notebook
base_path = "/kaggle/input/map-finetune-24/"
model_path = base_path + "qwen-finetuned-best-checkpoint/"
mappings_path = base_path + "label_mappings.json"

print("✅ Paths defined.")


# Load the label mappings
with open(mappings_path, 'r') as f:
    mappings = json.load(f)
    id_to_label = {int(k): v for k, v in mappings['id_to_label'].items()}

# --- CHANGE IS HERE ---
# Define base model and quantization
# Use the local path to the model you added as a data source
model_name = "/kaggle/input/qwen2.5-math/transformers/1.5b/1" 

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load the base model from the local path
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    num_labels=len(id_to_label),
    device_map="auto"
)

# Load the tokenizer from the local path
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
base_model.config.pad_token_id = tokenizer.pad_token_id


# Load the LoRA adapters and merge them into the base model
model = PeftModel.from_pretrained(base_model, model_path)
model.eval() # Set the model to evaluation mode

print("✅ Fine-tuned model and mappings loaded successfully from local files.")


# Load the test data
df_test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")

# Define the prompt template (must be identical to training)
prompt_template = """Analyze the student's reasoning for the following math problem.

### Question:
{QuestionText}

### Student's Answer Choice:
{MC_Answer}

### Student's Explanation:
{StudentExplanation}

### Analysis:"""

predictions = []
# Use tqdm for a progress bar
for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
    # Format the prompt
    prompt = prompt_template.format(
        QuestionText=row['QuestionText'],
        MC_Answer=row['MC_Answer'],
        StudentExplanation=row['StudentExplanation']
    )
    
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(model.device)
    
    # Get model predictions (logits)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get top 3 predictions
    top_3 = torch.topk(logits, 3).indices.squeeze().tolist()
    
    # Decode the predictions and join them
    pred_labels = [id_to_label[i] for i in top_3]
    predictions.append(" ".join(pred_labels))

print("✅ Predictions generated.")


# Create the submission DataFrame
submission_df = pd.DataFrame({
    'row_id': df_test['row_id'],
    'Category:Misconception': predictions
})

# Save to submission.csv
submission_df.to_csv('submission.csv', index=False)

print("✅ submission.csv created successfully!")
print("You can now submit your notebook.")




