!pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3 -q --no-cache
!pip install -U bitsandbytes -q
!pip install gtts -q


import platform,socket,re,uuid,json,psutil,logging

def getSystemInfo():
    try:
        info={}
        info['platform']=platform.system()
        info['platform-release']=platform.release()
        info['platform-version']=platform.version()
        info['architecture']=platform.machine()
        info['hostname']=socket.gethostname()
        info['ip-address']=socket.gethostbyname(socket.gethostname())
        info['mac-address']=':'.join(re.findall('..', '%012x' % uuid.getnode()))
        info['processor']=platform.processor()
        info['ram']=str(round(psutil.virtual_memory().total / (1024.0 **3)))+" GB"
        return json.dumps(info)
    except Exception as e:
        logging.exception(e)

json.loads(getSystemInfo())


import gc
import torch
from transformers import pipeline, BitsAndBytesConfig, AutoProcessor, AutoTokenizer, Gemma3ForConditionalGeneration
from accelerate import Accelerator
import base64
import io
from PIL import Image

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import os

os.environ['TRANSFORMERS_OFFLINE']='1'


train = pd.read_csv(r'/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv(r'/kaggle/input/playground-series-s5e4/test.csv')
test = pd.read_csv(r'/kaggle/input/playground-series-s5e4/sample_submission.csv')
target = "rainfall"


train.info()


def generate_visualizations(df):
    """
    Generates matplotlib visualizations for the new podcast dataset.
    Returns a list of base64-encoded PNG images of the visualizations.
    """

    visualizations = []

    # 1. Count plot for Genre
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Genre', data=df)
    plt.title('Count of Genres')
    plt.xlabel('Genre')
    plt.ylabel('Count')
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png', bbox_inches='tight')
    img_data.seek(0)
    img_base64 = base64.b64encode(img_data.getvalue()).decode('utf-8')
    visualizations.append(img_base64)
    plt.close()

    # 2. Count plot for Episode_Sentiment
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Episode_Sentiment', data=df)
    plt.title('Count of Episode Sentiments')
    plt.xlabel('Episode Sentiment')
    plt.ylabel('Count')
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png', bbox_inches='tight')
    img_data.seek(0)
    img_base64 = base64.b64encode(img_data.getvalue()).decode('utf-8')
    visualizations.append(img_base64)
    plt.close()

    # 3. Correlation Matrix (Heatmap) for numeric columns
    numeric_cols = [
        'Episode_Length_minutes', 
        'Host_Popularity_percentage',
        'Guest_Popularity_percentage',
        'Number_of_Ads',
        'Listening_Time_minutes'
    ]
    # Drop columns if they have nulls and can't be correlated
    corr_df = df[numeric_cols].dropna()
    plt.figure(figsize=(10, 8))
    correlation_matrix = corr_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix (Numeric Columns)')
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png', bbox_inches='tight')
    img_data.seek(0)
    img_base64 = base64.b64encode(img_data.getvalue()).decode('utf-8')
    visualizations.append(img_base64)
    plt.close()

    # 4. Distribution of Numeric Features
    for feature in numeric_cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png', bbox_inches='tight')
        img_data.seek(0)
        img_base64 = base64.b64encode(img_data.getvalue()).decode('utf-8')
        visualizations.append(img_base64)
        plt.close()

    # 5. Boxplots of Numeric Features vs. Episode_Sentiment
    for feature in numeric_cols:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Episode_Sentiment', y=feature, data=df)
        plt.title(f'Boxplot of {feature} vs. Episode_Sentiment')
        plt.xlabel('Episode Sentiment')
        plt.ylabel(feature)
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png', bbox_inches='tight')
        img_data.seek(0)
        img_base64 = base64.b64encode(img_data.getvalue()).decode('utf-8')
        visualizations.append(img_base64)
        plt.close()

    # 6. Scatter plot of Episode_Length_minutes vs. Listening_Time_minutes
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x='Episode_Length_minutes', 
        y='Listening_Time_minutes', 
        data=df
    )
    plt.title('Episode Length vs. Listening Time')
    plt.xlabel('Episode Length (minutes)')
    plt.ylabel('Listening Time (minutes)')
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png', bbox_inches='tight')
    img_data.seek(0)
    img_base64 = base64.b64encode(img_data.getvalue()).decode('utf-8')
    visualizations.append(img_base64)
    plt.close()

    return visualizations


visualizations = generate_visualizations(train)


from IPython.display import Audio, Markdown
from gtts import gTTS
import torch
from transformers import pipeline, BitsAndBytesConfig


def create_pipeline(model_path: str = "/kaggle/input/gemma-3/transformers/gemma-3-4b-it/1/"):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    accelerator = Accelerator()
    model_path = "/kaggle/input/gemma-3/transformers/gemma-3-4b-it/1/"

    # Load the processor
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load the model with quantization configuration
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )

    # Create the pipeline with model + tokenizer
    pipe = pipeline(
        "image-text-to-text",
        model=model,
        processor=processor,
        tokenizer=tokenizer,  # Pass tokenizer explicitly
    )

    pipe, model = accelerator.prepare(pipe, model)
    return pipe, model, tokenizer, accelerator


pipe, model, tokenizer, accelerator = create_pipeline()


# Get insights from the plots 
def process_images_with_batching(visualizations, pipe, batch_size=2): #adjust batch_size based on VRAM.
    all_outputs = []
    for i in range(0, len(visualizations), batch_size):
        batch_images = []
        batch_messages = []
        for img_base64 in visualizations[i:i + batch_size]:
            img_bytes = base64.b64decode(img_base64)
            img = Image.open(io.BytesIO(img_bytes))
            batch_images.append(img)
            batch_messages.append([
                {"role": "user", "content": "Let's begin our analysis"},
                {"role": "assistant", "content": "You are an analyst. You give great insights into data."},
                {"role": "user",
                 "content": [
                     {"type": "image"},
                     {"type": "text", "text": "Describe the key insights from this plot. Focus on insights and do not describe the plot itself too much. Think step-by-step, but only keep a minimum draft for each thinking step."}
                 ]}
            ])


        output = pipe(
            text=batch_messages,
            images=batch_images,
            max_new_tokens=64
        )
        for out in output:
            print(out[0]["generated_text"][3]['content'])
            all_outputs.append(out[0]["generated_text"][3]['content'])
    
    return all_outputs

outputs = process_images_with_batching(visualizations, pipe)


# remove old model
del pipe
del model
del tokenizer
del accelerator
torch.cuda.empty_cache()
_ = gc.collect()


# get fresh instance to have shorter context window
pipe, model, tokenizer, accelerator = create_pipeline(model_path="/kaggle/input/gemma-3/transformers/gemma-3-27b-it/1/")


# Clean up outputs
final_output_cleaned = "\n".join([o.strip() for o in outputs if o.strip() != ""])

# Truncate if necessary
max_length = 128000
if len(final_output_cleaned) > max_length:
    final_output_cleaned = final_output_cleaned[:max_length] + "\n\n[Truncated]"

# Prepare chat formatted input
chat_input = [
    {"role": "user", "content": "."},
    {"role": "assistant", "content": "You are a Kaggle Grandmaster and expert data scientist."},
    {"role": "user", "content": f"""Summarize the dataset analysis provided below in a maximum of 500 words.
    Think step-by-step, but only keep a minimum draft for each thinking step. Do not repeat any provided text verbatim.

After the summary, derive top recommendations to build an optimal machine learning pipeline based on the insights. These recommenfdations shal include preprocessing,
the correct selection of the best algorithm and hyperparameter tuning with a robost cv scheme (using sklearn). This part shall have
a maximum of 500 words as well.

Write an example main.py file that shows the end-to-end code in Python.

Dataset analysis:
{final_output_cleaned}
"""}
]

# Apply chat template & tokenize
inputs = tokenizer.apply_chat_template(chat_input, return_tensors="pt").to(model.device)
prompt_length = inputs.shape[1]

# Generate
outputs_summary = model.generate(inputs, max_new_tokens=5000, eos_token_id=tokenizer.eos_token_id)


# Decode
decoded_output = tokenizer.decode(outputs_summary[0][prompt_length:], skip_special_tokens=True)

print(decoded_output)


tts = gTTS(text=decoded_output, lang='en')
file='output_summary.mp3'
tts.save(file)


Audio(file, autoplay=False)


# remove old model
del pipe
del model
del tokenizer
del accelerator
torch.cuda.empty_cache()
_ = gc.collect()


# get fresh instance to have shorter context window
pipe, model, tokenizer, accelerator = create_pipeline(model_path="/kaggle/input/gemma-3/transformers/gemma-3-27b-it/1/")


# Prepare chat formatted input
chat_input = [
    {"role": "user", "content": "."},
    {"role": "assistant", "content": "You are a Kaggle super Grandmaster and expert machine learning engineer."},
    {"role": "user", "content": f"""Write code for a machine learning pipeline to predict the regression target {target}. Do not use any form of oversampling!
    For preprocessing and CV strategies consider sklearn only, no extra library.
    Think step-by-step, but only keep a minimum draft for each thinking step.
    
    You can assume that the Pandas DataFrames train, test and submissions have been loaded already. 
    The submission file can be created like in this example:

    submission[target] = y_probs
    submission.to_csv("submission.csv", index=False)

Do not repeat any provided text verbatim.

Here are insights an recommendations from a fellow data scientist that shall help you:
{decoded_output}
"""}
]

# Apply chat template & tokenize
inputs = tokenizer.apply_chat_template(chat_input, return_tensors="pt").to(model.device)
prompt_length = inputs.shape[1]

# Generate
outputs_summary = model.generate(inputs, max_new_tokens=10000, eos_token_id=tokenizer.eos_token_id)


# Decode
decoded_output = tokenizer.decode(outputs_summary[0][prompt_length:], skip_special_tokens=True)
print(decoded_output)

