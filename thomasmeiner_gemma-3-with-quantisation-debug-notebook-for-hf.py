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


train = pd.read_csv(r'/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv(r'/kaggle/input/playground-series-s5e3/test.csv')


train.info()


def generate_visualizations(df):
    """
    Generates matplotlib visualizations to summarize the given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        list: A list of base64 encoded PNG images of the visualizations.
    """

    visualizations = []

    # 1. Distribution of Rainfall (Target Variable)
    plt.figure(figsize=(8, 6))
    sns.countplot(x='rainfall', data=df)
    plt.title('Distribution of Rainfall')
    plt.xlabel('Rainfall (0: No Rain, 1: Rain)')
    plt.ylabel('Count')
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    img_base64 = base64.b64encode(img_data.getvalue()).decode('utf-8')
    visualizations.append(img_base64)
    plt.close()

    # 2. Correlation Matrix (Heatmap)
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    img_base64 = base64.b64encode(img_data.getvalue()).decode('utf-8')
    visualizations.append(img_base64)
    plt.close()

    # 3. Distribution of Numerical Features
    numerical_features = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
                          'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

    for feature in numerical_features:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png')
        img_data.seek(0)
        img_base64 = base64.b64encode(img_data.getvalue()).decode('utf-8')
        visualizations.append(img_base64)
        plt.close()

    # 4. Boxplots of Numerical Features vs. Rainfall
    for feature in numerical_features:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='rainfall', y=feature, data=df)
        plt.title(f'Boxplot of {feature} vs. Rainfall')
        plt.xlabel('Rainfall')
        plt.ylabel(feature)
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png')
        img_data.seek(0)
        img_base64 = base64.b64encode(img_data.getvalue()).decode('utf-8')
        visualizations.append(img_base64)
        plt.close()

    # 5. scatter plot of temp vs dewpoint
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='temparature', y='dewpoint', data=df, hue = 'rainfall')
    plt.title('Temperature vs Dewpoint')
    plt.xlabel('Temperature')
    plt.ylabel('Dewpoint')
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png')
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


def create_pipeline():
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
        tokenizer=tokenizer,  # Pass tokenizer explicitly
    )

    pipe, model = accelerator.prepare(pipe, model)
    return pipe, accelerator


pipe, accelerator = create_pipeline()


# Get insights from the plots 
def process_images_with_batching(visualizations, pipe, batch_size=2): #adjust batch_size based on VRAM.
    all_outputs =
    for i in range(0, len(visualizations), batch_size):
        batch_images =
        batch_messages =
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
                     {"type": "text", "text": "Describe the key insights from this plot. Focus on insights and do not describe the plot itself too much."}
                 ]}
            ])


        output = pipe(
            text=batch_messages,
            images=batch_images,
            max_new_tokens=200
        )
        for out in output:
            print(out[0]["generated_text"][3]['content'])
            all_outputs.append(out[0]["generated_text"][3]['content'])
    return all_outputs

outputs = process_images_with_batching(visualizations, pipe)


# remove old model
del pipe
del accelerator


# get fresh instance to have shorter context window
pipe, accelerator = create_pipeline()


final_output_cleaned = "\n".join([o.strip() for o in outputs if o.strip() != ""])

max_length = 128000
if len(final_output_cleaned) > max_length:
    final_output_cleaned = final_output_cleaned[:max_length] + "\n\n[Truncated]"

message = """You got insights about a dataset. Please gsummarize the main findings."""

output = pipe(
    text=[final_output_cleaned, message],
    max_new_tokens=5000
)


tts = gTTS(text=output[0][0]["generated_text"], lang='en')
file='output.mp3'
tts.save(file)


output[0]


Audio(file, autoplay=False)


display(Markdown(output[0][0]["generated_text"]))

