from IPython.display import clear_output


#!pip install -U transformers bitsandbytes peft accelerate
!pip install --no-index --find-links /kaggle/input/wsdm-wheels -r /kaggle/input/requirements/requirements.txt
clear_output()


import pandas as pd
import os
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig
)
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from peft import PeftModel
from huggingface_hub import login
import torch
from kaggle_secrets import UserSecretsClient
from concurrent.futures import ThreadPoolExecutor


class Config:
    input_directory = '/kaggle/input/wsdm-cup-multilingual-chatbot-arena'
    model = '/kaggle/input/gemma-2/transformers/gemma-2-9b-it/2'
    #model = '/kaggle/input/gemma-2-9b-4bit-it-unsloth/transformers/default/1/gemma-2-9b-it-4bit-unsloth_old'
    #model = 'google/gemma-2-9b-it'
    weights_lora = '/kaggle/input/googlegemma-2-9b-it-qlora-wsdm-2/transformers/default/1'
    max_length = 2048
    batch_size = 8
    dtype = torch.float16

config = Config()


def tokenize(prompt,response_a,response_b,tokenizer,max_length = config.max_length):
    prompt = ['<prompt>: '+p for p in prompt]
    response_a = ['\n\n'+'<response_a>: '+r_a for r_a in response_a]
    response_b = ['\n\n'+'<response_b>: '+r_b for r_b in response_b]
    text_input = [p+r_a+r_b for p,r_a,r_b in zip(prompt,response_a,response_b)]

    encoded_input = tokenizer(text_input,max_length=max_length,truncation=True,padding=False)
    return encoded_input['input_ids'],encoded_input['attention_mask']


quantization_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_compute_dtype = config.dtype,
    bnb_4bit_quant_type = 'nf4',
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_storage = config.dtype
)


#user_secrets = UserSecretsClient()
#os.environ['HF_TOKEN'] = user_secrets.get_secret('HF_TOKEN')
#login(token = os.environ['HF_TOKEN'])


tokenizer = AutoTokenizer.from_pretrained(config.model)
tokenizer.add_eos_token = True
tokenizer.padding_side = 'right'


device_0 = torch.device('cuda:0')
model_0 = AutoModelForSequenceClassification.from_pretrained(
    config.model,
    num_labels = 2,
    device_map = device_0,
    #dtype = config.dtype,
    use_cache = False,
    quantization_config = quantization_config
)

device_1 = torch.device('cuda:1')
model_1 = AutoModelForSequenceClassification.from_pretrained(
    config.model,
    num_labels = 2,
    device_map = device_1,
    #dtype = config.dtype,
    use_cache = False,
    quantization_config = quantization_config
)

model_0 = PeftModel.from_pretrained(model_0,config.weights_lora)
model_1 = PeftModel.from_pretrained(model_1,config.weights_lora)


test = pd.read_parquet(config.input_directory + '/test.parquet')


data = pd.DataFrame()
data['input_ids'],data['attention_mask'] = tokenize(test['prompt'],test['response_a'],test['response_b'],tokenizer)
data['id'] = test['id']

#aug_data = pd.DataFrame()
#aug_data['input_ids'],aug_data['attention_mask'] = tokenize(test['prompt'],test['response_b'],test['response_a'],tokenizer)
#aug_data['id'] = test['id']


@torch.no_grad()
@torch.cuda.amp.autocast()
def inference(df,model,device,batch_size = config.batch_size,max_length = config.max_length):
    predictions_all = []
    for start_idx in range(0,len(df),batch_size):
        end_idx = min(start_idx + batch_size,len(df))
        tmp = df.iloc[start_idx:end_idx]
        input_ids = tmp['input_ids'].to_list()
        attention_mask = tmp['attention_mask'].to_list()
        inputs = pad_without_fast_tokenizer_warning(
            tokenizer,
            {'input_ids':input_ids,'attention_mask':attention_mask},
            padding = 'longest',
            max_length = max_length,
            pad_to_multiple_of = None,
            return_tensors = 'pt',
        ).to(device)

        logits = model(**inputs).logits
        predictions = logits.argmax(dim = 1).cpu()
        predictions_all.extend(predictions.tolist())
    df['winner'] = predictions_all
    return df


sub_data_0 = data.iloc[0::2].copy()
sub_data_1 = data.iloc[1::2].copy()


with ThreadPoolExecutor(max_workers=2) as executor:
    sub_results = executor.map(inference,(sub_data_0,sub_data_1),(model_0,model_1),(device_0,device_1))


results = list(sub_results)
results_df = pd.concat(results)
results_df['winner'] = ['model_a' if pred==0 else 'model_b' for pred in results_df['winner']]
results_df[['id','winner']].to_csv(r'submission.csv', index=False)

