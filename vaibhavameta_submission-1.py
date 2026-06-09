# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os


print("hello world!")


import vllm
import torch


llm = vllm.LLM(
    "/kaggle/input/qwen2.5/transformers/14b-instruct-awq/1",
    quantization="awq",
    tensor_parallel_size=torch.cuda.device_count(),
    gpu_memory_utilization=0.9,
    trust_remote_code=True,
    dtype="half",
    enforce_eager=True,
    max_model_len=4096,
    disable_log_stats=True
)


# tokenizer = llm.get_tokenizer()


# tokenizer.encode("wheta")


%%time
llm.chat([{
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user",
            "content": "Hello."
        }], 
                vllm.SamplingParams(
                    n=1,  # Number of output sequences to return for each prompt.
                    top_p=0.8,  # Float that controls the cumulative probability of the top tokens to consider.
                    temperature=0,  # randomness of the sampling
                    seed=777, # Seed for reprodicibility
                    skip_special_tokens=False,  # Whether to skip special tokens in the output.
                    max_tokens=1026,  # Maximum number of tokens to generate per output sequence.
                )
            )


def query_model_vllm(prompt, skip_special_tokens=True):
    messages = [
        {
            "role": "user", "content": prompt
        }
    ]
    response = llm.chat(messages, 
                vllm.SamplingParams(
                    n=1,  # Number of output sequences to return for each prompt.
                    top_p=0.8,  # Float that controls the cumulative probability of the top tokens to consider.
                    temperature=0,  # randomness of the sampling
                    seed=777, # Seed for reprodicibility
                    skip_special_tokens=False,  # Whether to skip special tokens in the output.
                    max_tokens=5024,  # Maximum number of tokens to generate per output sequence.
                )
            )

    return response[0].outputs[0].text



def batch_query_model_vllm(prompts, skip_special_tokens=True):
    responses = llm.chat(
        prompts, 
        vllm.SamplingParams(
                    n=1,  # Number of output sequences to return for each prompt.
                    top_p=0.8,  # Float that controls the cumulative probability of the top tokens to consider.
                    temperature=0,  # randomness of the sampling
                    seed=777, # Seed for reprodicibility
                    skip_special_tokens=False,  # Whether to skip special tokens in the output.
                    max_tokens=5024,  # Maximum number of tokens to generate per output sequence.
                )
            )
    
    return [response.outputs[0].text for response in responses]


def get_chat_sturucture(prompt): 
    chat_struct = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    return chat_struct
    
    
    


%%time
batch_query_model_vllm([get_chat_sturucture("hello there!"), get_chat_sturucture("hey :(")])


%%time
query_model_vllm("hey :(")


## General
import pymupdf

import pymupdf4llm

import logging

## LLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

## JSON Extraction
import json
import re


env_config = {
    "LOG_LEVEL": "INFO"
}

# env_config = {
#     "LOG_LEVEL": "INFO"
# }
WE_USING_TRANSFORMERS = False


# Configure logger
logger = logging.getLogger('mdc-logger')
logger.setLevel(env_config['LOG_LEVEL'])

# Get handler
handler = logging.StreamHandler()
handler.setLevel(env_config['LOG_LEVEL'])

# Get formatter
formatter = logging.Formatter("%(asctime)s | %(levelname)s | [%(funcName)s(): %(lineno)s] | %(message)s")

handler.setFormatter(formatter)
logger.addHandler(handler)


# !pip install flash-attn --no-build-isolation


#!pip install autoawq


if WE_USING_TRANSFORMERS:
    model_path = "/kaggle/input/qwen-3/transformers/8b-awq/1"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, 
                                                 #quantization_config=BitsAndBytesConfig(load_in_4bit=True),
                                                 # attn_implementation="flash_attention_2",
                                                 device_map='auto')


# ## Assistant Model for Speculative Decoding
# assistant_model_path = "/kaggle/input/qwen2.5/transformers/0.5b-instruct/1"

# assistant_tokenizer = AutoTokenizer.from_pretrained(assistant_model_path)
# assistant_model = AutoModelForCausalLM.from_pretrained(assistant_model_path, torch_dtype=torch.bfloat16, quantization_config=BitsAndBytesConfig(load_in_4bit=True), device_map="auto")


def get_response_data(output):
    ## Thinking models
    think_start = output.find("<think>")
    think_end = output.find("</think>")

    return {"thought": output[think_start + 8: think_end], "answer": output[think_end + 9: ]}

def get_response_data_gemma(output):
    # Gemma
    token_string = "<end_of_turn>"
    eot_start = output.find(token_string)

    return output[eot_start: ].strip(token_string)


def get_response_data_qwen(output):
    # For Qwen 2.5
    start_token = "<|im_start|>"
    end_token = "<|im_end|>"

    responses = []

    for start_string in output.split(start_token):
        clean_string = start_string.strip('\n').strip(end_token).strip('\n')
        if clean_string:
            responses.append(clean_string)
    return responses[-1].strip("assistant\n")


def get_response_data_mistral(output):
    # For Mistral 3
    INST_END_TOKEN = '[/INST]'
    return output[output.find(INST_END_TOKEN) + len(INST_END_TOKEN): ].strip('<s>').strip("</s>")

    
    

def query_model(prompt, skip_special_tokens=True):
    messages = [
        {
            "role": "user", "content": prompt
        }
    ]
    
    inputs = tokenizer.apply_chat_template(messages, return_tensors='pt', padding_side="left").to("cuda")
    # tokens = tokenizer.apply_chat_template(messages, return_tensors='pt', padding_side="left", tokenize=False).to("cuda")
    # tokens = tokenizer.tokenize()
    # inputs = tokens.i
    #inputs = tokenizer.encode(prompt, return_tensors='pt', padding_side="left").to("cuda")

    # outputs = model.generate(inputs, assistant_model=assistant_model, tokenizer=tokenizer, assistant_tokenizer=assistant_tokenizer, max_new_tokens=50000)
    #outputs = assistant_model.generate(inputs, max_new_tokens=5000)
    #outputs = model.generate(inputs, assistant_early_exit=4, max_new_tokens=50000)

    outputs = model.generate(inputs,  max_new_tokens=50000)

    #return get_response_data(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    #print(tokenizer.chat_template)
    #return get_response_data_gemma(tokenizer.batch_decode(outputs)[0])
    return tokenizer.batch_decode(outputs, skip_special_tokens=skip_special_tokens)[0]


def extract_codeBlockData(text, returnInput=True):
    """
    given text, extract data that may be present inside a code block (enclosed withing ``` ```)
    only returns the data within the very first code block encountered.

    Args:
        text (str): string from which the data needs to be extracted
        returnInput (bool): if True, will return ib if there was no code block found.

    Returns:
        extracted_data (str/None): string if any data was extracted. None, in case of failure/No data found.
    """
    try:
        ## also dealing with incomplete code blocks (code block created, but not ended)
        if text.count("```") == 1:
            text += "```"
        pattern = r"```(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        text = matches[0].strip()
    except Exception as e:
        logger.debug(f"Error @extract_codeBlock(): {str(e)}")
        if not returnInput:
            text = None
    return text

@staticmethod
def extract_jsonCodeBlock(text, strict_flag=False):
    """
    given a json string (text), extract it into a dict;
    performs a strip('json'), and strip() on the string before a json.loads

    Args:
        text (str): text cotaining the json string
        strict_flag (bool): flag for 'strict' kwarg for json.loads; will be okay with control characters in the string

    Returns:
        text (dict/None): json object as a dict, if the extraction was successful; None, otherwise.
    """
    try:
        ### removing json at the beginning/end of the text
        text = text.strip("```").strip('json').strip('JSON').strip()
        text = json.loads(text, strict=strict_flag)
    except Exception as e:
        logger.debug(f"Error @extract_jsonCodeBlock(): {str(e)}")
        text = None
    return text


%%time
try:
    print(get_response_data(query_model_vllm("Wait a minute! Who are you?")))
except Exception as e:
    pass


# %%time
# query_model("Wait a minute! Who are you?", skip_special_tokens=True)


# %%time
# get_response_data(query_model("Wait a minute! Who are you?"))


# %%time
# get_response_data_mistral(query_model("Wait a minute! Who are you?"))


try:
    train_data_labels = pd.read_csv("/kaggle/input/make-data-count-finding-data-references/train_labels.csv")
except Exception as e:
    train_data_labels = pd.DataFrame()
    logger.error(f"Error: {str(e)}")


# Get toc along with the text
def extract_pdf_text(filepath, get_blocks=False):
    if get_blocks:
        extracted_text = []
    else:
        extracted_text = ""

    file = pymupdf.open(filepath)

    for page in file:
        if get_blocks:
            extracted_text.extend(page.get_text("blocks"))
        else:
            extracted_text += '\n' + page.get_text()

    return file.get_toc(simple=False), extracted_text


PROMPT = """The text provided to you is a snippet from a research paper. Your task is to identify mentions of datasets in the snippet, and provide appropriate details.
Provide a JSON with the following details for each mention of a dataset:
1. 'name': name of the dataset.
2. 'identifier': identifier of the dataset (unique identifier/url/accession number)
3. 'type': 'Primary',or 'Secondary'. Primary would mean the dataset was created/generated for this paper. Secondary if the dataset was taken from another paper/source (not directly related to this paper).
4. 'source': Source of the dataset (optional)
5. 'ref_no': Reference number (If applicable, provide the list number that can be looked up in the references section.)
6. 'details': Details (some brief explanation)

Reply with just 'None' if there is no mention of a dataset.

Make sure that:
1. The identifier is a reference to a **dataset** and not a paper.
2. Prefer **DOI references**, followed by other **identifiers such as PBD, GSE, etc.** Give lowest preference to obscure (not known/generally used) URLs.
---

Below is the snippet from the paper:
{TEXT}
"""


def get_data_references(filepath):
    responses = []
    try:
        toc, text = extract_pdf_text(filepath, get_blocks=True)
    
        counter = 0
        counter_2 = 0
        logger.debug(f"text list len: {len(text)}")
        
        for te in text:
            if "doi.org" in te[4] or 'pdb ' in te[4] or "gse" in te[4]:
                logger.debug(f"Finally found something! @{counter_2=}")
                logger.debug(f"Looking at {te[4]}")
                response = query_model_vllm(PROMPT.format(TEXT=te[4]))
                # response = query_model(PROMPT.format(TEXT=te[4]))
                # extracted_output = get_response_data_mistral(response)

                ## For reasoning models
                # extracted_output = get_response_data(response)
                # logger.debug(f"Extracted_output: {extracted_output}")
                # extracted_json = extract_jsonCodeBlock(extract_codeBlockData(extracted_output['answer'].strip()))

                ## For non-reasoning models
                #extracted_output = get_response_data(response)
                logger.debug(f"Extracted_output: {response}")
                extracted_json = extract_jsonCodeBlock(extract_codeBlockData(response.strip()))
    
                if extracted_json:
                    responses.append(extracted_json)
    
                counter += 1
                logger.debug(f"DONE {counter}")
                if counter >= 7:
                    logger.debug(f"Tried for {te[4]}")
                    #break
            counter_2 += 1
            if counter_2%20 == 0:
                logger.debug(f"Here after {counter_2=}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")

    return responses




    


def get_only_jsons(responses):
    only_jsons = []
    
    for response in responses:
        if isinstance(response, str):
            if response.lower().strip() == 'none':
                continue
            else:
                try:
                    only_jsons.append(json.loads(response, strict=strict_flag))
                except:
                    logger.error(f"Got string as response, but could't convert to a dict/json. got: {response}")
                    pass
        elif isinstance(response, dict):
            only_jsons.append(response)
        elif isinstance(response, list):
            for re in response:
                if isinstance(re, dict):
                    only_jsons.append(re)
                elif isinstance(re, str):
                    try:
                        only_jsons.append(json.loads(response, strict=strict_flag))
                    except:
                        logger.error(f"Got string as response, but could't convert to a dict/json. got: {response}")
                        pass       
        else:
            pass
    return only_jsons


def get_article_id(filepath):
    try:
        return '.'.join(filepath.split(".")[:-1])
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return filepath


def get_empty_submission(filepath):
    response = {
        "article_id": get_article_id(filepath),
        "dataset_id": "Missing",
        "type": "Missing"
    }
    return response


def get_submission_data(filepath, only_jsons):
    submission_data = []

    for json in only_jsons:
        try:
            article_id = get_article_id(filepath)
            submission_data.append(
                {
                    "article_id": article_id,
                    "dataset_id": json['identifier'],
                    "type": json['type'] 
                }
            )
        except Exception as e:
            logger.error(f"Error while generating submission data: {str(e)}")
    # print(submission_data)
    return submission_data
            


%%time
try:
    filepath = "/kaggle/input/make-data-count-finding-data-references/test/PDF/10.1002_2017jc013030.pdf"
    
    responses = get_data_references(filepath)
except Exception as e:
    responses = []
    logger.error(f"Error: {str(e)}")


try:
    from IPython.display import display
    display(responses)
except Exception as e:
    logger.error(f"Error: {str(e)}")


try:
    display(get_submission_data(filepath, get_only_jsons(responses)))
except Exception as e:
    logger.error(f"Error: {str(e)}")


try:
    display(train_data_labels[train_data_labels['article_id'] == '10.1002_2017jc013030'])
except Exception as e:
    logger.error(f"Error: {str(e)}")


# train_data_labels


submission_data = []
folder_path = "/kaggle/input/make-data-count-finding-data-references/test/PDF"
for filepath in os.listdir(folder_path):
    logger.info(f"Looking at: {filepath}")
    try:
        responses = get_data_references(f"{folder_path}/{filepath}")
        #responses = get_data_references("/kaggle/input/make-data-count-finding-data-references/test/PDF/10.1002_2017jc013030.pdf")
        submission_list = get_submission_data(filepath, get_only_jsons(responses))
        if submission_list:
            submission_data.extend(submission_list)
        # else:
        #     submission_data.append(get_empty_submission(filepath))
    except Exception as e:
        #submission_data.append(get_empty_submission(filepath))
        logger.error(f"Encountered an error while generating submission. Error: {str(e)}")
    


submission_data


submission_df = pd.DataFrame.from_dict(submission_data)
submission_df['row_id'] = submission_df.index


submission_df = submission_df[['row_id', 'article_id', 'dataset_id', 'type']]


submission_df.drop_duplicates(subset=['article_id', 'dataset_id'], keep='first', inplace=True)


submission_df.to_csv(f"/kaggle/working/submission.csv", index=False)


# submission_df




