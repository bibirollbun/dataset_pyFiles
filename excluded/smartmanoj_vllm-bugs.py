
!python -m pip install  /kaggle/input/browser-notifications-in-a-kaggle-kernel/*.whl 
!python -m pip install jupyter -q --no-index --find-links=/kaggle/input/browser-notifications-in-a-kaggle-kernel/ 
!python -m pip install -q /kaggle/input/browser-notifications-in-a-kaggle-kernel/jupyter-notify/dist/jupyternotify-0.1.15-py2.py3-none-any.whl --no-index



%reload_ext jupyternotify


# !pip install vllm==0.7.2 --target=/kaggle/working


from vllm import LLM, SamplingParams
import warnings
import os

warnings.simplefilter('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


llm_model_pth = '/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-32b-awq/1'


MAX_NUM_SEQS = 4
MAX_MODEL_LEN = 32_768
MAX_TOKENS = 32_768

llm = LLM(
    llm_model_pth,
    # dtype="half",               # The data type for the model weights and activations
    max_num_seqs=MAX_NUM_SEQS,   # Maximum number of sequences per iteration. Default is 256
    max_model_len=MAX_MODEL_LEN, # Model context length
    trust_remote_code=True,      # Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer
    tensor_parallel_size=4,      # The number of GPUs to use for distributed execution with tensor parallelism
    gpu_memory_utilization=0.95, # The ratio (between 0 and 1) of GPU memory to reserve for the model
    seed=2024,
)

tokenizer = llm.get_tokenizer()


def test_generate(content, temperature=0):

    sampling_params = SamplingParams(
        temperature=temperature,              # randomness of the sampling
        min_p=0.01,
        skip_special_tokens=True,     # Whether to skip special tokens in the output
        max_tokens=MAX_MODEL_LEN,
    )
    
    list_of_messages = [
        [
            {
                "role": "user",
                "content": content
            },
        ]
    ]

    list_of_texts = [
        tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True
        )
        for messages in list_of_messages
    ]
    print([len(tokenizer.encode(text)) for text in list_of_texts])

    request_output = llm.generate(prompts=list_of_texts, sampling_params=sampling_params)
    if not request_output:
        return ""
    return request_output[0].outputs[0].text


%%notify
from IPython.display import display, Latex
display(Latex(test_generate("Tell a random number with 4 decimal places between 1 to 10.")))
print('---'*20)
display(Latex(test_generate("Tell a random number with 4 decimal places between 1 to 10.")))
print('---'*20)
display(Latex(test_generate("Tell a random number with 4 decimal places between 1 to 10.",temperature=1)))
print('---'*20)
display(Latex(test_generate("Tell a random number with 4 decimal places between 1 to 10.",temperature=1)))


!wget https://raw.githubusercontent.com/vllm-project/vllm/main/collect_env.py
# For security purposes, please feel free to check the contents of collect_env.py before running it.
!python collect_env.py


# transformers test


!pip install autoawq


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


torch.cuda.empty_cache()


from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
model_name = "/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-32b-awq/1"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map = "sequential" ).cuda()


!nvidia-smi


!ps 8505


!chmod +x /kaggle/working/triton/backends/nvidia/bin/ptxas


def test_generate2(content, temperature=0.0):
    
    inputs = tokenizer(content, return_tensors="pt").to("cuda")
    import torch
    torch.manual_seed(42)
    with torch.no_grad():
      outputs = model.generate(**inputs, max_length=512, temperature=temperature, do_sample=temperature)
    
    return (tokenizer.decode(outputs[0], skip_special_tokens=True))


%%notify
from IPython.display import display, Latex
display(Latex(test_generate2("Tell a random number with 4 decimal places between 1 to 10.")))
print('---'*20)
display(Latex(test_generate2("Tell a random number with 4 decimal places between 1 to 10.")))
print('---'*20)
display(Latex(test_generate2("Tell a random number with 4 decimal places between 1 to 10.",temperature=1)))
print('---'*20)
display(Latex(test_generate2("Tell a random number with 4 decimal places between 1 to 10.",temperature=1)))


!pkill -9 8505 


!ps 8505

