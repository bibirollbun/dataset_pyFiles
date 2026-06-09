# !pip install -U "transformers>=4.42.3" bitsandbytes accelerate peft


!pip install transformers peft accelerate bitsandbytes \
    -U --no-index --find-links /kaggle/input/lmsys-wheel-files


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

VER=159

# FINAL SOLUTION IS USE_QLORA=FALSE, TRAIN_100_PERCENT=TRUE, ADD_33K=TRUE, DEBUG=FALSE
USE_QLORA = True
TRAIN_100_PERCENT = False
ADD_33K = False
DEBUG = True


import os
import copy
from dataclasses import dataclass

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    BitsAndBytesConfig,
    Gemma2ForSequenceClassification,
    GemmaTokenizerFast,
    Gemma2Config,
    PreTrainedTokenizerBase, 
    EvalPrediction,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from sklearn.metrics import log_loss, accuracy_score


tokenizer = GemmaTokenizerFast.from_pretrained("/kaggle/input/gemma2-9b-it-fp16")
tokenizer.add_eos_token = True  # We'll add <eos> at the end
tokenizer.padding_side = "right"


import torch
import torch.nn as nn
from transformers import Gemma2ForSequenceClassification, Gemma2Config

class CustomGemma2ForSequenceClassification(Gemma2ForSequenceClassification):
    # def __init__(self, config, num_labels_head1=58, num_labels_head2=58):
    def __init__(self, config, num_labels_head1=54, num_labels_head2=54,num_labels_head3=128):
        super().__init__(config)
        self.num_labels_head1 = num_labels_head1
        self.num_labels_head2 = num_labels_head2
        self.num_labels_head3 = num_labels_head3
        self.classifier_head1 = nn.Linear(config.hidden_size, num_labels_head1, bias=False)
        self.classifier_head2 = nn.Linear(config.hidden_size, num_labels_head2, bias=False)
        self.classifier_head3 = nn.Linear(config.hidden_size, num_labels_head3, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        device = input_ids.device

        if labels is not None:
            labels = labels.to(device)
            outputs = super().forward(input_ids, attention_mask=attention_mask, labels=labels[:, 0], output_hidden_states=True)
        else:
            outputs = super().forward(input_ids, attention_mask=attention_mask)

        last_token_indices = (torch.sum(attention_mask, dim=1) - 1).to(device)
        last_token_outputs = outputs.hidden_states[-1].to(device)[
            torch.arange(outputs.hidden_states[-1].shape[0], device=device), last_token_indices]

        outputs_head1 = self.classifier_head1(last_token_outputs).to(device)
        outputs_head2 = self.classifier_head2(last_token_outputs).to(device)
        outputs_head3 = self.classifier_head3(last_token_outputs).to(device)

        if labels is not None:
            labels_head1 = labels[:, 1].to(device)
            labels_head2 = labels[:, 2].to(device)
            labels_head3 = labels[:, 3].to(device)
            
            loss_head1 = nn.CrossEntropyLoss()(outputs_head1, labels_head1)
            loss_head2 = nn.CrossEntropyLoss()(outputs_head2, labels_head2)
            loss_head3 = nn.CrossEntropyLoss()(outputs_head3, labels_head3)
            loss = outputs.loss.to(device) + 0.1 * loss_head1 + 0.1 * loss_head2 + 0.1 * loss_head3
            return {"loss": loss, "logits": (outputs.logits, outputs_head1, outputs_head2, outputs_head3)}
        else:
            return {"logits": (outputs.logits, outputs_head1, outputs_head2, outputs_head3)}


from peft import PeftModel
config2 = Gemma2Config.from_pretrained("/kaggle/input/gemma2-9b-it-fp16")
# config2.num_labels = 3
config2.num_labels = 2
model = CustomGemma2ForSequenceClassification.from_pretrained(
    "/kaggle/input/gemma2-9b-it-fp16",
    config=config2,
    num_labels_head1=54,
    num_labels_head2=54,
    num_labels_head3=128,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.config.use_cache = False
model = PeftModel.from_pretrained(model, "/kaggle/input/lora_v159/pytorch/default/1/LoRA-v159")
model


model = model.merge_and_unload()
model


model.save_pretrained(f"merged-v{VER}") 
tokenizer.save_pretrained(f"merged-v{VER}")


del model
torch.cuda.empty_cache()


from transformers import BitsAndBytesConfig
if not USE_QLORA: 
    bnb_config = BitsAndBytesConfig(
        load_in_8bit = True,
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_skip_modules = ["score","classifier_head1", "classifier_head2", "classifier_head3"]
    )
else: # USE SAME QUANTIZATION THAT WE TRAINED WITH
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4", #nf4 or fp4
        bnb_4bit_use_double_quant = False,
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_skip_modules = ["score","classifier_head1", "classifier_head2", "classifier_head3"]
    )


config2 = Gemma2Config.from_pretrained("/kaggle/input/gemma2-9b-it-fp16")
config2.num_labels = 2
model = CustomGemma2ForSequenceClassification.from_pretrained(
    f"merged-v{VER}",
    config=config2,
    num_labels_head1=54,
    num_labels_head2=54,
    num_labels_head3=128,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config = bnb_config,
)
model.config.use_cache = False
model


os.system(f"rm -r merged-v{VER}") 


if USE_QLORA:
    model.save_pretrained(f"merged-v{VER}-4bit") 
    tokenizer.save_pretrained(f"merged-v{VER}-4bit")
else:
    model.save_pretrained(f"merged-v{VER}-8bit") 
    tokenizer.save_pretrained(f"merged-v{VER}-8bit")

