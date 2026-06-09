import os
import math
from math import exp
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


os.environ['TOKENIZERS_PARALLELISM'] = 'false'

PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HuggingFaceModelLoader:
    def __init__(self, model_path: str, load_in_4bit: bool, device_map: str):
        self.model_path = model_path
        self.load_in_8bit = load_in_4bit
        self.device_map = device_map

    def load_model(self) -> transformers.PreTrainedModel:
        if self.load_in_8bit:
            if DEVICE.type != 'cuda':
                raise ValueError('8-bit quantization requires a CUDA device')

            quantization_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_use_double_quant=False,
                bnb_4bit_compute_dtype=torch.float16,
            )

            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                device_map=self.device_map
            )
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map=self.device_map
            )

        model.eval()
        return model


class HuggingFaceTokenizer:
    def __init__(self, model_path: str):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, padding_side="right")
        self.bos_token = self.tokenizer.bos_token or self.tokenizer.cls_token
        self.eos_token = self.tokenizer.eos_token or self.tokenizer.sep_token
        if self.bos_token is None:
            self.bos_token = ""
        if self.eos_token is None:
            self.eos_token = ""

    def tokenize(self, texts: List[str]) -> dict:
        processed_texts = []

        for text in texts:
            combined_text = f"{self.bos_token}{text}{self.eos_token}"
            processed_texts.append(combined_text)

        model_inputs = self.tokenizer(
            processed_texts,
            return_tensors='pt',
            add_special_tokens=False,
            padding=True
        )

        if 'token_type_ids' in model_inputs:
            model_inputs.pop('token_type_ids')

        return model_inputs


class PerplexityCalculator:
    def __init__(self, model_loader, tokenizer):
        self.model = model_loader.load_model()
        self.tokenizer = tokenizer
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

    def get_perplexity(
        self,
        input_texts: Union[str, List[str]],
        batch_size: int = 32
    ) -> Union[float, List[float]]:

        single_input = isinstance(input_texts, str)
        input_texts = [input_texts] if single_input else input_texts

        loss_list = []
        num_texts = len(input_texts)
        batches = num_texts // batch_size + (num_texts % batch_size != 0)
        with torch.no_grad():
            for j in range(batches):
                start_idx = j * batch_size
                end_idx = (j + 1) * batch_size
                input_batch = input_texts[start_idx:end_idx]

                sequence_loss = self._compute_sequence_loss(input_batch)
                loss_list.extend(sequence_loss)

        ppl = [exp(i) for i in loss_list]

        return ppl[0] if single_input else ppl

    def _compute_sequence_loss(self, input_batch):
        with torch.no_grad():
            model_inputs = self.tokenizer.tokenize(input_batch)
            model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}

            output = self.model(**model_inputs, use_cache=False)
            logits = output['logits']

            label = model_inputs['input_ids']
            if hasattr(self.model.config, 'pad_token_id') and self.model.config.pad_token_id is not None:
                label[label == self.model.config.pad_token_id] = PAD_TOKEN_LABEL_ID

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = label[..., 1:].contiguous()

            token_loss = self.loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view(len(logits), -1)

            valid_length = (shift_labels != PAD_TOKEN_LABEL_ID).sum(dim=-1)
            sequence_loss = torch.sum(token_loss, -1) / valid_length
        return sequence_loss.cpu().tolist()


model_path = "/kaggle/input/gemma-2/transformers/gemma-2-9b/2"
model_loader = HuggingFaceModelLoader(model_path=model_path, load_in_4bit=False, device_map='auto')
tokenizer = HuggingFaceTokenizer(model_path)
scorer = PerplexityCalculator(model_loader, tokenizer)


from pprint import pprint

batch = ['reindeer mistletoe elf gingerbread family advent scrooge chimney fireplace ornament']*64

scores = scorer.get_perplexity(batch, batch_size=4)
pprint(scores)




