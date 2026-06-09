!pip install timm --upgrade
!pip install accelerate
!pip install git+https://github.com/huggingface/transformers.git


from datetime import datetime
import logging
import os
import sys
from typing import TextIO

import kagglehub
import psutil
from transformers import (
    AutoTokenizer,
    Gemma3nForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)
import torch


model_path = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")

class FakeLogger:
    """kind of looks like a real python logger, but won't compete with whatever 
    settings kaggle have for their real loggers. Also this will overwite the file
    every time it is started."""
    def __init__(self):
        self.logging_file = "/kaggle/working/performance.log"
        self.outstream = None

    def start(self):
        if self.outstream is not None:
            raise Exception("tried to start fake logging when it was already startred.")
        self.outstream = open(self.logging_file, mode="w")

    def stop(self):
        if self.outstream is None:
            raise Exception("Fake logging already stopped")
        self.outstream.close()
        self.outstream = None
        
    def info(self, message):
        if self.outstream is None:
            raise Exception("Fake logger not ready to accept input")
        self.outstream.write(message+"\n")

logger = FakeLogger()


class MemInfo:
    def __init__(self):
        self._pid = os.getpid()
        self._process = psutil.Process(self._pid)

    def rss_mb(self):
        return self._process.memory_info().rss / (1024**2)

    @property
    def pid(self):
        return self._pid

mem_info = MemInfo()


class ChatSession:
    def __init__(
        self, system_prompt: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer
    ):
        self._system_prompt = system_prompt
        self._model = model
        self._tokenizer = tokenizer
        self.reset()

    def reset(self):
        self._history = [{"role": "system", "content": self._system_prompt}]

    def _add(self, role, content):
        self._history.append({"role": role, "content": content})

    def prompt(self, prompt):
        self._add("user", prompt)
        encoded = self._tokenizer.apply_chat_template(
            self._history, return_tensors="pt", add_generation_prompt=True
        ).to(self._model.device)
        logger.info(f"About to generate using {len(encoded[0])} input tokens")
        logger.info(f"Memory usage is currently {mem_info.rss_mb():.2f}MB.")
        start = datetime.now()
        response = self._model.generate(encoded, max_length=1000)
        response = response[0][len(encoded[0]) :]
        delta = datetime.now() - start
        logger.info(f"Produced {len(response)} tokens in {delta.seconds} seconds")
        logger.info(f"Memory usage is now {mem_info.rss_mb():2f}MB.")
        decoded = self._tokenizer.decode(response, skip_special_tokens=True)
        self._add(decoded, "model")
        return decoded

    def start_session(self):
        print(
            "Type /bye to exit, /reset to reset the "
            "conversation, or type your question.\n"
        )
        while True:
            print("you: ", end="")
            prompt = input().strip()
            if prompt == "/bye":
                break
            if prompt == "/reset":
                self.reset()
                continue

            response = self.prompt(prompt)
            print(f"model: {response}\n")


def make_model():
    return Gemma3nForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).eval()


def make_tokenizer():
    return AutoTokenizer.from_pretrained(model_path)


def get_layer_sizes(model: PreTrainedModel):
    layer_sizes = {}
    total_size = 0

    for name, param in model.named_parameters():
        layer_size = param.numel() * param.element_size()
        total_size += layer_size
        layer_sizes[name] = (param.numel(), layer_size, param.dtype)

    return layer_sizes, total_size



logger.start()

logger.info(
    "Starting new run. Memory usage statistics refer to resident set size (RSS)"
)
logger.info(f"Before loading model, RSS memory usage is {mem_info.rss_mb():0.2f}MB")
model = make_model()
logger.info(f"The model loaded on {model.device}")
logger.info(f"Memory usage is now {mem_info.rss_mb():.2f}MB")

tokenizer = make_tokenizer()
logger.info(f"Tokenizer loaded, memory is now {mem_info.rss_mb():.2f}")

_, total_size = get_layer_sizes(model)
logger.info(
    f"Model paramaters appear to take up a total of {total_size / 1024**2:.2f} MB"
)

chat = ChatSession(
    "you are a helpful assistant, here to answer the users questions.",
    model,
    tokenizer,
)


def prompt(msg):
    print(f"you: {msg}")
    print(f"bot: {chat.prompt(msg)}")

prompt("Hello, my name is Brian.")
prompt("Do you remember my name?")

logger.info("Chat session ended.")

logger.stop()


!cat /kaggle/working/performance.log

