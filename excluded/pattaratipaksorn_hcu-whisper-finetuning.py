!pip install -q datasets transformers evaluate huggingface_hub jiwer pythainlp


!pip install -q -U "huggingface_hub[cli]"


import gc
import os
import evaluate
import numpy as np
import pandas as pd
import torch
import librosa
from typing import List
from torch.utils.data import Dataset
from datasets import load_dataset, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from tqdm.notebook import tqdm
from pythainlp.tokenize import word_tokenize
from huggingface_hub import login
from dataclasses import dataclass


os.environ["WANDB_MODE"] = "offline"


device = "cuda" if torch.cuda.is_available() else "cpu"
device


lang = "th"
task = "transcribe"
model_name = "openai/whisper-small"


model = WhisperForConditionalGeneration.from_pretrained(model_name)
model.generation_config.language = lang
model.generation_config.task = task
model.generation_config.forced_decoder_ids = None


for phase in ["train", "dev"]:
    
    drop_list = []

    df = pd.read_csv(f"/kaggle/input/hcu-speech-recognition-challenge-2025/{phase}.csv")

    for i in range(len(df)):
        audio_path = os.path.join(f"/kaggle/input/hcu-speech-recognition-challenge-2025/{phase}", df.audio[i])
        if not os.path.exists(audio_path):
            print(df.audio[i])
            drop_list.append(df.audio[i])
      
    df = df[~df.audio.isin(drop_list)].reset_index(drop=True)
    df.to_csv(f"/kaggle/working/{phase}.csv", index=False)


class CustomDataset(Dataset):
    def __init__(self, 
        ann_file_path: str, 
        audio_folder_path: str, 
        target_sr: int = 16000,
        feature_extractor: WhisperFeatureExtractor = None,
        tokenizer: WhisperTokenizer = None
    ):
        super(CustomDataset, self).__init__()
        self.audio_folder_path = audio_folder_path
        self.target_sr = target_sr
        self.ann_file = pd.read_csv(ann_file_path)
        self.ann_file["audio"] = self.ann_file.audio.apply(
            lambda x: os.path.join(audio_folder_path, x)
        )
        
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

    def __len__(self):
        return self.ann_file.shape[0]
    
    def __getitem__(self, idx):
        audio_path, sentence = self.ann_file.iloc[idx]
        
        array, sr = librosa.load(path=audio_path, sr=self.target_sr)
        input_features = self.feature_extractor(
            array, sampling_rate=sr
        ).input_features[0]

        labels = self.tokenizer(sentence, truncation=True, max_length=448).input_ids

        return dict(
            input_features=input_features,
            labels=labels
        )


feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
tokenizer = WhisperTokenizer.from_pretrained(model_name)
tokenizer.set_prefix_tokens(language=lang, task=task)

datasets = {
    phase: CustomDataset(
        ann_file_path=f"/kaggle/working/{phase}.csv",
        audio_folder_path=f"/kaggle/input/hcu-speech-recognition-challenge-2025/{phase}",
        target_sr=16_000,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    ) for phase in ["train", "dev"]
}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:

    def __init__(self, processor, decoder_start_token_id):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features):

        # Prepare input features by padding and converting to tensor
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Prepare labels by padding and converting to tensor
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Mask padding tokens in labels
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove the decoder start token
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


processor = WhisperProcessor.from_pretrained(model_name, language=lang, task=task)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)


metric = evaluate.load("cer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    cer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}


training_args = Seq2SeqTrainingArguments(
    output_dir="./model_output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    num_train_epochs=3.0,
    learning_rate=5e-5,
    gradient_checkpointing=True,
    fp16=True,
    bf16=False,
    optim="adamw_torch_fused", # adamw_torch_fused, adamw_8bit, adamw_torch, sgd
    eval_strategy="epoch",
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    generation_max_length=256,
    save_strategy="epoch",
    save_total_limit=5,
    logging_steps=50,
    report_to=None,
    push_to_hub=False,
)


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=datasets["train"],
    eval_dataset=datasets["dev"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor
)


trainer.train()


model.save_pretrained(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)


del model
gc.collect()
torch.cuda.empty_cache()


test_folder_path = "/kaggle/input/hcu-speech-recognition-challenge-2025/test"
df_test = pd.read_csv("/kaggle/input/hcu-speech-recognition-challenge-2025/test.csv")


# Read audio function
def read_audio_to_array(
    audio_path: str,
    target_sr: int,
) -> np.ndarray:
    
    # load audio
    array, sr = librosa.load(path=audio_path, sr=target_sr)
    
    return array

# Batch transcribe function
def batch_transcribe(
    model,
    processor,
    audio_path_list: str,
    batch_size: int,
    sampling_rate: int,
    torch_dtype: torch.dtype,
    device: str
) -> List[str]:

    model.eval()
    transcriptions = []

    # Process audio files in batches
    for i in tqdm(range(0, len(audio_path_list), batch_size)):
        audio_batch = audio_path_list[i : i + batch_size]
        
        # Read and preprocess audio files   
        array_list = [
            read_audio_to_array(audio_path, sampling_rate)
                for audio_path in audio_batch
        ]

        # Process audio arrays into input features
        input_features = processor(
            array_list, 
            sampling_rate=sampling_rate, 
            return_tensors="pt"
        ).input_features
        input_features = input_features.to(torch_dtype)
        
        # Generate transcriptions
        with torch.no_grad():
            output_ids = model.generate(
                input_features.to(device),
                max_new_tokens=20,
                temperature=0.4
            )
            output_texts = processor.batch_decode(output_ids, skip_special_tokens=True)

        transcriptions.extend(output_texts)
        
    return transcriptions


lang = "th"
task = "transcribe"
torch_dtype = torch.float16
model_path ="/kaggle/working/model_output"


# Load model & Processor
processor = WhisperProcessor.from_pretrained(model_path, lang=lang, task=task)
model = WhisperForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch_dtype)
model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=lang, task=task)
model = model.to(device)


audio_path_list = df_test.audio.apply(
    lambda x: os.path.join(test_folder_path, x)
).to_list()

transcriptions = batch_transcribe(
    model=model, 
    processor=processor, 
    audio_path_list=audio_path_list, 
    batch_size=16,
    sampling_rate=16_000,
    torch_dtype=torch_dtype, 
    device=device
)


df_test["sentence"] = transcriptions
df_test["sentence"] = df_test.sentence.apply(
    lambda t: " ".join(word_tokenize(t.lower().replace(" ", ""), engine="newmm"))
)


df_test


df_test.to_csv("submission.csv", index=False)

