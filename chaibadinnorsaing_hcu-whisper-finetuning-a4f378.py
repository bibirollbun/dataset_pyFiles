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


pip install bitsandbytes


os.environ["WANDB_MODE"] = "offline"


device = "cuda" if torch.cuda.is_available() else "cpu"
device


model_name = "openai/whisper-base"
lang = "th"
task = "transcribe"


model = WhisperForConditionalGeneration.from_pretrained(model_name)
processor = WhisperProcessor.from_pretrained(model_name)

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

        # เช็คว่ามีคอลัมน์ "audio" และ "sentence" หรือไม่
        if "audio" not in self.ann_file.columns or "sentence" not in self.ann_file.columns:
            raise ValueError(f"CSV file missing 'audio' or 'sentence' column. Found: {self.ann_file.columns}")

        self.ann_file["audio"] = self.ann_file["audio"].apply(lambda x: os.path.join(audio_folder_path, x))

        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

    def __len__(self):
        return self.ann_file.shape[0]
    
    def __getitem__(self, idx):
        row = self.ann_file.iloc[idx]

        audio_path = row["audio"]
        sentence = row["sentence"]

        # เช็คว่าไฟล์เสียงมีอยู่จริง
        if not os.path.exists(audio_path):
            print(f"⚠️ Warning: {audio_path} not found. Skipping.")
            return None  # หรือกำหนดค่า default

        # โหลดเสียง
        array, sr = librosa.load(path=audio_path, sr=self.target_sr)
        if array is None or len(array) == 0:
            print(f"⚠️ Warning: {audio_path} failed to load.")
            return None

        input_features = self.feature_extractor(array, sampling_rate=sr).input_features[0]

        # เช็คว่า sentence ถูกต้องหรือไม่
        if pd.isna(sentence) or not isinstance(sentence, str):
            print(f"⚠️ Warning: Missing sentence at index {idx}. Skipping.")
            return None

        labels = self.tokenizer(sentence, truncation=True, max_length=448).input_ids

        return dict(
            input_features=input_features,
            labels=labels
        )


from transformers import WhisperFeatureExtractor, WhisperTokenizer

# โหลดโมเดล
model_name = "openai/whisper-base"  # ตรวจสอบชื่อโมเดลให้ถูกต้อง
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
tokenizer = WhisperTokenizer.from_pretrained(model_name)

# ตรวจสอบว่า tokenizer รองรับ `set_prefix_tokens()`
if hasattr(tokenizer, "set_prefix_tokens"):
    tokenizer.set_prefix_tokens(language=lang, task=task)

# ตรวจสอบว่าทุกอย่างโหลดได้ถูกต้อง
assert feature_extractor is not None, "Feature extractor ไม่สามารถโหลดได้"
assert tokenizer is not None, "Tokenizer ไม่สามารถโหลดได้"

# สร้าง Dataset
datasets = {
    phase: CustomDataset(
        ann_file_path=f"/kaggle/working/{phase}.csv",
        audio_folder_path=f"/kaggle/input/hcu-speech-recognition-challenge-2025/{phase}",
        target_sr=16_000,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    ) for phase in ["train", "dev"]
}

print("✅ Dataset ถูกโหลดเรียบร้อยแล้ว!")



@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor, decoder_start_token_id):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features):
        # เตรียม input_features (เสียง) และ padding
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # เตรียม labels (ตัวอักษร) และ padding
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # ตรวจสอบว่า tokenizer มี attention_mask หรือไม่
        if "attention_mask" in labels_batch:
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        else:
            labels = labels_batch["input_ids"]

        # ลบ decoder start token ถ้าอยู่ที่ตำแหน่งแรก
        if labels.size(1) > 0 and (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
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
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=3.0,
    learning_rate=1e-4,
    gradient_checkpointing=True,
    fp16=True,  # ปิด fp16 ถ้า bf16 เปิดอยู่
    bf16=False,   # เปิด bf16 ถ้า GPU รองรับ
    optim="adamw_8bit",
    eval_strategy="epoch",
    per_device_eval_batch_size=4,
    predict_with_generate=True,  # แก้ syntax error
    generation_max_length=256,
    save_strategy="epoch",
    save_total_limit=5,
    logging_steps=50,
    report_to=None,
    push_to_hub=False,
)



# Make sure to pass the correct tokenizer and feature extractor
trainer = Seq2SeqTrainer(
    args=training_args,  # Your training arguments (as defined before)
    model=model,  # Your model
    train_dataset=datasets["train"],  # Your training dataset
    eval_dataset=datasets["dev"],  # Your evaluation dataset
    data_collator=data_collator,  # Data collator (for batching)
    compute_metrics=compute_metrics,  # Function to compute metrics (e.g., accuracy)
    tokenizer=processor.tokenizer,  # Correct tokenizer for text tasks
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
def read_audio_to_array(audio_path: str, target_sr: int) -> np.ndarray:
    try:
        array, sr = librosa.load(path=audio_path, sr=target_sr)  # ✅ แก้ typo
        return array
    except Exception as e:
        print(f"Error loading audio from {audio_path}: {e}")
        return np.array([])  # Return an empty array on error

# Batch transcribe function
def batch_transcribe(
    model,
    processor,
    audio_path_list: List[str],
    batch_size: int,  # ✅ ต้องมี batch_size ตรงนี้
    sampling_rate: int,
    torch_dtype: torch.dtype,
    device: str
):

    model.eval()  # Set model to evaluation mode
    transcriptions = []

    # Move model to the appropriate device
    model.to(device)

    # Process audio files in batches
    for i in tqdm(range(0, len(audio_path_list), batch_size)):
        audio_batch = audio_path_list[i : i + batch_size]
        
        # Read and preprocess audio files
        array_list = [
            read_audio_to_array(audio_path, sampling_rate)
                for audio_path in audio_batch
        ]
        
        # Filter out empty arrays (error case during loading)
        array_list = [arr for arr in array_list if arr.size > 0]

        if len(array_list) == 0:
            print(f"Skipping batch {i // batch_size + 1} due to errors in loading audio.")
            continue

        # Convert list of numpy arrays to tensor
        input_features = processor(
            array_list, 
            sampling_rate=sampling_rate, 
            return_tensors="pt"
        ).input_features
        input_features = input_features.to(device, dtype=torch_dtype)

        # Generate transcriptions
        with torch.no_grad():
            output_ids = model.generate(
                input_features,
                max_new_tokens=50,  
                temperature=0.7,  
                top_k=50,  
                top_p=0.95  
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


# import os
# import numpy as np
# import torch
# import librosa

# # ✅ ฟังก์ชันโหลดเสียง (แก้ไขให้คืนค่า NumPy array)
# def load_audio(audio_path, sampling_rate=16000):
#     audio, _ = librosa.load(audio_path, sr=sampling_rate)
#     return audio

# # ✅ ฟังก์ชัน Padding mel spectrogram
# def pad_mel(mel, target_length=3000):
#     if mel.shape[-1] < target_length:
#         pad_width = target_length - mel.shape[-1]
#         mel = np.pad(mel, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
#     return mel

# # ✅ สร้าง list ของไฟล์เสียงที่มี path เต็ม
# audio_path_list = df_test.audio.apply(
#     lambda x: os.path.join(test_folder_path, x)
# ).to_list()

# # ✅ ฟังก์ชัน batch_transcribe ที่ปรับปรุงเพื่อลด validation loss
# def batch_transcribe(model, processor, audio_path_list, batch_size, sampling_rate, device):
#     transcriptions = []

#     for i in range(0, len(audio_path_list), batch_size):
#         batch_audio_paths = audio_path_list[i : i + batch_size]

#         # ✅ โหลดไฟล์เสียง
#         batch_audio = [load_audio(path, sampling_rate) for path in batch_audio_paths]

#         # ✅ สร้าง mel spectrogram และ padding
#         batch_mels = []
#         for audio in batch_audio:
#             inputs = processor(audio, sampling_rate=sampling_rate, return_tensors="pt")
#             mel = inputs.input_features.squeeze(0).numpy()
#             mel = pad_mel(mel)
#             batch_mels.append(mel)

#         # ✅ แปลง mel spectrogram เป็น tensor และ float16
#         batch_mels = torch.tensor(np.array(batch_mels)).to(device, dtype=torch.float16)

#         # ✅ ใช้ beam search และ temperature scaling เพื่อลด validation loss
#         with torch.no_grad():
#             generated_ids = model.generate(
#                 batch_mels,
#                 max_length=300,  # จำกัดความยาวของข้อความที่ถอดรหัส
#                 num_beams=5,  # Beam search เพื่อให้ผลลัพธ์แม่นยำขึ้น
#                 temperature=0.7,  # ทำให้โมเดลมีความหลากหลายในการทำนาย
#                 repetition_penalty=1.2,  # ลดการซ้ำของคำที่ไม่จำเป็น
#                 early_stopping=True  # หยุดเมื่อแน่ใจว่าถอดรหัสครบแล้ว
#             )

#         batch_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

#         transcriptions.extend(batch_text)

#     return transcriptions

# # ✅ เรียกใช้ batch_transcribe พร้อมตั้งค่าที่เหมาะสม
# transcriptions = batch_transcribe(
#     model=model,
#     processor=processor,
#     audio_path_list=audio_path_list,
#     batch_size=16,  # สามารถปรับได้ตามขนาด GPU
#     sampling_rate=16000,  # ใช้ค่าเดียวกับที่โมเดลต้องการ
#     device="cuda" if torch.cuda.is_available() else "cpu"  # ใช้ GPU ถ้ามี
# )



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


# กำหนดค่าตัวแปรก่อนใช้งาน
batch_size = 8
sampling_rate = 16000
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"batch_size: {batch_size}, sampling_rate: {sampling_rate}, device: {device}")

transcriptions = []

for i in tqdm(range(0, len(audio_path_list), batch_size)):
    audio_batch = audio_path_list[i : i + batch_size]

    # โหลดเสียงและตรวจสอบค่า
    array_list = [read_audio_to_array(audio, sampling_rate) for audio in audio_batch]
    array_list = [arr for arr in array_list if arr.size > 0]  # ลบเสียงที่โหลดไม่สำเร็จ

    print(f"Batch {i // batch_size + 1}: โหลดเสียงสำเร็จ {len(array_list)} ไฟล์")

    if len(array_list) == 0:
        print("⚠️ ไม่มีไฟล์เสียงใน batch นี้, ข้ามไป")
        continue

    # แปลงเสียงเป็น features และส่งเข้าโมเดล
    input_features = processor(array_list, sampling_rate=sampling_rate, return_tensors="pt").input_features
    input_features = input_features.to(device, dtype=torch_dtype)

    with torch.no_grad():
        output_ids = model.generate(input_features, max_new_tokens=50)
        output_texts = processor.batch_decode(output_ids, skip_special_tokens=True)

    transcriptions.extend(output_texts)

print(f"✅ สรุปได้ transcriptions {len(transcriptions)} คำตอบ")



df_test["sentence"] = transcriptions
df_test["sentence"] = df_test.sentence.apply(
    lambda t: " ".join(word_tokenize(t.lower().replace(" ", ""), engine="newmm"))
)


df_test


df_test.to_csv("submission.csv", index=False)

