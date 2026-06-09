import os
import time
import tensorflow as tf
import keras
import keras_nlp
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import json

# --- Kaggle 환경 변수 확인 ---
print("Kaggle 환경 변수:")
for k, v in os.environ.items():
    if "TPU" in k:
        print(f"{k}={v}")

# --- 파일 경로 설정 ---
PARK_FILE = "/kaggle/input/txttxt/park.txt" # Place holder
YOON_FILE = "/kaggle/input/txttxt/yoon.txt" # Place holder

# --- Constants (Matching Cakeboss setup) ---
TOKEN_LIMIT = 512
LR_VALUE = 1e-4 # Matching Cakeboss
TRAIN_EPOCH = 1 # Matching Cakeboss
MODEL_ID = "gemma2_instruct_2b_en"
BATCH_SIZE = 1 # Enforce batch size of 1
GRADIENT_ACCUMULATION_STEPS = 2 # Matching Cakeboss
LORA_RANK = 4 # Matching Cakeboss

# --- Data Preprocessing ---
def preprocess_text(text):
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)
    text = text.replace('\n', ' ').strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def load_and_preprocess_data(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return preprocess_text(text)
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found.")
        exit()

# --- Model Loading with KerasNLP ---
def load_gemma_model_with_tokenizer(model_path):
    try:
        gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(model_path)
        gemma_lm.backbone.enable_lora(rank=LORA_RANK)
        print("Model loaded successfully.")
        return gemma_lm
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# --- Dataset Preparation ---
def prepare_dataset(text, sequence_length, batch_size, preprocessor):
    print("Preparing dataset...")
    start_time = time.time()

    # 텍스트를 토큰화
    tokenized_output = preprocessor([text])
    print(f"Tokenized output type: {type(tokenized_output)}")
    print(f"Tokenized output content: {tokenized_output}")

    token_ids = tokenized_output[0]
    padding_mask = tokenized_output[1]


    # Create dataset from the full tensor
    dataset = tf.data.Dataset.from_tensor_slices({
        "token_ids": token_ids[:sequence_length],
        "attention_mask": padding_mask[:sequence_length]
    })

    def create_dataset_element(data):
        input_ids = data["token_ids"][:-1]
        attention_mask = data["attention_mask"][:-1]
        target_ids = data["token_ids"][1:]
        return {"token_ids": input_ids, "attention_mask": attention_mask}, target_ids

    dataset = dataset.map(create_dataset_element)
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    end_time = time.time()
    print(f"Dataset preparation took: {end_time - start_time:.2f} seconds")
    return dataset

# --- Training Setup ---
def compile_model(model, lr_value):
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_value,
        weight_decay=0.01,
        clipnorm=1.0
    )
    optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])
    model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            weighted_metrics=["accuracy"]
        )
    print("Model compiled successfully.")

def setup_tensorboard_callback():
    return tf.keras.callbacks.TensorBoard(log_dir="./logs")

# --- Training Loop ---
def train_model(model, train_dataset, train_epoch, tensorboard_callback):
    print("Starting training...")
    model.fit(train_dataset, epochs=train_epoch, callbacks=[tensorboard_callback])
    print("Training finished.")

# --- Inference ---
def generate_text(prompt, model, max_length=2048, output_file=None):
    start_of_turn_user = "<start_of_turn>user\n"
    start_of_turn_model = "<start_of_turn>model\n"
    end_of_turn = "<end_of_turn>\n"

    prompt = start_of_turn_user + prompt + end_of_turn + start_of_turn_model
    print("Input Prompt:", prompt)
    start = time.time()
    output = model.generate(prompt, max_length=max_length)
    end = time.time()
    generated_text = output
    print(f"Total time taken: {end - start:.2f} seconds")
    print("\nGenerated Text:")
    print(generated_text)
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(generated_text)
        print(f"\nGenerated text saved to {output_file}")

# --- Evaluation ---
def evaluate_generated_text(yoon_text, output_file_path):
    print("\n=== Evaluation Started ===")
    print("\n=== Subjective Evaluation ===")
    print("1. How similar is the generated text to the style and reasoning of the training data?")
    print("2. How well does the generated text reflect the objective of the model?")
    print("3. Is the flow and meaning of the generated text natural?")
    print("4. Does the generated text maintain consistency and logical flow?")
    try:
        with open(output_file_path, "r", encoding="utf-8") as f:
            generated_text = f.read()
    except FileNotFoundError:
        print(f"Error: Generated text file '{output_file_path}' not found.")
        return
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([yoon_text, generated_text]) # Place holder
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    print(f"\n=== Objective Evaluation ===")
    print(f"Cosine Similarity between generated text and input text: {similarity:.4f}")
    print("\n=== Semantic Coherence Evaluation ===")
    print("Verifying if the generated text consistently maintains the presented basis and arguments.")
    print("Specifically, reviewing if the content is logically connected.")
    print("\n=== Evaluation Finished ===")

# --- Main Execution ---
if __name__ == "__main__":
    print("Available files in the input directory:")
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

    # --- GPU 설정 ---
    strategy = tf.distribute.get_strategy()
    print("REPLICAS: ", strategy.num_replicas_in_sync)

    print("Starting data preprocessing...")
    park_data = load_and_preprocess_data(PARK_FILE) # Place holder
    yoon_text = load_and_preprocess_data(YOON_FILE) # Place holder
    print("Data preprocessing finished.")

    print("Starting model loading...")
    gemma_lm = load_gemma_model_with_tokenizer(MODEL_ID)
    preprocessor = gemma_lm.preprocessor
    print("Model loading finished.")

    print("Starting dataset preparation...")
    # BATCH_SIZE is already set to 1, so no change needed here
    train_dataset = prepare_dataset(park_data, TOKEN_LIMIT, BATCH_SIZE, preprocessor)
    print("Dataset preparation finished.")

    print("Starting model compilation...")
    compile_model(gemma_lm, LR_VALUE)
    print("Model compilation finished.")

    print("Starting callback setup...")
    tensorboard_callback = setup_tensorboard_callback()
    print("Callback setup finished.")

    print("Starting model training...")
    train_model(gemma_lm, train_dataset, TRAIN_EPOCH, tensorboard_callback)
    print("Model training finished.")

    prompt = f"Write a text based on the training data." # Place holder
    output_file_path = "./generated_kerasnlp.txt" # Place holder

    print("Starting text generation...")
    generate_text(prompt, gemma_lm, output_file=output_file_path)
    print("Text generation finished.")

    print("Starting generated text evaluation...")
    evaluate_generated_text(yoon_text, output_file_path) # Place holder
    print("Generated text evaluation finished.")

    print("\nEnd of the process!")

