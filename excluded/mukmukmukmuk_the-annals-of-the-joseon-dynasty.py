!pip install -q -U keras-nlp tensorflow-text tensorflow-cpu
!pip install nltk
!pip install bert-score
!pip install rouge


import jax
import os


# Check available devices
jax.devices()


# Configure Keras backend to use JAX
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"


import keras
import keras_nlp
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score as bert_score
from rouge import Rouge
import matplotlib.pyplot as plt


# Define device mesh
device_mesh = keras.distribution.DeviceMesh(
    (1, 4),
    ["batch", "model"],
    devices=keras.distribution.list_devices()[:4],
)


# Define layout mapping
model_dim = "model"

layout_map = keras.distribution.LayoutMap(device_mesh)

layout_map["token_embedding/embeddings"] = (model_dim, None)
layout_map["decoder_block.*attention.*(query|key|value)/kernel"] = (model_dim, None, None)
layout_map["decoder_block.*attention_output/kernel"] = (model_dim, None, None)
layout_map["decoder_block.*ffw_gating.*/kernel"] = (None, model_dim)
layout_map["decoder_block.*ffw_linear/kernel"] = (model_dim, None)


# Set up model parallelism
model_parallel = keras.distribution.ModelParallel(
    layout_map=layout_map,
    batch_dim_name="batch",
)

keras.distribution.set_distribution(model_parallel)


df = pd.read_csv("/kaggle/input/the-joseon-wangjo-sillok/dataset.csv")


df.head(3)


def format_data_for_training(row):
    """
    Transforms each row of the dataset into a format suitable for training.

    Args:
        row (pd.Series): A single row of the dataset.

    Returns:
        dict: A dictionary containing the keys 'instruction', 'input', and 'output'.
    """
    instruction = (
        "Translate the old Korean text to modern Korean, ensuring accuracy and tone preservation."
    )
    input_text = (
        f"Old Title (구역 제목): {row['구역 제목']}\n"
        f"Old Text (구역 문장):\n{row['구역 문장']}\n"
        f"Notes (구역 주석):\n{row['구역 주석']}"
    )
    output_text = (
        f"Translated Title (신역 제목): {row['신역 제목']}\n"
        f"Translated Text (신역 문장):\n{row['신역 문장']}"
    )
    return {
        "instruction": instruction.strip(),
        "input": input_text.strip(),
        "output": output_text.strip()
    }

# apply formatting
formatted_data = df.apply(format_data_for_training, axis=1).tolist()


train_data, temp_data = train_test_split(formatted_data, test_size=0.1, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)


train_data[0]


def template_setting(data):
    """
    Convert data into proper template format

    Args:
        data (list of dict): Training data list

    Returns:
        list of str: converted data list
    """
    template_data = []
    for item in data:
        template = f"Instruction: {item['instruction']}\n\nInput:\n{item['input']}\n\nOutput:\n{item['output']}\n"
        template_data.append(template)
    return template_data

train_template = template_setting(train_data)
print(train_template[0])


# Load the pre-trained Gemma Causal Language Model
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("/kaggle/input/gemma2/keras/gemma2_instruct_2b_en/2")


# Access and inspect the first decoder block's weights.
decoder_block_1 = gemma_lm.backbone.get_layer('decoder_block_1')
print(type(decoder_block_1))

for variable in decoder_block_1.weights:
  print(f'{variable.path:<48}  {str(variable.shape):<14}  {str(variable.value.sharding.spec)}')


# Activate LoRA
gemma_lm.backbone.enable_lora(rank=8)


# compile the model
gemma_lm.preprocessor.sequence_length = 2048
gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=3e-5),
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
gemma_lm.summary()


def test_model(model, parsed_test_data, max_length=2048):
    """
    Perform inference using the model on the test data.

    Args:
        model: The pre-trained model for inference.
        parsed_test_data (list): List of dictionaries containing test instructions, inputs, and outputs.
        max_length (int): Maximum length for model-generated output.

    Returns:
        list: A list of dictionaries containing inputs, correct outputs, and model predictions.
    """
    results = []
    for item in parsed_test_data:  # Test with a subset of the data (e.g., first 5 examples)
        # Prepare model input
        prompt = f"Instruction: {item['instruction']}\n\nInput:\n{item['input']}\n\nOutput:\n"
        # Perform model inference
        model_response = model.generate(prompt, max_length=max_length)  # Adjust if generate() requires specific params
        # Store results
        results.append({
            "input": item["input"],
            "correct_output": item["output"],
            "model_response": model_response.strip()
        })
    return results


# Run inference on test data
test_results = test_model(gemma_lm, test_data[:1])

# Display results
for idx, result in enumerate(test_results):
    print(f"Example {idx + 1}:")
    print(f"Input:\n{result['input']}\n\n")
    print(f"Correct Output:\n{result['correct_output']}\n\n")
    print(f"Model Response:\n{result['model_response']}")
    print("-" * 50)


def evaluate_inference_results(results):
    """
    Evaluate inference results by comparing model responses with correct outputs.

    Args:
        results (list): List of dictionaries with keys: 'input', 'correct_output', 'model_response'.

    Returns:
        dict: Evaluation scores including BLEU, ROUGE, and BERTScore.
    """
    # Extract the 'Output:' part of model_response and correct_output
    correct_outputs = [item['correct_output'].split('Output:')[-1].strip() for item in results]
    model_responses = [item['model_response'].split('Output:')[-1].strip() for item in results]

    # Initialize scores
    evaluation_scores = {}

    # BLEU Score
    bleu_scores = [sentence_bleu([ref.split()], pred.split()) for ref, pred in zip(correct_outputs, model_responses)]
    evaluation_scores['BLEU'] = sum(bleu_scores) / len(bleu_scores)

    # ROUGE Score
    rouge = Rouge()
    rouge_scores = rouge.get_scores(model_responses, correct_outputs, avg=True)
    evaluation_scores['ROUGE'] = rouge_scores

    # BERTScore
    bert_p, bert_r, bert_f1 = bert_score(model_responses, correct_outputs, lang="ko", verbose=True)
    evaluation_scores['BERTScore'] = {
        'Precision': bert_p.mean().item(),
        'Recall': bert_r.mean().item(),
        'F1': bert_f1.mean().item()
    }

    return evaluation_scores

scores = evaluate_inference_results(test_results)


def print_evaluation_scores(scores):
    """
    Nicely formatted printout of evaluation scores.

    Args:
        scores (dict): Dictionary containing BLEU, ROUGE, and BERTScore.
    """
    print("=" * 40)
    print("Evaluation Results")
    print("=" * 40)
    print(f"BLEU Score: {scores['BLEU']:.2f}")
    print("\nROUGE Scores:")
    print(f"  ROUGE-1 (Precision): {scores['ROUGE']['rouge-1']['p']:.4f}")
    print(f"  ROUGE-1 (Recall): {scores['ROUGE']['rouge-1']['r']:.4f}")
    print(f"  ROUGE-1 (F1): {scores['ROUGE']['rouge-1']['f']:.4f}")
    print(f"  ROUGE-2 (Precision): {scores['ROUGE']['rouge-2']['p']:.4f}")
    print(f"  ROUGE-2 (Recall): {scores['ROUGE']['rouge-2']['r']:.4f}")
    print(f"  ROUGE-2 (F1): {scores['ROUGE']['rouge-2']['f']:.4f}")
    print(f"  ROUGE-L (Precision): {scores['ROUGE']['rouge-l']['p']:.4f}")
    print(f"  ROUGE-L (Recall): {scores['ROUGE']['rouge-l']['r']:.4f}")
    print(f"  ROUGE-L (F1): {scores['ROUGE']['rouge-l']['f']:.4f}")
    print("\nBERTScore:")
    print(f"  Precision: {scores['BERTScore']['Precision']:.4f}")
    print(f"  Recall: {scores['BERTScore']['Recall']:.4f}")
    print(f"  F1: {scores['BERTScore']['F1']:.4f}")
    print("=" * 40)

print_evaluation_scores(scores)


# Validation 데이터 준비
def prepare_validation_data(val_data):
    """
    Prepare validation data in (input, output) format for model evaluation.

    Args:
        val_data (list): Validation data as a list of dictionaries.

    Returns:
        tuple: Tuple containing inputs (x_val) and correct outputs (y_val).
    """
    x_val = []
    y_val = []
    for item in val_data:
        # 입력 텍스트
        x_val.append(f"Instruction: {item['instruction']}\n\nInput:\n{item['input']}\n\nOutput:\n")
        # 정답 텍스트
        y_val.append(item['output'])
    return x_val, y_val

# Validation 데이터 변환
x_val, y_val = prepare_validation_data(val_data)


# EarlyStopping Callback
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)


train_history=gemma_lm.fit(
    x=train_template,
    validation_data=(x_val, y_val),
    batch_size=2,
    epochs=50,
    callbacks=[early_stopping],
    verbose=2
)


plt.figure(figsize=(8, 5))
plt.plot(train_history.history['loss'], label='Train Loss')
plt.plot(train_history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train / Validation Loss')
plt.legend()
plt.show()


test_results = test_model(gemma_lm, test_data)


test_results[0]


def parse_results(test_results):
    """
    Parse test results into a structured DataFrame.

    Args:
        test_results (list): List of dictionaries containing 'input', 'correct_output', and 'predicted_output'.

    Returns:
        pd.DataFrame: DataFrame with columns for parsed data.
    """
    parsed_data = []

    for result in test_results:
        # Original Input
        input_text = result["input"]
        
        # Correct Output (split into title and text)
        correct_parts = result["correct_output"].split("\n")
        correct_title = correct_parts[0].replace("Translated Title (신역 제목):", "").strip() if len(correct_parts) > 0 else ""
        correct_text = correct_parts[2].strip() if len(correct_parts) > 2 else ""

        # Predicted Output (split into title and text)
        response_parts = result["model_response"].split("\n\n")
        response_input = response_parts[1].replace("Input:","").strip() if len(response_parts) > 1 else ""
        response_title = response_parts[2].replace("Output:", "").split("\n")[1].replace("Translated Title (신역 제목):", "").strip() if len(response_parts) > 2 else ""
        response_text = response_parts[2].replace("Output:", "").split("\n")[3].strip() if len(response_parts) > 2 and len(response_parts[2].split("\n")) > 3 else ""

        # Append parsed data
        parsed_data.append({
            "input": input_text,
            "correct_title": correct_title,
            "correct_text": correct_text,
            "response_input": response_input,
            "response_title": response_title,
            "response_text": response_text
        })

    return pd.DataFrame(parsed_data)


# Parse results
results_df = parse_results(test_results)


results_df.iloc[0]


# save results in csv format
results_df.to_csv("results.csv", index=False, encoding="utf-8-sig")


for idx, result in enumerate(test_results[:3]):
    print(f"Test {idx + 1}:")
    print(f"Input:\n{result['input']}")
    print(f"Correct Output:\n{result['correct_output']}")
    print(f"Model Response:\n{result['model_response']}\n")


def calculate_scores(results_df):
    """
    Calculate BLEU, BERTScore, and ROUGE scores for titles and content.

    Args:
        results_df (DataFrame): DataFrame containing correct and response texts.

    Returns:
        dict: Dictionary with calculated scores.
    """
    # BLEU Scores
    title_bleu_scores = [
        sentence_bleu([ref.split()], pred.split())
        for ref, pred in zip(results_df['correct_title'], results_df['response_title'])
    ]
    text_bleu_scores = [
        sentence_bleu([ref.split()], pred.split())
        for ref, pred in zip(results_df['correct_text'], results_df['response_text'])
    ]

    # BERTScore
    title_bert_scores = bert_score(
        results_df['response_title'].tolist(),
        results_df['correct_title'].tolist(),
        lang="ko",
        verbose=False
    )
    text_bert_scores = bert_score(
        results_df['response_text'].tolist(),
        results_df['correct_text'].tolist(),
        lang="ko",
        verbose=False
    )

    # ROUGE Scores
    rouge = Rouge()
    title_rouge_scores = rouge.get_scores(
        results_df['response_title'].tolist(),
        results_df['correct_title'].tolist(),
        avg=True
    )
    text_rouge_scores = rouge.get_scores(
        results_df['response_text'].tolist(),
        results_df['correct_text'].tolist(),
        avg=True
    )

    return {
        "BLEU": {
            "Title": sum(title_bleu_scores) / len(title_bleu_scores),
            "Content": sum(text_bleu_scores) / len(text_bleu_scores)
        },
        "BERTScore": {
            "Title": {
                "Precision": title_bert_scores[0].mean().item(),
                "Recall": title_bert_scores[1].mean().item(),
                "F1": title_bert_scores[2].mean().item()
            },
            "Content": {
                "Precision": text_bert_scores[0].mean().item(),
                "Recall": text_bert_scores[1].mean().item(),
                "F1": text_bert_scores[2].mean().item()
            }
        },
        "ROUGE": {
            "Title": title_rouge_scores,
            "Content": text_rouge_scores
        }
    }


def print_scores(scores):
    """
    Nicely formatted printout of evaluation scores.

    Args:
        scores (dict): Dictionary containing BLEU, ROUGE, and BERTScore.
    """
    print("=" * 40)
    print("Evaluation Results")
    print("=" * 40)

    # BLEU Scores
    print(f"BLEU Scores:")
    print(f"  Title: {scores['BLEU']['Title']:.2f}")
    print(f"  Content: {scores['BLEU']['Content']:.2f}")

    # ROUGE Scores
    print("\nROUGE Scores:")
    print(f"  Title:")
    print(f"    ROUGE-1 (Precision): {scores['ROUGE']['Title']['rouge-1']['p']:.4f}")
    print(f"    ROUGE-1 (Recall): {scores['ROUGE']['Title']['rouge-1']['r']:.4f}")
    print(f"    ROUGE-1 (F1): {scores['ROUGE']['Title']['rouge-1']['f']:.4f}")
    print(f"    ROUGE-2 (Precision): {scores['ROUGE']['Title']['rouge-2']['p']:.4f}")
    print(f"    ROUGE-2 (Recall): {scores['ROUGE']['Title']['rouge-2']['r']:.4f}")
    print(f"    ROUGE-2 (F1): {scores['ROUGE']['Title']['rouge-2']['f']:.4f}")
    print(f"    ROUGE-L (Precision): {scores['ROUGE']['Title']['rouge-l']['p']:.4f}")
    print(f"    ROUGE-L (Recall): {scores['ROUGE']['Title']['rouge-l']['r']:.4f}")
    print(f"    ROUGE-L (F1): {scores['ROUGE']['Title']['rouge-l']['f']:.4f}")

    print(f"  Content:")
    print(f"    ROUGE-1 (Precision): {scores['ROUGE']['Content']['rouge-1']['p']:.4f}")
    print(f"    ROUGE-1 (Recall): {scores['ROUGE']['Content']['rouge-1']['r']:.4f}")
    print(f"    ROUGE-1 (F1): {scores['ROUGE']['Content']['rouge-1']['f']:.4f}")
    print(f"    ROUGE-2 (Precision): {scores['ROUGE']['Content']['rouge-2']['p']:.4f}")
    print(f"    ROUGE-2 (Recall): {scores['ROUGE']['Content']['rouge-2']['r']:.4f}")
    print(f"    ROUGE-2 (F1): {scores['ROUGE']['Content']['rouge-2']['f']:.4f}")
    print(f"    ROUGE-L (Precision): {scores['ROUGE']['Content']['rouge-l']['p']:.4f}")
    print(f"    ROUGE-L (Recall): {scores['ROUGE']['Content']['rouge-l']['r']:.4f}")
    print(f"    ROUGE-L (F1): {scores['ROUGE']['Content']['rouge-l']['f']:.4f}")

    # BERTScore
    print("\nBERTScore:")
    print(f"  Title:")
    print(f"    Precision: {scores['BERTScore']['Title']['Precision']:.4f}")
    print(f"    Recall: {scores['BERTScore']['Title']['Recall']:.4f}")
    print(f"    F1: {scores['BERTScore']['Title']['F1']:.4f}")
    print(f"  Content:")
    print(f"    Precision: {scores['BERTScore']['Content']['Precision']:.4f}")
    print(f"    Recall: {scores['BERTScore']['Content']['Recall']:.4f}")
    print(f"    F1: {scores['BERTScore']['Content']['F1']:.4f}")

    print("=" * 40)


scores = calculate_scores(results_df)
print_scores(scores)


gemma_lm.backbone.save_lora_weights("sillok-model.lora.h5")

