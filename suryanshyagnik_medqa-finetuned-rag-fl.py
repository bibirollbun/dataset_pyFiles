%%capture
%pip install -q bitsandbytes
%pip install -q transformers
%pip install -q peft
%pip install -q accelerate
%pip install -q trl
%pip install -q torch
%pip install -q qdrant-client langchain pypdf sentence-transformers


!pip install langchain_community


%%capture
import os, torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig, TrainingArguments, pipeline
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from datasets import Dataset
from IPython.display import Markdown, display
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline


model = "/kaggle/input/m/google/gemma/transformers/2b-it/2"

bnbConfig = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model, quantization_config=bnbConfig, device_map="auto")

model = AutoModelForCausalLM.from_pretrained(
    model,
    device_map = "auto",
    quantization_config=bnbConfig
)


system =  "You are a skilled software engineer who consistently produces high-quality Python code."
user = "Write a Python code to display text in a star pattern."

prompt = f"System: {system} \n User: {user} \n AI: "
    
inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")

outputs = model.generate(**inputs, num_return_sequences=1, max_new_tokens=1000)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
Markdown(text.split("AI:")[1])


# Load dataset
data = pd.read_csv("/kaggle/input/dataset-python-question-answer/Dataset_Python_Question_Answer.csv")

# Split into three equal parts
split_ratio = len(data) // 3
data_1, data_2, data_3 = data[:split_ratio], data[split_ratio:2*split_ratio], data[2*split_ratio:]

# Convert to Hugging Face datasets
dataset_1 = Dataset.from_pandas(data_1)
dataset_2 = Dataset.from_pandas(data_2)
dataset_3 = Dataset.from_pandas(data_3)


def formatting_func(example):
    template = "Instruction:\n{instruction}\n\nResponse:\n{response}"
    line = template.format(instruction=example['Question'], response=example['Answer'])
    return [line]


import os
os.environ["WANDB_DISABLED"] = "true"


lora_config = LoraConfig(
    r = 8,
    target_modules = ["q_proj", "o_proj", "k_proj", "v_proj",
                      "gate_proj", "up_proj", "down_proj"],
    task_type = "CAUSAL_LM",
)


# Define training function
def fine_tune_model(model, dataset, output_dir):
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=50,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir=output_dir,
            optim="paged_adamw_8bit"
        ),
        peft_config=lora_config,
        formatting_func=formatting_func,
    )
    trainer.train()
    return trainer

# Fine-tune three separate models
fine_tune_model(model, dataset_1, "outputs_model_1")
fine_tune_model(model, dataset_2, "outputs_model_2")
fine_tune_model(model, dataset_3, "outputs_model_3")


system =  "You are a skilled software engineer who consistently produces high-quality Python code."
question =system + "What is the difference between a variable and an object"

prompt = f"Question: {question} \n Answer: "
    
inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")

outputs = model.generate(**inputs, num_return_sequences=1, max_new_tokens=512)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)

Markdown(text.split("Answer:")[1])


# Instantiate a PyPDFDirectoryLoader object with the specified directory path
pdf_loader = PyPDFDirectoryLoader("/kaggle/input/knowledge-base")

# Load PDF documents from the specified directory
pdfs = pdf_loader.load()


# import the HuggingFaceEmbeddings class, 
embeddings = HuggingFaceEmbeddings(
    # This argument specifies the pre-trained model name to be used for generating embeddings.
    # Here, "sentence-transformers/all-mpnet-base-v2" is a pre-trained sentence transformer model 
    # from the Sentence Transformers library (not Transformers).
    # Sentence transformer models are specifically trained to generate meaningful representations 
    # of sentences that capture semantic similarity.
    model_name="sentence-transformers/all-mpnet-base-v2",

    # This argument is likely specific to the HuggingFaceEmbeddings class and might 
    # not be present in the base Transformers library.
    # It sets the device to "cuda" to leverage the GPU for faster processing if available.
    model_kwargs={"device": "cuda"}
)


# Instantiate a RecursiveCharacterTextSplitter object with specified parameters
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Split documents into chunks using the RecursiveCharacterTextSplitter
all_splits = text_splitter.split_documents(pdfs)


# Create a Qdrant collection from the document splits
# For storing and searching document information we use a vector database called Qdrant. 

qdrant_collection = Qdrant.from_documents(
    all_splits,                # List of document splits
    embeddings,                # HuggingFaceEmbeddings object for generating embeddings
    location=":memory:",       # Location to store the collection (in memory)
    collection_name="all_documents"  # Name of the Qdrant collection
)


# Create a retriever
retriever = qdrant_collection.as_retriever()


# This code creates a pipeline for text generation using a pre-trained model (model) 
# and its tokenizer (tokenizer). It leverages mixed precision (torch.bfloat16) 
# for potentially faster inference and limits generated text to 512 tokens.
pipeline = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer,
    model_kwargs = {"torch.dtype": torch.bfloat16},
    max_new_tokens=512    
)


question = "What is the difference between a variable and an object"

message = [
    {"role": "user", "content": question},
]

prompt = pipeline.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

outputs = pipeline(
    prompt,
    max_new_tokens=512,
    add_special_tokens=True,
    do_sample=True,
    temperature=0.7,
    top_k=10,
    top_p=0.95
)
Markdown(outputs[0]["generated_text"][len(prompt):])


gemma_llm = HuggingFacePipeline(
    pipeline=pipeline,
    model_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 512,
        "add_special_tokens": True,
        "do_sample": True,
        "top_k": 10,
        "top_p": 0.95
    },
)
# Create a RetrievalQA object
qa = RetrievalQA.from_chain_type(
    llm=gemma_llm,  # Pass the text-generation pipeline object
    chain_type="stuff",
    retriever=retriever  # retriever object
)


question = "Write in detail about python"
message = [
    {"role": "user", "content": question},
]

prompt = pipeline.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, truncation=True)
result = qa.invoke(prompt)
Markdown(result['result'].split('Helpful Answer:')[1])


# Fine-tune three separate models and save them
trainer_1 = fine_tune_model(model, dataset_1, "outputs_model_1")
trainer_1.model.save_pretrained("model_1")  # Save model_1

trainer_2 = fine_tune_model(model, dataset_2, "outputs_model_2")
trainer_2.model.save_pretrained("model_2")  # Save model_2

trainer_3 = fine_tune_model(model, dataset_3, "outputs_model_3")
trainer_3.model.save_pretrained("model_3")  # Save model_3


def load_model(model_path, dtype=torch.float16, device="cpu"):
    """Load model with reduced precision and on CPU to save RAM."""
    return AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=device)


def federated_averaging(model_paths):
    """Perform federated averaging with memory optimization."""
    global_model = load_model(model_paths[0])
    global_model_state = global_model.state_dict()

    for key in global_model_state.keys():
        global_model_state[key] = global_model_state[key].to(torch.float32)  # Convert to float32 for accurate averaging

    num_models = len(model_paths)

    # Iterate through remaining models one by one to avoid memory overhead
    for model_path in model_paths[1:]:
        model = load_model(model_path)
        model_state = model.state_dict()

        for key in global_model_state.keys():
            global_model_state[key] += model_state[key].to(torch.float32)  # Accumulate in float32

        del model  # Free memory
        torch.cuda.empty_cache()

    # Compute final averaged parameters
    for key in global_model_state.keys():
        global_model_state[key] /= num_models  # Average across models

    # Reload the averaged weights into a model
    final_model = load_model(model_paths[0])  # Initialize from first model's structure
    final_model.load_state_dict(global_model_state)

    return final_model

# Define model paths instead of loading them all at once
model_paths = ["model_1", "model_2"]
global_model = federated_averaging(model_paths)


# Save the federated averaged model
save_path = "global_model"
global_model.save_pretrained(save_path)
print(f"Global model saved at: {save_path}")



# Load the global model
global_model = AutoModelForCausalLM.from_pretrained("global_model")

# Inspect the parameters
for name, param in global_model.named_parameters():
    print(f"{name}: {param.data}")


from transformers import pipeline

# Create the pipeline with a different variable name
text_generator = pipeline(
    "text-generation",
    model=global_model,
    tokenizer=tokenizer,
)

# Your example question
question = "What is the difference between a variable and an object"

# Create the message format
message = [
    {"role": "user", "content": question},
]

# Apply the chat template
prompt = text_generator.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

# Generate the output
outputs = text_generator(
    prompt,
    max_new_tokens=512,
    add_special_tokens=True,
    do_sample=True,
    temperature=0.7,
    top_k=10,
    top_p=0.95
)

# Display the result
Markdown(outputs[0]["generated_text"][len(prompt):])


import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from nltk.metrics import jaccard_distance
import numpy as np


# Download required NLTK data
nltk.download('punkt')

def calculate_metrics(generated_text, reference_text):
    """
    Calculate various metrics for text generation evaluation
    
    Args:
        generated_text (str): The model generated text
        reference_text (str): The ground truth reference text
    
    Returns:
        dict: Dictionary containing various metric scores
    """
    metrics = {}
    
    # Tokenize texts
    generated_tokens = word_tokenize(generated_text.lower())
    reference_tokens = [word_tokenize(reference_text.lower())]
    
    # BLEU Score
    smoother = SmoothingFunction().method1
    try:
        bleu_score = sentence_bleu(reference_tokens, generated_tokens, 
                                 smoothing_function=smoother)
        metrics['bleu'] = bleu_score
    except Exception as e:
        metrics['bleu'] = 0
        print(f"BLEU score calculation failed: {e}")

    # Jaccard Similarity (1 - distance)
    gen_set = set(generated_tokens)
    ref_set = set(reference_tokens[0])
    try:
        jaccard_sim = 1 - jaccard_distance(gen_set, ref_set)
        metrics['jaccard_similarity'] = jaccard_sim
    except Exception as e:
        metrics['jaccard_similarity'] = 0
        print(f"Jaccard calculation failed: {e}")
    
    # Token overlap ratio
    common_tokens = len(gen_set.intersection(ref_set))
    metrics['token_overlap'] = common_tokens / len(ref_set)
    
    # Length metrics
    metrics['generated_length'] = len(generated_tokens)
    metrics['reference_length'] = len(reference_tokens[0])
    metrics['length_ratio'] = len(generated_tokens) / len(reference_tokens[0])
    
    return metrics

def evaluate_model(text_generator, eval_data):
    """
    Evaluate the model on a set of test examples
    
    Args:
        text_generator: The pipeline instance
        eval_data: List of tuples containing (question, reference_answer)
    
    Returns:
        dict: Aggregated metrics across all examples
    """
    all_metrics = []
    
    for question, reference in eval_data:
        # Generate response
        message = [{"role": "user", "content": question}]
        prompt = text_generator.tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        
        outputs = text_generator(
            prompt,
            max_new_tokens=512,
            add_special_tokens=True,
            do_sample=True,
            temperature=0.7,
            top_k=10,
            top_p=0.95
        )
        
        generated_text = outputs[0]["generated_text"][len(prompt):]
        
        # Calculate metrics
        metrics = calculate_metrics(generated_text, reference)
        all_metrics.append(metrics)
        
        # Print individual results
        print(f"\nQuestion: {question}")
        print(f"Generated: {generated_text[:200]}...")
        print(f"Reference: {reference[:200]}...")
        print("Metrics:", {k: f"{v:.4f}" for k, v in metrics.items()})
    
    # Aggregate metrics
    aggregated_metrics = {}
    for metric in all_metrics[0].keys():
        values = [m[metric] for m in all_metrics]
        aggregated_metrics[f'avg_{metric}'] = np.mean(values)
        aggregated_metrics[f'std_{metric}'] = np.std(values)
    
    return aggregated_metrics

# Example usage:
eval_data = [
    (
        "What is the difference between a variable and an object?",
        "A variable is a named storage location that holds a value, while an object is an instance of a class that contains both data and methods."
    ),
    (
        "Explain what is inheritance in programming?",
        "Inheritance is a fundamental concept in object-oriented programming where a class can inherit properties and methods from another class. This promotes code reuse and establishes a relationship between parent and child classes."
    )
]

# Run evaluation
print("Running evaluation...")
metrics = evaluate_model(text_generator, eval_data)

print("\nAggregated Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")




