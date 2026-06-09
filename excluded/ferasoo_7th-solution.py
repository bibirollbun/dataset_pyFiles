%pip install -U bitsandbytes -U transformers -U peft -U accelerate


import os
import argparse
import logging
import pandas as pd
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from datasets import Dataset
from tqdm import tqdm
from peft import PeftModel, PeftConfig
import torch.nn.functional as F

# Hardcoded model paths and weights
MODEL_PATHS = [
    "/kaggle/input/kachallenges-series-1-model-weights/model1", # 0.9068
    "/kaggle/input/kachallenges-series-1-model-weights/model2", #0.9156 
    "/kaggle/input/kachallenges-series-1-model-weights/model3", # 0.9068
    "/kaggle/input/kachallenges-series-1-model-weights/model4", # 0.9156 
    "/kaggle/input/kachallenges-series-1-model-weights/model5", # 0.9127
]

# Weights for each model (will be normalized to sum to 1)
MODEL_WEIGHTS = [
    1,
    1, 
    1,
    1,
    1,
]

# Default settings
TEST_PATH = "/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv"
OUTPUT_FOLDER = "ensemble_predictions"
BATCH_SIZE = 8

TOPICS = {
    0: "Algebra",
    1: "Geometry and Trigonometry",
    2: "Calculus and Analysis",
    3: "Probability and Statistics",
    4: "Number Theory",
    5: "Combinatorics and Discrete Math",
    6: "Linear Algebra",
    7: "Abstract Algebra and Topology"
}

def preprocess_function(examples, tokenizer, max_length=512):
    """Tokenize the examples for prediction."""
    return tokenizer(
        examples["Question"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

def generate_predictions_with_probs(model, dataset, tokenizer, batch_size):
    """Generate predictions and probabilities for the test set."""
    model.eval()
    all_predictions = []
    all_probabilities = []
    
    # Create DataLoader-like batches
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i + batch_size]
        inputs = preprocess_function(batch, tokenizer)
        
        # Move inputs to the same device as model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return all_predictions, all_probabilities

def load_model(model_path):
    """Load a model and tokenizer from the given path."""

    # First load the PEFT config to get the base model name
    peft_config = PeftConfig.from_pretrained(model_path)


    # Define quantization config for 4-bit
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",  # You can try "fp4" as well
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    # Load the base model with quantization
    base_model = AutoModelForSequenceClassification.from_pretrained(
        peft_config.base_model_name_or_path,
        quantization_config=quant_config,
        device_map="auto",  # helps with loading on multi-GPU setups
        num_labels=8,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = "[PAD]"
            tokenizer.add_special_tokens({'pad_token': "[PAD]"})
    base_model.config.pad_token_id = tokenizer.pad_token_id

    # Load the PEFT model with adapters
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.to("cuda")
    return model, tokenizer

def main():    
    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    try:
        # Load test data
        print(f"Loading test data from: {TEST_PATH}")
        test_df = pd.read_csv(TEST_PATH)
        
        # Convert to Hugging Face Dataset
        test_dataset = Dataset.from_pandas(test_df)
        
        # Set up model weights
        # Check if number of weights matches number of models
        if len(MODEL_WEIGHTS) != len(MODEL_PATHS):
            raise ValueError(f"Number of weights ({len(MODEL_WEIGHTS)}) must match number of models ({len(MODEL_PATHS)})")
        
        # Normalize weights to sum to 1
        weights = np.array(MODEL_WEIGHTS)
        weights = weights / weights.sum()
            
        print(f"Using model weights: {weights}")
        
        # Initialize array for ensemble probabilities
        ensemble_probs = np.zeros((len(test_dataset), len(TOPICS)))
        
        # Process each model
        for idx, (model_path, weight) in enumerate(zip(MODEL_PATHS, weights)):
            print(f"Processing model {idx+1}/{len(MODEL_PATHS)}: {model_path} (weight: {weight:.4f})")
            
            # Load model and tokenizer
            model, tokenizer = load_model(model_path)
            
            # Generate predictions and probabilities
            predictions, probabilities = generate_predictions_with_probs(model, test_dataset, tokenizer, BATCH_SIZE)
            
            # Add weighted probabilities to ensemble
            ensemble_probs += weight * np.array(probabilities)
            
            # Create individual model prediction DataFrame
            model_df = pd.DataFrame({
                'id': test_df['id'],
                'label': predictions
            })
            
            # Add probability columns for each topic
            for topic_id, topic_name in TOPICS.items():
                model_df[f'prob_{topic_id}_{topic_name.replace(" ", "_")}'] = [probs[topic_id] for probs in probabilities]
            
            # Save individual model predictions
            model_output_path = os.path.join(OUTPUT_FOLDER, f"model_{idx+1}_predictions.csv")
            model_df.to_csv(model_output_path, index=False)
            print(f"Saved individual model predictions to: {model_output_path}")
            
            # Clean up model and tokenizer to free GPU memory
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Cleaned up model {idx+1} from GPU memory.")
        
        # Get ensemble predictions
        ensemble_predictions = np.argmax(ensemble_probs, axis=1)
        
        # Create ensemble prediction DataFrame
        ensemble_df = pd.DataFrame({
            'id': test_df['id'],
            'label': ensemble_predictions
        })
        
        # Add probability columns for ensemble
        for topic_id, topic_name in TOPICS.items():
            ensemble_df[f'prob_{topic_id}_{topic_name.replace(" ", "_")}'] = ensemble_probs[:, topic_id]
        
        # Save ensemble predictions
        ensemble_output_path = os.path.join(OUTPUT_FOLDER, "ensemble_predictions.csv")
        ensemble_df.to_csv(ensemble_output_path, index=False)
        print(f"Saved ensemble predictions to: {ensemble_output_path}")
        
        # Save a simple version for submission
        submission_path = os.path.join("submission.csv")
        ensemble_df[['id', 'label']].to_csv(submission_path, index=False)
        print(f"Saved submission file to: {submission_path}")
        
        # Log prediction distribution
        print(f"Ensemble prediction distribution:")
        value_counts = ensemble_df['label'].value_counts().sort_index()
        for topic_id, count in value_counts.items():
            print(f"Topic {topic_id} ({TOPICS[topic_id]}): {count} predictions")
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise
    
    print("Ensemble predictions completed successfully!")


main()




