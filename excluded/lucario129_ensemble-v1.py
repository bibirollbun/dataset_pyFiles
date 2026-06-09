import os
import gc
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Queue
import threading

import torch
import gc

def clear_gpu_memory():
    """
    Clears GPU memory by releasing cached memory and 
    running garbage collection to free unused objects.
    """
    try:
        print("\nClearing GPU memory...")
        torch.cuda.empty_cache()   # Free unused cached memory allocated by PyTorch
        gc.collect()               # Run Python garbage collection
        print("✓ GPU memory cleared successfully!")
    except Exception as e:
        print(f"✗ Error clearing GPU memory: {e}")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================================
# Data Processing Class
# ============================================================================

class TestDataProcessor:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.le = LabelEncoder()
        self.n_classes = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess test data"""
        # Load train to fit label encoder
        train = pd.read_csv(self.train_path)
        train.Misconception = train.Misconception.fillna('NA')
        train['target'] = train.Category + ':' + train.Misconception
        train['label'] = self.le.fit_transform(train['target'])
        self.n_classes = len(self.le.classes_)
        
        # Get correct answers from train
        correct = self._get_correct_answers(train)
        
        # Load and process test
        test = pd.read_csv(self.test_path)
        test = test.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')
        test.is_correct = test.is_correct.fillna(0)
        
        return test
    
    def _get_correct_answers(self, train):
        """Extract correct answers from training data"""
        idx = train.apply(lambda row: row.Category.split('_')[0], axis=1) == 'True'
        correct = train.loc[idx].copy()
        correct['c'] = correct.groupby(['QuestionId', 'MC_Answer']).MC_Answer.transform('count')
        correct = correct.sort_values('c', ascending=False)
        correct = correct.drop_duplicates(['QuestionId'])
        correct = correct[['QuestionId', 'MC_Answer']]
        correct['is_correct'] = 1
        return correct
    
    def format_input(self, row):
        """Format input text for model"""
        x = "This answer is correct." if row['is_correct'] else "This answer is incorrect."
        return (
            f"Question: {row['QuestionText']}\n"
            f"Answer: {row['MC_Answer']}\n"
            f"{x}\n"
            f"Student Explanation: {row['StudentExplanation']}"
        )
    
    def prepare_dataset(self, df, tokenizer, max_length=256):
        """Prepare dataset for inference"""
        df = df.copy()
        df['text'] = df.apply(self.format_input, axis=1)
        df['label'] = 0  # Dummy label for inference
        
        # Keep only required columns
        df = df[['row_id', 'label', 'text']]
        
        # Convert to HuggingFace dataset
        ds = Dataset.from_pandas(df)
        
        # Tokenization function
        def tokenize(batch):
            return tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length
            )
        
        # Apply tokenization
        ds = ds.map(tokenize, batched=True)
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        
        return ds


from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
import torch
import numpy as np
import gc
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from typing import List
from scipy.special import softmax

def run_inference_single_gpu(model_path, test_data, n_classes, batch_size, device_str, model_name):
    """Run inference on a single GPU and return embeddings"""
    try:
        device_id = int(device_str.split(":")[1])  # Extract GPU id
        torch.cuda.set_device(device_id)
        device = torch.device(device_str)

        print(f"\n[{model_name}] Loading model on {device}...")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=n_classes,
            torch_dtype=torch.float16
        ).to(device)

        model.config.pad_token_id = tokenizer.pad_token_id
        model.eval()
        print(f"[{model_name}] ✓ Model loaded on {device}")

        # Prepare dataset
        processor_temp = TestDataProcessor('', '')
        processor_temp.le = LabelEncoder()
        processor_temp.le.classes_ = np.load('/tmp/le_classes.npy', allow_pickle=True)
        processor_temp.n_classes = n_classes
        ds = processor_temp.prepare_dataset(test_data, tokenizer)

        dataloader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0
        )

        # Run inference
        all_embeddings = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"[{model_name}] Inference"):
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                final_hidden_state = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]

                # Mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(final_hidden_state.size())
                sum_hidden = torch.sum(final_hidden_state * mask_expanded, dim=1)
                sentence_embeddings = sum_hidden / attention_mask.sum(dim=1, keepdim=True)

                all_embeddings.append(sentence_embeddings.float().cpu().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)

        # Cleanup
        del model, tokenizer, ds, dataloader
        torch.cuda.empty_cache()
        gc.collect()

        print(f"[{model_name}] ✓ Inference complete, memory cleared")
        return embeddings

    except Exception as e:
        print(f"[{model_name}] ✗ Error: {str(e)}")
        return None

def run_model_inference(model, processor, test_data, batch_size=4):
    """Run inference for a single model and return embeddings"""
    np.save('/tmp/le_classes.npy', processor.le.classes_)
    print(f"Running inference for model: {model['name']}")

    embd = run_inference_single_gpu(
        model_path=model["path"],
        test_data=test_data,
        n_classes=processor.n_classes,
        batch_size=batch_size,
        device_str=model["device"],
        model_name=model["name"]
    )

    if embd is not None:
        print(f"[{model['name']}] ✓ Inference successful")
        return embd
    else:
        print(f"[{model['name']}] ✗ Failed to get results")
        return None

import torch.nn as nn

class LinearClassifier(nn.Module):
    """Simple linear classifier"""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)



batch_size = 4

processor = TestDataProcessor(
    train_path='/kaggle/input/map-charting-student-math-misunderstandings/train.csv',
    test_path='/kaggle/input/map-charting-student-math-misunderstandings/test.csv'
)

test_data = processor.load_and_preprocess_data()
print(f"Test samples: {len(test_data)}, Classes: {processor.n_classes}")

model_deepseek = {
    'path': '/kaggle/input/deekseepmath-7b-map-competition/MAP_EXP_09_FULL',
    'name': 'deepseek',
    'device': 'cuda:0'
}

model_qwen3 = {
    'path': '/kaggle/input/qwen3-8b-map-competition/MAP_EXP_16_FULL',
    'name': 'qwen3',
    'device': 'cuda:0'
}

print(f"\n{'='*60}")
print(f"Running inference for individual models")
print(f"{'='*60}")

# Run inference for embeddings
deepseekmath_embd = run_model_inference(model_deepseek, processor, test_data, batch_size)
clear_gpu_memory()

qwen_embd = run_model_inference(model_qwen3, processor, test_data, batch_size)
clear_gpu_memory()

# Concatenate embeddings
concat_embd = np.concatenate([deepseekmath_embd, qwen_embd], axis=1)





import torch.serialization

# allow your LinearClassifier class to be unpickled
torch.serialization.add_safe_globals([LinearClassifier])

# load full model directly
classifier_path = "/kaggle/input/custom_llm/pytorch/classifier-map/1/model_params.pth"
classifier = torch.load(classifier_path, map_location="cuda", weights_only=False)
classifier.eval()





# Predict
with torch.no_grad():
    logits = classifier(torch.tensor(concat_embd, dtype=torch.float32).to('cuda'))
    probs = torch.softmax(logits, dim=1).cpu().numpy()

# Get top-3 predictions
top_indices = np.argsort(-probs, axis=1)[:, :3]

predictions = []
for indices in top_indices:
    pred_labels = processor.le.inverse_transform(indices)
    predictions.append(' '.join(pred_labels))

# Submission
submission = pd.DataFrame({
    'row_id': test_data.row_id.values,
    'Category:Misconception': predictions
})

submission.to_csv('submission.csv', index=False)
print(f"\n✓ Submission saved to submission.csv")
print(submission.head(10))


submission


submission['Category:Misconception'][0]




