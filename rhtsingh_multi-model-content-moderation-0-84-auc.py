import pandas as pd
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class PromptDataset(Dataset):
    def __init__(self, prompts, labels, tokenizer, max_length=512):
        self.prompts = prompts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_prompt(row, prompt_style="default"):
    """Create instruction prompt for each row with different styles"""
    pos_ex1 = str(row.get('positive_example_1', 'No example provided'))
    pos_ex2 = str(row.get('positive_example_2', 'No example provided'))
    neg_ex1 = str(row.get('negative_example_1', 'No example provided'))
    neg_ex2 = str(row.get('negative_example_2', 'No example provided'))
    
    if pd.isna(row.get('positive_example_1')):
        pos_ex1 = "No example provided"
    if pd.isna(row.get('positive_example_2')):
        pos_ex2 = "No example provided"
    if pd.isna(row.get('negative_example_1')):
        neg_ex1 = "No example provided"
    if pd.isna(row.get('negative_example_2')):
        neg_ex2 = "No example provided"
    
    if prompt_style == "default":
        prompt = f"""You are a Reddit moderator reviewing comments for rule violations.

Rule: {row['rule']}

Examples of violations:
- {pos_ex1}
- {pos_ex2}

Examples of non-violations:
- {neg_ex1}
- {neg_ex2}

Comment to classify: "{row['body']}"

Does this comment violate the rule? Answer Yes or No.
Classification:"""
    
    elif prompt_style == "concise":
        prompt = f"""Rule: {row['rule']}
Violations: {pos_ex1}; {pos_ex2}
Non-violations: {neg_ex1}; {neg_ex2}
Comment: "{row['body']}"
Violates rule? (Yes/No):"""
    
    elif prompt_style == "detailed":
        prompt = f"""As a Reddit community moderator, carefully review this comment for rule violations.

COMMUNITY RULE TO ENFORCE:
{row['rule']}

EXAMPLES OF CONTENT THAT VIOLATES THIS RULE:
1. {pos_ex1}
2. {pos_ex2}

EXAMPLES OF ACCEPTABLE CONTENT:
1. {neg_ex1}
2. {neg_ex2}

COMMENT TO REVIEW:
"{row['body']}"

DECISION: Does this comment violate the rule stated above? Respond with Yes or No.
Answer:"""
    
    elif prompt_style == "examples_first":
        prompt = f"""Examples of rule violations:
- {pos_ex1}
- {pos_ex2}

Examples of acceptable comments:
- {neg_ex1}
- {neg_ex2}

Rule: {row['rule']}

Review this comment: "{row['body']}"

Does it violate the rule? Yes or No:"""
    
    return prompt

class WeightedTrainer(Trainer):
    """Custom Trainer that handles class weights for imbalanced datasets"""
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        if self.class_weights is not None:
            # Move class weights to the same device as labels
            device = labels.device
            weights = torch.tensor(self.class_weights, dtype=torch.float32).to(device)
            
            # Create weight tensor for each sample based on its label
            sample_weights = weights[labels]
            
            # Compute weighted cross entropy loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            loss = (loss * sample_weights).mean()
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

class SimplePromptClassifier:
    def __init__(self, model_name="/kaggle/input/modernbert/pytorch/base/2"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2,
            torch_dtype=torch.float32  # Force float32 for stability
        )
        
        # Disable compilation to avoid device issues
        self.model.config.reference_compile = False
        
        # Move model to GPU
        self.model = self.model.to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = "[PAD]"
    
    def calculate_class_weights(self, labels, weight_power=1.5):
        """Calculate class weights for imbalanced datasets
        
        Args:
            labels: Training labels
            weight_power: Power to raise the weights to (>1 increases emphasis on minority class)
        """
        classes = np.unique(labels)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=labels
        )
        
        # Adjust weights to emphasize class 0 more (since it has lower accuracy)
        # Raise weights to a power to increase the difference
        class_weights = np.power(class_weights, weight_power)
        
        # Normalize so the mean weight is 1.0
        class_weights = class_weights / class_weights.mean()
        
        print(f"Class weights - Class 0: {class_weights[0]:.3f}, Class 1: {class_weights[1]:.3f}")
        return class_weights
    
    def prepare_data(self, df, prompt_style="default"):
        """Convert dataframe to prompts and labels"""
        # Create a copy to avoid SettingWithCopyWarning
        df_copy = df.copy()
        df_copy['prompt'] = df_copy.apply(lambda row: create_prompt(row, prompt_style), axis=1)
        
        # Check if this is training data (has labels) or test data (no labels)
        if 'rule_violation' in df_copy.columns:
            return df_copy['prompt'].tolist(), df_copy['rule_violation'].tolist()
        else:
            # For test data, return None for labels
            return df_copy['prompt'].tolist(), None
    
    def train(self, train_df, val_df=None, output_dir="./results", 
              use_class_weights=True, weight_power=1.5, 
              learning_rate=5e-5, num_epochs=5, warmup_ratio=0.1):
        """Fine-tune the model with optional class weighting and improved parameters
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            output_dir: Output directory for model
            use_class_weights: Whether to use class weights
            weight_power: Power for class weight adjustment (higher = more emphasis on minority)
            learning_rate: Learning rate (try 5e-5 for better class 0)
            num_epochs: Number of training epochs
            warmup_ratio: Proportion of steps for warmup
        """
        print("Preparing training data...")
        train_prompts, train_labels = self.prepare_data(train_df)
        
        # Calculate class weights if requested
        class_weights = None
        if use_class_weights:
            print("Calculating class weights for imbalanced data...")
            class_weights = self.calculate_class_weights(train_labels, weight_power)
        
        train_dataset = PromptDataset(
            train_prompts, train_labels, self.tokenizer, max_length=512
        )
        
        val_dataset = None
        if val_df is not None:
            val_prompts, val_labels = self.prepare_data(val_df)
            val_dataset = PromptDataset(
                val_prompts, val_labels, self.tokenizer, max_length=512
            )
        
        # Calculate warmup steps based on ratio
        total_steps = len(train_dataset) * num_epochs // 4  # divided by batch_size * grad_accum
        warmup_steps = int(total_steps * warmup_ratio)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=warmup_steps,
            weight_decay=0.02,  # Slightly increased regularization
            learning_rate=learning_rate,
            logging_dir=f'{output_dir}/logs',
            logging_steps=50,
            eval_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False,
            seed=42,
            remove_unused_columns=False,
            metric_for_best_model="loss",  # Use loss for best model selection
            greater_is_better=False,
            fp16=True if torch.cuda.is_available() else False,  # Mixed precision training
            report_to="none",  # CRITICAL: Disable wandb and all reporting
            disable_tqdm=False,  # Keep progress bars
            push_to_hub=False,  # Don't try to push to HuggingFace
        )
        
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Use custom trainer with class weights
        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        print(f"Starting training with:")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  Weight power: {weight_power}")
        
        trainer.train()
        
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print("Training completed!")
    
    def predict_batch(self, prompts, batch_size=16):
        """Batch prediction for multiple prompts - much faster than single predictions"""
        all_predictions = []
        
        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            # Tokenize the batch
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Convert to probabilities
            probabilities = torch.softmax(logits, dim=1)
            violation_probs = probabilities[:, 1].cpu().numpy()
            
            all_predictions.extend(violation_probs)
        
        return np.array(all_predictions)
    
    def predict_single(self, prompt):
        """Get prediction probability for a single prompt (kept for compatibility)"""
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]
        
        probabilities = torch.softmax(logits, dim=0)
        violation_prob = probabilities[1].item()
        
        return violation_prob
    
    def predict(self, df, batch_size=16, use_tta=False):
        """Get predictions for a dataframe using batch processing
        
        Args:
            df: DataFrame with data
            batch_size: Batch size for prediction
            use_tta: If True, use test-time augmentation with multiple prompts
        """
        if use_tta:
            print("Using Test-Time Augmentation with multiple prompt styles...")
            prompt_styles = ["default", "concise", "detailed", "examples_first"]
            all_style_predictions = []
            
            for style in prompt_styles:
                print(f"  Predicting with {style} prompt style...")
                prompts, _ = self.prepare_data(df, prompt_style=style)
                predictions = self.predict_batch(prompts, batch_size=batch_size)
                all_style_predictions.append(predictions)
            
            # Average predictions across all prompt styles
            final_predictions = np.mean(all_style_predictions, axis=0)
            
            # You can also use weighted average or voting
            # Example of weighted average (give more weight to default style):
            # weights = [0.4, 0.2, 0.2, 0.2]  # default gets 40%, others get 20%
            # final_predictions = np.average(all_style_predictions, axis=0, weights=weights)
            
            print(f"TTA completed - averaged {len(prompt_styles)} prompt variations")
            return final_predictions
        else:
            prompts, _ = self.prepare_data(df)  # Labels will be None for test data
            print(f"Making predictions for {len(prompts)} samples (batch size: {batch_size})...")
            predictions = self.predict_batch(prompts, batch_size=batch_size)
            print(f"Predictions completed!")
            return predictions
    
    def evaluate(self, df, batch_size=16, use_tta=False):
        """Evaluate on a dataframe using batch predictions"""
        predictions = self.predict(df, batch_size=batch_size, use_tta=use_tta)
        
        # Only calculate metrics if we have true labels
        if 'rule_violation' not in df.columns:
            print("No true labels available for evaluation")
            return None, predictions
            
        true_labels = df['rule_violation'].values
        
        auc = roc_auc_score(true_labels, predictions)
        
        print(f"AUC Score: {auc:.4f}")
        print(f"Prediction stats: mean={predictions.mean():.3f}, std={predictions.std():.3f}")
        
        # Calculate per-class accuracy for insight
        threshold = 0.5
        binary_preds = (predictions > threshold).astype(int)
        
        for class_val in [0, 1]:
            class_mask = true_labels == class_val
            if class_mask.sum() > 0:
                class_acc = (binary_preds[class_mask] == true_labels[class_mask]).mean()
                print(f"Class {class_val} accuracy: {class_acc:.3f} (n={class_mask.sum()})")
        
        return auc, predictions

def main():
    # Load data
    train_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
    print(f"Loaded {len(train_df)} training samples")
    
    # Check rule distribution
    unique_rules = train_df['rule'].unique()
    print(f"Number of unique rules: {len(unique_rules)}")
    print("Rule distribution:")
    print(train_df['rule'].value_counts())
    print("\nTarget distribution:")
    target_counts = train_df['rule_violation'].value_counts()
    print(target_counts)
    
    # Check class imbalance
    imbalance_ratio = target_counts[0] / target_counts[1] if 1 in target_counts.index else float('inf')
    print(f"Class imbalance ratio (0/1): {imbalance_ratio:.2f}")
    
    # Use standard stratified split instead of rule-wise split
    train_split, val_split = train_test_split(
        train_df, test_size=0.2, random_state=42, 
        stratify=train_df['rule_violation']
    )
    
    print(f"\nTrain samples: {len(train_split)}")
    print(f"Validation samples: {len(val_split)}")
    print(f"Train target distribution: {train_split['rule_violation'].value_counts().to_dict()}")
    print(f"Val target distribution: {val_split['rule_violation'].value_counts().to_dict()}")
    
    # Initialize classifier
    classifier = SimplePromptClassifier()
    
    # Train with adjusted parameters for better class 0 accuracy
    use_weights = imbalance_ratio > 1.5 or imbalance_ratio < 0.67
    if use_weights:
        print(f"\nClass imbalance detected (ratio: {imbalance_ratio:.2f}). Using class weights.")
    
    # Balanced training parameters for better overall accuracy
    classifier.train(
        train_split, 
        val_split, 
        output_dir="./prompt_model",
        use_class_weights=use_weights,
        weight_power=1.2,  # Mild emphasis on minority class (was 2.0, too aggressive)
        learning_rate=3e-5,  # Moderate learning rate (was 5e-5)
        num_epochs=4,  # Balanced number of epochs
        warmup_ratio=0.15  # Slightly more warmup for stability
    )
    
    # Evaluate with batch prediction
    print("\nEvaluating on validation set...")
    auc, predictions = classifier.evaluate(val_split, batch_size=32, use_tta=False)
    
    # Also evaluate with TTA to see if it helps
    print("\nEvaluating on validation set with Test-Time Augmentation...")
    auc_tta, predictions_tta = classifier.evaluate(val_split, batch_size=32, use_tta=True)
    
    # Show some example predictions
    print("\nSample predictions (with TTA):")
    for i in range(min(3, len(val_split))):
        row = val_split.iloc[i]
        pred = predictions_tta[i]
        actual = row['rule_violation']
        print(f"Comment: {row['body'][:100]}...")
        print(f"Predicted: {pred:.3f}, Actual: {actual}")
        print("---")
    
    # Generate test predictions and submission
    import os
    if os.path.exists("/kaggle/input/jigsaw-agile-community-rules/test.csv"):
        print("\nGenerating test predictions...")
        test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
        print(f"Test data shape: {test_df.shape}")
        
        # Use TTA for final test predictions if it improved validation score
        use_tta_for_test = auc_tta > auc
        if use_tta_for_test:
            print("Using TTA for test predictions (improved validation AUC)")
        else:
            print("Not using TTA for test predictions (didn't improve validation AUC)")
        
        # Use batch prediction for faster inference
        test_predictions = classifier.predict(test_df, batch_size=32, use_tta=use_tta_for_test)
        
        submission_df = pd.DataFrame({
            'row_id': test_df['row_id'],
            'rule_violation': test_predictions
        })
        
        submission_df.to_csv("submission.csv", index=False)
        print("Submission saved to submission.csv")
        
        print(f"\nSubmission statistics:")
        print(f"  Mean: {test_predictions.mean():.4f}")
        print(f"  Std: {test_predictions.std():.4f}")
        print(f"  Min: {test_predictions.min():.4f}")
        print(f"  Max: {test_predictions.max():.4f}")
    else:
        print("No test.csv found - skipping test predictions")

if __name__ == "__main__":
    main()







