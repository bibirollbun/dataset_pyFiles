#!pip freeze | grep -v '^-e' > to_remove.txt

# 2) Uninstall them all:
#!pip uninstall -y -r to_remove.txt
#!rm -r python-packages


# 1. Upgrade pip so we get the latest installer
#!pip install --upgrade pip
#%pip install torch --index-url https://download.pytorch.org/whl/cu124

# 2. Install into our own directory, ignoring whatever’s already installed
#!mkdir -p python-packages
#!pip install --upgrade --ignore-installed --target python-packages \
#    setfit==1.1.3 \
#    sentence-transformers==5.1.0 \
#    transformers==4.55.0 \
#    datasets \
#    scikit-learn \
#    pandas \
#    numpy==1.26.4 \
#    scipy==1.12.0 \
#    tqdm \
#    torch==2.8.0 \
#    torchvision==0.23

#!zip -r python_packages.zip python-packages/



from IPython.display import FileLink
FileLink('python_packages.zip')


import logging
import pandas as pd
import numpy as np
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, Trainer, TrainingArguments

os.environ["WANDB_DISABLED"] = "true"


from tqdm.notebook import tqdm
import tqdm.notebook
tqdm.tqdm = tqdm.notebook.tqdm

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("setfit").setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)


def setup_distributed(rank, nproc):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group("nccl", rank=rank, world_size=nproc)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    dist.destroy_process_group()

def get_gpu_info():
    if not torch.cuda.is_available():
        return {"count" : 0, "devices": []}

    gpu_count = torch.cuda.device_count()
    devices = []

    print(f"Found {gpu_count} GPUs:")
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        devices.append({
            "id": i,
            "name": props.name,
            "memory_gb": memory_gb,
            "compute_capability": f"{props.major}.{props.minor}"
        })
        print(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")
    
    return {"count": gpu_count, "devices": devices}

gpu_info = get_gpu_info()
    
if gpu_info["count"] == 0:
    print("No GPUs found, using CPU")



def load_data(train_csv="/kaggle/input/jigsaw-agile-community-rules/train.csv", 
             test_csv="/kaggle/input/jigsaw-agile-community-rules/test.csv"):
   print("Loading the data...")
   
   # Load training data
   train_df = pd.read_csv(train_csv)
   print(f"Training dataset shape: {train_df.shape}")
   print(f"Training label distribution:\n{train_df['rule_violation'].value_counts()}")
   
   # Load test data
   try:
       test_df = pd.read_csv(test_csv)
       print(f"Test dataset shape: {test_df.shape}")
       if 'rule_violation' in test_df.columns:
           print(f"Test label distribution:\n{test_df['rule_violation'].value_counts()}")
       else:
           print("Test set has no labels (submission format)")
   except FileNotFoundError:
       print(f"Test file '{test_csv}' not found.")
       test_df = None
   
   # Prepare training texts
   train_texts = []
   for i, row in train_df.iterrows():
       combined_text = f"Rule: {row['rule']}\n\nPost: {row['body']}"
       train_texts.append(combined_text)
   train_labels = train_df['rule_violation'].tolist()
   
   # Prepare test texts
   test_texts = []
   test_labels = []
   if test_df is not None:
       for i, row in test_df.iterrows():
           combined_text = f"Rule: {row['rule']}\n\nPost: {row['body']}"
           test_texts.append(combined_text)
       
       if 'rule_violation' in test_df.columns:
           test_labels = test_df['rule_violation'].tolist()
       else:
           test_labels = None
   
   return train_texts, train_labels, test_texts, test_labels, train_df, test_df

train_texts, train_labels, test_texts, test_labels, train_df, test_df = load_data()
#train_texts, train_labels, test_texts, test_labels, train_df, test_df


def check_internet():
    try:
        import urllib.request
        urllib.request.urlopen('http://google.com', timeout=3)
        return True
    except:
        return False
        

def create_setfit_model(model_name="sentence-transformers/all-distilroberta-v1", use_distributed=False, rank=0):
    device = f"cuda:{rank}" if use_distributed else "cuda"
    model_dir = "/kaggle/input/all-distilroberta-v1/transformers/v1/1/all-distilroberta-v1"
    
    is_online = check_internet()
    
    if not is_online or os.path.exists(model_dir):
        print(f"Loading SetFit model from local path: {model_dir}")
        model = SetFitModel.from_pretrained(
            model_dir,
            use_differentiable_head=True,
            head_params={"out_features": 2},
            device=device,
            local_files_only=True
        )
    else:
        print(f"Creating SetFit model with body: {model_name}")
        os.makedirs(model_dir, exist_ok=True)
        model = SetFitModel.from_pretrained(
            model_name,
            use_differentiable_head=True,
            head_params={"out_features": 2},
            device=device
        )
        model.save_pretrained(model_dir)
    
    if torch.cuda.is_available():
        model = model.to('cuda')
        print(f"Model moved to GPU: {torch.cuda.get_device_name()}")
    
    return model

def setup_model(model, gpu_info, use_distributed=False, rank=0, nproc=1):

    if gpu_info["count"] == 0:
        print("No GPUs found, using CPU...")
        return model

    if use_distributed and nproc > 1:
        print(f"Using DDP on {nproc} GPUs")

        if hasattr(model, 'model_body'):
            model.model_body = DDP(model.model_body, device_ids=[rank])
        if hasattr(model, 'model_head'):
            model.model_head = DDP(model.model_head, device_ids=[rank])

    elif gpu_info["count"] > 1:
        print(f"Using DataParallel on {gpu_info['count']} GPUs")
        if hasattr(model, 'model_body'):
            model.model_body = DataParallel(model.model_body)#, device_ids=[rank])
        if hasattr(model, 'model_head'):
            model.model_head = DataParallel(model.model_head)#, device_ids=[rank])
    else:
        print(f"Using single GPU: {gpu_info['devices'][0]['name']}")

    return model
model = create_setfit_model()


class MultiGPUTrainer(Trainer):
    def __init__(self, *args, use_distributed=False, rank=0, nproc=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_distributed = use_distributed
        self.rank = rank
        self.nproc = nproc

        if nproc > 1:
            self.args.batch_size = self.args.batch_size * nproc
            print(f"Adjusted batch size to {self.args.batch_size} for {nproc} GPUs")

    def create_train_dataloader(self):
        data_loader = super().create_train_dataloader()
    
        if self.use_distributed and self.nproc > 1:
            dataset = data_loader.dataset
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.nproc,
                rank=self.rank,
                shuffle=True
            )
            from torch.utils.data import DataLoader
            data_loader = DataLoader(
                dataset,
                batch_size=data_loader.batch_size,
                sampler=sampler,
                num_workers=data_loader.num_workers,
                collate_fn=data_loader.collate_fn
            )
    
        return data_loader

def train_setfit_model(model, train_texts, train_labels, gpu_info,
          eval_texts=None, eval_labels=None,
          use_distributed=False, rank=0, nproc=1):
    if rank == 0:
        print("Setting up multi-GPU trainer...")

    if gpu_info["count"] > 0:
        avg_memory = sum(device["memory_gb"] for device in gpu_info["devices"]) / len(gpu_info["devices"])
        batch_size_base = min(16, max(32, int(avg_memory // 2)))
        batch_size = batch_size_base * gpu_info["count"] if not use_distributed else batch_size_base
    else:
        batch_size = 16
    train_dataset = Dataset.from_dict({
        "text": train_texts,
        "label": train_labels
    })

    eval_dataset = None
    if eval_texts is not None and eval_labels is not None:
        eval_dataset = Dataset.from_dict({
            "text": eval_texts,
            "label": eval_labels
        })
    args = TrainingArguments(
        output_dir="./checkpoints",
        batch_size=(64, 32),                    
        num_epochs=(3, 15),                     
        body_learning_rate=(8e-6, 4e-6),       
        head_learning_rate=0.003,
        loss=CosineSimilarityLoss,
        sampling_strategy="oversampling",
        num_iterations=100,                    
        use_amp=True,
        warmup_proportion=0.2,                  
        end_to_end=True,                        
        l2_weight=0.003,                        
        eval_strategy="steps" if eval_dataset is not None else "no",  # Only eval if we have eval data
        eval_steps=25 if eval_dataset is not None else None,
        eval_delay=50 if eval_dataset is not None else 0,
        save_strategy="steps" if eval_dataset is not None else "no",  # Match eval strategy
        save_steps=25 if eval_dataset is not None else None,
        save_total_limit=10,
        load_best_model_at_end=True if eval_dataset is not None else False,
        metric_for_best_model="eval_accuracy" if eval_dataset is not None else None,
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=10,
        show_progress_bar=True,
        seed=42,
    )

    trainer = MultiGPUTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        metric='accuracy',
        column_mapping={"text": "text", "label": "label"},
        use_distributed=use_distributed,
        rank=rank,
        nproc=nproc
    )

    
    if rank == 0:
        print(f"Training model with batch size: {batch_size}")
        print(f"Using {'distributed' if use_distributed else 'data parallel'} training")
    
    logging.getLogger("setfit").setLevel(logging.INFO)

    print("Starting training with progress monitoring...")
    trainer.train()
    return trainer

def train(rank, nproc, train_texts, train_labels, gpu_info, val_texts=None, val_labels=None):
    setup_distributed(rank, nproc)

    try:
        model = create_setfit_model(use_distributed=True, rank=rank)
        model = setup_model(model, gpu_info, use_distributed=True, rank=rank, nproc=nproc)

        trainer = train_setfit_model(
            model, train_texts, train_labels, gpu_info, 
            val_texts, val_labels,
            use_distributed=True, rank=rank, nproc=nproc
        )
        if rank == 0:
            print("Saving Model to ./reddit-rule-violation-setfit")
            trainer.model.save_pretrained("./reddit-rule-violation-setfit")
    finally:
        cleanup_distributed()
        


def evaluate_model(model, test_texts, test_labels, rank=0):
    if rank != 0:
        return {}

    print("Starting Evaluation...")
    if hasattr(model.model_body, 'module'):
        model.model_body = model.model_body.module
    if hasattr(model.model_head, 'module'):
        model.model_head = model.model_head.module

    predictions = model(test_texts)
    if hasattr(predictions, 'cpu'):
        predictions = predictions.cpu().numpy()
    
    accuracy = accuracy_score(test_labels, predictions)

    report = classification_report(test_labels, predictions, 
                                   target_names=['No Violation', 'Violation']
                                   , output_dict = True)
    print(f"Accuracy: {accuracy:.4f}\n")
    print(f"Classification Report:\n{classification_report(test_labels, predictions, target_names=['No Violation', 'Violation'])}")
    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_labels, predictions)
    print(cm)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm
    }

def generate_test_predictions(model, test_texts, test_df, output_path = "./submission.csv"):
        print("Generating test set predictions...")
        if hasattr(model.model_body, 'module'):
            model.model_body = model.model_body.module
        if hasattr(model.model_head, 'module'):
            model.model_head = model.model_head.module
        
        #predictions = model(test_texts)
        probabilities = model.predict_proba(test_texts)

        #if hasattr(predictions, 'cpu'):
        #    predictions = predictions.cpu().numpy()
        if hasattr(probabilities, 'cpu'):
            probabilities = probabilities.cpu().numpy()


        predictions_df = test_df[['row_id']].copy()
        #predictions_df['predicted_rule_violation'] = predictions
        predictions_df['rule_violation'] = probabilities[: ,1]
        #predictions_df['no_violation_probability'] = probabilities[: ,0]

        predictions_df.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")

        #print(f"\nPrediction Statistics:")
        #print(f"Total Predictions: {len(predictions)}")
        #print(f"Predicted violations: {sum(predictions)} ({sum(predictions) / len(predictions) * 100:.2f}%)")
        #print(f"Predicted no violations: {len(predictions) - sum(predictions)} ({(len(predictions) - sum(predictions))/len(predictions)*100:.2f}%)")
        #print(f"Average violation probability: {probabilities[:, 1].mean():.4f}")
        return predictions_df

def predict_new_examples(model, examples):

    if hasattr(model.model_body, 'module'):
        model.model_body = model.model_body.module
    if hasattr(model.model_head, 'module'):
        model.model_head = model.model_head.module
    
    texts = []
    for rule, post_body in examples:
        combined_text = f"Rule: {rule}\n\nPost: {post_body}"
        texts.append(combined_text)
    
    predictions = model(texts)
    probabilities = model.predict_proba(texts)
    
    if hasattr(predictions, 'cpu'):
        predictions = predictions.cpu().numpy()
    if hasattr(probabilities, 'cpu'):
        probabilities = probabilities.cpu().numpy()
    
    results = []
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        results.append({
            'text': examples[i][1][:100] + "...",
            'prediction': 'Violation' if pred == 1 else 'No Violation',
            'confidence': float(max(prob)),
            'probabilities': {'No Violation': float(prob[0]), 'Violation': float(prob[1])}
        })
    
    return results


def main():
   
    print("=== Multi-GPU SetFit Reddit Rule Violation Classification ===\n")
    
    gpu_info = get_gpu_info()
    
    train_texts, train_labels, test_texts, test_labels, train_df, test_df = load_data()
    
  
    if test_labels is not None:
        print("\nTest set has labels - creating validation split from training data")
        train_texts_split, val_texts, train_labels_split, val_labels = train_test_split(
            train_texts, train_labels, test_size=0.2, random_state=42, stratify=train_labels
        )
        print(f"Training size: {len(train_texts_split)}")
        print(f"Validation size: {len(val_texts)}")
        print(f"Test size: {len(test_texts)}")
    else:
        print("\nTest set has no labels - using test set for validation during training")
        train_texts_split, val_texts = train_texts, test_texts
        train_labels_split, val_labels = train_labels, None
        print(f"Training size: {len(train_texts_split)}")
        print(f"Test size (used for validation): {len(test_texts)}")
    
    if gpu_info["count"] > 1:
        use_distributed = False#input(f"\nFound {gpu_info['count']} GPUs. Use distributed training? (y/n): ").lower() == 'y'
        
        if use_distributed:
            print("Starting distributed training...")
            world_size = gpu_info["count"]
            nprocs = gpu_info["count"]  # Total number of processes

            mp.spawn(
                train,
                args=(nprocs, train_texts_split, train_labels_split, gpu_info, val_texts, val_labels),
                nprocs=nprocs,  # This tells mp.spawn how many processes to create
                join=True
            )

            print("Loading trained model for evaluation...")
            model = SetFitModel.from_pretrained("./reddit-rule-violation-setfit")
            
        else:
            print("Using DataParallel training...")
            model = create_setfit_model()
            #model = setup_model(model, use_distributed=False)
            
            trainer = train_setfit_model(model, train_texts_split, train_labels_split, gpu_info, 
                                                val_texts, val_labels)
            
            print("Saving model...")
            trainer.model.save_pretrained("./reddit-rule-violation-setfit")
            print("Model saved to: ./reddit-rule-violation-setfit")
            
            model = trainer.model
    else:
        print("Using single GPU/CPU training...")
        model = create_setfit_model()
        trainer = train_setfit_model(model, train_texts_split, train_labels_split, gpu_info, 
                                            val_texts, val_labels)
        model = trainer.model
    
    # Evaluate model on test set (if labels available)
    if test_labels is not None:
        print("\n=== Test Set Evaluation ===")
        metrics = evaluate_model(model, test_texts, test_labels)
    else:
        print("\n=== Test Set Prediction Generation ===")
        metrics = {}
    
    # Generate test predictions
    if test_df is not None:
        predictions_df = generate_test_predictions(model, test_texts, test_df)
    
    # Example predictions on sample data
    if metrics:  # Only if we evaluated (have test labels)
        print("\n=== Example Predictions ===")
        example_cases = [
            (
                "No Advertising: Spam, referral links, unsolicited advertising, and promotional content are not allowed.",
                "Check out this amazing product! Click here to buy now! www.example.com"
            ),
            (
                "No Advertising: Spam, referral links, unsolicited advertising, and promotional content are not allowed.",
                "I really enjoyed this movie. The cinematography was excellent and the story was compelling."
            ),
            (
                "No legal advice: Do not offer or request legal advice.",
                "You should definitely sue them. I'm a lawyer and this is clear cut."
            ),
            (
                "No legal advice: Do not offer or request legal advice.",
                "That sounds like a frustrating situation. Have you considered talking to them directly?"
            )
        ]
        
        results = predict_new_examples(model, example_cases)
        
        for i, result in enumerate(results, 1):
            print(f"\nExample {i}:")
            print(f"Text: {result['text']}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Probabilities: {result['probabilities']}")
    
    return model, metrics

if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Run training pipeline
    trained_model, metrics = main()
    
    if metrics:
        print(f"\n=== Training Complete ===")
        print(f"Final Accuracy: {metrics['accuracy']:.4f}")
        
        print("\n=== GPU Utilization Summary ===")
        gpu_info = get_gpu_info()
        for device in gpu_info["devices"]:
            print(f"GPU {device['id']}: {device['name']} - {device['memory_gb']:.1f} GB")

