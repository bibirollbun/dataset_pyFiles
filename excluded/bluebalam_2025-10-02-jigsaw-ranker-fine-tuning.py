import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import RerankingEvaluator
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers import SentenceTransformerTrainer
from datasets import Dataset
import torch

from tqdm.auto import tqdm

from datetime import datetime


def analyze_grouped_dataset(grouped_data):
    """
    Analyze the distribution of grouped triplet data
    
    Args:
        grouped_data: List of dicts with 'query', 'positives', 'negatives'
    
    Returns:
        Dictionary with dataset statistics
    """
    total_queries = len(grouped_data)
    total_positives = sum(len(item['positives']) for item in grouped_data)
    total_negatives = sum(len(item['negatives']) for item in grouped_data)
    
    pos_per_query = [len(item['positives']) for item in grouped_data]
    neg_per_query = [len(item['negatives']) for item in grouped_data]
    
    return {
        'total_queries': total_queries,
        'total_positives': total_positives,
        'total_negatives': total_negatives,
        'avg_positives_per_query': np.mean(pos_per_query),
        'avg_negatives_per_query': np.mean(neg_per_query),
        'min_positives_per_query': min(pos_per_query),
        'max_positives_per_query': max(pos_per_query),
        'min_negatives_per_query': min(neg_per_query),
        'max_negatives_per_query': max(neg_per_query)
    }


def split_grouped_data_content_aware(grouped_data, train_size=0.7, val_size=0.15, test_size=0.15, 
                                   random_state=42, min_pos_per_split=1, min_neg_per_split=1):
    """
    Split grouped data ensuring disjoint positive/negative content across splits
    
    Args:
        grouped_data: List of dicts with 'query', 'positives', 'negatives'
        train_size: Proportion for training set
        val_size: Proportion for validation set  
        test_size: Proportion for test set
        random_state: Random seed for reproducibility
        min_pos_per_split: Minimum positives per query per split
        min_neg_per_split: Minimum negatives per query per split
    
    Returns:
        Tuple of (train_grouped, val_grouped, test_grouped)
    """
    print(f"Splitting {len(grouped_data)} queries...")
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Sizes must sum to 1.0"
    
    np.random.seed(random_state)
    train_grouped, val_grouped, test_grouped = [], [], []
    
    for i, item in enumerate(grouped_data):
        query = item['query']
        positives = item['positives'].copy()
        negatives = item['negatives'].copy()
        
        n_pos, n_neg = len(positives), len(negatives)
        print(f"Query {i+1}: {n_pos} positives, {n_neg} negatives")
        
        # For small datasets, use simpler splitting
        if len(grouped_data) <= 3:
            # Put first query in train, second in val, third in test
            if i == 0:
                train_grouped.append({'query': query, 'positives': positives, 'negatives': negatives})
            elif i == 1 and len(grouped_data) > 1:
                val_grouped.append({'query': query, 'positives': positives, 'negatives': negatives})
            elif i == 2 and len(grouped_data) > 2:
                test_grouped.append({'query': query, 'positives': positives, 'negatives': negatives})
            continue
        
        # Check if we have enough content for splitting
        if n_pos < 3 * min_pos_per_split or n_neg < 3 * min_neg_per_split:
            print(f"WARNING: Query '{query[:50]}...' insufficient content. Assigning to training.")
            train_grouped.append({'query': query, 'positives': positives, 'negatives': negatives})
            continue
        
        # Shuffle and split content
        np.random.shuffle(positives)
        np.random.shuffle(negatives)
        
        # Calculate split sizes for positives
        n_pos_train = max(min_pos_per_split, int(n_pos * train_size))
        n_pos_val = max(min_pos_per_split, int(n_pos * val_size))
        n_pos_test = n_pos - n_pos_train - n_pos_val
        
        if n_pos_test < min_pos_per_split:
            n_pos_test = min_pos_per_split
            n_pos_val = max(min_pos_per_split, n_pos - n_pos_train - n_pos_test)
            n_pos_train = n_pos - n_pos_val - n_pos_test
        
        # Calculate split sizes for negatives
        n_neg_train = max(min_neg_per_split, int(n_neg * train_size))
        n_neg_val = max(min_neg_per_split, int(n_neg * val_size))
        n_neg_test = n_neg - n_neg_train - n_neg_val
        
        if n_neg_test < min_neg_per_split:
            n_neg_test = min_neg_per_split
            n_neg_val = max(min_neg_per_split, n_neg - n_neg_train - n_neg_test)
            n_neg_train = n_neg - n_neg_val - n_neg_test
        
        # Create splits
        train_pos = positives[:n_pos_train]
        val_pos = positives[n_pos_train:n_pos_train + n_pos_val]
        test_pos = positives[n_pos_train + n_pos_val:]
        
        train_neg = negatives[:n_neg_train]
        val_neg = negatives[n_neg_train:n_neg_train + n_neg_val]
        test_neg = negatives[n_neg_train + n_neg_val:]
        
        # Add to splits if both positives and negatives exist
        if train_pos and train_neg:
            train_grouped.append({'query': query, 'positives': train_pos, 'negatives': train_neg})
        if val_pos and val_neg:
            val_grouped.append({'query': query, 'positives': val_pos, 'negatives': val_neg})
        if test_pos and test_neg:
            test_grouped.append({'query': query, 'positives': test_pos, 'negatives': test_neg})
    
    print(f"Final splits: {len(train_grouped)} train, {len(val_grouped)} val, {len(test_grouped)} test")
    return train_grouped, val_grouped, test_grouped


def prepare_dataset_for_mnr(grouped_data):
    """
    Prepare dataset for MultipleNegativesRankingLoss with explicit negatives
    Format: (anchor, positive, negative_1, negative_2, ..., negative_n)
    
    Args:
        grouped_data: List of dicts with 'query', 'positives', 'negatives'
    
    Returns:
        Dataset object for MNR loss
    """
    if not grouped_data:
        raise ValueError("grouped_data cannot be empty")
    
    examples = [{"query": item['query'], "pos": item['positives'], 'neg': item['negatives']} for item in grouped_data]
        
    
    print(f"Created {len(examples)} training examples from {len(grouped_data)} queries")
    
    return examples


def create_reranking_evaluator_from_grouped(grouped_data, max_triplets_per_query=20, name="reranking"):
    """
    Create RerankingEvaluator from grouped data
    
    Args:
        grouped_data: List of dicts with 'query', 'positives', 'negatives'
        max_triplets_per_query: Maximum documents per query for reranking
        name: Name for the evaluator
    
    Returns:
        RerankingEvaluator object
    """
    samples = []
    np.random.seed(42)
    
    for item in grouped_data:
        query = item['query']
        positives = item['positives']
        negatives = item['negatives']
        
        # Sample negatives if too many
        sampled_negatives = negatives
        if len(negatives) > max_triplets_per_query:
            sampled_negatives = np.random.choice(
                negatives, size=max_triplets_per_query, replace=False
            ).tolist()
        
        # Create sample in the format expected by RerankingEvaluator
        sample = {
            'query': query,
            'positive': positives,  # List of positive documents
            'negative': sampled_negatives  # List of negative documents
        }
        samples.append(sample)
    
    return RerankingEvaluator(samples=samples, name=name)


def analyze_content_overlap(train_grouped, val_grouped, test_grouped):
    """
    Analyze content overlap across splits
    
    Args:
        train_grouped: Training grouped data
        val_grouped: Validation grouped data
        test_grouped: Test grouped data
    
    Returns:
        Dictionary with overlap statistics
    """
    def extract_content(data):
        pos_set, neg_set = set(), set()
        for item in data:
            pos_set.update(item['positives'])
            neg_set.update(item['negatives'])
        return pos_set, neg_set
    
    train_pos, train_neg = extract_content(train_grouped)
    val_pos, val_neg = extract_content(val_grouped)
    test_pos, test_neg = extract_content(test_grouped)
    
    return {
        'positive_overlaps': {
            'train_val': len(train_pos & val_pos),
            'train_test': len(train_pos & test_pos),
            'val_test': len(val_pos & test_pos)
        },
        'negative_overlaps': {
            'train_val': len(train_neg & val_neg),
            'train_test': len(train_neg & test_neg),
            'val_test': len(val_neg & test_neg)
        }
    }

def evaluate_model(model, test_grouped, max_triplets_per_query=20):
    """
    Evaluate trained model on test data using RerankingEvaluator
    
    Args:
        model: SentenceTransformer model
        test_grouped: Test grouped data
        max_triplets_per_query: Maximum documents per query for evaluation
    
    Returns:
        Evaluation score
    """
    evaluator = create_reranking_evaluator_from_grouped(test_grouped, max_triplets_per_query, name="test")
    return evaluator(model)




TRAIN_DATASET_PATH = "/kaggle/input/jigsaw-agile-community-rules/train.csv"
TEST_DATASET_PATH = "/kaggle/input/jigsaw-agile-community-rules/test.csv"


train_df = pd.read_csv(TRAIN_DATASET_PATH)
test_df = pd.read_csv(TEST_DATASET_PATH)


train_df.head()


bodies_train = train_df["body"].to_numpy()
rules_train = train_df["rule"].to_numpy()
subrs_train = train_df["subreddit"].to_numpy()
pos1s_train = train_df["positive_example_1"].to_numpy()
pos2s_train = train_df["positive_example_2"].to_numpy()
neg1s_train = train_df["negative_example_1"].to_numpy()
neg2s_train = train_df["negative_example_2"].to_numpy()
y_rule_violation_train = train_df["rule_violation"].to_numpy()


set(rules_train)


train_examples = {}

for i in tqdm(range(len(bodies_train))):
    query = f"r/{subrs_train[i]}. {rules_train[i]}."

    positives = set()
    negatives = set()

    positives.add(pos1s_train[i])
    positives.add(pos2s_train[i])

    negatives.add(neg1s_train[i])
    negatives.add(neg2s_train[i])

    if int(y_rule_violation_train[i]) == 1:
        positives.add(bodies_train[i])
    else:
        negatives.add(bodies_train[i])

    if query in train_examples:
        positives.update(train_examples[query]["positives"])
        negatives.update(train_examples[query]["negatives"])
        
    train_examples[query] = {"positives": list(positives), "negatives": list(negatives)}




train_examples_grouped = [{"query": q, 'positives': item['positives'], 'negatives': item['negatives']} for q, item in tqdm(train_examples.items())]


train_examples_grouped[0]


# Analyze dataset
stats = analyze_grouped_dataset(train_examples_grouped)
print("Dataset Analysis:")
print(f"- Total queries: {stats['total_queries']}")
print(f"- Total positives: {stats['total_positives']}")
print(f"- Total negatives: {stats['total_negatives']}")
print(f"- Avg positives per query: {stats['avg_positives_per_query']:.2f}")
print(f"- Avg negatives per query: {stats['avg_negatives_per_query']:.2f}")


# Split data with content separation
print("\n=== Splitting Data ===")
try:
    train_grouped, val_grouped, test_grouped = split_grouped_data_content_aware(
        train_examples_grouped,
        train_size=0.75,
        val_size=0.2,
        test_size=0.05,
        random_state=42
    )
    
    print(f"Split results:")
    print(f"- Training queries: {len(train_grouped) if train_grouped else 0}")
    print(f"- Validation queries: {len(val_grouped) if val_grouped else 0}")
    print(f"- Test queries: {len(test_grouped) if test_grouped else 0}")
    
    # Debug: Print actual content
    if train_grouped:
        print(f"First training query: {train_grouped[0]['query'][:50]}...")
        print(f"Positives: {len(train_grouped[0]['positives'])}, Negatives: {len(train_grouped[0]['negatives'])}")
    else:
        print("ERROR: No training data after split!")
        
except Exception as e:
    print(f"ERROR during data splitting: {e}")

# Analyze content overlap
print("\n=== Content Overlap Analysis ===")
try:
    if train_grouped and val_grouped and test_grouped:
        overlap_stats = analyze_content_overlap(train_grouped, val_grouped, test_grouped)
        print(f"Content overlaps:")
        print(f"- Positive overlaps: {overlap_stats['positive_overlaps']}")
        print(f"- Negative overlaps: {overlap_stats['negative_overlaps']}")
    else:
        print("Skipping overlap analysis due to empty splits")
except Exception as e:
    print(f"ERROR during overlap analysis: {e}")



# Training configuration

# We use `e5` since it gave us good reults for the down-stream task: classification.
# Other base models could be used for experimentation.
base_model_name = 'intfloat/e5-base'

# This is your output dir where the model will be saved:
output_dir = "./2025-10-02-fine-tuned-e5-base"

# Params:
batch_size = 4
learning_rate = 1e-5
eval_steps = 8
save_steps = 8
num_train_epochs = 3

# Since we use a ranking metric, we also specify the desired cut-off of the ranking to evaluate:
at_k = 10

# For the ranking metric to keep track during the fine-tuning process, we use ndcg@k
# We also experimented optimizing for `mrr` and `map`
metric_for_best_model = f"eval_validation_ndcg@{at_k}"


# Debug: Check if we have training data
if not train_grouped:
    print("ERROR: No training data available after splitting!")

print(f"Training data available: {len(train_grouped)} queries")





# Load model
model = SentenceTransformer(base_model_name, device="cuda")    



train_dataset_examples = prepare_dataset_for_mnr(train_grouped)



train_dataset_examples[0]


train_dataset = Dataset.from_list(train_dataset_examples)


loss = losses.MultipleNegativesRankingLoss(model)


# Create evaluator
if val_grouped:
    # Use RerankingEvaluator instead of TripletEvaluator
    evaluator = create_reranking_evaluator_from_grouped(val_grouped, max_triplets_per_query=(2*at_k), name="validation")
    print(f"RerankingEvaluator created with validation data: {len(val_grouped)} queries")
else:
    evaluator = None
    print("WARNING: No validation data available, training without evaluation")




len(list(set(bodies_train)))


# Training arguments
args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=learning_rate,
    warmup_ratio=0.05,
    fp16=torch.cuda.is_available(),
    bf16=False,
    batch_sampler="no_duplicates",
    eval_strategy="steps" if evaluator else "no",
    eval_steps=eval_steps if evaluator else None,
    save_strategy="steps",
    save_steps=save_steps,
    logging_steps=eval_steps,
    save_total_limit=3,
    load_best_model_at_end=True if evaluator else False,
    metric_for_best_model=metric_for_best_model if evaluator else None,
    greater_is_better=True,
    run_name=f"sentence-transformer-mnr",
    report_to="none",
    seed=42,
    data_seed=42,
)



# Create trainer

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=None,
    loss=loss,
    evaluator=evaluator,
)
print("Trainer created successfully")
    


print(f"Training configuration:")
print(f"- Model: {base_model_name}")
print(f"- Training queries: {len(train_grouped)}")
print(f"- Training examples: {len(train_dataset)}")
print(f"- Validation queries: {len(val_grouped)}")
print(f"- Batch size: {batch_size}")
print(f"- Epochs: {num_train_epochs}")
print(f"- Learning rate: {learning_rate}")
print(f"- metric_for_best_model: {metric_for_best_model}")


# Evaluate on validation
print("base model on validation set") 
if val_grouped:
    val_score = evaluate_model(model, val_grouped, max_triplets_per_query=(2*at_k))
    print(f"Eval score before fine tuning (VALIDATION SET): {val_score}")


# Evaluate on test set
print("base model on test set") 
if test_grouped:
    test_score = evaluate_model(SentenceTransformer(base_model_name, device="cuda"), test_grouped, max_triplets_per_query=(2*at_k))
    print(f"Test evaluation (TEST SET): {test_score}")


# Train
trainer.train()
trainer.save_model()
print("Training completed successfully!")


# Evaluate fine-tuned model on validation set
if val_grouped:
    val_score = evaluate_model(model, val_grouped, max_triplets_per_query=20)
    print(f"Eval score after fine tuning: {val_score}")


# Evaluate
print("new model")
if test_grouped:
    test_score = evaluate_model(SentenceTransformer(output_dir, device="cuda"), test_grouped, max_triplets_per_query=(2*at_k))
    print(f"Test evaluation: {test_score}")


print("~ fin ~")

