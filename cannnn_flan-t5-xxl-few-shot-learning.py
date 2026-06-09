# Standard library imports
import os
import re

# Third-party imports
import numpy as np  # For linear algebra operations
import pandas as pd  # For data processing, CSV/Excel I/O
from tqdm.auto import tqdm  # Progress bars for loops and pandas operations
import torch  # PyTorch for deep learning
from sklearn.metrics import roc_auc_score  # Metric for binary classification

# Transformers library for NLP models
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # Generic seq2seq models
from transformers import T5Tokenizer, T5ForConditionalGeneration  # T5 specific models

# KaggleHub for Kaggle integration
import kagglehub

# Enable tqdm progress bars for pandas operations
tqdm.pandas()



# -----------------------------
# Data Cleaning
# -----------------------------
class TextCleaner:
    """Class responsible for cleaning text from emojis and kaomojis."""

    @staticmethod
    def remove_emojis_and_kaomoji(text: str) -> str:
        """Remove emojis and kaomojis from the text."""
        emoji_pattern = re.compile(
            "[" 
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002700-\U000027BF"
            "\U0001F900-\U0001F9FF"
            "\U00002600-\U000026FF"
            "\U0001FA70-\U0001FAFF"
            "\U00002500-\U00002BEF"
            "]+",
            flags=re.UNICODE
        )
        text = emoji_pattern.sub("", text)
        kaomoji_pattern = re.compile(r"[^\w\s,.!?@#%&()\-+=:;'\"/\\]")
        text = kaomoji_pattern.sub("", text)
        return text

    def clean_dataframe(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Apply cleaning to specified columns in a DataFrame."""
        for col in columns:
            df[col] = df[col].progress_apply(self.remove_emojis_and_kaomoji)
        return df



# -----------------------------
# Data Aggregation
# -----------------------------
class DataAggregator:
    """Class responsible for aggregating examples, rules, subreddits, and optionally the target."""

    @staticmethod
    def aggregate_examples(df: pd.DataFrame, target_col: str | None = None) -> pd.DataFrame:
        """
        Aggregate examples, rules, subreddits, and optionally include the target column.

        Args:
            df (pd.DataFrame): Input DataFrame to aggregate.
            target_col (str | None): Name of the target column to preserve, e.g., 'rule_violation'.

        Returns:
            pd.DataFrame: Aggregated DataFrame.
        """
        agg_dict = {
            "positive_1": ("positive_example_1", lambda x: ", ".join(x.dropna())),
            "positive_2": ("positive_example_2", lambda x: ", ".join(x.dropna())),
            "negative_1": ("negative_example_1", lambda x: ", ".join(x.dropna())),
            "negative_2": ("negative_example_2", lambda x: ", ".join(x.dropna())),
            "rules": ("rule", lambda x: ", ".join(np.unique(x.dropna()))),
            "subreddits": ("subreddit", lambda x: ", ".join(np.unique(x.dropna())))
        }

        if target_col and target_col in df.columns:
            # If target exists, take the first value in each group
            agg_dict[target_col] = (target_col, "first")

        grouped = df.groupby(["body", "row_id"]).agg(**agg_dict).reset_index()

        # Combine positive and negative examples into single columns
        grouped["positive_examples"] = grouped["positive_1"] + ", " + grouped["positive_2"]
        grouped["negative_examples"] = grouped["negative_1"] + ", " + grouped["negative_2"]

        # Keep relevant columns
        columns_to_keep = ["row_id", "body", "positive_examples", "negative_examples", "rules", "subreddits"]
        if target_col and target_col in grouped.columns:
            columns_to_keep.append(target_col)

        return grouped[columns_to_keep]



# -----------------------------
# Model Interface
# -----------------------------
class BaseViolationModel:
    """Abstract model interface for rule violation prediction."""

    def predict_batch(self, batch: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for a batch of rows."""
        raise NotImplementedError


class T5ViolationModel(BaseViolationModel):
    """T5-based model for rule violation prediction."""

    def __init__(self, model_path: str, max_length: int = 512):
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.max_length = max_length
        self.device = self.model.device

        # Precompute the token ID for "Yes"
        self.yes_token_id = self.tokenizer("Yes", return_tensors="pt").input_ids[0][0].item()

    def predict_batch(self, batch: pd.DataFrame) -> np.ndarray:
        """
        Batch inference: generate rule violation probabilities for multiple rows at once.

        This version avoids looping over scores by using the logits of the first generated token
        directly across the entire batch, which is much faster and VRAM-efficient.
        """
        # Prepare prompts
        prompts = []
        for _, row in batch.iterrows():
            prompt = f"""
            Rule Violation Detection Task
            Comment:
            {row["body"]}

            Rule:
            {row["rules"]}

            Subreddit:
            {row["subreddits"]}

            Positive Examples:
            {row["positive_examples"]}

            Negative Examples:
            {row["negative_examples"]}

            Does the comment violate the rule? Answer with "Yes" or "No".
            """
            prompts.append(prompt.strip())

        # Tokenize entire batch at once
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        with torch.no_grad():
            # Generate one token per example
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True
            )

        # Extract logits of the first generated token across the batch
        # outputs.scores[0] has shape: [batch_size, vocab_size]
        first_token_logits = outputs.scores[0]  # Tensor of shape (batch_size, vocab_size)
        probs = torch.nn.functional.softmax(first_token_logits, dim=-1)  # Convert to probabilities

        # Gather probabilities corresponding to "Yes" token
        yes_probs = probs[:, self.yes_token_id].cpu().numpy()  # shape: (batch_size,)

        return yes_probs



# -----------------------------
# Prediction Pipeline
# -----------------------------
class ViolationPredictor:
    """Class to manage batch predictions using a model."""

    def __init__(self, model: BaseViolationModel, chunk_size: int = 16):
        self.model = model
        self.chunk_size = chunk_size

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make batched predictions on the entire DataFrame."""
        results = []
        for i in tqdm(range(0, len(df), self.chunk_size), desc="Processing chunks"):
            chunk = df.iloc[i:i + self.chunk_size].copy()
            # Use batched model inference
            chunk["rule_violation"] = self.model.predict_batch(chunk)
            results.append(chunk)
            torch.cuda.empty_cache()  # Clear GPU memory after each batch
        return pd.concat(results).reset_index(drop=True)


# -----------------------------
# Output
# -----------------------------
class SubmissionExporter:
    """Class to handle exporting predictions to CSV."""

    @staticmethod
    def export(df: pd.DataFrame, path: str):
        submission = df[["row_id", "rule_violation"]]
        submission.to_csv(path, index=False)
        print(f"Submission saved to {path}")



if __name__ == "__main__":
    # Reduce VRAM usage
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Load train and test data
    #train_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
    test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

    # Clean text
    cleaner = TextCleaner()
    #train_df = cleaner.clean_dataframe(
    #    train_df,
    #    ["body", "positive_example_1", "positive_example_2", "negative_example_1", "negative_example_2"]
    #)
    
    test_df = cleaner.clean_dataframe(
        test_df,
        ["body", "positive_example_1", "positive_example_2", "negative_example_1", "negative_example_2"]
    )

    # Aggregate examples
    aggregator = DataAggregator()
    #grouped_train = aggregator.aggregate_examples(train_df)
    grouped_test = aggregator.aggregate_examples(test_df)

    # Load T5 model
    kagglehub.model_download("google/flan-t5/pyTorch/xxl")
    model_path = "/kaggle/input/flan-t5/pytorch/xxl/1/flan-t5/xxl"
    violation_model = T5ViolationModel(model_path, max_length=512)

    # Make predictions on train set for calibration
    predictor = ViolationPredictor(violation_model, chunk_size=8)

    # Make predictions on test set
    grouped_test = predictor.predict_dataframe(grouped_test)

    # Export submission
    SubmissionExporter.export(grouped_test, "submission.csv")

