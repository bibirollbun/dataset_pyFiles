pip install -q -U wandb nltk rouge-score thefuzz python-Levenshtein bert-score evaluate transformers peft datasets janome numpy fuzzywuzzy bitsandbytes ml_dtypes tf_keras torch torchvision pytorch-lightning tensorflow scikit-learn tokenizers==0.20.1 huggingface_hub


path = "google/gemma-2/transformers/gemma-2-2b" # Make sure to change this to the path where the model is stored


config = {

    # Core Learning Parameters
    "learning_rate": 5e-5,                  # How fast the model learns (0.00005)
    "continuous_thoughts": 4,               # Number of latent space reasoning steps
    "stages": 4,                            # Number of training curriculum stages
    "training_thoughts_sequence_length": 50, # Number of thought sequence to generate

    # Inference and Evaluation Params       
    "fuzzy_matcher_threshold": 80,          # Fuzzy matcher threshold at 80%
    "cot_decoding_k": 5,                    # Number of paths to try before finding the best answer

    # Model Setup
    "max_length": 256,                      # Maximum text length to process
    "model_name": path,                     # Path to Gemma model
    "batch_size": 4,                        # Number of examples processed together
    "weight_decay": 0.01,                   # Helps prevent overfitting

    # Special Tokens
    "bot_id": "<bot>",                      # Marks start of latent reasoning
    "eot_id": "<eot>",                      # Marks end of latent reasoning
    "answer_id": "<answer>",                # Marks the begining of answer
    "debug": True,                          # Enables debugging output. Also allows you see the model's thoughts

    # Training Optimizations
    "bf16": True,                           # Uses BFloat16 for faster training
    "per_device_train_batch_size": 1,       # Samples per GPU/CPU
    "optim": "adamw_torch",                 # AdamW optimizer for efficiency
    "wandb_project": "gemma2-finetuning",   # Tracks training on Weights & Biases
    "logging_steps": 1,                     # How often to log training progress
    "bf16_full_eval": True,                 # Uses BFloat16 for evaluation
    "gradient_accumulation_steps": 1,       # How often to update weights
    "save_steps": 10000,                    # How often to save model
    "warmup_steps": 0.1,                    # Number of warmup steps
    "output_dir": "output",                 # Where to save model files
    "diversity_weight": 0.1,                # Reasoning diversity weight
    "coherence_weight": 0.1                 # Reasoning coherence weight
}


from datasets import load_dataset, DatasetDict
from datasets import config as dataset_config
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

dataset_name = "izumi-lab/llm-japanese-dataset"
dataset = load_dataset(dataset_name)

# For this tutorial, let's take 30,000k samples from the dataset
item = 30000

truncated_dataset = DatasetDict({
    split: dataset[split].select(range(item))
    for split in dataset.keys()
})


dataset = truncated_dataset
eval_dataset = dataset



for i in range(3):
    print("Instruction: ", dataset['train']["instruction"][i], "\n")
    print("Input: ", dataset['train']["input"][i], "\n")
    print("Output: ",dataset['train']["output"][i], "\n")
    print(f"{'='*200}\n")




# Implementing the LanguageDetector class

from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ScriptRange:
    """Represents a Unicode range for a writing system"""
    start: int
    end: int
    name: str
    
class LanguageDetector:
    def __init__(self):
        self.scripts: List[ScriptRange] = []
        self.language_mappings: Dict[str, List[str]] = {}
        
    def add_script(self, name: str, start: int, end: int) -> None:
        """
        Add a new script range to the detector
        
        Args:
            name: Name of the script (e.g., 'Hiragana', 'Latin')
            start: Starting Unicode code point
            end: Ending Unicode code point
        """
        self.scripts.append(ScriptRange(start, end, name))
    
    def map_scripts_to_language(self, language: str, script_names: List[str]) -> None:
        """
        Map multiple scripts to a single language
        
        Args:
            language: Name of the language (e.g., 'Japanese')
            script_names: List of script names that belong to this language
        """
        self.language_mappings[language] = script_names
    
    def detect(self, text: str) -> Dict[str, float]:
        """
        Detect the percentage of different languages/scripts in the text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary mapping language/script names to their percentage presence
        """
        # Count characters in each script
        char_counts: Dict[str, int] = {script.name: 0 for script in self.scripts}
        total_chars = 0
        
        for char in text:
            if char.isspace() or char in '.,!?()[]{}':
                continue
                
            code = ord(char)
            total_chars += 1
            
            # Check which script range the character falls into
            for script in self.scripts:
                if script.start <= code <= script.end:
                    char_counts[script.name] += 1
                    break
        
        if total_chars == 0:
            return {}
            
        # Calculate initial percentages
        percentages = {
            script: (count / total_chars) * 100
            for script, count in char_counts.items()
            if count > 0
        }
        
        # Combine scripts into languages where applicable
        final_percentages = {}
        used_scripts = set()
        
        # First, handle mapped languages
        for language, script_names in self.language_mappings.items():
            total = sum(percentages.get(script, 0) for script in script_names)
            if total > 0:
                final_percentages[language] = total
                used_scripts.update(script_names)
        
        # Then add remaining unmapped scripts
        for script, percentage in percentages.items():
            if script not in used_scripts:
                final_percentages[script] = percentage
        
        return {k: round(v, 1) for k, v in sorted(
            final_percentages.items(),
            key=lambda x: x[1],
            reverse=True
        )}

# Example setup and usage
def create_default_detector() -> LanguageDetector:
    """Create a detector with Japanese and English support"""
    detector = LanguageDetector()
    
    # Add Japanese scripts
    detector.add_script('Hiragana', 0x3040, 0x309F)
    detector.add_script('Katakana', 0x30A0, 0x30FF)
    detector.add_script('Kanji', 0x4E00, 0x9FFF)

    # Add English scripts
    detector.add_script('Latin', 0x0000, 0x024F)

    
    # Map scripts to languages
    detector.map_scripts_to_language('Japanese', ['Hiragana', 'Katakana', 'Kanji'])
    detector.map_scripts_to_language('English', ['Latin'])
    
    return detector

if __name__ == "__main__":
    detector = create_default_detector()
    
    test_texts = [
        'スナフキン',
        'レベッカ(REBECCA)',
        'Hello World',
        'こんにちは World!'
    ]
    
    for text in test_texts:
        result = detector.detect(text)
        print(f"Text: {text} ===>>>> {result}")


from transformers import PreTrainedTokenizer

def preprocess_function(
    examples, 
    detector=None,  # Make detector optional
    stages=1, 
    eos_token="<eos>",
    bos_token="<bos>",
    language_config=None,
):
    """
    Preprocess the input examples by constructing the prompt with reasoning steps.

    Args:
        examples (dict): A dictionary containing the input examples with keys "instruction", "input", and "output".
        detector: A language detection object or function that detects the language of a given text.
        stages (int): The number of reasoning stages to include in the prompt.
        eos_token (str): The end-of-sequence token.
        bos_token (str): The beginning-of-sequence token.
        language_config (dict): A dictionary mapping language keys to their respective translations for steps and labels.

    Returns:
        dict: A dictionary containing the preprocessed prompts.
    """

    if language_config is None:
        language_config = {
            "English": {
                "language_detection": "Question language detection",
                "understand_question": "Understand the question",
                "understand_answer": "Understand the answer",
                "response_language_detection": "Response language detection",
                "answer_label": "Answer:",
                "step_label": "Step",
            },
            "Japanese": {
                "language_detection": "言語の検出",
                "understand_question": "質問を理解する",
                "understand_answer": "答えを理解する",
                "response_language_detection": "応答言語の検出",
                "answer_label": "答え：",
                "step_label": "ステップ",
            },
            # Add more languages here as needed
        }

    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]

    bot = config["bot_id"]
    eot = config["eot_id"]
    answer_token = config["answer_id"]

    # Initialize output dictionaries with lists
    result = []

    for i in range(len(instructions)):
        instruction = instructions[i]
        input = inputs[i]
        output = outputs[i]

        if len(input) > 1:
            input = instruction + input
        else:
            input = instruction

        # Use the provided detector to detect languages
        input_language = detector.detect(input) if detector else {"English": 100.0}  # Default to English if no detector
        output_language = detector.detect(output) if detector else {"English": 100.0}  # Default to English if no detector

        steps = []

        # Determine the primary input and output languages
        # Use the language key from the detector's output that matches a key in language_config
        input_lang = next((lang for lang in input_language if lang in language_config), "English")
        output_lang = next((lang for lang in output_language if lang in language_config), "English")

        # Get the language-specific labels
        input_labels = language_config.get(input_lang, language_config["English"])
        output_labels = language_config.get(output_lang, language_config["English"])

        # Input language detection
        input_lang_str = ", ".join([f"{k}: {v}%" for k, v in input_language.items()])
        steps.append(f"{input_labels['language_detection']}: {input_lang_str}")
        steps.append(f"{input_labels['understand_question']}: {input}")
        steps.append(f"{input_labels['understand_answer']}: {output}")

        # Output language detection
        output_lang_str = ", ".join([f"{k}: {v}%" for k, v in output_language.items()]) if output_language else "Unknown"
        steps.append(f"{output_labels['response_language_detection']}: {output_lang_str}")

        # Format steps with step numbers
        steps = [f"{output_labels['step_label']} {i+1} : {step}" for i, step in enumerate(steps)]

        # Include only the steps relevant to the current stage
        if stages > 0:
            steps = steps[-stages:]  # Keep the last `stages` steps

        # Renumber steps to start from 1
        steps = [f"{output_labels['step_label']} {i+1} : {step.split(' : ')[1]}" for i, step in enumerate(steps)]

        # Construct the prompt
        prompt = bos_token + "\n" + input + eos_token + bot + eot + "\n" + "\n".join(steps) + "\n" + answer_token + output_labels['answer_label'] + output + eos_token

        result.append(prompt)

    return {
        "prompt": result
    }


import torch

def tokenizer_function(examples, tokenizer):
    """
    Tokenize the input prompt and prepare the input_ids, attention_mask, and labels for training.

    Args:
        examples (dict): A dictionary containing the input prompts.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenization.

    Returns:
        dict: A dictionary containing the tokenized input_ids, attention_mask, and labels.
    """

    prompt = examples["prompt"]
    eot = config["eot_id"]

    tokenized = tokenizer(
        prompt,
        max_length=config["max_length"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = tokenized["input_ids"].squeeze(0)
    attention_mask = tokenized["attention_mask"].squeeze(0)

    labels = input_ids.clone()
    batch_size = labels.shape[0]
    eot_id = tokenizer.convert_tokens_to_ids(eot)

    for i in range(batch_size):
        # Find the positions of <eot> in the input_ids
        eot_pos = (input_ids[i] == eot_id).nonzero(as_tuple=True)

        if len(eot_pos[0]) > 0:
            # Get the last occurrence of <eot>
            last_eot_pos = eot_pos[0][-1].item()
            
            # Mask everything before and including the last <eot>
            labels[i, :last_eot_pos] = -100

        # Mask padding
        labels[i, attention_mask[i] == 0] = -100


    value =  {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


    if torch.cuda.device_count() > 1:
        return value
    else:
        value["labels"] = labels
        return value


language_config = {
    "English": {
        "language_detection": "Question language detection",
        "understand_question": "Understand the question",
        "understand_answer": "Understand the answer",
        "response_language_detection": "Response language detection",
        "answer_label": "Answer:",
        "step_label": "Step",
    },
    "Japanese": {
        "language_detection": "言語の検出",
        "understand_question": "質問を理解する",
        "understand_answer": "答えを理解する",
        "response_language_detection": "応答言語の検出",
        "answer_label": "答え：",
        "step_label": "ステップ",
    },
    # Add more languages here as needed
}


# So we do not load every dataset as this takes a while
truncated_dataset = DatasetDict({
    split: dataset[split].select(range(5))
    for split in dataset.keys()
})

for stage in range(config["stages"]):
    dataset_ = truncated_dataset.map(
        (lambda x: preprocess_function(
            x, 
            detector=detector,
            stages=stage, 
            language_config=language_config
        )),
        batched=True,
        batch_size=config["batch_size"],
    )

    print(f"Stage: ========================>>>>>>>>>>>>>>>>> {stage}")
    for i in range(5):
        print("Input: ", dataset_['train']["prompt"][i], "\n")
        print(f"{'='*100}\n")




import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GemmaForCausalLM, DynamicCache, PreTrainedTokenizer
from typing import Optional, List, Union, Dict, Any
import logging

logger = logging.getLogger(__name__)

class LatentReasoningGemmaForCausalLM(GemmaForCausalLM):
    """
    A custom implementation of GemmaForCausalLM that supports latent reasoning 
    using the Coconut (Chain of Continuous Thought) paradigm.
    """

    DEFAULT_CONFIG = {
        # Core Learning Parameters
        "continuous_thoughts": 4,               # Number of latent space reasoning steps
        "stages": 4,                            # Number of training curriculum stages
        "training_thoughts_sequence_length": 50, # Number of thought sequence to generate

        # Inference and Evaluation Params       
        "fuzzy_matcher_threshold": 80,          # Fuzzy matcher threshold at 80%
        "cot_decoding_k": 5,                    # Number of paths to try before finding the best answer

        # Model Setup
        "max_length": 256,                      # Maximum text length to process

        # Special Tokens
        "bot_id": "<bot>",                      # Marks start of latent reasoning
        "eot_id": "<eot>",                      # Marks end of latent reasoning
        "answer_id": "<answer>",                # Marks the begining of answer
        "debug": True,                          # Enables debugging output. Also allows you see the model's thoughts

    }
    
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer: PreTrainedTokenizer = None
        self.current_stage = 0
        self.model_config = type(self).DEFAULT_CONFIG
        self.debug = self.model_config.get("debug", False)
        self.diversity_weight = self.model_config.get("diversity_weight", 0.1)
        self.coherence_weight = self.model_config.get("coherence_weight", 0.1)

    def get_input_ids(self, inputs_embeds):
        """Helper method to get input ids from embeddings."""
        embedding_matrix = self.get_input_embeddings().weight
        similarities = torch.matmul(inputs_embeds, embedding_matrix.T)
        token_ids = torch.argmax(similarities, dim=-1)
        return token_ids

    def thoughts_forward(self, num_thoughts, thought_ids, thought_mask, num_of_thought_tokens = 1):
        """
        Generate continuous thought embeddings.
        """
        all_thought_outputs = []
        batch_size = thought_ids.shape[0]
        
        # Get initial embeddings
        initial_embeds = self.get_input_embeddings()(thought_ids)
        current_embeds = initial_embeds
        current_mask = thought_mask

        for t in range(num_thoughts):
            # Forward pass through transformer
            outputs = self.model.forward(
                inputs_embeds=current_embeds,
                attention_mask=current_mask,
                past_key_values=None,
                use_cache=False,
                return_dict=True,
                output_hidden_states=True,  # Get hidden states from all layers
            )
            
            # Get hidden states from all layers for better representation
            hidden_states = outputs.hidden_states
            
            # Combine hidden states from different layers using attention
            layer_attention = torch.softmax(
                torch.randn(len(hidden_states), device=hidden_states[0].device), 
                dim=0
            )
            weighted_states = sum(w * h for w, h in zip(layer_attention, hidden_states))
            
            n = num_of_thought_tokens
            last_hidden = weighted_states[:, -n:, :]  # [batch_size, n, hidden_size]
            
            # Project to lower dimension for thought space
            thought_proj = nn.Sequential(
                nn.Linear(last_hidden.shape[-1], self.config.hidden_size // 2),
                nn.LayerNorm(self.config.hidden_size // 2),
                nn.GELU()
            ).to(last_hidden.device)
            projected_thought = thought_proj(last_hidden)  # [batch_size, n, hidden_size // 2]
            
            # Add noise to increase diversity
            noise = torch.randn_like(projected_thought) * 0.1  # Adjust noise scale as needed
            projected_thought = projected_thought + noise
            
            # Project back to embedding space
            embed_proj = nn.Linear(
                self.config.hidden_size // 2,
                self.config.hidden_size,
                device=projected_thought.device
            )
            next_token_embeds = embed_proj(projected_thought)  # [batch_size, n, hidden_size]
            
            # Apply layer normalization for stability
            next_token_embeds = nn.LayerNorm(
                self.config.hidden_size,
                device=next_token_embeds.device
            )(next_token_embeds)
            
            # Update embeddings and mask
            current_embeds = torch.cat([current_embeds, next_token_embeds], dim=1)
            current_mask = torch.cat([
                current_mask,
                torch.ones((batch_size, n), device=current_mask.device)
            ], dim=1)
            
            all_thought_outputs.append(last_hidden)

        # Ensure reasonable sequence length
        max_seq_len = self.model_config.get("max_length", 512)
        if current_embeds.shape[1] > max_seq_len:
            current_embeds = current_embeds[:, :max_seq_len, :]
            current_mask = current_mask[:, :max_seq_len]
        
        return all_thought_outputs, current_embeds, current_mask


    def train_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ):
        """
        Training forward pass with continuous thought generation and CoT alignment.
        """
        self.train()

        # Keep original labels if none provided
        if labels is None:
            labels = input_ids.clone()
            batch_size = labels.shape[0]
            eot_id = self.tokenizer.convert_tokens_to_ids(self.model_config["eot_id"])

            for i in range(batch_size):
                # Find the positions of <eot> in the input_ids
                eot_pos = (input_ids[i] == eot_id).nonzero(as_tuple=True)

                if len(eot_pos[0]) > 0:
                    # Get the last occurrence of <eot>
                    last_eot_pos = eot_pos[0][-1].item()
                    
                    # Mask everything before and including the last <eot>
                    labels[i, :last_eot_pos] = -100

                # Mask padding
                labels[i, attention_mask[i] == 0] = -100

        # Get input embeddings if not provided
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)


        # Generate continuous thoughts
        if self.current_stage > 0:
            num_thoughts = self.current_stage * self.model_config["continuous_thoughts"]
            all_thoughts, final_embeds, final_mask = self.thoughts_forward(
                num_thoughts=num_thoughts,
                thought_ids=input_ids,
                thought_mask=attention_mask,
                num_of_thought_tokens = self.model_config["training_thoughts_sequence_length"]
            )

            # Add auxiliary losses
            auxiliary_losses = []

            # Thought coherence loss
            if len(all_thoughts) > 1:
                coherence_loss = 0
                for t1, t2 in zip(all_thoughts[:-1], all_thoughts[1:]):
                    sim = F.cosine_similarity(t1, t2, dim=-1)
                    coherence_loss += (1 - sim).mean()
                auxiliary_losses.append(coherence_loss * self.coherence_weight)

            batch_size = labels.shape[0]

            for i in range(batch_size):
                # Find the start and end of CoT in the labels
                cot_start = None
                
                for j, token_id in enumerate(labels[i]):
                    if token_id == self.tokenizer.convert_tokens_to_ids(self.model_config["eot_id"]):
                        cot_start = j + 1  # Start of CoT


                # Debugging: Print CoT tokens and latent thoughts
                if cot_start is not None:
                    # Extract CoT tokens
                    cot_tokens = labels[i, cot_start:]  # [cot_seq_len]

                    # Get the latent thoughts for this batch
                    latent_thoughts = all_thoughts[i]  # [thought_seq_len, hidden_size]

                    # Project latent thoughts to logits
                    thought_logits = self.lm_head(latent_thoughts)  # [thought_seq_len, vocab_size]
                    thought_token_ids = torch.argmax(thought_logits, dim=-1)  # [thought_seq_len]


                    # Debugging: Print CoT tokens and latent thoughts
                    if self.debug:
                        # Decode CoT tokens
                        cot_tokens_list = cot_tokens.squeeze().tolist()  # Convert to 1D list
                        if isinstance(cot_tokens_list, int):  # Handle single token case
                            cot_tokens_list = [cot_tokens_list]
                        cot_text = self.tokenizer.decode(cot_tokens_list, skip_special_tokens=True)
                        print(f" ==================== \n Debug: CoT for batch {i}: {cot_text} \n ====================")

                        # Decode latent thoughts
                        thought_token_ids_list = thought_token_ids.squeeze().tolist()  # Convert to list

                        # Ensure thought_token_ids_list is a flat list
                        if isinstance(thought_token_ids_list, list) and all(isinstance(item, list) for item in thought_token_ids_list):
                            # Flatten the nested list
                            thought_token_ids_list = [token for sublist in thought_token_ids_list for token in sublist]
                        elif isinstance(thought_token_ids_list, int):  # Handle single token case
                            thought_token_ids_list = [thought_token_ids_list]

                        # Decode the flat list of token IDs
                        thought_text = self.tokenizer.decode(thought_token_ids_list, skip_special_tokens=False)
                        print(f"==================== \n Debug: Latent thoughts for batch {i}: {thought_text} \n ========================")


            # Forward pass with thoughts
            outputs = super().forward(
                inputs_embeds=final_embeds,
                attention_mask=final_mask,
                labels=labels,
                **kwargs
            )

            # Add auxiliary losses
            if auxiliary_losses:
                outputs.loss += sum(auxiliary_losses)

        else:

            if inputs_embeds is None:
                # Standard forward pass for initial stage
                outputs = super().forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )
            else:

                outputs = super().forward(
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )

        return outputs

    
    def infer_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[DynamicCache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ):
        """
        Inference forward pass with continuous thought generation.
        """

        batch_size = input_ids.shape[0]

        # Insert <bot> token to initiate latent reasoning
        if input_ids.shape[1] > 1:
            input_ids = torch.cat(
                [
                    input_ids,
                    torch.tensor(
                        [[self.tokenizer.convert_tokens_to_ids(self.model_config["bot_id"])]] * batch_size,
                        device=input_ids.device,
                    ),
                ],
                dim=1,
            )
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((batch_size, 1), device=attention_mask.device),
                ],
                dim=1,
            )

        # Generate continuous thoughts
        if self.model_config["stages"] - 1 > 0 and input_ids.shape[1] > 1:
            num_thoughts = (self.model_config["stages"] - 1) * self.model_config["continuous_thoughts"]
            all_thoughts, final_embeds, final_mask = self.thoughts_forward(
                num_thoughts, input_ids, attention_mask
            )

            # Add <eot> token to mark the end of latent reasoning
            eot_embeds = self.get_input_embeddings()(
                torch.tensor(
                    [[self.tokenizer.convert_tokens_to_ids(self.model_config["eot_id"])]] * batch_size,
                    device=final_embeds.device,
                )
            )
            final_embeds = torch.cat([final_embeds, eot_embeds], dim=1)
            final_mask = torch.cat([final_mask, torch.ones((batch_size, 1), device=final_mask.device)], dim=1)

            # Generate final output in language mode
            outputs = super().forward(
                inputs_embeds=final_embeds,
                attention_mask=final_mask,
                past_key_values=None,  # Reset past_key_values for answer generation
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )
        else:
            # Standard forward pass (no latent thoughts)
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        return outputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[DynamicCache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ):
        """Main forward function that routes to either training or inference."""
        forward_fn = self.train_forward if self.training else self.infer_forward
        return forward_fn(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )


from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
tokenizer.pad_token = tokenizer.eos_token

# Add special tokens
special_tokens = {
    "additional_special_tokens": [config["bot_id"], config["eot_id"], config["answer_id"]]
}
num_added_tokens = tokenizer.add_special_tokens(special_tokens)

# Load the Reasoning model configuration
model_config = AutoConfig.from_pretrained(config["model_name"])
latent_config = LatentReasoningGemmaForCausalLM.DEFAULT_CONFIG
LatentReasoningGemmaForCausalLM.DEFAULT_CONFIG = {
    **latent_config,
    **config
}
updated_latent_config = LatentReasoningGemmaForCausalLM.DEFAULT_CONFIG
model = LatentReasoningGemmaForCausalLM(config=model_config)

# Load the Reasoning model
model = model.from_pretrained(
    config["model_name"],
    torch_dtype=torch.bfloat16,
)
model.tokenizer = tokenizer
model.resize_token_embeddings(len(tokenizer))


# Load the normal model for comparison
model_without_reasoning = AutoModelForCausalLM.from_pretrained(config["model_name"])
model_without_reasoning.resize_token_embeddings(len(tokenizer))
model_without_reasoning = model_without_reasoning.cuda()


from typing import Tuple
from transformers import TextStreamer
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)

def generate_answer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    question: str,
    max_length: int = 128,
    k: int = config["cot_decoding_k"],
    temperature: float = 1.0,
    **generation_kwargs
) -> str:
    """
    Generates answer using CoT decoding and returns the best path.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        question: Input question
        max_length: Maximum sequence length
        k: Number of alternative paths to consider
        temperature: Sampling temperature
        **generation_kwargs: Additional generation arguments
        
    Returns:
        Best decoded sequence with highest confidence
    """
    # Initialize streamer
    streamer = TextStreamer(tokenizer, skip_prompt=False, skip_special_tokens=False)
    
    # Tokenize input
    inputs = tokenizer(question, max_length=max_length, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Get initial logits for CoT paths
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True
    )
    
    first_token_logits = outputs.logits[:, -1, :] / temperature
    
    # Get top-k tokens
    probs = F.softmax(first_token_logits, dim=-1)
    top_k_probs, top_k_tokens = torch.topk(probs, k, dim=-1)
    
    best_path = None
    best_confidence = -float('inf')
    
    # Generate continuation for each top-k token
    for i in range(k):
        # Prepare input with current top-k token
        curr_input_ids = torch.cat([
            input_ids,
            top_k_tokens[:, i:i+1]
        ], dim=1)
        
        curr_attention_mask = torch.cat([
            attention_mask,
            torch.ones((attention_mask.shape[0], 1), device=model.device)
        ], dim=1)
        
        # Generate with streamer for best path
        outputs = model.generate(
            input_ids=curr_input_ids,
            attention_mask=curr_attention_mask,
            max_length=max_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
            streamer=streamer if i == 0 else None,  # Only stream first path
            **generation_kwargs
        )
        
        # Calculate confidence for this path
        _, confidence = calculate_answer_confidence(
            outputs.sequences[0].tolist(),
            outputs.scores[-1],
            tokenizer
        )
        
        # Update best path if confidence is higher
        if confidence > best_confidence:
            best_confidence = confidence
            best_path = outputs.sequences[0]
            
    # Return the path with highest confidence
    return tokenizer.decode(best_path, skip_special_tokens=True)

def calculate_answer_confidence(
    sequence: List[int],
    final_logits: torch.Tensor,
    tokenizer: PreTrainedTokenizer
) -> Tuple[str, float]:
    """Calculate confidence score using min-margin approach."""
    # Extract answer from sequence
    answer = extract_answer(sequence, tokenizer)
    
    if not answer:
        return "", 0.0
    
    # Get probabilities
    probs = F.softmax(final_logits, dim=-1)
    
    # Calculate margins for answer tokens
    answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
    margins = []
    
    for token in answer_tokens:
        token_prob = probs[0, token].item()
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        second_best_prob = sorted_probs[0, 1].item()
        margin = token_prob - second_best_prob
        margins.append(margin)
        
    confidence = sum(margins) / len(margins)
    return answer, confidence

def extract_answer(sequence: List[int], tokenizer: PreTrainedTokenizer) -> str:
    """
    Extract final answer from sequence using <eot> token.
    Finds the answer between the last occurrence of <eot> and the end of sequence.
    """
    # Convert sequence to string
    decoded = tokenizer.decode(sequence)
    
    # Find last <eot> position
    eot_position = decoded.rfind(config["eot_id"])
    
    if eot_position != -1:
        # Extract everything after the last <eot>
        answer = decoded[eot_position + len(config["eot_id"]):].strip()
        return answer
        
    return decoded


import time

tick_start = 0

def tick():
    global tick_start
    tick_start = time.time()

def tock():
    print(f"TOTAL TIME ELAPSED: {time.time() - tick_start:.2f}s")


def text_gen(prompt, model, tokenizer):
    tick()
    input = f"{prompt}"
    print(f"Question: {prompt} \n ==========================================")
    output = generate_answer(model=model, tokenizer=tokenizer, question=input, k=5, max_length=config["max_length"] )
    print(f"Outputs: ========================")
    print(output)
    tock()
    print(f"\n\n\n\n")



# Test the function
text_gen("格闘家ボブ・サップの出身国はどこでしょう？", model=model_without_reasoning, tokenizer=tokenizer)
text_gen("人気漫画『ドラえもん』の登場人物で、ジャイアンの苗字は剛田ですが、スネ夫の苗字は何でしょう？",  model=model_without_reasoning, tokenizer=tokenizer)
text_gen("Translate 'Hello, how are you?' to Japanese.",  model=model_without_reasoning, tokenizer=tokenizer)
text_gen("「お元気ですか」を英語に訳すと",  model=model_without_reasoning, tokenizer=tokenizer)
text_gen("Translate to english `「ねえ、それは何のためにあるの？`", model=model_without_reasoning, tokenizer=tokenizer)


from transformers import (
    Trainer,
    TrainingArguments
) 
import wandb
import os
import torch
import evaluate
import numpy as np

# Initialize WandB
wandb.init(project=config["wandb_project"], config=config)

# Set up training arguments
training_args = TrainingArguments(
    output_dir=config["output_dir"],
    per_device_train_batch_size=config["per_device_train_batch_size"],
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    learning_rate=config["learning_rate"],
    warmup_ratio=config["warmup_steps"],
    logging_steps=config["logging_steps"],
    save_steps=config["save_steps"],
    bf16=config["bf16"],
    bf16_full_eval=config["bf16_full_eval"],
    optim=config["optim"],
    report_to="wandb",
    remove_unused_columns=False,
    dataloader_pin_memory=True,
    # gradient_checkpointing=True,
)

# Move model to GPU and wrap with DataParallel if multiple GPUs available
if torch.cuda.is_available():
    # Check if model is not already on CUDA
    if not next(model.parameters()).is_cuda:
        model = model.cuda()
    if torch.cuda.device_count() > 1:
        # Check if model isn't already wrapped with DataParallel
        if not isinstance(model, torch.nn.DataParallel):
            # Use DataParallel with explicit device IDs
            model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

def stage_trainer(stage=0):

    if isinstance(model, torch.nn.DataParallel):
        model.module.current_stage = stage
    else:
        model.current_stage = stage

    current_output_dir = f"{config['output_dir']}_stage{stage}"
    training_args.output_dir = current_output_dir
    training_args.num_train_epochs = 3
        

    # Load the Reasoning model configuration
    dataset_ = dataset.map(
        (lambda x: preprocess_function(
            x, 
            detector=detector,
            stages=stage, 
            eos_token=tokenizer.eos_token,
            bos_token=tokenizer.bos_token,
            language_config=language_config
        )),
        batched=True,
        batch_size=config["batch_size"]
    )

    # Tokenize the dataset
    dataset_ = dataset_.map(
        (lambda x: tokenizer_function(
            x, 
            tokenizer=tokenizer,
        )),
        batched=True,
        batch_size=config["batch_size"],
        remove_columns=["input", "instruction", "output", "prompt"]
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_["train"]
    )
    
    trainer.train()

    # Save checkpoints
    for folder in os.listdir(current_output_dir):
        if folder.startswith("checkpoint-"):
            checkpoint_folder = os.path.join(current_output_dir, folder)
            if os.path.isdir(checkpoint_folder):
                tokenizer.save_pretrained(checkpoint_folder)
                # If using DataParallel, save the base model
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(checkpoint_folder)

# Run training stages
for stage in range(config["stages"] + 1):
    stage_trainer(stage)


from transformers import AutoTokenizer, AutoConfig
import torch
torch.cuda.empty_cache()


def load_model(model_name = "output_stage1/checkpoint-10000"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_config = AutoConfig.from_pretrained(model_name)
    model = LatentReasoningGemmaForCausalLM(config=model_config)
    model = model.from_pretrained(model_name)
    model.tokenizer = tokenizer

    model = model.cuda()

    return  model, tokenizer


# Make sure to load the model from your specified path. In our case our path is "output_stage1/checkpoint-10000"
model, tokenizer = load_model(model_name= "output_stage1/checkpoint-10000")


text_gen("格闘家ボブ・サップの出身国はどこでしょう？", model=model, tokenizer=tokenizer)
text_gen("人気漫画『ドラえもん』の登場人物で、ジャイアンの苗字は剛田ですが、スネ夫の苗字は何でしょう？", model=model, tokenizer=tokenizer)
text_gen("「お元気ですか」を英語に訳すと ", model=model, tokenizer=tokenizer)
text_gen("Translate to english `「ねえ、それは何のためにあるの？`", model=model, tokenizer=tokenizer)
text_gen("「abc ～the first～」へようこそ！さて、ABC・・・と始まるアルファベットは、全部で何文字でしょう？`", model=model, tokenizer=tokenizer)


import nltk

try:
    nltk.data.find('tokenizers/punkt')
    nltk.download('punkt_tab')
except LookupError:
    nltk.download('punkt')



import torch


def preprocess_eval_dataset_function(
    examples, 
):
    """
    Preprocess the input examples by constructing the prompt with reasoning steps.

    Args:
        examples (dict): A dictionary containing the input examples with keys "instruction", "input", and "output".
    Returns:
        dict: A dictionary containing the preprocessed prompts.
    """

    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]

    new_inputs = []
    
    for i in range(len(instructions)):
        instruction = instructions[i]
        input = inputs[i]

        input = instruction + input
        new_inputs.append(input)


    return {"input": new_inputs, "output": outputs, "instructions": instructions}



# Preprocess eval dataset
eval_dataset_ = eval_dataset.map(
    preprocess_eval_dataset_function,
    batched=True,
    batch_size=config["batch_size"]
)


import torch
from typing import Dict, List, Union
from transformers import PreTrainedTokenizer, PreTrainedModel
from nltk.translate.bleu_score import sentence_bleu
from thefuzz import fuzz
from bert_score import score as bert_score
from nltk.tokenize import word_tokenize
import nltk
import tqdm
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class EvaluationMetrics:
    accuracy: float
    avg_fuzzy_score: float
    avg_bleu_score: float
    avg_bert_score_f1: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'accuracy': self.accuracy,
            'avg_fuzzy_score': self.avg_fuzzy_score,
            'avg_bleu_score': self.avg_bleu_score,
            'avg_bert_score_f1': self.avg_bert_score_f1
        }


def extract_answer_from_predicted_answer(text: str) -> str:
    """
    Extract the text after '答え：' or 'Answer:' from the input text.
    
    Args:
        text (str): The input text containing the answer.
    
    Returns:
        str: The extracted answer, or an empty string if no match is found.
    """
    prefixes = ["答え：", "Answer:"]
    
    for prefix in prefixes:
        if prefix in text:
            return text.split(prefix, 1)[1].strip()
    
    return text.strip()  # Return stripped text if no prefix found



# Detect if the text contains Japanese characters
def contains_japanese(text):
    # Hiragana (3040-309F), Katakana (30A0-30FF), Kanji (4E00-9FFF)
    for char in text:
        if ('\u3040' <= char <= '\u309F' or  # Hiragana
            '\u30A0' <= char <= '\u30FF' or  # Katakana
            '\u4E00' <= char <= '\u9FFF'):   # Kanji
            return True
    return False


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text based on language (Japanese or English).
    For Japanese, splits on spaces and punctuation while preserving important characters.
    For English, uses basic word tokenization.
    """

    if contains_japanese(text):
        # Simple Japanese tokenization: split on spaces and basic punctuation
        # while preserving Japanese punctuation
        import re
        # Split on spaces and common punctuation, but preserve Japanese punctuation
        tokens = re.findall(r'[^\s\.,!?]+|[。、！？]', text)
        return [token for token in tokens if token.strip()]
    else:
        # For English, use simple whitespace and punctuation splitting
        import re
        return re.findall(r'\w+|[^\w\s]', text.lower())


def compute_metrics(pred_answer: str, target_answer: str, threshold: int = 80) -> Dict[str, Union[float, bool]]:
    """
    Compute multiple evaluation metrics for comparing predicted and target answers.
    """
    # Preprocess answers
    pred_clean = extract_answer_from_predicted_answer(pred_answer)
    target_clean = target_answer.strip()
    
    # Convert to lowercase for consistent comparison
    pred_lower = pred_clean.lower()
    target_lower = target_clean.lower()
    
    # Calculate fuzzy match score
    fuzzy_score = fuzz.ratio(pred_lower, target_lower)
    
    # Tokenize for BLEU score
    pred_tokens = word_tokenize(pred_lower)
    target_tokens = word_tokenize(target_lower)
    
    # Calculate BLEU score
    try:
        bleu = sentence_bleu([target_tokens], pred_tokens, weights=(1.0,))
    except ZeroDivisionError:
        bleu = 0.0

    
    
    # Set language based on content
    lang = 'ja' if contains_japanese(target_clean) else 'en'
    
    # Calculate BERTScore with appropriate language model
    P, R, F1 = bert_score([pred_clean], [target_clean], lang=lang, verbose=False)
    bert_f1 = F1.item()
    
    return {
        'fuzzy_match': fuzzy_score >= threshold,
        'fuzzy_score': fuzzy_score,
        'bleu_score': bleu,
        'bert_score_f1': bert_f1
    }



@torch.no_grad()
def evaluate(
    dataloader,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    max_new_tokens: int,
    threshold: int = 80,
) -> EvaluationMetrics:
    """
    Evaluate the model using multiple metrics.
    
    Returns:
        EvaluationMetrics: Object containing all computed metrics
    """
    total_instances = 0
    total_correct = 0
    
    # Initialize metric aggregators
    total_metrics = {
        'fuzzy_score': 0,
        'bleu_score': 0,
        'bert_score_f1': 0
    }

    for batch in tqdm.tqdm(dataloader):
        inputs = batch["input"]
        outputs = batch["output"]
        batch_size = len(inputs)
        total_instances += batch_size

        for i in range(batch_size):
            input_text = inputs[i]
            target_answer = outputs[i]

            # Generate the answer
            pred_answer = generate_answer(
                model=model,
                tokenizer=tokenizer,
                question=input_text,
                max_length=max_new_tokens,
            )

            # Compute all metrics
            metrics = compute_metrics(pred_answer, target_answer, threshold)
            
            # Update counters
            if metrics['fuzzy_match']:
                total_correct += 1
            
            # Aggregate metrics
            for key in total_metrics:
                total_metrics[key] += metrics[key]

            if config["debug"]:
                pred_answer_extracted = extract_answer_from_predicted_answer(pred_answer)
                print(
                    f"Input: {input_text}\n"
                    f"Target: {target_answer}\n"
                    f"Predicted: {pred_answer_extracted}\n"
                    f"Metrics: {metrics}\n"
                )

    # Calculate averages
    accuracy = total_correct / total_instances
    for key in total_metrics:
        total_metrics[key] /= total_instances

    return EvaluationMetrics(
        accuracy=accuracy,
        avg_fuzzy_score=total_metrics['fuzzy_score'],
        avg_bleu_score=total_metrics['bleu_score'],
        avg_bert_score_f1=total_metrics['bert_score_f1']
    )


from torch.utils.data import DataLoader

# Load data for evaluation
dataloader = DataLoader(eval_dataset_["train"], batch_size=config["batch_size"], shuffle=False)

def test_evaluation(model, tokenizer):
    metrics = evaluate(dataloader, tokenizer, model, config["max_length"])
    print(f"Metrics: {metrics}")


# Evaluating every model stage

for i in range(config["stages"] + 1):
    model_name = f"output_stage{i}/checkpoint-10000"
    model, tokenizer = load_model(model_name = model_name)
    test_evaluation(model, tokenizer=tokenizer)
    print(f"Model : {model_name}")



pip install -q git+https://github.com/vicksEmmanuel/latent-gemma.git


from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from latent_gemma import LatentReasoningGemmaForCausalLM

model_path = "victorumesiobi/gemma-2-japanese-english-reasoning/transformers/1" # Replace with the path to which your model was downloaded too

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model_config = AutoConfig.from_pretrained(model_path)

config = {
    "max_length": 256
}
latent_config = LatentReasoningGemmaForCausalLM.DEFAULT_CONFIG
LatentReasoningGemmaForCausalLM.DEFAULT_CONFIG = {
    **latent_config,
    **config
}
updated_latent_config = LatentReasoningGemmaForCausalLM.DEFAULT_CONFIG
model = LatentReasoningGemmaForCausalLM(config=model_config)
model = model.from_pretrained(model_path)
model.tokenizer = tokenizer


text = "人気漫画『ドラえもん』の登場人物で、ジャイアンの苗字は剛田ですが、スネ夫の苗字は何でしょう？"
output = model.generate_answer(
    model=model, 
    tokenizer=tokenizer, 
    question=text, 
    k=5, 
    max_length=256
)

print(f"output: {output}")


input_ids = tokenizer(text, return_tensors="pt")

outputs = model.generate(**input_ids, max_new_tokens=32)
print(tokenizer.decode(outputs[0]))

