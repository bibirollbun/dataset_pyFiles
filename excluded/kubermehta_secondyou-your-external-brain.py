%%capture
# Install Unsloth for fast LoRA fine-tuning
!pip install unsloth

# Upgrade transformers and related packages
!pip install transformers -U
!pip install --no-deps --upgrade timm


# Environment setup for optimal performance
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '2'  # Faster HF downloads
os.environ['PYTHONIOENCODING'] = 'utf-8'       # Text encoding consistency
os.environ['PYTHONUTF8'] = '1'                 # Enable UTF-8 mode for Python
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"       # Single GPU setup

# Core imports
import torch
import json
import pandas as pd
import gc
import random
import numpy as np
from IPython.display import Markdown, display, clear_output

# Configure torch dynamo for faster inference
torch._dynamo.config.cache_size_limit = 256

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)} - {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")


import torch
from unsloth import FastLanguageModel

# Load Gemma 3N with 4-bit quantization for memory efficiency
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-3n-E4B-it",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

print(f"Model loaded: {model.config.model_type}")
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Max sequence length: {model.config.max_position_embeddings}")


# Configure LoRA for efficient fine-tuning with adaptive parameters
def get_optimal_lora_config(dataset_size):
    """Get optimal LoRA configuration based on dataset size."""
    if dataset_size < 20:
        return {"r": 4, "alpha": 8}  # Conservative for small datasets
    elif dataset_size < 100:
        return {"r": 8, "alpha": 16}  # Balanced approach
    else:
        return {"r": 16, "alpha": 32}  # More capacity for larger datasets

# Get optimal configuration
lora_config = get_optimal_lora_config(len(training_data))
print(f"  LoRA Configuration for {len(training_data)} training examples:")
print(f"   Rank (r): {lora_config['r']}")
print(f"   Alpha: {lora_config['alpha']}")

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_config['r'],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=lora_config['alpha'],
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print("  LoRA configuration applied successfully!")
print(f"  Model Statistics:")
print(f"   Trainable parameters: {trainable_params:,}")
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable ratio: {100 * trainable_params / total_params:.2f}%")

print(f"\n  LoRA Explanation:")
print(f"     Rank (r={lora_config['r']}): Controls adaptation capacity")
print(f"     Alpha ({lora_config['alpha']}): Scales the adaptation strength")
print(f"     Target modules: Key transformer components for personalization")
print(f"     Memory efficient: Only {trainable_params:,} parameters to train!")

if lora_config['r'] < 8:
    print(f"\n  Note: Using conservative LoRA rank due to small dataset")
    print(f"   Consider adding more training data for better personalization")


# Set up the chat template for Gemma 3N
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma-3",  # Same template as Gemma 3
)

print("  Chat template configured for Gemma 3N!")
print("  This ensures proper conversation formatting for training")
print("  Gemma 3N uses the same chat format as Gemma 3")


import jso 
import pandas as pd
import re
import os
from datetime import datetime

# Upload your ChatGPT data export (conversations.json)
# For Kaggle: Upload file using the file upload feature
# For Colab: Use files.upload() or mount Google Drive

def preprocess_message(content):
    """Clean and preprocess message content."""
    if not content or len(content.strip()) == 0:
        return None
    
    # Remove URLs
    content = re.sub(r'http\S+|www\S+', '', content)
    
    # Remove excessive whitespace
    content = re.sub(r'\s+', ' ', content).strip()
    
    # Skip very short messages (less than 10 words)
    if len(content.split()) < 10:
        return None
    
    return content

def load_chatgpt_data(file_path):
    """Load and parse ChatGPT conversation data with improved error handling."""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"  Error: File '{file_path}' not found.")
            print("  Please upload your ChatGPT export file (conversations.json)")
            print("  Instructions: Go to ChatGPT â†’ Settings â†’ Data Export â†’ Download")
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        conversations = []
        total_messages = 0
        filtered_messages = 0
        
        for conv in data:
            if 'mapping' in conv:
                messages = []
                for msg_id, msg_data in conv['mapping'].items():
                    if msg_data.get('message') and msg_data['message'].get('content'):
                        content = msg_data['message']['content']
                        if content.get('parts') and len(content['parts']) > 0:
                            total_messages += 1
                            
                            # Preprocess the message
                            processed_content = preprocess_message(content['parts'][0])
                            if processed_content:
                                messages.append({
                                    'role': msg_data['message']['author']['role'],
                                    'content': processed_content,
                                    'timestamp': msg_data['message'].get('create_time')
                                })
                            else:
                                filtered_messages += 1
                
                # Only include conversations with substantial content
                if len(messages) > 2:  # At least 2 message exchanges
                    conversations.append({
                        'title': conv.get('title', 'Untitled'),
                        'messages': messages
                    })
        
        print(f"  Loaded {len(conversations)} conversations")
        print(f"  Total messages processed: {total_messages}")
        print(f"  Messages filtered out: {filtered_messages}")
        
        # Warn about small dataset
        if len(conversations) < 50:
            print(f"  WARNING: Only {len(conversations)} conversations found.")
            print("  For effective fine-tuning, consider:")
            print("   - Uploading more conversation history")
            print("   - Adding other text sources (.md, .txt files)")
            print("   - Minimum 100+ conversations recommended")
        
        return conversations
        
    except FileNotFoundError:
        print(f"  Error: File '{file_path}' not found.")
        print("  Please upload your ChatGPT export file")
        return []
    except json.JSONDecodeError as e:
        print(f"  Error: '{file_path}' is not a valid JSON file.")
        print(f"  JSON Error: {e}")
        print("  Ensure it's a proper ChatGPT export file")
        return []
    except Exception as e:
        print(f"  Unexpected error loading data: {e}")
        print("  Please check your file format and try again")
        return []

def load_additional_text_sources(file_paths):
    """Load additional text sources (.md, .txt, .json files)."""
    additional_data = []
    
    for file_path in file_paths:
        try:
            if not os.path.exists(file_path):
                print(f"  Skipping {file_path} - file not found")
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    # Handle JSON files (could be other chat exports)
                    data = json.load(f)
                    # Add custom parsing logic here based on format
                else:
                    # Handle text/markdown files
                    content = f.read()
                    
                    # Split into conversation-like chunks
                    # This is a simple approach - can be enhanced
                    chunks = content.split('\n\n')
                    for chunk in chunks:
                        cleaned_chunk = preprocess_message(chunk)
                        if cleaned_chunk:
                            additional_data.append({
                                'title': f'Text from {os.path.basename(file_path)}',
                                'messages': [
                                    {'role': 'user', 'content': 'Please share your thoughts on this topic.'},
                                    {'role': 'assistant', 'content': cleaned_chunk}
                                ]
                            })
                            
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")
    
    print(f"  Loaded {len(additional_data)} additional text sources")
    return additional_data

# Dynamic file path input for different environments
def get_data_file_path():
    """Get the data file path dynamically based on environment."""
    
    # For Kaggle, check if file is in input directory
    kaggle_path = "/kaggle/input/conversations.json"
    if os.path.exists(kaggle_path):
        return kaggle_path
    
    # Check current directory
    local_path = "conversations.json"
    if os.path.exists(local_path):
        return local_path
    
    # Interactive input for other cases
    file_path = input("  Enter path to your ChatGPT conversations.json file: ").strip()
    return file_path

# Example: Load your data
# Uncomment these lines to load your actual data:
# file_path = get_data_file_path()
# conversations = load_chatgpt_data(file_path)

# For demonstration, we'll create enhanced sample data
sample_conversations = [
    {
        'title': 'Learning Python Programming',
        'messages': [
            {'role': 'user', 'content': 'How do I learn Python effectively? I want to become proficient quickly but thoroughly.'},
            {'role': 'assistant', 'content': 'Start with the basics: variables, data types, and control structures. Practice daily with small projects like calculators, to-do apps, or data analysis scripts. Use resources like Python.org tutorial, automate boring stuff, and build real projects that interest you.'}
        ]
    },
    {
        'title': 'Career Development Strategy',
        'messages': [
            {'role': 'user', 'content': 'What advice would you give someone starting their tech career? I feel overwhelmed by all the options.'},
            {'role': 'assistant', 'content': 'Focus on fundamentals first - pick one programming language and get really good at it. Build a portfolio of projects that demonstrate real problem-solving skills. Network genuinely with people in the field, contribute to open source, and don\'t try to learn everything at once. Depth over breadth initially.'}
        ]
    },
    {
        'title': 'Project Management Philosophy',
        'messages': [
            {'role': 'user', 'content': 'I\'m struggling with managing multiple projects at once. How do you stay organized and productive?'},
            {'role': 'assistant', 'content': 'I use a combination of time-blocking and priority matrices. Break large projects into smaller, actionable tasks. Set clear boundaries between projects and use tools like Notion or Obsidian for knowledge management. Most importantly, say no to projects that don\'t align with your core goals.'}
        ]
    }
]

print("  Enhanced sample data created for demonstration")
print(f"  Sample dataset contains {len(sample_conversations)} conversations")
print("  Replace with your actual data for better personalization")


def format_conversation_for_training(conversations, tokenizer):
    """Format conversations for training with Gemma chat template."""
    formatted_data = []
    
    for conv in conversations:
        messages = conv['messages']
        
        # Process conversation in pairs (user -> assistant)
        for i in range(0, len(messages) - 1, 2):
            if (i + 1 < len(messages) and 
                messages[i]['role'] == 'user' and 
                messages[i + 1]['role'] == 'assistant'):
                
                conversation = [
                    {"role": "user", "content": messages[i]['content']},
                    {"role": "assistant", "content": messages[i + 1]['content']}
                ]
                
                # Apply chat template
                formatted_text = tokenizer.apply_chat_template(
                    conversation, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                
                formatted_data.append({"text": formatted_text})
    
    return formatted_data

# Format the data
# training_data = format_conversation_for_training(conversations, tokenizer)

# For demonstration with sample data
training_data = format_conversation_for_training(sample_conversations, tokenizer)
print(f"Formatted {len(training_data)} training examples")
print("Sample formatted text:")
print(training_data[0]['text'][:200] + "..." if len(training_data) > 0 else "No data")


from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Validate dataset size before training
if len(training_data) < 5:
    print("  WARNING: Very small dataset detected!")
    print(f"  Current dataset size: {len(training_data)} examples")
    print("  Recommendations:")
    print("   - Add more conversation data for better results")
    print("   - Consider using a larger sample dataset")
    print("   - Minimum 50+ examples recommended for meaningful fine-tuning")
    print()
    
    # Add more sample data if dataset is too small
    if len(training_data) < 10:
        print("ğŸ”§ Adding additional sample conversations for demonstration...")
        additional_samples = [
            {"text": tokenizer.apply_chat_template([
                {"role": "user", "content": "What's your approach to learning new technologies?"},
                {"role": "assistant", "content": "I believe in hands-on learning combined with solid theoretical foundations. Start with official documentation, build small projects, and gradually increase complexity. Don't just follow tutorials - try to break things and understand why they work."}
            ], tokenize=False, add_generation_prompt=False)},
            {"text": tokenizer.apply_chat_template([
                {"role": "user", "content": "How do you handle work-life balance?"},
                {"role": "assistant", "content": "I set clear boundaries between work and personal time. Use time-blocking techniques, turn off work notifications after hours, and make sure to pursue hobbies that recharge me. Quality over quantity - focused work time is more valuable than long hours."}
            ], tokenize=False, add_generation_prompt=False)},
            {"text": tokenizer.apply_chat_template([
                {"role": "user", "content": "What programming patterns do you find most useful?"},
                {"role": "assistant", "content": "I'm a big fan of functional programming principles like immutability and pure functions. Design patterns like dependency injection and observer pattern are incredibly useful. Always prioritize readability and maintainability over clever code."}
            ], tokenize=False, add_generation_prompt=False)}
        ]
        training_data.extend(additional_samples)
        print(f"  Dataset expanded to {len(training_data)} examples")

# Create train/validation split
if len(training_data) >= 4:
    train_data, val_data = train_test_split(
        training_data, 
        test_size=0.2, 
        random_state=42,
        shuffle=True
    )
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    print(f"  Dataset split:")
    print(f"   Training: {len(train_data)} examples")
    print(f"   Validation: {len(val_data)} examples")
else:
    # If dataset is too small for split, use all for training
    train_dataset = Dataset.from_list(training_data)
    val_dataset = None
    print(f"  Using all {len(training_data)} examples for training (no validation split)")

# Enhanced training configuration with validation
training_args = TrainingArguments(
    output_dir="./secondyou-gemma3n",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=min(10, len(train_dataset) // 4),  # Adaptive warmup
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=max(1, len(train_dataset) // 10),  # Adaptive logging
    save_strategy="epoch",
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=42,
    # Add evaluation if we have validation data
    eval_strategy="epoch" if val_dataset else "no",
    per_device_eval_batch_size=1 if val_dataset else None,
    save_total_limit=2,  # Keep only 2 checkpoints to save space
    load_best_model_at_end=True if val_dataset else False,
    metric_for_best_model="eval_loss" if val_dataset else None,
    greater_is_better=False if val_dataset else None,
)

# Initialize trainer with improved configuration
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset if val_dataset else None,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    packing=False,
    args=training_args,
)

print("  Training configuration complete!")
print(f"ğŸ”§ Configuration details:")
print(f"   Dataset size: {len(train_dataset)}")
print(f"   Validation: {'Yes' if val_dataset else 'No'}")
print(f"   Batch size: {training_args.per_device_train_batch_size}")
print(f"   Learning rate: {training_args.learning_rate}")
print(f"   Epochs: {training_args.num_train_epochs}")
print(f"   Warmup steps: {training_args.warmup_steps}")

# LoRA parameter explanation
print(f"\n  LoRA Configuration Explanation:")
print(f"   Rank (r=8): Low rank for memory efficiency")
print(f"   Alpha (16): Controls adaptation strength")
print(f"   Target modules: Key attention and MLP layers")
print(f"     For larger datasets, consider r=16 or r=32")


import time
import torch

print("  Starting SecondYou training...")
print(f"  Training details:")
print(f"   Model: Gemma 3N with LoRA (r={lora_config['r']}, Î±={lora_config['alpha']})")
print(f"   Dataset: {len(train_dataset)} training examples")
if val_dataset:
    print(f"   Validation: {len(val_dataset)} examples")
print(f"   Epochs: {training_args.num_train_epochs}")
print(f"   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print()

start_time = time.time()

try:
    # Train the model with progress monitoring
    trainer_stats = trainer.train()
    
    end_time = time.time()
    training_duration = end_time - start_time
    
    print(f"  Training completed successfully!")
    print(f"  Training time: {training_duration:.2f} seconds ({training_duration/60:.1f} minutes)")
    print(f"  Final training loss: {trainer_stats.training_loss:.4f}")
    
    if val_dataset:
        # Get final evaluation metrics
        eval_results = trainer.evaluate()
        print(f"  Final validation loss: {eval_results.get('eval_loss', 'N/A'):.4f}")
    
    # Training quality assessment
    if trainer_stats.training_loss > 2.0:
        print(f"  Note: Training loss is relatively high ({trainer_stats.training_loss:.4f})")
        print(f"  Consider:")
        print(f"   - Training for more epochs")
        print(f"   - Adding more diverse training data")
        print(f"   - Adjusting learning rate")
    elif trainer_stats.training_loss < 0.5:
        print(f"  Note: Training loss is very low ({trainer_stats.training_loss:.4f})")
        print(f"  Possible overfitting - validate with test questions")
    else:
        print(f"  Training loss looks good ({trainer_stats.training_loss:.4f})")
    
    # Save the fine-tuned model
    print(f"\n  Saving your personalized SecondYou model...")
    model.save_pretrained("secondyou-gemma3n-final")
    tokenizer.save_pretrained("secondyou-gemma3n-final")
    print("  Model saved successfully!")
    
    # Save training metrics
    training_info = {
        'training_loss': float(trainer_stats.training_loss),
        'training_time_seconds': training_duration,
        'dataset_size': len(train_dataset),
        'lora_rank': lora_config['r'],
        'lora_alpha': lora_config['alpha'],
        'epochs': training_args.num_train_epochs,
        'learning_rate': training_args.learning_rate
    }
    
    if val_dataset:
        training_info['validation_loss'] = float(eval_results.get('eval_loss', 0))
        training_info['validation_size'] = len(val_dataset)
    
    import json
    with open('training_metrics.json', 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"  Training metrics saved to training_metrics.json")

except Exception as e:
    print(f"  Training failed with error: {e}")
    print(f"  Troubleshooting tips:")
    print(f"   - Check GPU memory (reduce batch size if needed)")
    print(f"   - Verify dataset format and size")
    print(f"   - Try reducing max_seq_length to 1024")
    raise e


def chat_with_secondyou(message, max_length=512, temperature=0.7, top_p=0.9):
    """Chat with your personalized AI with improved error handling."""
    
    try:
        if not message or len(message.strip()) == 0:
            return "Please provide a message to respond to."
        
        # Format the input message
        conversation = [{"role": "user", "content": message.strip()}]
        
        # Apply chat template with error handling
        try:
            input_text = tokenizer.apply_chat_template(
                conversation, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception as e:
            print(f"  Warning: Chat template error: {e}")
            # Fallback to simple format
            input_text = f"User: {message}\nAssistant: "
        
        # Tokenize with length check
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1500)
        
        # Move to correct device
        if torch.cuda.is_available():
            inputs = inputs.to(model.device)
        
        # Generate response with error handling
        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("  GPU out of memory. Trying with smaller parameters...")
                    # Retry with smaller parameters
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=min(max_length, 256),
                        do_sample=False,  # Use greedy decoding
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                else:
                    raise e
        
        # Decode and clean response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the new response
        if input_text in full_response:
            response = full_response[len(input_text):].strip()
        else:
            # Fallback extraction
            response = full_response.strip()
        
        # Clean up the response
        response = response.replace("Assistant:", "").strip()
        
        # Handle empty responses
        if not response:
            response = "I'm still learning to respond in your style. Could you try rephrasing your question?"
        
        return response
        
    except Exception as e:
        error_msg = f"Error generating response: {e}"
        print(f"  {error_msg}")
        return "Sorry, I encountered an error. Please try again with a different question."

print("  Enhanced chat function ready!")
print("  Features: Error handling, memory management, response cleaning")


def display_conversation(prompt, response):
    """Display a conversation in a formatted way."""
    from IPython.display import display, Markdown
    
    conversation_md = f"""
**You:** {prompt}

**SecondYou:** {response}

---
"""
    display(Markdown(conversation_md))

print("Display function ready!")


# Test your AI with various question types
test_questions = [
    {
        "category": "Technical",
        "question": "What's your favorite programming language and why?",
        "expected_style": "Should reflect personal preferences and reasoning"
    },
    {
        "category": "Learning",
        "question": "How do you approach learning new technologies?",
        "expected_style": "Should show learning methodology and philosophy"
    },
    {
        "category": "Career",
        "question": "What advice would you give to someone starting their career?",
        "expected_style": "Should reflect experience and personal insights"
    },
    {
        "category": "Problem-solving",
        "question": "Tell me about a challenging project you've worked on.",
        "expected_style": "Should demonstrate problem-solving approach"
    },
    {
        "category": "Personal productivity",
        "question": "How do you manage your time and stay productive?",
        "expected_style": "Should reflect personal productivity systems"
    }
]

print("ğŸ§ª Testing SecondYou AI across different categories:\n")

test_results = []

for i, test_case in enumerate(test_questions, 1):
    print(f"{'='*60}")
    print(f"Test {i}: {test_case['category']} Question")
    print(f"{'='*60}")
    print(f"  Question: {test_case['question']}")
    print(f"  Expected style: {test_case['expected_style']}")
    print()
    
    try:
        response = chat_with_secondyou(test_case['question'])
        print(f"  SecondYou Response:")
        print(f"   {response}")
        print()
        
        # Basic response quality checks
        response_length = len(response.split())
        if response_length < 10:
            print("  Response seems quite short - consider more training data")
        elif response_length > 100:
            print("  Detailed response - good engagement!")
        else:
            print("  Response length looks appropriate")
        
        test_results.append({
            'question': test_case['question'],
            'response': response,
            'category': test_case['category'],
            'response_length': response_length
        })
        
    except Exception as e:
        print(f"  Error testing question {i}: {e}")
        test_results.append({
            'question': test_case['question'],
            'response': f"Error: {e}",
            'category': test_case['category'],
            'response_length': 0
        })
    
    print(f"{'='*60}\n")

# Summary of test results
print("  Test Summary:")
print(f"   Total questions tested: {len(test_questions)}")
print(f"   Successful responses: {sum(1 for r in test_results if not r['response'].startswith('Error:'))}")
print(f"   Average response length: {sum(r['response_length'] for r in test_results) / len(test_results):.1f} words")

# Save test results
import json
with open('test_results.json', 'w') as f:
    json.dump(test_results, f, indent=2)

print(f"  Test results saved to test_results.json")


def interactive_chat():
    """Start an interactive chat session with SecondYou."""
    print(" SecondYou AI Interactive Chat Session")
    print("=" * 50)
    print(" Chat with your personalized AI!")
    print(" Commands:")
    print("   'quit', 'exit', 'bye' - End the chat")
    print("   'help' - Show this help message")
    print("   'clear' - Clear conversation history")
    print("   'save' - Save conversation to file")
    print("=" * 50)
    print()
    
    conversation_history = []
    
    while True:
        try:
            user_input = input(" You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print(" SecondYou: Thanks for chatting! Hope I was helpful. Goodbye! ")
                break
            elif user_input.lower() == 'help':
                print(" SecondYou: Here are the available commands:")
                print("   'quit', 'exit', 'bye' - End our chat")
                print("   'help' - Show this help message")
                print("   'clear' - Clear our conversation history")
                print("   'save' - Save our conversation to a file")
                print("   Or just ask me anything! ")
                continue
            elif user_input.lower() == 'clear':
                conversation_history = []
                print(" SecondYou: Conversation history cleared! Fresh start! ")
                continue
            elif user_input.lower() == 'save':
                if conversation_history:
                    filename = f"chat_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(filename, 'w') as f:
                        json.dump(conversation_history, f, indent=2)
                    print(f" SecondYou: Conversation saved to {filename} ")
                else:
                    print(" SecondYou: No conversation to save yet! Start chatting first ")
                continue
            elif not user_input:
                print(" SecondYou: I'm here! Please say something or type 'help' for commands. ")
                continue
            
            # Generate response
            print(" SecondYou: ", end="", flush=True)
            response = chat_with_secondyou(user_input)
            print(response)
            print()
            
            # Save to conversation history
            conversation_history.append({
                'timestamp': datetime.now().isoformat(),
                'user': user_input,
                'assistant': response
            })
            
        except KeyboardInterrupt:
            print("\n\n SecondYou: Chat interrupted. Goodbye! ")
            break
        except Exception as e:
            print(f" Error in chat: {e}")
            print(" SecondYou: Sorry, I had a technical hiccup. Let's try again! ")

# Enhanced chat function for Jupyter notebooks
def notebook_interactive_chat(max_rounds=10):
    """Interactive chat optimized for Jupyter notebooks."""
    print(" SecondYou Notebook Chat")
    print("=" * 40)
    print(f" Ask up to {max_rounds} questions!")
    print(" Type your question and press Enter")
    print(" Type 'stop' to end early")
    print("=" * 40)
    print()
    
    for round_num in range(1, max_rounds + 1):
        try:
            user_input = input(f"Round {round_num}/{max_rounds} - Your question: ").strip()
            
            if user_input.lower() in ['stop', 'quit', 'exit']:
                print(" SecondYou: Thanks for the chat! ")
                break
            
            if not user_input:
                print(" SecondYou: Please ask me something! ")
                continue
            
            response = chat_with_secondyou(user_input)
            display_conversation(user_input, response)
            
        except KeyboardInterrupt:
            print("\n SecondYou: Chat stopped. See you next time! ")
            break
        except Exception as e:
            print(f" Error: {e}")
            continue
    
    if round_num >= max_rounds:
        print(f" Completed {max_rounds} rounds! Start a new session to continue chatting.")

print(" Interactive chat functions ready!")
print(" Usage:")
print("   interactive_chat() - Full interactive mode")
print("   notebook_interactive_chat() - Notebook-optimized chat")
print()
print(" Uncomment the line below to start chatting:")
print("# notebook_interactive_chat()")

# Uncomment to start interactive chat:
# notebook_interactive_chat()


# Test 1: Personal productivity question
test_prompt_1 = "I'm feeling overwhelmed with all my projects. How do you usually manage multiple tasks?"
response_1 = chat_with_secondyou(test_prompt_1)
display_conversation(test_prompt_1, response_1)


# Test 2: Technical question
test_prompt_2 = "What programming language should I learn next and why?"
response_2 = chat_with_secondyou(test_prompt_2)
display_conversation(test_prompt_2, response_2)


# Test 3: Ask about learning and growth
test_prompt_3 = "I want to start a new creative project but I'm not sure where to begin. Any advice?"
response_3 = chat_with_secondyou(test_prompt_3)
display_conversation(test_prompt_3, response_3)

# Interactive testing - Try your own questions!
print(" Try asking your SecondYou AI anything!")
print(" Notice how it responds in your style and with your perspective")
print(" Uncomment the lines below to ask custom questions:")
print()
print("# your_question = \"Your question here\"")
print("# response = chat_with_secondyou(your_question)")
print("# display_conversation(your_question, response)")

# Uncomment these lines to test with your own questions:
# your_question = "What's your perspective on the future of AI?"
# response = chat_with_secondyou(your_question)
# display_conversation(your_question, response)


# Model Export and Deployment Preparation
print(" Preparing SecondYou for deployment...")
print(" Converting to multiple formats for different use cases")

# First, prepare the model for inference
print("\nâƒ£ Preparing model for inference...")
model = FastLanguageModel.for_inference(model)
print(" Model ready for inference")

# Create export directory
import os
os.makedirs("secondyou-exports", exist_ok=True)

export_formats = []

try:
    # GGUF format for local deployment (most popular)
    print("\n Converting to GGUF format...")
    
    # Q4_K_M - Best balance of size and quality (recommended)
    print("   Converting to Q4_K_M (recommended)...")
    model.save_pretrained_gguf(
        "secondyou-exports/secondyou-gemma3n-q4km", 
        tokenizer, 
        quantization_method="q4_k_m"
    )
    export_formats.append("Q4_K_M (4GB) - Recommended for most users")
    
    # Q8_0 - Higher quality, larger size
    print("   Converting to Q8_0 (high quality)...")
    model.save_pretrained_gguf(
        "secondyou-exports/secondyou-gemma3n-q8", 
        tokenizer, 
        quantization_method="q8_0"
    )
    export_formats.append("Q8_0 (7GB) - High quality, larger size")
    
    # F16 - Full precision (if you have lots of storage/memory)
    print("   Converting to F16 (full precision)...")
    model.save_pretrained_gguf(
        "secondyou-exports/secondyou-gemma3n-f16", 
        tokenizer, 
        quantization_method="f16"
    )
    export_formats.append("F16 (15GB) - Full precision, largest size")
    
    print(" GGUF conversion complete!")
    
except Exception as e:
    print(f" GGUF conversion had issues: {e}")
    print("   This might be due to environment limitations")

# Save the merged model (HuggingFace format)
print("\n Saving merged HuggingFace format...")
try:
    model.save_pretrained_merged(
        "secondyou-exports/secondyou-gemma3n-hf",
        tokenizer,
        save_method="merged_16bit"
    )
    export_formats.append("HuggingFace (8GB) - Standard format for inference")
    print(" HuggingFace format saved!")
except Exception as e:
    print(f" HuggingFace save had issues: {e}")

# Save LoRA adapters separately
print("\n Saving LoRA adapters...")
try:
    model.save_pretrained("secondyou-exports/secondyou-lora-adapters")
    tokenizer.save_pretrained("secondyou-exports/secondyou-lora-adapters")
    export_formats.append("LoRA Adapters (100MB) - Lightweight, requires base model")
    print(" LoRA adapters saved!")
except Exception as e:
    print(f" LoRA save had issues: {e}")

# Create deployment guide
deployment_guide = f"""
# SecondYou Deployment Guide

## Available Formats

{chr(10).join(f"- {fmt}" for fmt in export_formats)}

## Usage Instructions

### Option 1: GGUF with Ollama (Recommended)
1. Install Ollama: https://ollama.ai/
2. Copy the Q4_K_M file to Ollama models directory
3. Create a Modelfile:
   ```
   FROM ./secondyou-gemma3n-q4km.gguf
   PARAMETER temperature 0.7
   PARAMETER top_p 0.9
   SYSTEM "You are SecondYou, a personalized AI assistant trained on the user's conversation style and preferences."
   ```
4. Run: `ollama create secondyou -f Modelfile`
5. Chat: `ollama run secondyou`

### Option 2: HuggingFace Transformers
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "./secondyou-exports/secondyou-gemma3n-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./secondyou-exports/secondyou-gemma3n-hf")

# Chat function provided in the notebook
```

### Option 3: LoRA Adapters
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("unsloth/gemma-3n-E4B-it")
model = PeftModel.from_pretrained(base_model, "./secondyou-exports/secondyou-lora-adapters")
```

## Model Stats
- Base Model: Gemma 3N (4B parameters)
- Fine-tuning: LoRA with rank {lora_config['r']}
- Training Data: {len(train_dataset)} examples
- Training Time: {training_duration/60:.1f} minutes
"""

with open("secondyou-exports/DEPLOYMENT_GUIDE.md", "w") as f:
    f.write(deployment_guide)

print(f"\n Deployment Summary:")
print(f"   Formats created: {len(export_formats)}")
print(f"   Export directory: ./secondyou-exports/")
print(f"   Deployment guide: ./secondyou-exports/DEPLOYMENT_GUIDE.md")
print(f"\n SecondYou is ready for deployment!")
print(f" Check the DEPLOYMENT_GUIDE.md for usage instructions")


# Optional: Upload to Hugging Face Hub
from huggingface_hub import HfApi

def upload_to_huggingface(repo_name, token):
    """Upload your model to Hugging Face Hub."""
    try:
        # Push the merged model
        model.push_to_hub_merged(
            repo_name,
            tokenizer,
            save_method="merged_16bit",
            token=token,
            private=True  # Set to False if you want it public
        )
        
        print(f"Model uploaded successfully to: https://huggingface.co/{repo_name}")
        
    except Exception as e:
        print(f"Upload failed: {e}")
        print("Make sure you have a valid Hugging Face token")

# Example usage (uncomment and provide your details):
# upload_to_huggingface("your-username/secondyou-gemma3n", "your-hf-token")

print("Upload function ready!")
print("Uncomment the last line and provide your details to upload to Hugging Face.")

