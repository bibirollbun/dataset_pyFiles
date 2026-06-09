import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import random

# Set random seeds for reproducibility
torch.random.manual_seed(42)
random.seed(42)

def load_model():
    """Load the model and tokenizer with CPU configuration"""
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
        device_map="auto",  # This will default to CPU if no GPU is available
        torch_dtype=torch.float32,  # Use float32 for CPU
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

def generate_essay(topic_id, topic, pipe):
    """Generate an essay for a given topic"""
    
    # Craft prompt to generate controversial essays
    prompt = f"""Write a thought-provoking 100-word essay about {topic} that could be interpreted differently by different readers. 
    The essay should be well-written but present ideas that could be seen as controversial or debatable.
    Present multiple perspectives without clearly favoring one side."""
    
    messages = [
        {"role": "system", "content": "You are an essay writer focusing on complex topics."},
        {"role": "user", "content": prompt}
    ]
    
    generation_args = {
        "max_new_tokens": 200,  # Approximately 100 words
        "return_full_text": False,
        "temperature": 0.9,     # Higher temperature for more creativity
        "top_p": 0.9,          # Allow more diverse outputs
        "do_sample": True,
    }
    
    try:
        response = pipe(messages, **generation_args)[0]['generated_text']
        
        # Clean up the response
        essay = response.strip()
        words = essay.split()
        if len(words) > 120:  # Trim to approximately 100 words
            essay = ' '.join(words[:100]) + '.'
        
        return essay
    except Exception as e:
        print(f"Error generating essay for topic {topic_id}: {str(e)}")
        return f"Error generating essay for topic: {topic}"

def main():
    # Load the model
    print("Loading model...")
    pipe = load_model()
    print("Model loaded successfully!")
    
    # Read input file
    try:
        test = pd.read_csv('/kaggle/input/llms-you-cant-please-them-all/test.csv')
        print(f"Loaded {len(test)} topics from test.csv")
    except FileNotFoundError:
        print("Error: test.csv not found in the specified path")
        return
    
    essays = []  # List to store generated essays
    
    # Generate essays for each topic
    print("\nGenerating essays...")
    for idx, row in test.iterrows():
        topic = row['topic']
        topic_id = row['id']
        
        print(f"\nProcessing topic {topic_id}: {topic}")
        essay = generate_essay(topic_id, topic, pipe)
        essays.append(essay)
        
        # Print progress
        print(f"Essay {idx + 1}/{len(test)} completed")
        print("Generated essay:")
        print(essay)
        print('\n' + '*' * 50 + '\n')
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'id': test['id'],
        'essay': essays
    })
    
    # Save to CSV
    submission.to_csv('submission.csv', index=False)
    print("\nSubmission file has been saved as 'submission.csv'!")
    
    # Print sample of generated essays
    print("\nSample of generated essays:")
    print(submission.head())

if __name__ == "__main__":
    main()

