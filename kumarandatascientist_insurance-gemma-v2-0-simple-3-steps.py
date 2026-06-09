import pandas as pd

data = {
    "Question ID": range(1, 11),
    "Category": [
        "Insurance Basics",
        "Health Insurance",
        "Car Insurance",
        "Life Insurance",
        "Homeowners Insurance",
        "Insurance Basics",
        "Insurance Basics",
        "Travel Insurance",
        "Life Insurance",
        "Health Insurance"
    ],
    "Question": [
        "What is a deductible in an insurance policy?",
        "How do premiums affect coverage in health insurance?",
        "What factors influence car insurance rates?",
        "What is the difference between term and whole life insurance?",
        "How does homeowners insurance coverage work?",
        "What is the role of an insurance underwriter?",
        "How can you file a claim for property damage?",
        "What are the benefits of having travel insurance?",
        "How is the cash value of a life insurance policy determined?",
        "What is the process of underwriting in health insurance?"
    ],
    "Answer": [
        "", "", "", "", "", "", "", "", "", ""
    ]
}

df = pd.DataFrame(data)
print(df)

# Save the dataset to a CSV file
df.to_csv('insurance_questions_dataset.csv', index=False)



import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import keras
import keras_nlp
from kaggle_secrets import UserSecretsClient
from IPython.display import display, Markdown
import time
from tqdm import tqdm
from tensorflow.keras.mixed_precision import set_global_policy
import gc
import torch

# Configuration class to store all the parameters
class Config:
    def __init__(self, seed, dataset_path, preset, sequence_length, batch_size, lora_rank, learning_rate, epochs):
        self.seed = seed
        self.dataset_path = dataset_path
        self.preset = preset
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.lora_rank = lora_rank
        self.learning_rate = learning_rate
        self.epochs = epochs

# Main pipeline class to handle the insurance dataset and model training
class AndroGemmaInsurancePipeline:
    def __init__(self, config: Config):
        self.config = config
        self.setup_environment()
        # Load the pre-trained model from the preset
        self.gemma_causal_lm = keras_nlp.models.GemmaCausalLM.from_preset(config.preset)
        # Define the prompt attribute
        self.prompt = "\n\nCategory:\n{Category}\n\nQuestion:\n{Question}\n\nAnswer:\n{Answer}"

    # Set up the environment variables and secrets
    def setup_environment(self):
        """
        This method sets up various environment variables and secrets.
        It enables mixed precision training to optimize memory usage
        and sets a random seed for reproducibility.
        """
        os.environ["KERAS_BACKEND"] = "jax"
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"
        os.environ["JAX_PLATFORMS"] = ""
        user_secrets = UserSecretsClient()
        os.environ["KAGGLE_USERNAME"] = user_secrets.get_secret("username")
        os.environ["KAGGLE_KEY"] = user_secrets.get_secret("key")
        keras.utils.set_random_seed(self.config.seed)
        # Enable mixed precision
        set_global_policy('mixed_float16')

    # Load and preprocess the dataset
    def load_dataset(self):
        """
        This method reads the dataset from a CSV file,
        formats it into a prompt template, and stores it in a list.
        It also prints the time taken to load the dataset.
        """
        start_time = time.time()
        print("Loading dataset...")

        self.df = pd.read_csv(f"{self.config.dataset_path}/insurance_questions_dataset.csv")
        self.df["prompt"] = self.df.apply(lambda row: self.prompt.format(Category=row.Category, Question=row.Question, Answer=row.Answer), axis=1)
        self.data = self.df.prompt.tolist()

        end_time = time.time()
        print(f"Dataset loaded in {end_time - start_time:.2f} seconds.")

    # Add color formatting to the text for better visualization
    def colorize_text(self, text):
        """
        This method adds HTML color formatting to the text
        for better visualization in Markdown.
        """
        for word, color in zip(["Category", "Question", "Answer"], ["blue", "red", "green"]):
            text = text.replace(f"\n\n{word}:", f"\n\n**<font color='{color}'>{word}:</font>**")
        return text

    # Generate a response to a given query
    def query(self, category, question):
        """
        This method generates a response to a given query using the model
        and displays it with colorized text.
        """
        response = self.gemma_causal_lm.generate(
            self.prompt.format(
                Category=category,
                Question=question,
                Answer=""
            ), 
            max_length=self.config.sequence_length
        )
        display(Markdown(self.colorize_text(response)))
        return response

    # Train the model with the loaded dataset
    def train_model(self):
        """
        This method compiles and trains the model using the dataset.
        It also uses `tqdm` to show progress during training
        and prints the time taken to train the model.
        """
        start_time = time.time()
        print("Training model...")

        # Enable LoRA and compile the model
        self.gemma_causal_lm.backbone.enable_lora(rank=self.config.lora_rank)
        self.gemma_causal_lm.preprocessor.sequence_length = self.config.sequence_length
        self.gemma_causal_lm.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )
        
        # Fit the model with progress bar
        for epoch in tqdm(range(self.config.epochs), desc="Training Progress"):
            self.gemma_causal_lm.fit(self.data, epochs=1, batch_size=self.config.batch_size)

        end_time = time.time()
        print(f"Model trained in {end_time - start_time:.2f} seconds.")

    # Save the trained model to a preset directory
    def save_model(self, preset_dir):
        """
        This method saves the trained model to a specified directory
        and uploads it to Kaggle. It prints the time taken to save the model.
        """
        start_time = time.time()
        print("Saving model...")

        self.gemma_causal_lm.save_to_preset(preset_dir)
        kaggle_username = os.environ["KAGGLE_USERNAME"]
        kaggle_uri = f"kaggle://{kaggle_username}/gemma2-kaggle-docs/keras/gemma2_2b_en_insurance_gemma"
        keras_nlp.upload_preset(kaggle_uri, preset_dir)

        end_time = time.time()
        print(f"Model saved in {end_time - start_time:.2f} seconds.")

    # Clear memory after training
    def clear_memory(self):
        """
        This method clears the memory to free up resources.
        """
        print("Clearing memory...")
        del self.gemma_causal_lm
        gc.collect()
        torch.cuda.empty_cache()
        print("Memory cleared.")





# rerunning uncomment below 2 lines
torch.cuda.empty_cache()
gc.collect()

# Main function to run the entire pipeline
def main():
    # Configuration settings
    config = Config(
    seed=42,
    dataset_path="/kaggle/working/",
    preset="gemma2_2b_en",
    sequence_length=256,
    batch_size=1,
    lora_rank=4,
    learning_rate=8e-5,
    epochs=5
)

    # Initialize the pipeline
    pipeline = AndroGemmaInsurancePipeline(config)

    # Load the dataset
    pipeline.load_dataset()

    # Train the model
    pipeline.train_model()

    # Save the model
    pipeline.save_model("./gemma2_2b_en_insurance_gemma")

    # Query the model with an example question
    example_category = "Health Insurance"
    example_question = "What are the benefits of having health insurance?"
    pipeline.query(example_category, example_question)

if __name__ == "__main__":
    main()

# Clear pipeline variable
del pipeline
torch.cuda.empty_cache()
gc.collect()
print("Pipeline variable cleared.")





