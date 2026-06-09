!pip install --upgrade unsloth
!pip install --no-deps transformers==4.54.1  # the latest version, 4.55.0.dev0, results in error when running image processing
#!pip install --no-deps git+https://github.com/huggingface/transformers.git
!pip install --no-deps --upgrade timm

from unsloth import FastModel
import torch
import gc

# Set torch parameter to avoid error message, "FailOnRecompileLimitHit: recompile_limit reached with one_graph=True." when doing inference on images
torch._dynamo.config.cache_size_limit = 32   


# Initialize model
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
#    model_name = "unsloth/gemma-3n-E2B-it", # This runs out of memory for the recommend/analyze chats
    dtype = None, # None for auto detection
    max_seq_length = 1024, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)

# Helper function for inference
def do_gemma_3n_inference(model, messages, max_new_tokens = 128):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True, # Must add for generation
        tokenize = True,
        return_dict = True,
        return_tensors = "pt",
    ).to("cuda")

    with torch.no_grad():  # Disable gradient calculation during inference
        outputs = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            temperature = 1.0, top_p = 0.95, top_k = 64,
            return_dict_in_generate=True,  # Crucial: Get the full output
        )

    # Decode generated tokens
    outputs_excluding_inputs = outputs.sequences[:, inputs.input_ids.shape[1]:]    # exclude input tokens
    generated_text = tokenizer.batch_decode(outputs_excluding_inputs, skip_special_tokens=True)[0]

    # Cleanup to reduce VRAM usage
    del inputs
    torch.cuda.empty_cache()
    gc.collect()

    return generated_text


def query_ai_text_image(text, image_path=None): 
    ''' Query AI with a prompt that includes text and an image. '''
    if image_path is None:
        return "No image uploaded."
    messages = [{
        "role" : "user",
        "content": [
            { "type": "image", "image" : image_path },
            { "type": "text",  "text" : text }
        ]
    }]
    text = do_gemma_3n_inference(model, messages, max_new_tokens = 256)
    return text


def query_ai_text(text): 
    ''' Query AI with a text prompt. '''
    messages = [{
        "role" : "user",
        "content": [
            { "type": "text",  "text" : text }
        ]
    }]
    text = do_gemma_3n_inference(model, messages, max_new_tokens = 256)
    return text


import pandas as pd

# Define Inventory class object
class Inventory:
    column_names = ['title', 'author', 'year_published', 'isbn', 'description', 'copies_on_shelf', 'total_copies']

    def __init__(self, input_file_path, output_file_path): 
        ''' Initialize library inventory with data from an input csv file. Specify the file path for storing updated inventory. '''

        # Load input file, keeping only the relevant columns
        data = pd.read_csv(input_file_path)
        data = data[ [col for col in data.columns if col in self.column_names] ]

        # Check if input contains the required fields of "title" and "description"
        for col in ['title', 'description']: 
            if col not in data.columns: 
                raise Exception(f"Input book info must contain '{col}'.")
        
        # If the number of copies is not available in the input data, set it to the default value of 1
        for col in ['copies_on_shelf', 'total_copies']: 
            if col not in data.columns: 
                print(f"Input {col} not found. Setting to default value 1.")
                data[col] = 1

        self.data = data
        self.file_path = output_file_path
        self.save()

    
    def save(self): 
        ''' Save inventory data to file. '''
        self.data.to_csv(self.file_path, index=False)
    
    def get_index(self, title): 
        ''' Return a pandas Index list of book(s) that match a given title. '''
        idx = self.data[self.data.title.str.lower() == title.lower()].index
        if idx.size == 0: 
            return None
        if idx.size > 1: 
            raise Exception(f"Found {idx.size} books with the title '{title}'.") #TODO: Match on author as well.
        return idx[0]

    def check_out(self, title): 
        i = self.get_index(title)
        if i is None: 
            return "ERROR: Title not found in library collection." # TODO: Add book to collection
        if self.data.loc[i, 'copies_on_shelf'] == 0: 
            return "ERROR: Check out unsuccessful. There are 0 copies on shelf."
        self.data.loc[i, 'copies_on_shelf'] -= 1
        self.save()
        return f"Check out successful. {self.data.loc[i, 'copies_on_shelf']} of {self.data.loc[i, 'total_copies']} copies remaining."

    def check_in(self, title): 
        i = self.get_index(title)
        if i is None: 
            return "ERROR: Title not found in library collection."
        row = self.data.loc[i]
        if row.copies_on_shelf == row.total_copies: 
            return f"ERROR: Check in unsuccessful. {row.copies_on_shelf} of {row.total_copies} copies already on shelf."
        self.data.loc[i, 'copies_on_shelf'] += 1
        self.save()
        return f"Check in successful. {self.data.loc[i, 'copies_on_shelf']} of {self.data.loc[i, 'total_copies']} copies on shelf."

    def get_on_shelf_book_info(self): 
        ''' Return the title/author/description info of all books with available copies on shelf, in csv format. '''
        columns = ['title', 'author', 'description']
        return self.data[self.data.copies_on_shelf > 0][columns].to_csv()

    def get_df(self): 
        ''' Return inventory data. '''
        return self.data

    def get_dtypes(self): 
        ''' Get data types for each column. '''
        return self.data.dtypes

    def set_df(self, data): 
        ''' Set inventory as the input DataFrame. '''
        self.data = data


# Create mobile library Inventory object

# Input file path
initial_book_list = '/kaggle/input/caldecott-medal-winners-1938-2019/caldecott_winners.csv'

# Working file path
inventory_file_path = '/kaggle/working/inventory.csv'

# Initialize inventory
# NOTE: Due to runtime memory limitations, we only demonstrate the application on the subset of books that have short descriptions.
inventory = Inventory(initial_book_list, inventory_file_path)
inventory.set_df(inventory.get_df()[inventory.get_df().description.str.count(' ') < 50])


import gradio as gr
from datetime import datetime


# --- "Scan" tab ---
def scan_book(image, action):

    # Query AI to extract the title and author
    prompt = "Extract the title and author from this book cover image. Format the output as ('[title]', '[author]'). If unsuccessful, output ('Unknown Title', 'Unknown Author')."
    title, author = query_ai_text_image(prompt, image)

    # AI query success check
    if title == "Unknown Title" or author == "Unknown Author":
        return "Could not reliably extract book information from the image. Please try again with a clearer cover."

    # Get the right function (check out or check in)
    if action == 'out': 
        fn = inventory.check_out
    elif action == 'in': 
        fn = inventory.check_in
    else: 
        raise Exception(f'Unknown action {action}. Valid options are "out" or "in".')

    # Perform action and return results
    return f"Title: {title}\nAuthor: {author}\n" + fn(title)


# --- "Recommend" tab ---
recommend_examples = [
    ["Suggest five books for a toddler who loves animals."],
    ["Find 3 books for a preschooler interested in space."],
    ["What are some books about adventures?"]
]

def recommend_chat_response(message, history): 
    prompt = "You are a helpful librarian making book recommendations based on the user's description of the reader's background and interests. Respond with 3-5 books, unless otherwise specified by the user. Respond with a bullet point list formatted '[title] by [author]', followed by a short sentence of less than 20 words about why this book was chosen. You must only choose books from the following csv file: " + inventory.get_on_shelf_book_info()
    return query_ai_text(f"{prompt} \n User question: {message}")


# --- "Analyze" tab ---
analyze_examples = [
    ["What is the newest book we have?"], 
    ["Summarize popular themes in our collection."]
]

def analyze_chat_response(message, history): 
    prompt = "You are a helpful librarian answering questions about the library's collection of books, based only on this inventory data: " + inventory.get_df().to_csv(index=False)
    return query_ai_text(f"{prompt} \n User question: {message}")


# --- "Manage" tab ---
def save_inventory(df_input): 
    ''' Save the user-edited DataFrame as the inventory DataFrame. ''' # TODO: More robust error checks
    df = pd.DataFrame(df_input)

    # Explicitly convert columns to desired data types
    col_type = inventory.get_dtypes().to_list()
    for i,col in enumerate(df.columns): 
        df[col] = df[col].astype(col_type[i])

    # Save DataFrame
    inventory.set_df(df)
    inventory.save()


# --- Main gradio app ---
with gr.Blocks() as demo:
    gr.Markdown("# ğŸš� MoLi: Mobile Librarian ğŸ“š")
    gr.Markdown("Scan to check out/in, get book recommendations, and analyze your collection, powered by Google's Gemma 3n AI!")

    with gr.Tabs() as tabs: 

        # Scan book to check out or check in
        actions = ['out', 'in']
        with gr.Tab(label='Scan'):
            image_input = gr.Image(type='filepath', label="Upload book cover or take a photo", sources=['upload', 'webcam'], width=300)
            with gr.Row(): 
                button = {a: gr.Button(f'Check {a}') for a in actions}
            status_text = gr.Textbox(show_label=False)
            button['out'].click(fn=lambda x: scan_book(x, 'out'), inputs=image_input, outputs=status_text)
            button['in'].click(fn=lambda x: scan_book(x, 'in'), inputs=image_input, outputs=status_text)
            # # Somehow the following does not work:
            # for a, b in button.items():
            #     b.click(fn=lambda x: scan_book(x, a), inputs=image_input, outputs=status_text)

        with gr.Tab(label='Recommend'): 
            recommend_greeting = "Tell me the reader's background and interests, and I'll recommend some books available for check out!"
            gr.ChatInterface(
                fn=recommend_chat_response,
                type='messages',
                examples=recommend_examples,
                chatbot=gr.Chatbot(type='messages', placeholder=recommend_greeting),
            )
        
        with gr.Tab(label='Analyze'): 
            analyze_greeting = "Ask me anything about the library collection!"
            gr.ChatInterface(
                fn=analyze_chat_response,
                type='messages',
                examples=analyze_examples,
                chatbot=gr.Chatbot(type='messages', placeholder=analyze_greeting),
            )

        with gr.Tab(label='Manage'): 

            # Buttons
            with gr.Row(): 
                reload_button = gr.Button('Reload')
                save_button = gr.Button('Save changes')
            
            # Textbox to display status messages
            status_message = gr.Textbox(show_label=False, value='Please reload after check out or check in.')

            # Inventory table
            inventory_table = gr.DataFrame(
                value=inventory.get_df(),
                interactive=True, # Allow editing
                label="Current Library Inventory", 
                wrap=True
            )

            # Attach functions to buttons
            reload_button.click(fn=inventory.get_df, outputs=inventory_table).then(fn=lambda:f"Reloaded on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", outputs=[status_message])
            save_button.click(fn=save_inventory, inputs=inventory_table).then(fn=lambda:f"Saved on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", outputs=[status_message])


if __name__ == '__main__':
    demo.launch()

