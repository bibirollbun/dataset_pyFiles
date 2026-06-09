import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer
import kagglehub
import torch
import os
import xml.etree.ElementTree as ET
import io

import warnings
warnings.filterwarnings("ignore")



# Attempt to import pdfminer, and install it if it's not present
try:
    from pdfminer.converter import TextConverter
    from pdfminer.pdfinterp import PDFPageInterpreter
    from pdfminer.pdfinterp import PDFResourceManager
    from pdfminer.pdfpage import PDFPage
    PDF_MINER_INSTALLED = True
except ModuleNotFoundError:
    print("pdfminer.six not found.  Installing...")
    import subprocess
    try:
        subprocess.check_call(['pip', 'install', 'pdfminer.six'])
        from pdfminer.converter import TextConverter
        from pdfminer.pdfinterp import PDFPageInterpreter
        from pdfminer.pdfinterp import PDFResourceManager
        from pdfminer.pdfpage import PDFPage
        PDF_MINER_INSTALLED = True
        print("pdfminer.six installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing pdfminer.six: {e}")
        PDF_MINER_INSTALLED = False
        print("PDF processing will be disabled.")


# Define paths
TRAIN_LABELS_PATH = '/kaggle/input/make-data-count-finding-data-references/train_labels.csv'
SAMPLE_SUBMISSION_PATH = '/kaggle/input/make-data-count-finding-data-references/sample_submission.csv'

# Train Paths
TRAIN_PATH = '/kaggle/input/make-data-count-finding-data-references/train'
TRAIN_PDF_PATH = '/kaggle/input/make-data-count-finding-data-references/train/PDF'
TRAIN_XML_PATH = '/kaggle/input/make-data-count-finding-data-references/train/XML'

# Test Paths
TEST_PATH = '/kaggle/input/make-data-count-finding-data-references/test'
TEST_PDF_PATH = '/kaggle/input/make-data-count-finding-data-references/test/PDF'
TEST_XML_PATH = '/kaggle/input/make-data-count-finding-data-references/test/XML'



# Load data
train_labels = pd.read_csv(TRAIN_LABELS_PATH)
sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

# Display head of train labels
print("Train Labels Head:")
train_labels.head()


# --- EDA ---
print("\nTrain Labels Info:")
train_labels.info()

print("\nTrain Labels Value Counts:")
print(train_labels['type'].value_counts())

# Visualizing the distribution of 'type'
plt.figure(figsize=(8, 6))
sns.countplot(data=train_labels, x='type', palette='viridis')
plt.title('Distribution of Data Reference Types', fontsize=16, fontweight='bold')
plt.xlabel('Data Reference Type', fontsize=14, fontweight='bold')
plt.ylabel('Count', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.gca().set_facecolor('#f0f0f0')  # Optional: Change background color
plt.show()



# --- Model Loading ---

model_name = kagglehub.model_download("qwen-lm/qwen-3/transformers/0.6b")


# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
model.eval()  # Set the model to evaluation mode



# --- QwenChatbot Class ---

class QwenChatbot:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True)
        self.model.eval() # Set the model to evaluation mode here as well.
        self.history = []

    def generate_response(self, user_input, enable_thinking=True, max_new_tokens=512):  # Reduce max_new_tokens
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad(): # Disable gradient calculation during inference
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,  # Use the reduced value
                temperature=0.6 if enable_thinking else 0.7,
                top_p=0.95 if enable_thinking else 0.8,
                top_k=20,
                min_p=0
            )

        output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()

        #parsing thinking content
        try:
            # rindex finding 151668 ()
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        response = content

        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response, thinking_content
    
    def clear_history(self):
        self.history = []

# --- Data Extraction Helpers ---

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    if not PDF_MINER_INSTALLED:
        print("PDF processing is disabled because pdfminer.six is not installed.")
        return ""

    resource_manager = PDFResourceManager()
    output_string = io.StringIO()
    codec = 'utf-8'
    laparams = None
    converter = TextConverter(resource_manager, output_string, codec=codec, laparams=laparams)
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    
    with open(pdf_path, 'rb') as fh:
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            page_interpreter.process_page(page)
    
    text = output_string.getvalue()
    converter.close()
    output_string.close()
    return text

def extract_text_from_xml(xml_path):
    """Extracts text from an XML file, concatenating all text elements."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        text_parts = []
        for element in root.iter():
            if element.text:
                text_parts.append(element.text)
        return "\n".join(text_parts)  # Join with newlines for readability
    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_path}: {e}")
        return ""

# --- Function to Extract Citations ---
def extract_citations(file_path, file_type, chatbot):
    """Extracts citations from a single PDF or XML file using the chatbot."""
    article_id = os.path.basename(file_path).split('.')[0]  # Extract article_id from filename
    text = ""

    if file_type == "pdf" and PDF_MINER_INSTALLED:
        text = extract_text_from_pdf(file_path)
    elif file_type == "xml":
        text = extract_text_from_xml(file_path)
    else:
        print(f"Unsupported file type or PDF processing disabled for {file_path}")
        return []

    if not text:
        print(f"No text extracted from {file_path}")
        return []
    
    citations = []
    # Chunk the text to avoid exceeding token limits
    chunk_size = 500  # Further reduce chunk size to 500 or lower
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        prompt = f"Analyze the following research paper snippet and identify any references to datasets, databases, or repositories. Please list the dataset ID and the type of reference (Primary, Secondary, etc.) if possible. If no dataset is mentioned, respond with 'No dataset mentioned'.\n\n{chunk}"

        response, _ = chatbot.generate_response(prompt, enable_thinking=True, max_new_tokens=128)  # Reduce max_new_tokens even further

        # Extract dataset_id and type from response using regular expressions
        dataset_matches = re.findall(r"(https?://doi\.org/[^\s]+|[A-Z]+[0-9]+)(?:.*?(Primary|Secondary))?", response, re.IGNORECASE) # Improved regex

        for dataset_id, ref_type in dataset_matches:
            dataset_id = dataset_id.strip() # Clean whitespace
            dataset_id = dataset_id if dataset_id.startswith("https://doi.org") else f"https://doi.org/{dataset_id}" if "/" not in dataset_id else dataset_id
            ref_type = ref_type or "Other"  # Default to "Other" if not specified
            citations.append((article_id, dataset_id, ref_type))
            
            # Clear history after each chunk to reduce memory usage
            chatbot.clear_history()
    return citations

# --- Main Processing Loop ---
if __name__ == "__main__":
    chatbot = QwenChatbot(model_name)
    citations = []  # Initialize citations list here

    # Process training data
    # Example: Analyze a training PDF for data references
    example_pdf_filename_train = "10.1002_chem.201903120.pdf"  # Example train PDF file
    example_pdf_path_train = os.path.join(TRAIN_PDF_PATH, example_pdf_filename_train)

    if os.path.exists(example_pdf_path_train):
        if PDF_MINER_INSTALLED:
            pdf_text_train = extract_text_from_pdf(example_pdf_path_train)
            # Create a prompt asking the model to identify data references
            prompt_train = f"Analyze the following research paper abstract and identify any references to datasets, databases, or repositories. Please list the dataset ID and the type of reference (Primary, Secondary, etc.) if possible.\n\n{pdf_text_train[:500]}" # Limit to first 500 chars
            print(f"User (Train PDF): {prompt_train}")
            response_train, thinking_content_train = chatbot.generate_response(prompt_train, enable_thinking=True, max_new_tokens=128) # Reduce max_new_tokens
            print(f"Bot (Train PDF): {response_train}")
            print(f"Thinking Content (Train PDF): {thinking_content_train}")
            citations.extend(extract_citations(example_pdf_path_train, "pdf", chatbot))  # Append citations from this file
            chatbot.clear_history() # Clear after processing the file
        else:
            print("Skipping PDF processing because pdfminer.six could not be installed.")

    else:
        print(f"PDF file not found: {example_pdf_path_train}")
    print("----------------------")

    # Example: Analyze a training XML for data references
    example_xml_filename_train = "10.1002_chem.201903120.xml"  # Example train XML file
    example_xml_path_train = os.path.join(TRAIN_XML_PATH, example_xml_filename_train)

    if os.path.exists(example_xml_path_train):
        xml_text_train = extract_text_from_xml(example_xml_path_train)
        # Create a prompt asking the model to identify data references
        prompt_train = f"Analyze the following research paper abstract and identify any references to datasets, databases, or repositories. Please list the dataset ID and the type of reference (Primary, Secondary, etc.) if possible.\n\n{xml_text_train[:500]}" # Limit to first 500 chars

        print(f"User (Train XML): {prompt_train}")
        response_train, thinking_content_train = chatbot.generate_response(prompt_train, enable_thinking=True, max_new_tokens=128) # Reduce max_new_tokens
        print(f"Bot (Train XML): {response_train}")
        print(f"Thinking Content (Train XML): {thinking_content_train}")
        citations.extend(extract_citations(example_xml_path_train, "xml", chatbot)) # Append citations from this file
        chatbot.clear_history() # Clear after processing the file

    else:
        print(f"XML file not found: {example_xml_path_train}")
    print("----------------------")


    # Example: Analyze a test PDF for data references
    example_pdf_filename_test = "10.1002_anie.202007717.pdf"  # Example test PDF file
    example_pdf_path_test = os.path.join(TEST_PDF_PATH, example_pdf_filename_test)

    if os.path.exists(example_pdf_path_test):
        if PDF_MINER_INSTALLED:
            pdf_text_test = extract_text_from_pdf(example_pdf_path_test)
            # Create a prompt asking the model to identify data references
            prompt_test = f"Analyze the following research paper abstract and identify any references to datasets, databases, or repositories. Please list the dataset ID and the type of reference (Primary, Secondary, etc.) if possible.\n\n{pdf_text_test[:500]}" # Limit to first 500 chars
            print(f"User (Test PDF): {prompt_test}")
            response_test, thinking_content_test = chatbot.generate_response(prompt_test, enable_thinking=True, max_new_tokens=128) # Reduce max_new_tokens
            print(f"Bot (Test PDF): {response_test}")
            print(f"Thinking Content (Test PDF): {thinking_content_test}")
            citations.extend(extract_citations(example_pdf_path_test, "pdf", chatbot))  # Append citations from this file
            chatbot.clear_history() # Clear after processing the file

        else:
            print("Skipping PDF processing because pdfminer.six could not be installed.")

    else:
        print(f"PDF file not found: {example_pdf_path_test}")
    print("----------------------")

    # Example: Analyze a test XML for data references
    example_xml_filename_test = "10.1002_anie.202007717.xml"  # Example test XML file
    example_xml_path_test = os.path.join(TEST_XML_PATH, example_xml_filename_test)

    if os.path.exists(example_xml_path_test):
        xml_text_test = extract_text_from_xml(example_xml_path_test)
        # Create a prompt asking the model to identify data references
        prompt_test = f"Analyze the following research paper abstract and identify any references to datasets, databases, or repositories. Please list the dataset ID and the type of reference (Primary, Secondary, etc.) if possible.\n\n{xml_text_test[:500]}" # Limit to first 500 chars

        print(f"User (Test XML): {prompt_test}")
        response_test, thinking_content_test = chatbot.generate_response(prompt_test, enable_thinking=True, max_new_tokens=128) # Reduce max_new_tokens
        print(f"Bot (Test XML): {response_test}")
        print(f"Thinking Content (Test XML): {thinking_content_test}")
        citations.extend(extract_citations(example_xml_path_test, "xml", chatbot)) # Append citations from this file
        chatbot.clear_history() # Clear after processing the file

    else:
        print(f"XML file not found: {example_xml_path_test}")
    print("----------------------")



    # Submission
    submission = pd.DataFrame(citations, columns=['article_id', 'dataset_id', 'type'])
    print("\nInitial Submission Head:")
    print(submission.head())
    print("\nInitial Submission Shape:", submission.shape)

    # Apply post-processing steps
    dataset_id_counts = submission['dataset_id'].value_counts()
    frequent_dataset_ids = dataset_id_counts[dataset_id_counts >= 3].index
    submission = submission[~submission['dataset_id'].isin(frequent_dataset_ids)].sort_values(by=["article_id", "dataset_id", "type"], ascending=True).drop_duplicates(subset=['article_id', 'dataset_id'])
    submission['row_id'] = range(len(submission))

    # Ensure all DOIs are in the correct format
    submission['dataset_id'] = submission['dataset_id'].apply(lambda x: x if x.startswith('https://doi.org') else 'https://doi.org/' + x if x.startswith("10.") else x)


    #Final Checks and Output
    print("\nFinal Submission Head:")
    print(submission.head())
    print("\nFinal Submission Shape:", submission.shape)

    submission[['row_id', 'article_id', 'dataset_id', 'type']].to_csv("submission.csv", index=False)
    print("\nSubmission file created: submission.csv")

