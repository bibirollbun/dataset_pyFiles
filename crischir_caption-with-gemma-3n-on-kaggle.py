!pip install timm --upgrade
!pip install accelerate
!pip install git+https://github.com/huggingface/transformers.git


import kagglehub

GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")


# import transformers
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(GEMMA_PATH, trust_remote_code=True)
# prompt = "Why are there so many Geese on Kaggle?"
# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
# generation_config = GenerationConfig(max_new_tokens=150, do_sample=True, temperature=0.7)
# outputs = model.generate(**inputs, generation_config=generation_config)
# result = tokenizer.decode(outputs[0], skip_special_tokens=True)


# print(result)


from IPython.display import Image
IMAGE_URL="/kaggle/input/pozetest/80TMKEFE5Q62CA2QXRWGEQR740.jpeg"
Image(url=IMAGE_URL,height=250,width=250)


from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained(GEMMA_PATH)
model = AutoModelForImageTextToText.from_pretrained(GEMMA_PATH, torch_dtype="auto", device_map="auto")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": IMAGE_URL},
            # {"type": "text", "text": "Describe this image in 50 words"}
            {"type": "text", "text": "Describe this image in a comma-separated list of visual tags, similar to a WD14 style caption. Focus on objects, colors, actions, settings, and general descriptors. Do not use full sentences. Aim for a comprehensive list of tags, up to 50 distinct tags if applicable, without exceeding 100 words in total."}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device, dtype=model.dtype)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=64, disable_compile=True)
text = processor.batch_decode(
    outputs[:, input_len:],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True
)





print(text[0])


import os

def create_directory(path):
    """
    Creates a directory at the specified path.

    Args:
        path (str): The path of the directory to create.
    """
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Directory '{path}' created successfully (or already exists).")
    except OSError as e:
        print(f"Error creating directory '{path}': {e}")

def process_image_to_text(image_path):
    """
    Function to integrate with Gemma 3 for text extraction from an image.
    REPLACE THIS WITH YOUR ACTUAL GEMMA 3 INTEGRATION LOGIC.

    Args:
        image_path (str): The full path to the image file.

    Returns:
        str: The extracted text content from the image.
    """
    print(f"Calling Gemma 3 for image: {os.path.basename(image_path)}")

    # --- IMPORTANT: REPLACE THIS SECTION WITH YOUR REAL GEMMA 3 INTEGRATION ---
    # This is where you would typically:
    # 1. Read the image file content (e.g., in bytes).
    # 2. Prepare the image content for Gemma 3 (e.g., base64 encoding if it's an API,
    #    or passing bytes directly if it's an SDK designed for it).
    # 3. Make an API call or use the Gemma 3 SDK to send the image.
    # 4. Parse the response to get the extracted text.

    try:
        # Example if using google.generativeai (conceptual):
        # model = genai.GenerativeModel('gemma-3-vision') # Or whatever the model name is
        # with open(image_path, 'rb') as f:
        #     image_data = f.read()
        #
        # # Assuming Gemma 3 can take image bytes directly or a file path
        # # You might need to adjust the content structure based on Gemma 3's actual API/SDK
        # response = model.generate_content(['Extract all text from this image:', image_data])
        # extracted_text = response.text
        # return extracted_text

        # For demonstration without actual Gemma 3 integration,
        # we'll return a dummy string.
        messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": IMAGE_URL},
                                # {"type": "text", "text": "Describe this image in 50 words"}
                                {"type": "text", "text": "Describe this image in a comma-separated list of visual tags, similar to a WD14 style caption. Focus on objects, colors, actions, settings, and general descriptors. Do not use full sentences. Aim for a comprehensive list of tags, up to 50 distinct tags if applicable, without exceeding 100 words in total."}
                            ]
                        }
                    ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, dtype=model.dtype)
        input_len = inputs["input_ids"].shape[-1]
        
        outputs = model.generate(**inputs, max_new_tokens=512, disable_compile=True)
        text = processor.batch_decode(
            outputs[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        #         dummy_text = (
        #             f"This text was hypothetically extracted by Gemma 3 from "
        #             f"'{os.path.basename(image_path)}'.\n"
        #             f"*** PLEASE REPLACE THE CONTENT OF `process_image_to_text` "
        #             f"WITH YOUR ACTUAL GEMMA 3 API/SDK CALL. ***"
        #         )
        return text[0]
        
    except Exception as e:
        return f"Error integrating with Gemma 3 for {os.path.basename(image_path)}: {e}\n" \
               f"Please ensure your Gemma 3 integration (API key, library, model call) is correct."



def process_images_in_directory(input_directory, output_directory=None):
    """
    Reads image files from a directory, applies a text extraction function (Gemma 3),
    and writes the output to text files.

    Args:
        input_directory (str): The path to the directory containing image files.
        output_directory (str, optional): The directory where the text files will be saved.
                                          If None, text files will be saved in the input_directory.
                                          Defaults to None.
    """
    if not os.path.isdir(input_directory):
        print(f"Error: Input directory '{input_directory}' does not exist.")
        return

    if output_directory and not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")

    # Common image extensions. Add or remove as needed.
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')

    print(f"Starting image processing in: {input_directory}")
    processed_count = 0
    skipped_count = 0
    list_of_files=os.listdir(input_directory)
    for filename in list_of_files:
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(input_directory, filename)
            base_name = os.path.splitext(filename)[0] # Get filename without extension

            if output_directory:
                output_txt_path = os.path.join(output_directory, f"{base_name}.txt")
            else:
                output_txt_path = os.path.join(input_directory, f"{base_name}.txt")

            try:
                extracted_text = process_image_to_text(image_path)

                with open(output_txt_path, 'w', encoding='utf-8') as f:
                    f.write(extracted_text)
                print(f"Successfully processed '{filename}' and saved text to '{output_txt_path}'")
                processed_count += 1
            except Exception as e:
                print(f"Failed to process '{filename}': {e}")
        else:
            print(f"Skipping non-image file: {filename}")
            skipped_count += 1


    print(f"\nFinished processing. Total images processed: {processed_count}")
    print(f"Total non-image files skipped: {skipped_count}")
    if processed_count == 0:
        print("No image files with specified extensions found or successfully processed.")





# --- Configuration ---
new_dir = "/kaggle/working/prompts"
create_directory(new_dir)
input_image_directory = '/kaggle/input/pozetest/'
output_text_directory = "/kaggle/working/prompts"
# -------------------

# # --- Create a dummy directory and some dummy image files for testing ---
# # You can comment out this section if you already have your own images
# dummy_test_dir = 'gemma_test_images'
# os.makedirs(dummy_test_dir, exist_ok=True)
# print(f"\n--- Setting up dummy test directory: {dummy_test_dir} ---")
# with open(os.path.join(dummy_test_dir, 'invoice.png'), 'w') as f: f.write("dummy png content for Gemma")
# with open(os.path.join(dummy_test_dir, 'receipt.jpg'), 'w') as f: f.write("dummy jpg content for Gemma")
# with open(os.path.join(dummy_test_dir, 'document_page.gif'), 'w') as f: f.write("dummy gif content for Gemma")
# with open(os.path.join(dummy_test_dir, 'config.ini'), 'w') as f: f.write("This should be ignored by the script")
# print("Created dummy image files for testing (these are just empty files for the script to find).\n")
# input_image_directory = dummy_test_dir # Use the dummy directory for testing
# ----------------------------------------------------------------------

process_images_in_directory(input_image_directory, output_text_directory)

# --- Optional: Clean up dummy files ---
# import shutil
# if os.path.exists(dummy_test_dir):
#     print(f"\n--- Cleaning up dummy test directory: {dummy_test_dir} ---")
#     shutil.rmtree(dummy_test_dir)
#     print(f"Cleaned up dummy test directory: {dummy_test_dir}")


# --- Add this code to list files in the output directory ---
print("\n--- Listing files in the output directory ---")
try:
    with os.scandir(output_text_directory) as entries:
        print(f"Files and directories in '{output_text_directory}':")
        for entry in entries:
            print(f"- {entry.name}")
except FileNotFoundError:
    print(f"Error: Directory '{output_text_directory}' not found.")
except Exception as e:
    print(f"An error occurred while listing files: {e}")




