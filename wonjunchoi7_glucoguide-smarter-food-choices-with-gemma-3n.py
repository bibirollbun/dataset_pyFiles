!pip install timm --upgrade
!pip install accelerate
!pip install git+https://github.com/huggingface/transformers.git


!pip install timm==1.0.17
!pip install transformers==4.53.2


import kagglehub
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Download the model from Kaggle
# This path points to the model downloaded in the notebook
GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")

# Set up the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(GEMMA_PATH, trust_remote_code=True)



from IPython.display import Image

# Define the URL of the image we want to analyze
IMAGE_URL="https://ai.google.dev/static/gemma/docs/images/thali-indian-plate.jpg"

# Display the image to verify it's loaded correctly
Image(url=IMAGE_URL, height=250, width=250)


from IPython.display import Image

IMAGE_PATH = "/kaggle/input/food-photos/pizza.jpg"  # Adjust path to match your dataset structure
Image(filename=IMAGE_PATH, height=250, width=250)



# This prompt template forces the model into the desired persona and output structure for a text query.
user_question = "Can I eat a granola bar for breakfast?"

custom_prompt = (
    "You are the Visual & Text Analyzer within the GlucoGuide app. "
    "Your analysis must be brief, bulleted, and relentlessly focused on glucose impact. "
    "You will receive input that is either a text description of a meal, a photo, or both.\n\n"
    "Your Logic Flow:\n\n"
    "Analyze Input:\n"
    "- If photo and text are provided, use the text to clarify the items in the photo. The text is the primary source of truth.\n"
    "- If photo-only, analyze the image visually. State that portion sizes are a rough guess.\n"
    "- If text-only, base your analysis entirely on the user's description.\n"
    "- Handle Vague Text: If a user's text is vague (e.g., \"sandwich\"), you must state your assumptions (e.g., \"Assuming a turkey sandwich on white bread...\") to perform the analysis. Default to standard portion sizes and mention that you are doing so.\n\n"
    "Generate Response: Regardless of input type, your output format is always the same:\n"
    "- Lead with the Prediction: Start immediately with the predicted glucose impact.\n"
    "- Explain the \"Why\": Identify the glucose-driving foods and the estimated carb count.\n"
    "- Identify Balancers: Note any protein, fat, or fiber that will help slow glucose absorption.\n"
    "- Provide Glucose Management Tips: Offer actionable tips to manage the glucose response.\n"
    "- Disclaimer: Conclude with the mandatory short disclaimer."
    f"Question: {user_question}"
)


# Prepare the inputs for the model
inputs = tokenizer(custom_prompt, return_tensors="pt").to(model.device)

# Generate the response
generation_config = GenerationConfig(max_new_tokens=150, do_sample=True, temperature=0.7)
outputs = model.generate(**inputs, generation_config=generation_config)

# Decode and print the final result
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)





# This prompt template forces the model into the desired persona and output structure for an image.
prompt_text = (
    "<Image> You are the Visual & Text Analyzer within the GlucoGuide app. "
    "Your analysis must be brief, bulleted, and relentlessly focused on glucose impact. "
    "You will receive input that is either a text description of a meal, a photo, or both.\n\n"
    "Your Logic Flow:\n\n"
    "Analyze Input:\n"
    "- If photo and text are provided, use the text to clarify the items in the photo. The text is the primary source of truth.\n"
    "- If photo-only, analyze the image visually. State that portion sizes are a rough guess.\n"
    "- If text-only, base your analysis entirely on the user's description.\n"
    "- Handle Vague Text: If a user's text is vague (e.g., \"sandwich\"), you must state your assumptions (e.g., \"Assuming a turkey sandwich on white bread...\") to perform the analysis. Default to standard portion sizes and mention that you are doing so.\n\n"
    "Generate Response: Regardless of input type, your output format is always the same:\n"
    "- Lead with the Prediction: Start immediately with the predicted glucose impact.\n"
    "- Explain the \"Why\": Identify the glucose-driving foods and the estimated carb count.\n"
    "- Identify Balancers: Note any protein, fat, or fiber that will help slow glucose absorption.\n"
    "- Provide Glucose Management Tips: Offer actionable tips to manage the glucose response.\n"
    "- Disclaimer: Conclude with the mandatory short disclaimer."
)


import kagglehub
import transformers
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# Set up the processor and the image-to-text model
processor = AutoProcessor.from_pretrained(GEMMA_PATH)
model = AutoModelForImageTextToText.from_pretrained(GEMMA_PATH, torch_dtype="auto", device_map="auto")

# Prepare the messages for the model using the image and the custom prompt
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": IMAGE_URL},
            {"type": "text", "text": prompt_text}
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

# Generate a response
outputs = model.generate(**inputs, max_new_tokens=512, disable_compile=True)

# Decode and print the result
text = processor.batch_decode(
    outputs[:, input_len:],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True
)
print(text[0])


# Prepare the messages for the model using the image and the custom prompt
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": IMAGE_PATH},
            {"type": "text", "text": prompt_text}
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

# Generate a response
outputs = model.generate(**inputs, max_new_tokens=512, disable_compile=True)

# Decode and print the result
text = processor.batch_decode(
    outputs[:, input_len:],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True
)
print(text[0])


import base64
from IPython.display import Image, display

def mermaid_graph(graph, scale=2):
    graphbytes = graph.encode("ascii")
    base64_bytes = base64.b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    display(Image(url=f"https://mermaid.ink/img/{base64_string}"))

mermaid_graph("""
graph LR;
    A[Start] --> B[User Input]

    subgraph "Step 1: Input Processing"
        B --> C1{Input Type}
        C1 -->|Text| D1[Text Prompt Template]
        C1 -->|Image| D2[Image Prompt Template]
    end

    subgraph "Step 2: AI Analysis"
        D1 --> E[Gemma 3n Model]
        D2 --> E
        E --> F{Output Structure Check}
        F -->|Valid| G1[Verdict]
        F -->|Invalid| FX1[Fix with Template Rules]
        FX1 --> F
    end

    subgraph "Step 3: Structured Output"
        G1 --> G2[Explanation]
        G2 --> G3[Recommendation]
        G3 --> H[Display to User]
    end

    H --> I[End]

    %% Styling
    style D1 fill:#b3d9ff,stroke:#333,stroke-width:1px
    style D2 fill:#b3d9ff,stroke:#333,stroke-width:1px
    style E fill:#b3d9ff,stroke:#333,stroke-width:1px
    style F fill:#f9f,stroke:#333,stroke-width:1px
    style FX1 fill:#ccffcc,stroke:#333,stroke-width:1px
    style G1 fill:#eee,stroke:#333,stroke-width:1px
    style G2 fill:#eee,stroke:#333,stroke-width:1px
    style G3 fill:#eee,stroke:#333,stroke-width:1px
    style H fill:#e0e0ff,stroke:#333,stroke-width:1px
""")


import base64
from IPython.display import Image, display

def mermaid_graph(graph, scale=2):
    graphbytes = graph.encode("utf-8")  # Changed from 'ascii' to 'utf-8'
    base64_bytes = base64.b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    display(Image(url=f"https://mermaid.ink/img/{base64_string}"))

mermaid_graph("""
graph TD
    subgraph "Image Input Flow"
        A[User Attaches Image] --> B[Image Preview Shows]
        B --> C[User Sends Message with Image]
        C --> D_img[Processing Animation Starts]

        subgraph "Processing UI (Image)"
            D_img --> E[ImageProcessingIndicator Component]
            E --> F["📷 Rotating Camera Icon"]
            E --> G["'Analyzing image' Text"]
            E --> H["● ● ● Pulsing Dots"]
        end

        D_img --> I[isProcessingImage State = true]

        subgraph "Backend Processing (Image)"
            I --> J[MediaPipe Vision Analysis]
            I --> K[LLM Processing]
            J --> L[Analysis Complete]
            K --> L
        end

        L --> M[isProcessingImage State = false]
    end


    subgraph "Text Input Flow"
        A_txt[User Enters Text] --> C_txt[User Sends Message]
        C_txt --> D_txt[Processing Animation Starts]

        subgraph "Processing UI (Text)"
            D_txt --> G_txt["'Thinking...' Text"]
            D_txt --> H_txt["● ● ● Pulsing Dots"]
        end

        D_txt --> I_txt[isProcessingText State = true]

        subgraph "Backend Processing (Text)"
            I_txt --> K_txt[LLM Processing]
            K_txt --> L_txt[Processing Complete]
        end

        L_txt --> M_txt[isProcessingText State = false]
    end

    subgraph "Final Output (Shared)"
        M --> N[Animation Stops]
        M_txt --> N
        N --> O[Response Displayed]
    end



""")


