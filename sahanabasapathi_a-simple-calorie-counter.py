!pip install timm==1.0.17
!pip install transformers==4.53.2


import kagglehub

GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")


from transformers import AutoProcessor, AutoModelForImageTextToText
from IPython.display import Image


IMAGE_URL="https://static01.nyt.com/images/2022/06/27/dining/kc-mushroom-beef-burgers/merlin_209008674_b3fa58fa-9bb1-4cfe-a08a-40b4dda0de78-threeByTwoMediumAt2X.jpg?quality=75&auto=webp"
Image(url=IMAGE_URL,height=500,width=500)


def prompt():
    return f'''
            
        You are a personal health advisor. You are given an image of a prepared food item. Your goal is to analyze the image and provide detailed information under the following categories:
        
        Calorie Estimate:
        
        Provide an estimated total calorie count or a reasonable calorie range for the food shown.
        Base the estimate on standard portion sizes and common preparation methods.
        Mention assumptions made about ingredients, cooking methods (e.g., fried, baked), and portion size.
        
        Likely Ingredients:
        
        Identify as many individual ingredients as can be reasonably inferred from the visual features of the dish.
        Group them into categories if helpful (e.g., vegetables, proteins, sauces, grains).
        Note any ingredients that are likely but not visually identifiable, based on common recipes for similar dishes.
        
        Potential Allergens:
        List possible allergens commonly associated with the detected or assumed ingredients.
        Use standard allergen categories: gluten, dairy, eggs, nuts, soy, shellfish, fish, sesame, and mustard.
        
        Mention uncertainty if visual confirmation is difficult.
        
        Output Format:
        
        Calorie Estimate: [e.g., 350–450 kcal]
        
        Likely Ingredients:
        - [Ingredient 1]
        - [Ingredient 2]
        - ...
        
        Potential Allergens:
        - [Allergen 1]
        - [Allergen 2]
        - ...
        
        Notes:
        - [Any assumptions or uncertainties based on visual interpretation]
        Provide reasoning where appropriate and keep the analysis grounded in realistic, evidence-based assumptions about the food.
    '''


processor = AutoProcessor.from_pretrained(GEMMA_PATH)
model = AutoModelForImageTextToText.from_pretrained(GEMMA_PATH, torch_dtype="auto", device_map="auto")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": IMAGE_URL},
            {"type": "text", "text": prompt()}
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


print(text[0])




