import os
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pandas as pd
import csv
from IPython.display import HTML, display


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_name = "openai/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)


# Function to load animal classes from a file
def load_animal_classes(file_path):
    animal_classes = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            animal_name = ' '.join(parts[1:])  # Handle multi-word class names
            animal_classes.append(animal_name)
    return animal_classes


# cell for genrating text prompt from multiple templates 
def generate_text_prompts(animal_classes):
    templates = [
        "A photo of a {}.",
        "A realistic photo of a {}.",
        "An image of a {} in the wild.",
        "A {} in its natural habitat.",
        "A close-up of a {}."
    ]
    text_prompts = [template.format(animal) for animal in animal_classes for template in templates]
    return text_prompts


# Function to normalize embeddings
def normalize_embeddings(embeddings):
    return embeddings / embeddings.norm(dim=-1, keepdim=True)


# Function to classify test images
def classify_images(test_image_dir, animal_classes, text_embeddings):
    test_images = [
        os.path.join(test_image_dir, f)
        for f in os.listdir(test_image_dir)
        if f.endswith(".jpg")
    ]
    print(f"Found {len(test_images)} images for classification.")
    results = []

    for img_path in test_images:
        image = Image.open(img_path).convert("RGB")
        with torch.no_grad():
            # these lines preprocess the images which clip does according to it and then does image encoding
            image_inputs = processor(images=image, return_tensors="pt").to(device)
            image_embeddings = model.get_image_features(**image_inputs)
            image_embeddings = normalize_embeddings(image_embeddings)

            # these lines calculate similiarity scores 
            similarity_scores = image_embeddings @ text_embeddings.T
            best_match_idx = similarity_scores.argmax(dim=-1).item()
            predicted_class = animal_classes[best_match_idx]
            results.append({"Image Path": img_path, "Predicted Class": predicted_class})

            print(f"Image: {os.path.basename(img_path)}, Predicted Class: {predicted_class}")
    return results


# this cell saves result to csv file 
def save_results_to_csv(results, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Path', 'Predicted Class'])
        for result in results:
            writer.writerow([result['Image Path'], result['Predicted Class']])
    print(f"Classification results saved to {output_file}")


# this cell deals with sorting the csv file 
def sort_csv(input_file, output_file):
    df = pd.read_csv(input_file)
    df_sorted = df.sort_values(by='Image Path')
    df_sorted.to_csv(output_file, index=False)
    print(f"Sorted CSV file saved to {output_file}")


# Main function
if __name__ == '__main__':
     classes_file_path = "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/classes.txt"
    test_image_dir = "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/test"
    output_csv_file = 'predictions.csv'
    sorted_csv_file = 'sorted_predictions.csv'

    
    animal_classes = load_animal_classes(classes_file_path)
    print(f"Loaded {len(animal_classes)} animal classes.")

    # using function to generate text prompt and therefore gets text embeddings
    text_prompts = generate_text_prompts(animal_classes)
    with torch.no_grad():
        text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True).to(device)
        text_embeddings = model.get_text_features(**text_inputs)
        text_embeddings = normalize_embeddings(text_embeddings)

        # aggregating embeddings
        num_templates = len(text_prompts) // len(animal_classes)
        aggregated_embeddings = text_embeddings.view(len(animal_classes), num_templates, -1).mean(dim=1)
        aggregated_embeddings = normalize_embeddings(aggregated_embeddings)
        
    results = classify_images(test_image_dir, animal_classes, aggregated_embeddings)

    save_results_to_csv(results, output_csv_file)
    sort_csv(output_csv_file, sorted_csv_file)

    print(f"Classification completed. Results saved to {sorted_csv_file}.")


# this cell creates downloading link for the csv file 
file_path = sorted_csv_file
download_link = f'<a href="{file_path}" download>Download sorted_predictions.csv</a>'
display(HTML(download_link))

