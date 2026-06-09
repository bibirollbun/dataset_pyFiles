import pandas as pd

# Load the training data (assuming the CSV contains a 'description' column)
train_csv_path = '/kaggle/input/drawing-with-llms/train.csv'
train_df = pd.read_csv(train_csv_path)
print("Training Data Overview:")
print(train_df.head())
print("\nTotal training samples:", len(train_df))


def preprocess_text(text: str) -> str:
    """
    Simple preprocessing: remove leading/trailing whitespace and convert to lowercase.
    """
    return text.strip().lower()

# Apply preprocessing to the description column and create a new column.
train_df['clean_description'] = train_df['description'].apply(preprocess_text)
print(train_df[['description', 'clean_description']].head())


class Model:
    def __init__(self):
        # Initialize any parameters or load resources if needed.
        pass

    def predict(self, prompt: str) -> str:
        """
        Generate SVG code from a text prompt.
        
        This baseline uses simple keyword detection to decide which SVG shape to render.
        """
        prompt_lower = prompt.lower()
        
        # SVG canvas header and footer
        svg_header = '<svg width="300" height="300" xmlns="http://www.w3.org/2000/svg">'
        svg_footer = '</svg>'
        svg_content = ""
        
        # Determine the SVG content based on keywords in the prompt.
        if "circle" in prompt_lower:
            svg_content = '<circle cx="150" cy="150" r="50" fill="red" />'
        elif "square" in prompt_lower or "rectangle" in prompt_lower:
            svg_content = '<rect x="100" y="100" width="100" height="100" fill="blue" />'
        elif "triangle" in prompt_lower:
            svg_content = '<polygon points="150,50 100,200 200,200" fill="green" />'
        else:
            # Fallback: display the prompt as centered text.
            svg_content = (
                f'<text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" '
                f'font-size="20">{prompt}</text>'
            )
        
        svg_code = f"{svg_header}{svg_content}{svg_footer}"
        
        # Enforce SVG size limit (10,000 bytes max)
        if len(svg_code.encode('utf-8')) > 10000:
            svg_code = svg_code.encode('utf-8')[:10000].decode('utf-8', errors='ignore')
        
        return svg_code


# Instantiate the model
model = Model()

# Sample prompts for testing
sample_prompts = [
    "A red circle",
    "A blue square",
    "A green triangle",
    "Abstract shapes"
]

for prompt in sample_prompts:
    svg_output = model.predict(prompt)
    print(f"Prompt: {prompt}")
    print(svg_output)
    print("-" * 40)


# Load the test data (assuming a 'description' column exists)
test_csv_path = '/kaggle/input/drawing-with-llms/kaggle_evaluation/test.csv'
test_df = pd.read_csv(test_csv_path)
print("Test Data Overview:")
print(test_df.head())

# Generate SVG predictions for the test data.
test_df['svg'] = test_df['description'].apply(lambda x: model.predict(x))
print("\nSample predictions:")
print(test_df[['description', 'svg']].head())


# Prepare the submission DataFrame.
if 'id' in test_df.columns:
    submission_df = test_df[['id', 'svg']]
else:
    submission_df = test_df.reset_index()[['index', 'svg']].rename(columns={'index': 'id'})

# Save the submission file.
submission_filename = 'submission.csv'
submission_df.to_csv(submission_filename, index=False)
print(f"Submission file saved as {submission_filename}")


import kagglehub
package = kagglehub.package_import('dster/drawing-with-llms-starter-notebook/versions/3')
print("Kaggle Package imported successfully.")

