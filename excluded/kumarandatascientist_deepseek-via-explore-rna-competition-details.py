!cp /kaggle/input/deepseekkagglecompetitionanalyst/other/default/1/* /kaggle/working


import sys
sys.path.append('/kaggle/working/')


from DeepseekKaggleCompetitionPipeline import DeepseekKaggleCompetitionPipeline


# Example input for a Kaggle competition
competition_input = {
    "name": "Stanford RNA 3D Folding",
    "description": "Solve RNA structure prediction, one of biology's remaining grand challenges. RNA is vital to life’s most essential processes, but despite its significance, predicting its 3D structure is still difficult. Deep learning breakthroughs like AlphaFold have transformed protein structure prediction, but progress with RNA has been much slower due to limited data and evaluation methods. This competition builds on recent advances, like the deep learning foundation model RibonanzaNet, which emerged from a prior Kaggle competition. Now, you’ll take on the next challenge—predicting RNA’s full 3D structure. Your work could push RNA-based medicine forward, making treatments like cancer immunotherapies and CRISPR gene editing more accessible and effective. More fundamentally, your work may be the key step in illuminating the folds and functions of natural RNA molecules, which have been called the 'dark matter of biology'. This competition is made possible through a worldwide collaborative effort including the organizers, experimental RNA structural biologists, and predictors of the CASP16 and RNA-Puzzles competitions; Howard Hughes Medical Institute; the Institute of Protein Design; and Stanford University School of Medicine.",
    "further_details": (
        "share learning materials and references for this competitions"
    )
}

# Initialize the Kaggle competition pipeline
model_folder = "/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-7b/1"  # Update with your actual model path
pipeline = DeepseekKaggleCompetitionPipeline(model_folder=model_folder)

# Generate the detailed competition analysis in multiple formats
analysis = pipeline.generate_competition_analysis(competition_input)

# Display the outputs

# 1. Plain Text Response
print("Plain Text Output:\n")
print(analysis["text"])
print("\n" + "-"*60 + "\n")

# 2. Markdown Response
print("Markdown Output:\n")
print(analysis["markdown"])
print("\n" + "-"*60 + "\n")

# 3. JSON Response
import json
print("JSON Output:\n")
print(json.dumps(analysis["json"], indent=2))




