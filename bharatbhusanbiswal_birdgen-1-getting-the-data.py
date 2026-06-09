from pathlib import Path
import shutil
import random
import pandas as pd

root_dir = Path('/kaggle/input/birdsong-recognition/train_audio')
metadata_csv = Path('/kaggle/input/birdsong-recognition/train.csv')
assert root_dir.exists(), 'Invalid audio path'
assert metadata_csv.exists(), 'Invalid metadata path'

num_examples = 100 
subset_dir = Path(f'bird_call_filtered_{num_examples}')
subset_dir.mkdir(exist_ok=True)

df = pd.read_csv(metadata_csv)

df = df.query("rating >= 4")

species_counts = df['ebird_code'].value_counts()

selected_species = species_counts[species_counts >= num_examples].index.tolist()

print(f"Selected {len(selected_species)} species with >= {num_examples} high-rated recordings")

# Copy files
for species in selected_species:
    species_dir = root_dir / species
    if not species_dir.exists():
        continue
    
    audio_files = list(species_dir.glob('*.mp3'))
    
    audio_files = [f for f in audio_files if f.stem in df.query("ebird_code == @species")['filename'].values]
    
    if len(audio_files) >= num_examples:
        dest_dir = subset_dir / species
        dest_dir.mkdir(exist_ok=True)
        
        selected_files = random.sample(audio_files, num_examples)
        
        for f in selected_files:
            shutil.copy2(f, dest_dir / f.name)
        
        print(f"✓ Copied {num_examples} files for species {species}")



num_categories = 20
examples_per_category = 100
subset_dir = Path(f"/kaggle/working/bird_call_{num_categories}_{examples_per_category}")

zip_path = f"/kaggle/working/bird_call_{num_categories}_{examples_per_category}-new"

shutil.make_archive(zip_path, 'zip', subset_dir)

print(f"Archive created: {zip_path}.zip")





