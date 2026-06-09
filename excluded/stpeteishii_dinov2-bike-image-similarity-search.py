


!pip install protobuf==3.20.3


import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import torch
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("          DINOv2 IMAGE SIMILARITY SEARCH - QUERY MODE")
print("="*80)
print("\nPIPELINE:")
print("1. Load query image and extract DINOv2 features")
print("2. Load all images from folder and extract features")
print("3. Calculate cosine similarity between query and all images")
print("4. Display Top 5 most similar images")
print("="*80)
print()


class DINOv2SimilaritySearch:
    def __init__(self, model_name='facebook/dinov2-base'):
        """Initialize DINOv2 model for feature extraction"""
        print(f"Loading DINOv2 model: {model_name}...")
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        print("✓ Model loaded successfully\n")
        
    def extract_feature(self, image_path):
        """
        Extract 768-dimensional feature vector from image using [CLS] token
        """
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Extract [CLS] token (global image representation)
            cls_token = outputs.last_hidden_state[0, 0, :].numpy()
            return cls_token, image
        except Exception as e:
            print(f"✗ Error processing {image_path}: {e}")
            return None, None
    
    def search_similar_images(self, query_image_path, folder_path, top_k=5):
        """
        Find top-k images most similar to query image
        
        Args:
            query_image_path: Path to query image
            folder_path: Folder containing database images
            top_k: Number of top similar images to return
        """
        print("="*80)
        print(f"QUERY IMAGE: {Path(query_image_path).name}")
        print("="*80)
        
        # Extract query features
        print("\n[1/4] Extracting query image features...")
        query_feature, query_image = self.extract_feature(query_image_path)
        if query_feature is None:
            print("Error: Could not process query image")
            return
        print(f"✓ Query feature shape: {query_feature.shape}")
        
        # Load database images
        print(f"\n[2/4] Loading images from: {folder_path}")
        folder = Path(folder_path)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(folder.glob(f'*{ext}'))
            image_paths.extend(folder.glob(f'*{ext.upper()}'))
        
        image_paths = sorted(set(image_paths))
        
        # Remove query image from database if it exists in the folder
        query_path = Path(query_image_path)
        image_paths = [p for p in image_paths if p.resolve() != query_path.resolve()]
        
        if len(image_paths) == 0:
            print("Error: No images found in folder")
            return
        
        print(f"✓ Found {len(image_paths)} images in database")
        
        # Extract features from all database images
        print("\n[3/4] Extracting features from database images...")
        db_features = []
        db_images = []
        valid_paths = []
        
        for i, img_path in enumerate(image_paths):
            print(f"  Processing [{i+1}/{len(image_paths)}]: {img_path.name}", end='\r')
            feature, image = self.extract_feature(img_path)
            
            if feature is not None:
                db_features.append(feature)
                db_images.append(image)
                valid_paths.append(img_path)
        
        print(f"\n✓ Successfully processed {len(db_features)} database images")
        
        if len(db_features) == 0:
            print("Error: No valid images found in database")
            return
        
        db_features = np.array(db_features)
        
        # Calculate similarities
        print("\n[4/4] Calculating cosine similarities...")
        query_feature = query_feature.reshape(1, -1)
        similarities = cosine_similarity(query_feature, db_features)[0]
        
        # Get top-k most similar
        top_k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        print(f"✓ Found top {top_k} similar images\n")
        
        # Display results
        self._visualize_results(query_image, query_path, 
                               db_images, valid_paths, 
                               similarities, top_indices, top_k)
        
        # Print detailed results
        self._print_results(valid_paths, similarities, top_indices)
        
        return top_indices, similarities[top_indices], valid_paths
    
    def _visualize_results(self, query_image, query_path, 
                          db_images, db_paths, similarities, top_indices, top_k):
        """Visualize query image and top-k results"""
        
        # Create figure with query + top-k results
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('DINOv2 Image Similarity Search - Top 5 Results', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        # Display query image in first position
        axes[0].imshow(query_image)
        axes[0].set_title('QUERY IMAGE\n' + query_path.name, 
                         fontsize=13, fontweight='bold', 
                         color='white', backgroundcolor='red', pad=10)
        axes[0].axis('off')
        axes[0].set_facecolor('#f0f0f0')
        
        # Display top-k results
        for i, idx in enumerate(top_indices):
            ax = axes[i+1]
            ax.imshow(db_images[idx])
            
            # Color code by similarity
            sim = similarities[idx]
            if sim >= 0.9:
                color = 'green'
                label = 'Very Similar'
            elif sim >= 0.8:
                color = 'yellowgreen'
                label = 'Similar'
            elif sim >= 0.7:
                color = 'orange'
                label = 'Moderately Similar'
            else:
                color = 'orangered'
                label = 'Less Similar'
            
            title = f'RANK #{i+1} - {label}\n{db_paths[idx].name}\nSimilarity: {sim:.4f}'
            ax.set_title(title, fontsize=11, fontweight='bold', 
                        color='white', backgroundcolor=color, pad=10)
            ax.axis('off')
            ax.set_facecolor('#f0f0f0')
        
        plt.tight_layout()
        
        # Save result
        output_path = 'similarity_search_results.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Results saved to: {output_path}")
        plt.show()
    
    def _print_results(self, db_paths, similarities, top_indices):
        """Print detailed similarity scores"""
        print("\n" + "="*80)
        print("                         SIMILARITY RANKING")
        print("="*80)
        print(f"{'Rank':<6} {'Similarity':<12} {'Image Name':<50}")
        print("-"*80)
        
        for i, idx in enumerate(top_indices):
            sim = similarities[idx]
            name = db_paths[idx].name
            
            # Add visual indicator
            if sim >= 0.9:
                indicator = "★★★★★"
            elif sim >= 0.8:
                indicator = "★★★★☆"
            elif sim >= 0.7:
                indicator = "★★★☆☆"
            elif sim >= 0.6:
                indicator = "★★☆☆☆"
            else:
                indicator = "★☆☆☆☆"
            
            print(f"#{i+1:<5} {sim:<12.6f} {name:<50} {indicator}")
        
        print("="*80)
        print(f"\nSimilarity Statistics:")
        print(f"  • Highest similarity: {similarities[top_indices[0]]:.6f}")
        print(f"  • Lowest similarity:  {similarities[top_indices[-1]]:.6f}")
        print(f"  • Average similarity: {np.mean(similarities[top_indices]):.6f}")
        print(f"  • Total database size: {len(similarities)} images")
        print("="*80)


# ==================== MAIN USAGE ====================

def main():
    """
    Main function to run similarity search
    """
    # ========== CONFIGURATION ==========
    # Set your paths here
    DATABASE_FOLDER = "/kaggle/input/image-matching-challenge-2025/train/imc2023_haiper"   
    QUERY_IMAGE = F"{DATABASE_FOLDER}/bike_image_004.png"        
   
    TOP_K = 5                         # Number of similar images to find
    # ===================================
    
    # Check if paths exist
    if not os.path.exists(QUERY_IMAGE):
        print(f"❌ Error: Query image not found: {QUERY_IMAGE}")
        print("\nPlease update QUERY_IMAGE path in the code.")
        print("Example: QUERY_IMAGE = '/path/to/your/query.jpg'")
        return
    
    if not os.path.exists(DATABASE_FOLDER):
        print(f"❌ Error: Database folder not found: {DATABASE_FOLDER}")
        print("\nPlease update DATABASE_FOLDER path in the code.")
        print("Example: DATABASE_FOLDER = '/path/to/your/images'")
        return
    
    # Run similarity search
    searcher = DINOv2SimilaritySearch()
    results = searcher.search_similar_images(
        query_image_path=QUERY_IMAGE,
        folder_path=DATABASE_FOLDER,
        top_k=TOP_K
    )
    
    if results:
        print("\n✓ Search completed successfully!")


# ==================== ALTERNATIVE: DIRECT FUNCTION ====================

def find_similar_images(query_image_path, folder_path, top_k=5):
    """
    Quick function to find similar images
    
    Usage:
        find_similar_images("my_cat.jpg", "./animal_images", top_k=5)
    """
    searcher = DINOv2SimilaritySearch()
    return searcher.search_similar_images(query_image_path, folder_path, top_k)


if __name__ == "__main__":
    main()
    
    # Or use the quick function:
    # find_similar_images("query.jpg", "./images", top_k=5)

