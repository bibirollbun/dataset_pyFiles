import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import feature, color
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
import joblib
import pandas as pd


def load_image_subset(data_dir, n_cats=100, n_dogs=100):
    data_path = Path(data_dir)
    images = []
    labels = []
    filenames = []
    
    cat_files = sorted([f for f in os.listdir(data_path) if f.startswith('cat.')])[:n_cats]
    print(f"Loading {len(cat_files)} cat images...")
    
    for filename in cat_files:
        img = Image.open(data_path / filename)
        img_array = np.array(img)
        
        if len(img_array.shape) == 2:
            continue
        
        images.append(img_array)
        labels.append(0)
        filenames.append(filename)
    
    dog_files = sorted([f for f in os.listdir(data_path) if f.startswith('dog.')])[:n_dogs]
    print(f"Loading {len(dog_files)} dog images...")
    
    for filename in dog_files:
        img = Image.open(data_path / filename)
        img_array = np.array(img)
        
        if len(img_array.shape) == 2:
            continue
        
        images.append(img_array)
        labels.append(1)
        filenames.append(filename)
    
    print(f"Loaded {len(images)} images total")
    return images, labels, filenames


def extract_rgb_histogram(img_array, bins=256):
    r_hist, _ = np.histogram(img_array[:, :, 0], bins=bins, range=(0, 256))
    g_hist, _ = np.histogram(img_array[:, :, 1], bins=bins, range=(0, 256))
    b_hist, _ = np.histogram(img_array[:, :, 2], bins=bins, range=(0, 256))
    
    total_pixels = img_array.shape[0] * img_array.shape[1]
    features = np.concatenate([r_hist / total_pixels, g_hist / total_pixels, b_hist / total_pixels])
    
    return features.astype(np.float32)

def extract_edge_features(img_array, sigma=1.0, grid_size=10):
    if len(img_array.shape) == 3:
        gray = color.rgb2gray(img_array)
    else:
        gray = img_array
    
    edges = feature.canny(gray, sigma=sigma)
    h, w = edges.shape
    cell_h = h // grid_size
    cell_w = w // grid_size
    
    grid_densities = []
    for i in range(grid_size):
        for j in range(grid_size):
            cell = edges[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            density = cell.sum() / cell.size if cell.size > 0 else 0.0
            grid_densities.append(density)
    
    overall_density = edges.sum() / edges.size
    edge_concentration = np.std(grid_densities)
    features = np.array(grid_densities + [overall_density, edge_concentration])
    
    return features.astype(np.float32)

def extract_lbp_features(img_array, n_points=8, radius=1, bins=256):
    if len(img_array.shape) == 3:
        gray = color.rgb2gray(img_array)
    else:
        gray = img_array
    
    lbp = feature.local_binary_pattern(gray, n_points, radius, method='default')
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, 256), density=True)
    
    return hist.astype(np.float32)


train_dir = "../data/train"
images, labels, filenames = load_image_subset(train_dir, n_cats=100, n_dogs=100)

print(f"\nDataset: {len(images)} images")
print(f"Sample shape: {images[0].shape}")


print("Extracting features...")
rgb_features = []
edge_features = []
lbp_features = []

for i, img in enumerate(images):
    rgb_features.append(extract_rgb_histogram(img))
    edge_features.append(extract_edge_features(img))
    lbp_features.append(extract_lbp_features(img))
    
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(images)}")

X_rgb = np.array(rgb_features)
X_edge = np.array(edge_features)
X_lbp = np.array(lbp_features)
X_combined = np.hstack([X_rgb, X_edge, X_lbp])
y = np.array(labels)

print(f"\nCombined features shape: {X_combined.shape}")


features_dir = Path("../data/features")
features_dir.mkdir(exist_ok=True)

np.save(features_dir / "rgb_histogram_features.npy", X_rgb)
np.save(features_dir / "edge_features.npy", X_edge)
np.save(features_dir / "lbp_features.npy", X_lbp)
np.save(features_dir / "labels.npy", y)

print(f"Saved to {features_dir}")


print(f"Feature shape: {X_combined.shape}")
print(f"Expected: (200, 1126)")
print(f"\nRGB sum per image: {X_rgb.sum(axis=1).mean():.4f}")
print(f"LBP sum per image: {X_lbp.sum(axis=1).mean():.4f}")
print(f"\nLabel split: {(y==0).sum()} cats, {(y==1).sum()} dogs")


rf = RandomForestClassifier(random_state=42, n_jobs=-1)

print("Running 5-fold CV...")
cv_results = cross_validate(
    rf, X_combined, y, cv=5,
    scoring=['accuracy', 'neg_log_loss'],
    return_train_score=False
)

acc = cv_results['test_accuracy']
loss = -cv_results['test_neg_log_loss']

print(f"\nAccuracy: {acc.mean():.4f} ± {acc.std():.4f}")
print(f"Log Loss: {loss.mean():.4f} ± {loss.std():.4f}")


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

print("Grid search...")
grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
)

grid.fit(X_combined, y)

print(f"\nBest accuracy: {grid.best_score_:.4f}")
print(f"Best params: {grid.best_params_}")


best_rf = grid.best_estimator_

cv_best = cross_validate(
    best_rf, X_combined, y, cv=5,
    scoring=['accuracy', 'neg_log_loss'],
    return_train_score=False
)

acc_best = cv_best['test_accuracy']
loss_best = -cv_best['test_neg_log_loss']

print(f"Best model accuracy: {acc_best.mean():.4f} ± {acc_best.std():.4f}")
print(f"Best model log loss: {loss_best.mean():.4f} ± {loss_best.std():.4f}")


images_full, labels_full, filenames_full = load_image_subset(train_dir, n_cats=12500, n_dogs=12500)

print(f"\nDataset: {len(images_full)} images")


print("Extracting features from full dataset...")
rgb_features_full = []
edge_features_full = []
lbp_features_full = []

for i, img in enumerate(images_full):
    rgb_features_full.append(extract_rgb_histogram(img))
    edge_features_full.append(extract_edge_features(img))
    lbp_features_full.append(extract_lbp_features(img))
    
    if (i + 1) % 1000 == 0:
        print(f"  {i + 1}/{len(images_full)}")

X_rgb_full = np.array(rgb_features_full)
X_edge_full = np.array(edge_features_full)
X_lbp_full = np.array(lbp_features_full)
X_combined_full = np.hstack([X_rgb_full, X_edge_full, X_lbp_full])
y_full = np.array(labels_full)

print(f"\nCombined features shape: {X_combined_full.shape}")


np.save(features_dir / "full_rgb_histogram_features.npy", X_rgb_full)
np.save(features_dir / "full_edge_features.npy", X_edge_full)
np.save(features_dir / "full_lbp_features.npy", X_lbp_full)
np.save(features_dir / "full_labels.npy", y_full)

print(f"Saved full features to {features_dir}")


print(f"Feature shape: {X_combined_full.shape}")
print(f"Expected: (25000, 1126)")
print(f"\nLabel split: {(y_full==0).sum()} cats, {(y_full==1).sum()} dogs")


print("Training final model on full dataset...")
print(f"Using best params from phase 1: {grid.best_params_}")

final_model = RandomForestClassifier(
    n_estimators=grid.best_params_['n_estimators'],
    max_depth=grid.best_params_['max_depth'],
    min_samples_split=grid.best_params_['min_samples_split'],
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_combined_full, y_full)

models_dir = Path("../models")
models_dir.mkdir(exist_ok=True)
model_path = models_dir / "random_forest_final.pkl"

joblib.dump(final_model, model_path)
print(f"\nModel saved to {model_path}")


def load_test_images(data_dir):
    data_path = Path(data_dir)
    test_files = sorted([f for f in os.listdir(data_path) if f.endswith('.jpg')], 
                       key=lambda x: int(x.split('.')[0]))
    
    print(f"Processing {len(test_files)} test images...")
    
    test_ids = []
    rgb_feats = []
    edge_feats = []
    lbp_feats = []
    
    for i, filename in enumerate(test_files):
        img_id = int(filename.split('.')[0])
        test_ids.append(img_id)
        
        img = Image.open(data_path / filename)
        img_array = np.array(img)
        
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        
        rgb_feats.append(extract_rgb_histogram(img_array))
        edge_feats.append(extract_edge_features(img_array))
        lbp_feats.append(extract_lbp_features(img_array))
        
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(test_files)}")
    
    X_test = np.hstack([np.array(rgb_feats), np.array(edge_feats), np.array(lbp_feats)])
    print(f"\nTest features shape: {X_test.shape}")
    
    return test_ids, X_test

test_dir = "../data/test"
test_ids, X_test = load_test_images(test_dir)

np.save(features_dir / "test_features.npy", X_test)
np.save(features_dir / "test_ids.npy", np.array(test_ids))


print("Generating predictions...")

predictions = final_model.predict_proba(X_test)
dog_probs = predictions[:, 1]

submission = pd.DataFrame({
    'id': test_ids,
    'label': dog_probs
})

submission = submission.sort_values('id').reset_index(drop=True)
submission_path = Path("../submission.csv")
submission.to_csv(submission_path, index=False)

print(f"Submission saved to {submission_path}")
print(f"\nPreview:")
print(submission.head(10))

