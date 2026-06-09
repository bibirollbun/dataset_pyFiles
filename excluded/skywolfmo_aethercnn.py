pip install nmn


pip install -U flax


# ==============================================================================
# 0. IMPORTS
# ==============================================================================
import typing as tp
import os
import glob
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow_datasets as tfds
import tensorflow as tf
import optax
import orbax.checkpoint as orbax
import pandas as pd
from tqdm import tqdm

from flax import nnx
from  nmn.nnx.yatconv import YatConv
from  nmn.nnx.nmn import YatNMN

# ==============================================================================
# 1. CONFIGURATIONS (Now defined in main)
# ==============================================================================

# Typing
Array = jax.Array
Axis = int
Size = int

# ==============================================================================
# 2. DATA LOADING & PREPROCESSING
# ==============================================================================

def create_image_folder_dataset(path: str, validation_split: float, seed: int):
    """Creates train and test tf.data.Dataset from an image folder."""
    class_names = sorted([d.name for d in os.scandir(path) if d.is_dir()])
    if not class_names:
        raise ValueError(f"No subdirectories found in {path}. Each subdirectory should contain images for one class.")
    
    class_to_index = {name: i for i, name in enumerate(class_names)}

    all_image_paths = []
    all_image_labels = []
    for class_name in class_names:
        class_dir = os.path.join(path, class_name)
        image_paths = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.JPEG'):
             image_paths.extend(glob.glob(os.path.join(class_dir, ext)))
        all_image_paths.extend(image_paths)
        all_image_labels.extend([class_to_index[class_name]] * len(image_paths))
        
    if not all_image_paths:
        raise ValueError(f"No image files found in subdirectories of {path}.")

    # Create a tf.data.Dataset
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int32))
    image_label_ds = tf.data.Dataset.zip((path_ds, label_ds))

    # Shuffle and split
    dataset_size = len(all_image_paths)
    image_label_ds = image_label_ds.shuffle(buffer_size=dataset_size, seed=seed)
    
    val_count = int(dataset_size * validation_split)
    train_ds = image_label_ds.skip(val_count)
    test_ds = image_label_ds.take(val_count)
    
    # Get dataset size for train and test
    train_size = dataset_size - val_count
    
    return train_ds, test_ds, class_names, train_size

def get_image_processor(image_size: tuple[int, int], num_channels: int):
    """Returns a tf.data processing function."""
    def process_path(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=num_channels, expand_animations=False)
        img = tf.image.resize(img, image_size)
        img = tf.cast(img, tf.float32) / 255.0
        return {'image': img, 'label': label}
    return process_path


# ==============================================================================
# 4. MODEL ARCHITECTURES
# ==============================================================================

class YatCNN(nnx.Module):
    """YAT CNN model with custom layers."""
    def __init__(self, *, num_classes: int, input_channels: int, rngs: nnx.Rngs):
        self.conv1 = YatConv(input_channels, 8, kernel_size=(7, 7), rngs=rngs)
        self.conv2 = YatConv(8, 8, kernel_size=(5, 5), rngs=rngs)
        self.conv3 = YatConv(8, 8, kernel_size=(5, 5), rngs=rngs)
        self.conv4 = YatConv(8, 8, kernel_size=(5, 5), rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=0.3, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=0.3, rngs=rngs)
        self.dropout3 = nnx.Dropout(rate=0.3, rngs=rngs)
        self.dropout4 = nnx.Dropout(rate=0.3, rngs=rngs)
        
        self.out_conv = YatConv(8, num_classes, use_bias=False, use_alpha=False, kernel_size=(7, 7), rngs=rngs)

        self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))

    def __call__(self, x, training: bool = False, return_activations_for_layer: tp.Optional[str] = None):
        activations = {}
        x = self.conv1(x)
        activations['conv1'] = x
        if return_activations_for_layer == 'conv1': return x
        x = self.dropout1(x, deterministic=not training)
        x = self.avg_pool(x)
        
        x = self.conv2(x)
        activations['conv2'] = x
        if return_activations_for_layer == 'conv2': return x
        x = self.dropout2(x, deterministic=not training)
        x = self.avg_pool(x)
        
        x = self.conv2(x)
        activations['conv3'] = x
        if return_activations_for_layer == 'conv3': return x
        x = self.dropout3(x, deterministic=not training)
        x = self.avg_pool(x)
        
        x = self.conv2(x)
        activations['conv4'] = x
        if return_activations_for_layer == 'conv4': return x
        x = self.dropout4(x, deterministic=not training)
        x = self.avg_pool(x)

        x = self.out_conv(x)
        x = jnp.mean(x, axis=(1, 2))
        activations['global_avg_pool'] = x
        if return_activations_for_layer == 'global_avg_pool': return x
        
        if return_activations_for_layer is not None and return_activations_for_layer not in activations:
            print(f"Warning: Layer '{return_activations_for_layer}' not found in YatCNN. Available: {list(activations.keys())}")
        return x


# ==============================================================================
# 5. TRAINING & EVALUATION INFRASTRUCTURE
# ==============================================================================

def loss_fn(model, batch):
  logits = model(batch['image'], training=True)
  loss = optax.softmax_cross_entropy_with_integer_labels(
    logits=logits, labels=batch['label']
  ).mean()
  return loss, logits

@nnx.jit
def train_step(model, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
  """Train for a single step."""
  grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])
  optimizer.update(grads)

@nnx.jit
def eval_step(model, metrics: nnx.MultiMetric, batch):
  loss, logits = loss_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])

def _train_model_loop(
    model_class: tp.Type[nnx.Module],
    model_name: str,
    dataset_name: str,
    rng_seed: int,
    learning_rate: float,
    momentum: float,
    optimizer_constructor: tp.Callable,
    dataset_configs: dict,
    fallback_configs: dict,
):
    """Helper function to train a model and return it with its metrics history."""
    print(f"Initializing {model_name} model for dataset {dataset_name}...")

    # Determine data source and load metadata
    if os.path.isdir(dataset_name):
        config = dataset_configs.get('custom_folder', fallback_configs)
        
        image_size = config.get('input_dim', (64, 64))
        split_percentage = config.get('test_split_percentage', 0.2)
        current_batch_size = config.get('batch_size', fallback_configs['batch_size_folder'])
        print(f"Loading data from directory: {dataset_name} using tf.data pipeline")
        
        train_ds, test_ds, class_names, train_size = create_image_folder_dataset(
            dataset_name, validation_split=split_percentage, seed=42
        )
        
        num_classes = len(class_names)
        input_channels = config.get('input_channels', 3)
        current_num_epochs = config.get('num_epochs', fallback_configs['num_epochs'])
        current_eval_every = config.get('eval_every', fallback_configs['eval_every'])
        print(f"Inferred from folder: num_classes={num_classes}. Using 'custom_folder' config for training params.")

        image_processor = get_image_processor(image_size=image_size, num_channels=input_channels)
        train_ds = train_ds.map(image_processor, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.map(image_processor, num_parallel_calls=tf.data.AUTOTUNE)
        dataset_size = train_size
    else:
        config = dataset_configs.get(dataset_name)
        if not config:
            raise ValueError(f"Dataset '{dataset_name}' not in configs and not a valid directory.")
        
        num_classes = config['num_classes']
        input_channels = config['input_channels']
        current_num_epochs = config['num_epochs']
        current_eval_every = config['eval_every']
        current_batch_size = config['batch_size']
        image_key, label_key = config['image_key'], config['label_key']
        train_split_name, test_split_name = config['train_split'], config['test_split']
        
        preprocess_fn = lambda s: {'image': tf.cast(s[image_key], tf.float32) / 255.0, 'label': s[label_key]}
        
        train_ds = tfds.load(dataset_name, split=train_split_name, as_supervised=False).map(preprocess_fn)
        test_ds = tfds.load(dataset_name, split=test_split_name, as_supervised=False).map(preprocess_fn)
        
        dataset_size = train_ds.cardinality().numpy()

    # Initialize model, optimizer, and metrics
    model = model_class(num_classes=num_classes, input_channels=input_channels, rngs=nnx.Rngs(rng_seed))
    optimizer = nnx.Optimizer(model, optimizer_constructor(learning_rate, momentum))
    metrics_computer = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average('loss'),
    )
    metrics_history = {'train_loss': [], 'train_accuracy': [], 'test_loss': [], 'test_accuracy': []}
    global_step_counter = 0

    # Training loop
    steps_per_epoch = dataset_size // current_batch_size
    total_steps = current_num_epochs * steps_per_epoch
    print(f"Training for {total_steps} steps...")

    for epoch in range(current_num_epochs):
        epoch_train_iter = train_ds.shuffle(1024).batch(current_batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
        for batch_data in epoch_train_iter:
            train_step(model, optimizer, metrics_computer, batch_data)
            
            if global_step_counter > 0 and (global_step_counter % current_eval_every == 0 or global_step_counter == total_steps - 1):
                train_metrics = metrics_computer.compute()
                metrics_history['train_loss'].append(train_metrics['loss'])
                metrics_history['train_accuracy'].append(train_metrics['accuracy'])
                metrics_computer.reset()
                
                # Create a fresh iterator for test set
                current_test_iter = test_ds.batch(current_batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
                for test_batch in current_test_iter:
                    eval_step(model, metrics_computer, test_batch)
                test_metrics = metrics_computer.compute()
                metrics_history['test_loss'].append(test_metrics['loss'])
                metrics_history['test_accuracy'].append(test_metrics['accuracy'])
                metrics_computer.reset()
                print(f"    Step {global_step_counter}: Train Acc = {train_metrics['accuracy']:.4f}, Test Acc = {test_metrics['accuracy']:.4f}")

            global_step_counter += 1
            if global_step_counter >= total_steps: break
        if global_step_counter >= total_steps: break

    # Final evaluation
    print(f"âœ… Training complete on {dataset_name} after {global_step_counter} steps!")
    if metrics_history['test_accuracy']:
        print(f"   Final Test Accuracy: {metrics_history['test_accuracy'][-1]:.4f}")
    
    # Save the model state
    save_dir = os.path.abspath(f"./models/{model_name}_{dataset_name.replace('/', '_')}")
    state = nnx.state(model)
    checkpointer = orbax.PyTreeCheckpointer()
    print(f"ğŸ’¾ Saving model state to {save_dir}...")
    checkpointer.save(save_dir, state, force=True)
    print(f"   Model saved successfully!")
    
    return model, metrics_history


# ==============================================================================
# 6. ANALYSIS & VISUALIZATION FUNCTIONS
# ==============================================================================

def plot_training_curves(history, model_name):
    """Plot training curves for a model."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'Training Curves for {model_name}', fontsize=16, fontweight='bold')
    
    steps = range(len(history['train_loss']))
    
    ax1.plot(steps, history['train_loss'], 'b-', label=f'{model_name} Train Loss', linewidth=2)
    ax1.plot(steps, history['test_loss'], 'r--', label=f'{model_name} Test Loss', linewidth=2)
    ax1.set_title('Loss'); ax1.set_xlabel('Evaluation Steps'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    
    ax2.plot(steps, history['train_accuracy'], 'b-', label=f'{model_name} Train Accuracy', linewidth=2)
    ax2.plot(steps, history['test_accuracy'], 'r--', label=f'{model_name} Test Accuracy', linewidth=2)
    ax2.set_title('Accuracy'); ax2.set_xlabel('Evaluation Steps'); ax2.set_ylabel('Accuracy')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    
    plt.tight_layout(); plt.show()
    print("ğŸ“ˆ Training curves plotted successfully!")

def print_final_metrics(history, model_name):
    """Print a detailed table of final metrics."""
    print(f"\nğŸ“Š FINAL METRICS FOR {model_name}" + "\n" + "=" * 40)
    
    final_metrics = {metric: hist[-1] for metric, hist in history.items() if hist}
    
    print(f"{'Metric':<20} {'Value':<15}")
    print("-" * 35)
    
    for metric, value in final_metrics.items():
        print(f"{metric:<20} {value:<15.4f}")
    
    print("\nğŸ�† SUMMARY:")
    print(f"   Final Test Accuracy: {final_metrics.get('test_accuracy', 0):.4f}")

def analyze_convergence(history, model_name):
    """Analyze convergence speed and stability of the model."""
    print(f"\nğŸ”� CONVERGENCE ANALYSIS FOR {model_name}" + "\n" + "=" * 50)
    
    def calculate_convergence_metrics(h):
        if not h['test_accuracy']: return {'convergence_step': 'N/A', 'stability': 'N/A', 'overfitting': 'N/A', 'final_loss': 'N/A'}
        test_acc, train_acc, test_loss = h['test_accuracy'], h['train_accuracy'], h['test_loss']
        target_acc = 0.5 * test_acc[-1]
        convergence_step = next((i for i, acc in enumerate(test_acc) if acc >= target_acc), 'N/A')
        stability = np.std(test_acc[-len(test_acc)//4:])
        overfitting = train_acc[-1] - test_acc[-1]
        return {'convergence_step': convergence_step, 'stability': stability, 'overfitting': overfitting, 'final_loss': test_loss[-1]}
    
    conv_metrics = calculate_convergence_metrics(history)
    
    print(f"{'Metric':<25} {'Value':<15}")
    print("-" * 40)
    for metric, value in conv_metrics.items():
        print(f"{metric.replace('_', ' ').title():<25} {value:<15.4f}" if isinstance(value, (int, float)) else f"{metric.replace('_', ' ').title():<25} {value:<15}")

def detailed_test_evaluation(model, test_ds_iter, class_names: list[str], model_name: str):
    """Perform detailed evaluation on test set including per-class accuracy."""
    print(f"Running detailed test evaluation for {model_name}...")
    num_classes = len(class_names)
    predictions, true_labels = [], []
    
    for batch in test_ds_iter:
        preds = jnp.argmax(model(batch['image'], training=False), axis=1)
        predictions.extend(preds.tolist())
        true_labels.extend(batch['label'].tolist())
    
    predictions, true_labels = np.array(predictions), np.array(true_labels)
    
    print("\nğŸ�¯ PER-CLASS ACCURACY" + "\n" + "=" * 50)
    print(f"{'Class':<12} {'Accuracy':<10} {'Sample Count':<12}")
    print("-" * 50)
    
    for i in range(num_classes):
        mask = true_labels == i
        if np.sum(mask) > 0:
            acc = np.mean(predictions[mask] == true_labels[mask])
            print(f"{class_names[i]:<12} {acc:<10.4f} {np.sum(mask):<12}")
        elif num_classes <= 20:
             print(f"{class_names[i]:<12} {'N/A':<10} {np.sum(mask):<12}")

    return {'predictions': predictions, 'true_labels': true_labels, 'class_names': class_names}

def plot_confusion_matrix(predictions_data, model_name):
    """Plot confusion matrix for the model."""
    cm = confusion_matrix(predictions_data['true_labels'], predictions_data['predictions'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=predictions_data['class_names'], yticklabels=predictions_data['class_names'])
    plt.title(f'{model_name} - Confusion Matrix', fontweight='bold'); plt.xlabel('Predicted Label'); plt.ylabel('True Label')
    plt.tight_layout(); plt.show()
    print("ğŸ“Š Confusion matrix plotted successfully!")

def generate_submission_file(model, test_dir, class_names, image_size, input_channels, model_name, batch_size):
    """Generates a submission.csv file for Kaggle using batch prediction."""
    print(f"\nğŸ“„ Generating submission file for {model_name} using batch prediction (batch size: {batch_size})...")
    
    # Check if test directory exists
    if not os.path.isdir(test_dir):
        print(f"âš ï¸� Test directory not found at {test_dir}. Skipping submission generation.")
        return

    # Using glob to find all images, assuming .png, can be adjusted (Eurosat is jpg)
    test_image_paths = sorted(glob.glob(os.path.join(test_dir, '*.png')))

    if not test_image_paths:
        print(f"âš ï¸� No images found in {test_dir}. Tried searching for '*.png'.")
        return

    print(f"ğŸ”� Found {len(test_image_paths)} images in the test directory. Starting predictions...")

    # Define a processor for a single test image
    def process_test_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=input_channels, expand_animations=False)
        img = tf.image.resize(img, image_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    # Create a tf.data pipeline for batching
    path_ds = tf.data.Dataset.from_tensor_slices(test_image_paths)
    image_ds = path_ds.map(process_test_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = image_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    all_pred_indices = []
    try:
        for image_batch in tqdm(test_ds.as_numpy_iterator(), desc=f"Predicting for {model_name}"):
            logits = model(image_batch, training=False)
            pred_indices = jnp.argmax(logits, axis=1)
            all_pred_indices.extend(pred_indices.tolist())
    except Exception as e:
        print(f"â�—ï¸� An error occurred during batch prediction: {e}")
        return

    if len(all_pred_indices) != len(test_image_paths):
        print(f"â�—ï¸� Prediction count ({len(all_pred_indices)}) does not match image count ({len(test_image_paths)}). Aborting.")
        return

    # Create and save submission file
    test_image_filenames = [os.path.basename(p) for p in test_image_paths]
    predictions = [
        {'Id': filename, 'Label': class_names[pred_idx]}
        for filename, pred_idx in zip(test_image_filenames, all_pred_indices)
    ]
    
    submission_df = pd.DataFrame(predictions)
    submission_df.to_csv('submission.csv', index=False)
    print(f"\nâœ… Submission file 'submission.csv' created successfully with {len(predictions)} predictions.")
    print(f"   Saved to: {os.path.abspath('submission.csv')}")

def generate_summary_report(history, model_name):
    """Generate a comprehensive summary report of the model's performance."""
    print("\n" + "="*80 + f"\n                    COMPREHENSIVE SUMMARY REPORT FOR {model_name}\n" + "="*80)
    
    acc = history['test_accuracy'][-1] if history['test_accuracy'] else 0
    print(f"\nğŸ�† Final Test Accuracy: {acc:.4f}")

def visualize_kernels(model, model_name, layer_name='conv1', num_kernels_to_show=16):
    """Visualize the kernels of a convolutional layer for the model."""
    print(f"\nğŸ�¨ VISUALIZING KERNELS FROM LAYER: {layer_name} for {model_name}")
    def get_kernels(m, layer):
        try: return getattr(m, layer).kernel.value
        except (AttributeError, KeyError): print(f"Kernel not found in layer {layer} of {m.__class__.__name__}"); return None

    kernels = get_kernels(model, layer_name)
    if kernels is not None:
        num_kernels = min(num_kernels_to_show, kernels.shape[-1])
        cols = int(np.ceil(np.sqrt(num_kernels)))
        rows = int(np.ceil(num_kernels / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        fig.suptitle(f'{model_name} - Kernels (First {num_kernels})', fontsize=16)
        for i in range(num_kernels):
            ax = axes.flat[i] if num_kernels > 1 else axes
            k = kernels[:, :, 0, i]
            ax.imshow((k - k.min()) / (k.max() - k.min() + 1e-5), cmap='viridis')
            ax.set_title(f'Kernel {i+1}'); ax.axis('off')
        plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()

def get_activation_maps(model, layer_name, input_sample, training=False):
    """Extracts activation maps from a specified layer of the model."""
    try: return model(input_sample, training=training, return_activations_for_layer=layer_name)
    except Exception as e: print(f"Error getting activations for {layer_name} in {model.__class__.__name__}: {e}"); return None

def activation_map_visualization(model, model_name, test_ds_iter, layer_name='conv1', num_maps_to_show=16):
    """Visualize activation maps from a specified layer for a sample input."""
    print(f"\nğŸ—ºï¸� VISUALIZING ACTIVATION MAPS FROM LAYER: {layer_name} for {model_name}")
    try:
        sample_batch = next(iter(test_ds_iter))
    except (StopIteration, tf.errors.OutOfRangeError):
        print("ERROR: Test dataset iterator for activation maps is exhausted. Consider re-creating it or passing a fresh one.")
        return

    sample_image = sample_batch['image'][0:1]
    
    activations = get_activation_maps(model, layer_name, sample_image)
    if activations is not None and activations.ndim == 4:
        num_maps = min(num_maps_to_show, activations.shape[-1])
        cols, rows = int(np.ceil(np.sqrt(num_maps))), int(np.ceil(num_maps / int(np.ceil(np.sqrt(num_maps)))))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
        fig.suptitle(f'{model_name} - Activation Maps (First {num_maps})', fontsize=16)
        for i in range(num_maps):
            ax = axes.flat[i] if num_maps > 1 else axes
            ax.imshow(activations[0, :, :, i], cmap='viridis'); ax.set_title(f'Map {i+1}'); ax.axis('off')
        plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()

def saliency_map_analysis(model, model_name, test_ds_iter, class_names: list[str]):
    """Generate and visualize saliency maps for the model."""
    print(f"\nğŸ”¥ SALIENCY MAP ANALYSIS FOR {model_name}")
    try:
        sample_batch = next(iter(test_ds_iter))
    except (StopIteration, tf.errors.OutOfRangeError):
        print("ERROR: Test dataset iterator for saliency maps is exhausted. Consider re-creating it or passing a fresh one.")
        return
            
    sample_image, sample_label = sample_batch['image'][0:1], int(sample_batch['label'][0])
    true_class_name = class_names[sample_label]

    @partial(jax.jit, static_argnums=(0, 2))
    def get_saliency_map(m, image, class_index):
        grads = jax.grad(lambda img: m(img, training=False)[0, class_index])(image)
        return jnp.max(jnp.abs(grads[0]), axis=-1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Saliency Map for {model_name} (True Class: {true_class_name})", fontsize=16)

    logits = model(sample_image, training=False)
    pred_idx = int(jnp.argmax(logits, axis=1)[0])
    pred_name = class_names[pred_idx]
    
    saliency_true = get_saliency_map(model, sample_image, sample_label)
    
    axes[0].imshow(sample_image[0])
    axes[0].set_title(f"Input (Pred: {pred_name})"); axes[0].axis('off')

    im = axes[1].imshow(saliency_true, cmap='hot')
    axes[1].set_title(f"Saliency (for True: {true_class_name})"); axes[1].axis('off')
    fig.colorbar(im, ax=axes[1])
        
    plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.show()


# ==============================================================================
# 7. MAIN EXECUTION & DEMO FUNCTIONS
# ==============================================================================

def run_training_and_analysis(
    dataset_name: str, 
    dataset_configs: dict, 
    fallback_configs: dict, 
    learning_rate: float, 
    momentum: float
):
    """Runs a full training and analysis pipeline for the YatCNN model."""
    print("\n" + "="*80 + f"\n                    RUNNING TRAINING & ANALYSIS FOR: {dataset_name.upper()}\n" + "="*80)
    
    # --- Path handling ---
    is_path = os.path.isdir(dataset_name)
    train_path = dataset_name
    test_path = None

    if is_path:
        # If dataset_name is a directory, check for a 'train' subdirectory.
        potential_train_path = os.path.join(dataset_name, 'train')
        if os.path.isdir(potential_train_path):
            train_path = potential_train_path
            # The corresponding test path for submission is in the parent directory.
            test_path = os.path.join(dataset_name, 'test')
            print(f"INFO: Found 'train' and 'test' subdirectories. Using '{train_path}' for training.")
        else:
            print(f"INFO: 'train' subdirectory not found in '{dataset_name}'. "
                  f"Assuming '{dataset_name}' is the training data directory.")

    # Get class names and batch size
    if is_path:
        custom_config = dataset_configs.get('custom_folder', fallback_configs)
        _, _, class_names_comp, _ = create_image_folder_dataset(
            train_path, validation_split=custom_config.get('test_split_percentage', 0.2), seed=42
        )
        current_batch_size_for_eval = custom_config.get('batch_size', fallback_configs['batch_size_folder'])
    else:
        config = dataset_configs.get(dataset_name, {})
        ds_info = tfds.builder(dataset_name).info
        label_key = config.get('label_key', 'label')

        # Correctly access the label feature and get class names
        label_feature = ds_info.features[label_key]
        if hasattr(label_feature, 'names'):
            class_names_comp = label_feature.names
        else:
            class_names_comp = [f'Class {i}' for i in range(label_feature.num_classes)]
        current_batch_size_for_eval = config.get('batch_size', fallback_configs['batch_size'])
    
    # Train model
    model_name = "YatCNN"
    print(f"\nğŸš€ STEP 1: Training {model_name} Model...");
    model_train_path = train_path if is_path else dataset_name
    model, metrics_history = _train_model_loop(
        YatCNN, model_name, model_train_path, 0, learning_rate, momentum, optax.adamw,
        dataset_configs=dataset_configs, fallback_configs=fallback_configs
    )
    
    # Prepare test data iterator for detailed analysis
    def get_test_iter():
      if is_path:
          custom_config = dataset_configs.get('custom_folder', fallback_configs)
          image_size_for_eval = custom_config.get('input_dim', (64, 64))
          input_channels_for_eval = custom_config.get('input_channels', 3)
          split_percentage_for_eval = custom_config.get('test_split_percentage', 0.2)
          current_batch_size_for_eval = custom_config.get('batch_size', fallback_configs['batch_size_folder'])

          _, test_ds, _, _ = create_image_folder_dataset(
              train_path, validation_split=split_percentage_for_eval, seed=42
          )
          image_processor = get_image_processor(image_size=image_size_for_eval, num_channels=input_channels_for_eval)
          test_ds = test_ds.map(image_processor, num_parallel_calls=tf.data.AUTOTUNE)
          return test_ds.batch(current_batch_size_for_eval, True).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
      else:
          key, label = dataset_configs[dataset_name]['image_key'], dataset_configs[dataset_name]['label_key']
          preprocess = lambda s: {'image': tf.cast(s[key], tf.float32) / 255.0, 'label': s[label]}
          test_ds = tfds.load(dataset_name, split=dataset_configs[dataset_name]['test_split']).map(preprocess)
          return test_ds.batch(current_batch_size_for_eval, True).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
    
    # Run analysis
    print("\nğŸ“Š STEP 2: Running Analysis...")
    plot_training_curves(metrics_history, model_name)
    print_final_metrics(metrics_history, model_name)
    analyze_convergence(metrics_history, model_name)
    predictions_data = detailed_test_evaluation(model, get_test_iter(), class_names=class_names_comp, model_name=model_name)
    plot_confusion_matrix(predictions_data, model_name)
    generate_summary_report(metrics_history, model_name)
    
    print("\nğŸ”¬ STEP 3: Running Advanced Analysis...")
    visualize_kernels(model, model_name, layer_name='conv1')
    activation_map_visualization(model, model_name, get_test_iter(), layer_name='conv1')
    saliency_map_analysis(model, model_name, get_test_iter(), class_names=class_names_comp)
    
    # Generate submission file for Kaggle
    if test_path and os.path.isdir(test_path):
        print("\nğŸ“� STEP 4: Generating Kaggle Submission File...")
        custom_config = dataset_configs.get('custom_folder', fallback_configs)
        image_size = custom_config.get('input_dim', (32, 32))
        input_channels = custom_config.get('input_channels', 3)
        batch_size = custom_config.get('batch_size', fallback_configs.get('batch_size_folder', 32))
            
        generate_submission_file(
            model=model,
            test_dir=test_path,
            class_names=class_names_comp,
            image_size=image_size,
            input_channels=input_channels,
            model_name=model_name,
            batch_size=batch_size
        )
    
    print("\n" + "="*80 + f"\n                    ANALYSIS FOR {dataset_name.upper()} COMPLETE! âœ…\n" + "="*80)
    return {'model': model, 'metrics_history': metrics_history, 'predictions_data': predictions_data}


# ==============================================================================
# 8. SCRIPT ENTRYPOINT & USAGE GUIDE
# ==============================================================================

def main():
    """Main function to define configurations and run the training and analysis."""
    
    # Dataset Configurations
    dataset_configs = {
        'cifar10': {
            'num_classes': 10, 'input_channels': 3, 'train_split': 'train', 'test_split': 'test',
            'image_key': 'image', 'label_key': 'label', 'num_epochs': 5, 'eval_every': 200, 'batch_size': 64
        },
        'cifar100': {
            'num_classes': 100, 'input_channels': 3, 'train_split': 'train', 'test_split': 'test',
            'image_key': 'image', 'label_key': 'label', 'num_epochs': 75, 'eval_every': 200, 'batch_size': 64
        },
        'stl10': {
            'num_classes': 10, 'input_channels': 3, 'train_split': 'train', 'test_split': 'test',
            'image_key': 'image', 'label_key': 'label', 'num_epochs': 100, 'eval_every': 200, 'batch_size': 32
        },
        'eurosat/rgb': {
            'num_classes': 10, 'input_channels': 3, 'train_split': 'train[:80%]', 'test_split': 'train[80%:]',
            'image_key': 'image', 'label_key': 'label', 'num_epochs': 30, 'eval_every': 100, 'batch_size': 32
        },
        'eurosat/all': {
            'num_classes': 10, 'input_channels': 13, 'train_split': 'train[:80%]', 'test_split': 'train[80%:]',
            'image_key': 'image', 'label_key': 'label', 'num_epochs': 20, 'eval_every': 100, 'batch_size': 16
        },
        'downsampled_imagenet/64x64': {
            'num_classes': 1000, 'input_channels': 3, 'train_split': 'train', 'test_split': 'validation',
            'image_key': 'image', 'label_key': 'label', 'num_epochs': 5, 'eval_every': 100, 'batch_size': 256
        },
        # --- Configuration for your custom folder dataset ---
        # These settings are used when `dataset_to_run` is a local folder path.
        'custom_folder': {
            'num_classes': 'auto',      # 'auto' means it will be inferred from subdirectories
            'input_channels': 3,        # Number of image channels (e.g., 3 for RGB, 1 for grayscale)
            'input_dim': (32, 32),      # Image dimensions (height, width) for resizing
            'test_split_percentage': 0.2, # Percentage of data to use for the test set (e.g., 0.2 for 20%)
            'num_epochs': 100,           # Number of times to iterate over the training data
            'eval_every': 500,          # How often to evaluate the model (in steps)
            'batch_size': 256,           # Number of samples per batch
        }
    }

    # Training parameter fallbacks
    default_config = dataset_configs.get('cifar10', {})
    fallback_configs = {
        'num_epochs': default_config.get('num_epochs', 10),
        'eval_every': default_config.get('eval_every', 200),
        'batch_size': default_config.get('batch_size', 64),
        'batch_size_folder': 32, # Specific for folder datasets
    }
    
    # Common training parameters
    learning_rate = 0.003
    momentum = 0.9
    
    # =================================================
    # CHOOSE YOUR DATASET HERE
    # =================================================
    # Options: 'cifar10', 'cifar100', 'stl10', 'eurosat/rgb', 
    # or a path to your local folder, e.g., '/path/to/my_dataset'
    dataset_to_run = '/kaggle/input/ml-nomads-downscaling-laws-cifar-10' 
    # =================================================

    # --- Run the training and analysis ---
    run_training_and_analysis(
        dataset_name=dataset_to_run,
        dataset_configs=dataset_configs,
        fallback_configs=fallback_configs,
        learning_rate=learning_rate,
        momentum=momentum
    )


if __name__ == '__main__':
    print("\n" + "="*80)
    print("ğŸš€ TO RUN THE TRAINING AND ANALYSIS:")
    print("   Modify the `dataset_to_run` variable inside the `main()` function.")
    print("\n   Examples:")
    print("   dataset_to_run = 'cifar100'")
    print("   dataset_to_run = '/path/to/your/image_folder'")
    print("="*80)
    
    # Execute the main function
    main()








