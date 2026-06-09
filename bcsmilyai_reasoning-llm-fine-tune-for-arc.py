# ==============================================================================
# SAM1 Fine-tuning - OFFLINE Training (4x L4 GPU)
# ==============================================================================
# Prerequisites:
# 1. Add tokenized dataset as input: /kaggle/input/tokenised-reasoning-dataset/
# 2. Add tokenizer cache as input: /kaggle/input/<your-setup-notebook>/tokenizer_cache/
# 3. Add base model as input: /kaggle/input/sam-1-large-base/
# ==============================================================================

import os
import time

# Set environment variables
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

print("="*70)
print("ğŸš€ SAM1 FINE-TUNING - OFFLINE MODE".center(70))
print("="*70)

print("\nğŸ“š Importing TensorFlow...")
import tensorflow as tf
import keras

# ==============================================================================
# GPU Setup with Model Sharding
# ==============================================================================
print("\nğŸ”Œ Initializing GPU strategy...")

gpu_devices = tf.config.list_physical_devices('GPU')
if not gpu_devices:
    raise RuntimeError("No GPUs found! This notebook requires GPUs.")

print(f"âœ… Found {len(gpu_devices)} GPU(s)")

# CRITICAL: Memory management optimized for L4 (24GB each)
for gpu in gpu_devices:
    # Enable memory growth (don't allocate all at once)
    tf.config.experimental.set_memory_growth(gpu, True)
    
    # L4 has 24GB, use 22GB to be safe
    try:
        tf.config.set_logical_device_configuration(
            gpu,
            [tf.config.LogicalDeviceConfiguration(memory_limit=22528)]  # 22GB in MB
        )
        print(f"   GPU {gpu.name}: Memory limit set to 22GB")
    except:
        pass  # Already configured

# Use XLA compilation for speed and memory efficiency
tf.config.optimizer.set_jit(True)
print("   XLA JIT compilation enabled")

if len(gpu_devices) > 1:
    # Use MirroredStrategy for data parallelism (we'll do model sharding manually)
    strategy = tf.distribute.MirroredStrategy()
    device_type = "Multi-GPU-Sharded"
    print(f"âœ… Multi-GPU with MODEL SHARDING: {len(gpu_devices)} GPUs")
    
    if len(gpu_devices) == 4:
        print(f"   ğŸš€ 4-way sharding: 4 layers per GPU (optimized for L4)")
    else:
        layers_per = 16 // len(gpu_devices)
        print(f"   Each GPU will hold {layers_per} layers (pipeline parallelism)")
else:
    strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
    device_type = "Single-GPU"
    print(f"âœ… Single GPU Strategy")

print(f"\nğŸš€ Device: {device_type}")
print(f"ğŸ“Š Replicas: {strategy.num_replicas_in_sync}")
print(f"ğŸ”§ TensorFlow: {tf.__version__}")

# ==============================================================================
# Configuration
# ==============================================================================
import numpy as np
import pandas as pd
import json

class Cfg:
    def __init__(self):
        self.vocab_size = 50257
        self.d_model = 768
        self.n_layers = 16
        self.n_heads = 12
        self.ff_mult = 7.0
        self.max_len = 128
        self.dropout = 0.1
        self.rope_theta = 10_000
        self.seed = 42
        self.weight_decay = 0.1

class FineTuneCfg(Cfg):
    def __init__(self, strategy, device_type):
        super().__init__()
        self.max_len = 1024
        self.lr = 1e-4
        self.epochs = 1
        
        # Optimized batch sizes for L4 GPUs (24GB each!)
        if device_type == "Multi-GPU-Sharded":
            num_gpus = strategy.num_replicas_in_sync
            if num_gpus == 4:
                # 4x L4: 8 per GPU = 32 global (HUGE WIN!)
                self.per_core_batch_size = 8
            elif num_gpus == 2:
                self.per_core_batch_size = 6
            else:
                self.per_core_batch_size = 4
        elif device_type == "Single-GPU":
            self.per_core_batch_size = 8
        else:
            self.per_core_batch_size = 4
        
        self.global_batch = self.per_core_batch_size * strategy.num_replicas_in_sync
        
        # Mixed precision for GPU (saves memory!)
        if device_type in ["Multi-GPU-Sharded", "Single-GPU"]:
            keras.mixed_precision.set_global_policy('mixed_bfloat16')
            print(f"   Mixed precision enabled (bfloat16)")

cfg = FineTuneCfg(strategy, device_type)

# Paths - UPDATE THESE IF YOUR INPUT NAMES ARE DIFFERENT
BASE_MODEL_KERAS_PATH = "/kaggle/input/sam-1-large-base/keras/default/1/tf-sam/model.keras"
TOKENIZER_DIR = "/kaggle/input/setup-tokeniser/"  # UPDATE THIS PATH
TRAIN_CACHE = "/kaggle/input/tokenised-reasoning-dataset/tokenized_cache/train_tokens.npy"
VAL_CACHE = "/kaggle/input/tokenised-reasoning-dataset/tokenized_cache/val_tokens.npy"

# Output pathsy
HF_FINETUNED_DIR = "/kaggle/working/safe-sam-finetuned/"
TF_FINETUNED_DIR = "/kaggle/working/tf-sam-finetuned/"
FINETUNE_CHECKPOINT_DIR = "/kaggle/working/checkpoints-finetuned/"

for d in [HF_FINETUNED_DIR, TF_FINETUNED_DIR, FINETUNE_CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"\nâš™ï¸� Config: MaxLen={cfg.max_len}, Global Batch={cfg.global_batch}, Epochs={cfg.epochs}")

# Install required packages (offline compatible)
print("\nğŸ“¦ Installing packages (offline mode - using cached wheels)...")
#!pip install --no-cache-dir --quiet safetensors transformers

# ==============================================================================
# Model Architecture
# ==============================================================================
print("\nğŸ�—ï¸� Defining model architecture...")

@keras.saving.register_keras_serializable()
class RotaryEmbedding(keras.layers.Layer):
    def __init__(self, dim, max_len=2048, theta=10000, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.max_len = max_len
        self.theta = theta
        self.built_cache = False
    
    def build(self, input_shape):
        if not self.built_cache:
            inv_freq = 1.0 / (self.theta ** (tf.range(0, self.dim, 2, dtype=tf.float32) / self.dim))
            t = tf.range(self.max_len, dtype=tf.float32)
            freqs = tf.einsum("i,j->ij", t, inv_freq)
            emb = tf.concat([freqs, freqs], axis=-1)
            
            self.cos_cached = tf.constant(tf.cos(emb), dtype=tf.float32)
            self.sin_cached = tf.constant(tf.sin(emb), dtype=tf.float32)
            self.built_cache = True
        
        super().build(input_shape)
    
    def rotate_half(self, x):
        x1, x2 = tf.split(x, 2, axis=-1)
        return tf.concat([-x2, x1], axis=-1)
    
    def call(self, q, k):
        seq_len = tf.shape(q)[2]
        dtype = q.dtype
        cos = tf.cast(self.cos_cached[:seq_len, :], dtype)[None, None, :, :]
        sin = tf.cast(self.sin_cached[:seq_len, :], dtype)[None, None, :, :]
        
        q_rotated = (q * cos) + (self.rotate_half(q) * sin)
        k_rotated = (k * cos) + (self.rotate_half(k) * sin)
        
        return q_rotated, k_rotated
    
    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "max_len": self.max_len, "theta": self.theta})
        return config


@keras.saving.register_keras_serializable()
class RMSNorm(keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(name="scale", shape=(input_shape[-1],), initializer="ones")
    
    def call(self, x):
        variance = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
        return x * tf.math.rsqrt(variance + self.epsilon) * self.scale
    
    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


@keras.saving.register_keras_serializable()
class TransformerBlock(keras.layers.Layer):
    def __init__(self, d_model, n_heads, ff_dim, dropout, max_len, rope_theta, layer_idx=0, gpu_id=None, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        self.max_len = max_len
        self.rope_theta = rope_theta
        self.head_dim = d_model // n_heads
        self.layer_idx = layer_idx
        self.gpu_id = gpu_id
        
        self.pre_attn_norm = RMSNorm()
        self.pre_ffn_norm = RMSNorm()
        
        self.q_proj = keras.layers.Dense(d_model, use_bias=False, name="q_proj")
        self.k_proj = keras.layers.Dense(d_model, use_bias=False, name="k_proj")
        self.v_proj = keras.layers.Dense(d_model, use_bias=False, name="v_proj")
        self.out_proj = keras.layers.Dense(d_model, use_bias=False, name="o_proj")
        
        self.rope = RotaryEmbedding(self.head_dim, max_len=max_len, theta=rope_theta)
        
        self.gate_proj = keras.layers.Dense(ff_dim, use_bias=False, name="gate_proj")
        self.up_proj = keras.layers.Dense(ff_dim, use_bias=False, name="up_proj")
        self.down_proj = keras.layers.Dense(d_model, use_bias=False, name="down_proj")
        
        self.dropout = keras.layers.Dropout(dropout)
    
    def call(self, x, training=None):
        if self.gpu_id is not None:
            device = f'/gpu:{self.gpu_id}'
        else:
            device = None
        
        if device:
            with tf.device(device):
                return self._forward(x, training)
        else:
            return self._forward(x, training)
    
    def _forward(self, x, training):
        B, T, D = tf.shape(x)[0], tf.shape(x)[1], self.d_model
        dtype = x.dtype
        
        # Attention
        res = x
        y = self.pre_attn_norm(x)
        
        q = tf.transpose(tf.reshape(self.q_proj(y), [B, T, self.n_heads, self.head_dim]), [0, 2, 1, 3])
        k = tf.transpose(tf.reshape(self.k_proj(y), [B, T, self.n_heads, self.head_dim]), [0, 2, 1, 3])
        v = tf.transpose(tf.reshape(self.v_proj(y), [B, T, self.n_heads, self.head_dim]), [0, 2, 1, 3])
        
        q, k = self.rope(q, k)
        
        scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.head_dim, dtype))
        
        mask = tf.where(
            tf.linalg.band_part(tf.ones([T, T], dtype=dtype), -1, 0) == 0,
            tf.constant(-1e9, dtype=dtype),
            tf.constant(0.0, dtype=dtype)
        )
        scores += mask
        attn = tf.matmul(tf.nn.softmax(scores, axis=-1), v)
        
        attn = tf.reshape(tf.transpose(attn, [0, 2, 1, 3]), [B, T, D])
        x = res + self.dropout(self.out_proj(attn), training=training)
        
        # FFN (SwiGLU)
        res = x
        y = self.pre_ffn_norm(x)
        ffn = self.down_proj(keras.activations.silu(self.gate_proj(y)) * self.up_proj(y))
        
        return res + self.dropout(ffn, training=training)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_rate,
            "max_len": self.max_len,
            "rope_theta": self.rope_theta,
            "layer_idx": self.layer_idx,
            "gpu_id": self.gpu_id
        })
        return config


@keras.saving.register_keras_serializable()
class SAM1Model(keras.Model):
    def __init__(self, enable_sharding=False, num_gpus=1, **kwargs):
        super().__init__()
        if 'config' in kwargs and isinstance(kwargs['config'], dict):
            self.cfg = kwargs['config']
        elif 'vocab_size' in kwargs:
            self.cfg = kwargs
        else:
            self.cfg = kwargs.get('cfg', kwargs)
        
        self.enable_sharding = enable_sharding
        self.num_gpus = num_gpus
        
        # Embeddings on GPU 0
        with tf.device('/gpu:0' if enable_sharding else None):
            self.embed = keras.layers.Embedding(self.cfg['vocab_size'], self.cfg['d_model'], name="embed_tokens")
        
        ff_dim = int(self.cfg['d_model'] * self.cfg['ff_mult'])
        block_args = {
            'd_model': self.cfg['d_model'],
            'n_heads': self.cfg['n_heads'],
            'ff_dim': ff_dim,
            'dropout': self.cfg['dropout'],
            'max_len': self.cfg['max_len'],
            'rope_theta': self.cfg['rope_theta']
        }
        
        # Create transformer blocks with GPU assignment
        self.blocks = []
        layers_per_gpu = self.cfg['n_layers'] // num_gpus if enable_sharding else self.cfg['n_layers']
        
        for i in range(self.cfg['n_layers']):
            if enable_sharding and num_gpus > 1:
                gpu_id = i // layers_per_gpu
                if gpu_id >= num_gpus:
                    gpu_id = num_gpus - 1
                
                if i % layers_per_gpu == 0:
                    end_layer = min(i + layers_per_gpu - 1, self.cfg['n_layers'] - 1)
                    print(f"   ğŸ“� Layers {i}-{end_layer} â†’ GPU:{gpu_id}")
            else:
                gpu_id = None
            
            block = TransformerBlock(
                name=f"block_{i}",
                layer_idx=i,
                gpu_id=gpu_id,
                **block_args
            )
            self.blocks.append(block)
        
        # Final norm and LM head on last GPU
        final_device = f'/gpu:{num_gpus-1}' if enable_sharding and num_gpus > 1 else None
        with tf.device(final_device):
            self.norm = RMSNorm(name="final_norm")
            self.lm_head = keras.layers.Dense(self.cfg['vocab_size'], use_bias=False, name="lm_head")
    
    def call(self, input_ids, training=None):
        if self.enable_sharding and self.num_gpus > 1:
            with tf.device('/gpu:0'):
                x = self.embed(input_ids)
        else:
            x = self.embed(input_ids)
        
        for block in self.blocks:
            x = block(x, training=training)
        
        if self.enable_sharding and self.num_gpus > 1:
            with tf.device(f'/gpu:{self.num_gpus-1}'):
                x = self.norm(x)
                return self.lm_head(x)
        else:
            return self.lm_head(self.norm(x))
    
    def get_config(self):
        base_config = super().get_config()
        base_config['config'] = self.cfg
        base_config['enable_sharding'] = self.enable_sharding
        base_config['num_gpus'] = self.num_gpus
        return base_config

print("âœ… Model architecture defined")

# ==============================================================================
# Load Tokenizer (Offline from cached files)
# ==============================================================================
print("\nğŸ“¦ Loading tokenizer from cache...")

from transformers import AutoTokenizer

# Load tokenizer from cached directory
tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)

# Load metadata
with open(os.path.join(TOKENIZER_DIR, "tokenizer_metadata.json"), 'r') as f:
    tok_metadata = json.load(f)

eos_token = tok_metadata['eos_token']
eos_token_id = tok_metadata['eos_token_id']
custom_tokens = tok_metadata['custom_tokens']

# Update config with actual vocab size
cfg.vocab_size = tok_metadata['vocab_size']

print(f"âœ… Tokenizer loaded (vocab_size={cfg.vocab_size})")
print(f"   EOS token: '{eos_token}' (ID: {eos_token_id})")
print(f"   <think> ID: {tok_metadata['think_token_id']}")
print(f"   </think> ID: {tok_metadata['end_think_token_id']}")

# ==============================================================================
# Load Base Model and Resize Embeddings
# ==============================================================================
print("\nğŸ”§ Loading base model...")

enable_sharding = (device_type == "Multi-GPU-Sharded")
num_gpus = strategy.num_replicas_in_sync if enable_sharding else 1

if enable_sharding:
    print(f"ğŸ”€ MODEL SHARDING ENABLED: Splitting 16 layers across {num_gpus} GPUs")

with strategy.scope():
    model = SAM1Model(enable_sharding=enable_sharding, num_gpus=num_gpus, **vars(cfg))
    _ = model(tf.zeros((1, cfg.max_len), dtype=tf.int32))
    
    if os.path.exists(BASE_MODEL_KERAS_PATH):
        print("ğŸ“¥ Loading base model checkpoint...")
        try:
            print("   Creating temporary model with old vocab size...")
            old_cfg = vars(cfg).copy()
            old_cfg['vocab_size'] = 50257  # Original GPT-2 vocab size
            
            temp_model = SAM1Model(enable_sharding=enable_sharding, num_gpus=num_gpus, **old_cfg)
            _ = temp_model(tf.zeros((1, cfg.max_len), dtype=tf.int32))
            
            temp_model.load_weights(BASE_MODEL_KERAS_PATH, skip_mismatch=True)
            print("   âœ… Weights loaded into temp model")
            
            print("   ğŸ“‹ Extracting weights from temp model...")
            
            old_embed_weights = temp_model.embed.weights[0].numpy()
            old_vocab_size = old_embed_weights.shape[0]
            old_lm_head_weights = temp_model.lm_head.weights[0].numpy()
            
            print("   ğŸ”„ Copying transformer block weights...")
            for i, (old_block, new_block) in enumerate(zip(temp_model.blocks, model.blocks)):
                for old_w, new_w in zip(old_block.weights, new_block.weights):
                    new_w.assign(old_w)
            
            model.norm.scale.assign(temp_model.norm.scale)
            
            print("   âœ… Transformer weights copied")
            
            del temp_model
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"   âš ï¸� Failed to load base weights: {e}")
            print("   Using random initialization")
            old_embed_weights = model.embed.weights[0].numpy()
            old_vocab_size = old_embed_weights.shape[0]
            old_lm_head_weights = model.lm_head.weights[0].numpy()
    else:
        print("âš ï¸� Base model not found, using random initialization")
        old_embed_weights = model.embed.weights[0].numpy()
        old_vocab_size = old_embed_weights.shape[0]
        old_lm_head_weights = model.lm_head.weights[0].numpy()
    
    print(f"\nğŸ”§ Checking embedding size...")
    print(f"   Old vocab size: {old_vocab_size}")
    print(f"   New vocab size: {cfg.vocab_size}")
    print(f"   Tokens added: {cfg.vocab_size - old_vocab_size}")
    
    if old_vocab_size != cfg.vocab_size:
        print(f"ğŸ”€ Resizing embeddings from {old_vocab_size} to {cfg.vocab_size}...")
        
        new_embed_weights = np.zeros((cfg.vocab_size, cfg.d_model), dtype=np.float32)
        new_embed_weights[:old_vocab_size] = old_embed_weights[:old_vocab_size]
        new_embed_weights[old_vocab_size:] = np.mean(old_embed_weights, axis=0, keepdims=True)
        
        model.embed.set_weights([new_embed_weights])
        
        new_lm_head_weights = np.zeros((cfg.d_model, cfg.vocab_size), dtype=np.float32)
        new_lm_head_weights[:, :old_vocab_size] = old_lm_head_weights[:, :old_vocab_size]
        new_lm_head_weights[:, old_vocab_size:] = np.mean(old_lm_head_weights, axis=1, keepdims=True)
        
        model.lm_head.set_weights([new_lm_head_weights])
        
        print(f"âœ… Embeddings resized!")
        print(f"   - Embedding layer: {old_vocab_size} â†’ {cfg.vocab_size}")
        print(f"   - LM head: {old_vocab_size} â†’ {cfg.vocab_size}")
        print(f"   - New tokens initialized with mean of existing embeddings")
    else:
        print("âœ… Vocab size unchanged, no resizing needed")

# ==============================================================================
# Load Cached Dataset
# ==============================================================================
print("\nğŸ“Š Loading cached dataset...")

if os.path.exists(TRAIN_CACHE) and os.path.exists(VAL_CACHE):
    print("âœ… Loading cached tokenized data...")
    train_tokens = np.load(TRAIN_CACHE)
    val_tokens = np.load(VAL_CACHE)
    print(f"   Loaded from cache!")
    print(f"   Train: {train_tokens.shape}, Val: {val_tokens.shape}")
else:
    raise FileNotFoundError(f"Cached data not found! Please ensure these files exist:\n  {TRAIN_CACHE}\n  {VAL_CACHE}")

# Create TF datasets
def create_dataset(tokens, batch_size, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices(tokens)
    if shuffle:
        ds = ds.shuffle(1024, seed=cfg.seed)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

train_ds = create_dataset(train_tokens, cfg.global_batch, shuffle=True)
val_ds = create_dataset(val_tokens, cfg.global_batch, shuffle=False)
num_train_steps = len(train_ds)

print(f"âœ… TF Datasets ready: {num_train_steps} train steps, {len(val_ds)} val steps")

# ==============================================================================
# Training Setup and Callbacks
# ==============================================================================
class KaggleProgressLogger(keras.callbacks.Callback):
    def __init__(self, print_freq=50, **kwargs):
        super().__init__(**kwargs)
        self.print_freq = print_freq
        self.step_times = []
    
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.global_step = 0
        print("\n" + "="*70)
        print("ğŸ”¥ FINE-TUNING START ğŸ”¥".center(70))
        print("="*70)
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        self.epoch_step = 0
        self.total_steps = self.params.get('steps', self.params['steps'])
        print(f"\n{'â”€'*70}")
        print(f"ğŸ“š Epoch {epoch + 1}/{self.params['epochs']}")
        print(f"{'â”€'*70}")
    
    def on_train_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()
    
    def on_train_batch_end(self, batch, logs=None):
        # Track step time
        step_time = time.time() - self.batch_start_time
        self.step_times.append(step_time)
        if len(self.step_times) > 100:
            self.step_times.pop(0)
        
        self.global_step += 1
        self.epoch_step = batch + 1
        
        # Print at specified frequency or last step
        if self.epoch_step % self.print_freq == 0 or self.epoch_step == self.total_steps:
            loss = logs.get('loss', 0.0)
            acc = logs.get('accuracy', 0.0) * 100
            
            # Calculate learning rate from optimizer
            lr = self.model.optimizer.learning_rate
            if hasattr(lr, 'numpy'):
                current_lr = float(lr.numpy())
            elif hasattr(lr, '__call__'):
                current_lr = float(lr(self.model.optimizer.iterations).numpy())
            else:
                current_lr = float(lr)
            
            # Calculate ETA
            avg_step_time = sum(self.step_times) / len(self.step_times)
            remaining_steps = self.total_steps - self.epoch_step
            eta_seconds = avg_step_time * remaining_steps
            
            # Format output
            step_str = f"{self.epoch_step}/{self.total_steps}"
            loss_str = f"loss: {loss:.4f}"
            acc_str = f"acc: {acc:.2f}%"
            lr_str = f"lr: {current_lr:.2e}"
            time_str = f"{avg_step_time:.2f}s/step"
            eta_str = f"ETA: {self._format_time(eta_seconds)}"
            
            print(f"  Step {step_str:>12} â”‚ {loss_str} â”‚ {acc_str} â”‚ {lr_str} â”‚ {time_str} â”‚ {eta_str}")
    
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss', 0.0)
        val_acc = logs.get('val_accuracy', 0.0) * 100
        epoch_time = time.time() - self.epoch_start_time
        
        print(f"{'â”€'*70}")
        print(f"ğŸ“Š Epoch {epoch + 1} Summary:")
        print(f"   Train Loss: {logs.get('loss', 0.0):.4f} â”‚ Train Acc: {logs.get('accuracy', 0.0)*100:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} â”‚ Val Acc:   {val_acc:.2f}%")
        print(f"   Duration:   {self._format_time(epoch_time)}")
        print(f"{'â”€'*70}")
    
    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        print("\n" + "="*70)
        print(f"âœ… FINE-TUNING COMPLETE".center(70))
        print(f"Total Duration: {self._format_time(total_time)}".center(70))
        print(f"Total Steps: {self.global_step}".center(70))
        print("="*70 + "\n")
    
    def _format_time(self, seconds):
        """Format seconds into HH:MM:SS or MM:SS"""
        s = int(seconds)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        else:
            return f"{m:02d}:{s:02d}"

print("\nğŸ”§ Compiling model...")

with strategy.scope():
    schedule = keras.optimizers.schedules.CosineDecay(cfg.lr, num_train_steps * cfg.epochs)
    
    # Use mixed precision optimizer for GPU
    optimizer = keras.optimizers.AdamW(learning_rate=schedule, weight_decay=cfg.weight_decay)
    optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)
    print("   Using LossScaleOptimizer for mixed precision")
    
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

print("âœ… Model compiled")

# Callbacks
checkpoint_path = os.path.join(FINETUNE_CHECKPOINT_DIR, "ckpt.weights.h5")
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=0
)
logger_cb = KaggleProgressLogger(print_freq=50)

# ==============================================================================
# Training
# ==============================================================================
print(f"\nStarting training on {strategy.num_replicas_in_sync} GPUs...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=cfg.epochs,
    callbacks=[checkpoint_cb, logger_cb],
    verbose=0
)

# ==============================================================================
# Save and Export
# ==============================================================================
print("\nğŸš€ Exporting fine-tuned model...")

if os.path.exists(checkpoint_path):
    model.load_weights(checkpoint_path)
    print("âœ… Loaded best checkpoint")
else:
    print("âš ï¸� Using final weights (no checkpoint found)")

# Save Keras model
final_keras_path = os.path.join(TF_FINETUNED_DIR, "model.keras")
model.save(final_keras_path)
print(f"âœ… Keras model saved to {final_keras_path}")

# Helper function
def create_hf_config_dict():
    return {
        "model_type": "sam1",
        "architectures": ["SAM1ForCausalLM"],
        "vocab_size": cfg.vocab_size,
        "max_position_embeddings": cfg.max_len,
        "hidden_size": cfg.d_model,
        "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,
        "intermediate_size": int(cfg.d_model * cfg.ff_mult),
        "hidden_act": "silu",
        "rope_theta": cfg.rope_theta,
        "rms_norm_eps": 1e-5,
        "bos_token_id": eos_token_id,
        "eos_token_id": eos_token_id,
        "pad_token_id": eos_token_id,
        "auto_map": {"AutoModel": "modeling_sam1.SAM1ForCausalLM"},
        "custom_tokens": custom_tokens
    }

# Save tokenizer to TF directory
print("\nğŸ’¾ Saving tokenizer...")
tok.save_pretrained(TF_FINETUNED_DIR)
print(f"âœ… Tokenizer saved to {TF_FINETUNED_DIR}")

# Save TF files
with open(os.path.join(TF_FINETUNED_DIR, "config.json"), 'w') as f:
    json.dump(create_hf_config_dict(), f, indent=2)
pd.DataFrame(history.history).to_csv(os.path.join(TF_FINETUNED_DIR, "training_history.csv"), index=False)
print("âœ… TF model files saved")

# Convert to HuggingFace Safetensors
print(f"\nğŸ“¦ Converting to HuggingFace format...")

from safetensors.tensorflow import save_file

safetensors_weights = {}

# Embeddings and final layers
safetensors_weights["model.embed_tokens.weight"] = model.embed.weights[0]
safetensors_weights["model.norm.weight"] = model.norm.scale
safetensors_weights["lm_head.weight"] = tf.transpose(model.lm_head.kernel)

# Transformer blocks
for i, block in enumerate(model.blocks):
    # Norms
    safetensors_weights[f"model.layers.{i}.input_layernorm.weight"] = block.pre_attn_norm.scale
    safetensors_weights[f"model.layers.{i}.post_attention_layernorm.weight"] = block.pre_ffn_norm.scale
    
    # Attention
    safetensors_weights[f"model.layers.{i}.self_attn.q_proj.weight"] = tf.transpose(block.q_proj.kernel)
    safetensors_weights[f"model.layers.{i}.self_attn.k_proj.weight"] = tf.transpose(block.k_proj.kernel)
    safetensors_weights[f"model.layers.{i}.self_attn.v_proj.weight"] = tf.transpose(block.v_proj.kernel)
    safetensors_weights[f"model.layers.{i}.self_attn.o_proj.weight"] = tf.transpose(block.out_proj.kernel)
    
    # MLP
    safetensors_weights[f"model.layers.{i}.mlp.gate_proj.weight"] = tf.transpose(block.gate_proj.kernel)
    safetensors_weights[f"model.layers.{i}.mlp.up_proj.weight"] = tf.transpose(block.up_proj.kernel)
    safetensors_weights[f"model.layers.{i}.mlp.down_proj.weight"] = tf.transpose(block.down_proj.kernel)

# Save HuggingFace files
with open(os.path.join(HF_FINETUNED_DIR, "config.json"), 'w') as f:
    json.dump(create_hf_config_dict(), f, indent=2)

# Save tokenizer to HF directory as well
tok.save_pretrained(HF_FINETUNED_DIR)
print(f"âœ… Tokenizer saved to {HF_FINETUNED_DIR}")

# Save safetensors
save_file(safetensors_weights, os.path.join(HF_FINETUNED_DIR, "model.safetensors"))
pd.DataFrame(history.history).to_csv(os.path.join(HF_FINETUNED_DIR, "training_history.csv"), index=False)

print("âœ… HuggingFace model saved")
print("\nğŸ�� EXPORT COMPLETE! ğŸ��")
print(f"   - TF Model: {TF_FINETUNED_DIR}")
print(f"   - HF Model: {HF_FINETUNED_DIR}")
print("\nğŸ“� Your fine-tuned model is ready in /kaggle/working/")
print("   You can download it or use it for inference!")

