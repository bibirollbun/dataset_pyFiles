!git clone https://huggingface.co/convaiinnovations/hindi-embedding-foundational-model


%cd hindi-embedding-foundational-model


%%writefile convaicausallm_model.py
import torch
import torch.nn as nn
import math
from transformers import PreTrainedModel, PretrainedConfig
from typing import Optional, Tuple

class ConvaiCausalLMConfig(PretrainedConfig):
    model_type = "convaicausallm"
    
    def __init__(
        self,
        vocab_size=16000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=16,
        num_key_value_heads=4,
        intermediate_size=3072,
        hidden_act="silu",
        max_position_embeddings=512,
        rope_theta=10000.0,  # Base parameter for RoPE
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cos, sin) with given dimensions.
    This matches the RoPE implementation in many popular models like LLaMA, Mistral etc.
    """
    # Ensure dim is even for complex numbers
    assert dim % 2 == 0, "Dimension must be even"
    
    # Create position indices for caching
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end).float()
    freqs = torch.outer(t, freqs)  # [end, dim/2]
    
    # Create complex exponentials (cos, sin pairs)
    cos, sin = torch.cos(freqs), torch.sin(freqs)
    return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """
    Apply rotary position embeddings to q and k tensors.
    
    Args:
        q: Query tensor - [batch_size, seq_len, num_heads, head_dim]
        k: Key tensor - [batch_size, seq_len, num_kv_heads, head_dim] 
        cos, sin: Cosine and sine position encodings - [seq_len, head_dim/2]
        position_ids: Optional position indices - defaults to incremental positions
    """
    # Extract shapes
    batch, seq_len, n_heads, head_dim = q.shape
    _, kv_seq_len, n_kv_heads, _ = k.shape
    
    # Handle position IDs or use sequential positions
    if position_ids is None:
        # Default: Just use sequential positions
        position_ids = torch.arange(seq_len, device=q.device)
        position_ids = position_ids.unsqueeze(0).expand(batch, -1)
        
    # Get the cosine and sine for the positions we're using
    cos = cos[position_ids].unsqueeze(-2)  # [batch, seq, 1, dim/2]
    sin = sin[position_ids].unsqueeze(-2)  # [batch, seq, 1, dim/2]
    
    # q and k must be arranged in pairs for rotation
    q_embed_dim = q.shape[-1]
    q_half_dim = q_embed_dim // 2
    
    # Split the embedding dimensions into pairs
    q_half1, q_half2 = q[..., :q_half_dim], q[..., q_half_dim:]
    k_half1, k_half2 = k[..., :q_half_dim], k[..., q_half_dim:]
    
    # Apply rotary embeddings to each pair of dimensions
    # For each pair (a, b), we compute (a*cos - b*sin, a*sin + b*cos)
    q_out_half1 = q_half1 * cos - q_half2 * sin
    q_out_half2 = q_half1 * sin + q_half2 * cos
    k_out_half1 = k_half1 * cos - k_half2 * sin
    k_out_half2 = k_half1 * sin + k_half2 * cos
    
    # Concatenate back to original shape
    q_out = torch.cat([q_out_half1, q_out_half2], dim=-1)
    k_out = torch.cat([k_out_half1, k_out_half2], dim=-1)
    
    return q_out, k_out


class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # For MQA/GQA support
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Precompute rotary position encoding frequencies
        max_seq_len = config.max_position_embeddings
        self.max_seq_len = max_seq_len
        
        # Register frequencies as buffers
        cos, sin = precompute_freqs_cis(self.head_dim, max_seq_len, config.rope_theta)
        self.register_buffer("cos", cos)  # [max_seq_len, dim/2]
        self.register_buffer("sin", sin)  # [max_seq_len, dim/2]
        
        # Create causal mask for attention
        self.register_buffer(
            "causal_mask", 
            torch.triu(torch.ones(max_seq_len, max_seq_len) * -1e9, diagonal=1)
        )

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project queries, keys, values
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for attention computation
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Apply rotary position embeddings
        q_rotary, k_rotary = apply_rotary_pos_emb(q, k, self.cos, self.sin)
        
        # Reshape for attention computation
        q_rotary = q_rotary.transpose(1, 2)  # [batch, heads, seq, dim]
        k_rotary = k_rotary.transpose(1, 2)  # [batch, kv_heads, seq, dim]
        v = v.transpose(1, 2)  # [batch, kv_heads, seq, dim]
        
        # Handle Multi-Query Attention / Grouped-Query Attention
        if self.num_key_value_groups > 1:
            # Repeat k, v for each query in the group
            k_rotary = k_rotary.repeat_interleave(self.num_key_value_groups, dim=1)  # [b, n_heads, seq, head_dim]
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)  # [b, n_heads, seq, head_dim]
        
        # Compute attention scores: [batch, n_heads, seq_len, seq_len]
        attn_scores = torch.matmul(q_rotary, k_rotary.transpose(-1, -2)) / (self.head_dim ** 0.5)
        
        # Apply causal mask - only attend to previous tokens
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        attn_scores = attn_scores + causal_mask
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: [batch, 1, 1, seq_len]
            attn_scores = attn_scores + attention_mask
            
        # Normalize the attention scores to probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attn_probs, v)  # [b, n_heads, seq, head_dim]
        
        # Reshape back to [batch_size, seq_length, hidden_size]
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, -1)
        
        # Final projection
        output = self.o_proj(context)
        
        return output


class ConvaiCausalLM(PreTrainedModel):
    config_class = ConvaiCausalLMConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": GroupedQueryAttention(config),
                "mlp": nn.Sequential(
                    nn.Linear(config.hidden_size, config.intermediate_size),
                    nn.SiLU(),
                    nn.Linear(config.intermediate_size, config.hidden_size)
                ),
                "input_layernorm": nn.LayerNorm(config.hidden_size),
                "post_attention_layernorm": nn.LayerNorm(config.hidden_size)
            }) for _ in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _prepare_attention_mask(self, attention_mask, input_shape, device):
        # Prepare masks for attention
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
            
        # Make broadcastable shape: [batch, 1, 1, seq_len]
        extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Convert to additive mask (0 for valid, -10000 for masked)
        extended_mask = (1.0 - extended_mask) * -10000.0
        
        return extended_mask
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = self._prepare_attention_mask(
                attention_mask, (batch_size, seq_len), device
            )
        
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Apply each layer
        for layer in self.layers:
            residual = hidden_states
            
            # First norm and attention
            hidden_states = layer["input_layernorm"](hidden_states)
            hidden_states = layer["self_attn"](hidden_states, attention_mask)
            hidden_states = residual + hidden_states
            
            # Second norm and MLP
            residual = hidden_states
            hidden_states = layer["post_attention_layernorm"](hidden_states)
            hidden_states = layer["mlp"](hidden_states)
            hidden_states = residual + hidden_states
            
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        return logits


%%writefile train_convaicausallm_large_corpus.py
import os
import torch
import random
import json
import shutil
import time
import argparse
import traceback
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from hindi_embeddings import SentencePieceTokenizerWrapper
from convaicausallm_model import ConvaiCausalLM, ConvaiCausalLMConfig

class StreamingHindiDataset(IterableDataset):
    """Streaming dataset for processing large Hindi corpus file"""
    def __init__(self, file_path, tokenizer, block_size=512, buffer_size=100000, 
                 epoch_size_mb=500, shuffle=True):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.buffer_size = buffer_size
        self.epoch_size_mb = epoch_size_mb  # How many MB to process per epoch
        self.shuffle = shuffle
        
        # Check file existence
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist")
        
        # Calculate file statistics
        self.file_size = os.path.getsize(file_path)
        self.file_size_mb = self.file_size / (1024 * 1024)
        
        print(f"[DATA] File size: {self.file_size_mb:.2f} MB")
        print(f"[DATA] Will process approximately {self.epoch_size_mb} MB per epoch")
        
        # Calculate how much of the file to read per epoch
        self.epoch_fraction = min(1.0, self.epoch_size_mb / self.file_size_mb)
        self.bytes_per_epoch = int(self.file_size * self.epoch_fraction)
        
        print(f"[DATA] Using {self.epoch_fraction*100:.1f}% of file per epoch ({self.bytes_per_epoch/(1024*1024):.1f} MB)")
        
        # Estimate tokens and examples
        avg_bytes_per_token = 4.5  # Estimate based on typical Hindi text
        estimated_tokens_per_epoch = self.bytes_per_epoch / avg_bytes_per_token
        self.estimated_examples_per_epoch = int(estimated_tokens_per_epoch / block_size * 2)  # *2 for stride
        
        print(f"[DATA] Estimated examples per epoch: {self.estimated_examples_per_epoch}")
    
    def __iter__(self):
        # Decide on starting position in file
        if self.file_size > self.bytes_per_epoch and self.shuffle:
            # Start at a random position within the file
            max_start_pos = self.file_size - self.bytes_per_epoch
            start_pos = random.randint(0, max_start_pos)
        else:
            # Read from the beginning if file is smaller than epoch size
            start_pos = 0
        
        print(f"[DATA] Reading from position {start_pos/(1024*1024):.1f} MB")
        
        worker_info = torch.utils.data.get_worker_info()
        
        # Handle multiple workers
        if worker_info is not None:
            # Split the data among workers
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            
            # Each worker gets a slice of the bytes to read
            worker_bytes = self.bytes_per_epoch // num_workers
            worker_start = start_pos + worker_id * worker_bytes
            worker_end = min(worker_start + worker_bytes, self.file_size)
            
            print(f"[DATA] Worker {worker_id} processing {worker_bytes/(1024*1024):.1f} MB")
        else:
            # Single worker case
            worker_start = start_pos
            worker_end = min(start_pos + self.bytes_per_epoch, self.file_size)
        
        # Initialize tokenized buffer
        token_buffer = []
        
        # Open file and seek to the worker's start position
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            f.seek(worker_start)
            
            # Read the worker's portion of the file
            bytes_to_read = worker_end - worker_start
            text = f.read(bytes_to_read)
            
            # Tokenize the text
            tokenized = self.tokenizer.sp_model.EncodeAsIds(text)
            
            # Fill buffer with tokenized text
            token_buffer.extend(tokenized)
            
            # Keep track of position for overlapping sequences
            pos = 0
            stride = self.block_size // 2
            
            # Generate examples with stride
            while pos + self.block_size <= len(token_buffer):
                yield torch.tensor(token_buffer[pos:pos+self.block_size], dtype=torch.long)
                pos += stride
    
    def __len__(self):
        # Return estimated number of examples with a buffer to avoid display issues
        return int(self.estimated_examples_per_epoch * 1.1)


def train_model(args):
    start_time = time.time()
    
    # Detect number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"[INFO] Found {num_gpus} GPUs")
    
    # Load tokenizer
    tokenizer_model_path = os.path.join(args.tokenizer_path, "tokenizer.model")
    print(f"[INFO] Loading tokenizer from: {tokenizer_model_path}")
    
    tokenizer = SentencePieceTokenizerWrapper(tokenizer_model_path)
    print(f"[INFO] Loaded tokenizer with vocab size: {tokenizer.vocab_size}")
    
    # Create streaming dataset with full data utilization
    print(f"[INFO] Creating streaming dataset from: {args.corpus_path}")
    dataset = StreamingHindiDataset(
        file_path=args.corpus_path,
        tokenizer=tokenizer,
        block_size=args.block_size,
        epoch_size_mb=args.epoch_size_mb,
        shuffle=True
    )
    
    # Create dataloader with auto-prefetching
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size * num_gpus,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    # Initialize model
    print(f"[MODEL] Initializing model")
    
    # Try to load from checkpoint first if resuming
    checkpoint_path = None
    if args.resume:
        # Automatically find the latest checkpoint
        checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        if os.path.exists(checkpoint_dir):
            checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("step_")]
            if checkpoints:
                # Sort by step number
                checkpoints.sort(key=lambda x: int(x.split("_")[1]), reverse=True)
                checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0])
                print(f"[RESUME] Found checkpoint: {checkpoint_path}")
    
    # Create new model if not resuming or no checkpoint found
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print(f"[MODEL] Creating new model")
        config = ConvaiCausalLMConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=16,
            num_key_value_heads=4,
            intermediate_size=3072,
            max_position_embeddings=args.block_size
        )
        model = ConvaiCausalLM(config)
        global_step = 0
        
        # Save the config 
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump(config.to_dict(), f, indent=2)
    else:
        # Load from checkpoint
        try:
            print(f"[RESUME] Loading model from checkpoint: {checkpoint_path}")
            config_path = os.path.join(checkpoint_path, "config.json")
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            
            config = ConvaiCausalLMConfig(**config_dict)
            model = ConvaiCausalLM(config)
            
            # Find model file in checkpoint
            model_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.bin')]
            if model_files:
                model_path = os.path.join(checkpoint_path, model_files[0])
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                print(f"[RESUME] Loaded model weights from: {model_path}")
            
            # Extract step number from checkpoint folder
            global_step = int(os.path.basename(checkpoint_path).split("_")[1])
            print(f"[RESUME] Resuming from step: {global_step}")
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint: {e}")
            traceback.print_exc()
            print("[INFO] Creating new model instead")
            
            config = ConvaiCausalLMConfig(
                vocab_size=tokenizer.vocab_size,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=16,
                num_key_value_heads=4,
                intermediate_size=3072,
                max_position_embeddings=args.block_size
            )
            model = ConvaiCausalLM(config)
            global_step = 0
    
    # Use DataParallel for multi-GPU training
    if num_gpus > 1:
        print(f"[MODEL] Using DataParallel across {num_gpus} GPUs")
        model = torch.nn.DataParallel(model)
    
    # Move model to GPU(s)
    model = model.cuda()
    
    # Use mixed precision for faster training
    if args.mixed_precision:
        print("[TRAIN] Using mixed precision training")
        # Use the newer syntax to avoid the warning
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None
        print("[TRAIN] Using full precision training")
    
    # Optimizer with weight decay for certain parameters
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Calculate total steps and adjust for resumption
    total_steps = args.max_steps if args.max_steps > 0 else len(dataloader) * args.num_epochs
    warmup_steps = min(2000, int(total_steps * 0.1))  # More warmup steps
    
    # If resuming, adjust the total steps to account for steps already taken
    if global_step > 0:
        remaining_steps = total_steps - global_step
        print(f"[RESUME] Adjusting schedule: {remaining_steps} steps remaining out of {total_steps} total")
        total_steps = remaining_steps
    
    print(f"[TRAIN] Initial learning rate: {args.learning_rate}")
    print(f"[TRAIN] Total training steps: {total_steps}")
    print(f"[TRAIN] Warmup steps: {warmup_steps}")
    
    # Use cosine schedule with warmup - smooth decay that avoids dropping to zero too quickly
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=args.lr_cycles,
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Checkpointing
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Track best model
    best_loss = float('inf')
    
    # Training loop
    training_start = time.time()
    
    print(f"\n{'='*40}")
    print(f"[TRAIN] Starting training at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*40}\n")
    
    try:
        for epoch in range(args.num_epochs):
            epoch_start_time = time.time()
            model.train()
            total_loss = 0
            steps_in_epoch = 0
            
            # Print epoch header
            print(f"\n{'-'*20} Epoch {epoch+1}/{args.num_epochs} {'-'*20}")
            
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to GPU
                batch = batch.cuda(non_blocking=True)
                
                # Prepare inputs and labels
                inputs = batch[:, :-1]
                labels = batch[:, 1:]
                
                # Forward pass with mixed precision if enabled
                if scaler:
                    with torch.amp.autocast('cuda'):  # Use newer API
                        outputs = model(inputs)
                        loss = torch.nn.functional.cross_entropy(
                            outputs.reshape(-1, outputs.size(-1)),
                            labels.reshape(-1)
                        )
                else:
                    outputs = model(inputs)
                    loss = torch.nn.functional.cross_entropy(
                        outputs.reshape(-1, outputs.size(-1)),
                        labels.reshape(-1)
                    )
                
                # Backward pass with mixed precision if enabled
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                    optimizer.step()
                
                # Step scheduler after optimizer
                scheduler.step()
                
                # Get current learning rate with proper formatting
                current_lr = scheduler.get_last_lr()[0]
                
                optimizer.zero_grad(set_to_none=True)
                
                # Track metrics
                batch_loss = loss.item()
                total_loss += batch_loss
                steps_in_epoch += 1
                global_step += 1
                
                # Log progress with standard LLM metrics
                if batch_idx % args.log_interval == 0:
                    elapsed = time.time() - training_start
                    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                    steps_per_sec = global_step / elapsed if elapsed > 0 else 0
                    
                    print(f"Step: {global_step:6d} | "
                          f"Loss: {batch_loss:.4f} | "
                          f"LR: {current_lr:.8f} | "  # More decimal places
                          f"Time: {elapsed_str} | "
                          f"Steps/sec: {steps_per_sec:.2f} | "
                          f"Epoch: {epoch+1}/{args.num_epochs} | "
                          f"Batch: {batch_idx+1}")  # Remove denominator to avoid confusion
                
                # Save checkpoint with better error handling
                if global_step % args.save_interval == 0:
                    try:
                        checkpoint_path = os.path.join(checkpoint_dir, f"step_{global_step}")
                        os.makedirs(checkpoint_path, exist_ok=True)
                        
                        # Save unwrapped model if using DataParallel
                        if num_gpus > 1:
                            # Create a temporary file first
                            temp_path = os.path.join(checkpoint_path, "temp_model.bin")
                            torch.save(model.module.state_dict(), temp_path)
                            
                            # If save was successful, finalize it
                            final_path = os.path.join(checkpoint_path, "pytorch_model.bin")
                            shutil.move(temp_path, final_path)
                            
                            # Save config too
                            with open(os.path.join(checkpoint_path, "config.json"), "w") as f:
                                json.dump(config.to_dict(), f, indent=2)
                        else:
                            # Create a temporary file first
                            temp_path = os.path.join(checkpoint_path, "temp_model.bin")
                            torch.save(model.state_dict(), temp_path)
                            
                            # If save was successful, finalize it
                            final_path = os.path.join(checkpoint_path, "pytorch_model.bin")
                            shutil.move(temp_path, final_path)
                            
                            # Save config too
                            with open(os.path.join(checkpoint_path, "config.json"), "w") as f:
                                json.dump(config.to_dict(), f, indent=2)
                        
                        print(f"[CHECKPOINT] Successfully saved checkpoint to {checkpoint_path}")
                    except Exception as e:
                        print(f"[ERROR] Failed to save checkpoint: {e}")
                        traceback.print_exc()
                
                # Check if we've reached max steps
                if args.max_steps > 0 and global_step >= args.max_steps:
                    print(f"[TRAIN] Reached maximum steps ({args.max_steps}). Stopping training.")
                    break
            
            # End of epoch
            avg_loss = total_loss / steps_in_epoch
            epoch_time = time.time() - epoch_start_time
            
            print(f"\n{'-'*20} Epoch {epoch+1} Summary {'-'*20}")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Epoch time: {epoch_time:.2f}s")
            print(f"Current learning rate: {current_lr:.8f}")
            print(f"Steps per second: {steps_in_epoch / epoch_time:.2f}")
            
            # Save if best
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_path = os.path.join(args.output_dir, "best_model")
                try:
                    os.makedirs(best_model_path, exist_ok=True)
                    
                    # Save unwrapped model if using DataParallel
                    if num_gpus > 1:
                        # Create a temporary file first 
                        temp_path = os.path.join(best_model_path, "temp_model.bin")
                        torch.save(model.module.state_dict(), temp_path)
                        
                        # If save was successful, finalize it
                        final_path = os.path.join(best_model_path, "pytorch_model.bin") 
                        shutil.move(temp_path, final_path)
                        
                        # Save config too
                        with open(os.path.join(best_model_path, "config.json"), "w") as f:
                            json.dump(config.to_dict(), f, indent=2)
                    else:
                        # Create a temporary file first
                        temp_path = os.path.join(best_model_path, "temp_model.bin")
                        torch.save(model.state_dict(), temp_path)
                        
                        # If save was successful, finalize it
                        final_path = os.path.join(best_model_path, "pytorch_model.bin")
                        shutil.move(temp_path, final_path)
                        
                        # Save config too
                        with open(os.path.join(best_model_path, "config.json"), "w") as f:
                            json.dump(config.to_dict(), f, indent=2)
                    
                    print(f"[CHECKPOINT] New best model saved with loss: {best_loss:.4f}")
                except Exception as e:
                    print(f"[ERROR] Failed to save best model: {e}")
                    traceback.print_exc()
            
            # Check if we've reached max steps
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        
        # Save final model
        print(f"\n{'='*40}")
        print(f"[TRAIN] Training complete! Saving final model...")
        
        try:
            # Save unwrapped model if using DataParallel
            if num_gpus > 1:
                # Create a temporary file first
                temp_path = os.path.join(args.output_dir, "temp_model.bin")
                torch.save(model.module.state_dict(), temp_path)
                
                # If save was successful, finalize it
                final_path = os.path.join(args.output_dir, "pytorch_model.bin")
                shutil.move(temp_path, final_path)
            else:
                # Create a temporary file first
                temp_path = os.path.join(args.output_dir, "temp_model.bin")
                torch.save(model.state_dict(), temp_path)
                
                # If save was successful, finalize it
                final_path = os.path.join(args.output_dir, "pytorch_model.bin")
                shutil.move(temp_path, final_path)
            
            # Copy tokenizer to output directory
            tokenizer_out_path = os.path.join(args.output_dir, "tokenizer.model")
            shutil.copy(tokenizer_model_path, tokenizer_out_path)
            print(f"[INFO] Copied tokenizer to {tokenizer_out_path}")
            
            # Save config in final dir too
            with open(os.path.join(args.output_dir, "config.json"), "w") as f:
                json.dump(config.to_dict(), f, indent=2)
                
            print(f"[CHECKPOINT] Successfully saved final model to {args.output_dir}")
        except Exception as e:
            print(f"[ERROR] Failed to save final model: {e}")
            traceback.print_exc()
        
        # Print training summary
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\n{'='*40}")
        print(f"[SUMMARY] Training completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[SUMMARY] Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        print(f"[SUMMARY] Total steps: {global_step}")
        print(f"[SUMMARY] Final loss: {avg_loss:.4f}")
        print(f"[SUMMARY] Best loss: {best_loss:.4f}")
        print(f"[SUMMARY] Final learning rate: {current_lr:.8f}")
        print(f"[SUMMARY] Model saved to: {args.output_dir}")
        print(f"{'='*40}\n")
    
    except Exception as e:
        print(f"[ERROR] Error during training: {e}")
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(description="Train ConvaiCausalLM on Hindi corpus")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to the corpus text file")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the directory containing the tokenizer")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the model")
    parser.add_argument("--block_size", type=int, default=512, help="Block size for training")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    parser.add_argument("--lr_cycles", type=float, default=0.5, help="Number of cycles for cosine scheduler")
    parser.add_argument("--max_steps", type=int, default=0, help="Maximum number of steps (0 for no limit)")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval (batches)")
    parser.add_argument("--save_interval", type=int, default=1000, help="Save interval (steps)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes for data loading")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--epoch_size_mb", type=int, default=500, help="MB of data to process per epoch")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Print setup info
    print(f"\n{'='*40}")
    print(f"[SETUP] Hindi LLM Training")
    print(f"[SETUP] PyTorch version: {torch.__version__}")
    print(f"[SETUP] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[SETUP] CUDA version: {torch.version.cuda}")
        print(f"[SETUP] GPU(s): {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"[SETUP] GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"{'='*40}\n")
    
    # Run training
    try:
        train_model(args)
    except Exception as e:
        print(f"[ERROR] Training failed with error: {e}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()





!rm -rf /kaggle/working/hindi-embedding-foundational-model/convaicausallm-output


!python train_convaicausallm_large_corpus.py \
  --corpus_path /kaggle/working/hindi-embedding-foundational-model/hindi_corpus.txt \
  --tokenizer_path /kaggle/working/hindi-embedding-foundational-model \
  --output_dir ./convaicausallm-output \
  --batch_size 16 \
  --num_epochs 1 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --lr_cycles 0.5 \
  --mixed_precision \
  --epoch_size_mb 500 \
  --log_interval 25 \
  --max_steps 8000



%%writefile convert_to_safetensor.py
import os
import argparse
import torch
from safetensors.torch import save_file
from convaicausallm_model import ConvaiCausalLMConfig, ConvaiCausalLM

def convert_model(model_path, output_path=None):
    """
    Convert a PyTorch .bin model to SafeTensors format.
    
    Args:
        model_path: Path to the directory containing the model files
        output_path: Path to save the converted model (defaults to original path)
    """
    print(f"Converting model in {model_path} to SafeTensors format")
    
    # Check if model exists
    pytorch_model_path = os.path.join(model_path, "pytorch_model.bin")
    if not os.path.exists(pytorch_model_path):
        raise FileNotFoundError(f"PyTorch model not found at {pytorch_model_path}")
    
    # Load config
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")
    
    print("Loading model config...")
    import json
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    config = ConvaiCausalLMConfig(**config_dict)
    
    # Create output path if not specified
    if output_path is None:
        output_path = model_path
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize model
    print("Initializing model...")
    model = ConvaiCausalLM(config)
    
    # Load state dict
    print(f"Loading weights from {pytorch_model_path}...")
    state_dict = torch.load(pytorch_model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    
    # Copy config and tokenizer
    if output_path != model_path:
        print("Copying config and tokenizer...")
        import shutil
        shutil.copy(config_path, os.path.join(output_path, "config.json"))
        
        tokenizer_path = os.path.join(model_path, "tokenizer.model")
        if os.path.exists(tokenizer_path):
            shutil.copy(tokenizer_path, os.path.join(output_path, "tokenizer.model"))
    
    # Convert to SafeTensors
    safetensors_path = os.path.join(output_path, "model.safetensors")
    print(f"Converting and saving to {safetensors_path}...")
    
    # Get state dict from model
    state_dict = model.state_dict()
    
    # Save with safetensors
    save_file(state_dict, safetensors_path)
    print(f"Model successfully converted and saved to {safetensors_path}")
    
    # Verify saved model
    safetensors_size = os.path.getsize(safetensors_path) / (1024 * 1024)
    pytorch_size = os.path.getsize(pytorch_model_path) / (1024 * 1024)
    
    print(f"SafeTensors model size: {safetensors_size:.2f} MB")
    print(f"PyTorch model size: {pytorch_size:.2f} MB")
    
    return safetensors_path

def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to SafeTensors format")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the directory containing the model files")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save the converted model (defaults to original path)")
    
    args = parser.parse_args()
    convert_model(args.model_path, args.output_path)

if __name__ == "__main__":
    main()


%%writefile hindi_llm_inference_safetensors.py
import os
import torch
import argparse
import time
from hindi_embeddings import SentencePieceTokenizerWrapper
from convaicausallm_model import ConvaiCausalLM, ConvaiCausalLMConfig
from safetensors.torch import load_file

class HindiLLMGenerator:
    """Text generator for Hindi LLM model using SafeTensors format"""
    
    def __init__(self, model_path, device=None):
        """
        Initialize the Hindi text generator.
        
        Args:
            model_path: Path to the directory containing model files
            device: Device to use for inference ('cuda', 'cpu', or specific GPU index)
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        tokenizer_path = os.path.join(model_path, "tokenizer.model")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        
        self.tokenizer = SentencePieceTokenizerWrapper(tokenizer_path)
        print(f"Loaded tokenizer with vocab size: {self.tokenizer.vocab_size}")
        
        # Load model config
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found at {config_path}")
            
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            
        self.config = ConvaiCausalLMConfig(**config_dict)
        print(f"Loaded model config: hidden_size={self.config.hidden_size}, layers={self.config.num_hidden_layers}")
        
        # Load model - try safetensors first, fall back to PyTorch bin if needed
        safetensors_path = os.path.join(model_path, "model.safetensors")
        pytorch_path = os.path.join(model_path, "pytorch_model.bin")
        
        self.model = ConvaiCausalLM(self.config)
        
        # Check which format is available and load accordingly
        if os.path.exists(safetensors_path):
            print(f"Loading model from SafeTensors: {safetensors_path}")
            # Load state dict using safetensors
            state_dict = load_file(safetensors_path, device="cpu")
            self.model.load_state_dict(state_dict)
        elif os.path.exists(pytorch_path):
            print(f"Loading model from PyTorch bin: {pytorch_path}")
            # Fall back to PyTorch format
            self.model.load_state_dict(torch.load(pytorch_path, map_location="cpu"))
        else:
            raise FileNotFoundError(f"No model weights found in {model_path}")
        
        # Move model to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded and ready for generation")
    
    def generate(self, prompt, max_length=100, temperature=0.8, top_k=50, top_p=0.9, 
                 repetition_penalty=1.1, do_sample=True, num_return_sequences=1):
        """
        Generate text based on the input prompt.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text (including prompt)
            temperature: Temperature for sampling (higher = more random)
            top_k: Number of highest probability tokens to consider for sampling
            top_p: Cumulative probability threshold for nucleus sampling
            repetition_penalty: Penalty for repeating tokens
            do_sample: If False, use greedy decoding instead of sampling
            num_return_sequences: Number of sequences to generate
            
        Returns:
            List of generated text sequences
        """
        # Tokenize the prompt
        input_ids = self.tokenizer.sp_model.EncodeAsIds(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        # Store original prompt length to return only the generated part
        prompt_length = len(input_ids)
        
        # Keep track of all generated sequences
        all_sequences = []
        
        # Generate sequences
        for _ in range(num_return_sequences):
            # Start with the input tensor
            output_sequence = input_tensor.clone()
            
            # Generate tokens one by one
            for _ in range(max_length - len(input_ids)):
                with torch.no_grad():
                    # Get the model's output for the current sequence
                    outputs = self.model(output_sequence)
                    next_token_logits = outputs[0, -1, :]
                    
                    # Apply temperature
                    if temperature > 0:
                        next_token_logits = next_token_logits / temperature
                    
                    # Apply repetition penalty
                    if repetition_penalty > 1.0:
                        for token_id in output_sequence[0].tolist():
                            next_token_logits[token_id] /= repetition_penalty
                    
                    # Filter with top-k sampling
                    if top_k > 0:
                        top_k_values, top_k_indices = torch.topk(next_token_logits, top_k)
                        next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                        next_token_logits.scatter_(0, top_k_indices, top_k_values)
                    
                    # Filter with top-p/nucleus sampling
                    if top_p < 1.0 and do_sample:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # Shift the indices to the right to keep the first token above the threshold
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Sample or choose the next token
                    if do_sample:
                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                    
                    # Add the next token to the sequence
                    output_sequence = torch.cat([output_sequence, next_token.unsqueeze(0)], dim=1)
                    
                    # Check if we've generated an end token
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
            
            # Decode the generated sequence
            generated_ids = output_sequence[0].tolist()
            generated_text = self.tokenizer.sp_model.DecodeIds(generated_ids)
            
            # Add to the list of sequences
            all_sequences.append(generated_text)
        
        return all_sequences
    
    def interactive_generation(self):
        """Interactive text generation loop"""
        print("\n" + "="*50)
        print("Hindi LLM Interactive Text Generation")
        print("Type 'exit' to quit, 'settings' to change generation parameters")
        print("="*50 + "\n")
        
        # Default generation parameters
        params = {
            "max_length": 100,
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "do_sample": True
        }
        
        while True:
            try:
                prompt = input("\nPrompt: ")
                
                if prompt.lower() == 'exit':
                    print("Exiting interactive mode")
                    break
                    
                elif prompt.lower() == 'settings':
                    print("\nCurrent generation settings:")
                    for param, value in params.items():
                        print(f"{param}: {value}")
                    
                    param_to_change = input("\nEnter parameter to change (or 'done'): ")
                    while param_to_change.lower() != 'done':
                        if param_to_change in params:
                            new_value = input(f"Enter new value for {param_to_change}: ")
                            try:
                                # Convert to appropriate type
                                if param_to_change == "do_sample":
                                    params[param_to_change] = new_value.lower() in ['true', 'yes', '1', 't', 'y']
                                else:
                                    params[param_to_change] = type(params[param_to_change])(new_value)
                                print(f"Updated {param_to_change} to {params[param_to_change]}")
                            except ValueError:
                                print(f"Invalid value for {param_to_change}. Using previous value.")
                        else:
                            print(f"Unknown parameter: {param_to_change}")
                        
                        param_to_change = input("Enter parameter to change (or 'done'): ")
                    
                    continue
                
                if not prompt:
                    print("Please enter a prompt")
                    continue
                
                print("\nGenerating...")
                start_time = time.time()
                
                # Generate text
                generated_texts = self.generate(
                    prompt=prompt,
                    **params
                )
                
                generation_time = time.time() - start_time
                
                # Print generated texts
                print("\n" + "-"*50)
                for i, text in enumerate(generated_texts):
                    print(f"[Generated Text {i+1}]")
                    print(text)
                    print()
                
                print(f"Generation completed in {generation_time:.2f}s")
                print("-"*50)
                
            except KeyboardInterrupt:
                print("\nExiting interactive mode")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Hindi LLM text generation using SafeTensors")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--device", type=str, default=None, help="Device to use for inference (cuda/cpu)")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for generation")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--no_sample", action="store_true", help="Use greedy decoding instead of sampling")
    parser.add_argument("--num_sequences", type=int, default=1, help="Number of sequences to generate")
    
    args = parser.parse_args()
    
    # Create text generator
    generator = HindiLLMGenerator(
        model_path=args.model_path,
        device=args.device
    )
    
    # Run in interactive mode or generate from prompt
    if args.interactive:
        generator.interactive_generation()
    elif args.prompt:
        generated_texts = generator.generate(
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            do_sample=not args.no_sample,
            num_return_sequences=args.num_sequences
        )
        
        # Print generated texts
        print("\n" + "-"*50)
        for i, text in enumerate(generated_texts):
            print(f"[Generated Text {i+1}]")
            print(text)
            print()
        print("-"*50)
    else:
        print("Please provide a prompt or use --interactive")

if __name__ == "__main__":
    main()


!python convert_to_safetensor.py --model_path ./convaicausallm-output


!python hindi_llm_inference_safetensors.py --model_path ./convaicausallm-output --prompt "भारत एक विशाल देश है" --max_length 150 --temperature 0.9


%%writefile upload_to_huggingface.py
import os
import argparse
import shutil
import json
from huggingface_hub import HfApi, upload_folder, create_repo

def create_readme(model_path, output_path):
    """Create README.md with model usage instructions"""
    readme_content = """# Hindi-CausalLM

A Hindi language model based on ConvaiCausalLM architecture with 12 transformer layers and 768 hidden dimensions, trained on a large Hindi corpus.

## Model Description

This model is a Hindi language generation model based on a modern causal language model architecture. It has been trained on a large corpus of Hindi text to generate coherent and contextually relevant Hindi text. The model uses Rotary Position Embeddings (RoPE) to enhance its understanding of token positions.

## Model Architecture

- **Architecture:** ConvaiCausalLM
- **Layers:** 12 transformer layers
- **Hidden size:** 768
- **Attention heads:** 16
- **Key-value heads:** 4 (grouped-query attention)
- **Position encoding:** Rotary Position Embeddings (RoPE)
- **Vocabulary size:** 16,000 tokens
- **Context length:** 512 tokens
- **Parameters:** ~125M

## Usage

You can use this model with the following code:

```python
import torch
import math
import os
from hindi_embeddings import SentencePieceTokenizerWrapper
from safetensors.torch import load_file
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig


class ConvaiCausalLMConfig(PretrainedConfig):
    model_type = "convaicausallm"
    
    def __init__(
        self,
        vocab_size=16000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=16,
        num_key_value_heads=4,
        intermediate_size=3072,
        hidden_act="silu",
        max_position_embeddings=512,
        rope_theta=10000.0,  # Base parameter for RoPE
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta


def precompute_freqs_cis(dim, end, theta=10000.0):
  
    # Ensure dim is even for complex numbers
    assert dim % 2 == 0, "Dimension must be even"
    
    # Create position indices for caching
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end).float()
    freqs = torch.outer(t, freqs)  # [end, dim/2]
    
    # Create complex exponentials (cos, sin pairs)
    cos, sin = torch.cos(freqs), torch.sin(freqs)
    return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):

    # Extract shapes
    batch, seq_len, n_heads, head_dim = q.shape
    _, kv_seq_len, n_kv_heads, _ = k.shape
    
    # Handle position IDs or use sequential positions
    if position_ids is None:
        # Default: Just use sequential positions
        position_ids = torch.arange(seq_len, device=q.device)
        position_ids = position_ids.unsqueeze(0).expand(batch, -1)
        
    # Get the cosine and sine for the positions we're using
    cos = cos[position_ids].unsqueeze(-2)  # [batch, seq, 1, dim/2]
    sin = sin[position_ids].unsqueeze(-2)  # [batch, seq, 1, dim/2]
    
    # q and k must be arranged in pairs for rotation
    q_embed_dim = q.shape[-1]
    q_half_dim = q_embed_dim // 2
    
    # Split the embedding dimensions into pairs
    q_half1, q_half2 = q[..., :q_half_dim], q[..., q_half_dim:]
    k_half1, k_half2 = k[..., :q_half_dim], k[..., q_half_dim:]
    
    # Apply rotary embeddings to each pair of dimensions
    # For each pair (a, b), we compute (a*cos - b*sin, a*sin + b*cos)
    q_out_half1 = q_half1 * cos - q_half2 * sin
    q_out_half2 = q_half1 * sin + q_half2 * cos
    k_out_half1 = k_half1 * cos - k_half2 * sin
    k_out_half2 = k_half1 * sin + k_half2 * cos
    
    # Concatenate back to original shape
    q_out = torch.cat([q_out_half1, q_out_half2], dim=-1)
    k_out = torch.cat([k_out_half1, k_out_half2], dim=-1)
    
    return q_out, k_out


class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # For MQA/GQA support
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Precompute rotary position encoding frequencies
        max_seq_len = config.max_position_embeddings
        self.max_seq_len = max_seq_len
        
        # Register frequencies as buffers
        cos, sin = precompute_freqs_cis(self.head_dim, max_seq_len, config.rope_theta)
        self.register_buffer("cos", cos)  # [max_seq_len, dim/2]
        self.register_buffer("sin", sin)  # [max_seq_len, dim/2]
        
        # Create causal mask for attention
        self.register_buffer(
            "causal_mask", 
            torch.triu(torch.ones(max_seq_len, max_seq_len) * -1e9, diagonal=1)
        )

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project queries, keys, values
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for attention computation
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Apply rotary position embeddings
        q_rotary, k_rotary = apply_rotary_pos_emb(q, k, self.cos, self.sin)
        
        # Reshape for attention computation
        q_rotary = q_rotary.transpose(1, 2)  # [batch, heads, seq, dim]
        k_rotary = k_rotary.transpose(1, 2)  # [batch, kv_heads, seq, dim]
        v = v.transpose(1, 2)  # [batch, kv_heads, seq, dim]
        
        # Handle Multi-Query Attention / Grouped-Query Attention
        if self.num_key_value_groups > 1:
            # Repeat k, v for each query in the group
            k_rotary = k_rotary.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Compute attention scores
        attn_scores = torch.matmul(q_rotary, k_rotary.transpose(-1, -2)) / (self.head_dim ** 0.5)
        
        # Apply causal mask - only attend to previous tokens
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        attn_scores = attn_scores + causal_mask
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
            
        # Normalize the attention scores to probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attn_probs, v)  # [b, n_heads, seq, head_dim]
        
        # Reshape back to [batch_size, seq_length, hidden_size]
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, -1)
        
        # Final projection
        output = self.o_proj(context)
        
        return output


class ConvaiCausalLM(PreTrainedModel):
    config_class = ConvaiCausalLMConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": GroupedQueryAttention(config),
                "mlp": nn.Sequential(
                    nn.Linear(config.hidden_size, config.intermediate_size),
                    nn.SiLU(),
                    nn.Linear(config.intermediate_size, config.hidden_size)
                ),
                "input_layernorm": nn.LayerNorm(config.hidden_size),
                "post_attention_layernorm": nn.LayerNorm(config.hidden_size)
            }) for _ in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _prepare_attention_mask(self, attention_mask, input_shape, device):
        # Prepare masks for attention
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
            
        # Make broadcastable shape: [batch, 1, 1, seq_len]
        extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Convert to additive mask (0 for valid, -10000 for masked)
        extended_mask = (1.0 - extended_mask) * -10000.0
        
        return extended_mask
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = self._prepare_attention_mask(
                attention_mask, (batch_size, seq_len), device
            )
        
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Apply each layer
        for layer in self.layers:
            residual = hidden_states
            
            # First norm and attention
            hidden_states = layer["input_layernorm"](hidden_states)
            hidden_states = layer["self_attn"](hidden_states, attention_mask)
            hidden_states = residual + hidden_states
            
            # Second norm and MLP
            residual = hidden_states
            hidden_states = layer["post_attention_layernorm"](hidden_states)
            hidden_states = layer["mlp"](hidden_states)
            hidden_states = residual + hidden_states
            
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        return logits


class HindiLLMGenerator:
    def __init__(self, model_path, device=None):
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        tokenizer_path = os.path.join(model_path, "tokenizer.model")
        self.tokenizer = SentencePieceTokenizerWrapper(tokenizer_path)
        
        # Load model config
        config_path = os.path.join(model_path, "config.json")
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            
        self.config = ConvaiCausalLMConfig(**config_dict)
        
        # Load model - try safetensors first, fall back to PyTorch bin if needed
        safetensors_path = os.path.join(model_path, "model.safetensors")
        pytorch_path = os.path.join(model_path, "pytorch_model.bin")
        
        self.model = ConvaiCausalLM(self.config)
        
        # Check which format is available and load accordingly
        if os.path.exists(safetensors_path):
            print(f"Loading model from SafeTensors")
            state_dict = load_file(safetensors_path, device="cpu")
            self.model.load_state_dict(state_dict)
        elif os.path.exists(pytorch_path):
            print(f"Loading model from PyTorch bin")
            self.model.load_state_dict(torch.load(pytorch_path, map_location="cpu"))
        
        # Move model to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()
    
    def generate(self, prompt, max_length=100, temperature=0.8, top_k=50, top_p=0.9, 
                 repetition_penalty=1.1, do_sample=True):
        # Tokenize the prompt
        input_ids = self.tokenizer.sp_model.EncodeAsIds(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        # Start with the input tensor
        output_sequence = input_tensor.clone()
        
        # Generate tokens one by one
        for _ in range(max_length - len(input_ids)):
            with torch.no_grad():
                # Get the model's output for the current sequence
                outputs = self.model(output_sequence)
                next_token_logits = outputs[0, -1, :]
                
                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply repetition penalty
                if repetition_penalty > 1.0:
                    for token_id in output_sequence[0].tolist():
                        next_token_logits[token_id] /= repetition_penalty
                
                # Filter with top-k sampling
                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(0, top_k_indices, top_k_values)
                
                # Filter with top-p/nucleus sampling
                if top_p < 1.0 and do_sample:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample or choose the next token
                if do_sample:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                
                # Add the next token to the sequence
                output_sequence = torch.cat([output_sequence, next_token.unsqueeze(0)], dim=1)
                
                # Check if we've generated an end token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode the generated sequence
        generated_ids = output_sequence[0].tolist()
        generated_text = self.tokenizer.sp_model.DecodeIds(generated_ids)
        
        return generated_text

# Example usage
if __name__ == "__main__":
    generator = HindiLLMGenerator("path/to/model")
    result = generator.generate("भारत एक विशाल देश है")
    print(result)
```

## Example Prompts

Try the model with these example prompts:

```
भारत एक विशाल देश है
मुझे हिंदी में एक कहानी सुनाओ
आज का मौसम बहुत अच्छा है
हिंदी साहित्य की प्रमुख विशेषताएं
```

## Dependencies

- PyTorch
- SentencePiece
- safetensors (for faster model loading)
- The model's custom implementation files (hindi_embeddings.py and convaicausallm_model.py)

## Model Limitations

- The model has been trained on a specific corpus of Hindi text and may not perform well on certain domains or topics not well-represented in the training data.
- It may sometimes generate repetitive or less coherent text for longer generations.
- The model does not have knowledge of events after its training cutoff.

## Ethical Considerations

This model should be used responsibly. Users should be aware that language models can sometimes generate content that reflects biases present in their training data.

## Contact

For any questions or feedback about this model, please reach out through the Hugging Face community tab.
"""

    # Write the README.md file
    readme_path = os.path.join(output_path, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print(f"Created README.md at {readme_path}")
    return readme_path

def create_modelcard(model_path, output_path):
    """Create model card with information about the model"""
    # Load config to extract model details
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {}  # Default empty config if not found
    
    # Extract relevant information or use defaults
    hidden_size = config.get("hidden_size", 768)
    num_layers = config.get("num_hidden_layers", 12)
    num_heads = config.get("num_attention_heads", 16)
    kv_heads = config.get("num_key_value_heads", 4)
    vocab_size = config.get("vocab_size", 16000)
    
    # Calculate approximate parameter count (rough estimation)
    embedding_params = vocab_size * hidden_size  # Token embeddings
    attention_params = num_layers * (3 * hidden_size * hidden_size)  # Self-attention projections
    ffn_params = num_layers * (4 * hidden_size * hidden_size)  # Feed-forward networks
    other_params = hidden_size * vocab_size  # Final projection
    total_params = (embedding_params + attention_params + ffn_params + other_params) / 1000000  # In millions
    
    modelcard_content = f"""---
language:
- hi
tags:
- hindi
- text-generation
- causal-lm
- lm
- rope
license: mit
datasets:
- custom_hindi_corpus
---

# Hindi-CausalLM

A Hindi language generation model with the following specifications:

## Model Architecture
- **Type**: Causal Language Model with Transformer architecture
- **Hidden size**: {hidden_size}
- **Layers**: {num_layers}
- **Attention heads**: {num_heads}
- **Key-value heads**: {kv_heads} (using grouped-query attention)
- **Position encoding**: Rotary Position Embeddings (RoPE)
- **Vocabulary size**: {vocab_size}
- **Parameters**: ~{total_params:.1f}M
- **Context window**: 512 tokens
- **Trained on**: Large corpus of Hindi text

## Training

The model was trained on a large corpus of Hindi text using a cosine learning rate schedule with warmup. Training utilized mixed-precision and distributed data parallel across multiple GPUs.

## Capabilities

This model can:
- Generate coherent Hindi text
- Continue text from a given prompt
- Create stories, explanations, and other content in Hindi

## Limitations

- Performance varies based on the similarity of the input to the training data
- May occasionally generate repetitive content for longer texts
- May produce grammatically incorrect Hindi in some contexts
- Has no knowledge of events beyond its training corpus

## Intended Use

This model is intended for Hindi language generation tasks, creative writing assistance, and as a foundation for fine-tuning on specific tasks.

## Ethical Considerations

Users should be aware that like all language models, this model may reproduce biases or generate problematic content in certain contexts.
"""
    
    # Write the model card file
    modelcard_path = os.path.join(output_path, "README.md")  # HF uses README.md as the model card
    with open(modelcard_path, "w", encoding="utf-8") as f:
        f.write(modelcard_content)
    
    print(f"Created model card at {modelcard_path}")
    return modelcard_path

def upload_to_huggingface(model_path, username, token, model_name=None):
    """Upload model to Hugging Face Hub"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory {model_path} not found")
    
    # Set default model name if not provided
    if model_name is None:
        model_name = "hindi-causal-lm"
    
    # Create full repo name
    repo_id = f"{username}/{model_name}"
    print(f"Preparing to upload model to {repo_id}")
    
    # Create a temporary directory for upload
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Created temporary directory: {tmp_dir}")
        
        # Copy model files to temp directory
        for file_name in ["config.json", "tokenizer.model", "pytorch_model.bin", "model.safetensors"]:
            src_path = os.path.join(model_path, file_name)
            if os.path.exists(src_path):
                dst_path = os.path.join(tmp_dir, file_name)
                shutil.copy(src_path, dst_path)
                print(f"Copied {file_name} to temporary directory")
        
        # Create README.md (will serve as the model card for HF)
        create_modelcard(model_path, tmp_dir)
        
        # Initialize Hugging Face API
        api = HfApi(token=token)
        
        # Create or update repository
        print(f"Creating/updating repository {repo_id}...")
        create_repo(repo_id, token=token, exist_ok=True)
        
        # Upload files to Hugging Face
        print(f"Uploading files to {repo_id}...")
        upload_folder(
            folder_path=tmp_dir,
            repo_id=repo_id,
            token=token,
            commit_message="Upload Hindi CausalLM model with RoPE",
        )
        
        print(f"Upload complete! Model available at: https://huggingface.co/{repo_id}")
        return f"https://huggingface.co/{repo_id}"

def main():
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face Hub")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the model directory")
    parser.add_argument("--username", type=str, required=True,
                        help="Hugging Face username")
    parser.add_argument("--token", type=str, required=True,
                        help="Hugging Face API token")
    parser.add_argument("--model_name", type=str, default="hindi-causal-lm",
                        help="Name for the model on Hugging Face Hub")
    
    args = parser.parse_args()
    
    # Upload to Hugging Face
    upload_to_huggingface(
        model_path=args.model_path,
        username=args.username,
        token=args.token,
        model_name=args.model_name
    )

if __name__ == "__main__":
    main()


%%writefile fix_config.py
import json
import os
import argparse

def fix_config(config_path, output_path=None):
    """Fix the config.json file to be compatible with Hugging Face Hub"""
    
    # Use the same path for output if not specified
    if output_path is None:
        output_path = config_path
    
    print(f"Reading config from {config_path}")
    
    # Read the existing config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Fix the architectures field
    config['architectures'] = ["ConvaiCausalLM"]
    
    # Set model type
    config['model_type'] = "convaicausallm"
    
    # Set token IDs based on common values or tokenizer config
    # These values should match your tokenizer
    config['bos_token_id'] = 1  # Usually <s>
    config['eos_token_id'] = 2  # Usually </s>
    config['pad_token_id'] = 0  # Usually <pad>
    
    # Add auto_map to help with loading
    config['auto_map'] = {
        "AutoConfig": "configuration_convaicausallm.ConvaiCausalLMConfig",
        "AutoModelForCausalLM": "modeling_convaicausallm.ConvaiCausalLM"
    }
    
    # Clean up unnecessary generation parameters from model config
    generation_params = [
        "max_length", "min_length", "do_sample", "early_stopping", 
        "num_beams", "num_beam_groups", "diversity_penalty", "temperature",
        "top_k", "top_p", "typical_p", "repetition_penalty", "length_penalty",
        "no_repeat_ngram_size", "encoder_no_repeat_ngram_size", "bad_words_ids",
        "num_return_sequences", "output_scores", "return_dict_in_generate",
        "forced_bos_token_id", "forced_eos_token_id", "remove_invalid_values",
        "exponential_decay_length_penalty", "suppress_tokens", "begin_suppress_tokens"
    ]
    
    # Add these parameters to task_specific_params instead
    task_params = {}
    for param in generation_params:
        if param in config:
            task_params[param] = config[param]
            # Keep some essential ones in both places
            if param not in ["temperature", "top_k", "top_p", "repetition_penalty"]:
                config.pop(param, None)
    
    # Add task-specific parameters in a structured way
    config["task_specific_params"] = {
        "text-generation": task_params
    }
    
    # Write the fixed config
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Fixed config saved to {output_path}")
    return config

def main():
    parser = argparse.ArgumentParser(description="Fix config.json for Hugging Face compatibility")
    parser.add_argument("--config_path", type=str, required=True,
                       help="Path to the config.json file")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Path to save the fixed config.json (defaults to the same path)")
    
    args = parser.parse_args()
    fix_config(args.config_path, args.output_path)

if __name__ == "__main__":
    main()


!python fix_config.py --config_path /kaggle/working/hindi-embedding-foundational-model/convaicausallm-output/config.json


!python upload_to_huggingface.py \
  --model_path /kaggle/working/hindi-embedding-foundational-model/convaicausallm-output \
  --username convaiinnovations \
  --token token_id \
  --model_name hindi-causal-lm




