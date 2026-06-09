!pip install -q bitsandbytes==0.41.3 peft==0.7.1 accelerate==0.27.2 transformers==4.36.2 sentencepiece


pip install deepspeed


#!/usr/bin/env python3
"""
finetune_arc_qlora_kaggle.py - Optimized QLoRA fine-tuning for ARC-AGI-2 competition
https://www.kaggle.com/competitions/arc-prize-2025
"""
import os
import sys
import subprocess
import inspect
import torch
import json

def create_zero2_config(output_dir):
    """Create a custom ZeRO-2 config file with optimized settings"""
    config = {
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "overlap_comm": True,
            "contiguous_gradients": True
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto"
            }
        },
        "bf16": {
            "enabled": True
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
        "zero_allow_untested_optimizer": True
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    
    # Write config to file
    config_path = os.path.join(output_dir, "zero2_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    return config_path

def main():

    LLAVA_DIR = "/kaggle/input/llavadoc/LLaVA"
    DATA_JSON = "/kaggle/input/converteddataset/converteddataset/arc_llava_data_converted/arc_llava_training.json"
    IMAGE_FOLDER = "/kaggle/input/converteddataset/converteddataset/arc_llava_images_converted"

    MODEL_PATH = "/kaggle/input/vicuna-13b/pytorch/default/1/v1.6-vicuna-13b"
    OUTPUT_DIR = "/kaggle/working/arc-agi2-qlora-vicuna-13b"
    
    # Add LLaVA to Python path
    sys.path.append(LLAVA_DIR)
    print(f"Added {LLAVA_DIR} to Python path")
    
    try:
        # Verify GPU configuration
        print("\nGPU configuration:")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Verify paths
        print("\nVerifying paths:")
        for path in [MODEL_PATH, DATA_JSON, IMAGE_FOLDER]:
            exists = os.path.exists(path)
            print(f"{path}: {'✓' if exists else '✗'}")
            if not exists:
                parent = os.path.dirname(path)
                if os.path.exists(parent):
                    print(f"Contents of {parent}:")
                    subprocess.run(f"ls -la {parent}", shell=True)
        
        # Create custom ZeRO-2 config optimized for checkpointing
        ds_config_path = create_zero2_config("/kaggle/working")
        print(f"Created optimized DeepSpeed ZeRO-2 config at: {ds_config_path}")
        
        # Import training function
        from llava.train.train import train
        
        # ARC-specific hyperparameters - optimized for reasoning tasks
        print("\nStarting fine-tuning with ARC-optimized parameters...")
        sys.argv = [
            "train.py",
            "--deepspeed", ds_config_path,
            
            # Model configuration
            "--model_name_or_path", MODEL_PATH,
            "--vision_tower", "openai/clip-vit-large-patch14",
            "--version", "arc_agi2",
            "--mm_vision_select_layer", "-2",  # Use second-to-last layer for vision features
            
            # QLoRA configuration for 13B model
            "--lora_enable", "True",
            "--bits", "4",  # 4-bit quantization
            "--double_quant", "True",  # Use double quantization for further memory savings
            "--quant_type", "nf4",  # Normalized float 4-bit quantization
            "--lora_r", "128",  # Adaptive rank for 13B model
            "--lora_alpha", "256",
            "--lora_dropout", "0.05",
            "--target_modules", "q_proj,v_proj,k_proj,o_proj,down_proj,gate_proj,up_proj",
            
            # Data configuration
            "--data_path", DATA_JSON,
            "--image_folder", IMAGE_FOLDER,
            "--image_aspect_ratio", "square",
            
            # Training settings
            "--bf16", "True",  # Use bfloat16 for efficiency
            "--output_dir", OUTPUT_DIR,
            "--num_train_epochs", "3",
            "--per_device_train_batch_size", "4",  # Smaller batch size for 13B model
            "--per_device_eval_batch_size", "4",
            "--gradient_accumulation_steps", "8",  # Increased gradient accumulation for stability
            
            # Fix: Make evaluation and save strategies match but reduce frequency to avoid slow checkpoints
            "--evaluation_strategy", "steps", 
            "--eval_steps", "1000",  # Reduced frequency
            "--save_strategy", "steps", 
            "--save_steps", "1000",  # Reduced frequency
            "--save_total_limit", "2",  # Keep fewer checkpoints
            
            "--learning_rate", "2e-4",  # Slightly higher LR for ARC tasks
            "--weight_decay", "0.01",  # Increase regularization for better generalization
            "--warmup_ratio", "0.05",  # More warmup for stability
            "--lr_scheduler_type", "cosine",
            "--logging_steps", "10",
            "--tf32", "True",
            "--model_max_length", "2048",  # Increased context for reasoning
            "--gradient_checkpointing", "True",
            "--dataloader_num_workers", "4",
            "--lazy_preprocess", "True",
            "--report_to", "none",
            
            # Optimizer settings for ARC reasoning tasks
            "--optim", "adamw_torch_fused",
            "--adam_beta1", "0.9",
            "--adam_beta2", "0.95",  # Higher beta2 for reasoning tasks
            
            # Early stopping with matching save/eval strategies
            "--load_best_model_at_end", "True", 
            "--metric_for_best_model", "eval_loss",
            "--greater_is_better", "False",
        ]
        
        # Start training
        print("Training started with DeepSpeed ZeRO Stage 2 for faster checkpointing...")
        trainer = train(attn_implementation="flash_attention_2")
        
        # Explicitly save final model after training is complete
        print("\nTraining complete, saving final model...")
        try:
            # Try trainer's save_model first
            trainer.save_model(output_dir=OUTPUT_DIR + "/final-model")
            print("Model saved using trainer.save_model()")
        except Exception as e:
            print(f"Standard save failed: {e}")
            try:
                # Alternative: use DeepSpeed's save_checkpoint
                if hasattr(trainer, 'deepspeed'):
                    checkpoint_dir = os.path.join(OUTPUT_DIR, "checkpoint-final")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    trainer.deepspeed.save_checkpoint(checkpoint_dir)
                    print(f"DeepSpeed checkpoint saved to {checkpoint_dir}")
                    
                    # Try to convert DeepSpeed checkpoint to HF format
                    try:
                        print("Converting DeepSpeed checkpoint to HF format...")
                        subprocess.run(f"cd {checkpoint_dir} && python zero_to_fp32.py . pytorch_model.bin", shell=True)
                        print("Conversion complete!")
                    except Exception as conversion_error:
                        print(f"Conversion error: {conversion_error}")
            except Exception as ds_save_error:
                print(f"DeepSpeed save failed: {ds_save_error}")
        
        print("\nAll done! Check for model files in:")
        print(f"1. {OUTPUT_DIR} (regular checkpoints)")
        print(f"2. {OUTPUT_DIR}/final-model (final model)")
        print(f"3. {OUTPUT_DIR}/checkpoint-final (DeepSpeed checkpoint)")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        
        # Additional diagnostic information
        print("\nModel path contents:")
        subprocess.run(f"ls -la {MODEL_PATH}", shell=True)
        
        # Check if we're out of memory
        if "CUDA out of memory" in str(e):
            print("\nCUDA OOM detected. Current GPU memory usage:")
            subprocess.run("nvidia-smi", shell=True)
            print("\nTry reducing batch size, model_max_length, or lora_r parameters")

if __name__ == "__main__":
    main() 

