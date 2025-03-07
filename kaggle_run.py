import os
import sys
import json
import torch
import argparse
from pathlib import Path


# Определяем конфигурацию модели прямо здесь
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, Tuple, Type
import torch.nn as nn

@dataclass
class SmallRecRNNConfig:
    """Configuration for the smaller RecRNN model (30M parameters)"""
    # Model architecture
    vocab_size: int = 1205  # Based on the provided vocab.json
    hidden_size: int = 768  # Default value, can be overridden
    intermediate_size: int = 2048  # Can be calculated as hidden_size * 2
    num_hidden_layers: int = 4  # Reduced from 8 (1 encoder + 2 recurrent + 1 decoder)
    recurrent_layers: int = 2  # Reduced recurrent layers
    encoder_layers: int = 1  # Reduced encoder layers
    decoder_layers: int = 1  # Reduced decoder layers
    num_attention_heads: int = 8  # Can be scaled with hidden_size
    block_size: int = 256  # Context window size
    
    # Recurrence settings
    mean_recurrence: int = 6  # Default recurrence depth
    use_cache: bool = True  # Enable caching
    recurrent_chunk_size: Optional[int] = None  # Chunk size for recurrent processing
    
    # Regularization settings
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    
    # Activation function
    hidden_act: str = "gelu"
    
    # Other settings
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    
    def __post_init__(self):
        # Ensure parameters are consistent
        assert self.num_hidden_layers == self.encoder_layers + self.recurrent_layers + self.decoder_layers

from simple_tokenizer import SimpleTokenizer, Tokenizer
from optimized_collator import optimized_collate_fn

def run_small_model():
    parser = argparse.ArgumentParser(description="Train a small RecRNN model")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare data, don't train")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze data, don't train")
    parser.add_argument("--hidden-size", type=int, default=768, help="Hidden size of the model")
    parser.add_argument("--batch-size", type=int, default=32, help="Micro batch size")
    parser.add_argument("--optimize-data", action="store_true", help="Use optimized data format with single token targets")
    args = parser.parse_args()

    # Определяем пути к данным и выходным файлам с учетом структуры Kaggle
    data_dir = Path("/kaggle/input/paper-data/data/comparison.1000.12.6")
    output_dir = Path("/kaggle/working/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.analyze_only:
        print("Analyzing dataset...")
        import subprocess
        subprocess.run(["python", "analyze_dataset.py"])
        return

    # Prepare data
    print("Preparing data...")
    if args.optimize_data:
        # Use optimized data format (single token target)
        if not (Path("/kaggle/working/optimized")).exists():
            import subprocess
            subprocess.run(["python", "prepare_optimized_data.py"])
        else:
            print("Optimized data already exists. Skipping preparation.")
        data_prefix = "optimized"
    else:
        data_prefix = "data"
    
    if args.prepare_only:
        print("Data preparation complete. Exiting as requested.")
        return
    
    # Create model configuration
    model_config = SmallRecRNNConfig(
        vocab_size=1205,  # From the vocab.json file
        hidden_size=args.hidden_size,
        intermediate_size=args.hidden_size * 2,
        num_hidden_layers=4,
        recurrent_layers=2,
        encoder_layers=1,
        decoder_layers=1,
        num_attention_heads=args.hidden_size // 128,  # Scale attention heads with model size
        block_size=256
    )
    
    # Store model config for reference
    with open(output_dir / "small_model_config.json", "w") as f:
        json.dump({k: v for k, v in model_config.__dict__.items() 
                  if not k.startswith("_")}, f, indent=2)
    
    # Print information about training configuration
    print("\nModel Configuration:")
    print(f"  Hidden Size: {model_config.hidden_size}")
    print(f"  Intermediate Size: {model_config.intermediate_size}")
    print(f"  Attention Heads: {model_config.num_attention_heads}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Data Format: {'Optimized (single token target)' if args.optimize_data else 'Standard'}")
    
    if args.optimize_data:
        print("\nTraining with optimized collator...")
        # Run optimized trainer directly
        from model_trainer import train_model
        train_model(
            train_data_path=str(Path("/kaggle/working/optimized/train_optimized.json")),
            val_data_path=str(Path("/kaggle/working/optimized/valid_optimized.json")),
            output_dir=str(output_dir),
            epochs=10,
            batch_size=args.batch_size,
            learning_rate=1e-4,
            seed=42
        )
    else:
        print("\nThis mode is not fully implemented in Kaggle environment.")
        print("Please use --optimize-data flag for training in Kaggle.")
    
    print("\nTraining completed. Files saved to:", output_dir)

if __name__ == "__main__":
    run_small_model()
