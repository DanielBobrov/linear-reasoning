import os
import sys
import json
import torch
import argparse
from pathlib import Path
import subprocess

# Add the directory to the path so we can import our modules
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

# Import the configuration
from config.small_recurnn import SmallRecRNNConfig
from simple_tokenizer import Tokenizer
from optimized_collator import optimized_collate_fn

def run_small_model():
    parser = argparse.ArgumentParser(description="Train a small RecRNN model")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare data, don't train")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze data, don't train")
    parser.add_argument("--hidden-size", type=int, default=768, help="Hidden size of the model")
    parser.add_argument("--batch-size", type=int, default=32, help="Micro batch size")
    parser.add_argument("--optimize-data", action="store_true", help="Use optimized data format with single token targets")
    args = parser.parse_args()

    # Directory paths
    data_dir = Path("/kaggle/input/paper-data/data/comparison.1000.12.6")
    output_dir = Path("kaggle/working/") / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.analyze_only:
        print("Analyzing dataset...")
        subprocess.run(["python", "analyze_dataset.py"])
        return

    # First, let's prepare the data
    print("Preparing data...")
    if args.optimize_data:
        # Use optimized data format (single token target)
        if not (data_dir / "optimized").exists():
            subprocess.run(["python", "prepare_optimized_data.py"])
        else:
            print("Optimized data already exists. Skipping preparation.")
        data_prefix = "optimized"
    else:
        # Use standard tokenized format
        if not (data_dir / "tokenized").exists():
            subprocess.run(["python", "prepare_tokenized_data.py"])
        else:
            print("Tokenized data already exists. Skipping preparation.")
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
        json.dump(vars(model_config), f, indent=2)
    
    # Run the training script
    print("Starting training...")
    
    # Base command
    cmd = [
        "python", "train.py",
        "--run_name", "small_recurrent_model",
        "--model_impl", "recurrent",  
        "--model_config", "SmallRecRNNConfig",
        "--tokenizer_path", str(data_dir / "vocab.json"),
        "--train_data_dir", str(data_dir),
        "--val_data_dir", str(data_dir),
        "--out_dir", str(output_dir),
        "--block_size", "256",
        "--micro_batch_size", str(args.batch_size),
        "--batch_size", str(args.batch_size * 4),
        "--max_tokens", "1000000000",  # 1B tokens
        "--gradient_accumulation_steps", "4",
        "--optim_config.lr", "0.0001",
        "--warmup_steps", "100",
        "--eval_step_interval", "500",
        "--save_step_interval", "1000",
        "--log_step_interval", "10",
        "--fabric_strategy", "single",
        "--seed", "42"
    ]
    
    # Add data configuration based on format
    if args.optimize_data:
        # We use the optimized data with a custom collator
        data_config = {
            "train_data": [{"type": "pqds", "data_dir": str(data_dir), "prefix": "train_optimized"}],
            "val_data": [{"type": "pqds", "data_dir": str(data_dir), "prefix": "valid_optimized"}]
        }
        cmd.extend([
            "--data_config.train_data", json.dumps(data_config["train_data"]),
            "--data_config.val_data", json.dumps(data_config["val_data"]),
            # Add a flag to use the optimized collator
            "--use_optimized_collator", "True"
        ])
    else:
        data_config = {
            "train_data": [{"type": "pqds", "data_dir": str(data_dir), "prefix": "train"}],
            "val_data": [{"type": "pqds", "data_dir": str(data_dir), "prefix": "valid"}]
        }
        cmd.extend([
            "--data_config.train_data", json.dumps(data_config["train_data"]),
            "--data_config.val_data", json.dumps(data_config["val_data"])
        ])
    
    print("Running command:")
    print(" ".join(cmd))
    
    # Execute the command
    subprocess.run(cmd)

if __name__ == "__main__":
    run_small_model()
