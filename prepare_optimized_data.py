import json
import os
from pathlib import Path
import numpy as np
from simple_tokenizer import SimpleTokenizer
from tqdm import tqdm

def prepare_optimized_data():
    """Tokenize and optimize data, reducing target to just the attribute token to predict"""
    data_dir = Path("/kaggle/input/paper-data/data/comparison.1000.12.6")
    
    # Check if files exist
    vocab_path = data_dir / "vocab.json"
    train_path = data_dir / "train.json"
    valid_path = data_dir / "valid.json"
    test_path = data_dir / "test.json"
    
    # Verify which files exist
    available_files = []
    for path in [train_path, valid_path, test_path]:
        if path.exists():
            available_files.append(path.name)
    
    if not vocab_path.exists() or not available_files:
        print("Required files are missing. Need at least vocab.json and one of train.json, valid.json, test.json")
        return
    
    print(f"Found files: vocab.json and {available_files}")
    
    # Load tokenizer
    tokenizer = SimpleTokenizer(vocab_path)

    data_dir = Path("/kaggle/working")
    
    # Create output directories
    train_out_dir = data_dir / "train"
    valid_out_dir = data_dir / "valid"
    test_out_dir = data_dir / "test"
    optimized_dir = data_dir / "optimized"
    
    for directory in [train_out_dir, valid_out_dir, test_out_dir, optimized_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Process each available file
    file_mapping = [
        ("train.json", train_out_dir),
        ("valid.json", valid_out_dir),
        ("test.json", test_out_dir)
    ]
    
    for file_name, output_dir in file_mapping:
        file_path = data_dir / file_name
        if not file_path.exists():
            print(f"Skipping {file_name} as it doesn't exist.")
            continue
            
        print(f"Processing {file_name}...")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Tokenize and optimize data
        optimized_data = []
        for item in tqdm(data, desc=f"Optimizing {file_name}"):
            input_tokens = tokenizer.encode(item['input_text'])
            target_tokens = tokenizer.encode(item['target_text'])
            
            # Extract just the attribute token (second-to-last token)
            # Target is now just a single token - the attribute to predict
            attribute_token = target_tokens[-2] if len(target_tokens) >= 2 else None
            
            if attribute_token is None:
                print(f"Warning: item has no attribute token to predict: {item}")
                continue
                
            optimized_item = {
                'input_ids': input_tokens,
                'target_id': attribute_token,  # Single token target
                'type': item.get('type', 'unknown'),
            }
            optimized_data.append(optimized_item)
        
        # Save optimized data
        output_file = optimized_dir / f"{file_name.replace('.json', '_optimized.json')}"
        with open(output_file, 'w') as f:
            json.dump(optimized_data, f)
        
        # Also save a version in the expected directory structure
        simplified_output = output_dir / f"{file_name.split('.')[0]}_optimized.json"
        with open(simplified_output, 'w') as f:
            json.dump(optimized_data, f)
            
        # Calculate statistics
        input_lengths = [len(item['input_ids']) for item in optimized_data]
        attribute_values = [item['target_id'] for item in optimized_data]
        unique_targets = len(set(attribute_values))
        
        print(f"Statistics for {file_name}:")
        print(f"  Total samples: {len(optimized_data)}")
        print(f"  Input lengths: min={min(input_lengths)}, max={max(input_lengths)}, avg={np.mean(input_lengths):.2f}")
        print(f"  Unique target attributes: {unique_targets}")
        print(f"  Optimized data saved to {output_file}")
        print(f"  Data also saved to {simplified_output}")
    
    # Create a metadata file that the dataloader can use
    metadata = {
        "vocab_size": tokenizer.vocab_size,
        "mask_token_id": tokenizer.mask_token_id,
        "optimized_format": True,
        "single_token_target": True
        # Удаляем ссылки на end_token_id, так как этого атрибута больше нет в SimpleTokenizer
    }
    
    with open(data_dir / "optimized" / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\nOptimized data preparation complete.")

if __name__ == "__main__":
    prepare_optimized_data()
