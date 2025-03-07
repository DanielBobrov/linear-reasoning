import json
import os
from pathlib import Path
import numpy as np
from simple_tokenizer import SimpleTokenizer
from tqdm import tqdm

def prepare_optimized_data():
    """Tokenize and optimize data, reducing target to just the attribute token to predict"""
    # Пути в Kaggle - проверяем различные возможные пути к данным
    possible_data_dirs = [
        Path("/kaggle/input/paper-data/data/comparison.1000.12.6"),
        Path("/kaggle/working/data")
    ]
    
    # Находим первый существующий каталог с данными
    data_dir = None
    for dir_path in possible_data_dirs:
        if dir_path.exists() and (dir_path / "vocab.json").exists():
            data_dir = dir_path
            break
    
    if data_dir is None:
        print("ERROR: Could not find data directory with vocab.json")
        # Ищем vocab.json в любом месте
        vocab_paths = list(Path("/kaggle").glob("**/vocab.json"))
        if vocab_paths:
            data_dir = vocab_paths[0].parent
            print(f"Found potential data directory: {data_dir}")
        else:
            print("No data directory found. Cannot proceed.")
            return
    
    # Директория для записи результатов
    working_dir = Path("/kaggle/working")
    
    print(f"Using data directory: {data_dir}")
    print(f"Using output directory: {working_dir}")
    
    # Check if files exist
    vocab_path = data_dir / "vocab.json"
    train_path = data_dir / "train.json"
    valid_path = data_dir / "valid.json"
    test_path = data_dir / "test.json"
    
    if not vocab_path.exists():
        print(f"ERROR: vocab.json not found at {vocab_path}")
        return
    
    # Verify which files exist
    available_files = []
    for path in [train_path, valid_path, test_path]:
        if path.exists():
            available_files.append(path.name)
    
    if not available_files:
        print("ERROR: No json data files found. Need at least one of train.json, valid.json, test.json")
        return
    
    print(f"Found files: vocab.json and {available_files}")
    
    # Load tokenizer
    tokenizer = SimpleTokenizer(vocab_path)
    
    # Create output directories only in writable location
    optimized_dir = working_dir / "optimized"
    optimized_dir.mkdir(parents=True, exist_ok=True)
    
    # Определим максимальный индекс токена в словаре
    max_token_id = tokenizer.vocab_size - 1
    print(f"Vocab size: {tokenizer.vocab_size}, Max token ID: {max_token_id}")
    
    # Распечатаем несколько токенов для диагностики
    print("Sample tokens from vocabulary:")
    for i in range(0, max_token_id, max(1, max_token_id // 10)):
        print(f"  ID {i}: {tokenizer.id_to_token.get(i, 'UNKNOWN')}")
    print(f"  ID {max_token_id}: {tokenizer.id_to_token.get(max_token_id, 'UNKNOWN')}")
    
    # Process each available file
    for file_name in ["train.json", "valid.json", "test.json"]:
        file_path = data_dir / file_name
        if not file_path.exists():
            print(f"Skipping {file_name} as it doesn't exist.")
            continue
            
        print(f"Processing {file_name}...")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Tokenize and optimize data
        optimized_data = []
        invalid_items = 0
        max_seen_token_id = -1
        
        for item in tqdm(data, desc=f"Optimizing {file_name}"):
            try:
                input_text = item['input_text']
                target_text = item['target_text']
                
                # Диагностическая информация для первых нескольких элементов
                if len(optimized_data) < 2:
                    print(f"\nSample {len(optimized_data)+1}:")
                    print(f"  Input: {input_text}")
                    print(f"  Target: {target_text}")
                
                # Явная проверка формата входного текста
                if not (input_text.startswith('<') and input_text.endswith('>')):
                    print(f"Warning: Input text does not follow expected format: {input_text}")
                    invalid_items += 1
                    continue
                
                # Токенизация с тщательным отслеживанием ошибок
                try:
                    input_tokens = tokenizer.encode(input_text)
                    if len(optimized_data) < 2:
                        print(f"  Input tokens: {input_tokens}")
                except Exception as e:
                    print(f"Error encoding input '{input_text}': {str(e)}")
                    invalid_items += 1
                    continue
                    
                try:
                    target_tokens = tokenizer.encode(target_text)
                    if len(optimized_data) < 2:
                        print(f"  Target tokens: {target_tokens}")
                except Exception as e:
                    print(f"Error encoding target '{target_text}': {str(e)}")
                    invalid_items += 1
                    continue
                
                # Отслеживание наибольшего ID токена для диагностики
                max_input_id = max(input_tokens) if input_tokens else -1
                max_target_id = max(target_tokens) if target_tokens else -1
                max_seen_token_id = max(max_seen_token_id, max_input_id, max_target_id)
                
                # Проверка на валидность индексов токенов
                invalid_input_ids = [tid for tid in input_tokens if tid < 0 or tid >= tokenizer.vocab_size]
                invalid_target_ids = [tid for tid in target_tokens if tid < 0 or tid >= tokenizer.vocab_size]
                
                if invalid_input_ids or invalid_target_ids:
                    print(f"Warning: Out-of-range token IDs in item {item}:")
                    if invalid_input_ids:
                        print(f"  Invalid input IDs: {invalid_input_ids}")
                    if invalid_target_ids:
                        print(f"  Invalid target IDs: {invalid_target_ids}")
                    invalid_items += 1
                    continue
                
                # Extract just the attribute token (second-to-last token)
                if len(target_tokens) < 2:
                    print(f"Warning: Target sequence too short: {target_text}")
                    invalid_items += 1
                    continue
                
                attribute_token = target_tokens[-2]
                
                # Дополнительная проверка атрибут-токена
                if attribute_token < 0 or attribute_token >= tokenizer.vocab_size:
                    print(f"Warning: Attribute token ID {attribute_token} is out of vocab range [0-{tokenizer.vocab_size-1}]")
                    invalid_items += 1
                    continue
                
                optimized_item = {
                    'input_ids': input_tokens,
                    'target_id': attribute_token,  # Single token target
                    'type': item.get('type', 'unknown'),
                }
                optimized_data.append(optimized_item)
                
            except Exception as e:
                print(f"Error processing item: {e}")
                print(f"Problematic item: {item}")
                invalid_items += 1
                continue  # Продолжаем обработку других элементов вместо завершения
        
        print(f"\nStats for {file_name}:")
        print(f"  Processed: {len(data)} items")
        print(f"  Valid: {len(optimized_data)} items")
        print(f"  Invalid: {invalid_items} items")
        print(f"  Largest token ID seen: {max_seen_token_id} (vocab size: {tokenizer.vocab_size})")
        
        # Save optimized data ТОЛЬКО в /kaggle/working
        output_file = optimized_dir / f"{file_name.replace('.json', '_optimized.json')}"
        with open(output_file, 'w') as f:
            json.dump(optimized_data, f)
        
        print(f"Saved to: {output_file}")
    
    # Create a metadata file that the dataloader can use
    metadata = {
        "vocab_size": tokenizer.vocab_size,
        "mask_token_id": tokenizer.mask_token_id,
        "optimized_format": True,
        "single_token_target": True
    }
    
    with open(optimized_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\nOptimized data preparation complete.")
    print(f"Files in optimized directory: {list(optimized_dir.glob('*'))}")

if __name__ == "__main__":
    prepare_optimized_data()
