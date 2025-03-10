import json
import os
import argparse
from pathlib import Path
import numpy as np
from simple_tokenizer import SimpleTokenizer
from tqdm import tqdm

def prepare_optimized_data():
    """Tokenize and optimize data, reducing target to just the attribute token to predict"""
    # Добавляем аргумент --force для принудительной пересоздания данных
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='Force recreation of optimized data')
    parser.add_argument('--verbose', action='store_true', help='Show detailed progress and diagnostics')
    args = parser.parse_args()
    
    # Функция для условной печати - только если verbose режим активен
    def verbose_print(*print_args, **print_kwargs):
        if args.verbose:
            print(*print_args, **print_kwargs)

    # Пути в Kaggle - проверяем различные возможные пути к данным
    possible_data_dirs = [
        # Path("/kaggle/input/paper-data/data/comparison.1000.12.6"),
        Path("/kaggle/input/paper-data/data/data/composition.2000.200.18.0"),
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
    
    # Проверяем, существуют ли оптимизированные данные и нужно ли их пересоздать
    if optimized_dir.exists() and not args.force:
        print(f"Optimized data directory already exists at {optimized_dir}. Use --force to recreate.")
        return
    
    # Создаем директорию заново
    if optimized_dir.exists():
        print("Removing existing optimized data directory...")
        import shutil
        shutil.rmtree(optimized_dir)
    
    optimized_dir.mkdir(parents=True, exist_ok=True)
    
    # Определим максимальный индекс токена в словаре
    max_token_id = tokenizer.vocab_size - 1
    verbose_print(f"Vocab size: {tokenizer.vocab_size}, Max token ID: {max_token_id}")
    
    if args.verbose:
        # Распечатаем несколько токенов для диагностики
        verbose_print("Sample tokens from vocabulary:")
        samples = list(range(0, max_token_id, max(1, max_token_id // 5)))[:5]  # Берем только 5 образцов
        for i in samples:
            verbose_print(f"  ID {i}: {tokenizer.id_to_token.get(i, 'UNKNOWN')}")
        verbose_print(f"  ID {max_token_id}: {tokenizer.id_to_token.get(max_token_id, 'UNKNOWN')}")
    
    # Process each available file
    for file_name in ["train.json", "valid.json", "test.json"]:
        file_path = data_dir / file_name
        if not file_path.exists():
            verbose_print(f"Skipping {file_name} as it doesn't exist.")
            continue
            
        print(f"Processing {file_name}...")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Tokenize and optimize data
        optimized_data = []
        error_items = []
        
        for item_idx, item in enumerate(tqdm(data, desc=f"Optimizing {file_name}")):
            try:
                input_text = item['input_text']
                target_text = item['target_text']
                
                # Диагностическая информация для первых нескольких элементов
                if args.verbose and len(optimized_data) < 2:
                    verbose_print(f"\nSample {len(optimized_data)+1}:")
                    verbose_print(f"  Input: {input_text}")
                    verbose_print(f"  Target: {target_text}")
                
                # Строгая проверка формата текста
                if not (input_text.startswith('<') and input_text.endswith('>')):
                    raise ValueError(f"Неверный формат входного текста (должен начинаться с '<' и заканчиваться '>'): {input_text}")
                
                if not (target_text.startswith('<') and target_text.endswith('>')):
                    raise ValueError(f"Неверный формат целевого текста (должен начинаться с '<' и заканчиваться '>'): {target_text}")
                
                # Токенизация с явной обработкой ошибок
                input_tokens = tokenizer.encode(input_text)
                target_tokens = tokenizer.encode(target_text)
                
                if args.verbose and len(optimized_data) < 2:
                    verbose_print(f"  Input tokens: {input_tokens}")
                    verbose_print(f"  Target tokens: {target_tokens}")
                
                # Проверка длины целевого текста
                if len(target_tokens) < 2:
                    raise ValueError(f"Слишком короткий целевой текст, должно быть минимум 2 токена: {target_text}")
                
                # Extract just the attribute token (second-to-last token)
                attribute_token = target_tokens[-2]
                
                # Строгая проверка валидности токенов
                if not (0 <= attribute_token < tokenizer.vocab_size):
                    raise ValueError(f"Целевой атрибут-токен {attribute_token} вне допустимого диапазона [0, {tokenizer.vocab_size-1}]")
                
                optimized_item = {
                    'input_ids': input_tokens,
                    'target_id': attribute_token,  # Single token target
                    'type': item.get('type', 'unknown'),
                }
                optimized_data.append(optimized_item)
                
            except Exception as e:
                error_info = {
                    "index": item_idx,
                    "item": item,
                    "error": str(e)
                }
                error_items.append(error_info)
                print(f"Ошибка в элементе {item_idx}: {str(e)}")
        
        print(f"\nСтатистика для {file_name}:")
        print(f"  Обработано: {len(data)} элементов")
        print(f"  Успешно преобразовано: {len(optimized_data)} элементов")
        print(f"  С ошибками: {len(error_items)} элементов")
        
        # Save optimized data to output directory
        output_file = optimized_dir / f"{file_name.replace('.json', '_optimized.json')}"
        with open(output_file, 'w') as f:
            json.dump(optimized_data, f)
            
        # Save errors to a separate file for analysis
        if error_items:
            error_file = optimized_dir / f"{file_name.replace('.json', '_errors.json')}"
            with open(error_file, 'w') as f:
                json.dump(error_items, f, indent=2)
            print(f"Ошибки сохранены в файл: {error_file} для анализа")
        
        print(f"Данные сохранены в: {output_file} - {len(optimized_data)} примеров")
    
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
