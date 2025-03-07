#!/usr/bin/env python3
import os
import sys
from pathlib import Path

def check_repo():
    """Проверка целостности репозитория - все ли файлы на месте."""
    repo_dir = Path(__file__).parent.absolute()
    
    required_files = [
        "simple_tokenizer.py",
        "optimized_collator.py",
        "prepare_optimized_data.py",
        "model_trainer.py",
        "kaggle_run.py",
        "analyze_dataset.py",
        "config/small_recurnn.py",
        "standard_train_kaggle.ipynb",
        "optimized_train_kaggle.ipynb",
    ]
    
    print("Checking repository structure...")
    
    all_ok = True
    for file_path in required_files:
        full_path = repo_dir / file_path
        if not full_path.exists():
            print(f"ERROR: Missing required file: {file_path}")
            all_ok = False
        else:
            print(f"OK: Found {file_path}")
    
    # Проверяем наличие директории с данными
    data_dir = repo_dir / "data"
    if not data_dir.exists():
        print("WARNING: No data directory found. Create it and add your data files.")
    else:
        print("OK: Data directory exists")
        
        # Проверяем наличие ключевых файлов данных
        for data_file in ["vocab.json", "train.json", "valid.json", "test.json"]:
            if (data_dir / data_file).exists():
                print(f"OK: Found data/{data_file}")
            else:
                print(f"WARNING: Missing data file: data/{data_file}")
    
    if all_ok:
        print("\nRepository structure looks good! All required files are present.")
    else:
        print("\nRepository structure has issues. Please fix the missing files.")
        sys.exit(1)

if __name__ == "__main__":
    check_repo()
