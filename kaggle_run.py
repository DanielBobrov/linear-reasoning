import os
import sys
import json
import torch
import argparse
from pathlib import Path

# Добавляем корневую директорию в PYTHONPATH
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

# Определяем конфигурацию модели прямо здесь
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, Tuple, Type
import torch.nn as nn

@dataclass
class SmallRecRNNConfig:
    """Configuration for the smaller RecRNN model (30M parameters)"""
    # Model architecture
    vocab_size: int  # Будет определено динамически
    hidden_size: int = 512  # Уменьшенный размер для модели ~30M параметров
    intermediate_size: int = 1024  # 2x hidden_size для эффективных вычислений
    num_hidden_layers: int = 4  # Reduced from 8 (1 encoder + 2 recurrent + 1 decoder)
    recurrent_layers: int = 2  # Reduced recurrent layers
    encoder_layers: int = 1  # Reduced encoder layers
    decoder_layers: int = 1  # Reduced decoder layers
    num_attention_heads: int = 8  # Can be scaled with hidden_size
    block_size: int = 256  # Context window size
    
    # Recurrence settings from original model
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
    
    def estimate_params(self):
        """Оценка количества параметров модели"""
        # Эмбеддинги
        embedding_params = self.vocab_size * self.hidden_size
        
        # Слои трансформера
        layer_params = 4 * (self.hidden_size ** 2)
        
        # Учитываем proj слои и attention heads
        attn_params = self.hidden_size * self.hidden_size * self.num_attention_heads 
        
        # Feedforward params
        ff_params = self.hidden_size * self.intermediate_size * 2
        
        # Суммарно за один слой
        per_layer = attn_params + ff_params + self.hidden_size * 2
        
        # Общее количество слоев
        total_layers_params = per_layer * self.num_hidden_layers
        
        # Выходной слой
        output_params = self.hidden_size * self.vocab_size
        
        # Итого
        total_params = embedding_params + total_layers_params + output_params
        
        return total_params

from simple_tokenizer import SimpleTokenizer, Tokenizer
from optimized_collator import optimized_collate_fn

def run_small_model():
    parser = argparse.ArgumentParser(description="Train a small RecRNN model")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare data, don't train")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze data, don't train")
    parser.add_argument("--hidden-size", type=int, default=768, help="Hidden size of the model")
    parser.add_argument("--batch-size", type=int, default=32, help="Micro batch size")
    parser.add_argument("--optimize-data", action="store_true", help="Use optimized data format with single token targets")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs") 
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    # Определяем пути к данным и выходным файлам с учетом структуры Kaggle
    data_dir = Path("/kaggle/input/paper-data/data/comparison.1000.12.6")
    working_dir = Path("/kaggle/working")
    output_dir = working_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Сначала загружаем токенизатор, чтобы получить размер словаря
    vocab_paths = [
        data_dir / "vocab.json",
        working_dir / "data/vocab.json"
    ]
    
    vocab_path = None
    for path in vocab_paths:
        if path.exists():
            vocab_path = path
            break
    
    if vocab_path is None:
        print("ERROR: vocab.json не найден! Ищем альтернативные пути...")
        # Ищем vocab.json в любом месте
        vocab_paths = list(Path("/kaggle").glob("**/vocab.json"))
        if vocab_paths:
            vocab_path = str(vocab_paths[0])
            print(f"Найден словарь: {vocab_path}")
        else:
            print("Словарь не найден! Используем значение по умолчанию - 1205 токенов")
            vocab_path = None
            vocab_size = 1205  # Fallback значение в случае отсутствия словаря
    
    if vocab_path:
        # Загружаем токенизатор и определяем размер словаря динамически
        print(f"Загружаем токенизатор из: {vocab_path}")
        tokenizer = SimpleTokenizer(vocab_path)
        vocab_size = tokenizer.vocab_size
        print(f"Размер словаря: {vocab_size} токенов")
    
    if args.analyze_only:
        print("Analyzing dataset...")
        import subprocess
        subprocess.run(["python", "analyze_dataset.py"])
        return

    # Prepare data
    print("Preparing data...")
    # Необходимо принудительно пересоздать оптимизированные данные, поскольку возможно 
    # они были созданы с другой версией словаря
    import subprocess
    print("Forcing recreation of optimized data to ensure consistency with current vocabulary")
    subprocess.run(["python", "prepare_optimized_data.py", "--force"])
    
    if args.prepare_only:
        print("Data preparation complete. Exiting as requested.")
        return
    
    # Create model configuration - теперь с правильными параметрами для ~30M параметров
    model_config = SmallRecRNNConfig(
        vocab_size=vocab_size,
        hidden_size=512,  # Reduced for 30M model
        intermediate_size=1024,  # 2x hidden_size
        num_hidden_layers=4,
        recurrent_layers=2,
        encoder_layers=1,
        decoder_layers=1,
        num_attention_heads=8,
        block_size=256,
        mean_recurrence=6  # Keep original recurrence depth
    )
    
    # Calculate estimated parameters
    estimated_params = model_config.estimate_params()
    
    # Store model config for reference
    with open(output_dir / "small_model_config.json", "w") as f:
        config_dict = {k: v for k, v in model_config.__dict__.items() if not k.startswith("_")}
        config_dict["estimated_parameters"] = estimated_params
        json.dump(config_dict, f, indent=2)
    
    # Print information about training configuration
    print("\nModel Configuration:")
    print(f"  Vocabulary Size: {model_config.vocab_size}")
    print(f"  Hidden Size: {model_config.hidden_size}")
    print(f"  Intermediate Size: {model_config.intermediate_size}")
    print(f"  Attention Heads: {model_config.num_attention_heads}")
    print(f"  Total Layers: {model_config.num_hidden_layers} (Encoder: {model_config.encoder_layers}, "
          f"Recurrent: {model_config.recurrent_layers}, Decoder: {model_config.decoder_layers})")
    print(f"  Mean Recurrence: {model_config.mean_recurrence}")
    print(f"  Estimated Parameters: {estimated_params/1_000_000:.2f}M")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.learning_rate}")
    
    # В Kaggle мы используем только оптимизированный тренер, так как стандартный требует train.py
    print("\nTraining with optimized model_trainer...")
    from model_trainer import train_model
    train_model(
        train_data_path=str(working_dir / "optimized" / "train_optimized.json"),
        val_data_path=str(working_dir / "optimized" / "valid_optimized.json"),
        output_dir=str(output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=42
    )
    
    print("\nTraining completed. Files saved to:", output_dir)

if __name__ == "__main__":
    run_small_model()
