import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random

# Импортируем наши модули
from simple_tokenizer import SimpleTokenizer
from optimized_collator import optimized_collate_fn

class SingleTokenClassifier(nn.Module):
    """Простой классификатор, который предсказывает один токен на основе входной последовательности."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Простая конфигурация трансформера
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Слои энкодера (упрощенные блоки трансформера)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size, 
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.encoder_layers)
        
        # Рекуррентный блок (рекурсивные слои трансформера)
        rec_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size, 
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=0.1,
            batch_first=True
        )
        self.recurrent_block = nn.TransformerEncoder(rec_layer, num_layers=config.recurrent_layers)
        self.mean_recurrence = getattr(config, "mean_recurrence", 6)
        
        # Слои декодера
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size, 
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=config.decoder_layers)
        
        # Выходной слой
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)
        
    def forward(self, x, attention_mask=None):
        # Эмбеддинг
        x = self.embedding(x)
        
        # Энкодер
        if attention_mask is not None:
            x = self.encoder(x, src_key_padding_mask=attention_mask)
        else:
            x = self.encoder(x)
            
        # Рекуррентный блок (применяется несколько раз)
        for _ in range(self.mean_recurrence):
            if attention_mask is not None:
                x = self.recurrent_block(x, src_key_padding_mask=attention_mask)
            else:
                x = self.recurrent_block(x)
        
        # Декодер
        if attention_mask is not None:
            x = self.decoder(x, src_key_padding_mask=attention_mask)
        else:
            x = self.decoder(x)
            
        # Mean pooling для классификации
        if attention_mask is not None:
            # Создаем маску для исключения паддинг-токенов из среднего
            mask = ~attention_mask
            mask = mask.float()
            # Применяем маску и усредняем
            x = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        else:
            # Простое усреднение, если нет маски
            x = torch.mean(x, dim=1)
        
        # Классификация
        logits = self.classifier(x)
        return logits

class OptimizedDataset:
    """Простой датасет для загрузки предварительно токенизированных данных"""
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def train_model(
    config_path: str = None,
    train_data_path: str = None,
    val_data_path: str = None,
    output_dir: str = "output",
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    seed: int = 42,
    device: str = None
):
    """Обучение модели на оптимизированных данных."""
    # Настройка устройства
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Устанавливаем seed для воспроизводимости
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Пути по умолчанию на основе структуры проекта
    base_dir = Path(__file__).parent.absolute()
    
    # В Kaggle мы имеем другие пути
    if train_data_path is None:
        train_data_path = "/kaggle/working/optimized/train_optimized.json"
    if val_data_path is None:
        val_data_path = "/kaggle/working/optimized/valid_optimized.json"
    
    # Проверяем существование файлов
    if not Path(train_data_path).exists():
        print(f"ОШИБКА: Файл {train_data_path} не найден!")
        # Ищем альтернативные пути
        alt_paths = list(Path("/kaggle/working").glob("**/*train*optimized*.json"))
        if alt_paths:
            print(f"Найдены альтернативные файлы: {alt_paths}")
            train_data_path = str(alt_paths[0])
        else:
            print("Альтернативные файлы не найдены. Убедитесь, что prepare_optimized_data.py был выполнен.")
            return
    
    if not Path(val_data_path).exists():
        print(f"ОШИБКА: Файл {val_data_path} не найден!")
        # Ищем альтернативные пути
        alt_paths = list(Path("/kaggle/working").glob("**/*valid*optimized*.json"))
        if alt_paths:
            print(f"Найдены альтернативные файлы: {alt_paths}")
            val_data_path = str(alt_paths[0])
        else:
            print("Альтернативные файлы не найдены. Убедитесь, что prepare_optimized_data.py был выполнен.")
            return
    
    # Создаем выходную директорию
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Загружаем токенизатор
    vocab_paths = [
        "/kaggle/input/paper-data/data/comparison.1000.12.6/vocab.json",
        "/kaggle/working/data/vocab.json"
    ]
    
    vocab_path = None
    for path in vocab_paths:
        if Path(path).exists():
            vocab_path = path
            break
    
    if vocab_path is None:
        print("ОШИБКА: vocab.json не найден!")
        # Ищем альтернативные пути
        alt_paths = list(Path("/kaggle").glob("**/vocab.json"))
        if alt_paths:
            print(f"Найдены альтернативные файлы: {alt_paths}")
            vocab_path = str(alt_paths[0])
        else:
            print("vocab.json не найден. Обучение невозможно.")
            return
    
    print(f"Loading tokenizer from: {vocab_path}")
    tokenizer = SimpleTokenizer(vocab_path)
    
    # Создаем конфигурацию модели с vocab_size из загруженного токенизатора
    from kaggle_run import SmallRecRNNConfig
    model_config = SmallRecRNNConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,  # Может быть настроен на основе требований
        intermediate_size=2048,
        num_hidden_layers=4,
        recurrent_layers=2,
        encoder_layers=1,
        decoder_layers=1,
        num_attention_heads=8,
        block_size=256,
        mean_recurrence=6,
    )
    
    print(f"Model configuration: vocab_size={model_config.vocab_size}, hidden_size={model_config.hidden_size}")
    
    # Создаем модель
    print("Creating model...")
    model = SingleTokenClassifier(model_config).to(device)
    
    # Загружаем датасеты
    print(f"Loading datasets from:")
    print(f"  Train: {train_data_path}")
    print(f"  Valid: {val_data_path}")
    
    train_dataset = OptimizedDataset(train_data_path)
    val_dataset = OptimizedDataset(val_data_path)
    
    print(f"Training data: {len(train_dataset)} samples")
    print(f"Validation data: {len(val_dataset)} samples")
    
    # Создаем загрузчики данных с обычной функцией collate без лишних проверок
    # Убираем "безопасный" collator, так как токенизатор должен выбрасывать исключения при проблемах
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=lambda batch: optimized_collate_fn(batch, tokenizer, block_size=model_config.block_size)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=lambda batch: optimized_collate_fn(batch, tokenizer, block_size=model_config.block_size)
    )
    
    # Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Цикл обучения
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Обучение
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        with tqdm(train_loader, unit="batch") as t:
            t.set_description(f"Epoch {epoch+1}/{epochs}")
            
            for input_ids, target_ids, _ in t:
                # Skip empty batches
                if input_ids.size(0) == 0:
                    continue
                
                # Перемещаем данные на устройство
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                
                # Создаем маску внимания (1 для паддинг-токенов)
                attention_mask = (input_ids == tokenizer.pad_id)
                
                # Прямой проход
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, target_ids)
                
                # Обратный проход и оптимизация
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Отслеживаем статистику
                train_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                train_total += target_ids.size(0)
                train_correct += (predicted == target_ids).sum().item()
                
                # Обновляем progress bar
                t.set_postfix(loss=f"{loss.item():.4f}", accuracy=f"{100*train_correct/train_total:.2f}%")
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(avg_train_loss)
        
        # Валидация
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for input_ids, target_ids, _ in tqdm(val_loader, desc="Validating"):
                # Перемещаем данные на устройство
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                
                # Создаем маску внимания (1 для паддинг-токенов)
                attention_mask = (input_ids == tokenizer.pad_id)
                
                # Прямой проход
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, target_ids)
                
                # Отслеживаем статистику
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_total += target_ids.size(0)
                val_correct += (predicted == target_ids).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        # Сохраняем лучшую модель
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
            }, output_dir / "best_model.pt")
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
    
    # Сохраняем финальную модель
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
    }, output_dir / "final_model.pt")
    
    # Сохраняем историю обучения
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
    }
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f)
    
    print("Training completed!")
    return model

if __name__ == "__main__":
    # Пример использования
    train_model(
        epochs=5, 
        batch_size=32, 
        learning_rate=1e-4
    )
