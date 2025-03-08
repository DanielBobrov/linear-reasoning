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
    """Классификатор на основе рекуррентной архитектуры RecRNN из статьи."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Проверка параметров
        self.vocab_size = config.vocab_size
        print(f"Initializing model with vocab_size={self.vocab_size}, hidden_size={config.hidden_size}")
        
        # Добавляем запас для словаря если необходимо
        if self.vocab_size < 1500:
            print(f"WARNING: Vocab size {self.vocab_size} seems small. Adding padding to be safe.")
            self.vocab_size = max(self.vocab_size + 10, 1200)
            
        # Эмбеддинг слой
        self.embedding = nn.Embedding(self.vocab_size, config.hidden_size, padding_idx=0)
        
        # Специфическая для RecRNN нормализация перед энкодером
        self.pre_encoder_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Слои энкодера - используем norm_first=True согласно статье
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size, 
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation=config.hidden_act,
            batch_first=True,
            norm_first=True  # Как в оригинальной статье RecRNN
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.encoder_layers)
        
        # Нормализация для "сэндвича" вокруг рекуррентного блока
        self.pre_recurrent_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_recurrent_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Рекуррентный блок с правильной архитектурой pre-LN
        rec_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size, 
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation=config.hidden_act,
            batch_first=True,
            norm_first=True  # Как в оригинальной статье RecRNN
        )
        self.recurrent_block = nn.TransformerEncoder(rec_layer, num_layers=config.recurrent_layers)
        self.mean_recurrence = getattr(config, "mean_recurrence", 6)
        
        # Нормализация перед декодером
        self.pre_decoder_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Слои декодера также с pre-LN
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size, 
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation=config.hidden_act,
            batch_first=True,
            norm_first=True  # Как в оригинальной статье RecRNN
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=config.decoder_layers)
        
        # Финальная нормализация перед классификацией
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Выходной слой
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Применяем инициализацию весов как в оригинале
        self.apply(self._init_weights)
        
        # Оценка параметров для проверки
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialized with {num_params/1_000_000:.2f}M parameters")
        
    def _init_weights(self, module):
        """Улучшенная инициализация весов для стабильности обучения"""
        if isinstance(module, nn.Linear):
            # Используем более строгую инициализацию для линейных слоев
            nn.init.xavier_uniform_(module.weight, gain=1.0 / (2.0 ** 0.5))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range / 2.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        
    def forward(self, x, attention_mask=None, num_recurrence=None):
        # Преобразуем padding mask в attention mask
        key_padding_mask = attention_mask if attention_mask is not None else None
        
        # Эмбеддинг
        x = self.embedding(x)
        
        # Нормализация перед энкодером
        x = self.pre_encoder_norm(x)
        
        # Энкодер
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        
        # Случайное количество итераций, если указано
        if num_recurrence is None:
            num_recurrence = self.mean_recurrence
            
        # Начальное состояние с нормализацией
        x = self.pre_recurrent_norm(x)
            
        # Рекуррентный блок (применяется указанное число раз)
        for _ in range(num_recurrence):
            x = self.recurrent_block(x, src_key_padding_mask=key_padding_mask)
        
        # Нормализация после рекуррентного блока
        x = self.post_recurrent_norm(x)
        
        # Нормализация перед декодером
        x = self.pre_decoder_norm(x)
        
        # Декодер
        x = self.decoder(x, src_key_padding_mask=key_padding_mask)
        
        # Финальная нормализация
        x = self.final_norm(x)
            
        # Mean pooling с учетом padding mask
        if key_padding_mask is not None:
            # Инвертируем маску: True для токенов, которые нужно сохранить
            mask = ~key_padding_mask
            mask = mask.float()
            # Применяем маску и усредняем
            x = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        else:
            # Простое усреднение
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    
    # Устанавливаем pad_id в корректное значение
    tokenizer.pad_id = 0  # Используем 0 вместо -1
    
    # Сканируем данные для проверки максимального индекса токена перед созданием модели
    try:
        with open(train_data_path, 'r') as f:
            train_samples = json.load(f)
            max_token_id = -1
            for item in train_samples[:1000]:  # Берем первые 1000 для скорости
                max_token_id = max(max_token_id, max(item['input_ids'] + [item['target_id']]))
            print(f"Maximum token ID in data: {max_token_id}, vocab_size={tokenizer.vocab_size}")
            
            # Настраиваем vocab_size, если нужно
            if max_token_id >= tokenizer.vocab_size:
                print(f"WARNING: Maximum token ID {max_token_id} exceeds vocab_size {tokenizer.vocab_size}")
                tokenizer.vocab_size = max_token_id + 10  # Добавляем запас
    except Exception as e:
        print(f"Error scanning data: {e}")
    
    # Создаем конфигурацию модели с vocab_size из загруженного токенизатора
    from kaggle_run import SmallRecRNNConfig
    model_config = SmallRecRNNConfig(
        vocab_size=tokenizer.vocab_size, 
        hidden_size=512,  # Уменьшенный размер для модели ~30M параметров
        intermediate_size=1024,
        num_hidden_layers=4,
        recurrent_layers=2,
        encoder_layers=1, 
        decoder_layers=1,
        num_attention_heads=8,
        block_size=256,
        mean_recurrence=6,
    )
    
    # Проверка ожидаемого размера модели
    estimated_params = model_config.estimate_params() if hasattr(model_config, "estimate_params") else "N/A"
    print(f"Estimated model parameters: ~{estimated_params/1_000_000:.2f}M" 
          if isinstance(estimated_params, (int, float)) else f"Estimation not available")
    
    print(f"Final model configuration: vocab_size={model_config.vocab_size}, hidden_size={model_config.hidden_size}")
    
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
    # Убираем "безопасный" collатор, так как токенизатор должен выбрасывать исключения при проблемах
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
    
    # Используем более стабильные гиперпараметры для оптимизатора: меньший LR и более сильный weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate * 0.1, weight_decay=0.1, eps=1e-8)
    
    # Добавляем планировщик скорости обучения с более щадящими параметрами 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=3, verbose=True, min_lr=1e-7
    )
    
    # Настраиваем ранний останов
    early_stop_patience = 5  # Количество эпох без улучшения для раннего останова
    early_stop_counter = 0
    
    # Оптимизированный цикл обучения
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    lr_history = []
    all_step_metrics = []  # Создаем список для хранения всех метрик по шагам
    
    # Создаем директорию для детализированных метрик
    metrics_dir = Path(output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = metrics_dir / "training_metrics.csv"
    
    # Записываем заголовок файла метрик
    with open(metrics_file, 'w') as f:
        f.write("epoch,step,train_loss,val_loss,learning_rate\n")
    
    # Настраиваем прогресс-бар для обновления реже
    update_interval = max(1, len(train_dataset) // (batch_size * 20))  # Примерно 20 обновлений за эпоху
    global_progress = tqdm(total=epochs * len(train_dataset) // batch_size, 
                          desc="Training", 
                          position=0,
                          leave=True,
                          ncols=100,
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    # Используем RunningMean только для потери, но не для точности (экономим вычисления)
    running_loss = torch.zeros(1, device=device)
    loss_count = 0
    
    for epoch in range(epochs):
        # Обучение
        model.train()  # Важно: перед каждой эпохой устанавливаем режим обучения
        train_loss = 0.0
        batch_count = 0
        epoch_step_metrics = []
        
        # Итерация по батчам без вложенного прогресс-бара
        for step, (input_ids, target_ids, _) in enumerate(train_loader):
            # Проверка на пустые батчи
            if input_ids.size(0) == 0:
                continue
                
            # Перемещаем данные на устройство
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            # Создаем маску внимания (1 для паддинг-токенов)
            attention_mask = (input_ids == tokenizer.pad_id)
            
            # Используем случайное число итераций как в статье
            # Исправление: используем тензоры float для torch.normal или альтернативный метод
            if model.training:
                # Случайное количество итераций в пределах [1, 2*mean_recurrence]
                # Исправленная версия с float тензорами
                num_recurrence = max(1, int(torch.normal(
                    mean=torch.tensor([float(model_config.mean_recurrence)]), 
                    std=torch.tensor([float(model_config.mean_recurrence/2)])
                ).item()))
                num_recurrence = min(num_recurrence, 2 * model_config.mean_recurrence)
            else:
                num_recurrence = model_config.mean_recurrence
                
            # Прямой проход с указанным числом рекуррентных итераций
            logits = model(input_ids, attention_mask, num_recurrence=num_recurrence)
            loss = criterion(logits, target_ids)
            
            # Обратный проход и оптимизация
            optimizer.zero_grad()
            loss.backward()
            
            # Добавим градиентный клиппинг для стабильности
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Отслеживаем статистику без вычисления точности
            train_loss += loss.item()
            running_loss += loss.item()
            loss_count += 1
            batch_count += 1
            
            # Обновляем прогресс-бар реже для повышения производительности
            if batch_count % update_interval == 0:
                current_lr = optimizer.param_groups[0]['lr']
                avg_loss = running_loss.item() / max(1, loss_count)
                global_progress.set_description(
                    f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.2f}, LR: {current_lr:.2e}"
                )
                global_progress.update(update_interval)
                
                # Записываем только базовые метрики на этом шаге
                step_idx = step + epoch * len(train_loader)
                step_data = {
                    "step": step_idx,
                    "loss": avg_loss,
                    "lr": current_lr
                }
                epoch_step_metrics.append(step_data)
                all_step_metrics.append(step_data)  # Добавляем также в общий список
                
                # Сбрасываем счетчики для следующего интервала
                running_loss.zero_()
                loss_count = 0
        
        # Обновляем оставшийся прогресс в конце эпохи
        remaining_steps = batch_count % update_interval
        if remaining_steps > 0:
            global_progress.update(remaining_steps)
        
        # Вычисляем среднюю потерю за эпоху
        avg_train_loss = train_loss / batch_count if batch_count > 0 else float('inf')
        train_losses.append(avg_train_loss)
        
        # Валидация - считаем точность только раз в эпоху
        model.eval()  # Важно: устанавливаем режим оценки перед валидацией
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_batch_count = 0
        
        with torch.no_grad():
            for input_ids, target_ids, _ in val_loader:
                # Пропускаем пустые батчи
                if input_ids.size(0) == 0:
                    continue
                    
                # Перемещаем данные на устройство
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                
                # Создаем маску внимания
                attention_mask = (input_ids == tokenizer.pad_id)
                
                # Прямой проход
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, target_ids)
                
                # Отслеживаем потерю
                val_loss += loss.item()
                val_batch_count += 1
                
                # Считаем точность только на валидации
                _, predicted = torch.max(logits, 1)
                val_total += target_ids.size(0)
                val_correct += (predicted == target_ids).sum().item()
        
        # Вычисляем метрики валидации
        avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')
        val_accuracy = 100 * val_correct / val_total if val_total > 0 else 0
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Вычисляем приблизительную точность на тренировочном наборе только в конце эпохи
        if epoch % 1 == 0:  # Каждую эпоху или реже
            model.eval()
            train_correct = 0
            train_total = 0
            with torch.no_grad():
                # Используем только небольшую выборку для оценки точности
                sample_size = min(1000, len(train_dataset))
                sample_indices = torch.randperm(len(train_dataset))[:sample_size]
                for idx in sample_indices:
                    item = train_dataset[idx]
                    input_ids = torch.tensor([item['input_ids']], device=device)
                    target_id = torch.tensor([item['target_id']], device=device)
                    
                    attention_mask = (input_ids == tokenizer.pad_id)
                    logits = model(input_ids, attention_mask)
                    _, predicted = torch.max(logits, 1)
                    
                    train_total += 1
                    if predicted.item() == target_id.item():
                        train_correct += 1
            
            train_accuracy = 100 * train_correct / train_total if train_total > 0 else 0
            train_accuracies.append(train_accuracy)
            model.train()  # Возвращаемся в режим обучения
        else:
            # Если не вычисляем точность в этой эпохе, используем предыдущее значение или 0
            train_accuracy = train_accuracies[-1] if train_accuracies else 0
            train_accuracies.append(train_accuracy)
        
        # Шаг планировщика скорости обучения
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        
        # Обновляем файл с метриками после эпохи
        with open(metrics_file, 'a') as f:
            f.write(f"{epoch},-1,{avg_train_loss:.6f},{avg_val_loss:.6f},{current_lr:.8e}\n")
        
        # Печатаем статистику эпохи
        print(f"\nEpoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, "
              f"LR: {current_lr:.2e}")
        
        # Диагностика
        if epoch == 0 or epoch == epochs - 1 or epoch % 5 == 0:
            # Выводим параметры модели для отладки
            print("\nDiagnostic check:")
            for name, param in model.named_parameters():
                if 'weight' in name and len(param.shape) >= 2:
                    print(f"  {name}: shape={param.shape}, mean={param.mean().item():.4f}, "
                          f"std={param.std().item():.4f}, max={param.max().item():.4f}, "
                          f"min={param.min().item():.4f}, "
                          f"has_nan={torch.isnan(param).any().item()}")
            
            # Проверка градиентов
            param_norm = 0
            grad_norm = 0
            for p in model.parameters():
                param_norm += p.norm().item() ** 2
                if p.grad is not None:
                    grad_norm += p.grad.norm().item() ** 2
            print(f"  Total parameter norm: {param_norm**0.5:.4f}")
            print(f"  Total gradient norm: {grad_norm**0.5:.4f}")
        
        # Сохраняем лучшую модель
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
            }, output_dir / "best_model.pt")
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        else:
            early_stop_counter += 1
            print(f"Validation loss did not improve. Early stop counter: {early_stop_counter}/{early_stop_patience}")
            
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break
    
    # Закрываем прогресс-бар в конце
    global_progress.close()
    
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
    
    # Сохраняем историю обучения с расширенными метриками
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_accuracy': train_accuracies,
        'val_accuracy': val_accuracies,
        'learning_rate': lr_history,
        'step_metrics': all_step_metrics  # Используем собранные метрики
    }
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f)
        
    # Создаем простой график для быстрого визуального анализа
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 10))
        
        # График потерь
        plt.subplot(2, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # График точности
        plt.subplot(2, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        
        # График LR
        plt.subplot(2, 2, 3)
        plt.plot(lr_history)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(output_dir / "training_metrics.png")
        print(f"Training metrics plot saved to {output_dir / 'training_metrics.png'}")
    except Exception as e:
        print(f"Could not create metrics plot: {e}")
    
    print("Training completed!")
    return model

if __name__ == "__main__":
    # Пример использования
    train_model(
        epochs=5, 
        batch_size=32, 
        learning_rate=1e-4
    )
