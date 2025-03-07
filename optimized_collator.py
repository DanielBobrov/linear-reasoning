import torch
from typing import List, Dict, Any, Tuple, Optional

def optimized_collate_fn(
    batch: List[Dict[str, Any]],
    tokenizer=None,
    block_size: int = 256,
    pad_to_block_size: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Оптимизированная функция collate для работы с одиночными целевыми токенами.
    
    Args:
        batch: Список словарей с ключами 'input_ids' и 'target_id'
        tokenizer: Объект токенизатора с атрибутом pad_id
        block_size: Максимальная длина последовательности
        pad_to_block_size: Дополнять ли последовательности до block_size
        
    Returns:
        input_ids: Тензор формы [batch_size, seq_len]
        target_ids: Тензор формы [batch_size]
        metadata: Список типов данных или другой метаинформации
    """
    if not batch:
        return torch.tensor([]), torch.tensor([]), []
    
    pad_id = tokenizer.pad_id if tokenizer is not None else -1
    
    # Получаем максимальную длину последовательностей в этом батче
    max_length = max(len(item["input_ids"]) for item in batch)
    
    # Ограничиваем block_size
    max_length = min(max_length, block_size)
    
    # Если запрошено дополнение до block_size
    if pad_to_block_size:
        max_length = block_size
    
    # Дополнительная проверка данных на корректность и согласованность с моделью
    vocab_size = tokenizer.vocab_size
    invalid_samples = 0
    valid_batch = []
    
    for item in batch:
        # Проверяем входные данные
        valid_input = all(0 <= token_id < vocab_size for token_id in item['input_ids'])
        valid_target = 0 <= item['target_id'] < vocab_size
        
        if valid_input and valid_target:
            valid_batch.append(item)
        else:
            invalid_samples += 1
            # Печатаем подробную информацию о неправильных данных
            if not valid_input:
                invalid_tokens = [t for t in item['input_ids'] if not (0 <= t < vocab_size)]
                print(f"Warning: Invalid input token IDs found: {invalid_tokens}, vocab_size={vocab_size}")
            if not valid_target:
                print(f"Warning: Invalid target token ID: {item['target_id']}, vocab_size={vocab_size}")
    
    if invalid_samples > 0:
        print(f"Skipped {invalid_samples} invalid samples out of {len(batch)}")
        
    # Если все образцы неправильные, возвращаем пустые тензоры
    if not valid_batch:
        print("Warning: Entire batch is invalid! Returning empty tensors.")
        return torch.zeros((0, 1), dtype=torch.long), torch.zeros((0,), dtype=torch.long), []
    
    # Пересчитываем batch с оставшимися валидными элементами
    batch = valid_batch
    
    # Подготавливаем тензоры
    input_ids = torch.full((len(batch), max_length), pad_id, dtype=torch.long)
    target_ids = torch.zeros(len(batch), dtype=torch.long)
    metadata = []
    
    # Заполняем тензоры
    for i, item in enumerate(batch):
        seq_len = min(len(item["input_ids"]), max_length)
        input_ids[i, :seq_len] = torch.tensor(item["input_ids"][:seq_len], dtype=torch.long)
        target_ids[i] = item["target_id"]
        metadata.append(item.get("type", "unknown"))
    
    return input_ids, target_ids, metadata
