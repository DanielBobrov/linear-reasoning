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
    
    # Обязательно проверяем vocab_size
    vocab_size = tokenizer.vocab_size
    print(f"Collator using vocab_size={vocab_size}, checking {len(batch)} samples")
    
    # Получаем максимальную длину последовательностей в этом батче
    max_length = max(len(item["input_ids"]) for item in batch)
    
    # Ограничиваем block_size
    max_length = min(max_length, block_size)
    
    # Если запрошено дополнение до block_size
    if pad_to_block_size:
        max_length = block_size
    
    # Проверка всех данных на соответствие размерам vocab_size
    valid_batch = []
    max_id_seen = -1
    
    for item in batch:
        is_valid = True
        # Проверяем входные данные
        for token_id in item["input_ids"]:
            max_id_seen = max(max_id_seen, token_id)
            if token_id < 0 or token_id >= vocab_size:
                print(f"Warning: Input token ID {token_id} out of range (vocab_size={vocab_size})")
                is_valid = False
                break
                
        # Проверяем целевой токен
        target_id = item["target_id"]
        max_id_seen = max(max_id_seen, target_id)
        if target_id < 0 or target_id >= vocab_size:
            print(f"Warning: Target token ID {target_id} out of range (vocab_size={vocab_size})")
            is_valid = False
        
        if is_valid:
            valid_batch.append(item)
    
    # Если есть невалидные элементы, выводим статистику
    invalid_count = len(batch) - len(valid_batch)
    if invalid_count > 0:
        print(f"WARNING: Filtered {invalid_count} invalid samples with tokens outside range [0, {vocab_size-1}]")
        print(f"Highest token ID observed: {max_id_seen}")
    
    # Если все образцы неправильные, возвращаем пустые тензоры
    if not valid_batch:
        print("ERROR: Entire batch is invalid! Returning empty tensors.")
        return torch.zeros((0, 1), dtype=torch.long), torch.zeros((0,), dtype=torch.long), []
    
    # Используем только валидные элементы
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
