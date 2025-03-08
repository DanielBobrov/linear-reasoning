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
    
    # Обеспечиваем корректный pad_id без проверок каждого токена
    pad_id = tokenizer.pad_id if tokenizer is not None and hasattr(tokenizer, 'pad_id') else 0
    if pad_id < 0:
        pad_id = 0  # Для безопасности используем 0
    
    # Получаем максимальную длину последовательностей в этом батче
    max_length = max(len(item["input_ids"]) for item in batch)
    
    # Ограничиваем block_size
    max_length = min(max_length, block_size)
    
    # Если запрошено дополнение до block_size
    if pad_to_block_size:
        max_length = block_size
    
    # Подготавливаем тензоры
    input_ids = torch.full((len(batch), max_length), pad_id, dtype=torch.long)
    target_ids = torch.zeros(len(batch), dtype=torch.long)
    metadata = []
    
    # Заполняем тензоры без проверки каждого токена
    for i, item in enumerate(batch):
        seq_len = min(len(item["input_ids"]), max_length)
            input_ids[i, :seq_len] = torch.tensor(item["input_ids"][:seq_len], dtype=torch.long)
            target_ids[i] = item["target_id"]
            metadata.append(item.get("type", "unknown"))
            
    return input_ids, target_ids, metadata
