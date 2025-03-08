from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, Tuple, Type

@dataclass
class SmallRecRNNConfig:
    """Configuration for the smaller RecRNN model (30M parameters)"""
    # Model architecture
    vocab_size: int = 1205  # Based on the provided vocab.json
    hidden_size: int = 512  # Reduced significantly from original
    intermediate_size: int = 1024  # 2x hidden_size as in original
    num_hidden_layers: int = 4  # Total layers: encoder + recurrent + decoder
    recurrent_layers: int = 2  # Number of layers in recurrent block
    encoder_layers: int = 1  # Number of encoder layers
    decoder_layers: int = 1  # Number of decoder layers
    num_attention_heads: int = 8  # Reduced from original
    block_size: int = 256  # Context window size
    
    # Recurrence settings - эти параметры из оригинальной модели
    mean_recurrence: int = 6  # Default recurrence depth
    use_cache: bool = True  # Enable caching for inference
    recurrent_chunk_size: Optional[int] = None  # Chunk size for recurrent processing
    
    # Regularization settings - оставляем как в оригинальной модели
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

    # Функция для оценки количества параметров (примерно)
    def estimate_params(self):
        # Оценка эмбеддингов
        embedding_params = self.vocab_size * self.hidden_size
        
        # Оценка слоев трансформера (приближенно)
        # Каждый слой имеет примерно 4 * h^2 параметров (где h - hidden_size)
        layer_params = 4 * (self.hidden_size ** 2)
        
        # Учитываем proj слои и attention heads
        attn_params = self.hidden_size * self.hidden_size * self.num_attention_heads 
        
        # Feedforward params
        ff_params = self.hidden_size * self.intermediate_size * 2
        
        # Суммарно за один слой
        per_layer = attn_params + ff_params + self.hidden_size * 2  # + нормализации
        
        # Общее количество слоев
        total_layers_params = per_layer * self.num_hidden_layers
        
        # Выходной слой
        output_params = self.hidden_size * self.vocab_size
        
        # Итоговое количество
        total_params = embedding_params + total_layers_params + output_params
        
        return total_params
