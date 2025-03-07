from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, Tuple, Type
import torch
import torch.nn as nn

@dataclass
class SmallRecRNNConfig:
    """Configuration for the smaller RecRNN model (30M parameters)"""
    # Model architecture
    vocab_size: int = 1205  # Based on the provided vocab.json
    hidden_size: int = 768  # Default value, can be overridden
    intermediate_size: int = 2048  # Can be calculated as hidden_size * 2
    num_hidden_layers: int = 4  # Reduced from 8 (1 encoder + 2 recurrent + 1 decoder)
    recurrent_layers: int = 2  # Reduced recurrent layers
    encoder_layers: int = 1  # Reduced encoder layers
    decoder_layers: int = 1  # Reduced decoder layers
    num_attention_heads: int = 8  # Can be scaled with hidden_size
    block_size: int = 256  # Context window size
    
    # Recurrence settings
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
