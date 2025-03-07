import json
import os
import torch
from typing import List, Dict, Optional, Union

class SimpleTokenizer:
    """A simplified tokenizer that uses vocab.json directly with minimal special token handling"""
    
    def __init__(self, vocab_path):
        # Load vocabulary
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        
        # Create token to id mapping
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
        
        # Minimal setup - только необходимые атрибуты
        self.vocab_size = len(self.vocab)
        self.pad_id = -1  # Default padding ID
        
        # Мы сохраняем только mask_token_id, так как он используется в данных
        self.mask_token_id = self.token_to_id.get("<mask>") if "<mask>" in self.token_to_id else None
        
    def encode(self, text: str) -> List[int]:
        """Encode text into token ids"""
        # For our case, the text already contains tokens like <e_X>
        tokens = text.split()
        ids = []
        
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                # Если токен не найден - это критическая ошибка
                raise ValueError(f"Token not found in vocabulary: '{token}'. This indicates a bug in tokenization logic!")
        
        return ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token ids back to text"""
        tokens = []
        for id in token_ids:
            if id in self.id_to_token:
                tokens.append(self.id_to_token[id])
            else:
                # Если ID токена за пределами словаря - это критическая ошибка
                raise ValueError(f"Token ID {id} is out of vocabulary range (0-{self.vocab_size-1}). This indicates a bug in the model or data processing!")
        
        return " ".join(tokens)
    
    def batch_encode(self, texts: List[str]):
        """Encode a batch of texts"""
        return [self.encode(text) for text in texts]
    
    @property
    def processor(self):
        """Return an object with properties needed by the model"""
        class TokenizerProcessor:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer
                
            @property
            def vocab_size(self):
                return self.tokenizer.vocab_size
        
        return TokenizerProcessor(self)


class Tokenizer:
    """Wrapper class for compatibility with the training script"""
    
    def __init__(self, tokenizer_path):
        self.tokenizer = SimpleTokenizer(tokenizer_path)
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_id = self.tokenizer.pad_id
        self.processor = self.tokenizer.processor

