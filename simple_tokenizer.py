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
        
        print(f"Loaded vocabulary with {len(self.vocab)} tokens from {vocab_path}")
        print(f"First 5 tokens: {self.vocab[:5]}")
        print(f"Last 5 tokens: {self.vocab[-5:]}")
        
        # Create token to id mapping
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
        
        # Minimal setup - только необходимые атрибуты
        self.vocab_size = len(self.vocab)
        self.pad_id = 0  # Используем 0 вместо -1
        
        # Мы сохраняем только mask_token_id, так как он используется в данных
        self.mask_token_id = self.token_to_id.get("<mask>") if "<mask>" in self.token_to_id else None
        
    def encode(self, text: str) -> List[int]:
        """Encode text into token ids"""
        # For our case, the text already contains tokens like <e_X>
        tokens = ["<"+i+">" for i in text[1:-1].split("><")]
        ids = []
        for i, token in enumerate(tokens):
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                raise ValueError(f"Token '{token}' (#{i+1}) not found in vocabulary in text: '{text}'")
        
        return ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token ids back to text"""
        tokens = []
        for i, id in enumerate(token_ids):
            if id in self.id_to_token:
                tokens.append(self.id_to_token[id])
            else:
                raise ValueError(f"Token ID {id} (position {i}) not found in vocabulary (range: 0-{self.vocab_size-1})")
        
        return "".join(tokens)  # Объединяем без пробелов для формата <token1><token2>
    
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

