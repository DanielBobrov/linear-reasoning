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
import math

from simple_tokenizer import SimpleTokenizer
from optimized_collator import optimized_collate_fn

class SingleTokenClassifier(nn.Module):
    """RecRNN model architecture based on the recurrent depth paper."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        
        # Embedding layer and scaling
        self.embedding = nn.Embedding(self.vocab_size, config.hidden_size, padding_idx=0)
        self.embedding_scale = math.sqrt(config.hidden_size)
        
        # The "Prelude" (P) block - embeds input data into latent space
        prelude_layers = []
        for _ in range(config.encoder_layers):
            prelude_layers.append(self._create_sandwich_layer(config))
        self.prelude = nn.ModuleList(prelude_layers)
        
        # The "Core" (R) recurrent block
        core_layers = []
        for _ in range(config.recurrent_layers):
            core_layers.append(self._create_sandwich_layer(config))
        self.core_block = nn.ModuleList(core_layers)
        
        # Adapter matrix to combine embedded input and previous state
        self.adapter = nn.Linear(2 * config.hidden_size, config.hidden_size)
        
        # The "Coda" (C) block - un-embeds from latent space
        coda_layers = []
        for _ in range(config.decoder_layers):
            coda_layers.append(self._create_sandwich_layer(config))
        self.coda = nn.ModuleList(coda_layers)
        
        # Final normalization and output projection
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Recurrence settings
        self.mean_recurrence = getattr(config, "mean_recurrence", 10)
        self.state_init_std = math.sqrt(2/5)  # σ for random state initialization
        
        # Apply initialization
        self.apply(self._init_weights)
        
    def _create_sandwich_layer(self, config):
        """Creates a sandwich-format layer as described in the paper."""
        layer = nn.ModuleDict({
            'norm1': nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            'norm2': nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            'norm3': nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            'norm4': nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            'attention': nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                dropout=config.hidden_dropout_prob,
                batch_first=True
            ),
            'mlp': nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size),
                nn.SiLU(),
                nn.Linear(config.intermediate_size, config.hidden_size),
                nn.Dropout(config.hidden_dropout_prob)
            )
        })
        return layer
    
    def _sandwich_forward(self, x, layer, attention_mask=None):
        """Forward pass through a sandwich-format layer."""
        # Attention part with sandwich norm
        x_norm1 = layer['norm1'](x)
        attn_output, _ = layer['attention'](
            query=x_norm1, 
            key=x_norm1, 
            value=x_norm1, 
            key_padding_mask=attention_mask,
            attn_mask=self._get_causal_mask(x.size(1)) if attention_mask is None else None
        )
        x = x + layer['norm2'](attn_output)
        
        # MLP part with sandwich norm
        x_norm3 = layer['norm3'](x)
        mlp_output = layer['mlp'](x_norm3)
        x = x + layer['norm4'](mlp_output)
        
        return x
    
    def _get_causal_mask(self, seq_len):
        """Creates a causal mask for self-attention."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask.to(next(self.parameters()).device)
        
    def _init_weights(self, module):
        """Initialize weights according to the paper."""
        if isinstance(module, nn.Linear):
            if module is self.classifier or "mlp" in str(module):
                # For output and MLP out projections
                std = 1/math.sqrt(5 * self.config.hidden_size * (self.config.encoder_layers + 
                                self.mean_recurrence * self.config.recurrent_layers + 
                                self.config.decoder_layers))
                nn.init.normal_(module.weight, mean=0.0, std=std)
            else:
                # For other linear layers
                std = math.sqrt(2/5) / math.sqrt(self.config.hidden_size)
                nn.init.normal_(module.weight, mean=0.0, std=std)
                
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range / 2.0)
            
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        
    def forward(self, x, attention_mask=None, num_recurrence=None):
        batch_size, seq_len = x.shape
        device = x.device
        
        # Determine number of recurrence steps
        if num_recurrence is None:
            num_recurrence = self.mean_recurrence
        
        # Apply embedding and scaling
        x = self.embedding(x) * self.embedding_scale
        
        # Get key_padding_mask for attention layers
        key_padding_mask = attention_mask if attention_mask is not None else None
        
        # Prelude: Process input through prelude layers
        e = x  # Store embedded input
        for layer in self.prelude:
            e = self._sandwich_forward(e, layer, key_padding_mask)
        
        # Initialize random state: s0 ~ N(0, σ²In·h)
        s = torch.randn(batch_size, seq_len, self.config.hidden_size, 
                      device=device) * self.state_init_std
        
        # Core: Apply recurrent block for num_recurrence iterations
        for _ in range(num_recurrence):
            # Combine state and embedded input using adapter
            combined = torch.cat([s, e], dim=-1)
            adapter_out = self.adapter(combined)
            
            # Process through core layers
            s = adapter_out
            for layer in self.core_block:
                s = self._sandwich_forward(s, layer, key_padding_mask)
        
        # Coda: Final processing
        c = s
        for layer in self.coda:
            c = self._sandwich_forward(c, layer, key_padding_mask)
        
        c = self.final_norm(c)
        
        # Mean pooling over sequence dimension if needed
        if key_padding_mask is not None:
            mask = ~key_padding_mask
            mask = mask.float()
            c = (c * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        else:
            c = torch.mean(c, dim=1)
        
        # Projection to vocabulary
        return self.classifier(c)
    
    def sample_num_recurrence(self):
        """Sample recurrence number from log-normal Poisson distribution as described in the paper."""
        mean_recurrence = self.mean_recurrence
        sigma = 0.5  # Fixed variance as in paper
        
        # Sample from log-normal Poisson distribution
        # Исправленный вызов torch.normal - корректная сигнатура для скалярных аргументов
        mean_val = math.log(mean_recurrence) - 0.5 * sigma**2
        tau = torch.normal(mean_val, sigma, (1,))
        
        # Use Poisson with rate e^τ
        rate = math.exp(tau.item())
        r = torch.poisson(torch.tensor([rate])).item() + 1
        
        # Ensure minimum of 1 step and reasonable maximum
        return max(1, min(int(r), 2 * mean_recurrence))

class OptimizedDataset:
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
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    seed: int = 42,
    device: str = None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    base_dir = Path(__file__).parent.absolute()
    
    if train_data_path is None:
        train_data_path = "/kaggle/working/optimized/train_optimized.json"
    if val_data_path is None:
        val_data_path = "/kaggle/working/optimized/valid_optimized.json"
    
    if not Path(train_data_path).exists():
        print(f"ОШИБКА: Файл {train_data_path} не найден!")
        alt_paths = list(Path("/kaggle/working").glob("**/*train*optimized*.json"))
        if alt_paths:
            print(f"Найдены альтернативные файлы: {alt_paths}")
            train_data_path = str(alt_paths[0])
        else:
            print("Альтернативные файлы не найдены. Убедитесь, что prepare_optimized_data.py был выполнен.")
            return
    
    if not Path(val_data_path).exists():
        print(f"ОШИБКА: Файл {val_data_path} не найден!")
        alt_paths = list(Path("/kaggle/working").glob("**/*valid*optimized*.json"))
        if alt_paths:
            print(f"Найдены альтернативные файлы: {alt_paths}")
            val_data_path = str(alt_paths[0])
        else:
            print("Альтернативные файлы не найдены. Убедитесь, что prepare_optimized_data.py был выполнен.")
            return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vocab_paths = [
        "/kaggle/input/paper-data/data/data/comparison.1000.12.6/vocab.json",
        "/kaggle/working/data/vocab.json"
    ]
    
    vocab_path = None
    for path in vocab_paths:
        if Path(path).exists():
            vocab_path = path
            break
    
    if vocab_path is None:
        print("ОШИБКА: vocab.json не найден!")
        alt_paths = list(Path("/kaggle").glob("**/vocab.json"))
        if alt_paths:
            print(f"Найдены альтернативные файлы: {alt_paths}")
            vocab_path = str(alt_paths[0])
        else:
            print("vocab.json не найден. Обучение невозможно.")
            return
    
    print(f"Loading tokenizer from: {vocab_path}")
    tokenizer = SimpleTokenizer(vocab_path)
    
    tokenizer.pad_id = 0
    
    try:
        with open(train_data_path, 'r') as f:
            train_samples = json.load(f)
            max_token_id = -1
            for item in train_samples[:1000]:
                max_token_id = max(max_token_id, max(item['input_ids'] + [item['target_id']]))
            print(f"Maximum token ID in data: {max_token_id}, vocab_size={tokenizer.vocab_size}")
            
            if max_token_id >= tokenizer.vocab_size:
                print(f"WARNING: Maximum token ID {max_token_id} exceeds vocab_size {tokenizer.vocab_size}")
                tokenizer.vocab_size = max_token_id + 10
    except Exception as e:
        print(f"Error scanning data: {e}")
    
    from kaggle_run import SmallRecRNNConfig
    model_config = SmallRecRNNConfig(
        vocab_size=tokenizer.vocab_size, 
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        recurrent_layers=2,
        encoder_layers=1, 
        decoder_layers=1,
        num_attention_heads=8,
        block_size=256,
        mean_recurrence=10,
    )
    
    estimated_params = model_config.estimate_params() if hasattr(model_config, "estimate_params") else "N/A"
    print(f"Estimated model parameters: ~{estimated_params/1_000_000:.2f}M" 
          if isinstance(estimated_params, (int, float)) else f"Estimation not available")
    
    print(f"Final model configuration: vocab_size={model_config.vocab_size}, hidden_size={model_config.hidden_size}")
    
    print("Creating model...")
    model = SingleTokenClassifier(model_config).to(device)
    
    print(f"Loading datasets from:")
    print(f"  Train: {train_data_path}")
    print(f"  Valid: {val_data_path}")
    
    train_dataset = OptimizedDataset(train_data_path)
    val_dataset = OptimizedDataset(val_data_path)
    
    print(f"Training data: {len(train_dataset)} samples")
    print(f"Validation data: {len(val_dataset)} samples")
    
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
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate * 0.1, weight_decay=0.1, eps=1e-8)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=3, verbose=True, min_lr=1e-7
    )
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    lr_history = []
    all_step_metrics = []
    
    metrics_dir = Path(output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = metrics_dir / "training_metrics.csv"
    
    with open(metrics_file, 'w') as f:
        f.write("epoch,step,train_loss,val_loss,learning_rate\n")
    
    update_interval = max(1, len(train_dataset) // (batch_size * 20))
    global_progress = tqdm(total=epochs * len(train_dataset) // batch_size, 
                          desc="Training", 
                          position=0,
                          leave=True,
                          ncols=100,
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    running_loss = torch.zeros(1, device=device)
    loss_count = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        batch_count = 0
        epoch_step_metrics = []
        
        for step, (input_ids, target_ids, _) in enumerate(train_loader):
            if input_ids.size(0) == 0:
                continue
                
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            attention_mask = (input_ids == tokenizer.pad_id)
            
            if model.training:
                # Use the log-normal Poisson sampling for recurrence steps
                num_recurrence = model.sample_num_recurrence()
            else:
                num_recurrence = model_config.mean_recurrence
                
            logits = model(input_ids, attention_mask, num_recurrence=num_recurrence)
            loss = criterion(logits, target_ids)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            running_loss += loss.item()
            loss_count += 1
            batch_count += 1
            
            if batch_count % update_interval == 0:
                current_lr = optimizer.param_groups[0]['lr']
                avg_loss = running_loss.item() / max(1, loss_count)
                global_progress.set_description(
                    f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.2f}, LR: {current_lr:.2e}"
                )
                global_progress.update(update_interval)
                
                step_idx = step + epoch * len(train_loader)
                step_data = {
                    "step": step_idx,
                    "loss": avg_loss,
                    "lr": current_lr
                }
                epoch_step_metrics.append(step_data)
                all_step_metrics.append(step_data)
                
                running_loss.zero_()
                loss_count = 0
        
        remaining_steps = batch_count % update_interval
        if remaining_steps > 0:
            global_progress.update(remaining_steps)
        
        avg_train_loss = train_loss / batch_count if batch_count > 0 else float('inf')
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_batch_count = 0
        
        with torch.no_grad():
            for input_ids, target_ids, _ in val_loader:
                if input_ids.size(0) == 0:
                    continue
                    
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                
                attention_mask = (input_ids == tokenizer.pad_id)
                
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, target_ids)
                
                val_loss += loss.item()
                val_batch_count += 1
                
                _, predicted = torch.max(logits, 1)
                val_total += target_ids.size(0)
                val_correct += (predicted == target_ids).sum().item()
        
        avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')
        val_accuracy = 100 * val_correct / val_total if val_total > 0 else 0
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        if epoch % 1 == 0:
            model.eval()
            train_correct = 0
            train_total = 0
            with torch.no_grad():
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
            model.train()
        else:
            train_accuracy = train_accuracies[-1] if train_accuracies else 0
            train_accuracies.append(train_accuracy)
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        
        with open(metrics_file, 'a') as f:
            f.write(f"{epoch},-1,{avg_train_loss:.6f},{avg_val_loss:.6f},{current_lr:.8e}\n")
        
        print(f"\nEpoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, "
              f"LR: {current_lr:.2e}")
        
        if epoch == 0 or epoch == epochs - 1 or epoch % 5 == 0:
            print("\nDiagnostic check:")
            for name, param in model.named_parameters():
                if 'weight' in name and len(param.shape) >= 2:
                    print(f"  {name}: shape={param.shape}, mean={param.mean().item():.4f}, "
                          f"std={param.std().item():.4f}, max={param.max().item():.4f}, "
                          f"min={param.min().item():.4f}, "
                          f"has_nan={torch.isnan(param).any().item()}")
            
            param_norm = 0
            grad_norm = 0
            for p in model.parameters():
                param_norm += p.norm().item() ** 2
                if p.grad is not None:
                    grad_norm += p.grad.norm().item() ** 2
            print(f"  Total parameter norm: {param_norm**0.5:.4f}")
            print(f"  Total gradient norm: {grad_norm**0.5:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
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
    
    global_progress.close()
    
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
    }, output_dir / "final_model.pt")  
    
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_accuracy': train_accuracies,
        'val_accuracy': val_accuracies,
        'learning_rate': lr_history,
        'step_metrics': all_step_metrics
    }
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f)
        
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        
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
    train_model(
        epochs=20,
        batch_size=32, 
        learning_rate=1e-4
    )
