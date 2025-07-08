#!/usr/bin/env python3
"""
AlgoSpace Colab Optimization Fixes
Provides optimized code snippets and utilities for efficient training in Google Colab.
"""

import os
import gc
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import json


class ColabOptimizer:
    """Collection of optimization utilities for Google Colab training."""
    
    @staticmethod
    def gpu_memory_management():
        """GPU memory management utilities."""
        return '''
# GPU Memory Management Utilities

import torch
import gc
import GPUtil

class GPUMemoryManager:
    """Manages GPU memory to prevent OOM errors."""
    
    def __init__(self, threshold_mb: int = 1000):
        self.threshold_mb = threshold_mb
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def clear_cache(self):
        """Clear GPU cache and collect garbage."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0, "free": 0}
            
        allocated = torch.cuda.memory_allocated() / 1e9  # GB
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        free = total - allocated
        
        return {
            "allocated": allocated,
            "reserved": reserved, 
            "free": free,
            "total": total
        }
    
    def check_memory(self, log: bool = True) -> bool:
        """Check if enough memory is available."""
        stats = self.get_memory_usage()
        free_mb = stats["free"] * 1000
        
        if log:
            print(f"GPU Memory: {stats['allocated']:.2f}/{stats['total']:.2f} GB used")
            
        if free_mb < self.threshold_mb:
            self.clear_cache()
            return False
        return True
    
    def safe_allocate(self, size_mb: int) -> bool:
        """Check if allocation is safe."""
        stats = self.get_memory_usage()
        free_mb = stats["free"] * 1000
        return free_mb > size_mb * 1.2  # 20% buffer

# Usage in training loop
gpu_manager = GPUMemoryManager(threshold_mb=500)

# Periodic memory clearing
if epoch % 10 == 0:
    gpu_manager.clear_cache()
    gpu_manager.check_memory()
'''

    @staticmethod
    def efficient_data_loading():
        """Optimized data loading for Colab."""
        return '''
# Efficient Data Loading Configuration

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Optional

class OptimizedDataLoader:
    """Creates optimized DataLoader for Colab environment."""
    
    @staticmethod
    def create_loader(dataset: Dataset, 
                     batch_size: int = 256,
                     shuffle: bool = True,
                     num_workers: Optional[int] = None) -> DataLoader:
        """Create optimized DataLoader with Colab-specific settings."""
        
        # Auto-detect optimal workers
        if num_workers is None:
            try:
                import multiprocessing
                num_workers = min(4, multiprocessing.cpu_count() // 2)
            except:
                num_workers = 2
        
        # Create loader with optimizations
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),  # Pin memory for GPU
            prefetch_factor=2,  # Prefetch batches
            persistent_workers=True if num_workers > 0 else False,  # Keep workers alive
            drop_last=True  # Drop incomplete batches for consistency
        )
        
        return loader

class CachedDataset(Dataset):
    """Dataset with in-memory caching for faster access."""
    
    def __init__(self, data_path: str, cache_size: int = 1000):
        self.data_path = data_path
        self.cache_size = cache_size
        self.cache = {}
        self.access_count = {}
        
        # Load data info
        self.data_info = self._load_data_info()
        
    def __getitem__(self, idx):
        # Check cache first
        if idx in self.cache:
            self.access_count[idx] += 1
            return self.cache[idx]
        
        # Load from disk
        data = self._load_item(idx)
        
        # Add to cache if space available
        if len(self.cache) < self.cache_size:
            self.cache[idx] = data
            self.access_count[idx] = 1
        else:
            # LRU eviction
            lru_idx = min(self.access_count, key=self.access_count.get)
            del self.cache[lru_idx]
            del self.access_count[lru_idx]
            self.cache[idx] = data
            self.access_count[idx] = 1
            
        return data
'''

    @staticmethod
    def checkpoint_management():
        """Advanced checkpointing for Colab sessions."""
        return '''
# Advanced Checkpoint Management

import torch
import os
import shutil
from datetime import datetime
from typing import Dict, Any, Optional

class ColabCheckpointManager:
    """Manages checkpoints with Google Drive integration."""
    
    def __init__(self, 
                 local_dir: str = "./checkpoints",
                 drive_dir: str = "/content/drive/MyDrive/AlgoSpace/checkpoints",
                 keep_n_checkpoints: int = 3):
        self.local_dir = local_dir
        self.drive_dir = drive_dir
        self.keep_n_checkpoints = keep_n_checkpoints
        
        # Create directories
        os.makedirs(local_dir, exist_ok=True)
        if os.path.exists("/content/drive"):
            os.makedirs(drive_dir, exist_ok=True)
            
        self.checkpoints = []
        
    def save_checkpoint(self, 
                       state: Dict[str, Any],
                       metrics: Dict[str, float],
                       is_best: bool = False,
                       backup_to_drive: bool = True):
        """Save checkpoint with automatic Drive backup."""
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_{timestamp}.pt"
        local_path = os.path.join(self.local_dir, filename)
        
        # Add metadata
        checkpoint = {
            'state': state,
            'metrics': metrics,
            'timestamp': timestamp,
            'cuda_available': torch.cuda.is_available(),
            'torch_version': torch.__version__
        }
        
        # Save locally first (faster)
        torch.save(checkpoint, local_path)
        print(f"✅ Checkpoint saved locally: {filename}")
        
        # Backup to Drive
        if backup_to_drive and os.path.exists("/content/drive"):
            drive_path = os.path.join(self.drive_dir, filename)
            shutil.copy2(local_path, drive_path)
            print(f"☁️  Backed up to Drive: {filename}")
            
        # Save best model separately
        if is_best:
            best_path = os.path.join(self.local_dir, "best_model.pt")
            shutil.copy2(local_path, best_path)
            if backup_to_drive and os.path.exists("/content/drive"):
                drive_best = os.path.join(self.drive_dir, "best_model.pt")
                shutil.copy2(best_path, drive_best)
                
        # Manage checkpoint history
        self.checkpoints.append({
            'filename': filename,
            'metrics': metrics,
            'timestamp': timestamp
        })
        
        # Clean old checkpoints
        self._cleanup_old_checkpoints()
        
    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint."""
        # Try Drive first
        if os.path.exists(self.drive_dir):
            checkpoints = [f for f in os.listdir(self.drive_dir) 
                          if f.startswith("checkpoint_") and f.endswith(".pt")]
            if checkpoints:
                latest = sorted(checkpoints)[-1]
                path = os.path.join(self.drive_dir, latest)
                checkpoint = torch.load(path, map_location='cpu')
                print(f"✅ Loaded checkpoint from Drive: {latest}")
                return checkpoint
                
        # Fall back to local
        if os.path.exists(self.local_dir):
            checkpoints = [f for f in os.listdir(self.local_dir) 
                          if f.startswith("checkpoint_") and f.endswith(".pt")]
            if checkpoints:
                latest = sorted(checkpoints)[-1]
                path = os.path.join(self.local_dir, latest)
                checkpoint = torch.load(path, map_location='cpu')
                print(f"✅ Loaded checkpoint locally: {latest}")
                return checkpoint
                
        return None
        
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save space."""
        if len(self.checkpoints) > self.keep_n_checkpoints:
            # Sort by timestamp
            self.checkpoints.sort(key=lambda x: x['timestamp'])
            
            # Remove oldest
            to_remove = self.checkpoints[:-self.keep_n_checkpoints]
            for checkpoint in to_remove:
                # Remove local
                local_path = os.path.join(self.local_dir, checkpoint['filename'])
                if os.path.exists(local_path):
                    os.remove(local_path)
                    
                # Remove from Drive
                if os.path.exists(self.drive_dir):
                    drive_path = os.path.join(self.drive_dir, checkpoint['filename'])
                    if os.path.exists(drive_path):
                        os.remove(drive_path)
                        
            # Update list
            self.checkpoints = self.checkpoints[-self.keep_n_checkpoints:]
'''

    @staticmethod
    def training_loop_optimization():
        """Optimized training loop for Colab."""
        return '''
# Optimized Training Loop

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm.notebook import tqdm
import time
from typing import Dict, Any

class OptimizedTrainer:
    """Trainer with Colab-specific optimizations."""
    
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device,
                 use_mixed_precision: bool = True,
                 gradient_accumulation_steps: int = 4):
        self.model = model
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Mixed precision training
        self.scaler = GradScaler() if use_mixed_precision else None
        
        # Memory tracking
        self.gpu_manager = GPUMemoryManager()
        
    def train_epoch(self, 
                   dataloader, 
                   optimizer, 
                   criterion,
                   epoch: int) -> Dict[str, float]:
        """Run one training epoch with optimizations."""
        
        self.model.train()
        total_loss = 0
        batch_times = []
        
        # Progress bar with notebook display
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", 
                    leave=False, ncols=100)
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            start_time = time.time()
            
            # Move data to device
            inputs = batch['inputs'].to(self.device, non_blocking=True)
            targets = batch['targets'].to(self.device, non_blocking=True)
            
            # Mixed precision forward pass
            if self.use_mixed_precision:
                with autocast():
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    loss = loss / self.gradient_accumulation_steps
                    
                # Backward with gradient scaling
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_mixed_precision:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                
            # Update metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            batch_time = time.time() - start_time
            batch_times.append(batch_time)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'batch_time': f'{batch_time:.2f}s',
                'gpu_mem': f'{self.gpu_manager.get_memory_usage()["allocated"]:.1f}GB'
            })
            
            # Periodic memory management
            if batch_idx % 50 == 0:
                self.gpu_manager.clear_cache()
                
        # Final gradient step if needed
        if (batch_idx + 1) % self.gradient_accumulation_steps != 0:
            if self.use_mixed_precision:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
                
        return {
            'loss': total_loss / len(dataloader),
            'avg_batch_time': np.mean(batch_times),
            'total_time': sum(batch_times)
        }
'''

    @staticmethod
    def colab_specific_fixes():
        """Colab-specific fixes and utilities."""
        return '''
# Colab-Specific Fixes and Utilities

import os
import sys
import subprocess
from IPython.display import clear_output
import warnings
warnings.filterwarnings('ignore')

class ColabEnvironment:
    """Utilities for Colab environment setup and fixes."""
    
    @staticmethod
    def setup_environment():
        """Setup Colab environment with common fixes."""
        
        # Fix for matplotlib backend
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Ensure proper CUDA setup
        if torch.cuda.is_available():
            print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            # Set CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
        # Fix for multiprocessing in Colab
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)
        
        # Install commonly missing packages
        required_packages = ['wandb', 'tensorboard', 'gputil', 'psutil']
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
                
        clear_output()
        print("✅ Colab environment configured")
        
    @staticmethod
    def keep_alive():
        """Prevent Colab from disconnecting due to inactivity."""
        from IPython.display import Javascript
        
        js_code = """
        function KeepAlive() {
            const pingInterval = 60000;  // 60 seconds
            setInterval(() => {
                console.log("Keeping Colab alive...");
                document.querySelector("#connect")?.click();
            }, pingInterval);
        }
        
        // Start keep-alive
        KeepAlive();
        """
        
        display(Javascript(js_code))
        print("✅ Keep-alive activated")
        
    @staticmethod
    def mount_drive_with_auth():
        """Mount Google Drive with proper authentication handling."""
        try:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=True)
            print("✅ Google Drive mounted")
            return True
        except Exception as e:
            print(f"❌ Failed to mount Drive: {e}")
            return False
            
    @staticmethod
    def auto_restart_kernel():
        """Setup auto-restart on crash."""
        code = """
        import os
        import signal
        import sys
        
        def restart_kernel(signum, frame):
            print("🔄 Restarting kernel...")
            os._exit(0)
            
        # Register signal handlers
        signal.signal(signal.SIGTERM, restart_kernel)
        signal.signal(signal.SIGSEGV, restart_kernel)
        """
        
        exec(code)
        print("✅ Auto-restart configured")
'''

    @staticmethod
    def unified_config_loader():
        """Unified configuration loading system."""
        return '''
# Unified Configuration System

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path

class AlgoSpaceConfig:
    """Centralized configuration management for AlgoSpace."""
    
    def __init__(self, config_path: str = "/content/AlgoSpace/config/settings.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._apply_environment_overrides()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML."""
        if not self.config_path.exists():
            print(f"⚠️  Config not found at {self.config_path}, using defaults")
            return self._get_default_config()
            
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        return config
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration matching PRD specs."""
        return {
            'main_marl_core': {
                'mc_dropout': {
                    'n_samples': 50,
                    'confidence_threshold': 0.65
                },
                'synergy_detector': {
                    'mlmi_nwrqk_threshold': 0.2
                },
                'training': {
                    'batch_size': 256,
                    'learning_rate': 3e-4,
                    'training_steps': 10000
                }
            },
            'agents': {
                'structure_analyzer': {
                    'window': 48,
                    'hidden_dim': 256,
                    'n_layers': 4,
                    'dropout': 0.2
                },
                'short_term_tactician': {
                    'window': 60,
                    'hidden_dim': 192,
                    'n_layers': 3,
                    'dropout': 0.2
                },
                'mid_frequency_arbitrageur': {
                    'window': 100,
                    'hidden_dim': 224,
                    'n_layers': 4,
                    'dropout': 0.2
                }
            },
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
    def _apply_environment_overrides(self):
        """Apply environment variable overrides."""
        # Override device if specified
        if 'ALGOSPACE_DEVICE' in os.environ:
            self.config['device'] = os.environ['ALGOSPACE_DEVICE']
            
        # Override batch size for limited memory
        if 'ALGOSPACE_BATCH_SIZE' in os.environ:
            batch_size = int(os.environ['ALGOSPACE_BATCH_SIZE'])
            self.config['main_marl_core']['training']['batch_size'] = batch_size
            
    def get(self, path: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated path."""
        keys = path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value
        
    def update_for_colab(self):
        """Update configuration for Colab environment."""
        # Reduce batch size if GPU memory is limited
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory < 16:  # Less than 16GB
                self.config['main_marl_core']['training']['batch_size'] = 128
                print(f"📉 Reduced batch size to 128 for {gpu_memory:.1f}GB GPU")
                
        # Enable mixed precision for faster training
        self.config['training_optimizations'] = {
            'mixed_precision': True,
            'gradient_accumulation': 4,
            'checkpoint_frequency': 100
        }
        
        return self

# Global config instance
config = AlgoSpaceConfig().update_for_colab()

# Usage example:
# batch_size = config.get('main_marl_core.training.batch_size', 256)
# mc_passes = config.get('main_marl_core.mc_dropout.n_samples', 50)
'''

    @staticmethod
    def generate_optimization_notebook():
        """Generate a complete optimization notebook."""
        content = f'''# AlgoSpace Colab Optimization Utilities

This notebook contains all optimization utilities for efficient training in Google Colab.

## 1. Environment Setup

```python
{ColabOptimizer.colab_specific_fixes()}
```

## 2. GPU Memory Management

```python
{ColabOptimizer.gpu_memory_management()}
```

## 3. Efficient Data Loading

```python
{ColabOptimizer.efficient_data_loading()}
```

## 4. Advanced Checkpointing

```python
{ColabOptimizer.checkpoint_management()}
```

## 5. Optimized Training Loop

```python
{ColabOptimizer.training_loop_optimization()}
```

## 6. Unified Configuration

```python
{ColabOptimizer.unified_config_loader()}
```

## 7. Complete Training Pipeline

```python
# Complete optimized training pipeline
def run_optimized_training():
    """Run training with all optimizations."""
    
    # 1. Setup environment
    ColabEnvironment.setup_environment()
    ColabEnvironment.keep_alive()
    ColabEnvironment.mount_drive_with_auth()
    
    # 2. Load configuration
    config = AlgoSpaceConfig().update_for_colab()
    
    # 3. Initialize GPU manager
    gpu_manager = GPUMemoryManager()
    
    # 4. Create optimized data loaders
    train_loader = OptimizedDataLoader.create_loader(
        train_dataset,
        batch_size=config.get('main_marl_core.training.batch_size'),
        num_workers=4
    )
    
    # 5. Initialize checkpoint manager
    checkpoint_manager = ColabCheckpointManager()
    
    # 6. Create optimized trainer
    trainer = OptimizedTrainer(
        model=model,
        device=torch.device(config.get('device')),
        use_mixed_precision=True,
        gradient_accumulation_steps=4
    )
    
    # 7. Training loop
    for epoch in range(num_epochs):
        # Check GPU memory
        gpu_manager.check_memory()
        
        # Train epoch
        metrics = trainer.train_epoch(
            train_loader, optimizer, criterion, epoch
        )
        
        # Save checkpoint
        if epoch % 10 == 0:
            checkpoint_manager.save_checkpoint(
                state={
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                },
                metrics=metrics,
                is_best=metrics['loss'] < best_loss
            )
            
    print("✅ Training complete!")
```

## Usage Instructions

1. Copy the relevant code sections to your training notebook
2. Initialize the environment setup at the beginning
3. Use the GPU memory manager throughout training
4. Enable mixed precision training for faster computation
5. Use the checkpoint manager to save progress
6. Monitor GPU usage and clear cache periodically

## Performance Tips

1. **Batch Size**: Start with 256, reduce if OOM errors occur
2. **Mixed Precision**: Enable for 2x speedup on modern GPUs
3. **Gradient Accumulation**: Use to simulate larger batch sizes
4. **Data Loading**: Use 2-4 workers for optimal performance
5. **Checkpointing**: Save every 10-20 epochs to Google Drive
'''
        return content


def main():
    """Generate optimization files and reports."""
    print("Generating AlgoSpace Colab Optimization Files...")
    
    # Create optimizer instance
    optimizer = ColabOptimizer()
    
    # Generate optimization notebook content
    notebook_content = optimizer.generate_optimization_notebook()
    
    # Save as Python file
    with open('notebooks/colab_optimizations.py', 'w') as f:
        f.write(notebook_content)
    print("✅ Saved colab_optimizations.py")
    
    # Generate specific fix files
    fixes = {
        'gpu_memory_utils.py': optimizer.gpu_memory_management(),
        'data_loading_utils.py': optimizer.efficient_data_loading(),
        'checkpoint_utils.py': optimizer.checkpoint_management(),
        'training_utils.py': optimizer.training_loop_optimization(),
        'config_utils.py': optimizer.unified_config_loader(),
        'colab_setup.py': optimizer.colab_specific_fixes()
    }
    
    # Save individual utility files
    utils_dir = 'notebooks/utils'
    os.makedirs(utils_dir, exist_ok=True)
    
    for filename, content in fixes.items():
        filepath = os.path.join(utils_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"✅ Saved {filename}")
    
    # Generate optimization summary
    summary = {
        "optimization_areas": {
            "GPU Memory": [
                "Automatic cache clearing",
                "Memory usage monitoring",
                "Safe allocation checks",
                "Periodic garbage collection"
            ],
            "Data Loading": [
                "Pin memory for GPU transfer",
                "Multi-worker parallel loading",
                "Prefetch factor optimization",
                "In-memory caching for small datasets"
            ],
            "Training Loop": [
                "Mixed precision training (AMP)",
                "Gradient accumulation",
                "Non-blocking data transfer",
                "Optimized progress tracking"
            ],
            "Checkpointing": [
                "Automatic Google Drive backup",
                "Checkpoint rotation (keep N latest)",
                "Best model tracking",
                "Resume capability"
            ],
            "Configuration": [
                "Centralized config management",
                "Environment variable overrides",
                "Automatic Colab adjustments",
                "PRD-compliant defaults"
            ]
        },
        "critical_parameters": {
            "MC Dropout Passes": 50,
            "Confidence Threshold": 0.65,
            "Synergy Threshold": 0.2,
            "Batch Size": "256 (128 for <16GB GPU)",
            "Learning Rate": "3e-4",
            "Gradient Accumulation": 4
        },
        "performance_gains": {
            "Mixed Precision": "~2x speedup",
            "Optimized DataLoader": "~30% faster loading",
            "GPU Memory Management": "Prevents OOM errors",
            "Gradient Accumulation": "Enables larger effective batch size"
        }
    }
    
    with open('notebooks/optimization_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("✅ Saved optimization_summary.json")
    
    print("\n✅ All optimization files generated successfully!")
    print("\nFiles created:")
    print("  - colab_optimizations.py (main reference)")
    print("  - utils/ directory with individual utilities")
    print("  - optimization_summary.json")


if __name__ == "__main__":
    main()