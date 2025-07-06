"""
General helper functions for training.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
from pathlib import Path
import json
from datetime import datetime

import structlog

logger = structlog.get_logger()


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Set random seed seed={seed}")


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get the best available device.
    
    Args:
        prefer_gpu: Whether to prefer GPU if available
        
    Returns:
        torch device
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU device={gpu_name}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    return device


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    filepath: Union[str, Path],
    additional_info: Optional[Dict[str, Any]] = None
):
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Current metrics
        filepath: Path to save checkpoint
        additional_info: Additional information to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        'torch_version': torch.__version__
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    # Ensure directory exists
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save checkpoint
    torch.save(checkpoint, filepath)
    
    # Save metrics separately as JSON for easy reading
    metrics_path = filepath.with_suffix('.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': checkpoint['timestamp']
        }, f, indent=2)
    
    logger.info(f"Saved checkpoint filepath={str(filepath}")
        epoch=epoch,
        metrics=metrics
    )


def load_checkpoint(
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load training checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        device: Device to load tensors to
        
    Returns:
        Checkpoint dictionary
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    # Load checkpoint
    if device:
        checkpoint = torch.load(filepath, map_location=device)
    else:
        checkpoint = torch.load(filepath)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Loaded checkpoint filepath={str(filepath}")
        epoch=checkpoint.get('epoch', 0),
        metrics=checkpoint.get('metrics', {})
    )
    
    return checkpoint


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_optimizer(
    model: nn.Module,
    config: Dict[str, Any]
) -> torch.optim.Optimizer:
    """
    Create optimizer from configuration.
    
    Args:
        model: Model to optimize
        config: Optimizer configuration
        
    Returns:
        Optimizer instance
    """
    optimizer_type = config.get('type', 'adam')
    learning_rate = config.get('learning_rate', 3e-4)
    weight_decay = config.get('weight_decay', 1e-5)
    
    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=config.get('betas', (0.9, 0.999),
            eps=config.get('eps', 1e-8)
        )
    elif optimizer_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=config.get('betas', (0.9, 0.999),
            eps=config.get('eps', 1e-8)
        )
    elif optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=config.get('momentum', 0.9),
            weight_decay=weight_decay,
            nesterov=config.get('nesterov', True)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    logger.info(f"Created optimizer type={optimizer_type} lr={learning_rate} weight_decay={weight_decay}")
    
    return optimizer


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any]
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler from configuration.
    
    Args:
        optimizer: Optimizer instance
        config: Scheduler configuration
        
    Returns:
        Scheduler instance or None
    """
    scheduler_type = config.get('type', None)
    
    if scheduler_type is None:
        return None
    
    if scheduler_type.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get('T_max', 100),
            eta_min=config.get('eta_min', 1e-6)
        )
    elif scheduler_type.lower() == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('step_size', 50),
            gamma=config.get('gamma', 0.5)
        )
    elif scheduler_type.lower() == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.get('gamma', 0.95)
        )
    elif scheduler_type.lower() == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.get('mode', 'min'),
            factor=config.get('factor', 0.5),
            patience=config.get('patience', 10),
            min_lr=config.get('min_lr', 1e-6)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    logger.info(f"Created scheduler type={scheduler_type} config={config}")
    
    return scheduler


class MetricsTracker:
    """
    Track and aggregate training metrics.
    """
    
    def __init__(self, metrics_to_track: List[str]):
        """
        Initialize metrics tracker.
        
        Args:
            metrics_to_track: List of metric names to track
        """
        self.metrics_to_track = metrics_to_track
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {metric: [] for metric in self.metrics_to_track}
        self.counts = {metric: 0 for metric in self.metrics_to_track}
    
    def update(self, metric_name: str, value: float, count: int = 1):
        """
        Update a metric.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            count: Number of samples
        """
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)
            self.counts[metric_name] += count
    
    def get_average(self, metric_name: str) -> float:
        """
        Get average value for a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Average value
        """
        if metric_name in self.metrics and self.metrics[metric_name]:
            return np.mean(self.metrics[metric_name])
        return 0.0
    
    def get_all_averages(self) -> Dict[str, float]:
        """
        Get average values for all metrics.
        
        Returns:
            Dictionary of average values
        """
        return {
            metric: self.get_average(metric)
            for metric in self.metrics_to_track
        }
    
    def get_latest(self, metric_name: str) -> float:
        """
        Get latest value for a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Latest value
        """
        if metric_name in self.metrics and self.metrics[metric_name]:
            return self.metrics[metric_name][-1]
        return 0.0


def create_experiment_name(config: Dict[str, Any]) -> str:
    """
    Create experiment name from configuration.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Experiment name
    """
    components = []
    
    # Add algorithm
    if 'algorithm' in config:
        components.append(config['algorithm'])
    
    # Add key hyperparameters
    if 'learning_rate' in config:
        components.append(f"lr{config['learning_rate']}")
    
    if 'batch_size' in config:
        components.append(f"bs{config['batch_size']}")
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    components.append(timestamp)
    
    return "_".join(components)


def log_system_info():
    """Log system information for reproducibility."""
    info = {
        'python_version': torch.version.__version__,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        info['gpu_names'] = [
            torch.cuda.get_device_name(i) 
            for i in range(torch.cuda.device_count())
        ]
    
    logger.info(f"System information {**info}")
    
    return info