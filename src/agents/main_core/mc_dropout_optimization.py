"""
GPU-optimized batching utilities for MC Dropout consensus mechanism.

This module provides performance optimizations for MC Dropout including
parallel batching, streaming statistics, and adaptive sampling strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MCDropoutBatcher:
    """
    Optimizes MC Dropout by batching multiple forward passes.
    
    Reduces memory transfers and improves GPU utilization.
    """
    
    def __init__(self, device: torch.device, max_batch_size: int = 50):
        self.device = device
        self.max_batch_size = max_batch_size
        
    def batch_forward(
        self,
        model: nn.Module,
        input_state: torch.Tensor,
        n_samples: int
    ) -> torch.Tensor:
        """
        Perform batched forward passes.
        
        Args:
            model: Neural network model
            input_state: Input tensor [1, features]
            n_samples: Number of MC samples
            
        Returns:
            Predictions tensor [n_samples, 1, outputs]
        """
        # Calculate optimal batch configuration
        n_batches = (n_samples + self.max_batch_size - 1) // self.max_batch_size
        samples_per_batch = n_samples // n_batches
        remainder = n_samples % n_batches
        
        all_outputs = []
        
        for batch_idx in range(n_batches):
            # Determine batch size
            batch_size = samples_per_batch
            if batch_idx < remainder:
                batch_size += 1
                
            # Replicate input
            batch_input = input_state.repeat(batch_size, 1)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(batch_input)
                
            # Handle different output formats
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Take first element (usually mean)
            elif isinstance(outputs, dict):
                outputs = outputs.get('action_logits', outputs.get('mu'))
                
            all_outputs.append(outputs)
            
        # Concatenate all batches
        return torch.cat(all_outputs, dim=0).unsqueeze(1)


class StreamingStatistics:
    """
    Calculate statistics in streaming fashion to reduce memory usage.
    
    Useful for very large numbers of MC samples.
    """
    
    def __init__(self):
        self.n = 0
        self.mean = None
        self.m2 = None
        self.min_val = None
        self.max_val = None
        
    def update(self, x: torch.Tensor):
        """Update statistics with new batch."""
        batch_size = x.size(0)
        
        if self.n == 0:
            self.mean = x.mean(dim=0)
            self.m2 = torch.zeros_like(self.mean)
            self.min_val = x.min(dim=0)[0]
            self.max_val = x.max(dim=0)[0]
            self.n = batch_size
        else:
            # Welford's online algorithm
            new_mean = self.mean + (x.sum(dim=0) - batch_size * self.mean) / (self.n + batch_size)
            
            # Update M2
            for i in range(batch_size):
                delta = x[i] - self.mean
                delta2 = x[i] - new_mean
                self.m2 += delta * delta2
                
            self.mean = new_mean
            self.n += batch_size
            
            # Update min/max
            self.min_val = torch.min(self.min_val, x.min(dim=0)[0])
            self.max_val = torch.max(self.max_val, x.max(dim=0)[0])
            
    def get_statistics(self) -> Dict[str, torch.Tensor]:
        """Get final statistics."""
        variance = self.m2 / (self.n - 1) if self.n > 1 else torch.zeros_like(self.mean)
        std = torch.sqrt(variance)
        
        return {
            'mean': self.mean,
            'std': std,
            'min': self.min_val,
            'max': self.max_val,
            'n_samples': self.n
        }


class AdaptiveMCDropout:
    """
    Adaptive MC Dropout that adjusts number of samples based on uncertainty.
    
    Saves computation when uncertainty is low.
    """
    
    def __init__(
        self,
        min_samples: int = 10,
        max_samples: int = 50,
        uncertainty_threshold: float = 0.1
    ):
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.uncertainty_threshold = uncertainty_threshold
        
    def adaptive_sampling(
        self,
        model: nn.Module,
        input_state: torch.Tensor,
        quick_check_samples: int = 10
    ) -> Dict[str, Any]:
        """
        Perform adaptive sampling based on initial uncertainty estimate.
        
        Args:
            model: Neural network model
            input_state: Input tensor
            quick_check_samples: Initial samples for uncertainty estimate
            
        Returns:
            Dictionary with samples and metadata
        """
        # Quick uncertainty check
        model.train()
        
        quick_samples = []
        with torch.no_grad():
            for _ in range(quick_check_samples):
                output = model(input_state)
                if isinstance(output, dict):
                    probs = F.softmax(output['action_logits'], dim=-1)
                else:
                    probs = F.softmax(output, dim=-1)
                quick_samples.append(probs)
                
        quick_samples = torch.stack(quick_samples)
        
        # Estimate initial uncertainty
        initial_std = quick_samples.std(dim=0).mean().item()
        
        # Determine number of samples needed
        if initial_std < self.uncertainty_threshold:
            n_samples = self.min_samples
            confidence = 'high'
        elif initial_std < self.uncertainty_threshold * 2:
            n_samples = (self.min_samples + self.max_samples) // 2
            confidence = 'medium'
        else:
            n_samples = self.max_samples
            confidence = 'low'
            
        # Perform full sampling if needed
        if n_samples > quick_check_samples:
            additional_samples = []
            
            with torch.no_grad():
                for _ in range(n_samples - quick_check_samples):
                    output = model(input_state)
                    if isinstance(output, dict):
                        probs = F.softmax(output['action_logits'], dim=-1)
                    else:
                        probs = F.softmax(output, dim=-1)
                    additional_samples.append(probs)
                    
            all_samples = torch.cat([
                quick_samples,
                torch.stack(additional_samples)
            ], dim=0)
        else:
            all_samples = quick_samples[:n_samples]
            
        model.eval()
        
        return {
            'samples': all_samples,
            'n_samples': n_samples,
            'initial_uncertainty': initial_std,
            'confidence_level': confidence,
            'computation_saved': 1.0 - (n_samples / self.max_samples)
        }


class GPUMemoryOptimizer:
    """
    Manages GPU memory for efficient MC Dropout execution.
    
    Prevents OOM errors and optimizes memory allocation.
    """
    
    def __init__(self, device: torch.device, reserved_memory_mb: int = 512):
        self.device = device
        self.reserved_memory = reserved_memory_mb * 1024 * 1024  # Convert to bytes
        
    def get_optimal_batch_size(
        self,
        model: nn.Module,
        input_shape: tuple,
        max_samples: int = 50
    ) -> int:
        """
        Determine optimal batch size based on available GPU memory.
        
        Args:
            model: Neural network model
            input_shape: Shape of single input
            max_samples: Maximum number of MC samples
            
        Returns:
            Optimal batch size
        """
        if self.device.type != 'cuda':
            return max_samples  # No memory constraints on CPU
            
        # Get available memory
        torch.cuda.synchronize()
        available_memory = torch.cuda.mem_get_info(self.device.index)[0] - self.reserved_memory
        
        # Estimate memory per sample
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape[1:], device=self.device)
        
        # Measure memory for single forward pass
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(dummy_input)
            
        memory_per_sample = torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated()
        
        # Calculate safe batch size (with 20% safety margin)
        safe_batch_size = int((available_memory * 0.8) / memory_per_sample)
        
        # Return constrained batch size
        return min(safe_batch_size, max_samples)
        
    def optimize_model_for_inference(self, model: nn.Module):
        """
        Optimize model for MC Dropout inference.
        
        Args:
            model: Neural network model
        """
        # Enable cudnn benchmarking for consistent input sizes
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            
        # Use automatic mixed precision if available
        if hasattr(torch.cuda, 'amp'):
            model = model.to(dtype=torch.float16)
            
        # Enable gradient checkpointing if model supports it
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            
        return model


class ParallelMCProcessor:
    """
    Processes MC Dropout samples in parallel across multiple GPUs.
    
    For high-throughput production environments.
    """
    
    def __init__(self, devices: List[torch.device]):
        self.devices = devices
        self.n_devices = len(devices)
        
    def parallel_forward(
        self,
        models: List[nn.Module],
        input_state: torch.Tensor,
        n_samples: int
    ) -> torch.Tensor:
        """
        Run MC samples in parallel across devices.
        
        Args:
            models: List of model replicas (one per device)
            input_state: Input tensor
            n_samples: Total number of MC samples
            
        Returns:
            Combined predictions from all devices
        """
        # Distribute samples across devices
        samples_per_device = n_samples // self.n_devices
        remainder = n_samples % self.n_devices
        
        device_outputs = []
        
        # Process on each device
        for i, (device, model) in enumerate(zip(self.devices, models)):
            # Determine samples for this device
            device_samples = samples_per_device
            if i < remainder:
                device_samples += 1
                
            # Move input to device
            device_input = input_state.to(device)
            
            # Batch forward on device
            with torch.no_grad():
                # Replicate input
                batch_input = device_input.repeat(device_samples, 1)
                
                # Forward pass
                outputs = model(batch_input)
                
                # Handle output format
                if isinstance(outputs, dict):
                    outputs = outputs['action_logits']
                elif isinstance(outputs, tuple):
                    outputs = outputs[0]
                    
                # Move back to first device
                device_outputs.append(outputs.to(self.devices[0]))
                
        # Combine all outputs
        return torch.cat(device_outputs, dim=0)


class MCDropoutOptimizationConfig:
    """Configuration for MC Dropout optimizations."""
    
    def __init__(self):
        self.batch_optimization = True
        self.streaming_stats = True
        self.adaptive_sampling = True
        self.gpu_memory_optimization = True
        self.multi_gpu = False
        
        # Batching parameters
        self.max_batch_size = 50
        self.min_batch_size = 10
        
        # Adaptive sampling parameters
        self.adaptive_min_samples = 10
        self.adaptive_max_samples = 50
        self.uncertainty_threshold = 0.1
        
        # Memory parameters
        self.reserved_memory_mb = 512
        
    def get_optimizer_suite(self, device: torch.device) -> Dict[str, Any]:
        """Get configured optimization components."""
        optimizers = {}
        
        if self.batch_optimization:
            optimizers['batcher'] = MCDropoutBatcher(
                device=device,
                max_batch_size=self.max_batch_size
            )
            
        if self.streaming_stats:
            optimizers['streaming'] = StreamingStatistics()
            
        if self.adaptive_sampling:
            optimizers['adaptive'] = AdaptiveMCDropout(
                min_samples=self.adaptive_min_samples,
                max_samples=self.adaptive_max_samples,
                uncertainty_threshold=self.uncertainty_threshold
            )
            
        if self.gpu_memory_optimization and device.type == 'cuda':
            optimizers['memory'] = GPUMemoryOptimizer(
                device=device,
                reserved_memory_mb=self.reserved_memory_mb
            )
            
        return optimizers


def benchmark_mc_dropout_performance(
    model: nn.Module,
    input_shape: tuple,
    n_samples: int = 50,
    device: torch.device = torch.device('cpu')
) -> Dict[str, float]:
    """
    Benchmark MC Dropout performance with different optimizations.
    
    Args:
        model: Neural network model
        input_shape: Shape of input tensor
        n_samples: Number of MC samples
        device: Device to run on
        
    Returns:
        Performance metrics
    """
    import time
    
    model = model.to(device)
    input_state = torch.randn(*input_shape, device=device)
    
    metrics = {}
    
    # Baseline: Sequential sampling
    model.train()
    start_time = time.time()
    
    sequential_samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            output = model(input_state)
            if isinstance(output, dict):
                output = output['action_logits']
            sequential_samples.append(output)
            
    sequential_samples = torch.stack(sequential_samples)
    sequential_time = time.time() - start_time
    metrics['sequential_time'] = sequential_time
    
    # Optimized: Batch sampling
    batcher = MCDropoutBatcher(device=device)
    
    start_time = time.time()
    batch_samples = batcher.batch_forward(model, input_state, n_samples)
    batch_time = time.time() - start_time
    metrics['batch_time'] = batch_time
    
    # Calculate speedup
    metrics['speedup'] = sequential_time / batch_time
    
    # Memory usage
    if device.type == 'cuda':
        torch.cuda.synchronize()
        metrics['peak_memory_mb'] = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        
    # Verify outputs are similar
    if sequential_samples.shape == batch_samples.shape:
        # Both should produce similar distributions
        seq_mean = sequential_samples.mean(dim=0)
        batch_mean = batch_samples.squeeze(1).mean(dim=0)
        metrics['output_difference'] = torch.abs(seq_mean - batch_mean).mean().item()
        
    logger.info(f"MC Dropout Benchmark Results: {metrics}")
    
    return metrics