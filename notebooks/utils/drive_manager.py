"""Google Drive Manager for Colab Training.

This module handles all Google Drive operations for efficient data management
and checkpoint handling during training.
"""

import os
import json
import pickle
import shutil
import zipfile
import h5py
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging
from tqdm import tqdm


logger = logging.getLogger(__name__)


class DriveManager:
    """Manages data and model storage on Google Drive."""
    
    def __init__(self, drive_base_path: str = "/content/drive/MyDrive/AlgoSpace"):
        """Initialize Drive manager.
        
        Args:
            drive_base_path: Base path in Google Drive
        """
        self.base_path = Path(drive_base_path)
        self.data_path = self.base_path / "data"
        self.checkpoint_path = self.base_path / "checkpoints"
        self.model_path = self.base_path / "models"
        self.results_path = self.base_path / "results"
        
        # Create directories
        self._create_directories()
        
        # Track file versions
        self.version_tracker = self._load_version_tracker()
        
        logger.info(f"DriveManager initialized at {self.base_path}")
    
    def _create_directories(self):
        """Create necessary directories."""
        dirs = [
            self.data_path,
            self.checkpoint_path,
            self.model_path,
            self.results_path,
            self.data_path / "raw",
            self.data_path / "processed",
            self.checkpoint_path / "training",
            self.checkpoint_path / "best",
            self.model_path / "production",
            self.results_path / "evaluation",
            self.results_path / "plots"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _load_version_tracker(self) -> Dict[str, Any]:
        """Load or create version tracker."""
        tracker_path = self.base_path / "version_tracker.json"
        
        if tracker_path.exists():
            with open(tracker_path, 'r') as f:
                return json.load(f)
        else:
            return {
                'checkpoints': {},
                'models': {},
                'data': {}
            }
    
    def _save_version_tracker(self):
        """Save version tracker."""
        tracker_path = self.base_path / "version_tracker.json"
        with open(tracker_path, 'w') as f:
            json.dump(self.version_tracker, f, indent=2)
    
    def upload_data(self, local_path: Union[str, Path], 
                   data_name: str,
                   compress: bool = True) -> str:
        """Upload data to Drive with optional compression.
        
        Args:
            local_path: Local file or directory path
            data_name: Name for the data
            compress: Whether to compress before upload
            
        Returns:
            Drive path where data was saved
        """
        local_path = Path(local_path)
        
        if compress and local_path.is_dir():
            # Compress directory
            print(f"📦 Compressing {data_name}...")
            zip_path = Path(f"/tmp/{data_name}.zip")
            shutil.make_archive(str(zip_path.with_suffix('')), 'zip', local_path)
            local_path = zip_path
        
        # Determine destination
        if local_path.suffix in ['.csv', '.parquet', '.h5', '.hdf5']:
            dest_path = self.data_path / "raw" / local_path.name
        elif local_path.suffix == '.zip':
            dest_path = self.data_path / "compressed" / local_path.name
        else:
            dest_path = self.data_path / "processed" / local_path.name
        
        # Copy file
        print(f"📤 Uploading to {dest_path}...")
        shutil.copy2(local_path, dest_path)
        
        # Update version tracker
        self.version_tracker['data'][data_name] = {
            'path': str(dest_path),
            'timestamp': datetime.now().isoformat(),
            'size': os.path.getsize(dest_path),
            'compressed': compress
        }
        self._save_version_tracker()
        
        print(f"✅ Data uploaded: {dest_path}")
        return str(dest_path)
    
    def download_data(self, data_name: str, 
                     local_path: Optional[str] = None,
                     decompress: bool = True) -> str:
        """Download data from Drive.
        
        Args:
            data_name: Name of the data
            local_path: Local destination (optional)
            decompress: Whether to decompress if compressed
            
        Returns:
            Local path where data was saved
        """
        if data_name not in self.version_tracker['data']:
            raise ValueError(f"Data '{data_name}' not found in Drive")
        
        data_info = self.version_tracker['data'][data_name]
        drive_path = Path(data_info['path'])
        
        if not drive_path.exists():
            raise FileNotFoundError(f"Data file not found: {drive_path}")
        
        # Determine local path
        if local_path is None:
            local_path = Path("/tmp") / drive_path.name
        else:
            local_path = Path(local_path)
        
        # Copy file
        print(f"📥 Downloading {data_name}...")
        shutil.copy2(drive_path, local_path)
        
        # Decompress if needed
        if decompress and data_info.get('compressed', False) and local_path.suffix == '.zip':
            print(f"📦 Decompressing...")
            extract_path = local_path.with_suffix('')
            shutil.unpack_archive(local_path, extract_path)
            local_path = extract_path
        
        print(f"✅ Data downloaded: {local_path}")
        return str(local_path)
    
    def save_checkpoint(self, checkpoint: Dict[str, Any],
                       name: str,
                       is_best: bool = False) -> str:
        """Save training checkpoint.
        
        Args:
            checkpoint: Checkpoint dictionary
            name: Checkpoint name
            is_best: Whether this is the best model
            
        Returns:
            Path where checkpoint was saved
        """
        # Add metadata
        checkpoint['metadata'] = {
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'is_best': is_best
        }
        
        # Determine path
        if is_best:
            checkpoint_path = self.checkpoint_path / "best" / f"{name}_best.pt"
        else:
            checkpoint_path = self.checkpoint_path / "training" / f"{name}.pt"
        
        # Save checkpoint
        print(f"💾 Saving checkpoint: {name}")
        torch.save(checkpoint, checkpoint_path)
        
        # Update version tracker
        self.version_tracker['checkpoints'][name] = {
            'path': str(checkpoint_path),
            'timestamp': checkpoint['metadata']['timestamp'],
            'episode': checkpoint.get('episode', 0),
            'metrics': checkpoint.get('metrics', {}),
            'is_best': is_best
        }
        self._save_version_tracker()
        
        # Clean old checkpoints
        self._cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, name: Optional[str] = None,
                       load_best: bool = False) -> Dict[str, Any]:
        """Load checkpoint from Drive.
        
        Args:
            name: Checkpoint name (if None, loads latest)
            load_best: Whether to load best checkpoint
            
        Returns:
            Checkpoint dictionary
        """
        if load_best:
            # Find best checkpoint
            best_checkpoints = {
                k: v for k, v in self.version_tracker['checkpoints'].items()
                if v.get('is_best', False)
            }
            if not best_checkpoints:
                raise ValueError("No best checkpoint found")
            
            # Get most recent best
            name = max(best_checkpoints.items(), 
                      key=lambda x: x[1]['timestamp'])[0]
        
        elif name is None:
            # Load latest checkpoint
            if not self.version_tracker['checkpoints']:
                raise ValueError("No checkpoints found")
            
            name = max(self.version_tracker['checkpoints'].items(),
                      key=lambda x: x[1]['timestamp'])[0]
        
        # Load checkpoint
        checkpoint_info = self.version_tracker['checkpoints'][name]
        checkpoint_path = Path(checkpoint_info['path'])
        
        print(f"📂 Loading checkpoint: {name}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        return checkpoint
    
    def save_model(self, models: Dict[str, torch.nn.Module],
                  name: str,
                  configs: Optional[Dict[str, Any]] = None,
                  metrics: Optional[Dict[str, float]] = None,
                  production: bool = False) -> str:
        """Save trained models.
        
        Args:
            models: Dictionary of models
            name: Model name
            configs: Model configurations
            metrics: Performance metrics
            production: Whether this is for production
            
        Returns:
            Path where models were saved
        """
        # Create model bundle
        model_bundle = {
            'models': {k: v.state_dict() for k, v in models.items()},
            'configs': configs or {},
            'metrics': metrics or {},
            'metadata': {
                'name': name,
                'timestamp': datetime.now().isoformat(),
                'production': production
            }
        }
        
        # Determine path
        if production:
            model_path = self.model_path / "production" / f"{name}_prod.pt"
        else:
            model_path = self.model_path / f"{name}.pt"
        
        # Save models
        print(f"💾 Saving models: {name}")
        torch.save(model_bundle, model_path)
        
        # Save TorchScript versions for production
        if production:
            script_path = self.model_path / "production" / f"{name}_scripted"
            script_path.mkdir(exist_ok=True)
            
            for agent_name, model in models.items():
                model.eval()
                scripted = torch.jit.script(model)
                scripted.save(script_path / f"{agent_name}.pt")
        
        # Update version tracker
        self.version_tracker['models'][name] = {
            'path': str(model_path),
            'timestamp': model_bundle['metadata']['timestamp'],
            'metrics': metrics or {},
            'production': production
        }
        self._save_version_tracker()
        
        return str(model_path)
    
    def load_model(self, name: str) -> Dict[str, Any]:
        """Load model from Drive.
        
        Args:
            name: Model name
            
        Returns:
            Model bundle dictionary
        """
        if name not in self.version_tracker['models']:
            raise ValueError(f"Model '{name}' not found")
        
        model_info = self.version_tracker['models'][name]
        model_path = Path(model_info['path'])
        
        print(f"📂 Loading model: {name}")
        model_bundle = torch.load(model_path, map_location='cpu')
        
        return model_bundle
    
    def save_results(self, results: Dict[str, Any],
                    name: str,
                    plots: Optional[Dict[str, Any]] = None):
        """Save evaluation results and plots.
        
        Args:
            results: Results dictionary
            name: Results name
            plots: Optional plots to save
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save results JSON
        results_file = self.results_path / "evaluation" / f"{name}_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save plots
        if plots:
            plot_dir = self.results_path / "plots" / f"{name}_{timestamp}"
            plot_dir.mkdir(exist_ok=True)
            
            for plot_name, fig in plots.items():
                if hasattr(fig, 'savefig'):
                    fig.savefig(plot_dir / f"{plot_name}.png", 
                               dpi=300, bbox_inches='tight')
        
        print(f"✅ Results saved: {name}")
    
    def list_available(self, category: str = 'all') -> Dict[str, List[str]]:
        """List available files in Drive.
        
        Args:
            category: Category to list ('all', 'data', 'checkpoints', 'models')
            
        Returns:
            Dictionary of available files
        """
        available = {}
        
        if category in ['all', 'data']:
            available['data'] = list(self.version_tracker['data'].keys())
        
        if category in ['all', 'checkpoints']:
            available['checkpoints'] = list(self.version_tracker['checkpoints'].keys())
        
        if category in ['all', 'models']:
            available['models'] = list(self.version_tracker['models'].keys())
        
        return available
    
    def get_info(self, name: str, category: str) -> Dict[str, Any]:
        """Get information about a stored item.
        
        Args:
            name: Item name
            category: Category ('data', 'checkpoints', 'models')
            
        Returns:
            Item information
        """
        if category not in self.version_tracker:
            raise ValueError(f"Invalid category: {category}")
        
        if name not in self.version_tracker[category]:
            raise ValueError(f"Item '{name}' not found in {category}")
        
        return self.version_tracker[category][name]
    
    def _cleanup_old_checkpoints(self, keep_recent: int = 5, keep_best: int = 3):
        """Clean up old checkpoints to save space.
        
        Args:
            keep_recent: Number of recent checkpoints to keep
            keep_best: Number of best checkpoints to keep
        """
        # Get non-best checkpoints
        regular_checkpoints = {
            k: v for k, v in self.version_tracker['checkpoints'].items()
            if not v.get('is_best', False)
        }
        
        # Sort by timestamp
        sorted_checkpoints = sorted(
            regular_checkpoints.items(),
            key=lambda x: x[1]['timestamp'],
            reverse=True
        )
        
        # Remove old checkpoints
        for name, info in sorted_checkpoints[keep_recent:]:
            checkpoint_path = Path(info['path'])
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                print(f"🗑️ Removed old checkpoint: {name}")
            del self.version_tracker['checkpoints'][name]
        
        # Clean best checkpoints
        best_checkpoints = {
            k: v for k, v in self.version_tracker['checkpoints'].items()
            if v.get('is_best', False)
        }
        
        sorted_best = sorted(
            best_checkpoints.items(),
            key=lambda x: x[1]['timestamp'],
            reverse=True
        )
        
        for name, info in sorted_best[keep_best:]:
            checkpoint_path = Path(info['path'])
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            del self.version_tracker['checkpoints'][name]
        
        self._save_version_tracker()
    
    def create_training_package(self, output_name: str = "training_package") -> str:
        """Create a complete training package for download.
        
        Args:
            output_name: Name for the package
            
        Returns:
            Path to created package
        """
        package_dir = self.base_path / "packages" / output_name
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy best models
        print("📦 Creating training package...")
        
        # Models
        models_dir = package_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        for name, info in self.version_tracker['models'].items():
            if info.get('production', False):
                src = Path(info['path'])
                dst = models_dir / src.name
                shutil.copy2(src, dst)
        
        # Best checkpoints
        checkpoints_dir = package_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)
        
        for name, info in self.version_tracker['checkpoints'].items():
            if info.get('is_best', False):
                src = Path(info['path'])
                dst = checkpoints_dir / src.name
                shutil.copy2(src, dst)
        
        # Results
        results_dir = package_dir / "results"
        if self.results_path.exists():
            shutil.copytree(self.results_path, results_dir, dirs_exist_ok=True)
        
        # Create info file
        info_file = package_dir / "package_info.json"
        package_info = {
            'created': datetime.now().isoformat(),
            'contents': {
                'models': os.listdir(models_dir) if models_dir.exists() else [],
                'checkpoints': os.listdir(checkpoints_dir) if checkpoints_dir.exists() else [],
                'results': os.listdir(results_dir) if results_dir.exists() else []
            }
        }
        
        with open(info_file, 'w') as f:
            json.dump(package_info, f, indent=2)
        
        # Create zip
        zip_path = self.base_path / f"{output_name}.zip"
        shutil.make_archive(str(zip_path.with_suffix('')), 'zip', package_dir)
        
        print(f"✅ Training package created: {zip_path}")
        return str(zip_path)


class DataStreamer:
    """Efficient data streaming from Drive for training."""
    
    def __init__(self, data_path: str, batch_size: int = 32,
                 cache_size: int = 1000):
        """Initialize data streamer.
        
        Args:
            data_path: Path to HDF5 data file
            batch_size: Batch size
            cache_size: Number of samples to cache
        """
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.cache_size = cache_size
        
        # Open HDF5 file
        self.h5_file = h5py.File(self.data_path, 'r')
        
        # Get dataset info
        self.datasets = list(self.h5_file.keys())
        self.n_samples = len(self.h5_file[self.datasets[0]])
        
        # Initialize cache
        self.cache = {}
        self.cache_indices = []
    
    def __len__(self) -> int:
        """Get number of batches."""
        return (self.n_samples + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        """Iterate over batches."""
        indices = np.arange(self.n_samples)
        np.random.shuffle(indices)
        
        for start_idx in range(0, self.n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            yield self._get_batch(batch_indices)
    
    def _get_batch(self, indices: np.ndarray) -> Dict[str, torch.Tensor]:
        """Get batch of data.
        
        Args:
            indices: Batch indices
            
        Returns:
            Batch dictionary
        """
        batch = {}
        
        for dataset_name in self.datasets:
            # Check cache
            cached_data = []
            uncached_indices = []
            
            for idx in indices:
                if idx in self.cache_indices:
                    cached_data.append(self.cache[dataset_name][self.cache_indices.index(idx)])
                else:
                    uncached_indices.append(idx)
            
            # Load uncached data
            if uncached_indices:
                new_data = self.h5_file[dataset_name][uncached_indices]
                
                # Update cache
                self._update_cache(dataset_name, uncached_indices, new_data)
                
                # Combine cached and new data
                all_data = cached_data + list(new_data)
            else:
                all_data = cached_data
            
            # Convert to tensor
            batch[dataset_name] = torch.tensor(np.array(all_data), dtype=torch.float32)
        
        return batch
    
    def _update_cache(self, dataset_name: str, indices: List[int], data: np.ndarray):
        """Update cache with new data.
        
        Args:
            dataset_name: Dataset name
            indices: Data indices
            data: Data to cache
        """
        if dataset_name not in self.cache:
            self.cache[dataset_name] = []
        
        # Add to cache
        for idx, sample in zip(indices, data):
            if len(self.cache_indices) >= self.cache_size:
                # Remove oldest
                old_idx = self.cache_indices.pop(0)
                for ds in self.cache:
                    self.cache[ds].pop(0)
            
            self.cache_indices.append(idx)
            self.cache[dataset_name].append(sample)
    
    def close(self):
        """Close HDF5 file."""
        self.h5_file.close()