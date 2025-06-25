"""Checkpoint Manager for Colab Training.

This module handles checkpoint saving, loading, and recovery for robust
training in Google Colab environment.
"""

import torch
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging
import hashlib
import numpy as np
from collections import OrderedDict


logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages training checkpoints with automatic recovery."""
    
    def __init__(self, drive_manager, checkpoint_dir: Optional[str] = None):
        """Initialize checkpoint manager.
        
        Args:
            drive_manager: DriveManager instance
            checkpoint_dir: Optional custom checkpoint directory
        """
        self.drive_manager = drive_manager
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else drive_manager.checkpoint_path
        
        # Checkpoint settings
        self.keep_last_n = 5
        self.keep_best_n = 3
        self.save_interval = 100  # episodes
        self.time_interval = 1800  # 30 minutes
        
        # State tracking
        self.last_save_time = datetime.now()
        self.last_save_episode = 0
        self.best_metric = -float('inf')
        self.checkpoint_history = self._load_history()
        
        logger.info(f"CheckpointManager initialized at {self.checkpoint_dir}")
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load checkpoint history."""
        history_file = self.checkpoint_dir / "checkpoint_history.json"
        
        if history_file.exists():
            with open(history_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_history(self):
        """Save checkpoint history."""
        history_file = self.checkpoint_dir / "checkpoint_history.json"
        
        with open(history_file, 'w') as f:
            json.dump(self.checkpoint_history, f, indent=2)
    
    def should_save(self, episode: int, force: bool = False) -> bool:
        """Check if should save checkpoint.
        
        Args:
            episode: Current episode
            force: Force save regardless of intervals
            
        Returns:
            Whether to save checkpoint
        """
        if force:
            return True
        
        # Check episode interval
        if episode - self.last_save_episode >= self.save_interval:
            return True
        
        # Check time interval
        if (datetime.now() - self.last_save_time).seconds >= self.time_interval:
            return True
        
        return False
    
    def save(self, state: Dict[str, Any], 
            metrics: Dict[str, float],
            is_best: bool = False,
            tag: Optional[str] = None) -> str:
        """Save checkpoint.
        
        Args:
            state: State dictionary containing models, optimizers, etc.
            metrics: Current metrics
            is_best: Whether this is the best model
            tag: Optional tag for the checkpoint
            
        Returns:
            Path where checkpoint was saved
        """
        episode = state.get('episode', 0)
        
        # Create checkpoint
        checkpoint = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'state': state,
            'tag': tag,
            'checksum': self._calculate_checksum(state)
        }
        
        # Generate filename
        if tag:
            filename = f"checkpoint_ep{episode}_{tag}.pt"
        else:
            filename = f"checkpoint_ep{episode}.pt"
        
        # Save using drive manager
        save_path = self.drive_manager.save_checkpoint(
            checkpoint, 
            filename.replace('.pt', ''),
            is_best=is_best
        )
        
        # Update history
        self.checkpoint_history.append({
            'filename': filename,
            'episode': episode,
            'timestamp': checkpoint['timestamp'],
            'metrics': metrics,
            'is_best': is_best,
            'tag': tag,
            'path': save_path
        })
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
        
        # Update state
        self.last_save_time = datetime.now()
        self.last_save_episode = episode
        
        # Save history
        self._save_history()
        
        logger.info(f"Saved checkpoint: {filename} (best={is_best})")
        return save_path
    
    def save_atomic(self, state: Dict[str, Any], 
                   metrics: Dict[str, float]) -> str:
        """Save checkpoint atomically to prevent corruption.
        
        Args:
            state: State dictionary
            metrics: Metrics
            
        Returns:
            Checkpoint path
        """
        # Save to temporary file first
        temp_path = self.checkpoint_dir / f"temp_checkpoint_{datetime.now().timestamp()}.pt"
        
        try:
            # Save checkpoint
            checkpoint = {
                'state': state,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            torch.save(checkpoint, temp_path)
            
            # Verify checkpoint
            test_load = torch.load(temp_path, map_location='cpu')
            assert 'state' in test_load
            
            # Move to final location
            episode = state.get('episode', 0)
            final_path = self.checkpoint_dir / f"checkpoint_ep{episode}.pt"
            temp_path.rename(final_path)
            
            return str(final_path)
            
        except Exception as e:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
            raise e
    
    def load(self, checkpoint_name: Optional[str] = None,
            load_best: bool = False,
            tag: Optional[str] = None) -> Dict[str, Any]:
        """Load checkpoint.
        
        Args:
            checkpoint_name: Specific checkpoint name
            load_best: Load best checkpoint
            tag: Load checkpoint with specific tag
            
        Returns:
            Checkpoint dictionary
        """
        if load_best:
            # Find best checkpoint
            best_checkpoints = [
                cp for cp in self.checkpoint_history 
                if cp.get('is_best', False)
            ]
            
            if not best_checkpoints:
                logger.warning("No best checkpoint found, loading latest")
                return self.load_latest()
            
            # Get most recent best
            checkpoint_info = max(best_checkpoints, key=lambda x: x['timestamp'])
            checkpoint_name = checkpoint_info['filename'].replace('.pt', '')
            
        elif tag:
            # Find checkpoint with tag
            tagged_checkpoints = [
                cp for cp in self.checkpoint_history
                if cp.get('tag') == tag
            ]
            
            if not tagged_checkpoints:
                raise ValueError(f"No checkpoint found with tag: {tag}")
            
            checkpoint_info = max(tagged_checkpoints, key=lambda x: x['timestamp'])
            checkpoint_name = checkpoint_info['filename'].replace('.pt', '')
            
        elif checkpoint_name is None:
            # Load latest
            return self.load_latest()
        
        # Load checkpoint
        checkpoint = self.drive_manager.load_checkpoint(checkpoint_name)
        
        # Verify checksum
        if 'checksum' in checkpoint:
            calculated_checksum = self._calculate_checksum(checkpoint['state'])
            if calculated_checksum != checkpoint['checksum']:
                logger.warning("Checkpoint checksum mismatch!")
        
        return checkpoint
    
    def load_latest(self) -> Dict[str, Any]:
        """Load latest checkpoint."""
        if not self.checkpoint_history:
            raise ValueError("No checkpoints found")
        
        # Get latest checkpoint
        latest = max(self.checkpoint_history, key=lambda x: x['timestamp'])
        checkpoint_name = latest['filename'].replace('.pt', '')
        
        return self.drive_manager.load_checkpoint(checkpoint_name)
    
    def get_resume_info(self) -> Dict[str, Any]:
        """Get information for resuming training.
        
        Returns:
            Resume information including episode, metrics, etc.
        """
        try:
            checkpoint = self.load_latest()
            
            resume_info = {
                'episode': checkpoint['state'].get('episode', 0),
                'timestamp': checkpoint.get('timestamp'),
                'metrics': checkpoint.get('metrics', {}),
                'available': True
            }
            
            # Calculate time since last save
            if 'timestamp' in checkpoint:
                last_save = datetime.fromisoformat(checkpoint['timestamp'])
                time_since = datetime.now() - last_save
                resume_info['hours_since_save'] = time_since.total_seconds() / 3600
            
            return resume_info
            
        except Exception as e:
            logger.warning(f"No checkpoint to resume from: {e}")
            return {'available': False, 'episode': 0}
    
    def create_recovery_point(self, state: Dict[str, Any], 
                            reason: str = "manual") -> str:
        """Create a recovery checkpoint.
        
        Args:
            state: Current state
            reason: Reason for recovery point
            
        Returns:
            Recovery checkpoint path
        """
        recovery_dir = self.checkpoint_dir / "recovery"
        recovery_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"recovery_{timestamp}_{reason}.pt"
        
        checkpoint = {
            'state': state,
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'episode': state.get('episode', 0)
        }
        
        path = recovery_dir / filename
        torch.save(checkpoint, path)
        
        logger.info(f"Created recovery point: {filename}")
        return str(path)
    
    def get_checkpoint_info(self) -> List[Dict[str, Any]]:
        """Get information about all checkpoints.
        
        Returns:
            List of checkpoint information
        """
        info = []
        
        for cp in self.checkpoint_history:
            cp_info = {
                'filename': cp['filename'],
                'episode': cp['episode'],
                'timestamp': cp['timestamp'],
                'metrics': cp.get('metrics', {}),
                'is_best': cp.get('is_best', False),
                'tag': cp.get('tag')
            }
            
            # Check if file exists
            path = Path(cp.get('path', ''))
            cp_info['exists'] = path.exists() if path else False
            
            info.append(cp_info)
        
        return sorted(info, key=lambda x: x['timestamp'], reverse=True)
    
    def _cleanup_checkpoints(self):
        """Clean up old checkpoints to save space."""
        # Separate regular and best checkpoints
        regular_checkpoints = [
            cp for cp in self.checkpoint_history
            if not cp.get('is_best', False)
        ]
        
        best_checkpoints = [
            cp for cp in self.checkpoint_history
            if cp.get('is_best', False)
        ]
        
        # Sort by timestamp
        regular_checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)
        best_checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Remove old regular checkpoints
        for cp in regular_checkpoints[self.keep_last_n:]:
            path = Path(cp.get('path', ''))
            if path.exists():
                path.unlink()
                logger.info(f"Removed old checkpoint: {cp['filename']}")
            self.checkpoint_history.remove(cp)
        
        # Remove old best checkpoints
        for cp in best_checkpoints[self.keep_best_n:]:
            path = Path(cp.get('path', ''))
            if path.exists():
                path.unlink()
            self.checkpoint_history.remove(cp)
    
    def _calculate_checksum(self, state: Dict[str, Any]) -> str:
        """Calculate checksum for state verification.
        
        Args:
            state: State dictionary
            
        Returns:
            Checksum string
        """
        # Create a string representation of model states
        state_str = ""
        
        if 'models' in state:
            for name, model_state in sorted(state['models'].items()):
                # Use first few parameters for checksum
                for key, tensor in list(model_state.items())[:5]:
                    if isinstance(tensor, torch.Tensor):
                        state_str += f"{key}:{tensor.sum().item():.6f}"
        
        # Calculate hash
        return hashlib.md5(state_str.encode()).hexdigest()
    
    def compare_checkpoints(self, checkpoint1: str, checkpoint2: str) -> Dict[str, Any]:
        """Compare two checkpoints.
        
        Args:
            checkpoint1: First checkpoint name
            checkpoint2: Second checkpoint name
            
        Returns:
            Comparison results
        """
        cp1 = self.load(checkpoint1)
        cp2 = self.load(checkpoint2)
        
        comparison = {
            'checkpoint1': checkpoint1,
            'checkpoint2': checkpoint2,
            'metrics_diff': {},
            'episode_diff': cp2['state']['episode'] - cp1['state']['episode']
        }
        
        # Compare metrics
        metrics1 = cp1.get('metrics', {})
        metrics2 = cp2.get('metrics', {})
        
        all_metrics = set(metrics1.keys()) | set(metrics2.keys())
        
        for metric in all_metrics:
            val1 = metrics1.get(metric, 0)
            val2 = metrics2.get(metric, 0)
            comparison['metrics_diff'][metric] = {
                'value1': val1,
                'value2': val2,
                'diff': val2 - val1,
                'pct_change': ((val2 - val1) / val1 * 100) if val1 != 0 else 0
            }
        
        return comparison


class CheckpointScheduler:
    """Schedules checkpoint saves based on various criteria."""
    
    def __init__(self, checkpoint_manager: CheckpointManager):
        """Initialize checkpoint scheduler.
        
        Args:
            checkpoint_manager: CheckpointManager instance
        """
        self.checkpoint_manager = checkpoint_manager
        self.schedules = {
            'episode': {'interval': 100, 'last': 0},
            'time': {'interval': 1800, 'last': datetime.now()},  # 30 min
            'metric': {'threshold': 0.01, 'last_value': None}
        }
    
    def should_checkpoint(self, episode: int, metrics: Dict[str, float]) -> Tuple[bool, str]:
        """Check if should create checkpoint.
        
        Args:
            episode: Current episode
            metrics: Current metrics
            
        Returns:
            Tuple of (should_checkpoint, reason)
        """
        # Episode-based
        if episode - self.schedules['episode']['last'] >= self.schedules['episode']['interval']:
            return True, 'episode_interval'
        
        # Time-based
        time_elapsed = (datetime.now() - self.schedules['time']['last']).seconds
        if time_elapsed >= self.schedules['time']['interval']:
            return True, 'time_interval'
        
        # Metric improvement
        key_metric = metrics.get('sharpe_ratio', 0)
        if self.schedules['metric']['last_value'] is not None:
            improvement = key_metric - self.schedules['metric']['last_value']
            if improvement > self.schedules['metric']['threshold']:
                return True, 'metric_improvement'
        
        return False, ''
    
    def update_schedule(self, episode: int, metrics: Dict[str, float], saved: bool):
        """Update schedule after checkpoint decision.
        
        Args:
            episode: Current episode
            metrics: Current metrics  
            saved: Whether checkpoint was saved
        """
        if saved:
            self.schedules['episode']['last'] = episode
            self.schedules['time']['last'] = datetime.now()
            self.schedules['metric']['last_value'] = metrics.get('sharpe_ratio', 0)