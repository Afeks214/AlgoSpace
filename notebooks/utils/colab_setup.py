"""Google Colab Setup and Management Utilities.

This module provides utilities for setting up and managing Google Colab
environments for MARL training.
"""

import os
import sys
import subprocess
import torch
import psutil
import GPUtil
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import json


logger = logging.getLogger(__name__)


class ColabSetup:
    """Manages Google Colab environment setup and configuration."""
    
    def __init__(self, project_name: str = "AlgoSpace"):
        """Initialize Colab setup manager.
        
        Args:
            project_name: Name of the project
        """
        self.project_name = project_name
        self.is_colab = self._detect_colab()
        self.device = None
        self.gpu_info = {}
        
        if self.is_colab:
            self._setup_environment()
    
    def _detect_colab(self) -> bool:
        """Detect if running in Google Colab."""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def _setup_environment(self):
        """Setup Colab environment."""
        print("🚀 Setting up Google Colab environment...")
        
        # GPU setup
        self._setup_gpu()
        
        # Install dependencies
        self._install_dependencies()
        
        # Setup paths
        self._setup_paths()
        
        # Configure logging
        self._setup_logging()
        
        print("✅ Colab environment setup complete!")
    
    def _setup_gpu(self):
        """Setup and verify GPU availability."""
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            
            # Get GPU info
            gpu = GPUtil.getGPUs()[0]
            self.gpu_info = {
                'name': gpu.name,
                'memory_total': f"{gpu.memoryTotal}MB",
                'memory_free': f"{gpu.memoryFree}MB",
                'memory_used': f"{gpu.memoryUsed}MB",
                'utilization': f"{gpu.load * 100:.1f}%"
            }
            
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(0.9)
            
            print(f"🎮 GPU: {self.gpu_info['name']}")
            print(f"💾 Memory: {self.gpu_info['memory_free']} free / {self.gpu_info['memory_total']} total")
        else:
            self.device = torch.device('cpu')
            print("⚠️ No GPU available, using CPU")
    
    def _install_dependencies(self):
        """Install required dependencies."""
        print("📦 Installing dependencies...")
        
        requirements = [
            "torch>=2.0.0",
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "h5py>=3.0.0",
            "pyyaml>=5.4.0",
            "tensorboard>=2.8.0",
            "wandb>=0.12.0",
            "optuna>=3.0.0",
            "mlflow>=2.0.0",
            "tqdm>=4.62.0",
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
            "scikit-learn>=1.0.0"
        ]
        
        for package in requirements:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
    
    def _setup_paths(self):
        """Setup project paths."""
        # Add project root to path
        project_root = Path(f"/content/drive/MyDrive/{self.project_name}")
        if project_root.exists():
            sys.path.insert(0, str(project_root))
        
        # Create necessary directories
        dirs_to_create = [
            "checkpoints",
            "logs",
            "data",
            "models",
            "results"
        ]
        
        for dir_name in dirs_to_create:
            dir_path = project_root / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Configure logging for Colab."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def mount_drive(self, force_remount: bool = False):
        """Mount Google Drive.
        
        Args:
            force_remount: Force remount even if already mounted
            
        Returns:
            Path to Drive mount point
        """
        if self.is_colab:
            from google.colab import drive
            
            mount_point = '/content/drive'
            
            if force_remount or not os.path.exists(mount_point):
                drive.mount(mount_point, force_remount=force_remount)
                print(f"✅ Google Drive mounted at {mount_point}")
            else:
                print(f"ℹ️ Google Drive already mounted at {mount_point}")
            
            return Path(mount_point)
        else:
            print("⚠️ Not running in Colab, skipping Drive mount")
            return None
    
    def keep_alive(self):
        """Keep Colab session alive by preventing idle timeout."""
        if self.is_colab:
            from IPython.display import Javascript, display
            display(Javascript('''
                function KeepAlive() {
                    const pingInterval = 60000;  // 1 minute
                    setInterval(() => {
                        console.log("Keeping Colab alive...");
                        document.querySelector("#connect")?.click();
                    }, pingInterval);
                }
                KeepAlive();
            '''))
            print("🔄 Keep-alive script activated")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        info = {
            'timestamp': datetime.now().isoformat(),
            'is_colab': self.is_colab,
            'device': str(self.device),
            'gpu': self.gpu_info,
            'cpu': {
                'cores': psutil.cpu_count(),
                'usage': f"{psutil.cpu_percent()}%"
            },
            'memory': {
                'total': f"{psutil.virtual_memory().total / 1e9:.1f}GB",
                'available': f"{psutil.virtual_memory().available / 1e9:.1f}GB",
                'used': f"{psutil.virtual_memory().percent}%"
            },
            'disk': {
                'total': f"{psutil.disk_usage('/').total / 1e9:.1f}GB",
                'free': f"{psutil.disk_usage('/').free / 1e9:.1f}GB",
                'used': f"{psutil.disk_usage('/').percent}%"
            }
        }
        
        return info
    
    def check_gpu_memory(self) -> Dict[str, float]:
        """Check current GPU memory usage."""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1e9,
                'reserved': torch.cuda.memory_reserved() / 1e9,
                'free': (torch.cuda.get_device_properties(0).total_memory - 
                        torch.cuda.memory_allocated()) / 1e9
            }
        return {}
    
    def optimize_memory(self):
        """Optimize memory usage."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        import gc
        gc.collect()
        
        print("🧹 Memory optimized")
    
    def setup_wandb(self, api_key: Optional[str] = None, 
                   project: Optional[str] = None,
                   entity: Optional[str] = None):
        """Setup Weights & Biases for experiment tracking.
        
        Args:
            api_key: W&B API key
            project: Project name
            entity: Entity name
        """
        try:
            import wandb
            
            if api_key:
                wandb.login(key=api_key)
            else:
                # Try to login with stored credentials
                wandb.login()
            
            # Initialize run
            run = wandb.init(
                project=project or f"{self.project_name}-training",
                entity=entity,
                reinit=True,
                settings=wandb.Settings(start_method="thread")
            )
            
            # Log system info
            wandb.config.update(self.get_system_info())
            
            print(f"📊 W&B initialized: {run.url}")
            return run
            
        except Exception as e:
            print(f"⚠️ Failed to setup W&B: {e}")
            return None
    
    def download_from_drive(self, file_id: str, destination: str):
        """Download file from Google Drive using file ID.
        
        Args:
            file_id: Google Drive file ID
            destination: Local destination path
        """
        if self.is_colab:
            from google.colab import auth
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaIoBaseDownload
            import io
            
            auth.authenticate_user()
            drive_service = build('drive', 'v3')
            
            request = drive_service.files().get_media(fileId=file_id)
            
            fh = io.FileIO(destination, 'wb')
            downloader = MediaIoBaseDownload(fh, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    print(f"Download {int(status.progress() * 100)}%")
            
            print(f"✅ Downloaded to {destination}")
    
    def create_training_summary(self, metrics: Dict[str, Any], 
                              save_path: Optional[str] = None) -> str:
        """Create a training summary report.
        
        Args:
            metrics: Training metrics
            save_path: Optional path to save summary
            
        Returns:
            Summary as string
        """
        summary = f"""
# Training Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Environment
- Platform: {'Google Colab' if self.is_colab else 'Local'}
- Device: {self.device}
- GPU: {self.gpu_info.get('name', 'N/A')}

## Training Metrics
"""
        
        for key, value in metrics.items():
            if isinstance(value, float):
                summary += f"- {key}: {value:.4f}\n"
            else:
                summary += f"- {key}: {value}\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(summary)
        
        return summary


class SessionMonitor:
    """Monitor Colab session status and manage long-running training."""
    
    def __init__(self, max_runtime_hours: float = 23.5):
        """Initialize session monitor.
        
        Args:
            max_runtime_hours: Maximum runtime before saving (default 23.5 hours)
        """
        self.start_time = datetime.now()
        self.max_runtime_hours = max_runtime_hours
        self.check_interval = 300  # 5 minutes
        
    def get_runtime_hours(self) -> float:
        """Get current runtime in hours."""
        return (datetime.now() - self.start_time).total_seconds() / 3600
    
    def is_ending_soon(self, buffer_minutes: int = 30) -> bool:
        """Check if session is ending soon.
        
        Args:
            buffer_minutes: Minutes before max runtime to trigger
            
        Returns:
            True if session ending soon
        """
        runtime_hours = self.get_runtime_hours()
        return runtime_hours >= (self.max_runtime_hours - buffer_minutes / 60)
    
    def get_remaining_time(self) -> Dict[str, float]:
        """Get remaining session time."""
        runtime_hours = self.get_runtime_hours()
        remaining_hours = max(0, self.max_runtime_hours - runtime_hours)
        
        return {
            'runtime_hours': runtime_hours,
            'remaining_hours': remaining_hours,
            'remaining_minutes': remaining_hours * 60
        }
    
    def should_checkpoint(self, last_checkpoint_time: datetime,
                         checkpoint_interval_minutes: int = 30) -> bool:
        """Check if should create checkpoint.
        
        Args:
            last_checkpoint_time: Time of last checkpoint
            checkpoint_interval_minutes: Interval between checkpoints
            
        Returns:
            True if should checkpoint
        """
        # Check if ending soon
        if self.is_ending_soon():
            return True
        
        # Check regular interval
        time_since_checkpoint = (datetime.now() - last_checkpoint_time).total_seconds() / 60
        return time_since_checkpoint >= checkpoint_interval_minutes
    
    def display_status(self):
        """Display session status."""
        status = self.get_remaining_time()
        print(f"⏱️ Runtime: {status['runtime_hours']:.1f}h / "
              f"Remaining: {status['remaining_hours']:.1f}h")


# Utility functions for common Colab operations

def setup_colab_training(project_name: str = "AlgoSpace",
                        mount_drive: bool = True,
                        setup_wandb: bool = True,
                        keep_alive: bool = True) -> ColabSetup:
    """Quick setup for Colab training.
    
    Args:
        project_name: Project name
        mount_drive: Whether to mount Google Drive
        setup_wandb: Whether to setup W&B
        keep_alive: Whether to activate keep-alive
        
    Returns:
        Configured ColabSetup instance
    """
    setup = ColabSetup(project_name)
    
    if mount_drive:
        setup.mount_drive()
    
    if setup_wandb:
        setup.setup_wandb()
    
    if keep_alive:
        setup.keep_alive()
    
    # Display system info
    info = setup.get_system_info()
    print("\n📊 System Information:")
    print(json.dumps(info, indent=2))
    
    return setup