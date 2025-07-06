#!/usr/bin/env python3
"""
Quick Status Check for Terminal 1 Monitoring
Provides instant status snapshot for watch commands
"""

import sys
import time
import psutil
from pathlib import Path
from datetime import datetime

def quick_status_check():
    """Quick status check for watch command"""
    project_root = Path("/home/QuantNova/AlgoSpace-4")
    
    print("🔍 ALGOSPACE QUICK STATUS")
    print("=" * 40)
    print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
    
    # System resources
    memory = psutil.virtual_memory()
    cpu = psutil.cpu_percent()
    print(f"💻 CPU: {cpu:5.1f}% | RAM: {memory.percent:5.1f}%")
    
    # Check virtual environment
    venv_active = 'algospace_env' in sys.executable or 'VIRTUAL_ENV' in sys.environ
    print(f"🐍 VEnv: {'✅' if venv_active else '❌'}")
    
    # Check PyTorch
    try:
        import torch
        print(f"🔥 PyTorch: ✅ {torch.__version__}")
    except ImportError:
        print("🔥 PyTorch: ❌ Not available")
    
    # Check current task
    status_file = project_root / ".current_task"
    if status_file.exists():
        task = status_file.read_text().strip()
        print(f"📋 Task: {task}")
    
    # Check for recent errors in logs
    log_dir = project_root / "logs"
    if log_dir.exists():
        log_files = list(log_dir.glob("*.log"))
        if log_files:
            latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
            try:
                with open(latest_log, 'r') as f:
                    lines = f.readlines()
                    recent_errors = [l for l in lines[-10:] if 'error' in l.lower()]
                    print(f"❌ Recent Errors: {len(recent_errors)}")
            except:
                print("❌ Recent Errors: Unable to check")
    
    print("=" * 40)

if __name__ == "__main__":
    quick_status_check()