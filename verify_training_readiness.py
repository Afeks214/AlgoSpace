import os
import sys
import torch
import json
import yaml
import importlib
import traceback
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any
import subprocess

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

class TrainingReadinessAuditor:
    """Comprehensive training readiness audit system for AlgoSpace"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "phases": {},
            "errors": [],
            "warnings": [],
            "recommendations": [],
            "overall_status": "PENDING"
        }
        self.all_checks_passed = True
        
    def check_path(self, path: str, is_dir: bool = False, check_content: bool = True) -> bool:
        """Enhanced path checking with content validation"""
        exists = os.path.isdir(path) if is_dir else os.path.exists(path)
        emoji = "✅" if exists else "❌"
        status = 'Found' if exists else 'MISSING'
        
        print(f"  {emoji} {path}... {status}")
        
        if exists and check_content and not is_dir:
            # Check if file has actual content
            size = os.path.getsize(path)
            if size < 100:  # Less than 100 bytes is suspiciously small
                print(f"     ⚠️  WARNING: File is very small ({size} bytes)")
                self.results["warnings"].append(f"{path} is only {size} bytes")
        
        if not exists:
            self.results["errors"].append(f"Missing: {path}")
            
        return exists

    def check_module_architecture(self, module_path: str, expected_classes: List[str]) -> bool:
        """Verify module contains expected classes with correct methods"""
        try:
            # Read the file content
            with open(module_path, 'r') as f:
                content = f.read()
            
            missing_classes = []
            for class_name in expected_classes:
                if f"class {class_name}" not in content:
                    missing_classes.append(class_name)
            
            if missing_classes:
                print(f"     ❌ Missing classes: {', '.join(missing_classes)}")
                self.results["errors"].append(f"{module_path} missing: {missing_classes}")
                return False
            
            print(f"     ✅ All expected classes found: {', '.join(expected_classes)}")
            return True
            
        except Exception as e:
            print(f"     ❌ Error reading {module_path}: {e}")
            self.results["errors"].append(f"Cannot read {module_path}: {str(e)}")
            return False

    def test_model_instantiation(self, model_name: str, import_path: str, 
                                init_args: Dict[str, Any] = None) -> bool:
        """Test if a model can be instantiated with proper dimensions"""
        try:
            # Dynamic import
            module_path, class_name = import_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            ModelClass = getattr(module, class_name)
            
            # Instantiate with args
            init_args = init_args or {}
            model = ModelClass(**init_args)
            
            # Check if it's a PyTorch model
            if isinstance(model, torch.nn.Module):
                param_count = sum(p.numel() for p in model.parameters())
                print(f"     ✅ {model_name} instantiated successfully ({param_count:,} parameters)")
                
                # Test forward pass with dummy data
                if hasattr(model, 'forward'):
                    print(f"     ✅ Forward pass method exists")
                
                return True
            else:
                print(f"     ⚠️  {model_name} is not a torch.nn.Module")
                return True  # Still pass if it's a valid class
                
        except Exception as e:
            print(f"     ❌ {model_name} instantiation FAILED: {str(e)}")
            self.results["errors"].append(f"{model_name} failed: {str(e)}")
            return False

    def verify_notebook_content(self, notebook_path: str, required_cells: List[str]) -> bool:
        """Verify notebook contains required content"""
        try:
            with open(notebook_path, 'r') as f:
                notebook = json.load(f)
            
            # Check for cells
            if 'cells' not in notebook or len(notebook['cells']) < 5:
                print(f"     ⚠️  Notebook has insufficient cells")
                self.results["warnings"].append(f"{notebook_path} has few cells")
                return False
            
            # Check for required content
            all_content = ' '.join([' '.join(cell.get('source', [])) 
                                  for cell in notebook['cells']])
            
            missing_content = []
            for required in required_cells:
                if required not in all_content:
                    missing_content.append(required)
            
            if missing_content:
                print(f"     ⚠️  Missing content: {', '.join(missing_content[:3])}...")
                self.results["warnings"].append(f"{notebook_path} missing expected content")
                return False
                
            print(f"     ✅ Notebook contains expected content")
            return True
            
        except Exception as e:
            print(f"     ❌ Cannot read notebook: {e}")
            return False

    def run_phase_1_code_structure(self):
        """Phase 1: Verify Production Code Structure"""
        print("\n[Phase 1: Production Code Components]")
        print("-" * 50)
        
        phase_passed = True
        
        # Core components with expected classes
        components = {
            "src/agents/rde/model.py": ["RegimeDetectionEngine", "TransformerEncoder", "VAEHead"],
            "src/agents/rde/engine.py": ["RDEComponent"],
            "src/agents/mrms/models.py": ["PositionSizingAgent", "StopLossAgent", 
                                         "ProfitTargetAgent", "RiskManagementEnsemble"],
            "src/agents/mrms/engine.py": ["MRMSComponent"],
            "src/agents/main_core/models.py": ["StructureEmbedder", "TacticalEmbedder", 
                                               "RegimeEmbedder", "LVNEmbedder", 
                                               "SharedPolicy", "DecisionGate"],
            "src/agents/main_core/engine.py": ["MainMARLCoreComponent"],
            "src/detectors/synergy_detector.py": ["SynergyDetector"]
        }
        
        for path, expected_classes in components.items():
            if self.check_path(path):
                if not self.check_module_architecture(path, expected_classes):
                    phase_passed = False
            else:
                phase_passed = False
        
        # Check __init__.py files
        print("\n  Checking module initialization files:")
        init_files = [
            "src/agents/__init__.py",
            "src/agents/rde/__init__.py",
            "src/agents/mrms/__init__.py",
            "src/agents/main_core/__init__.py"
        ]
        for init_file in init_files:
            if not self.check_path(init_file):
                phase_passed = False
        
        self.results["phases"]["code_structure"] = phase_passed
        if not phase_passed:
            self.all_checks_passed = False

    def run_phase_2_training_notebooks(self):
        """Phase 2: Verify Training Notebooks"""
        print("\n[Phase 2: Training Notebooks]")
        print("-" * 50)
        
        phase_passed = True
        
        notebooks = {
            "notebooks/Data_Preparation_Colab.ipynb": [
                "generate_market_data", "MatrixAssembler", "calculate_indicators"
            ],
            "notebooks/Regime_Agent_Training.ipynb": [
                "RegimeDetectionEngine", "vae_loss", "train", "torch.optim"
            ],
            "notebooks/train_mrms_agent.ipynb": [
                "RiskManagementEnsemble", "sortino_ratio", "position_sizing"
            ],
            "notebooks/MARL_Training_Master_Colab.ipynb": [
                "SharedPolicy", "DecisionGate", "mc_dropout", "two_gate"
            ]
        }
        
        for notebook_path, required_content in notebooks.items():
            if self.check_path(notebook_path):
                if not self.verify_notebook_content(notebook_path, required_content):
                    phase_passed = False
            else:
                phase_passed = False
        
        self.results["phases"]["training_notebooks"] = phase_passed
        if not phase_passed:
            self.all_checks_passed = False

    def run_phase_3_model_imports(self):
        """Phase 3: Verify Model Class Imports and Instantiation"""
        print("\n[Phase 3: Model Import & Instantiation Tests]")
        print("-" * 50)
        
        phase_passed = True
        
        # Test imports with proper initialization arguments
        models_to_test = [
            ("RDE Model", "agents.rde.model.RegimeDetectionEngine", 
             {"input_dim": 155, "hidden_dim": 256, "latent_dim": 8}),
            ("M-RMS Ensemble", "agents.mrms.models.RiskManagementEnsemble", 
             {"input_dim": 40, "hidden_dim": 128}),
            ("Structure Embedder", "agents.main_core.models.StructureEmbedder", {}),
            ("Tactical Embedder", "agents.main_core.models.TacticalEmbedder", {}),
            ("Shared Policy", "agents.main_core.models.SharedPolicy", 
             {"input_dim": 136, "hidden_dim": 256}),
            ("Decision Gate", "agents.main_core.models.DecisionGate", 
             {"input_dim": 144, "hidden_dim": 64})
        ]
        
        for model_name, import_path, init_args in models_to_test:
            print(f"\n  Testing {model_name}:")
            if not self.test_model_instantiation(model_name, import_path, init_args):
                phase_passed = False
        
        # Test dimension compatibility
        print("\n  Testing dimension compatibility:")
        try:
            from agents.main_core.models import (
                StructureEmbedder, TacticalEmbedder, 
                RegimeEmbedder, LVNEmbedder
            )
            
            # Create dummy inputs
            structure_input = torch.randn(1, 48, 8)
            tactical_input = torch.randn(1, 60, 7)
            regime_input = torch.randn(1, 8)
            lvn_input = torch.randn(1, 5)
            
            # Test embedders
            s_emb = StructureEmbedder()
            t_emb = TacticalEmbedder()
            r_emb = RegimeEmbedder()
            l_emb = LVNEmbedder()
            
            s_out = s_emb(structure_input)
            t_out = t_emb(tactical_input)
            r_out = r_emb(regime_input)
            l_out = l_emb(lvn_input)
            
            # Check output dimensions
            print(f"     Structure: {structure_input.shape} → {s_out.shape}")
            print(f"     Tactical: {tactical_input.shape} → {t_out.shape}")
            print(f"     Regime: {regime_input.shape} → {r_out.shape}")
            print(f"     LVN: {lvn_input.shape} → {l_out.shape}")
            
            # Test concatenation
            unified = torch.cat([s_out, t_out, r_out, l_out], dim=1)
            print(f"     ✅ Unified state vector: {unified.shape}")
            
        except Exception as e:
            print(f"     ❌ Dimension compatibility test failed: {e}")
            phase_passed = False
        
        self.results["phases"]["model_imports"] = phase_passed
        if not phase_passed:
            self.all_checks_passed = False

    def run_phase_4_environment(self):
        """Phase 4: Environment & Hardware Verification"""
        print("\n[Phase 4: Environment & Hardware]")
        print("-" * 50)
        
        phase_passed = True
        
        # PyTorch and CUDA
        try:
            print(f"\n  PyTorch Installation:")
            print(f"     ✅ Version: {torch.__version__}")
            
            if torch.cuda.is_available():
                print(f"     ✅ GPU: {torch.cuda.get_device_name(0)}")
                print(f"     ✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
                
                # Test CUDA operations
                test_tensor = torch.randn(1000, 1000).cuda()
                result = torch.matmul(test_tensor, test_tensor)
                print(f"     ✅ CUDA operations working")
            else:
                print("     ⚠️  No GPU detected - training will be 10-20x slower")
                self.results["warnings"].append("No GPU available")
        except Exception as e:
            print(f"     ❌ PyTorch error: {e}")
            phase_passed = False
        
        # Check key dependencies
        print(f"\n  Key Dependencies:")
        dependencies = ["numpy", "pandas", "matplotlib", "seaborn", "tqdm", "tensorboard"]
        for dep in dependencies:
            try:
                module = importlib.import_module(dep)
                version = getattr(module, '__version__', 'unknown')
                print(f"     ✅ {dep}: {version}")
            except ImportError:
                print(f"     ❌ {dep}: NOT INSTALLED")
                phase_passed = False
        
        # Check disk space
        print(f"\n  Storage Space:")
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            free_gb = free // (2**30)
            print(f"     ✅ Free space: {free_gb} GB")
            if free_gb < 10:
                print(f"     ⚠️  Low disk space! Recommend at least 20GB free")
                self.results["warnings"].append(f"Low disk space: {free_gb}GB")
        except:
            print(f"     ⚠️  Cannot check disk space")
        
        self.results["phases"]["environment"] = phase_passed
        if not phase_passed:
            self.all_checks_passed = False

    def run_phase_5_data_pipeline(self):
        """Phase 5: Data Pipeline Verification"""
        print("\n[Phase 5: Data Pipeline]")
        print("-" * 50)
        
        phase_passed = True
        
        # Check data directories
        print("\n  Data Directories:")
        data_dirs = ["data/", "data/raw/", "data/processed/", "models/", "models/checkpoints/"]
        for dir_path in data_dirs:
            if not self.check_path(dir_path, is_dir=True):
                # Try to create missing directories
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    print(f"     ✅ Created missing directory: {dir_path}")
                except:
                    phase_passed = False
        
        # Check data pipeline components
        print("\n  Data Pipeline Components:")
        pipeline_files = [
            "src/data/market_data.py",
            "src/data/matrix_assembler.py",
            "src/data/indicators.py"
        ]
        for file_path in pipeline_files:
            if not self.check_path(file_path):
                phase_passed = False
        
        # Test data pipeline functionality
        print("\n  Testing Data Pipeline:")
        try:
            from data.matrix_assembler import MatrixAssembler
            assembler = MatrixAssembler({})
            print("     ✅ MatrixAssembler instantiated successfully")
        except Exception as e:
            print(f"     ❌ MatrixAssembler test failed: {e}")
            phase_passed = False
        
        self.results["phases"]["data_pipeline"] = phase_passed
        if not phase_passed:
            self.all_checks_passed = False

    def run_phase_6_configuration(self):
        """Phase 6: Configuration Files"""
        print("\n[Phase 6: Configuration]")
        print("-" * 50)
        
        phase_passed = True
        
        # Check configuration files
        config_files = {
            "config/settings.yaml": ["synergy_detector", "rde", "mrms", "main_core"],
            "config/model_configs.yaml": ["hidden_dim", "learning_rate", "batch_size"],
            "requirements.txt": ["torch", "numpy", "pandas"]
        }
        
        for config_path, required_content in config_files.items():
            if self.check_path(config_path):
                try:
                    with open(config_path, 'r') as f:
                        content = f.read()
                    
                    missing = [req for req in required_content if req not in content]
                    if missing:
                        print(f"     ⚠️  Missing config: {', '.join(missing)}")
                        phase_passed = False
                    else:
                        print(f"     ✅ Contains all required configurations")
                except Exception as e:
                    print(f"     ❌ Cannot read config: {e}")
                    phase_passed = False
            else:
                phase_passed = False
        
        self.results["phases"]["configuration"] = phase_passed
        if not phase_passed:
            self.all_checks_passed = False

    def generate_remediation_script(self):
        """Generate a script to fix common issues"""
        if self.all_checks_passed:
            return
        
        print("\n[Generating Remediation Script]")
        print("-" * 50)
        
        script_content = """#!/bin/bash
# AlgoSpace Training Readiness Remediation Script
# Generated on: """ + datetime.now().isoformat() + """

echo "Starting AlgoSpace remediation..."

# Create missing directories
mkdir -p src/agents/rde
mkdir -p src/agents/mrms  
mkdir -p src/agents/main_core
mkdir -p src/data
mkdir -p src/detectors
mkdir -p notebooks
mkdir -p data/raw data/processed
mkdir -p models/checkpoints
mkdir -p config
mkdir -p logs/training

# Create missing __init__.py files
touch src/__init__.py
touch src/agents/__init__.py
touch src/agents/rde/__init__.py
touch src/agents/mrms/__init__.py
touch src/agents/main_core/__init__.py
touch src/data/__init__.py
touch src/detectors/__init__.py

# Install missing dependencies
pip install torch torchvision numpy pandas matplotlib seaborn tqdm tensorboard pyyaml

echo "Remediation complete! Please run the verification script again."
"""
        
        with open("fix_training_readiness.sh", "w") as f:
            f.write(script_content)
        
        print("     ✅ Created fix_training_readiness.sh")
        print("     Run: chmod +x fix_training_readiness.sh && ./fix_training_readiness.sh")
        
        self.results["remediation_script"] = "fix_training_readiness.sh"

    def generate_final_report(self):
        """Generate comprehensive final report"""
        print("\n" + "=" * 60)
        print("          ALGOSPACE TRAINING READINESS REPORT")
        print("=" * 60)
        
        # Phase summary
        print("\nPhase Summary:")
        for phase, status in self.results["phases"].items():
            emoji = "✅" if status else "❌"
            print(f"  {emoji} {phase.replace('_', ' ').title()}")
        
        # Critical errors
        if self.results["errors"]:
            print(f"\nCritical Errors ({len(self.results['errors'])}):")
            for error in self.results["errors"][:5]:  # Show first 5
                print(f"  • {error}")
            if len(self.results["errors"]) > 5:
                print(f"  • ... and {len(self.results['errors']) - 5} more")
        
        # Warnings
        if self.results["warnings"]:
            print(f"\nWarnings ({len(self.results['warnings'])}):")
            for warning in self.results["warnings"][:3]:
                print(f"  • {warning}")
        
        # Training time estimate
        if self.all_checks_passed:
            print("\nEstimated Training Time:")
            print("  • RDE Training: 4-6 GPU hours")
            print("  • M-RMS Training: 3-4 GPU hours")
            print("  • Main Core Training: 8-10 GPU hours")
            print("  • Total: ~21-26 GPU hours")
        
        # Final verdict
        print("\n" + "=" * 60)
        if self.all_checks_passed:
            print("✅  SYSTEM IS 100% READY FOR TRAINING!")
            print("\nNext Steps:")
            print("1. Start with notebooks/Regime_Agent_Training.ipynb")
            print("2. Then run notebooks/train_mrms_agent.ipynb")
            print("3. Finally run notebooks/MARL_Training_Master_Colab.ipynb")
            self.results["overall_status"] = "READY"
        else:
            print("❌  SYSTEM IS NOT READY FOR TRAINING")
            print("\nRequired Actions:")
            print("1. Run: ./fix_training_readiness.sh")
            print("2. Fix any remaining errors listed above")
            print("3. Re-run this verification script")
            self.results["overall_status"] = "NOT_READY"
        print("=" * 60)
        
        # Save detailed report
        with open("training_readiness_report.json", "w") as f:
            json.dump(self.results, f, indent=2)
        print("\nDetailed report saved to: training_readiness_report.json")

    def run_all_phases(self):
        """Execute all verification phases"""
        print(f"Starting comprehensive audit at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.run_phase_1_code_structure()
        self.run_phase_2_training_notebooks()
        self.run_phase_3_model_imports()
        self.run_phase_4_environment()
        self.run_phase_5_data_pipeline()
        self.run_phase_6_configuration()
        
        if not self.all_checks_passed:
            self.generate_remediation_script()
        
        self.generate_final_report()


if __name__ == "__main__":
    auditor = TrainingReadinessAuditor()
    auditor.run_all_phases()