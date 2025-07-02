#!/usr/bin/env python3
"""
Simplified AlgoSpace Training Readiness Checker
Focuses on file structure verification without external dependencies
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any

class StructureAuditor:
    """Lightweight structure verification for AlgoSpace"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "phases": {},
            "errors": [],
            "warnings": [],
            "overall_status": "PENDING"
        }
        self.all_checks_passed = True
        
    def check_path(self, path: str, is_dir: bool = False) -> bool:
        """Check if path exists and log result"""
        exists = os.path.isdir(path) if is_dir else os.path.exists(path)
        emoji = "✅" if exists else "❌"
        status = 'Found' if exists else 'MISSING'
        
        print(f"  {emoji} {path}... {status}")
        
        if exists and not is_dir:
            # Check file size
            size = os.path.getsize(path)
            if size < 100:
                print(f"     ⚠️  WARNING: File is very small ({size} bytes)")
                self.results["warnings"].append(f"{path} is only {size} bytes")
            else:
                print(f"     ✅ File size: {size:,} bytes")
        
        if not exists:
            self.results["errors"].append(f"Missing: {path}")
            
        return exists

    def check_module_classes(self, module_path: str, expected_classes: List[str]) -> bool:
        """Check if module contains expected classes"""
        try:
            with open(module_path, 'r') as f:
                content = f.read()
            
            missing_classes = []
            found_classes = []
            for class_name in expected_classes:
                if f"class {class_name}" in content:
                    found_classes.append(class_name)
                else:
                    missing_classes.append(class_name)
            
            if missing_classes:
                print(f"     ❌ Missing classes: {', '.join(missing_classes)}")
                print(f"     ✅ Found classes: {', '.join(found_classes)}")
                self.results["errors"].append(f"{module_path} missing: {missing_classes}")
                return False
            
            print(f"     ✅ All classes found: {', '.join(expected_classes)}")
            return True
            
        except Exception as e:
            print(f"     ❌ Error reading {module_path}: {e}")
            self.results["errors"].append(f"Cannot read {module_path}: {str(e)}")
            return False

    def run_code_structure_audit(self):
        """Phase 1: Verify AlgoSpace Code Structure"""
        print("\n[Phase 1: AlgoSpace Code Components]")
        print("-" * 60)
        
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
        
        print("\n  Core Agent Components:")
        for path, expected_classes in components.items():
            if self.check_path(path):
                if not self.check_module_classes(path, expected_classes):
                    phase_passed = False
            else:
                phase_passed = False
        
        # Check __init__.py files
        print("\n  Module Initialization Files:")
        init_files = [
            "src/__init__.py",
            "src/agents/__init__.py",
            "src/agents/rde/__init__.py",
            "src/agents/mrms/__init__.py",
            "src/agents/main_core/__init__.py",
            "src/data/__init__.py",
            "src/detectors/__init__.py"
        ]
        for init_file in init_files:
            if not self.check_path(init_file):
                phase_passed = False
        
        # Check data pipeline components
        print("\n  Data Pipeline Components:")
        data_files = [
            "src/data/market_data.py",
            "src/data/matrix_assembler.py",
            "src/data/indicators.py"
        ]
        for data_file in data_files:
            if not self.check_path(data_file):
                phase_passed = False
        
        self.results["phases"]["code_structure"] = phase_passed
        if not phase_passed:
            self.all_checks_passed = False

    def run_notebook_audit(self):
        """Phase 2: Verify Training Notebooks"""
        print("\n[Phase 2: Training Notebooks]")
        print("-" * 60)
        
        phase_passed = True
        
        # Check for notebook directory
        print("\n  Notebook Directory:")
        if not self.check_path("notebooks/", is_dir=True):
            print("     Creating notebooks directory...")
            os.makedirs("notebooks", exist_ok=True)
        
        # Expected notebooks
        notebooks = {
            "notebooks/Data_Preparation_Colab.ipynb": "Data preparation and synthetic generation",
            "notebooks/Regime_Agent_Training.ipynb": "RDE Transformer+VAE training",
            "notebooks/train_mrms_agent.ipynb": "M-RMS ensemble training",
            "notebooks/MARL_Training_Master_Colab.ipynb": "Main MARL Core training"
        }
        
        print("\n  Training Notebooks:")
        for notebook_path, description in notebooks.items():
            exists = self.check_path(notebook_path)
            if exists:
                # Check if it's a valid notebook
                try:
                    with open(notebook_path, 'r') as f:
                        notebook_data = json.load(f)
                    if 'cells' not in notebook_data:
                        print(f"     ⚠️  Invalid notebook format")
                        phase_passed = False
                    else:
                        cell_count = len(notebook_data['cells'])
                        print(f"     ✅ Valid notebook with {cell_count} cells")
                except:
                    print(f"     ❌ Cannot parse notebook JSON")
                    phase_passed = False
            else:
                phase_passed = False
        
        self.results["phases"]["notebooks"] = phase_passed
        if not phase_passed:
            self.all_checks_passed = False

    def run_directory_audit(self):
        """Phase 3: Verify Directory Structure"""
        print("\n[Phase 3: Directory Structure]")
        print("-" * 60)
        
        phase_passed = True
        
        # Required directories
        required_dirs = [
            "src/",
            "src/agents/",
            "src/agents/rde/",
            "src/agents/mrms/",
            "src/agents/main_core/",
            "src/data/",
            "src/detectors/",
            "data/",
            "models/",
            "config/",
            "notebooks/"
        ]
        
        print("\n  Core Directories:")
        for dir_path in required_dirs:
            if not self.check_path(dir_path, is_dir=True):
                # Try to create missing directories
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    print(f"     ✅ Created: {dir_path}")
                except Exception as e:
                    print(f"     ❌ Cannot create {dir_path}: {e}")
                    phase_passed = False
        
        # Training-specific directories
        training_dirs = [
            "data/raw/",
            "data/processed/",
            "models/checkpoints/",
            "logs/",
            "logs/training/"
        ]
        
        print("\n  Training Directories:")
        for dir_path in training_dirs:
            if not self.check_path(dir_path, is_dir=True):
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    print(f"     ✅ Created: {dir_path}")
                except Exception as e:
                    print(f"     ❌ Cannot create {dir_path}: {e}")
                    phase_passed = False
        
        self.results["phases"]["directories"] = phase_passed
        if not phase_passed:
            self.all_checks_passed = False

    def run_config_audit(self):
        """Phase 4: Configuration Files"""
        print("\n[Phase 4: Configuration Files]")
        print("-" * 60)
        
        phase_passed = True
        
        # Configuration files
        config_files = [
            "config/settings.yaml",
            "config/model_configs.yaml",
            "config/training_config.yaml",
            "requirements.txt"
        ]
        
        print("\n  Configuration Files:")
        for config_file in config_files:
            exists = self.check_path(config_file)
            if exists:
                # Check content is reasonable
                try:
                    with open(config_file, 'r') as f:
                        content = f.read()
                    if len(content.strip()) < 50:
                        print(f"     ⚠️  Configuration seems minimal")
                        self.results["warnings"].append(f"{config_file} has minimal content")
                    else:
                        print(f"     ✅ Configuration looks substantial")
                except Exception as e:
                    print(f"     ❌ Cannot read config: {e}")
                    phase_passed = False
            else:
                phase_passed = False
        
        self.results["phases"]["configuration"] = phase_passed
        if not phase_passed:
            self.all_checks_passed = False

    def check_environment(self):
        """Phase 5: Basic Environment Check"""
        print("\n[Phase 5: Environment Check]")
        print("-" * 60)
        
        phase_passed = True
        
        # Python version
        print(f"\n  Python Environment:")
        print(f"     ✅ Python version: {sys.version.split()[0]}")
        
        # Check basic imports
        print(f"\n  Basic Dependencies:")
        basic_deps = ["os", "sys", "json", "datetime", "typing"]
        for dep in basic_deps:
            try:
                __import__(dep)
                print(f"     ✅ {dep}: Available")
            except ImportError:
                print(f"     ❌ {dep}: Missing")
                phase_passed = False
        
        # Check if advanced deps are available
        print(f"\n  Advanced Dependencies:")
        advanced_deps = ["torch", "numpy", "pandas", "matplotlib"]
        available_deps = []
        for dep in advanced_deps:
            try:
                module = __import__(dep)
                version = getattr(module, '__version__', 'unknown')
                print(f"     ✅ {dep}: {version}")
                available_deps.append(dep)
            except ImportError:
                print(f"     ❌ {dep}: Not installed")
        
        if len(available_deps) < 2:
            print(f"     ⚠️  Missing critical dependencies for training")
            self.results["warnings"].append("Missing training dependencies")
        
        # Disk space
        print(f"\n  Storage:")
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            free_gb = free // (2**30)
            print(f"     ✅ Free space: {free_gb} GB")
            if free_gb < 10:
                print(f"     ⚠️  Low disk space for training")
                self.results["warnings"].append(f"Low disk space: {free_gb}GB")
        except:
            print(f"     ⚠️  Cannot check disk space")
        
        self.results["phases"]["environment"] = phase_passed
        if not phase_passed:
            self.all_checks_passed = False

    def generate_remediation_plan(self):
        """Generate remediation steps"""
        if self.all_checks_passed:
            return
        
        print("\n[Remediation Plan]")
        print("-" * 60)
        
        fixes = []
        
        if not self.results["phases"].get("code_structure", True):
            fixes.append("• Implement missing agent components (RDE, M-RMS, Main Core)")
            fixes.append("• Create missing __init__.py files for proper imports")
        
        if not self.results["phases"].get("notebooks", True):
            fixes.append("• Create missing training notebooks")
            fixes.append("• Populate notebooks with training code")
        
        if not self.results["phases"].get("directories", True):
            fixes.append("• Create missing directory structure")
        
        if not self.results["phases"].get("configuration", True):
            fixes.append("• Create configuration files")
            fixes.append("• Define training hyperparameters")
        
        if not self.results["phases"].get("environment", True):
            fixes.append("• Install missing Python dependencies")
            fixes.append("• Set up proper Python environment")
        
        print("  Required Actions:")
        for fix in fixes:
            print(f"    {fix}")
        
        # Create auto-fix script
        script_content = f"""#!/bin/bash
# AlgoSpace Structure Fix Script
# Generated: {datetime.now().isoformat()}

echo "Creating missing directories..."
mkdir -p src/agents/rde src/agents/mrms src/agents/main_core
mkdir -p src/data src/detectors
mkdir -p notebooks config data/raw data/processed
mkdir -p models/checkpoints logs/training

echo "Creating __init__.py files..."
touch src/__init__.py
touch src/agents/__init__.py
touch src/agents/rde/__init__.py
touch src/agents/mrms/__init__.py
touch src/agents/main_core/__init__.py
touch src/data/__init__.py
touch src/detectors/__init__.py

echo "Installing Python dependencies..."
pip install --break-system-packages torch numpy pandas matplotlib seaborn tqdm pyyaml

echo "Structure remediation complete!"
echo "Next: Implement missing agent components and training notebooks"
"""
        
        with open("fix_structure.sh", "w") as f:
            f.write(script_content)
        
        print(f"\n  ✅ Auto-fix script created: fix_structure.sh")
        print(f"     Run: chmod +x fix_structure.sh && ./fix_structure.sh")

    def generate_final_report(self):
        """Generate comprehensive report"""
        print("\n" + "=" * 70)
        print("            ALGOSPACE TRAINING READINESS AUDIT REPORT")
        print("=" * 70)
        
        # Phase summary
        print(f"\nAudit Summary:")
        total_phases = len(self.results["phases"])
        passed_phases = sum(1 for status in self.results["phases"].values() if status)
        
        for phase, status in self.results["phases"].items():
            emoji = "✅" if status else "❌"
            print(f"  {emoji} {phase.replace('_', ' ').title()}")
        
        print(f"\nOverall Score: {passed_phases}/{total_phases} phases passed")
        
        # Issues summary
        if self.results["errors"]:
            print(f"\nCritical Issues ({len(self.results['errors'])}):")
            for error in self.results["errors"][:5]:
                print(f"  • {error}")
            if len(self.results["errors"]) > 5:
                print(f"  • ... and {len(self.results['errors']) - 5} more")
        
        if self.results["warnings"]:
            print(f"\nWarnings ({len(self.results['warnings'])}):")
            for warning in self.results["warnings"][:3]:
                print(f"  • {warning}")
        
        # Architecture overview
        print(f"\nAlgoSpace Architecture Verification:")
        print(f"  • Main MARL Core: {'✅' if os.path.exists('src/agents/main_core/models.py') else '❌'} Shared Policy System")
        print(f"  • RDE Component: {'✅' if os.path.exists('src/agents/rde/model.py') else '❌'} Regime Detection")
        print(f"  • M-RMS Component: {'✅' if os.path.exists('src/agents/mrms/models.py') else '❌'} Risk Management")
        print(f"  • Data Pipeline: {'✅' if os.path.exists('src/data/matrix_assembler.py') else '❌'} Matrix Assembly")
        
        # Final verdict
        print("\n" + "=" * 70)
        if self.all_checks_passed:
            print("🎉  STRUCTURE VERIFICATION PASSED!")
            print("\nAlgoSpace architecture is properly implemented!")
            print("\nNext Steps for Training Readiness:")
            print("1. Install PyTorch and ML dependencies")
            print("2. Create comprehensive training notebooks")
            print("3. Run full training readiness verification")
            print("4. Begin 3-phase training: RDE → M-RMS → Main Core")
            self.results["overall_status"] = "STRUCTURE_READY"
        else:
            print("❌  STRUCTURE VERIFICATION FAILED")
            print(f"\nIssues found: {len(self.results['errors'])} errors, {len(self.results['warnings'])} warnings")
            print("\nRequired Actions:")
            print("1. Run ./fix_structure.sh to create missing directories")
            print("2. Implement missing agent components")
            print("3. Create training notebooks")
            print("4. Re-run this verification")
            self.results["overall_status"] = "STRUCTURE_NOT_READY"
        
        print("=" * 70)
        
        # Save report
        with open("structure_audit_report.json", "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nDetailed report saved to: structure_audit_report.json")

    def run_full_audit(self):
        """Execute complete structure audit"""
        print(f"AlgoSpace Structure Audit Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.run_code_structure_audit()
        self.run_notebook_audit()
        self.run_directory_audit()
        self.run_config_audit()
        self.check_environment()
        
        if not self.all_checks_passed:
            self.generate_remediation_plan()
        
        self.generate_final_report()


if __name__ == "__main__":
    auditor = StructureAuditor()
    auditor.run_full_audit()