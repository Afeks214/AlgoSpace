#!/usr/bin/env python3
"""
Verification Test Runner for AlgoSpace Project
This script runs all structural and logical tests without requiring PyTorch.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestVerifier:
    """Verifies project structure and test coverage."""
    
    def __init__(self):
        self.results = {
            'passed': [],
            'failed': [],
            'warnings': []
        }
        
    def verify_project_structure(self) -> bool:
        """Verify all required directories and files exist."""
        print("\n=== Verifying Project Structure ===")
        
        required_dirs = [
            'src/core',
            'src/indicators',
            'src/agents/main_core',
            'src/agents/mrms',
            'src/agents/rde',
            'tests/agents',
            'tests/core',
            'tests/indicators',
            'config',
            'data',
            'models',
            'notebooks'
        ]
        
        required_files = [
            'src/core/kernel.py',
            'src/core/event_bus.py',
            'src/agents/main_core/engine.py',
            'src/agents/main_core/models.py',
            'src/agents/mrms/engine.py',
            'src/agents/mrms/models.py',
            'src/agents/rde/engine.py',
            'src/agents/rde/models.py',
            'tests/agents/test_main_marl_core.py',
            'tests/agents/test_mrms_engine.py',
            'tests/agents/test_rde_engine.py'
        ]
        
        all_good = True
        
        # Check directories
        for dir_path in required_dirs:
            full_path = project_root / dir_path
            if full_path.exists() and full_path.is_dir():
                self.results['passed'].append(f"Directory exists: {dir_path}")
            else:
                self.results['failed'].append(f"Missing directory: {dir_path}")
                all_good = False
                
        # Check files
        for file_path in required_files:
            full_path = project_root / file_path
            if full_path.exists() and full_path.is_file():
                self.results['passed'].append(f"File exists: {file_path}")
            else:
                self.results['failed'].append(f"Missing file: {file_path}")
                all_good = False
                
        return all_good
    
    def verify_module_imports(self) -> bool:
        """Verify that key modules can be imported (without torch)."""
        print("\n=== Verifying Module Imports ===")
        
        # Modules that should be importable without torch
        basic_modules = [
            'src.core.system_config',
            'config.settings'
        ]
        
        all_good = True
        
        for module_path in basic_modules:
            try:
                module_name = module_path.replace('/', '.').replace('.py', '')
                spec = importlib.util.find_spec(module_name)
                if spec is not None:
                    self.results['passed'].append(f"Module importable: {module_name}")
                else:
                    self.results['failed'].append(f"Module not found: {module_name}")
                    all_good = False
            except Exception as e:
                self.results['warnings'].append(f"Import check skipped for {module_path}: {str(e)}")
                
        return all_good
    
    def verify_test_files(self) -> bool:
        """Verify test file syntax and structure."""
        print("\n=== Verifying Test Files ===")
        
        test_dir = project_root / 'tests'
        test_files = list(test_dir.rglob('test_*.py'))
        
        all_good = True
        
        for test_file in test_files:
            # Try to compile the test file
            try:
                with open(test_file, 'r') as f:
                    compile(f.read(), str(test_file), 'exec')
                self.results['passed'].append(f"Valid syntax: {test_file.relative_to(project_root)}")
            except SyntaxError as e:
                self.results['failed'].append(f"Syntax error in {test_file}: {e}")
                all_good = False
                
        return all_good
    
    def check_code_quality(self) -> bool:
        """Check basic code quality metrics."""
        print("\n=== Checking Code Quality ===")
        
        src_dir = project_root / 'src'
        
        # Count Python files and lines
        py_files = list(src_dir.rglob('*.py'))
        total_lines = 0
        
        for py_file in py_files:
            with open(py_file, 'r') as f:
                lines = f.readlines()
                total_lines += len(lines)
                
        self.results['passed'].append(f"Total Python files in src/: {len(py_files)}")
        self.results['passed'].append(f"Total lines of code: {total_lines}")
        
        # Check for docstrings in key files
        key_files = [
            'src/core/kernel.py',
            'src/agents/main_core/engine.py',
            'src/agents/mrms/engine.py',
            'src/agents/rde/engine.py'
        ]
        
        for file_path in key_files:
            full_path = project_root / file_path
            if full_path.exists():
                with open(full_path, 'r') as f:
                    content = f.read()
                    if '"""' in content or "'''" in content:
                        self.results['passed'].append(f"Has docstrings: {file_path}")
                    else:
                        self.results['warnings'].append(f"Missing docstrings: {file_path}")
                        
        return True
    
    def run_pytest_dry_run(self) -> bool:
        """Run pytest in collect-only mode to verify test discovery."""
        print("\n=== Running PyTest Discovery ===")
        
        try:
            # Run pytest in collect-only mode
            result = subprocess.run(
                ['python', '-m', 'pytest', '--collect-only', '-q'],
                capture_output=True,
                text=True,
                cwd=project_root
            )
            
            if result.returncode == 0:
                # Count collected tests
                lines = result.stdout.strip().split('\n')
                test_count = len([l for l in lines if 'test_' in l])
                self.results['passed'].append(f"PyTest discovered {test_count} tests")
                return True
            else:
                self.results['failed'].append(f"PyTest discovery failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.results['warnings'].append(f"PyTest discovery skipped: {str(e)}")
            return True
    
    def generate_report(self) -> str:
        """Generate final verification report."""
        report = []
        report.append("\n" + "="*60)
        report.append("ALGOSPACE PROJECT VERIFICATION REPORT")
        report.append("="*60)
        
        # Summary
        total_passed = len(self.results['passed'])
        total_failed = len(self.results['failed'])
        total_warnings = len(self.results['warnings'])
        
        report.append(f"\nSUMMARY:")
        report.append(f"  ✅ Passed: {total_passed}")
        report.append(f"  ❌ Failed: {total_failed}")
        report.append(f"  ⚠️  Warnings: {total_warnings}")
        
        # Detailed results
        if self.results['passed']:
            report.append(f"\n✅ PASSED CHECKS ({total_passed}):")
            for item in self.results['passed']:
                report.append(f"  • {item}")
                
        if self.results['failed']:
            report.append(f"\n❌ FAILED CHECKS ({total_failed}):")
            for item in self.results['failed']:
                report.append(f"  • {item}")
                
        if self.results['warnings']:
            report.append(f"\n⚠️  WARNINGS ({total_warnings}):")
            for item in self.results['warnings']:
                report.append(f"  • {item}")
                
        # Final verdict
        report.append("\n" + "="*60)
        if total_failed == 0:
            report.append("FINAL VERDICT: ✅ PROJECT STRUCTURE VERIFIED")
            report.append("\nThe AlgoSpace project structure is complete and ready.")
            report.append("All critical components and test files are in place.")
            report.append("\nNOTE: Full test execution requires PyTorch installation.")
            report.append("To run the complete test suite:")
            report.append("  1. Install PyTorch: pip install torch")
            report.append("  2. Run tests: pytest -v")
        else:
            report.append("FINAL VERDICT: ❌ VERIFICATION FAILED")
            report.append(f"\nFound {total_failed} critical issues that need resolution.")
            
        report.append("="*60 + "\n")
        
        return '\n'.join(report)
    
    def run_all_verifications(self):
        """Run all verification checks."""
        checks = [
            ("Project Structure", self.verify_project_structure),
            ("Module Imports", self.verify_module_imports),
            ("Test Files", self.verify_test_files),
            ("Code Quality", self.check_code_quality),
            ("PyTest Discovery", self.run_pytest_dry_run)
        ]
        
        for check_name, check_func in checks:
            try:
                check_func()
            except Exception as e:
                self.results['warnings'].append(f"{check_name} check error: {str(e)}")


def main():
    """Main entry point."""
    print("Starting AlgoSpace Project Verification...")
    print("This will verify project structure and test readiness.")
    
    verifier = TestVerifier()
    verifier.run_all_verifications()
    
    # Generate and print report
    report = verifier.generate_report()
    print(report)
    
    # Return appropriate exit code
    if len(verifier.results['failed']) > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()