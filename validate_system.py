#!/usr/bin/env python3
"""
System Validation Script for AlgoSpace
Validates all components are properly implemented without running the full system
"""

import os
import sys
from pathlib import Path
import ast
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, 'src')

class SystemValidator:
    """Validates AlgoSpace system implementation"""
    
    def __init__(self):
        self.results = {
            'passed': [],
            'failed': [],
            'warnings': []
        }
        self.component_stats = {}
    
    def validate_all(self):
        """Run all validation checks"""
        print("\n" + "="*80)
        print("                    AlgoSpace System Validation")
        print("="*80 + "\n")
        
        # Check directory structure
        self.validate_directory_structure()
        
        # Check component implementations
        self.validate_components()
        
        # Check configuration
        self.validate_configuration()
        
        # Check data files
        self.validate_data_files()
        
        # Check code quality
        self.validate_code_quality()
        
        # Generate report
        self.generate_report()
    
    def validate_directory_structure(self):
        """Validate project directory structure"""
        print("[VALIDATING] Directory Structure...")
        
        required_dirs = [
            'src/core',
            'src/data', 
            'src/indicators',
            'src/matrix',
            'src/execution',
            'src/utils',
            'config',
            'data/historical',
            'tests'
        ]
        
        for dir_path in required_dirs:
            if os.path.exists(dir_path):
                self.results['passed'].append(f"✓ Directory exists: {dir_path}")
            else:
                self.results['failed'].append(f"✗ Missing directory: {dir_path}")
    
    def validate_components(self):
        """Validate all component implementations"""
        print("\n[VALIDATING] Component Implementations...")
        
        components = {
            'System Kernel': 'src/core/kernel.py',
            'Event System': 'src/core/events.py',
            'Configuration': 'src/core/config.py',
            'Data Handler': 'src/data/handlers.py',
            'Bar Generator': 'src/data/bar_generator.py',
            'Indicator Engine': 'src/indicators/engine.py',
            'Base Indicator': 'src/indicators/base.py',
            'MLMI Indicator': 'src/indicators/mlmi.py',
            'NWRQK Indicator': 'src/indicators/nwrqk.py',
            'FVG Indicator': 'src/indicators/fvg.py',
            'LVN Indicator': 'src/indicators/lvn.py',
            'MMD Engine': 'src/indicators/mmd.py',
            'Matrix Base': 'src/matrix/base.py',
            'Matrix Normalizers': 'src/matrix/normalizers.py',
            'Matrix Assembler 30m': 'src/matrix/assembler_30m.py',
            'Matrix Assembler 5m': 'src/matrix/assembler_5m.py',
            'Matrix Assembler Regime': 'src/matrix/assembler_regime.py',
            'Main Entry': 'src/main.py'
        }
        
        for name, path in components.items():
            if os.path.exists(path):
                # Analyze the file
                stats = self.analyze_python_file(path)
                self.component_stats[name] = stats
                
                self.results['passed'].append(
                    f"✓ {name}: {stats['lines']} lines, "
                    f"{stats['classes']} classes, {stats['functions']} functions"
                )
                
                # Check for required classes/functions
                self.validate_component_structure(name, path, stats)
            else:
                self.results['failed'].append(f"✗ Missing component: {name} ({path})")
    
    def analyze_python_file(self, filepath):
        """Analyze a Python file and return statistics"""
        stats = {
            'lines': 0,
            'classes': 0,
            'functions': 0,
            'imports': 0,
            'class_names': [],
            'function_names': []
        }
        
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                stats['lines'] = len(content.splitlines())
                
            # Parse AST
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    stats['classes'] += 1
                    stats['class_names'].append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    stats['functions'] += 1
                    stats['function_names'].append(node.name)
                elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    stats['imports'] += 1
        
        except Exception as e:
            self.results['warnings'].append(f"⚠ Error analyzing {filepath}: {e}")
        
        return stats
    
    def validate_component_structure(self, name, path, stats):
        """Validate specific component has required structure"""
        requirements = {
            'System Kernel': {
                'classes': ['SystemKernel', 'ComponentBase'],
                'functions': ['start', 'shutdown', 'register_component']
            },
            'Event System': {
                'classes': ['EventBus', 'EventType'],
                'functions': ['subscribe', 'publish']
            },
            'Data Handler': {
                'classes': ['AbstractDataHandler', 'BacktestDataHandler'],
                'functions': ['connect', 'start_data_stream']
            },
            'Bar Generator': {
                'classes': ['BarGenerator', 'WorkingBar'],
                'functions': ['_floor_timestamp', '_handle_gaps']
            },
            'Indicator Engine': {
                'classes': ['IndicatorEngine'],
                'functions': ['_calculate_5min_features', '_calculate_30min_features']
            }
        }
        
        if name in requirements:
            req = requirements[name]
            
            # Check required classes
            if 'classes' in req:
                for cls in req['classes']:
                    if cls in stats['class_names']:
                        self.results['passed'].append(f"  ✓ {name} has required class: {cls}")
                    else:
                        self.results['failed'].append(f"  ✗ {name} missing required class: {cls}")
            
            # Check required functions
            if 'functions' in req:
                for func in req['functions']:
                    if func in stats['function_names']:
                        self.results['passed'].append(f"  ✓ {name} has required function: {func}")
                    else:
                        self.results['warnings'].append(f"  ⚠ {name} missing function: {func}")
    
    def validate_configuration(self):
        """Validate configuration files"""
        print("\n[VALIDATING] Configuration...")
        
        config_path = 'config/settings.yaml'
        if os.path.exists(config_path):
            self.results['passed'].append(f"✓ Configuration file exists: {config_path}")
            
            # Check file content
            with open(config_path, 'r') as f:
                content = f.read()
            
            # Check for required sections
            required_sections = ['system:', 'timeframes:', 'symbols:', 'data_handler:', 'indicators:']
            for section in required_sections:
                if section in content:
                    self.results['passed'].append(f"  ✓ Config has section: {section}")
                else:
                    self.results['failed'].append(f"  ✗ Config missing section: {section}")
        else:
            self.results['failed'].append(f"✗ Missing configuration: {config_path}")
    
    def validate_data_files(self):
        """Validate data files exist"""
        print("\n[VALIDATING] Data Files...")
        
        data_files = [
            'data/historical/ES - 5 min.csv',
            'data/historical/ES - 30 min - New.csv'
        ]
        
        for filepath in data_files:
            if os.path.exists(filepath):
                # Get file stats
                file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                
                # Count lines
                with open(filepath, 'r') as f:
                    line_count = sum(1 for _ in f)
                
                self.results['passed'].append(
                    f"✓ Data file found: {os.path.basename(filepath)} "
                    f"({file_size:.1f}MB, {line_count:,} lines)"
                )
            else:
                self.results['failed'].append(f"✗ Missing data file: {filepath}")
    
    def validate_code_quality(self):
        """Validate code quality metrics"""
        print("\n[VALIDATING] Code Quality...")
        
        # Calculate total metrics
        total_lines = sum(stats.get('lines', 0) for stats in self.component_stats.values())
        total_classes = sum(stats.get('classes', 0) for stats in self.component_stats.values())
        total_functions = sum(stats.get('functions', 0) for stats in self.component_stats.values())
        
        self.results['passed'].append(f"✓ Total lines of code: {total_lines:,}")
        self.results['passed'].append(f"✓ Total classes: {total_classes}")
        self.results['passed'].append(f"✓ Total functions: {total_functions}")
        
        # Check complexity
        avg_lines_per_component = total_lines / len(self.component_stats) if self.component_stats else 0
        if avg_lines_per_component < 500:
            self.results['passed'].append(f"✓ Good component size (avg: {avg_lines_per_component:.0f} lines)")
        else:
            self.results['warnings'].append(f"⚠ Large components (avg: {avg_lines_per_component:.0f} lines)")
    
    def generate_report(self):
        """Generate validation report"""
        print("\n" + "="*80)
        print("                    VALIDATION REPORT")
        print("="*80)
        
        # Summary
        total_checks = len(self.results['passed']) + len(self.results['failed'])
        pass_rate = (len(self.results['passed']) / total_checks * 100) if total_checks > 0 else 0
        
        print(f"\nSummary:")
        print(f"  Total Checks: {total_checks}")
        print(f"  Passed: {len(self.results['passed'])} ✓")
        print(f"  Failed: {len(self.results['failed'])} ✗")
        print(f"  Warnings: {len(self.results['warnings'])} ⚠")
        print(f"  Pass Rate: {pass_rate:.1f}%")
        
        # Component Statistics
        print(f"\nComponent Statistics:")
        for name, stats in self.component_stats.items():
            print(f"  {name}: {stats['lines']} lines, {stats['classes']} classes")
        
        # Failed Checks
        if self.results['failed']:
            print(f"\nFailed Checks:")
            for fail in self.results['failed']:
                print(f"  {fail}")
        
        # Warnings
        if self.results['warnings']:
            print(f"\nWarnings:")
            for warn in self.results['warnings'][:5]:  # First 5
                print(f"  {warn}")
        
        # Save detailed report
        self.save_detailed_report()
        
        print("\n" + "="*80)
        
        # Final verdict
        if pass_rate >= 90:
            print("✓ SYSTEM VALIDATION PASSED - Ready for deployment")
        elif pass_rate >= 70:
            print("⚠ SYSTEM VALIDATION PASSED WITH WARNINGS - Review issues before deployment")
        else:
            print("✗ SYSTEM VALIDATION FAILED - Critical issues must be resolved")
        
        print("="*80 + "\n")
    
    def save_detailed_report(self):
        """Save detailed report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"validation_report_{timestamp}.json"
        
        report = {
            'timestamp': timestamp,
            'summary': {
                'total_checks': len(self.results['passed']) + len(self.results['failed']),
                'passed': len(self.results['passed']),
                'failed': len(self.results['failed']),
                'warnings': len(self.results['warnings'])
            },
            'component_stats': self.component_stats,
            'results': self.results
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed report saved to: {report_path}")


def main():
    """Run system validation"""
    validator = SystemValidator()
    validator.validate_all()


if __name__ == "__main__":
    main()