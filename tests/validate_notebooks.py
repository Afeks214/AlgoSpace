"""
Notebook validation script to ensure training notebooks are compatible with production pipeline.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import nbformat
from nbformat import v4 as nbv

logger = logging.getLogger(__name__)


class NotebookValidator:
    """Validate Jupyter notebooks for production compatibility."""
    
    def __init__(self, notebooks_dir: str = "notebooks"):
        self.notebooks_dir = Path(notebooks_dir)
        self.validation_results = {}
        
    def validate_all_notebooks(self) -> Dict[str, Any]:
        """Validate all notebooks in the notebooks directory."""
        logger.info("Starting notebook validation...")
        
        notebooks = list(self.notebooks_dir.rglob("*.ipynb"))
        
        results = {
            'total_notebooks': len(notebooks),
            'validated_notebooks': 0,
            'failed_notebooks': 0,
            'notebook_details': {},
            'common_issues': [],
            'recommendations': []
        }
        
        for notebook_path in notebooks:
            try:
                notebook_result = self.validate_notebook(notebook_path)
                results['notebook_details'][str(notebook_path)] = notebook_result
                
                if notebook_result['valid']:
                    results['validated_notebooks'] += 1
                else:
                    results['failed_notebooks'] += 1
                    
            except Exception as e:
                logger.error(f"Failed to validate {notebook_path}: {e}")
                results['notebook_details'][str(notebook_path)] = {
                    'valid': False,
                    'error': str(e),
                    'issues': ['Validation failed']
                }
                results['failed_notebooks'] += 1
        
        # Analyze common issues
        self._analyze_common_issues(results)
        
        return results
    
    def validate_notebook(self, notebook_path: Path) -> Dict[str, Any]:
        """Validate a single notebook."""
        logger.info(f"Validating notebook: {notebook_path}")
        
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        
        result = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'data_pipeline_compatible': False,
            'production_ready': False,
            'hardcoded_paths': [],
            'imports_valid': True,
            'config_management': False
        }
        
        # Check cells for various issues
        for cell_idx, cell in enumerate(notebook.cells):
            if cell.cell_type == 'code':
                self._validate_code_cell(cell, cell_idx, result)
        
        # Check overall structure
        self._validate_notebook_structure(notebook, result)
        
        # Determine if valid
        result['valid'] = len(result['issues']) == 0
        
        return result
    
    def _validate_code_cell(self, cell: nbv.new_code_cell, cell_idx: int, result: Dict):
        """Validate a code cell."""
        source = cell.source
        
        # Check for hardcoded paths
        hardcoded_patterns = [
            '/content/',  # Google Colab paths
            'C:\\',       # Windows absolute paths
            '/Users/',    # Mac absolute paths
            '/home/specific_user'  # Linux user-specific paths
        ]
        
        for pattern in hardcoded_patterns:
            if pattern in source and pattern != '/home/QuantNova/AlgoSpace-4':  # Allow our project path
                result['hardcoded_paths'].append({
                    'cell': cell_idx,
                    'pattern': pattern,
                    'line': source
                })
                result['issues'].append(f"Hardcoded path in cell {cell_idx}: {pattern}")
        
        # Check for data pipeline compatibility
        pipeline_indicators = [
            'src.data.handlers',
            'src.components.bar_generator',
            'src.indicators.engine',
            'src.matrix.',
            'BarGenerator',
            'IndicatorEngine',
            'MatrixAssembler'
        ]
        
        for indicator in pipeline_indicators:
            if indicator in source:
                result['data_pipeline_compatible'] = True
                break
        
        # Check for proper imports
        problematic_imports = [
            'import sys',
            'sys.path.append',
            'from google.colab',
        ]
        
        for prob_import in problematic_imports:
            if prob_import in source and 'google.colab' in prob_import:
                result['warnings'].append(f"Colab-specific import in cell {cell_idx}")
            elif prob_import in source and 'sys.path' in prob_import:
                result['warnings'].append(f"Manual path manipulation in cell {cell_idx}")
        
        # Check for configuration management
        config_indicators = [
            'config/',
            'settings.yaml',
            'os.environ',
            'CONFIG',
            'load_config'
        ]
        
        for indicator in config_indicators:
            if indicator in source:
                result['config_management'] = True
        
        # Check for production readiness indicators
        production_indicators = [
            'model.eval()',
            'torch.no_grad()',
            'model.save',
            'checkpoint',
            'state_dict'
        ]
        
        production_count = sum(1 for indicator in production_indicators if indicator in source)
        if production_count >= 2:
            result['production_ready'] = True
    
    def _validate_notebook_structure(self, notebook: nbformat.NotebookNode, result: Dict):
        """Validate overall notebook structure."""
        cells = notebook.cells
        
        # Check for markdown documentation
        markdown_cells = [cell for cell in cells if cell.cell_type == 'markdown']
        if len(markdown_cells) == 0:
            result['warnings'].append("No documentation (markdown cells) found")
        
        # Check for proper organization
        code_cells = [cell for cell in cells if cell.cell_type == 'code']
        if len(code_cells) == 0:
            result['issues'].append("No code cells found")
        
        # Check for outputs (should be cleared for production)
        cells_with_output = [cell for cell in code_cells if cell.get('outputs', [])]
        if len(cells_with_output) > len(code_cells) * 0.5:
            result['warnings'].append("Many cells have outputs - consider clearing for production")
    
    def _analyze_common_issues(self, results: Dict):
        """Analyze common issues across all notebooks."""
        all_issues = []
        all_warnings = []
        
        for notebook_path, details in results['notebook_details'].items():
            all_issues.extend(details.get('issues', []))
            all_warnings.extend(details.get('warnings', []))
        
        # Count issue frequency
        issue_counts = {}
        for issue in all_issues:
            # Generalize the issue
            if 'Hardcoded path' in issue:
                key = 'Hardcoded paths'
            elif 'import' in issue.lower():
                key = 'Import issues'
            else:
                key = issue
            
            issue_counts[key] = issue_counts.get(key, 0) + 1
        
        # Store most common issues
        results['common_issues'] = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Generate recommendations
        recommendations = []
        
        if any('Hardcoded paths' in issue for issue, count in results['common_issues']):
            recommendations.append("Replace hardcoded paths with configuration variables")
        
        if any('Colab' in warning for warning in all_warnings):
            recommendations.append("Remove Colab-specific code for production deployment")
        
        data_compatible_count = sum(1 for details in results['notebook_details'].values() 
                                  if details.get('data_pipeline_compatible', False))
        if data_compatible_count < results['total_notebooks'] * 0.5:
            recommendations.append("Ensure notebooks use the standard data pipeline components")
        
        production_ready_count = sum(1 for details in results['notebook_details'].values() 
                                   if details.get('production_ready', False))
        if production_ready_count < results['total_notebooks'] * 0.8:
            recommendations.append("Add model saving and evaluation mode code for production readiness")
        
        results['recommendations'] = recommendations


def main():
    """Run notebook validation."""
    logging.basicConfig(level=logging.INFO)
    
    validator = NotebookValidator()
    results = validator.validate_all_notebooks()
    
    # Save results
    with open('notebook_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n=== NOTEBOOK VALIDATION RESULTS ===")
    print(f"Total notebooks: {results['total_notebooks']}")
    print(f"Valid notebooks: {results['validated_notebooks']}")
    print(f"Failed notebooks: {results['failed_notebooks']}")
    print(f"Success rate: {results['validated_notebooks']/results['total_notebooks']*100:.1f}%")
    
    if results['common_issues']:
        print(f"\n🔍 COMMON ISSUES:")
        for issue, count in results['common_issues'][:5]:
            print(f"  - {issue}: {count} occurrences")
    
    if results['recommendations']:
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in results['recommendations']:
            print(f"  - {rec}")
    
    print(f"\nDetailed results saved to: notebook_validation_results.json")
    
    return results


if __name__ == '__main__':
    main()