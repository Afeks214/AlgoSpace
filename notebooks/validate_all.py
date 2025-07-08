#!/usr/bin/env python3
"""
AlgoSpace Notebook Validation Script
Performs comprehensive validation of all training notebooks for AlgoSpace architecture alignment.
"""

import os
import json
import nbformat
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import re


class NotebookValidator:
    """Validates AlgoSpace training notebooks against PRD specifications."""
    
    def __init__(self, notebooks_dir: str = "./notebooks", config_path: str = "./config/settings.yaml"):
        self.notebooks_dir = Path(notebooks_dir)
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.prd_specs = self._define_prd_specifications()
        self.validation_results = {}
        self.discrepancies = []
        
    def _load_config(self) -> Dict:
        """Load the main configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load config from {self.config_path}: {e}")
            return {}
    
    def _define_prd_specifications(self) -> Dict:
        """Define the expected PRD specifications."""
        return {
            "architecture": {
                "two_gate_system": True,
                "mc_dropout_passes": 50,
                "confidence_threshold": 0.65,
                "synergy_threshold": 0.2,  # MLMI-NWRQK > 0.2
                "frozen_rde": True,
                "rde_latent_space": 8,
                "mrms_risk_proposals": 4,
                "training_episodes": 10000,
                "batch_size": 256
            },
            "feature_dimensions": {
                "structure": {
                    "input_shape": (48, 8),
                    "output_dim": 64,
                    "window": "30m"
                },
                "tactical": {
                    "input_shape": (60, 7),
                    "output_dim": 48,
                    "window": "5m"
                },
                "regime": {
                    "input_dim": 8,
                    "output_dim": 16,
                    "source": "RDE"
                },
                "lvn": {
                    "input_features": 5,
                    "output_dim": 8
                }
            },
            "agents": {
                "structure_analyzer": {
                    "window": 48,
                    "hidden_dim": 256,
                    "n_layers": 4,
                    "dropout": 0.2
                },
                "short_term_tactician": {
                    "window": 60,
                    "hidden_dim": 192,
                    "n_layers": 3,
                    "dropout": 0.2
                },
                "mid_frequency_arbitrageur": {
                    "window": 100,
                    "hidden_dim": 224,
                    "n_layers": 4,
                    "dropout": 0.2
                }
            },
            "critical_parameters": {
                "mc_dropout_passes": 50,
                "confidence_threshold": 0.65,
                "synergy_mlmi_nwrqk": 0.2,
                "communication_rounds": 3,
                "min_agent_agreement": 2,
                "daily_trade_limit": 10
            }
        }
    
    def validate_notebook_architecture(self, notebook_path: Path) -> Dict:
        """Validate a single notebook against architecture specifications."""
        results = {
            "notebook": notebook_path.name,
            "passed": True,
            "checks": {},
            "discrepancies": []
        }
        
        # Load notebook
        try:
            with open(notebook_path, 'r') as f:
                nb = nbformat.read(f, as_version=4)
        except Exception as e:
            results["passed"] = False
            results["error"] = f"Failed to load notebook: {e}"
            return results
        
        # Extract code cells
        code_cells = [cell['source'] for cell in nb.cells if cell['cell_type'] == 'code']
        full_code = '\n'.join(code_cells)
        
        # Check for two-gate system
        two_gate_check = self._check_two_gate_system(full_code)
        results["checks"]["two_gate_system"] = two_gate_check
        if not two_gate_check["passed"]:
            results["passed"] = False
            results["discrepancies"].append(two_gate_check["message"])
        
        # Check MC Dropout configuration
        mc_dropout_check = self._check_mc_dropout(full_code)
        results["checks"]["mc_dropout"] = mc_dropout_check
        if not mc_dropout_check["passed"]:
            results["passed"] = False
            results["discrepancies"].append(mc_dropout_check["message"])
        
        # Check synergy detection
        synergy_check = self._check_synergy_detection(full_code)
        results["checks"]["synergy_detection"] = synergy_check
        if not synergy_check["passed"]:
            results["passed"] = False
            results["discrepancies"].append(synergy_check["message"])
        
        # Check feature dimensions
        dimension_check = self._check_feature_dimensions(full_code)
        results["checks"]["feature_dimensions"] = dimension_check
        if not dimension_check["passed"]:
            results["passed"] = False
            results["discrepancies"].extend(dimension_check["messages"])
        
        # Check for hardcoded values
        hardcoded_check = self._check_hardcoded_values(full_code)
        results["checks"]["hardcoded_values"] = hardcoded_check
        if hardcoded_check["found"]:
            results["warnings"] = hardcoded_check["values"]
        
        return results
    
    def _check_two_gate_system(self, code: str) -> Dict:
        """Check if the two-gate decision system is implemented."""
        gate_patterns = [
            r"gate[_\s]*1|first[_\s]*gate|synergy[_\s]*gate",
            r"gate[_\s]*2|second[_\s]*gate|decision[_\s]*gate|final[_\s]*gate"
        ]
        
        gate1_found = bool(re.search(gate_patterns[0], code, re.IGNORECASE))
        gate2_found = bool(re.search(gate_patterns[1], code, re.IGNORECASE))
        
        return {
            "passed": gate1_found and gate2_found,
            "gate1": gate1_found,
            "gate2": gate2_found,
            "message": "Two-gate system not properly implemented" if not (gate1_found and gate2_found) else "Two-gate system found"
        }
    
    def _check_mc_dropout(self, code: str) -> Dict:
        """Check MC Dropout configuration."""
        # Look for MC Dropout passes configuration
        mc_patterns = [
            r"n_passes\s*=\s*(\d+)",
            r"n_forward_passes\s*=\s*(\d+)",
            r"mc_dropout.*passes.*=\s*(\d+)",
            r"dropout_passes\s*=\s*(\d+)"
        ]
        
        for pattern in mc_patterns:
            match = re.search(pattern, code, re.IGNORECASE)
            if match:
                passes = int(match.group(1))
                expected = self.prd_specs["critical_parameters"]["mc_dropout_passes"]
                return {
                    "passed": passes == expected,
                    "found": passes,
                    "expected": expected,
                    "message": f"MC Dropout passes: {passes} (expected: {expected})"
                }
        
        return {
            "passed": False,
            "found": None,
            "expected": 50,
            "message": "MC Dropout configuration not found"
        }
    
    def _check_synergy_detection(self, code: str) -> Dict:
        """Check synergy detection implementation."""
        synergy_patterns = [
            r"mlmi.*nwrqk.*>.*0\.\d+",
            r"synergy.*threshold.*=\s*0\.\d+",
            r"MLMI.*NWRQK.*threshold"
        ]
        
        for pattern in synergy_patterns:
            match = re.search(pattern, code, re.IGNORECASE)
            if match:
                # Try to extract threshold value
                threshold_match = re.search(r"0\.\d+", match.group(0))
                if threshold_match:
                    threshold = float(threshold_match.group(0))
                    expected = self.prd_specs["critical_parameters"]["synergy_mlmi_nwrqk"]
                    return {
                        "passed": abs(threshold - expected) < 0.01,
                        "found": threshold,
                        "expected": expected,
                        "message": f"Synergy threshold: {threshold} (expected: {expected})"
                    }
        
        return {
            "passed": False,
            "found": None,
            "expected": 0.2,
            "message": "Synergy detection (MLMI-NWRQK > 0.2) not found"
        }
    
    def _check_feature_dimensions(self, code: str) -> Dict:
        """Check if feature dimensions match specifications."""
        results = {
            "passed": True,
            "messages": [],
            "dimensions": {}
        }
        
        # Define patterns for each embedder
        dimension_patterns = {
            "structure": [
                r"structure.*48.*8",
                r"window.*=.*48",
                r"shape.*48.*8"
            ],
            "tactical": [
                r"tactical.*60.*7",
                r"window.*=.*60",
                r"shape.*60.*7"
            ],
            "regime": [
                r"regime.*8.*16",
                r"latent.*=.*8",
                r"regime_dim.*=.*8"
            ],
            "lvn": [
                r"lvn.*5.*8",
                r"lvn_features.*=.*5",
                r"lvn.*input.*5"
            ]
        }
        
        for embedder, patterns in dimension_patterns.items():
            found = False
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    found = True
                    results["dimensions"][embedder] = "matched"
                    break
            
            if not found:
                results["passed"] = False
                expected = self.prd_specs["feature_dimensions"][embedder]
                results["messages"].append(f"{embedder} dimensions not matching PRD: {expected}")
                results["dimensions"][embedder] = "not found"
        
        return results
    
    def _check_hardcoded_values(self, code: str) -> Dict:
        """Check for hardcoded values that should be in config."""
        hardcoded_patterns = {
            "learning_rate": r"learning_rate\s*=\s*[\d.e-]+",
            "batch_size": r"batch_size\s*=\s*\d+",
            "episodes": r"episodes?\s*=\s*\d+",
            "confidence": r"confidence.*=\s*0\.\d+",
            "threshold": r"threshold\s*=\s*[\d.]+",
            "gpu_device": r"cuda:\d+|device\s*=\s*['\"]cuda['\"]"
        }
        
        found_values = {}
        for name, pattern in hardcoded_patterns.items():
            matches = re.findall(pattern, code, re.IGNORECASE)
            if matches:
                found_values[name] = matches
        
        return {
            "found": len(found_values) > 0,
            "values": found_values,
            "count": sum(len(v) for v in found_values.values())
        }
    
    def check_data_flow(self, notebook_path: Path) -> Dict:
        """Validate data flow through the notebook."""
        results = {
            "notebook": notebook_path.name,
            "data_flow": [],
            "issues": []
        }
        
        try:
            with open(notebook_path, 'r') as f:
                nb = nbformat.read(f, as_version=4)
            
            # Track data transformations
            data_vars = set()
            transformations = []
            
            for cell in nb.cells:
                if cell['cell_type'] == 'code':
                    source = cell['source']
                    
                    # Look for data loading
                    if re.search(r"pd\.read|load.*data|torch\.load", source):
                        transformations.append({
                            "type": "data_load",
                            "cell": nb.cells.index(cell)
                        })
                    
                    # Look for data transformations
                    if re.search(r"transform|preprocess|normalize|reshape", source):
                        transformations.append({
                            "type": "transformation",
                            "cell": nb.cells.index(cell)
                        })
                    
                    # Look for model inputs
                    if re.search(r"model\(|forward\(|\.predict\(", source):
                        transformations.append({
                            "type": "model_input",
                            "cell": nb.cells.index(cell)
                        })
            
            results["data_flow"] = transformations
            
        except Exception as e:
            results["issues"].append(f"Error analyzing data flow: {e}")
        
        return results
    
    def check_colab_optimization(self, notebook_path: Path) -> Dict:
        """Check for Colab-specific optimizations and issues."""
        results = {
            "notebook": notebook_path.name,
            "optimizations": {},
            "issues": []
        }
        
        try:
            with open(notebook_path, 'r') as f:
                nb = nbformat.read(f, as_version=4)
            
            code_cells = [cell['source'] for cell in nb.cells if cell['cell_type'] == 'code']
            full_code = '\n'.join(code_cells)
            
            # Check for GPU memory management
            gpu_checks = {
                "memory_clearing": bool(re.search(r"torch\.cuda\.empty_cache|gc\.collect", full_code)),
                "device_checks": bool(re.search(r"cuda\.is_available|device.*cuda", full_code)),
                "mixed_precision": bool(re.search(r"autocast|mixed.*precision|fp16|amp", full_code))
            }
            results["optimizations"]["gpu"] = gpu_checks
            
            # Check for data loading optimizations
            data_checks = {
                "batch_loading": bool(re.search(r"DataLoader.*num_workers", full_code)),
                "pin_memory": bool(re.search(r"pin_memory\s*=\s*True", full_code)),
                "prefetch": bool(re.search(r"prefetch|DataLoader.*prefetch", full_code))
            }
            results["optimizations"]["data_loading"] = data_checks
            
            # Check for checkpointing
            checkpoint_checks = {
                "save_checkpoint": bool(re.search(r"torch\.save|save.*checkpoint", full_code)),
                "resume_capability": bool(re.search(r"load.*checkpoint|resume.*checkpoint", full_code)),
                "google_drive": bool(re.search(r"drive\.mount|/content/drive", full_code))
            }
            results["optimizations"]["checkpointing"] = checkpoint_checks
            
            # Check for progress tracking
            progress_checks = {
                "tqdm": bool(re.search(r"from tqdm|tqdm\(", full_code)),
                "logging": bool(re.search(r"logging\.|logger\.", full_code)),
                "tensorboard": bool(re.search(r"tensorboard|SummaryWriter", full_code)),
                "wandb": bool(re.search(r"import wandb|wandb\.", full_code))
            }
            results["optimizations"]["progress_tracking"] = progress_checks
            
            # Identify potential issues
            if not gpu_checks["memory_clearing"]:
                results["issues"].append("No GPU memory clearing found - may cause OOM errors")
            
            if not data_checks["batch_loading"]:
                results["issues"].append("DataLoader not using multiple workers - slow data loading")
            
            if not checkpoint_checks["save_checkpoint"]:
                results["issues"].append("No checkpointing found - progress may be lost")
            
            if not progress_checks["tqdm"] and not progress_checks["logging"]:
                results["issues"].append("No progress tracking - difficult to monitor training")
            
        except Exception as e:
            results["issues"].append(f"Error checking optimizations: {e}")
        
        return results
    
    def validate_all_notebooks(self) -> Dict:
        """Validate all notebooks in the directory."""
        all_results = {
            "validation_date": datetime.now().isoformat(),
            "notebooks_checked": 0,
            "passed": 0,
            "failed": 0,
            "warnings": 0,
            "results": {}
        }
        
        # Find all notebooks
        notebook_files = list(self.notebooks_dir.rglob("*.ipynb"))
        all_results["notebooks_checked"] = len(notebook_files)
        
        for notebook_path in notebook_files:
            print(f"Validating {notebook_path.name}...")
            
            # Architecture validation
            arch_results = self.validate_notebook_architecture(notebook_path)
            
            # Data flow validation
            flow_results = self.check_data_flow(notebook_path)
            
            # Colab optimization check
            colab_results = self.check_colab_optimization(notebook_path)
            
            # Combine results
            combined_results = {
                "architecture": arch_results,
                "data_flow": flow_results,
                "colab_optimization": colab_results
            }
            
            all_results["results"][notebook_path.name] = combined_results
            
            # Update counters
            if arch_results["passed"]:
                all_results["passed"] += 1
            else:
                all_results["failed"] += 1
            
            if arch_results.get("warnings"):
                all_results["warnings"] += 1
        
        return all_results
    
    def generate_report(self, results: Dict, output_path: str = "validation_report.json"):
        """Generate a comprehensive validation report."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nValidation report saved to: {output_path}")
        print(f"Total notebooks: {results['notebooks_checked']}")
        print(f"Passed: {results['passed']}")
        print(f"Failed: {results['failed']}")
        print(f"Warnings: {results['warnings']}")


def main():
    """Main validation function."""
    print("AlgoSpace Notebook Validation Script")
    print("=" * 50)
    
    validator = NotebookValidator()
    results = validator.validate_all_notebooks()
    
    # Generate reports
    validator.generate_report(results, "notebooks/validation_report.json")
    
    # Generate alignment report
    print("\nGenerating alignment report...")
    alignment_issues = []
    for notebook, result in results["results"].items():
        if result["architecture"]["discrepancies"]:
            alignment_issues.extend([
                f"**{notebook}**: {disc}" 
                for disc in result["architecture"]["discrepancies"]
            ])
    
    with open("notebooks/alignment_report.md", "w") as f:
        f.write("# AlgoSpace Notebook Alignment Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if alignment_issues:
            f.write("## Architecture Discrepancies Found\n\n")
            for issue in alignment_issues:
                f.write(f"- {issue}\n")
        else:
            f.write("## ✅ All notebooks aligned with PRD specifications\n")
        
        f.write("\n## Summary\n\n")
        f.write(f"- Total notebooks validated: {results['notebooks_checked']}\n")
        f.write(f"- Passed validation: {results['passed']}\n")
        f.write(f"- Failed validation: {results['failed']}\n")
        f.write(f"- Warnings: {results['warnings']}\n")
    
    print("Alignment report saved to: notebooks/alignment_report.md")
    
    print("\n✅ Validation complete!")


if __name__ == "__main__":
    main()