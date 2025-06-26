"""Model Evaluation Framework for MARL Trading System.

This module provides comprehensive evaluation tools for assessing the performance
of trained multi-agent trading models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
import json
from datetime import datetime
import torch

from src.training.monitoring import ModelEvaluator, BacktestEngine
from src.core.performance_metrics import PerformanceCalculator


logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """Comprehensive evaluation system for MARL trading models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize comprehensive evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        
        # Initialize sub-evaluators
        self.model_evaluator = ModelEvaluator(config)
        self.backtest_engine = BacktestEngine(config.get('backtesting', {}))
        self.perf_calculator = PerformanceCalculator()
        
        logger.info("Initialized ComprehensiveEvaluator")
    
    def evaluate_models(self, agents: Dict[str, Any], 
                       test_data: pd.DataFrame,
                       env_config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive model evaluation.
        
        Args:
            agents: Trained agent models
            test_data: Test dataset
            env_config: Environment configuration
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info("Starting comprehensive model evaluation...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config
        }
        
        # 1. Live environment evaluation
        from src.training.environment import MultiAgentTradingEnv
        test_env = MultiAgentTradingEnv(env_config)
        
        logger.info("Running live environment evaluation...")
        results['live_evaluation'] = self.model_evaluator.evaluate(
            agents, test_env, render=False
        )
        
        # 2. Historical backtest
        logger.info("Running historical backtest...")
        results['backtest'] = self.backtest_engine.backtest(
            agents, test_data
        )
        
        # 3. Statistical analysis
        logger.info("Performing statistical analysis...")
        results['statistical_analysis'] = self._statistical_analysis(
            results['live_evaluation'], 
            results['backtest']
        )
        
        # 4. Risk analysis
        logger.info("Performing risk analysis...")
        results['risk_analysis'] = self._risk_analysis(
            results['live_evaluation'],
            results['backtest']
        )
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _statistical_analysis(self, live_results: Dict[str, Any],
                            backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical analysis of results.
        
        Args:
            live_results: Live evaluation results
            backtest_results: Backtest results
            
        Returns:
            Statistical analysis
        """
        analysis = {}
        
        # Compare live vs backtest performance
        live_metrics = live_results.get('metrics', {})
        backtest_metrics = backtest_results
        
        # Performance comparison
        comparison_metrics = ['sharpe_ratio', 'max_drawdown', 'win_rate']
        analysis['performance_comparison'] = {}
        
        for metric in comparison_metrics:
            if metric in live_metrics and metric in backtest_metrics:
                live_val = live_metrics[metric]
                backtest_val = backtest_metrics[metric]
                
                analysis['performance_comparison'][metric] = {
                    'live': live_val,
                    'backtest': backtest_val,
                    'difference': live_val - backtest_val,
                    'relative_difference': (live_val - backtest_val) / backtest_val if backtest_val != 0 else 0
                }
        
        # Statistical significance tests would go here
        
        return analysis
    
    def _risk_analysis(self, live_results: Dict[str, Any],
                      backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform risk analysis.
        
        Args:
            live_results: Live evaluation results  
            backtest_results: Backtest results
            
        Returns:
            Risk analysis
        """
        analysis = {}
        
        # Extract metrics
        live_metrics = live_results.get('metrics', {})
        
        # Risk metrics
        analysis['risk_metrics'] = {
            'max_drawdown': live_metrics.get('max_drawdown', 0),
            'sharpe_ratio': live_metrics.get('sharpe_ratio', 0),
            'sortino_ratio': live_metrics.get('sortino_ratio', 0),
            'calmar_ratio': live_metrics.get('calmar_ratio', 0)
        }
        
        # Risk assessment
        risk_score = 0
        if analysis['risk_metrics']['max_drawdown'] < -0.20:
            risk_score += 2  # High risk
        elif analysis['risk_metrics']['max_drawdown'] < -0.10:
            risk_score += 1  # Medium risk
            
        if analysis['risk_metrics']['sharpe_ratio'] < 0.5:
            risk_score += 2  # Poor risk-adjusted returns
        elif analysis['risk_metrics']['sharpe_ratio'] < 1.0:
            risk_score += 1
            
        analysis['risk_assessment'] = {
            'risk_score': risk_score,
            'risk_level': 'High' if risk_score >= 3 else 'Medium' if risk_score >= 1 else 'Low'
        }
        
        return analysis
    
    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results.
        
        Args:
            results: Evaluation results to save
        """
        output_dir = Path(self.config.get('output_path', 'evaluation_results'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        report_file = output_dir / f"comprehensive_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to {report_file}")


def create_evaluation_pipeline(config_path: str) -> ComprehensiveEvaluator:
    """Create evaluation pipeline from configuration.
    
    Args:
        config_path: Path to evaluation configuration
        
    Returns:
        Configured evaluator
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return ComprehensiveEvaluator(config['evaluation'])