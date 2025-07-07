"""
Real-time monitoring and visualization for MC Dropout consensus mechanism.

This module provides comprehensive monitoring capabilities including
performance tracking, uncertainty evolution, and decision quality metrics.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from typing import Dict, List, Any, Optional, TYPE_CHECKING
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import logging
from collections import deque
import json

if TYPE_CHECKING:
    from .mc_dropout import ConsensusResult

logger = logging.getLogger(__name__)


class MCDropoutMonitor:
    """
    Real-time monitoring system for MC Dropout consensus mechanism.
    
    Tracks performance, uncertainty evolution, and decision quality.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.window_size = config.get('window_size', 1000)
        
        # Metrics storage
        self.decision_history = deque(maxlen=self.window_size)
        self.uncertainty_history = deque(maxlen=self.window_size)
        self.performance_metrics = deque(maxlen=self.window_size)
        self.calibration_metrics = deque(maxlen=self.window_size)
        
        # Real-time statistics
        self.running_stats = RunningStatistics(window_size=100)
        
        # Visualization
        self.dashboard = None
        if config.get('enable_dashboard', True):
            self._init_dashboard()
            
    def record_decision(
        self,
        consensus_result: 'ConsensusResult',
        execution_result: Optional[Dict[str, Any]] = None
    ):
        """Record a decision and its outcome."""
        timestamp = datetime.now()
        
        # Extract key metrics
        record = {
            'timestamp': timestamp,
            'decision': consensus_result.should_proceed,
            'confidence': consensus_result.uncertainty_metrics.confidence_score,
            'calibrated_confidence': consensus_result.uncertainty_metrics.calibrated_confidence,
            'total_uncertainty': consensus_result.uncertainty_metrics.total_uncertainty,
            'epistemic_uncertainty': consensus_result.uncertainty_metrics.epistemic_uncertainty,
            'aleatoric_uncertainty': consensus_result.uncertainty_metrics.aleatoric_uncertainty,
            'convergence': consensus_result.convergence_info['converged'],
            'effective_samples': consensus_result.convergence_info['effective_samples'],
            'outlier_count': len(consensus_result.outlier_samples),
            'execution_result': execution_result
        }
        
        # Store record
        self.decision_history.append(record)
        
        # Store uncertainty metrics separately for detailed tracking
        self.uncertainty_history.append({
            'timestamp': timestamp,
            'total_uncertainty': consensus_result.uncertainty_metrics.total_uncertainty,
            'epistemic_uncertainty': consensus_result.uncertainty_metrics.epistemic_uncertainty,
            'aleatoric_uncertainty': consensus_result.uncertainty_metrics.aleatoric_uncertainty,
            'confidence': consensus_result.uncertainty_metrics.confidence_score,
            'calibrated_confidence': consensus_result.uncertainty_metrics.calibrated_confidence,
            'boundary_distance': consensus_result.uncertainty_metrics.decision_boundary_distance
        })
        
        # Update running statistics
        self.running_stats.update(record)
            
    def get_current_stats(self) -> Dict[str, float]:
        """Get current performance statistics."""
        if not self.decision_history:
            return {}
            
        recent = list(self.decision_history)[-100:]
        
        # Decision statistics
        decisions = [r['decision'] for r in recent]
        decision_rate = np.mean(decisions)
        
        # Confidence statistics
        confidences = [r['confidence'] for r in recent]
        cal_confidences = [r['calibrated_confidence'] for r in recent]
        
        # Uncertainty statistics
        epistemics = [r['epistemic_uncertainty'] for r in recent]
        aleatorics = [r['aleatoric_uncertainty'] for r in recent]
        
        # Quality metrics
        converged = [r['convergence'] for r in recent]
        convergence_rate = np.mean(converged)
        
        # Performance (if execution results available)
        executed = [r for r in recent if r['execution_result'] is not None]
        win_rate = 0
        avg_pnl = 0
        
        if executed:
            wins = [r['execution_result'].get('profitable', False) for r in executed]
            win_rate = np.mean(wins)
            
            pnls = [r['execution_result'].get('pnl', 0) for r in executed]
            avg_pnl = np.mean(pnls)
            
        return {
            'decision_rate': decision_rate,
            'avg_confidence': np.mean(confidences),
            'avg_calibrated_confidence': np.mean(cal_confidences),
            'confidence_std': np.std(confidences),
            'avg_epistemic': np.mean(epistemics),
            'avg_aleatoric': np.mean(aleatorics),
            'uncertainty_ratio': np.mean(epistemics) / (np.mean(aleatorics) + 1e-8),
            'convergence_rate': convergence_rate,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_decisions': len(self.decision_history)
        }
        
    def create_uncertainty_plot(self) -> go.Figure:
        """Create interactive uncertainty decomposition plot."""
        if not self.uncertainty_history:
            return go.Figure()
            
        df = pd.DataFrame(list(self.uncertainty_history)[-500:])
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Uncertainty Evolution',
                'Uncertainty Decomposition',
                'Confidence Distribution',
                'Decision Boundary Distance'
            )
        )
        
        # Uncertainty evolution
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['total_uncertainty'],
                name='Total',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['epistemic_uncertainty'],
                name='Epistemic',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['aleatoric_uncertainty'],
                name='Aleatoric',
                line=dict(color='green', width=2)
            ),
            row=1, col=1
        )
        
        # Uncertainty decomposition pie
        latest = df.iloc[-1]
        fig.add_trace(
            go.Pie(
                labels=['Epistemic', 'Aleatoric'],
                values=[
                    latest['epistemic_uncertainty'],
                    latest['aleatoric_uncertainty']
                ],
                hole=0.3
            ),
            row=1, col=2
        )
        
        # Confidence distribution
        fig.add_trace(
            go.Histogram(
                x=df['calibrated_confidence'],
                nbinsx=20,
                name='Calibrated',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Histogram(
                x=df['confidence'],
                nbinsx=20,
                name='Raw',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Decision boundary distance
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['boundary_distance'],
                mode='markers',
                marker=dict(
                    color=df['boundary_distance'],
                    colorscale='RdYlGn',
                    cmin=-1,
                    cmax=1,
                    size=8,
                    colorbar=dict(title='Distance')
                )
            ),
            row=2, col=2
        )
        
        # Add threshold line
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
            row=2, col=2
        )
        
        fig.update_layout(
            title='MC Dropout Uncertainty Analysis',
            height=800,
            showlegend=True
        )
        
        return fig
        
    def create_performance_dashboard(self) -> go.Figure:
        """Create comprehensive performance dashboard."""
        stats = self.get_current_stats()
        
        fig = make_subplots(
            rows=3, cols=3,
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}],
                [{"colspan": 2}, None, {"type": "scatter"}]
            ],
            subplot_titles=(
                '', '', '',
                'Decision Rate', 'Uncertainty Breakdown', 'Convergence',
                'Calibration Quality', '', 'P&L Distribution'
            )
        )
        
        # KPI indicators
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=stats['decision_rate'] * 100,
                title={'text': "Decision Rate %"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 50], 'color': "lightgray"},
                           {'range': [50, 80], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=stats['avg_calibrated_confidence'] * 100,
                title={'text': "Avg Confidence %"},
                gauge={'axis': {'range': [50, 100]},
                       'bar': {'color': "green"}}
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=stats['win_rate'] * 100,
                title={'text': "Win Rate %"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "gold"}}
            ),
            row=1, col=3
        )
        
        # Time series plots
        if self.decision_history:
            df = pd.DataFrame(list(self.decision_history)[-200:])
            
            # Decision rate over time
            df['decision_ma'] = df['decision'].rolling(20).mean()
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['decision_ma'],
                    mode='lines',
                    name='20-period MA'
                ),
                row=2, col=1
            )
            
            # Uncertainty breakdown
            uncertainty_data = df[['epistemic_uncertainty', 'aleatoric_uncertainty']].mean()
            fig.add_trace(
                go.Bar(
                    x=['Epistemic', 'Aleatoric'],
                    y=uncertainty_data.values,
                    marker_color=['red', 'green']
                ),
                row=2, col=2
            )
            
            # Convergence rate
            df['convergence_ma'] = df['convergence'].rolling(20).mean()
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['convergence_ma'],
                    mode='lines',
                    name='Convergence Rate'
                ),
                row=2, col=3
            )
            
            # Calibration quality (if available)
            if self.calibration_metrics:
                cal_df = pd.DataFrame(list(self.calibration_metrics)[-100:])
                fig.add_trace(
                    go.Scatter(
                        x=cal_df['confidence_bin'],
                        y=cal_df['actual_accuracy'],
                        mode='markers',
                        name='Actual',
                        marker=dict(size=8)
                    ),
                    row=3, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode='lines',
                        name='Perfect Calibration',
                        line=dict(dash='dash')
                    ),
                    row=3, col=1
                )
                
            # P&L distribution
            if 'execution_result' in df.columns:
                pnls = [r.get('pnl', 0) for r in df['execution_result'] if r]
                if pnls:
                    fig.add_trace(
                        go.Histogram(
                            x=pnls,
                            nbinsx=30,
                            name='P&L Distribution'
                        ),
                        row=3, col=3
                    )
                    
        fig.update_layout(
            title='MC Dropout Performance Dashboard',
            height=1000,
            showlegend=False
        )
        
        return fig
        
    def generate_report(self) -> str:
        """Generate text report of MC Dropout performance."""
        stats = self.get_current_stats()
        
        report = "MC Dropout Consensus Report\n"
        report += "=" * 50 + "\n\n"
        
        report += "Summary Statistics:\n"
        report += f"  Total Decisions: {stats['total_decisions']}\n"
        report += f"  Decision Rate: {stats['decision_rate']:.2%}\n"
        report += f"  Average Confidence: {stats['avg_confidence']:.3f}\n"
        report += f"  Calibrated Confidence: {stats['avg_calibrated_confidence']:.3f}\n"
        report += f"  Convergence Rate: {stats['convergence_rate']:.2%}\n\n"
        
        report += "Uncertainty Analysis:\n"
        report += f"  Average Total: {stats.get('avg_total', 0):.3f}\n"
        report += f"  Average Epistemic: {stats['avg_epistemic']:.3f}\n"
        report += f"  Average Aleatoric: {stats['avg_aleatoric']:.3f}\n"
        report += f"  Uncertainty Ratio: {stats['uncertainty_ratio']:.2f}\n\n"
        
        if stats['win_rate'] > 0:
            report += "Trading Performance:\n"
            report += f"  Win Rate: {stats['win_rate']:.2%}\n"
            report += f"  Average P&L: ${stats['avg_pnl']:.2f}\n\n"
            
        # Recent decisions summary
        if self.decision_history:
            recent = list(self.decision_history)[-10:]
            report += "Recent Decisions:\n"
            
            for i, decision in enumerate(recent):
                report += f"  {i+1}. {decision['timestamp'].strftime('%H:%M:%S')} - "
                report += f"{'PROCEED' if decision['decision'] else 'REJECT'} "
                report += f"(conf: {decision['confidence']:.3f})\n"
                
        return report
        
    def save_metrics(self, filepath: str):
        """Save monitoring metrics to file."""
        metrics = {
            'stats': self.get_current_stats(),
            'decision_history': [self._serialize_record(r) for r in list(self.decision_history)[-100:]],
            'uncertainty_history': [self._serialize_record(r) for r in list(self.uncertainty_history)[-100:]],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        logger.info(f"Metrics saved to {filepath}")
        
    def _serialize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize record for JSON storage."""
        serialized = {}
        for k, v in record.items():
            if isinstance(v, datetime):
                serialized[k] = v.isoformat()
            elif isinstance(v, torch.Tensor):
                serialized[k] = v.tolist()
            elif isinstance(v, np.ndarray):
                serialized[k] = v.tolist()
            else:
                serialized[k] = v
        return serialized
        
    def _init_dashboard(self):
        """Initialize real-time dashboard if enabled."""
        # This would initialize a web dashboard using Dash or similar
        # For now, we'll log that it's ready
        logger.info("MC Dropout monitoring dashboard initialized")


class RunningStatistics:
    """Efficient running statistics calculator."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.data = deque(maxlen=window_size)
        
    def update(self, record: Dict[str, Any]):
        """Update with new record."""
        self.data.append(record)
            
    def get_stats(self) -> Dict[str, float]:
        """Calculate current statistics."""
        if not self.data:
            return {}
            
        # Extract arrays
        confidences = [r['confidence'] for r in self.data]
        uncertainties = [r['total_uncertainty'] for r in self.data]
        
        return {
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'mean_uncertainty': np.mean(uncertainties),
            'trend_confidence': np.polyfit(range(len(confidences)), confidences, 1)[0] if len(confidences) > 1 else 0
        }


class AlertSystem:
    """Alert system for MC Dropout anomalies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_thresholds = {
            'low_confidence': config.get('low_confidence_threshold', 0.5),
            'high_uncertainty': config.get('high_uncertainty_threshold', 0.8),
            'low_convergence': config.get('low_convergence_threshold', 0.7),
            'high_outliers': config.get('high_outliers_threshold', 5)
        }
        self.alert_history = deque(maxlen=100)
        
    def check_alerts(self, consensus_result: 'ConsensusResult') -> List[Dict[str, Any]]:
        """Check for alert conditions."""
        alerts = []
        timestamp = datetime.now()
        
        # Low confidence alert
        if consensus_result.uncertainty_metrics.calibrated_confidence < self.alert_thresholds['low_confidence']:
            alerts.append({
                'type': 'low_confidence',
                'severity': 'warning',
                'message': f"Low confidence: {consensus_result.uncertainty_metrics.calibrated_confidence:.3f}",
                'timestamp': timestamp
            })
            
        # High uncertainty alert
        if consensus_result.uncertainty_metrics.total_uncertainty > self.alert_thresholds['high_uncertainty']:
            alerts.append({
                'type': 'high_uncertainty',
                'severity': 'warning',
                'message': f"High uncertainty: {consensus_result.uncertainty_metrics.total_uncertainty:.3f}",
                'timestamp': timestamp
            })
            
        # Low convergence alert
        if not consensus_result.convergence_info['converged']:
            alerts.append({
                'type': 'low_convergence',
                'severity': 'error',
                'message': f"MC sampling not converged (R-hat: {consensus_result.convergence_info['r_hat']:.3f})",
                'timestamp': timestamp
            })
            
        # High outliers alert
        if len(consensus_result.outlier_samples) > self.alert_thresholds['high_outliers']:
            alerts.append({
                'type': 'high_outliers',
                'severity': 'warning',
                'message': f"High number of outlier samples: {len(consensus_result.outlier_samples)}",
                'timestamp': timestamp
            })
            
        # Store alerts
        for alert in alerts:
            self.alert_history.append(alert)
            logger.warning(f"MC Dropout Alert: {alert['message']}")
            
        return alerts


class PerformanceProfiler:
    """Profile MC Dropout performance for optimization."""
    
    def __init__(self):
        self.timing_history = deque(maxlen=1000)
        self.memory_history = deque(maxlen=1000)
        
    def profile_evaluation(self, func, *args, **kwargs) -> Tuple[Any, Dict[str, float]]:
        """Profile a function execution."""
        import time
        import tracemalloc
        
        # Start profiling
        tracemalloc.start()
        start_time = time.perf_counter()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # End profiling
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calculate metrics
        metrics = {
            'execution_time_ms': (end_time - start_time) * 1000,
            'memory_current_mb': current / 1024 / 1024,
            'memory_peak_mb': peak / 1024 / 1024
        }
        
        # Store history
        self.timing_history.append(metrics['execution_time_ms'])
        self.memory_history.append(metrics['memory_peak_mb'])
        
        return result, metrics
        
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.timing_history:
            return {}
            
        return {
            'avg_time_ms': np.mean(self.timing_history),
            'p95_time_ms': np.percentile(self.timing_history, 95),
            'p99_time_ms': np.percentile(self.timing_history, 99),
            'avg_memory_mb': np.mean(self.memory_history),
            'peak_memory_mb': max(self.memory_history)
        }