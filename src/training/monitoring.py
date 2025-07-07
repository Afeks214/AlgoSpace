"""Training Monitoring and Evaluation Framework for MARL System.

This module implements comprehensive monitoring, logging, and evaluation
capabilities for tracking training progress and model performance.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import logging
import json
from datetime import datetime
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboardX import SummaryWriter
import mlflow
import mlflow.pytorch
from sklearn.metrics import confusion_matrix, classification_report

from src.training.environment import MultiAgentTradingEnv
from src.core.performance_metrics import PerformanceCalculator


logger = logging.getLogger(__name__)


class TrainingMonitor:
    """Comprehensive training monitoring system."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize training monitor.
        
        Args:
            config: Monitoring configuration including:
                - log_dir: Directory for logs
                - metrics: List of metrics to track
                - log_frequency: How often to log
                - use_mlflow: Whether to use MLflow
                - use_tensorboard: Whether to use TensorBoard
                - alert_thresholds: Performance thresholds for alerts
        """
        self.config = config
        self.log_dir = Path(config.get('log_dir', 'logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics to track
        self.tracked_metrics = config.get('metrics', [
            'episode_reward', 'sharpe_ratio', 'max_drawdown',
            'win_rate', 'avg_trade_duration', 'total_trades'
        ])
        
        # Logging frequency
        self.log_frequency = config.get('log_frequency', 10)
        
        # Initialize logging backends
        self.use_mlflow = config.get('mlflow', {}).get('enabled', True)
        self.use_tensorboard = config.get('tensorboard', {}).get('enabled', True)
        
        # Performance alerts
        self.alert_thresholds = config.get('alerts', {})
        
        # Metric history
        self.metric_history = defaultdict(list)
        self.episode_count = 0
        
        # Performance calculator
        self.perf_calculator = PerformanceCalculator()
        
        # Initialize backends
        self._initialize_backends()
        
        logger.info("Initialized TrainingMonitor")
    
    def _initialize_backends(self):
        """Initialize logging backends."""
        # TensorBoard
        if self.use_tensorboard:
            tb_dir = self.log_dir / 'tensorboard' / datetime.now().strftime('%Y%m%d_%H%M%S')
            self.tb_writer = SummaryWriter(str(tb_dir))
            logger.info(f"TensorBoard logging to {tb_dir}")
        
        # MLflow
        if self.use_mlflow:
            mlflow_config = self.config.get('mlflow', {})
            tracking_uri = mlflow_config.get('tracking_uri', 'http://localhost:5000')
            experiment_name = mlflow_config.get('experiment_name', 'MARL_Trading')
            
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            
            # Start MLflow run
            self.mlflow_run = mlflow.start_run()
            logger.info(f"MLflow tracking to {tracking_uri}")
    
    def log_episode(self, episode: int, episode_data: Dict[str, Any],
                   agent_metrics: Dict[str, Dict[str, float]]):
        """Log episode metrics.
        
        Args:
            episode: Episode number
            episode_data: Episode-level data
            agent_metrics: Agent-specific metrics
        """
        self.episode_count = episode
        
        # Extract metrics
        metrics = self._extract_metrics(episode_data, agent_metrics)
        
        # Update history
        for metric_name, value in metrics.items():
            self.metric_history[metric_name].append(value)
        
        # Log to backends
        if episode % self.log_frequency == 0:
            self._log_to_backends(metrics, agent_metrics)
            
            # Check alerts
            self._check_alerts(metrics)
        
        # Console logging
        if episode % (self.log_frequency * 10) == 0:
            self._log_summary(metrics)
    
    def _extract_metrics(self, episode_data: Dict[str, Any],
                        agent_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Extract metrics from episode data.
        
        Args:
            episode_data: Raw episode data
            agent_metrics: Agent-specific metrics
            
        Returns:
            Extracted metrics
        """
        metrics = {}
        
        # Episode-level metrics
        metrics['episode_reward'] = episode_data.get('total_reward', 0)
        metrics['episode_length'] = episode_data.get('episode_length', 0)
        metrics['total_trades'] = episode_data.get('total_trades', 0)
        
        # Portfolio metrics
        portfolio = episode_data.get('final_portfolio', {})
        metrics['final_value'] = portfolio.get('total_value', 0)
        metrics['total_pnl'] = portfolio.get('realized_pnl', 0)
        
        # Performance metrics
        if 'returns' in episode_data:
            returns = episode_data['returns']
            metrics['sharpe_ratio'] = self.perf_calculator.calculate_sharpe_ratio(returns)
            metrics['max_drawdown'] = self.perf_calculator.calculate_max_drawdown(returns)
            metrics['win_rate'] = self.perf_calculator.calculate_win_rate(returns)
        
        # Agent-specific aggregated metrics
        for agent_name, agent_data in agent_metrics.items():
            for metric_name, value in agent_data.items():
                metrics[f"{agent_name}_{metric_name}"] = value
        
        return metrics
    
    def _log_to_backends(self, metrics: Dict[str, float],
                        agent_metrics: Dict[str, Dict[str, float]]):
        """Log metrics to various backends.
        
        Args:
            metrics: Episode metrics
            agent_metrics: Agent-specific metrics
        """
        # TensorBoard logging
        if self.use_tensorboard:
            # Scalar metrics
            for name, value in metrics.items():
                self.tb_writer.add_scalar(f'metrics/{name}', value, self.episode_count)
            
            # Agent-specific metrics
            for agent_name, agent_data in agent_metrics.items():
                for metric_name, value in agent_data.items():
                    self.tb_writer.add_scalar(
                        f'agents/{agent_name}/{metric_name}', 
                        value, 
                        self.episode_count
                    )
            
            # Histograms for key metrics
            if self.episode_count % (self.log_frequency * 5) == 0:
                self._log_histograms()
        
        # MLflow logging
        if self.use_mlflow:
            mlflow.log_metrics(metrics, step=self.episode_count)
            
            # Log agent metrics with prefix
            for agent_name, agent_data in agent_metrics.items():
                agent_metrics_prefixed = {
                    f"{agent_name}.{k}": v for k, v in agent_data.items()
                }
                mlflow.log_metrics(agent_metrics_prefixed, step=self.episode_count)
    
    def _log_histograms(self):
        """Log histogram data to TensorBoard."""
        # Recent returns distribution
        if 'episode_reward' in self.metric_history:
            recent_rewards = self.metric_history['episode_reward'][-100:]
            if recent_rewards:
                self.tb_writer.add_histogram(
                    'distributions/episode_rewards',
                    np.array(recent_rewards),
                    self.episode_count
                )
        
        # Trade outcomes
        if 'win_rate' in self.metric_history:
            recent_win_rates = self.metric_history['win_rate'][-100:]
            if recent_win_rates:
                self.tb_writer.add_histogram(
                    'distributions/win_rates',
                    np.array(recent_win_rates),
                    self.episode_count
                )
    
    def _check_alerts(self, metrics: Dict[str, float]):
        """Check if any metrics violate alert thresholds.
        
        Args:
            metrics: Current metrics
        """
        alerts = []
        
        # Check minimum thresholds
        if 'min_sharpe_ratio' in self.alert_thresholds:
            if metrics.get('sharpe_ratio', 0) < self.alert_thresholds['min_sharpe_ratio']:
                alerts.append(f"Sharpe ratio below threshold: {metrics.get('sharpe_ratio', 0):.2f}")
        
        if 'max_drawdown' in self.alert_thresholds:
            if abs(metrics.get('max_drawdown', 0)) > self.alert_thresholds['max_drawdown']:
                alerts.append(f"Max drawdown exceeded: {abs(metrics.get('max_drawdown', 0)):.2%}")
        
        if 'min_win_rate' in self.alert_thresholds:
            if metrics.get('win_rate', 0) < self.alert_thresholds['min_win_rate']:
                alerts.append(f"Win rate below threshold: {metrics.get('win_rate', 0):.2%}")
        
        # Log alerts
        for alert in alerts:
            logger.warning(f"PERFORMANCE ALERT: {alert}")
            if self.use_mlflow:
                mlflow.log_param(f"alert_episode_{self.episode_count}", alert)
    
    def _log_summary(self, metrics: Dict[str, float]):
        """Log summary to console.
        
        Args:
            metrics: Current metrics
        """
        summary = f"\n{'='*60}\n"
        summary += f"Episode {self.episode_count} Summary\n"
        summary += f"{'='*60}\n"
        
        for metric_name in self.tracked_metrics:
            if metric_name in metrics:
                value = metrics[metric_name]
                if isinstance(value, float):
                    summary += f"{metric_name:.<30} {value:>15.4f}\n"
                else:
                    summary += f"{metric_name:.<30} {value:>15}\n"
        
        summary += f"{'='*60}\n"
        logger.info(summary)
    
    def log_model_update(self, update_info: Dict[str, Dict[str, float]]):
        """Log model update information.
        
        Args:
            update_info: Update information for each agent
        """
        if self.use_tensorboard:
            for agent_name, info in update_info.items():
                for metric_name, value in info.items():
                    self.tb_writer.add_scalar(
                        f'training/{agent_name}/{metric_name}',
                        value,
                        self.episode_count
                    )
    
    def save_checkpoint(self, models: Dict[str, torch.nn.Module],
                       optimizers: Dict[str, torch.optim.Optimizer],
                       episode: int, is_best: bool = False):
        """Save training checkpoint.
        
        Args:
            models: Agent models
            optimizers: Agent optimizers
            episode: Current episode
            is_best: Whether this is the best model
        """
        checkpoint_dir = self.log_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Checkpoint data
        checkpoint = {
            'episode': episode,
            'metric_history': dict(self.metric_history),
            'models': {name: model.state_dict() for name, model in models.items()},
            'optimizers': {name: opt.state_dict() for name, opt in optimizers.items()},
            'config': self.config
        }
        
        # Save checkpoint
        if is_best:
            checkpoint_path = checkpoint_dir / 'best_checkpoint.pt'
        else:
            checkpoint_path = checkpoint_dir / f'checkpoint_ep{episode}.pt'
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Log to MLflow
        if self.use_mlflow and is_best:
            mlflow.pytorch.log_model(models, "best_models")
    
    def plot_training_curves(self):
        """Generate and save training curves."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot each tracked metric
        for idx, metric_name in enumerate(self.tracked_metrics[:6]):
            if metric_name in self.metric_history:
                ax = axes[idx]
                data = self.metric_history[metric_name]
                
                # Smooth data
                window = min(100, len(data) // 10)
                if window > 1:
                    smoothed = pd.Series(data).rolling(window).mean()
                    ax.plot(smoothed, label='Smoothed', linewidth=2)
                    ax.plot(data, alpha=0.3, label='Raw')
                else:
                    ax.plot(data)
                
                ax.set_title(metric_name.replace('_', ' ').title())
                ax.set_xlabel('Episode')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = self.log_dir / 'training_curves.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved training curves to {fig_path}")
        
        # Log to MLflow
        if self.use_mlflow:
            mlflow.log_artifact(str(fig_path))
    
    def close(self):
        """Close monitoring resources."""
        # Save final plots
        self.plot_training_curves()
        
        # Close TensorBoard
        if self.use_tensorboard:
            self.tb_writer.close()
        
        # End MLflow run
        if self.use_mlflow:
            mlflow.end_run()
        
        logger.info("TrainingMonitor closed")


class ModelEvaluator:
    """Comprehensive model evaluation system."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.n_eval_episodes = config.get('n_episodes', 100)
        self.metrics_to_compute = config.get('metrics', [
            'total_return', 'sharpe_ratio', 'sortino_ratio',
            'max_drawdown', 'calmar_ratio', 'win_rate', 'profit_factor'
        ])
        
        # Performance calculator
        self.perf_calculator = PerformanceCalculator()
        
        logger.info(f"Initialized ModelEvaluator with {self.n_eval_episodes} episodes")
    
    def evaluate(self, agents: Dict[str, Any], env: MultiAgentTradingEnv,
                render: bool = False) -> Dict[str, Any]:
        """Evaluate trained agents.
        
        Args:
            agents: Dictionary of trained agents
            env: Trading environment
            render: Whether to render episodes
            
        Returns:
            Evaluation results
        """
        logger.info("Starting model evaluation...")
        
        # Set agents to eval mode
        for agent in agents.values():
            if hasattr(agent, 'eval'):
                agent.eval()
        
        # Collect evaluation data
        eval_data = self._collect_evaluation_data(agents, env, render)
        
        # Compute metrics
        metrics = self._compute_metrics(eval_data)
        
        # Generate report
        report = self._generate_report(metrics, eval_data)
        
        # Set agents back to train mode
        for agent in agents.values():
            if hasattr(agent, 'train'):
                agent.train()
        
        return report
    
    def _collect_evaluation_data(self, agents: Dict[str, Any], 
                                env: MultiAgentTradingEnv,
                                render: bool) -> Dict[str, List]:
        """Collect data from evaluation episodes.
        
        Args:
            agents: Trained agents
            env: Environment
            render: Whether to render
            
        Returns:
            Collected evaluation data
        """
        eval_data = {
            'returns': [],
            'positions': [],
            'trades': [],
            'portfolio_values': [],
            'actions': defaultdict(list),
            'rewards': defaultdict(list)
        }
        
        for episode in range(self.n_eval_episodes):
            observations = env.reset()
            episode_returns = []
            episode_positions = []
            episode_trades = []
            done = False
            
            while not done:
                # Get actions from agents
                actions = {}
                for agent_name, agent in agents.items():
                    obs = observations[agent_name]
                    with torch.no_grad():
                        action = agent.act(obs, deterministic=True)
                    actions[agent_name] = action
                    eval_data['actions'][agent_name].append(action)
                
                # Step environment
                observations, rewards, done, info = env.step(actions)
                
                # Collect data
                episode_returns.append(info.get('return', 0))
                episode_positions.append(info.get('position', 0))
                if info.get('trade_executed', False):
                    episode_trades.append(info['trade_result'])
                
                # Store rewards
                for agent_name, reward in rewards.items():
                    eval_data['rewards'][agent_name].append(reward)
                
                if render:
                    env.render()
            
            # Store episode data
            eval_data['returns'].append(episode_returns)
            eval_data['positions'].append(episode_positions)
            eval_data['trades'].append(episode_trades)
            eval_data['portfolio_values'].append(info.get('portfolio_value', 0))
            
            if (episode + 1) % 10 == 0:
                logger.info(f"Evaluated {episode + 1}/{self.n_eval_episodes} episodes")
        
        return eval_data
    
    def _compute_metrics(self, eval_data: Dict[str, List]) -> Dict[str, float]:
        """Compute evaluation metrics.
        
        Args:
            eval_data: Collected evaluation data
            
        Returns:
            Computed metrics
        """
        metrics = {}
        
        # Flatten returns
        all_returns = []
        for episode_returns in eval_data['returns']:
            all_returns.extend(episode_returns)
        
        # Basic metrics
        metrics['total_episodes'] = len(eval_data['returns'])
        metrics['total_trades'] = sum(len(trades) for trades in eval_data['trades'])
        metrics['avg_trades_per_episode'] = metrics['total_trades'] / metrics['total_episodes']
        
        # Performance metrics
        if all_returns:
            returns_array = np.array(all_returns)
            
            metrics['total_return'] = np.sum(returns_array)
            metrics['avg_return'] = np.mean(returns_array)
            metrics['std_return'] = np.std(returns_array)
            
            # Risk metrics
            metrics['sharpe_ratio'] = self.perf_calculator.calculate_sharpe_ratio(returns_array)
            metrics['sortino_ratio'] = self.perf_calculator.calculate_sortino_ratio(returns_array)
            metrics['max_drawdown'] = self.perf_calculator.calculate_max_drawdown(returns_array)
            metrics['calmar_ratio'] = self.perf_calculator.calculate_calmar_ratio(returns_array)
        
        # Trading metrics
        all_trades = []
        for episode_trades in eval_data['trades']:
            all_trades.extend(episode_trades)
        
        if all_trades:
            winning_trades = [t for t in all_trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in all_trades if t.get('pnl', 0) < 0]
            
            metrics['win_rate'] = len(winning_trades) / len(all_trades)
            
            if winning_trades and losing_trades:
                avg_win = np.mean([t['pnl'] for t in winning_trades])
                avg_loss = abs(np.mean([t['pnl'] for t in losing_trades]))
                metrics['profit_factor'] = avg_win / avg_loss if avg_loss > 0 else np.inf
            else:
                metrics['profit_factor'] = 0
        
        # Agent-specific metrics
        for agent_name, rewards in eval_data['rewards'].items():
            metrics[f'{agent_name}_avg_reward'] = np.mean(rewards)
            metrics[f'{agent_name}_total_reward'] = np.sum(rewards)
        
        return metrics
    
    def _generate_report(self, metrics: Dict[str, float], 
                        eval_data: Dict[str, List]) -> Dict[str, Any]:
        """Generate evaluation report.
        
        Args:
            metrics: Computed metrics
            eval_data: Raw evaluation data
            
        Returns:
            Comprehensive evaluation report
        """
        report = {
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'summary': self._generate_summary(metrics),
            'detailed_analysis': self._detailed_analysis(eval_data)
        }
        
        # Save report
        report_path = Path(self.config.get('output_path', 'evaluation_results'))
        report_path.mkdir(parents=True, exist_ok=True)
        
        report_file = report_path / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to {report_file}")
        
        # Generate visualizations
        self._generate_visualizations(eval_data, report_path)
        
        return report
    
    def _generate_summary(self, metrics: Dict[str, float]) -> str:
        """Generate text summary of evaluation results.
        
        Args:
            metrics: Computed metrics
            
        Returns:
            Summary text
        """
        summary = "Model Evaluation Summary\n"
        summary += "=" * 50 + "\n\n"
        
        # Key metrics
        key_metrics = ['sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor']
        for metric in key_metrics:
            if metric in metrics:
                value = metrics[metric]
                if metric == 'max_drawdown':
                    summary += f"{metric:.<30} {value:>15.2%}\n"
                elif metric == 'win_rate':
                    summary += f"{metric:.<30} {value:>15.2%}\n"
                else:
                    summary += f"{metric:.<30} {value:>15.4f}\n"
        
        summary += "\n" + "=" * 50 + "\n"
        
        return summary
    
    def _detailed_analysis(self, eval_data: Dict[str, List]) -> Dict[str, Any]:
        """Perform detailed analysis of evaluation data.
        
        Args:
            eval_data: Raw evaluation data
            
        Returns:
            Detailed analysis results
        """
        analysis = {}
        
        # Position analysis
        all_positions = []
        for episode_positions in eval_data['positions']:
            all_positions.extend(episode_positions)
        
        if all_positions:
            analysis['position_stats'] = {
                'mean': np.mean(all_positions),
                'std': np.std(all_positions),
                'min': np.min(all_positions),
                'max': np.max(all_positions),
                'long_ratio': sum(1 for p in all_positions if p > 0) / len(all_positions),
                'short_ratio': sum(1 for p in all_positions if p < 0) / len(all_positions),
                'neutral_ratio': sum(1 for p in all_positions if p == 0) / len(all_positions)
            }
        
        # Trade analysis
        all_trades = []
        for episode_trades in eval_data['trades']:
            all_trades.extend(episode_trades)
        
        if all_trades:
            trade_pnls = [t.get('pnl', 0) for t in all_trades]
            trade_durations = [t.get('duration', 0) for t in all_trades]
            
            analysis['trade_stats'] = {
                'total_trades': len(all_trades),
                'avg_pnl': np.mean(trade_pnls),
                'std_pnl': np.std(trade_pnls),
                'max_win': np.max(trade_pnls) if trade_pnls else 0,
                'max_loss': np.min(trade_pnls) if trade_pnls else 0,
                'avg_duration': np.mean(trade_durations) if trade_durations else 0
            }
        
        # Action analysis
        for agent_name, actions in eval_data['actions'].items():
            if actions:
                action_array = np.array(actions)
                if len(action_array.shape) == 1:
                    # Discrete actions
                    unique, counts = np.unique(action_array, return_counts=True)
                    analysis[f'{agent_name}_action_distribution'] = {
                        int(action): int(count) for action, count in zip(unique, counts)
                    }
                else:
                    # Continuous actions
                    analysis[f'{agent_name}_action_stats'] = {
                        'mean': action_array.mean(axis=0).tolist(),
                        'std': action_array.std(axis=0).tolist()
                    }
        
        return analysis
    
    def _generate_visualizations(self, eval_data: Dict[str, List], output_path: Path):
        """Generate evaluation visualizations.
        
        Args:
            eval_data: Evaluation data
            output_path: Path to save visualizations
        """
        # Portfolio value over time
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Portfolio values
        ax = axes[0, 0]
        portfolio_values = eval_data['portfolio_values']
        ax.plot(portfolio_values)
        ax.set_title('Portfolio Value Evolution')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Portfolio Value')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Returns distribution
        ax = axes[0, 1]
        all_returns = []
        for episode_returns in eval_data['returns']:
            all_returns.extend(episode_returns)
        
        if all_returns:
            ax.hist(all_returns, bins=50, alpha=0.7, edgecolor='black')
            ax.set_title('Returns Distribution')
            ax.set_xlabel('Return')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Position distribution
        ax = axes[1, 0]
        all_positions = []
        for episode_positions in eval_data['positions']:
            all_positions.extend(episode_positions)
        
        if all_positions:
            position_types = ['Long', 'Neutral', 'Short']
            position_counts = [
                sum(1 for p in all_positions if p > 0),
                sum(1 for p in all_positions if p == 0),
                sum(1 for p in all_positions if p < 0)
            ]
            ax.pie(position_counts, labels=position_types, autopct='%1.1f%%')
            ax.set_title('Position Distribution')
        
        # Plot 4: Trade outcomes
        ax = axes[1, 1]
        all_trades = []
        for episode_trades in eval_data['trades']:
            all_trades.extend(episode_trades)
        
        if all_trades:
            trade_pnls = [t.get('pnl', 0) for t in all_trades]
            wins = [pnl for pnl in trade_pnls if pnl > 0]
            losses = [pnl for pnl in trade_pnls if pnl < 0]
            
            ax.hist([wins, losses], label=['Wins', 'Losses'], 
                   bins=30, alpha=0.7, color=['green', 'red'])
            ax.set_title('Trade Outcome Distribution')
            ax.set_xlabel('P&L')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = output_path / 'evaluation_plots.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved evaluation plots to {fig_path}")


class BacktestEngine:
    """Backtesting engine for trained models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize backtest engine.
        
        Args:
            config: Backtesting configuration
        """
        self.config = config
        self.include_costs = config.get('include_costs', True)
        self.slippage_model = config.get('slippage', {})
        self.position_limits = config.get('position_limits', {})
        
        logger.info("Initialized BacktestEngine")
    
    def backtest(self, agents: Dict[str, Any], data: pd.DataFrame,
                initial_capital: float = 100000) -> Dict[str, Any]:
        """Run backtest on historical data.
        
        Args:
            agents: Trained agents
            data: Historical market data
            initial_capital: Starting capital
            
        Returns:
            Backtest results
        """
        logger.info(f"Starting backtest on {len(data)} bars of data")
        
        # Initialize portfolio
        portfolio = {
            'cash': initial_capital,
            'position': 0.0,
            'equity_curve': [initial_capital],
            'trades': [],
            'timestamps': []
        }
        
        # Run backtest
        for i in range(100, len(data)):  # Start after warmup period
            # Prepare observations for agents
            observations = self._prepare_observations(data, i)
            
            # Get agent decisions
            actions = {}
            for agent_name, agent in agents.items():
                with torch.no_grad():
                    action = agent.act(observations[agent_name], deterministic=True)
                actions[agent_name] = action
            
            # Process trading decision
            trade_decision = self._aggregate_decisions(actions)
            
            # Execute trade
            if trade_decision['execute']:
                trade_result = self._execute_trade(
                    portfolio, trade_decision, data.iloc[i]
                )
                if trade_result['executed']:
                    portfolio['trades'].append(trade_result)
            
            # Update portfolio value
            current_price = data.iloc[i]['close']
            portfolio_value = self._calculate_portfolio_value(portfolio, current_price)
            portfolio['equity_curve'].append(portfolio_value)
            portfolio['timestamps'].append(data.index[i])
        
        # Calculate performance metrics
        results = self._calculate_backtest_metrics(portfolio, data)
        
        # Generate backtest report
        self._generate_backtest_report(results, portfolio)
        
        return results
    
    def _prepare_observations(self, data: pd.DataFrame, index: int) -> Dict[str, np.ndarray]:
        """Prepare observations for agents from historical data.
        
        Args:
            data: Historical data
            index: Current bar index
            
        Returns:
            Observations for each agent
        """
        # This would implement the actual data preparation
        # Similar to how live data is prepared for agents
        observations = {}
        
        # Simplified example - would need full matrix generation
        observations['regime'] = np.random.randn(96, 12).astype(np.float32)
        observations['structure'] = np.random.randn(48, 8).astype(np.float32)
        observations['tactical'] = np.random.randn(60, 7).astype(np.float32)
        observations['risk'] = {
            'matrices': np.random.randn(204, 25).astype(np.float32),
            'portfolio': np.random.randn(10).astype(np.float32)
        }
        
        return observations
    
    def _aggregate_decisions(self, actions: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate agent decisions into trading decision.
        
        Args:
            actions: Agent actions
            
        Returns:
            Aggregated trading decision
        """
        # Simplified decision aggregation
        # In practice, this would implement the full decision logic
        
        decision = {
            'execute': np.random.random() > 0.95,  # 5% trade probability
            'direction': np.random.choice(['long', 'short']),
            'size': np.random.uniform(0.1, 0.5)
        }
        
        return decision
    
    def _execute_trade(self, portfolio: Dict[str, Any], 
                      decision: Dict[str, Any],
                      bar_data: pd.Series) -> Dict[str, Any]:
        """Execute trade in backtest.
        
        Args:
            portfolio: Current portfolio state
            decision: Trading decision
            bar_data: Current bar data
            
        Returns:
            Trade execution result
        """
        result = {
            'executed': False,
            'timestamp': bar_data.name,
            'price': bar_data['close'],
            'size': 0,
            'direction': decision['direction'],
            'cost': 0,
            'pnl': 0
        }
        
        # Apply slippage
        if self.slippage_model.get('type') == 'percentage':
            slippage = self.slippage_model.get('value', 0.0005)
            if decision['direction'] == 'long':
                result['price'] *= (1 + slippage)
            else:
                result['price'] *= (1 - slippage)
        
        # Check position limits
        max_position = self.position_limits.get('max_position', 1.0)
        if abs(portfolio['position']) + decision['size'] > max_position:
            return result
        
        # Execute trade
        result['executed'] = True
        result['size'] = decision['size']
        
        # Calculate cost
        if self.include_costs:
            result['cost'] = result['size'] * result['price'] * 0.001  # 0.1% cost
        
        # Update portfolio
        if decision['direction'] == 'long':
            portfolio['position'] += result['size']
        else:
            portfolio['position'] -= result['size']
        
        portfolio['cash'] -= result['size'] * result['price'] + result['cost']
        
        return result
    
    def _calculate_portfolio_value(self, portfolio: Dict[str, Any], 
                                 current_price: float) -> float:
        """Calculate current portfolio value.
        
        Args:
            portfolio: Portfolio state
            current_price: Current market price
            
        Returns:
            Total portfolio value
        """
        return portfolio['cash'] + portfolio['position'] * current_price
    
    def _calculate_backtest_metrics(self, portfolio: Dict[str, Any],
                                  data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate backtest performance metrics.
        
        Args:
            portfolio: Final portfolio state
            data: Market data
            
        Returns:
            Performance metrics
        """
        equity_curve = np.array(portfolio['equity_curve'])
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        metrics = {
            'total_return': (equity_curve[-1] - equity_curve[0]) / equity_curve[0],
            'sharpe_ratio': PerformanceCalculator.calculate_sharpe_ratio(returns),
            'max_drawdown': PerformanceCalculator.calculate_max_drawdown(equity_curve),
            'total_trades': len(portfolio['trades']),
            'final_value': equity_curve[-1]
        }
        
        # Trade analysis
        if portfolio['trades']:
            trade_pnls = [t['pnl'] for t in portfolio['trades'] if 'pnl' in t]
            if trade_pnls:
                winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
                metrics['win_rate'] = len(winning_trades) / len(trade_pnls)
                
                if winning_trades:
                    losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
                    if losing_trades:
                        metrics['profit_factor'] = (
                            sum(winning_trades) / abs(sum(losing_trades))
                        )
        
        return metrics
    
    def _generate_backtest_report(self, results: Dict[str, Any],
                                portfolio: Dict[str, Any]):
        """Generate backtest report with visualizations.
        
        Args:
            results: Backtest results
            portfolio: Portfolio data
        """
        # Create report directory
        report_dir = Path(self.config.get('output_path', 'backtest_results'))
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        report_file = report_dir / f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate equity curve plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Equity curve
        ax1.plot(portfolio['timestamps'], portfolio['equity_curve'])
        ax1.set_title('Equity Curve')
        ax1.set_ylabel('Portfolio Value')
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        equity_curve = np.array(portfolio['equity_curve'])
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        
        ax2.fill_between(portfolio['timestamps'], drawdown, 0, alpha=0.3, color='red')
        ax2.set_title('Drawdown')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown %')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = report_dir / 'backtest_equity_curve.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Backtest report saved to {report_dir}")


class TacticalEmbedderMonitor:
    """Monitor tactical embedder behavior and momentum patterns."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        from src.agents.main_core.models import MomentumAnalyzer
        self.momentum_analyzer = MomentumAnalyzer(window_size)
        self.performance_history = []
        
    def record_inference(
        self,
        embeddings: torch.Tensor,
        uncertainties: torch.Tensor,
        attention_weights: torch.Tensor,
        lstm_states: Optional[List[torch.Tensor]] = None
    ):
        """Record inference statistics and analyze momentum."""
        # Analyze momentum patterns
        momentum_metrics = self.momentum_analyzer.analyze(
            embeddings, 
            attention_weights,
            lstm_states or []
        )
        
        # Record performance metrics
        self.performance_history.append({
            'timestamp': datetime.now(),
            'embedding_norm': embeddings.norm(dim=-1).mean().item(),
            'uncertainty_mean': uncertainties.mean().item(),
            'uncertainty_std': uncertainties.std().item(),
            'momentum_clarity': momentum_metrics['momentum_clarity'],
            'recent_focus_ratio': momentum_metrics['recent_focus_ratio'],
            'patterns': momentum_metrics.get('patterns', [])
        })
        
        # Keep window size
        if len(self.performance_history) > self.window_size:
            self.performance_history.pop(0)
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get current monitoring statistics."""
        if not self.performance_history:
            return {}
            
        recent = self.performance_history[-100:]
        
        # Pattern frequency
        all_patterns = []
        for record in recent:
            all_patterns.extend(record.get('patterns', []))
            
        pattern_counts = {}
        for pattern in all_patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
        return {
            'avg_momentum_clarity': np.mean([r['momentum_clarity'] for r in recent]),
            'avg_recent_focus': np.mean([r['recent_focus_ratio'] for r in recent]),
            'avg_uncertainty': np.mean([r['uncertainty_mean'] for r in recent]),
            'pattern_frequencies': pattern_counts,
            'embedding_stability': 1.0 / (np.std([r['embedding_norm'] for r in recent]) + 1e-8)
        }
        
    def create_momentum_report(self) -> str:
        """Generate human-readable momentum analysis report."""
        stats = self.get_statistics()
        
        report = "Tactical Embedder Momentum Report\n"
        report += "=" * 40 + "\n\n"
        
        report += f"Momentum Clarity: {stats.get('avg_momentum_clarity', 0):.3f}\n"
        report += f"Recent Focus Ratio: {stats.get('avg_recent_focus', 0):.3f}\n"
        report += f"Average Uncertainty: {stats.get('avg_uncertainty', 0):.3f}\n"
        report += f"Embedding Stability: {stats.get('embedding_stability', 0):.3f}\n\n"
        
        report += "Pattern Frequencies:\n"
        for pattern, count in stats.get('pattern_frequencies', {}).items():
            report += f"  - {pattern}: {count} occurrences\n"
            
        return report