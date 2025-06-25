"""Hyperparameter Optimization and Distributed Training for MARL System.

This module implements automated hyperparameter optimization using Optuna
and distributed training capabilities for efficient model training.
"""

import os
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler, RandomSampler, GridSampler
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
import yaml
import logging
import mlflow
from datetime import datetime
import multiprocessing as mp
from functools import partial

from src.training.marl_trainer import MAPPOTrainer
from src.training.environment import MultiAgentTradingEnv


logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """Hyperparameter optimization for MARL training using Optuna."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize hyperparameter optimizer.
        
        Args:
            config: Optimization configuration including:
                - n_trials: Number of optimization trials
                - sampler: Sampling strategy (tpe, random, grid)
                - search_space: Parameter search space definitions
                - objective_metric: Metric to optimize
                - direction: minimize or maximize
                - study_name: Name for the Optuna study
                - storage: Database URL for distributed optimization
        """
        self.config = config
        self.n_trials = config.get('n_trials', 100)
        self.objective_metric = config.get('objective_metric', 'sharpe_ratio')
        self.direction = config.get('direction', 'maximize')
        
        # Initialize sampler
        sampler_type = config.get('sampler', 'tpe')
        if sampler_type == 'tpe':
            self.sampler = TPESampler(seed=config.get('seed', 42))
        elif sampler_type == 'random':
            self.sampler = RandomSampler(seed=config.get('seed', 42))
        elif sampler_type == 'grid':
            search_space = config.get('search_space', {})
            self.sampler = GridSampler(search_space)
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")
        
        # Create study
        study_name = config.get('study_name', f'marl_opt_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        storage = config.get('storage', None)
        
        self.study = optuna.create_study(
            study_name=study_name,
            direction=self.direction,
            sampler=self.sampler,
            storage=storage,
            load_if_exists=True
        )
        
        # Search space configuration
        self.search_space = config.get('search_space', {})
        
        # Training configuration template
        self.base_config = config.get('base_config', {})
        
        # Best trial tracking
        self.best_trial = None
        self.best_value = float('-inf') if self.direction == 'maximize' else float('inf')
        
        logger.info(f"Initialized HyperparameterOptimizer with {self.n_trials} trials")
    
    def objective(self, trial: Trial) -> float:
        """Objective function for optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Objective metric value
        """
        # Sample hyperparameters
        hyperparams = self._sample_hyperparameters(trial)
        
        # Update training configuration
        train_config = self._update_config(self.base_config, hyperparams)
        
        # Log trial parameters
        logger.info(f"Trial {trial.number}: {hyperparams}")
        
        try:
            # Train model with sampled hyperparameters
            trainer = MAPPOTrainer(train_config)
            
            # Short training run for hyperparameter evaluation
            eval_episodes = train_config.get('hyperopt_eval_episodes', 1000)
            trainer.train(n_episodes=eval_episodes, save_freq=eval_episodes+1)
            
            # Evaluate performance
            eval_metrics = trainer.evaluate(n_episodes=10)
            
            # Get objective metric
            objective_value = eval_metrics.get(self.objective_metric, 0.0)
            
            # Report intermediate values for pruning
            trial.report(objective_value, eval_episodes)
            
            # Check if trial should be pruned
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned")
                raise optuna.TrialPruned()
            
            # Log to MLflow if enabled
            if self.config.get('log_to_mlflow', True):
                self._log_trial_to_mlflow(trial, hyperparams, eval_metrics)
            
            logger.info(f"Trial {trial.number} completed: {self.objective_metric}={objective_value:.4f}")
            
            return objective_value
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return float('-inf') if self.direction == 'maximize' else float('inf')
    
    def _sample_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        """Sample hyperparameters based on search space.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Sampled hyperparameters
        """
        hyperparams = {}
        
        for param_name, param_config in self.search_space.items():
            param_type = param_config['type']
            
            if param_type == 'uniform':
                value = trial.suggest_uniform(
                    param_name,
                    param_config['low'],
                    param_config['high']
                )
            elif param_type == 'loguniform':
                value = trial.suggest_loguniform(
                    param_name,
                    param_config['low'],
                    param_config['high']
                )
            elif param_type == 'int':
                value = trial.suggest_int(
                    param_name,
                    param_config['low'],
                    param_config['high']
                )
            elif param_type == 'categorical':
                value = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
            
            hyperparams[param_name] = value
        
        return hyperparams
    
    def _update_config(self, base_config: Dict[str, Any], 
                      hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration with sampled hyperparameters.
        
        Args:
            base_config: Base training configuration
            hyperparams: Sampled hyperparameters
            
        Returns:
            Updated configuration
        """
        config = base_config.copy()
        
        # Map hyperparameters to config structure
        param_mapping = {
            'learning_rate': ['mappo', 'learning_rate'],
            'ppo_epochs': ['mappo', 'ppo_epochs'],
            'clip_param': ['mappo', 'clip_param'],
            'entropy_coef': ['mappo', 'entropy_coef'],
            'gae_lambda': ['mappo', 'gae_lambda'],
            'batch_size': ['mappo', 'batch_size'],
            'value_loss_coef': ['mappo', 'value_loss_coef'],
            'max_grad_norm': ['mappo', 'max_grad_norm'],
            # Add more mappings as needed
        }
        
        for param_name, param_value in hyperparams.items():
            if param_name in param_mapping:
                # Navigate config hierarchy
                config_path = param_mapping[param_name]
                current = config
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[config_path[-1]] = param_value
        
        return config
    
    def _log_trial_to_mlflow(self, trial: Trial, hyperparams: Dict[str, Any],
                            metrics: Dict[str, float]):
        """Log trial results to MLflow.
        
        Args:
            trial: Optuna trial
            hyperparams: Trial hyperparameters
            metrics: Evaluation metrics
        """
        with mlflow.start_run(nested=True):
            # Log hyperparameters
            mlflow.log_params(hyperparams)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log trial metadata
            mlflow.log_param("trial_number", trial.number)
            mlflow.log_param("trial_state", trial.state.name)
    
    def optimize(self, n_trials: Optional[int] = None,
                n_jobs: int = 1,
                callbacks: Optional[List[Callable]] = None) -> optuna.Study:
        """Run hyperparameter optimization.
        
        Args:
            n_trials: Number of trials (overrides config)
            n_jobs: Number of parallel jobs
            callbacks: List of callback functions
            
        Returns:
            Completed Optuna study
        """
        n_trials = n_trials or self.n_trials
        
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        
        # Add default callbacks
        if callbacks is None:
            callbacks = []
        
        # Add pruning callback
        callbacks.append(
            optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=100
            )
        )
        
        # Run optimization
        self.study.optimize(
            self.objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            callbacks=callbacks,
            show_progress_bar=True
        )
        
        # Get best trial
        self.best_trial = self.study.best_trial
        self.best_value = self.study.best_value
        
        logger.info(f"Optimization completed. Best value: {self.best_value:.4f}")
        logger.info(f"Best parameters: {self.best_trial.params}")
        
        # Save results
        self._save_results()
        
        return self.study
    
    def _save_results(self):
        """Save optimization results."""
        results = {
            'best_value': self.best_value,
            'best_params': self.best_trial.params,
            'best_trial_number': self.best_trial.number,
            'n_trials': len(self.study.trials),
            'study_name': self.study.study_name,
            'optimization_history': [
                {
                    'number': t.number,
                    'value': t.value,
                    'params': t.params,
                    'state': t.state.name
                }
                for t in self.study.trials
            ]
        }
        
        # Save to file
        output_path = Path(self.config.get('output_path', 'optimization_results'))
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_file = output_path / f"{self.study.study_name}_results.yaml"
        with open(results_file, 'w') as f:
            yaml.dump(results, f)
        
        logger.info(f"Optimization results saved to {results_file}")
    
    def visualize_results(self):
        """Generate visualization of optimization results."""
        try:
            import optuna.visualization as vis
            
            # Create visualizations
            fig_importance = vis.plot_param_importances(self.study)
            fig_history = vis.plot_optimization_history(self.study)
            fig_parallel = vis.plot_parallel_coordinate(self.study)
            fig_slice = vis.plot_slice(self.study)
            
            # Save figures
            output_path = Path(self.config.get('output_path', 'optimization_results'))
            output_path.mkdir(parents=True, exist_ok=True)
            
            fig_importance.write_html(output_path / "param_importance.html")
            fig_history.write_html(output_path / "optimization_history.html")
            fig_parallel.write_html(output_path / "parallel_coordinate.html")
            fig_slice.write_html(output_path / "slice_plot.html")
            
            logger.info("Visualization plots saved")
            
        except ImportError:
            logger.warning("Plotly not installed. Skipping visualization.")


class DistributedTrainer:
    """Distributed training manager for MARL system."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize distributed trainer.
        
        Args:
            config: Distributed training configuration
        """
        self.config = config
        self.backend = config.get('backend', 'nccl')
        self.world_size = config.get('world_size', torch.cuda.device_count())
        self.master_addr = config.get('master_addr', 'localhost')
        self.master_port = config.get('master_port', '12355')
        
        logger.info(f"Initialized DistributedTrainer with world_size={self.world_size}")
    
    def setup(self, rank: int):
        """Setup distributed training environment.
        
        Args:
            rank: Process rank
        """
        os.environ['MASTER_ADDR'] = self.master_addr
        os.environ['MASTER_PORT'] = self.master_port
        
        # Initialize process group
        dist.init_process_group(
            backend=self.backend,
            rank=rank,
            world_size=self.world_size
        )
        
        # Set device
        torch.cuda.set_device(rank)
        
        logger.info(f"Process {rank} initialized")
    
    def cleanup(self):
        """Cleanup distributed training."""
        dist.destroy_process_group()
    
    def train_distributed(self, rank: int, config: Dict[str, Any]):
        """Train model in distributed setting.
        
        Args:
            rank: Process rank
            config: Training configuration
        """
        # Setup process
        self.setup(rank)
        
        try:
            # Update config for distributed training
            config['device'] = f'cuda:{rank}'
            config['distributed'] = {
                'enabled': True,
                'rank': rank,
                'world_size': self.world_size
            }
            
            # Create trainer
            trainer = MAPPOTrainer(config)
            
            # Wrap models in DDP
            for agent_name, agent in trainer.agents.items():
                trainer.agents[agent_name] = DDP(
                    agent,
                    device_ids=[rank],
                    output_device=rank
                )
            
            # Distributed sampler for data
            if hasattr(trainer, 'data_loader'):
                trainer.data_loader.sampler = DistributedSampler(
                    trainer.data_loader.dataset,
                    num_replicas=self.world_size,
                    rank=rank
                )
            
            # Train model
            trainer.train(
                n_episodes=config['training']['n_episodes'],
                save_freq=config['training']['save_frequency']
            )
            
            # Save model only from rank 0
            if rank == 0:
                trainer.save_models(config['training']['n_episodes'], is_final=True)
            
        finally:
            self.cleanup()
    
    def launch(self, config: Dict[str, Any]):
        """Launch distributed training.
        
        Args:
            config: Training configuration
        """
        mp.spawn(
            self.train_distributed,
            args=(config,),
            nprocs=self.world_size,
            join=True
        )
        
        logger.info("Distributed training completed")


class MixedPrecisionTrainer:
    """Mixed precision training utilities."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize mixed precision trainer.
        
        Args:
            config: Mixed precision configuration
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.opt_level = config.get('opt_level', 'O1')
        
        if self.enabled:
            try:
                from apex import amp
                self.amp = amp
                logger.info(f"Mixed precision training enabled with opt_level={self.opt_level}")
            except ImportError:
                logger.warning("apex not installed. Disabling mixed precision training.")
                self.enabled = False
    
    def initialize(self, models: Dict[str, torch.nn.Module],
                  optimizers: Dict[str, torch.optim.Optimizer]) -> Tuple[Dict, Dict]:
        """Initialize models and optimizers for mixed precision.
        
        Args:
            models: Dictionary of models
            optimizers: Dictionary of optimizers
            
        Returns:
            Updated models and optimizers
        """
        if not self.enabled:
            return models, optimizers
        
        # Initialize each model-optimizer pair
        amp_models = {}
        amp_optimizers = {}
        
        for agent_name in models.keys():
            model = models[agent_name]
            optimizer = optimizers[agent_name]
            
            amp_model, amp_optimizer = self.amp.initialize(
                model,
                optimizer,
                opt_level=self.opt_level,
                loss_scale='dynamic'
            )
            
            amp_models[agent_name] = amp_model
            amp_optimizers[agent_name] = amp_optimizer
        
        return amp_models, amp_optimizers
    
    def scale_loss(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> torch.Tensor:
        """Scale loss for mixed precision training.
        
        Args:
            loss: Loss tensor
            optimizer: Optimizer
            
        Returns:
            Scaled loss
        """
        if not self.enabled:
            return loss
        
        with self.amp.scale_loss(loss, optimizer) as scaled_loss:
            return scaled_loss


class TrainingScheduler:
    """Learning rate and training schedulers."""
    
    def __init__(self, config: Dict[str, Any], optimizers: Dict[str, torch.optim.Optimizer]):
        """Initialize training schedulers.
        
        Args:
            config: Scheduler configuration
            optimizers: Dictionary of optimizers
        """
        self.config = config
        self.schedulers = {}
        
        # Create schedulers for each optimizer
        for agent_name, optimizer in optimizers.items():
            scheduler_config = config.get(agent_name, config.get('default', {}))
            scheduler_type = scheduler_config.get('type', 'cosine')
            
            if scheduler_type == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=scheduler_config.get('T_max', 1000),
                    eta_min=scheduler_config.get('eta_min', 1e-6)
                )
            elif scheduler_type == 'linear':
                scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor=scheduler_config.get('end_factor', 0.1),
                    total_iters=scheduler_config.get('total_iters', 1000)
                )
            elif scheduler_type == 'exponential':
                scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer,
                    gamma=scheduler_config.get('gamma', 0.99)
                )
            elif scheduler_type == 'plateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode=scheduler_config.get('mode', 'max'),
                    factor=scheduler_config.get('factor', 0.5),
                    patience=scheduler_config.get('patience', 10),
                    min_lr=scheduler_config.get('min_lr', 1e-6)
                )
            else:
                raise ValueError(f"Unknown scheduler type: {scheduler_type}")
            
            self.schedulers[agent_name] = scheduler
        
        logger.info(f"Initialized {len(self.schedulers)} learning rate schedulers")
    
    def step(self, metrics: Optional[Dict[str, float]] = None):
        """Step all schedulers.
        
        Args:
            metrics: Metrics for plateau scheduler
        """
        for agent_name, scheduler in self.schedulers.items():
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if metrics and agent_name in metrics:
                    scheduler.step(metrics[agent_name])
            else:
                scheduler.step()
    
    def get_last_lr(self) -> Dict[str, float]:
        """Get last learning rates.
        
        Returns:
            Dictionary of last learning rates
        """
        last_lrs = {}
        for agent_name, scheduler in self.schedulers.items():
            last_lrs[agent_name] = scheduler.get_last_lr()[0]
        return last_lrs


class EarlyStopping:
    """Early stopping handler for training."""
    
    def __init__(self, patience: int = 100, min_delta: float = 0.001,
                 mode: str = 'max'):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs with no improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.improvement_sign = -1
        else:
            self.improvement_sign = 1
    
    def __call__(self, score: float) -> bool:
        """Check if training should stop.
        
        Args:
            score: Current score
            
        Returns:
            Whether to stop training
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.improvement_sign * (score - self.best_score) > self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs")
        
        return self.early_stop
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False