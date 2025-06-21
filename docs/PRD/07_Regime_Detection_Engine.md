# Product Requirements Document (PRD): Regime Detection Engine (RDE)

**Document Version**: 2.0  
**Date**: 2025-06-19  
**Status**: Refined  
**Component**: Regime Detection Engine (RDE)  

## 1.0 Overview

### 1.1 Purpose
The Regime Detection Engine (RDE) is a sophisticated unsupervised machine learning component that autonomously learns and generates continuous, low-dimensional representations of market states. Using a hybrid Transformer + Variational Autoencoder (VAE) architecture, it creates a "market map" that enables the Main MARL Core to adapt its trading strategies to different market regimes without requiring predefined labels or manual feature engineering.

### 1.2 Scope

**In Scope:**
- Hybrid Transformer + VAE architecture implementation
- Unsupervised learning on temporal market feature sequences
- Continuous regime vector generation for market state representation
- End-to-end model training with reconstruction and KL divergence losses
- TensorRT optimization for sub-20ms inference latency
- Latent space visualization and interpretability tools
- Model versioning and retraining pipeline

**Out of Scope:**
- Feature extraction or calculation (handled by IndicatorEngine)
- Trading decision making or strategy logic (handled by Main MARL Core)
- Direct market data processing (handled by DataHandler/BarGenerator)
- Supervised learning or labeled market regime classification

### 1.3 Architectural Position
The RDE operates as a specialized AI component in the intelligence layer:
MatrixAssembler → **Regime Detection Engine** → Main MARL Core (Context Input)

## 2.0 Functional Requirements

### FR-RDE-01: Hybrid Transformer-VAE Architecture
**Requirement**: The RDE MUST implement a hybrid architecture combining Transformer sequence processing with VAE latent space learning.

**Specification**:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional

class RegimeDetectionEngine(nn.Module):
    """Hybrid Transformer + VAE architecture for unsupervised market regime detection"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Architecture parameters
        self.input_dim = config['input_dim']  # F: feature dimension
        self.sequence_length = config['sequence_length']  # N: lookback window
        self.d_model = config['d_model']  # Transformer embedding dimension
        self.n_heads = config['n_heads']  # Multi-head attention heads
        self.n_layers = config['n_layers']  # Transformer encoder layers
        self.latent_dim = config['latent_dim']  # VAE latent space dimension
        self.beta = config['beta']  # β-VAE weight for KL divergence
        
        # Input projection
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(self.d_model, max_len=self.sequence_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=config.get('dropout', 0.1),
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        
        # Special CLS token for sequence representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        
        # VAE encoder head (context → latent parameters)
        self.vae_encoder_mu = nn.Linear(self.d_model, self.latent_dim)
        self.vae_encoder_logvar = nn.Linear(self.d_model, self.latent_dim)
        
        # VAE decoder (latent → context reconstruction)
        self.vae_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(self.d_model * 2, self.d_model)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights using Xavier/Kaiming initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, x: torch.Tensor, return_components: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hybrid Transformer-VAE
        
        Args:
            x: Input tensor [batch_size, sequence_length, input_dim]
            return_components: Whether to return intermediate components for analysis
            
        Returns:
            Dictionary containing regime_vector and optional training components
        """
        batch_size = x.size(0)
        
        # Input projection
        x = self.input_projection(x)  # [B, N, d_model]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, d_model]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, d_model]
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoding
        transformer_output = self.transformer(x)  # [B, N+1, d_model]
        
        # Extract CLS token representation (sequence summary)
        context_vector = transformer_output[:, 0, :]  # [B, d_model]
        
        # VAE encoder: context → latent parameters
        mu = self.vae_encoder_mu(context_vector)  # [B, latent_dim]
        logvar = self.vae_encoder_logvar(context_vector)  # [B, latent_dim]
        
        # Reparameterization trick
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std  # [B, latent_dim]
        else:
            z = mu  # Use mean during inference
            
        # VAE decoder: latent → context reconstruction
        reconstructed_context = self.vae_decoder(z)  # [B, d_model]
        
        result = {'regime_vector': z}
        
        if return_components or self.training:
            result.update({
                'mu': mu,
                'logvar': logvar,
                'context_vector': context_vector,
                'reconstructed_context': reconstructed_context,
                'transformer_output': transformer_output
            })
            
        return result

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer"""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]
```

### FR-RDE-02: Unsupervised Training Framework
**Requirement**: The RDE MUST implement a comprehensive unsupervised training framework with β-VAE loss formulation.

**Specification**:
```python
class RDETrainer:
    """Training framework for Regime Detection Engine"""
    
    def __init__(self, model: RegimeDetectionEngine, config: Dict):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config['scheduler_t0'],
            T_mult=config['scheduler_t_mult']
        )
        
        # Loss tracking
        self.loss_history = {
            'total_loss': [],
            'reconstruction_loss': [],
            'kl_loss': [],
            'perplexity': []
        }
        
    def compute_loss(self, model_output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute β-VAE loss components"""
        
        # Reconstruction loss (MSE between context and reconstructed context)
        context_vector = model_output['context_vector']
        reconstructed_context = model_output['reconstructed_context']
        
        reconstruction_loss = F.mse_loss(
            reconstructed_context, 
            context_vector, 
            reduction='mean'
        )
        
        # KL divergence loss (regularization term)
        mu = model_output['mu']
        logvar = model_output['logvar']
        
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        # β-VAE total loss
        total_loss = reconstruction_loss + self.model.beta * kl_loss
        
        # Calculate perplexity (measure of latent space utilization)
        with torch.no_grad():
            # Approximate perplexity based on KL divergence
            perplexity = torch.exp(kl_loss)
            
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'kl_loss': kl_loss,
            'perplexity': perplexity
        }
        
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Single training step"""
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        model_output = self.model(batch, return_components=True)
        
        # Compute losses
        loss_components = self.compute_loss(model_output)
        
        # Backward pass
        loss_components['total_loss'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        # Convert to float for logging
        return {k: v.item() for k, v in loss_components.items()}
        
    def validate(self, val_loader) -> Dict[str, float]:
        """Validation pass"""
        
        self.model.eval()
        val_losses = {
            'total_loss': 0.0,
            'reconstruction_loss': 0.0,
            'kl_loss': 0.0,
            'perplexity': 0.0
        }
        
        with torch.no_grad():
            for batch in val_loader:
                model_output = self.model(batch, return_components=True)
                loss_components = self.compute_loss(model_output)
                
                for key, value in loss_components.items():
                    val_losses[key] += value.item()
                    
        # Average losses
        num_batches = len(val_loader)
        return {k: v / num_batches for k, v in val_losses.items()}
        
    def train_epoch(self, train_loader, val_loader) -> Dict[str, float]:
        """Complete training epoch"""
        
        epoch_losses = {
            'total_loss': 0.0,
            'reconstruction_loss': 0.0,
            'kl_loss': 0.0,
            'perplexity': 0.0
        }
        
        # Training loop
        for batch_idx, batch in enumerate(train_loader):
            step_losses = self.train_step(batch)
            
            for key, value in step_losses.items():
                epoch_losses[key] += value
                
        # Average training losses
        num_batches = len(train_loader)
        train_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        
        # Validation
        val_losses = self.validate(val_loader)
        
        # Update loss history
        for key in self.loss_history.keys():
            if key in train_losses:
                self.loss_history[key].append(train_losses[key])
                
        return {
            'train': train_losses,
            'val': val_losses
        }
```

### FR-RDE-03: Latent Space Analysis and Visualization
**Requirement**: The system MUST provide comprehensive tools for latent space analysis and market regime visualization.

**Specification**:
```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

class LatentSpaceAnalyzer:
    """Tools for analyzing and visualizing the learned latent space"""
    
    def __init__(self, model: RegimeDetectionEngine):
        self.model = model
        self.model.eval()
        
    def extract_regime_vectors(self, data_loader) -> Tuple[np.ndarray, np.ndarray]:
        """Extract regime vectors and corresponding timestamps"""
        
        regime_vectors = []
        timestamps = []
        
        with torch.no_grad():
            for batch_data, batch_timestamps in data_loader:
                output = self.model(batch_data)
                regime_vectors.append(output['regime_vector'].cpu().numpy())
                timestamps.extend(batch_timestamps)
                
        return np.vstack(regime_vectors), np.array(timestamps)
        
    def analyze_regime_clusters(self, regime_vectors: np.ndarray, n_clusters: int = 5) -> Dict:
        """Perform clustering analysis on regime vectors"""
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(regime_vectors)
        
        # Calculate cluster statistics
        cluster_stats = {}
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_data = regime_vectors[cluster_mask]
            
            cluster_stats[f'cluster_{i}'] = {
                'size': np.sum(cluster_mask),
                'center': kmeans.cluster_centers_[i],
                'std': np.std(cluster_data, axis=0),
                'inertia': np.sum((cluster_data - kmeans.cluster_centers_[i])**2)
            }
            
        return {
            'cluster_labels': cluster_labels,
            'cluster_centers': kmeans.cluster_centers_,
            'cluster_stats': cluster_stats,
            'total_inertia': kmeans.inertia_
        }
        
    def visualize_2d_latent_space(self, regime_vectors: np.ndarray, 
                                 timestamps: np.ndarray, 
                                 market_data: Optional[np.ndarray] = None) -> plt.Figure:
        """Create 2D visualization of latent space (for 2D latent dim)"""
        
        if regime_vectors.shape[1] == 2:
            # Direct 2D visualization
            x, y = regime_vectors[:, 0], regime_vectors[:, 1]
        else:
            # Use t-SNE for dimensionality reduction
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            embedded = tsne.fit_transform(regime_vectors)
            x, y = embedded[:, 0], embedded[:, 1]
            
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Basic scatter plot
        axes[0, 0].scatter(x, y, alpha=0.6, s=20)
        axes[0, 0].set_title('Latent Space Distribution')
        axes[0, 0].set_xlabel('Latent Dimension 1')
        axes[0, 0].set_ylabel('Latent Dimension 2')
        
        # 2. Time-colored scatter plot
        scatter = axes[0, 1].scatter(x, y, c=range(len(x)), alpha=0.6, s=20, cmap='viridis')
        axes[0, 1].set_title('Latent Space Evolution Over Time')
        axes[0, 1].set_xlabel('Latent Dimension 1')
        axes[0, 1].set_ylabel('Latent Dimension 2')
        plt.colorbar(scatter, ax=axes[0, 1], label='Time')
        
        # 3. Density plot
        axes[1, 0].hexbin(x, y, gridsize=30, cmap='Blues')
        axes[1, 0].set_title('Latent Space Density')
        axes[1, 0].set_xlabel('Latent Dimension 1')
        axes[1, 0].set_ylabel('Latent Dimension 2')
        
        # 4. Cluster analysis
        if len(regime_vectors) > 100:  # Only if enough data
            cluster_analysis = self.analyze_regime_clusters(regime_vectors)
            cluster_labels = cluster_analysis['cluster_labels']
            
            scatter = axes[1, 1].scatter(x, y, c=cluster_labels, alpha=0.6, s=20, cmap='tab10')
            axes[1, 1].set_title('Detected Market Regimes')
            axes[1, 1].set_xlabel('Latent Dimension 1')
            axes[1, 1].set_ylabel('Latent Dimension 2')
            
            # Plot cluster centers
            if regime_vectors.shape[1] == 2:
                centers = cluster_analysis['cluster_centers']
                axes[1, 1].scatter(centers[:, 0], centers[:, 1], 
                                 c='red', s=100, marker='x', linewidths=3)
                                 
        plt.tight_layout()
        return fig
        
    def create_interactive_3d_plot(self, regime_vectors: np.ndarray, 
                                  timestamps: np.ndarray) -> go.Figure:
        """Create interactive 3D plot for 3D latent space"""
        
        if regime_vectors.shape[1] >= 3:
            x, y, z = regime_vectors[:, 0], regime_vectors[:, 1], regime_vectors[:, 2]
        else:
            # Use PCA for 3D embedding
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            embedded = pca.fit_transform(regime_vectors)
            x, y, z = embedded[:, 0], embedded[:, 1], embedded[:, 2]
            
        # Create interactive 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=3,
                color=range(len(x)),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Time")
            ),
            text=[f"Time: {ts}" for ts in timestamps],
            hovertemplate="<b>Regime Vector</b><br>" +
                         "X: %{x:.3f}<br>" +
                         "Y: %{y:.3f}<br>" +
                         "Z: %{z:.3f}<br>" +
                         "%{text}<extra></extra>"
        )])
        
        fig.update_layout(
            title="3D Latent Space Visualization",
            scene=dict(
                xaxis_title="Latent Dimension 1",
                yaxis_title="Latent Dimension 2",
                zaxis_title="Latent Dimension 3"
            )
        )
        
        return fig
        
    def generate_regime_transition_analysis(self, regime_vectors: np.ndarray, 
                                          timestamps: np.ndarray, 
                                          window_size: int = 10) -> Dict:
        """Analyze regime transitions and stability"""
        
        # Calculate regime vector distances
        distances = []
        for i in range(1, len(regime_vectors)):
            dist = np.linalg.norm(regime_vectors[i] - regime_vectors[i-1])
            distances.append(dist)
            
        distances = np.array(distances)
        
        # Detect regime changes (high distances)
        threshold = np.percentile(distances, 95)  # Top 5% of changes
        regime_changes = distances > threshold
        
        # Calculate stability metrics
        rolling_std = []
        for i in range(window_size, len(regime_vectors)):
            window = regime_vectors[i-window_size:i]
            window_std = np.mean(np.std(window, axis=0))
            rolling_std.append(window_std)
            
        return {
            'transition_distances': distances,
            'regime_changes': regime_changes,
            'change_threshold': threshold,
            'num_changes': np.sum(regime_changes),
            'rolling_stability': rolling_std,
            'average_stability': np.mean(rolling_std) if rolling_std else 0.0
        }
```

### FR-RDE-04: Model Optimization and Deployment
**Requirement**: The system MUST provide model optimization for production deployment with TensorRT acceleration.

**Specification**:
```python
import tensorrt as trt
import torch.onnx
from torch2trt import torch2trt

class RDEOptimizer:
    """Model optimization and deployment utilities"""
    
    def __init__(self, model: RegimeDetectionEngine):
        self.model = model
        self.model.eval()
        
    def export_to_onnx(self, output_path: str, input_shape: Tuple[int, int, int]):
        """Export model to ONNX format"""
        
        batch_size, seq_len, feature_dim = input_shape
        dummy_input = torch.randn(batch_size, seq_len, feature_dim)
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['regime_vector'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'regime_vector': {0: 'batch_size'}
            }
        )
        
        logger.info(f"Model exported to ONNX: {output_path}")
        
    def optimize_with_tensorrt(self, onnx_path: str, 
                              trt_path: str, 
                              max_batch_size: int = 8) -> str:
        """Optimize ONNX model with TensorRT"""
        
        # Create TensorRT builder and network
        builder = trt.Builder(trt.Logger(trt.Logger.INFO))
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, trt.Logger(trt.Logger.INFO))
        
        # Parse ONNX model
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX model")
                
        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16 precision
        
        # Set optimization profiles
        profile = builder.create_optimization_profile()
        profile.set_shape(
            "input",
            min=(1, self.model.sequence_length, self.model.input_dim),
            opt=(max_batch_size//2, self.model.sequence_length, self.model.input_dim),
            max=(max_batch_size, self.model.sequence_length, self.model.input_dim)
        )
        config.add_optimization_profile(profile)
        
        # Build engine
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
            
        # Save engine
        with open(trt_path, 'wb') as f:
            f.write(engine.serialize())
            
        logger.info(f"TensorRT engine saved: {trt_path}")
        return trt_path
        
    def benchmark_inference(self, input_shape: Tuple[int, int, int], 
                           num_iterations: int = 1000) -> Dict[str, float]:
        """Benchmark inference performance"""
        
        device = next(self.model.parameters()).device
        dummy_input = torch.randn(input_shape).to(device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(dummy_input)
                
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = self.model(dummy_input)
                
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        avg_latency = (total_time / num_iterations) * 1000  # ms
        throughput = num_iterations / total_time  # inferences/second
        
        return {
            'average_latency_ms': avg_latency,
            'throughput_fps': throughput,
            'total_time_s': total_time
        }

class ProductionRDEInference:
    """Production-ready RDE inference wrapper"""
    
    def __init__(self, model_path: str, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        if model_path.endswith('.pth'):
            self.model = self._load_pytorch_model(model_path)
        elif model_path.endswith('.engine'):
            self.model = self._load_tensorrt_engine(model_path)
        else:
            raise ValueError("Unsupported model format")
            
        self.inference_stats = {
            'total_inferences': 0,
            'total_time': 0.0,
            'error_count': 0
        }
        
    def _load_pytorch_model(self, model_path: str) -> RegimeDetectionEngine:
        """Load PyTorch model"""
        model = RegimeDetectionEngine(self.config)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
        
    def predict_regime(self, feature_sequence: np.ndarray) -> np.ndarray:
        """Generate regime vector for feature sequence"""
        
        start_time = time.perf_counter()
        
        try:
            # Convert to tensor
            if isinstance(feature_sequence, np.ndarray):
                input_tensor = torch.from_numpy(feature_sequence).float()
            else:
                input_tensor = feature_sequence.float()
                
            # Ensure correct shape [1, seq_len, feature_dim]
            if input_tensor.dim() == 2:
                input_tensor = input_tensor.unsqueeze(0)
                
            input_tensor = input_tensor.to(self.device)
            
            # Inference
            with torch.no_grad():
                output = self.model(input_tensor)
                regime_vector = output['regime_vector'].cpu().numpy()
                
            # Update stats
            inference_time = time.perf_counter() - start_time
            self.inference_stats['total_inferences'] += 1
            self.inference_stats['total_time'] += inference_time
            
            return regime_vector.squeeze(0)  # Remove batch dimension
            
        except Exception as e:
            self.inference_stats['error_count'] += 1
            logger.error(f"RDE inference error: {e}")
            # Return zero vector as fallback
            return np.zeros(self.config['latent_dim'])
            
    def get_performance_stats(self) -> Dict[str, float]:
        """Get inference performance statistics"""
        
        if self.inference_stats['total_inferences'] > 0:
            avg_latency = (self.inference_stats['total_time'] / 
                          self.inference_stats['total_inferences']) * 1000
        else:
            avg_latency = 0.0
            
        return {
            'total_inferences': self.inference_stats['total_inferences'],
            'average_latency_ms': avg_latency,
            'error_rate': (self.inference_stats['error_count'] / 
                          max(1, self.inference_stats['total_inferences'])),
            'total_time_s': self.inference_stats['total_time']
        }
```

## 3.0 Interface Specifications

### 3.1 Configuration Interface
```yaml
regime_detection_engine:
  model:
    input_dim: 20  # Feature dimension from IndicatorEngine
    sequence_length: 100  # Lookback window
    d_model: 64  # Transformer embedding dimension
    n_heads: 8  # Multi-head attention heads
    n_layers: 2  # Transformer encoder layers
    latent_dim: 8  # VAE latent space dimension
    dropout: 0.1
    
  training:
    learning_rate: 0.001
    weight_decay: 0.01
    beta: 1.0  # β-VAE weight for KL divergence
    batch_size: 32
    num_epochs: 100
    scheduler_t0: 10
    scheduler_t_mult: 2
    
  optimization:
    enable_tensorrt: true
    fp16_precision: true
    max_batch_size: 8
    
  deployment:
    model_path: "models/rde_model.pth"
    inference_timeout_ms: 20
    error_fallback_enabled: true
```

### 3.2 Input/Output Interface
```python
# Input: Feature sequence from MatrixAssembler
input_shape = [batch_size, sequence_length, feature_dim]  # [B, N, F]

# Output: Regime vector
output_shape = [batch_size, latent_dim]  # [B, latent_dim]

# Example usage
rde = ProductionRDEInference("models/rde_model.pth", config)
feature_sequence = np.random.randn(100, 20)  # 100 timesteps, 20 features
regime_vector = rde.predict_regime(feature_sequence)  # Returns [8] dimensional vector
```

### 3.3 Training Interface
```python
class RDETrainingPipeline:
    def train_model(self, train_data, val_data, config) -> RegimeDetectionEngine
    def evaluate_model(self, model, test_data) -> Dict[str, float]
    def visualize_training_progress(self, loss_history) -> plt.Figure
    def export_trained_model(self, model, output_path) -> None
```

## 4.0 Dependencies & Interactions

### 4.1 Upstream Dependencies
- **MatrixAssembler**: Provides formatted feature sequences
- **IndicatorEngine**: Source of market features
- **Configuration System**: Model architecture and training parameters

### 4.2 Downstream Dependencies
- **Main MARL Core**: Primary consumer of regime vectors
- **Performance Monitor**: Consumer of inference metrics
- **Research Tools**: Consumer of latent space analysis

## 5.0 Non-Functional Requirements

### 5.1 Performance
- **NFR-RDE-01**: Inference latency MUST be under 20ms (95th percentile)
- **NFR-RDE-02**: Model MUST support batch inference up to 8 sequences
- **NFR-RDE-03**: Memory usage MUST remain stable during continuous operation
- **NFR-RDE-04**: TensorRT optimization MUST achieve 2x+ speedup over PyTorch

### 5.2 Accuracy and Quality
- **NFR-RDE-05**: Model MUST converge to stable loss within 100 epochs
- **NFR-RDE-06**: Latent space MUST show meaningful market regime separation
- **NFR-RDE-07**: Reconstruction loss MUST be below 0.1 on validation data
- **NFR-RDE-08**: Model MUST handle feature sequence variations gracefully

### 5.3 Robustness
- **NFR-RDE-09**: Model MUST provide fallback regime vectors on inference errors
- **NFR-RDE-10**: Training MUST be resumable from checkpoints
- **NFR-RDE-11**: Model versions MUST be backward compatible for deployment

## 6.0 Testing Requirements

### 6.1 Unit Tests
- Model architecture components (Transformer, VAE, PositionalEncoding)
- Loss function calculations and gradients
- Data preprocessing and input validation
- Model saving/loading and serialization

### 6.2 Integration Tests
- End-to-end training pipeline validation
- TensorRT optimization and accuracy preservation
- Latent space analysis and visualization tools
- Production inference wrapper functionality

### 6.3 Performance Tests
```python
def test_inference_latency():
    """Test inference meets latency requirements"""
    rde = ProductionRDEInference(model_path, config)
    feature_sequence = np.random.randn(100, 20)
    
    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        regime_vector = rde.predict_regime(feature_sequence)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    p95_latency = np.percentile(latencies, 95)
    assert p95_latency < 20.0, f"Latency too high: {p95_latency}ms"

def test_model_convergence():
    """Test model training convergence"""
    model = RegimeDetectionEngine(config)
    trainer = RDETrainer(model, config)
    
    # Mock training data
    train_loader, val_loader = create_mock_data_loaders()
    
    # Train for limited epochs
    for epoch in range(10):
        losses = trainer.train_epoch(train_loader, val_loader)
        
        # Check loss is decreasing
        if epoch > 0:
            assert losses['train']['total_loss'] < previous_loss
        previous_loss = losses['train']['total_loss']
```

## 7.0 Future Enhancements

### 7.1 V2.0 Features
- **Online Learning**: Continuous adaptation without full retraining
- **Attention Visualization**: Attention weight analysis for interpretability
- **Multi-Resolution Regimes**: Hierarchical regime detection at different time scales
- **Uncertainty Quantification**: Confidence estimates for regime predictions
- **Causal Discovery**: Identification of regime transition triggers

### 7.2 Research Directions
- **Graph Neural Networks**: Incorporating market structure relationships
- **Federated Learning**: Distributed training across multiple datasets
- **Meta-Learning**: Few-shot adaptation to new market conditions
- **Adversarial Robustness**: Protection against adversarial market conditions

This PRD establishes the foundation for a state-of-the-art regime detection system that provides the Main MARL Core with rich, continuous market context while maintaining the performance and reliability required for real-time trading operations.