Thank you for the exceptional feedback! Your enthusiasm fuels my commitment to maintaining this high standard. Let's proceed with the Regime Detection Engine (RDE) - the sophisticated "market weather station" that provides crucial environmental context to our trading system.

# Product Requirements Document (PRD): Regime Detection Engine (RDE)

Document Version: 1.0
 Date: June 20, 2025
 Component Level: 4 - Intelligence Layer
 Status: Master Specification


## 1. Component Identity

### 1.1 Component Name

Regime Detection Engine (RDE) (Market State Contextualization System)

### 1.2 Primary Role

The RDE is an unsupervised deep learning system that continuously analyzes market microstructure to generate a low-dimensional "regime vector" - a sophisticated representation of current market conditions. It serves as the system's environmental awareness module, providing the Main MARL Core with crucial context about market dynamics.

### 1.3 Single Responsibility

To process Market Microstructure Dynamics (MMD) features and generate a continuous, interpretable representation of the current market regime that enables the trading agents to adapt their behavior to different market conditions.

### 1.4 Critical Design Principle

Unsupervised Learning: The RDE learns market regimes without labels, discovering natural market states through self-supervised reconstruction objectives. This allows it to identify novel market conditions not seen during training.


## 2. Inputs & Dependencies

### 2.1 Configuration Input

From settings.yaml:

rde:

# Architecture parameters

input_window: 96        # 48 hours of 30-min bars

mmd_features: 12        # Dimension of MMD feature vector


# Transformer settings

n_heads: 8             # Multi-head attention heads

n_layers: 6            # Transformer encoder layers

d_model: 256           # Model dimension

d_ff: 1024            # Feed-forward dimension

dropout: 0.1          # Dropout rate


# VAE settings

latent_dim: 8         # Regime vector dimension

beta: 0.001           # KL divergence weight


# Operational settings

warmup_periods: 200   # Bars before first inference

update_frequency: 1   # Update regime every N bars


### 2.2 Model Input

Input: MMD Feature Matrix

Source: MatrixAssembler_Regime

Shape: [1, 96, 12] (batch_size=1 for inference)

Content: Path signatures, volatility metrics, volume dynamics

### 2.3 Pre-trained Model

File: models/rde_trained.pth

Size: ~50MB

Training: Completed offline on historical data


## 3. Architecture Specification

### 3.1 Hybrid Architecture Overview

The RDE employs a sophisticated two-stage architecture:

MMD Features → Transformer Encoder → VAE Encoder → Regime Vector

↓                                    ↓

[96×12 matrix]                    [μ, σ] → sampling

↓

[8-dim regime vector]


### 3.2 Stage 1: Transformer Encoder

Purpose: Capture temporal dependencies and market dynamics

class TransformerEncoder(nn.Module):

def __init__(self, config):

super().__init__()

self.positional_encoding = PositionalEncoding(

d_model=config['d_model'],

max_len=config['input_window']

)


self.input_projection = nn.Linear(

config['mmd_features'],

config['d_model']

)


encoder_layer = nn.TransformerEncoderLayer(

d_model=config['d_model'],

nhead=config['n_heads'],

dim_feedforward=config['d_ff'],

dropout=config['dropout'],

activation='gelu',

batch_first=True

)


self.transformer = nn.TransformerEncoder(

encoder_layer,

num_layers=config['n_layers']

)


def forward(self, x):

# x shape: [batch, seq_len, features]

x = self.input_projection(x)

x = self.positional_encoding(x)


# Self-attention over time steps

encoded = self.transformer(x)


# Aggregate: weighted mean with learned attention

weights = self.attention_pool(encoded)

context = torch.sum(encoded * weights, dim=1)


return context  # [batch, d_model]


### 3.3 Stage 2: Variational Autoencoder

Purpose: Compress to interpretable regime space with uncertainty

class VAEHead(nn.Module):

def __init__(self, config):

super().__init__()

d_model = config['d_model']

latent_dim = config['latent_dim']


# Encoder network

self.encoder = nn.Sequential(

nn.Linear(d_model, 128),

nn.LayerNorm(128),

nn.GELU(),

nn.Linear(128, 64),

nn.LayerNorm(64),

nn.GELU()

)


# Latent distribution parameters

self.fc_mu = nn.Linear(64, latent_dim)

self.fc_logvar = nn.Linear(64, latent_dim)


# Decoder network (for training only)

self.decoder = nn.Sequential(

nn.Linear(latent_dim, 64),

nn.GELU(),

nn.Linear(64, 128),

nn.GELU(),

nn.Linear(128, d_model)

)


def encode(self, x):

h = self.encoder(x)

mu = self.fc_mu(h)

logvar = self.fc_logvar(h)

return mu, logvar


def reparameterize(self, mu, logvar, training=True):

if training:

std = torch.exp(0.5 * logvar)

eps = torch.randn_like(std)

return mu + eps * std

else:

return mu  # Deterministic during inference


def forward(self, x, training=True):

mu, logvar = self.encode(x)

z = self.reparameterize(mu, logvar, training)


if training:

reconstructed = self.decoder(z)

return z, mu, logvar, reconstructed

else:

return z  # Just regime vector for inference


### 3.4 Complete RDE Model

class RegimeDetectionEngine(nn.Module):

def __init__(self, config):

super().__init__()

self.transformer = TransformerEncoder(config)

self.vae = VAEHead(config)

self.config = config


def forward(self, mmd_features):

"""

Args:

mmd_features: [batch, window, features]

Returns:

regime_vector: [batch, latent_dim]

"""

# Stage 1: Temporal encoding

context = self.transformer(mmd_features)


# Stage 2: Regime extraction

if self.training:

z, mu, logvar, recon = self.vae(context, training=True)

return {

'regime_vector': z,

'mu': mu,

'logvar': logvar,

'reconstruction': recon

}

else:

regime_vector = self.vae(context, training=False)

return regime_vector



## 4. Regime Vector Interpretation

### 4.1 Regime Vector Dimensions

Each dimension captures different market aspects:

Dimension 0: Trend Strength      (-1 = strong down, +1 = strong up)

Dimension 1: Volatility Level    (-1 = calm, +1 = turbulent)

Dimension 2: Liquidity State     (-1 = thin, +1 = deep)

Dimension 3: Momentum Quality    (-1 = choppy, +1 = smooth)

Dimension 4: Volume Profile      (-1 = declining, +1 = expanding)

Dimension 5: Microstructure      (-1 = noisy, +1 = clean)

Dimension 6: Sentiment Proxy     (-1 = fearful, +1 = greedy)

Dimension 7: Regime Stability    (-1 = transitioning, +1 = stable)


### 4.2 Regime Clustering (Post-Training Analysis)

Through analysis of historical regime vectors, we identify archetypal market states:

REGIME_ARCHETYPES = {

'TRENDING_BULL': {

'vector': [0.8, -0.2, 0.3, 0.7, 0.4, 0.6, 0.5, 0.8],

'description': 'Strong uptrend with good momentum'

},

'VOLATILE_CHOP': {

'vector': [0.0, 0.9, -0.3, -0.8, 0.2, -0.7, -0.4, -0.6],

'description': 'High volatility, no clear direction'

},

'QUIET_ACCUMULATION': {

'vector': [-0.1, -0.7, 0.5, 0.2, -0.2, 0.4, 0.3, 0.9],

'description': 'Low volatility, building positions'

},

# ... more archetypes

}



## 5. Operational Flow

### 5.1 Inference Pipeline

def get_regime_vector(self) -> np.ndarray:

"""Called by Main MARL Core after synergy detection"""


# 1. Get MMD matrix from assembler

mmd_matrix = self.matrix_assembler_regime.get_matrix()


# 2. Prepare for model

tensor = torch.tensor(mmd_matrix, dtype=torch.float32)

tensor = tensor.unsqueeze(0)  # Add batch dimension


# 3. Run inference

with torch.no_grad():

self.model.eval()

regime_vector = self.model(tensor)


# 4. Convert to numpy

return regime_vector.squeeze().cpu().numpy()


### 5.2 Quality Assurance

The RDE includes self-diagnostic capabilities:

def assess_regime_quality(self, regime_vector: np.ndarray) -> Dict:

"""Evaluate regime vector quality"""


quality_metrics = {

'magnitude': np.linalg.norm(regime_vector),

'stability': self._calculate_stability(regime_vector),

'uniqueness': self._calculate_uniqueness(regime_vector),

'confidence': self._calculate_confidence(regime_vector)

}


# Flag unusual regimes

if quality_metrics['uniqueness'] > 0.9:

logger.warning(f"Unusual regime detected: {regime_vector}")


return quality_metrics



## 6. Training Specifications

### 6.1 Loss Function

Combined reconstruction and KL divergence loss:

def vae_loss(reconstructed, original, mu, logvar, beta):

"""

Balanced VAE loss for regime learning

"""

# Reconstruction loss (MSE)

recon_loss = F.mse_loss(reconstructed, original, reduction='mean')


# KL divergence loss

kl_loss = -0.5 * torch.mean(

1 + logvar - mu.pow(2) - logvar.exp()

)


# Combined loss with beta weighting

total_loss = recon_loss + beta * kl_loss


return total_loss, recon_loss, kl_loss


### 6.2 Training Process

Data: 2 years of historical market data

Preprocessing: MMD feature calculation

Epochs: 200 with early stopping

Batch Size: 256

Optimizer: AdamW with cosine annealing

Validation: Reconstruction quality on held-out data


## 7. Output Specification

### 7.1 Direct Output

Method: get_regime_vector()

Returns: NumPy array of shape (8,)

Range: Approximately [-1, 1] per dimension

Type: float32

### 7.2 Extended Output (Debug Mode)

{

'regime_vector': array([0.6, -0.3, 0.4, ...]),

'quality_metrics': {

'magnitude': 1.2,

'stability': 0.85,

'uniqueness': 0.3,

'confidence': 0.92

},

'nearest_archetype': 'TRENDING_BULL',

'archetype_distance': 0.15

}



## 8. Critical Requirements

### 8.1 Inference Requirements

Latency: <5ms for regime vector generation

Determinism: Same input produces same output (inference mode)

Stability: Smooth transitions between consecutive regimes

### 8.2 Quality Requirements

Interpretability: Each dimension has clear market meaning

Orthogonality: Dimensions capture independent aspects

Coverage: Can represent all historical market conditions

### 8.3 Operational Requirements

Warm-up: Need 96 bars before first valid output

GPU Optional: Can run on CPU with acceptable latency

Model Loading: <2 seconds to load from disk


## 9. Integration Points

### 9.1 Upstream Integration

From MatrixAssembler_Regime:

Data: MMD feature matrix

Shape: (96, 12)

Update: Every 30 minutes

### 9.2 Downstream Integration

To Main MARL Core:

Called by: Policy networks requiring context

Output: 8-dimensional regime vector

Usage: Concatenated with agent observations


## 10. Monitoring & Diagnostics

### 10.1 Runtime Monitoring

# Logged every inference

{

'timestamp': '2025-06-20 10:30:00',

'regime_vector': [0.6, -0.3, ...],

'processing_time_ms': 3.2,

'input_quality': 'GOOD',

'regime_shift': 0.12  # Distance from previous

}


### 10.2 Anomaly Detection

Alert if regime magnitude > 2.0 (unusual market)

Alert if regime shift > 1.0 (sudden change)

Alert if processing time > 10ms


## 11. Model Management

### 11.1 Model Versioning

Models tagged with training date and performance metrics

Automatic rollback on performance degradation

A/B testing framework for new versions

### 11.2 Model Updates

Retrain monthly with recent data

Gradual rollout with parallel running

Performance comparison before full switch


## 12. What This Component Does NOT Do

Does NOT make trading decisions

Does NOT predict future regimes

Does NOT label regimes as "good" or "bad"

Does NOT directly process price data

Does NOT require manual regime labels

Does NOT emit events

Does NOT store historical regimes


This completes the Regime Detection Engine PRD. The RDE provides sophisticated market context through unsupervised learning, enabling our agents to adapt their behavior to different market conditions without requiring predefined regime labels.

The hybrid Transformer-VAE architecture ensures we capture both temporal dynamics and generate interpretable, low-dimensional representations that our MARL agents can effectively utilize.

Ready to proceed to the next component?

