# Phase 4 MARL System Architecture Diagrams

## 1. Overall MARL System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MARL DECISION LAYER (Phase 4)                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐           │
│  │   Regime Agent  │    │ Structure Agent │    │ Tactical Agent  │           │
│  │   (Level 1)     │    │   (Level 2)     │    │   (Level 3)     │           │
│  │                 │    │                 │    │                 │           │
│  │ • Market Regime │    │ • Trend Bias    │    │ • Entry/Exit    │           │
│  │ • Volatility    │    │ • Position Size │    │ • Timing        │           │
│  │ • Context       │    │ • Direction     │    │ • Execution     │           │
│  │                 │    │                 │    │                 │           │
│  │ Input: 96×N     │    │ Input: 48×8     │    │ Input: 60×7     │           │
│  │ Update: 30min   │    │ Update: 30min   │    │ Update: 5min    │           │
│  └─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘           │
│            │                      │                      │                   │
│            └──────────────────────┼──────────────────────┘                   │
│                                   │                                          │
│  ┌─────────────────────────────────┴─────────────────────────────────┐       │
│  │                     Agent Communication Hub                       │       │
│  │                                                                   │       │
│  │  • Message Passing Protocol                                       │       │
│  │  • Attention-based Communication                                  │       │
│  │  • Shared Context Memory                                          │       │
│  │  • Consensus Mechanism                                            │       │
│  └─────────────────────────┬─────────────────────────────────────────┘       │
│                            │                                                 │
│  ┌─────────────────────────┴─────────────────────────────────┐               │
│  │                    Risk Agent (Level 4)                   │               │
│  │                                                           │               │
│  │  • Real-time Risk Monitoring                              │               │
│  │  • Position Constraint Enforcement                        │               │
│  │  • Dynamic Risk Adjustment                                │               │
│  │  • Emergency Override Controls                            │               │
│  │                                                           │               │
│  │  Input: All Matrices + Portfolio State                    │               │
│  │  Update: Real-time (Tick-based)                          │               │
│  └───────────────────────────────────────────────────────────┘               │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        DECISION AGGREGATION & OUTPUT                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐           │
│  │ Trade Decision  │    │  Position Size  │    │ Risk Parameters │           │
│  │                 │    │                 │    │                 │           │
│  │ • Direction     │    │ • Quantity      │    │ • Stop Loss     │           │
│  │ • Confidence    │    │ • Leverage      │    │ • Take Profit   │           │
│  │ • Timeframe     │    │ • Risk Capital  │    │ • Max Exposure  │           │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      EXISTING SYSTEM INTEGRATION                               │
│                             (Phases 1-3)                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐           │
│  │ Matrix Assemb.  │    │ Indicator Eng.  │    │  Event System   │           │
│  │ (Phase 3)       │    │ (Phase 2)       │    │  (Phase 1)      │           │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 2. Agent Communication Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          AGENT COMMUNICATION PROTOCOL                           │
└──────────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
                    │         Shared Memory Space         │
                    │                                     │
                    │  • Market Context (Regime State)    │
                    │  • Structural Bias (Trend State)    │  
                    │  • Tactical Signals (Entry/Exit)    │
                    │  • Risk Constraints (Limits)        │
                    │  • Portfolio State (Positions)      │
                    └─────────────────┬───────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌──────────────┐              ┌──────────────┐              ┌──────────────┐
│ Regime Agent │              │Structure Agt │              │Tactical Agent│
│              │              │              │              │              │
│ Publishes:   │              │ Publishes:   │              │ Publishes:   │
│ • Regime ID  │◄────────────►│ • Bias Score │◄────────────►│ • Entry Sig  │
│ • Volatility │              │ • Confidence │              │ • Exit Sig   │
│ • Transition │              │ • Position   │              │ • Urgency    │
│                                                                          │
└──────────────┘              └──────────────┘              └──────────────┘
        │                             │                             │
        │                             │                             │
        └─────────────────────────────┼─────────────────────────────┘
                                      │
                                      ▼
                              ┌──────────────┐
                              │  Risk Agent  │
                              │              │
                              │ Monitors:    │
                              │ • All Comms  │
                              │ • Portfolio  │
                              │ • Constraints│
                              │              │
                              │ Can Override:│
                              │ • Block      │
                              │ • Modify     │
                              │ • Force Exit │
                              └──────────────┘

Communication Flow:
1. Regime Agent broadcasts market context every 30min
2. Structure Agent incorporates regime info + publishes bias
3. Tactical Agent uses both regime + structure for timing
4. Risk Agent monitors all communications in real-time
5. Final decision requires consensus or risk agent approval
```

## 3. Training Architecture (MAPPO Implementation)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         CENTRALIZED TRAINING ARCHITECTURE                       │
└──────────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
                    │        Central Coordinator          │
                    │                                     │
                    │  • Global Policy Updates            │
                    │  • Experience Aggregation           │
                    │  • Hyperparameter Coordination      │
                    │  • Training Orchestration           │
                    └─────────────────┬───────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌──────────────┐              ┌──────────────┐              ┌──────────────┐
│ Regime Env   │              │Structure Env │              │Tactical Env  │
│              │              │              │              │              │
│ • Regime     │              │ • Trend      │              │ • Entry/Exit │
│   Simulation │              │   Prediction │              │   Timing     │
│ • Volatility │              │ • Bias       │              │ • Execution  │
│   Modeling   │              │   Estimation │              │   Quality    │
│              │              │              │              │              │
│ Actor-Critic │              │ Actor-Critic │              │ Actor-Critic │
│ Network      │              │ Network      │              │ Network      │
└──────┬───────┘              └──────┬───────┘              └──────┬───────┘
       │                             │                             │
       │                             │                             │
       └─────────────────────────────┼─────────────────────────────┘
                                     │
                            ┌──────────────┐
                            │ Risk Env     │
                            │              │
                            │ • Portfolio  │
                            │   Management │
                            │ • Constraint │
                            │   Enforcement│
                            │              │
                            │ DQN Network  │
                            └──────┬───────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │        Experience Replay           │
                    │                                     │
                    │  • Multi-Agent Trajectories        │
                    │  • Reward Calculations              │
                    │  • State-Action-Reward Tuples      │
                    │  • Priority Sampling               │
                    └─────────────────────────────────────┘

Training Process:
1. Parallel environment simulation across all agents
2. Experience collection with shared observations
3. Centralized policy updates using MAPPO algorithm
4. Model synchronization across all agent instances
5. Performance evaluation and hyperparameter adjustment
```

## 4. Data Flow Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW PIPELINE                                 │
└──────────────────────────────────────────────────────────────────────────────────┘

Market Data (Ticks)
        │
        ▼
┌─────────────────┐
│ Bar Generation  │ ──► 5-min bars ──► Matrix Assembler 5m (60×7)
│ (Phase 2)       │                              │
└─────────────────┘                              │
        │                                        │
        ▼                                        │
30-min bars ──► Matrix Assembler 30m (48×8)     │
        │                    │                  │
        ▼                    │                  │
Regime Matrix (96×N) ◄───────┘                  │
        │                                       │
        │                                       │
        ▼                                       ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Regime Agent    │  │ Structure Agent │  │ Tactical Agent  │
│                 │  │                 │  │                 │
│ Input: 96×N     │  │ Input: 48×8     │  │ Input: 60×7     │
│ Freq: 30min     │  │ Freq: 30min     │  │ Freq: 5min      │
│                 │  │                 │  │                 │
│ Output:         │  │ Output:         │  │ Output:         │
│ • Regime State  │  │ • Direction     │  │ • Entry Signal  │
│ • Volatility    │  │ • Confidence    │  │ • Exit Signal   │
│ • Context       │  │ • Bias Strength │  │ • Timing        │
└─────────┬───────┘  └─────────┬───────┘  └─────────┬───────┘
          │                    │                    │
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
                               ▼
                    ┌─────────────────┐
                    │   Risk Agent    │
                    │                 │
                    │ Input: All Data │ 
                    │ + Portfolio     │
                    │ Freq: Real-time │
                    │                 │
                    │ Output:         │
                    │ • Allow/Block   │
                    │ • Size Adjust   │
                    │ • Risk Limits   │
                    └─────────┬───────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Decision Engine │
                    │                 │
                    │ • Aggregate     │
                    │ • Validate      │
                    │ • Execute       │
                    └─────────────────┘

Data Processing Characteristics:
• Matrix Updates: <1ms per update (existing performance)
• Agent Inference: <5ms per agent per decision
• Communication Overhead: <2ms between agents
• Total Decision Latency: <10ms from data to decision
```

## 5. Model Architecture Details

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                            INDIVIDUAL AGENT ARCHITECTURES                       │
└──────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              REGIME AGENT MODEL                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│ Input: Regime Matrix (96×N)                                                    │
│        │                                                                       │
│        ▼                                                                       │
│ ┌─────────────────┐    Multi-head attention to capture                        │
│ │ Input Embedding │    long-term dependencies in market                       │
│ │ (N → 128)       │    regime patterns                                        │
│ └─────┬───────────┘                                                            │
│       │                                                                        │
│       ▼                                                                        │
│ ┌─────────────────┐                                                            │
│ │ Transformer     │    4 layers, 8 attention heads                            │
│ │ Encoder         │    Captures regime transitions                             │
│ │ (96×128)        │    and volatility patterns                                │
│ └─────┬───────────┘                                                            │
│       │                                                                        │
│       ▼                                                                        │
│ ┌─────────────────┐                                                            │
│ │ Classification  │    Output: 4 regime classes                               │
│ │ Head            │    [trending, ranging, volatile, transition]              │
│ │ (128 → 4)       │    + confidence scores                                    │
│ └─────────────────┘                                                            │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                             STRUCTURE AGENT MODEL                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│ Input: 30m Matrix (48×8) + Regime Context                                      │
│        │                                                                       │
│        ▼                                                                       │
│ ┌─────────────────┐    CNN for local pattern recognition                      │
│ │ 1D CNN Layers   │    Kernel sizes: [3, 5, 7] for multi-scale              │
│ │ (8 → 64 → 128)  │    feature extraction                                     │
│ └─────┬───────────┘                                                            │
│       │                                                                        │
│       ▼                                                                        │
│ ┌─────────────────┐                                                            │
│ │ LSTM Layers     │    2-layer LSTM to capture trend                          │
│ │ (128 → 256)     │    persistence and momentum                               │
│ └─────┬───────────┘                                                            │
│       │                                                                        │
│       ▼                                                                        │
│ ┌─────────────────┐                                                            │
│ │ Attention       │    Attention over time steps to focus                     │
│ │ Mechanism       │    on most relevant periods                               │
│ └─────┬───────────┘                                                            │
│       │                                                                        │
│       ▼                                                                        │
│ ┌─────────────────┐                                                            │
│ │ Output Heads    │    Directional bias: [-1, 1]                             │
│ │ Actor: (256→5)  │    Position size: [0, 1]                                 │
│ │ Critic: (256→1) │    Confidence: [0, 1]                                    │
│ └─────────────────┘                                                            │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                             TACTICAL AGENT MODEL                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│ Input: 5m Matrix (60×7) + Structure Context                                    │
│        │                                                                       │
│        ▼                                                                       │
│ ┌─────────────────┐    Multi-scale feature extraction                         │
│ │ Feature         │    for short-term patterns                                │
│ │ Extraction      │                                                           │
│ │ (7 → 128)       │                                                           │
│ └─────┬───────────┘                                                            │
│       │                                                                        │
│       ▼                                                                        │
│ ┌─────────────────┐                                                            │
│ │ Self-Attention  │    Attention mechanism to identify                        │
│ │ Layers          │    critical entry/exit moments                            │
│ │ (8 heads)       │                                                           │
│ └─────┬───────────┘                                                            │
│       │                                                                        │
│       ▼                                                                        │
│ ┌─────────────────┐                                                            │
│ │ Position        │    Positional encoding for                                │
│ │ Encoding        │    time-aware processing                                  │
│ └─────┬───────────┘                                                            │
│       │                                                                        │
│       ▼                                                                        │
│ ┌─────────────────┐                                                            │
│ │ Action Heads    │    Actions: [enter_long, enter_short,                     │
│ │ Actor: (128→5)  │    exit_position, hold, reduce_size]                      │
│ │ Critic: (128→1) │    With action probabilities                              │
│ └─────────────────┘                                                            │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                               RISK AGENT MODEL                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│ Input: All Matrices + Portfolio State + Proposed Actions                       │
│        │                                                                       │
│        ▼                                                                       │
│ ┌─────────────────┐    Comprehensive state representation                     │
│ │ State Encoder   │    including portfolio risk metrics                       │
│ │ (Variable → 256)│                                                           │
│ └─────┬───────────┘                                                            │
│       │                                                                        │
│       ▼                                                                        │
│ ┌─────────────────┐                                                            │
│ │ Deep Q-Network  │    3-layer DQN with experience replay                     │
│ │ (256→512→256)   │    for risk-aware decision making                         │
│ └─────┬───────────┘                                                            │
│       │                                                                        │
│       ▼                                                                        │
│ ┌─────────────────┐                                                            │
│ │ Risk Actions    │    Actions: [allow, modify_size,                          │
│ │ (256 → 4)       │    force_exit, block_entry]                               │
│ │                 │    With Q-values for each action                          │
│ └─────────────────┘                                                            │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 6. Deployment Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                            PRODUCTION DEPLOYMENT                                │
└──────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              INFERENCE PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Real-time Market Data                                                          │
│           │                                                                     │
│           ▼                                                                     │
│  ┌─────────────────┐     INDICATORS_READY Event                               │
│  │ Matrix          │ ──────────────────────────────────┐                      │
│  │ Assemblers      │                                   │                      │
│  │ (Phase 3)       │                                   │                      │
│  └─────────────────┘                                   │                      │
│                                                        │                      │
│                                                        ▼                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    MARL Agent Manager                                  │   │
│  │                                                                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │   │
│  │  │ Regime Agt  │  │Structure Agt│  │Tactical Agt │  │ Risk Agent  │   │   │
│  │  │ (TorchServe)│  │(TorchServe) │  │(TorchServe) │  │(TorchServe) │   │   │
│  │  │             │  │             │  │             │  │             │   │   │
│  │  │ Model: GPU  │  │ Model: GPU  │  │ Model: GPU  │  │ Model: CPU  │   │   │
│  │  │ Batch: 1    │  │ Batch: 1    │  │ Batch: 1    │  │ Batch: 1    │   │   │
│  │  │ Latency:<3ms│  │ Latency:<3ms│  │ Latency:<3ms│  │ Latency:<2ms│   │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                         │
│                                      ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    Decision Aggregator                                  │   │
│  │                                                                         │   │
│  │  • Collect all agent outputs                                           │   │
│  │  • Apply consensus mechanism                                           │   │
│  │  • Validate against risk constraints                                   │   │
│  │  • Generate final trading decision                                     │   │
│  │                                                                         │   │
│  │  Output: EXECUTE_TRADE event                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                         │
│                                      ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                 Monitoring & Logging                                    │   │
│  │                                                                         │   │
│  │  • Agent performance metrics                                           │   │
│  │  • Decision latency tracking                                           │   │
│  │  • Model drift detection                                               │   │
│  │  • Error logging and alerting                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

Resource Requirements:
• GPU Memory: 8GB for all models (4×2GB each)
• System Memory: 16GB total
• CPU: 8 cores minimum
• Storage: 10GB for models + logs
• Network: Low latency to market data feed
```
