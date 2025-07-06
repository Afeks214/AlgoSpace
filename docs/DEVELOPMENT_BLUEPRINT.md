# AlgoSpace Development Blueprint - PHASE 2
Last Updated: 2025-06-21
Status: ACTIVE - CHECK BEFORE EVERY CODE CHANGE

## GOLDEN RULES
1. NO file creation without checking this blueprint
2. NO notebooks except those listed in ML Models section
3. NO new features not explicitly listed here
4. ALWAYS use existing code from notebooks when available
5. EVERY file must have a clear purpose tied to the plan

## APPROVED FILE STRUCTURE

### Python Modules (.py files ONLY)
```
src/
├── core/
│   ├── __init__.py              ✓ Create empty
│   ├── events.py                ✓ Event system with EventType enum
│   ├── config.py                ✓ YAML config loader
│   └── kernel.py                ✓ System orchestration
├── data/
│   ├── __init__.py              ✓ Create empty
│   ├── handlers.py              ✓ AbstractDataHandler, BacktestDataHandler
│   ├── bar_generator.py         ✓ Tick to 5m/30m bar conversion
│   └── validators.py            ✓ Data quality checks
├── indicators/
│   ├── __init__.py              ✓ Create empty
│   ├── base.py                  ✓ BaseIndicator abstract class
│   ├── mlmi.py                  ✓ EXTRACT from Strategy_Implementation.ipynb
│   ├── nwrqk.py                 ✓ EXTRACT from Strategy_Implementation.ipynb
│   ├── fvg.py                   ✓ EXTRACT from Strategy_Implementation.ipynb
│   ├── lvn.py                   ✓ EXTRACT from LVN_Implementation.ipynb
│   └── mmd.py                   ✓ EXTRACT from MMD_Production_edition.ipynb
├── execution/
│   ├── __init__.py              ✓ Create empty
│   ├── handler.py               ✓ Order execution management
│   └── risk.py                  ✓ Position sizing and risk limits
└── utils/
    ├── __init__.py              ✓ Create empty (exists)
    ├── logger.py                ✗ ALREADY EXISTS - DO NOT RECREATE
    └── metrics.py               ✓ Performance calculations
```

### Jupyter Notebooks (.ipynb files ONLY)
```
notebooks/
├── training/                    ✓ Create directory
│   ├── 01_lstm_agent_30m.ipynb ✓ LSTM for 30m agent
│   ├── 02_lstm_agent_5m.ipynb  ✓ LSTM for 5m agent
│   ├── 03_regime_mmd.ipynb     ✓ MMD regime training
│   ├── 04_risk_agent.ipynb     ✓ Risk management agent
│   └── 05_mappo_policy.ipynb   ✓ MAPPO training
└── backtesting/                 ✓ Create directory
    └── validation.ipynb         ✓ Strategy validation
```

## FORBIDDEN FILES - DO NOT CREATE
- Any .ipynb files not listed above
- Any test files (unless explicitly requested)
- Any documentation files except this blueprint
- Any config files except config.py
- Any additional utility files
- Any UI/visualization files
- Any database-related files

## CODE EXTRACTION MAPPING

### From Strategy_Implementation.ipynb:
1. MLMI Implementation → src/indicators/mlmi.py
   - Cells: 1331cd64, 381cc2a1
   - Keep ALL numba optimizations
   - Extract: calculate_wma, calculate_rsi_with_ma, MLMICalculator class

2. NW-RQK Implementation → src/indicators/nwrqk.py
   - Cell: 381cc2a1
   - Keep parallel numba processing
   - Extract: rational_quadratic_kernel functions

3. FVG Detection → src/indicators/fvg.py
   - Cells: 2cc4dd54, 64b01841
   - Keep numba optimizations
   - Extract: detect_fvg_patterns, FVGInvalidation class

### From MMD_Production_edition.ipynb:
1. MMD Engine → src/indicators/mmd.py
   - Keep ALL numba JIT optimizations
   - Extract: compute_mmd, gaussian_kernel, label_regimes

### From LVN_Implementation.ipynb:
1. LVN Analysis → src/indicators/lvn.py
   - Extract Phase 1 & 2 only (identification and features)
   - Save ML model for later notebook implementation

## IMPLEMENTATION CHECKPOINTS

### Week 1 Checklist:
□ Create this DEVELOPMENT_BLUEPRINT.md
□ Create directory structure (dirs only)
□ Extract and migrate indicators (5 files)
□ Create event system (events.py)
□ Create config loader (config.py)

### Week 2 Checklist:
□ Create data handlers (handlers.py)
□ Create bar generator (bar_generator.py)
□ Create base indicator class (base.py)
□ Test indicator integrations

### Week 3 Checklist:
□ Create ML training notebooks (6 files)
□ NO OTHER FILES

### DO NOT CREATE YET:
- Full kernel implementation
- Live data handlers
- Broker connections
- Web interfaces
- Additional indicators
- Helper utilities

## VALIDATION QUESTIONS (Ask before creating ANY file)
1. Is this file explicitly listed in APPROVED FILE STRUCTURE?
2. Is this a .py file (for system) or .ipynb file (for ML only)?
3. Am I extracting existing code or writing from scratch?
4. Does this file have a clear Week assignment?
5. Have I checked for existing implementations first?

## STATUS TRACKING
- [x] Blueprint created and reviewed
- [ ] Week 1 files: 0/10 completed
- [ ] Week 2 files: 0/4 completed  
- [ ] Week 3 files: 0/6 completed