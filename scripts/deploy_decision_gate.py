"""
File: scripts/deploy_decision_gate.py (NEW FILE)
Production deployment script for DecisionGate
"""

import torch
import yaml
import logging
import argparse
from pathlib import Path
import sys
import asyncio
import time
import subprocess
import shutil

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.agents.main_core.decision_gate_integration import DecisionGateSystem
from src.agents.main_core.decision_threshold_learning import AdaptiveThresholdLearner
from src.agents.main_core.decision_gate_transformer import DecisionGateTransformer, DecisionOutput
from src.agents.main_core.decision_interpretability import DecisionInterpreter

logger = logging.getLogger(__name__)


def validate_deployment():
    """Validate DecisionGate deployment."""
    print("🔍 Validating DecisionGate deployment...")
    
    # Load configuration
    with open('config/decision_gate_config.yaml', 'r') as f:
        config = yaml.safe_load(f)['decision_gate']
        
    # Initialize system
    system = DecisionGateSystem(config)
    
    # Run validation tests
    tests_passed = True
    
    # Test 1: Model loading
    print("\n1. Testing model loading...")
    try:
        system._load_weights(config['model']['model_path'])
        print("   ✅ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        tests_passed = False
        
    # Test 2: Inference speed
    print("\n2. Testing inference speed...")
    
    test_state = torch.randn(1, 512)
    test_risk = create_test_risk_proposal()
    test_consensus = {
        'should_qualify': torch.tensor([True]),
        'qualify_prob': torch.tensor([0.8])
    }
    test_context = {'regime': 'trending'}
    
    start = time.time()
    result = asyncio.run(system.make_decision(
        test_state,
        test_risk,
        test_consensus,
        test_context
    ))
    elapsed = (time.time() - start) * 1000
    
    if elapsed < config['integration']['timeout_ms']:
        print(f"   ✅ Inference time: {elapsed:.2f}ms")
    else:
        print(f"   ❌ Inference too slow: {elapsed:.2f}ms")
        tests_passed = False
        
    # Test 3: Safety checks
    print("\n3. Testing safety mechanisms...")
    
    # Test with extreme values
    extreme_risk = test_risk.copy()
    extreme_risk['portfolio_heat'] = 0.5  # Way too high
    
    result = asyncio.run(system.make_decision(
        test_state,
        extreme_risk,
        test_consensus,
        test_context
    ))
    
    if result['decision'] == 'REJECT':
        print("   ✅ Safety checks working")
    else:
        print("   ❌ Safety checks failed")
        tests_passed = False
        
    # Test 4: Interpretability
    print("\n4. Testing interpretability...")
    
    if 'interpretation' in result:
        report = system.interpreter.create_decision_report(result['interpretation'])
        if len(report) > 0:
            print("   ✅ Interpretability working")
        else:
            print("   ❌ Interpretability failed")
            tests_passed = False
    else:
        print("   ❌ No interpretation generated")
        tests_passed = False
        
    return tests_passed


def deploy_decision_gate(args):
    """Deploy DecisionGate to production."""
    print("🚀 Deploying DecisionGate Transformer...")
    
    # Validate first
    if not validate_deployment():
        print("\n❌ Validation failed. Aborting deployment.")
        return False
        
    print("\n✅ Validation passed. Proceeding with deployment...")
    
    # Create deployment package
    deployment_path = Path(args.deployment_dir) / 'decision_gate'
    deployment_path.mkdir(parents=True, exist_ok=True)
    
    # Copy necessary files
    files_to_deploy = [
        'src/agents/main_core/decision_gate_transformer.py',
        'src/agents/main_core/decision_gate_attention.py',
        'src/agents/main_core/decision_interpretability.py',
        'src/agents/main_core/decision_threshold_learning.py',
        'src/agents/main_core/decision_gate_integration.py',
        'config/decision_gate_config.yaml'
    ]
    
    for file_path in files_to_deploy:
        src = Path(file_path)
        dst = deployment_path / src.name
        
        if src.exists():
            shutil.copy2(src, dst)
            print(f"   ✅ Deployed: {src.name}")
        else:
            print(f"   ❌ Missing: {src.name}")
            
    # Create startup script
    startup_script = deployment_path / 'start_decision_gate.sh'
    with open(startup_script, 'w') as f:
        f.write("""#!/bin/bash
# DecisionGate startup script

echo "Starting DecisionGate Transformer..."

# Set environment
export PYTHONPATH=/app:$PYTHONPATH
export DECISION_GATE_CONFIG=/app/decision_gate/decision_gate_config.yaml

# Start service
python -m src.agents.main_core.decision_gate_integration \\
    --config $DECISION_GATE_CONFIG \\
    --mode production \\
    --log-level INFO

echo "DecisionGate started."
""")
    
    startup_script.chmod(0o755)
    print(f"   ✅ Created startup script")
    
    # Create Docker image if requested
    if args.docker:
        create_docker_image(deployment_path)
        
    print(f"\n✅ DecisionGate deployed to: {deployment_path}")
    print("\nNext steps:")
    print("1. Copy deployment to production server")
    print("2. Run startup script to begin service")
    print("3. Monitor logs and performance metrics")
    
    return True


def create_docker_image(deployment_path: Path):
    """Create Docker image for DecisionGate."""
    dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy DecisionGate files
COPY decision_gate/ /app/decision_gate/
COPY src/ /app/src/

# Set environment
ENV PYTHONPATH=/app:$PYTHONPATH
ENV DECISION_GATE_CONFIG=/app/decision_gate/decision_gate_config.yaml

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "from src.agents.main_core.decision_gate_integration import DecisionGateSystem; print('OK')"

# Start command
CMD ["python", "-m", "src.agents.main_core.decision_gate_integration", "--mode", "production"]
"""
    
    dockerfile_path = deployment_path / 'Dockerfile'
    with open(dockerfile_path, 'w') as f:
        f.write(dockerfile_content)
        
    print("   ✅ Created Dockerfile")
    
    # Build image
    try:
        subprocess.run(
            ['docker', 'build', '-t', 'algospace/decision-gate:latest', str(deployment_path)],
            check=True
        )
        print("   ✅ Built Docker image: algospace/decision-gate:latest")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Docker build failed: {e}")


def create_test_risk_proposal():
    """Create test risk proposal for validation."""
    return {
        'position_size': 100,
        'position_size_pct': 0.02,
        'leverage': 1.0,
        'dollar_risk': 200,
        'portfolio_heat': 0.06,
        'stop_loss_distance': 20,
        'stop_loss_atr_multiple': 1.5,
        'use_trailing_stop': True,
        'take_profit_distance': 60,
        'risk_reward_ratio': 3.0,
        'expected_return': 600,
        'risk_metrics': {
            'portfolio_risk_score': 0.4,
            'correlation_risk': 0.2,
            'concentration_risk': 0.1,
            'market_risk_multiplier': 1.2
        },
        'confidence_scores': {
            'overall_confidence': 0.75,
            'sl_confidence': 0.8,
            'tp_confidence': 0.7,
            'size_confidence': 0.8
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy DecisionGate Transformer")
    parser.add_argument(
        '--deployment-dir',
        type=str,
        default='deployments',
        help='Directory for deployment files'
    )
    parser.add_argument(
        '--docker',
        action='store_true',
        help='Build Docker image'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only run validation'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.validate_only:
        success = validate_deployment()
        sys.exit(0 if success else 1)
    else:
        success = deploy_decision_gate(args)
        sys.exit(0 if success else 1)