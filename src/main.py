# src/main.py
"""
The main entry point for the AlgoSpace application.
This script initializes and runs the System Kernel.
"""
import sys
import signal
import argparse
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.kernel import AlgoSpaceKernel

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='AlgoSpace Trading System - Main Entry Point',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main                          # Run with default config
  python -m src.main --config my_config.yaml  # Run with custom config
  python -m src.main --dry-run               # Initialize only, don't run
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/settings.yaml',
        help='Path to configuration file (default: config/settings.yaml)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Initialize system but do not start trading'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    
    return parser.parse_args()


def setup_signal_handlers(kernel: AlgoSpaceKernel):
    """
    Setup signal handlers for graceful shutdown.
    
    Args:
        kernel: The AlgoSpace kernel instance.
    """
    def handle_shutdown(sig, frame):
        logger.info(f"Signal {sig} received. Initiating graceful shutdown...")
        kernel.shutdown()
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)   # Ctrl+C
    signal.signal(signal.SIGTERM, handle_shutdown)  # Termination signal
    
    # Windows doesn't support SIGHUP
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, handle_shutdown)  # Hangup signal


def ensure_directories():
    """Ensure required directories exist."""
    directories = [
        'logs',
        'data/historical',
        'data/realtime',
        'models/saved',
        'models/checkpoints',
        'output/trades',
        'output/reports',
        'config'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("Required directories verified/created")


def main():
    """
    Main entry point for the AlgoSpace trading system.
    
    This function:
    1. Parses command line arguments
    2. Sets up logging
    3. Creates required directories
    4. Initializes the kernel
    5. Sets up signal handlers
    6. Runs the system
    """
    # Parse arguments
    args = parse_arguments()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("="*60)
    logger.info("AlgoSpace Trading System Starting")
    logger.info("="*60)
    logger.info(f"Configuration file: {args.config}")
    logger.info(f"Log level: {args.log_level}")
    logger.info(f"Dry run: {args.dry_run}")
    
    # Ensure directories exist
    ensure_directories()
    
    # Create kernel instance
    kernel = AlgoSpaceKernel(config_path=args.config)
    
    # Setup signal handlers
    setup_signal_handlers(kernel)
    
    try:
        # Initialize the kernel
        logger.info("Initializing AlgoSpace kernel...")
        kernel.initialize()
        
        if args.dry_run:
            logger.info("Dry run mode - system initialized but not started")
            status = kernel.get_status()
            logger.info(f"System status: {status}")
            logger.info("Dry run complete. Exiting.")
            sys.exit(0)
        
        # Run the system
        logger.info("Starting AlgoSpace trading system...")
        kernel.run()
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        kernel.shutdown()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Critical error in main: {e}", exc_info=True)
        kernel.shutdown()
        sys.exit(1)
    finally:
        logger.info("AlgoSpace system terminated")


if __name__ == "__main__":
    main()