"""
AlgoSpace Main Entry Point - System Orchestration

This module serves as the main entry point for the AlgoSpace MARL Trading System.
It orchestrates the startup, coordination, and shutdown of all system components
according to the Master PRD specifications.

Key Responsibilities:
- Initialize System Kernel
- Register and configure all components
- Manage component dependencies
- Handle graceful startup and shutdown
- Provide system monitoring and health checks
"""

import asyncio
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import core system components
from src.core.kernel import SystemKernel
from src.core.events import EventType, Event
from src.utils.logger import setup_logging, get_logger

# Import data pipeline components
from src.data.handlers import create_data_handler
from src.data.bar_generator import BarGenerator

# Import indicator components
from src.indicators.engine import IndicatorEngine


class AlgoSpaceSystem:
    """
    Main system orchestrator for AlgoSpace
    
    Coordinates all components according to dependency order and
    ensures proper initialization, operation, and shutdown.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize AlgoSpace system
        
        Args:
            config_path: Optional path to configuration file
        """
        # Setup logging first
        setup_logging()
        self.logger = get_logger("AlgoSpaceSystem")
        
        # Initialize system kernel
        self.kernel = SystemKernel(config_path)
        self.config = self.kernel.get_config()
        self.event_bus = self.kernel.get_event_bus()
        
        # Component references
        self.data_handler = None
        self.bar_generator = None
        self.indicator_engine = None
        
        # System state
        self.is_initialized = False
        self.start_time = None
        
        # Subscribe to system events
        self._setup_system_monitoring()
        
        self.logger.info("AlgoSpace System initialized",
                        mode=self.config.system_mode,
                        environment=self.config.get('system.environment', 'development'))
    
    def _setup_system_monitoring(self) -> None:
        """Setup monitoring for critical system events"""
        # Monitor system lifecycle
        self.event_bus.subscribe(EventType.SYSTEM_START, self._on_system_start)
        self.event_bus.subscribe(EventType.SYSTEM_SHUTDOWN, self._on_system_shutdown)
        self.event_bus.subscribe(EventType.SYSTEM_ERROR, self._on_system_error)
        
        # Monitor data flow
        self.event_bus.subscribe(EventType.BACKTEST_COMPLETE, self._on_backtest_complete)
        self.event_bus.subscribe(EventType.CONNECTION_LOST, self._on_connection_lost)
        
        # Monitor indicators
        self.event_bus.subscribe(EventType.INDICATORS_READY, self._on_indicators_ready)
        
        self.logger.debug("System monitoring established")
    
    async def initialize_components(self) -> None:
        """
        Initialize all system components in dependency order
        
        Component initialization order:
        1. DataHandler (no dependencies)
        2. BarGenerator (depends on DataHandler events)
        3. IndicatorEngine (depends on BarGenerator events)
        4. Future: SynergyDetector, MatrixAssemblers, MARL Core, etc.
        """
        try:
            self.logger.info("--- Initializing System Components ---")
            
            # 1. Create and register DataHandler
            self.logger.info("Creating DataHandler...")
            self.data_handler = create_data_handler(self.kernel)
            self.kernel.register_component(
                name="DataHandler",
                component=self.data_handler,
                dependencies=[]
            )
            
            # 2. Create and register BarGenerator
            self.logger.info("Creating BarGenerator...")
            self.bar_generator = BarGenerator(
                name="BarGenerator",
                kernel=self.kernel
            )
            self.kernel.register_component(
                name="BarGenerator",
                component=self.bar_generator,
                dependencies=["DataHandler"]
            )
            
            # 3. Create and register IndicatorEngine
            self.logger.info("Creating IndicatorEngine...")
            self.indicator_engine = IndicatorEngine(
                name="IndicatorEngine",
                kernel=self.kernel
            )
            self.kernel.register_component(
                name="IndicatorEngine",
                component=self.indicator_engine,
                dependencies=["BarGenerator"]
            )
            
            # Future components would be registered here:
            # - SynergyDetector (depends on IndicatorEngine)
            # - MatrixAssemblers (depends on IndicatorEngine)
            # - RegimeDetectionEngine (depends on IndicatorEngine)
            # - RiskManagementSubsystem (depends on SynergyDetector)
            # - MainMARLCore (depends on all above)
            # - ExecutionHandler (depends on MainMARLCore)
            
            self.is_initialized = True
            
            self.logger.info("--- All Components Initialized Successfully ---",
                           total_components=len(self.kernel.components))
            
            # Log component summary
            self._log_component_summary()
            
        except Exception as e:
            self.logger.error("Failed to initialize components", error=str(e))
            raise
    
    def _log_component_summary(self) -> None:
        """Log summary of initialized components"""
        summary = {
            "total_components": len(self.kernel.components),
            "components": list(self.kernel.components.keys()),
            "mode": self.config.system_mode,
            "symbol": self.config.primary_symbol,
            "timeframes": self.config.timeframes
        }
        
        if self.config.is_backtest:
            summary["backtest_file"] = self.config.get('data_handler.backtest_file', 'Not specified')
            summary["start_date"] = self.config.get('backtesting.start_date', 'Not specified')
            summary["end_date"] = self.config.get('backtesting.end_date', 'Not specified')
        
        self.logger.info("Component initialization summary", **summary)
    
    async def start(self) -> None:
        """
        Start the AlgoSpace system
        
        This method:
        1. Initializes all components
        2. Starts the system kernel
        3. Begins processing
        """
        try:
            self.logger.info("="*60)
            self.logger.info("        AlgoSpace Trading System Starting")
            self.logger.info("="*60)
            
            self.start_time = datetime.now()
            
            # Initialize components if not already done
            if not self.is_initialized:
                await self.initialize_components()
            
            # Start the kernel (which starts all components)
            await self.kernel.start()
            
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
            await self.shutdown()
        except Exception as e:
            self.logger.critical("System startup failed", error=str(e))
            await self.shutdown()
            raise
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the system"""
        try:
            self.logger.info("="*60)
            self.logger.info("      AlgoSpace System Shutdown Initiated")
            self.logger.info("="*60)
            
            # Calculate uptime
            if self.start_time:
                uptime = datetime.now() - self.start_time
                self.logger.info(f"System uptime: {uptime}")
            
            # Shutdown kernel (handles component shutdown)
            await self.kernel.shutdown()
            
        except Exception as e:
            self.logger.error("Error during shutdown", error=str(e))
        finally:
            self.logger.info("AlgoSpace shutdown complete")
    
    # Event handlers for system monitoring
    
    def _on_system_start(self, event: Event) -> None:
        """Handle system start event"""
        self.logger.info("System started successfully", 
                        timestamp=event.payload.get('timestamp'),
                        components=event.payload.get('components'))
    
    def _on_system_shutdown(self, event: Event) -> None:
        """Handle system shutdown event"""
        self.logger.info("System shutting down",
                        uptime_seconds=event.payload.get('uptime_seconds'))
    
    def _on_system_error(self, event: Event) -> None:
        """Handle system error event"""
        self.logger.error("SYSTEM ERROR",
                         component=event.payload.get('component'),
                         error=event.payload.get('error'),
                         context=event.payload.get('context'))
    
    def _on_backtest_complete(self, event: Event) -> None:
        """Handle backtest completion"""
        self.logger.info("Backtest completed",
                        total_bars=event.payload.get('total_bars'),
                        total_ticks=event.payload.get('total_ticks'))
        
        # Initiate graceful shutdown after backtest
        asyncio.create_task(self.shutdown())
    
    def _on_connection_lost(self, event: Event) -> None:
        """Handle connection lost event"""
        self.logger.warning("Connection lost - attempting recovery",
                          timestamp=event.timestamp,
                          source=event.source)
    
    def _on_indicators_ready(self, event: Event) -> None:
        """Handle indicators ready event"""
        feature_count = event.payload.get('feature_count', 0)
        
        # Log every 100th event to avoid spam
        if not hasattr(self, '_indicator_event_count'):
            self._indicator_event_count = 0
        
        self._indicator_event_count += 1
        
        if self._indicator_event_count % 100 == 0:
            self.logger.info(f"Indicators processed: {self._indicator_event_count} events",
                           feature_count=feature_count)


async def run_system(config_path: str = None) -> None:
    """
    Main system runner
    
    Args:
        config_path: Optional configuration file path
    """
    system = AlgoSpaceSystem(config_path)
    
    try:
        await system.start()
    except Exception as e:
        logger = get_logger("main")
        logger.critical("System failed to start", error=str(e))
        sys.exit(1)


def main():
    """
    Main entry point with CLI argument parsing
    """
    parser = argparse.ArgumentParser(
        description="AlgoSpace MARL Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run with default config
  python main.py -c custom.yaml     # Run with custom config
  python main.py --debug           # Run with debug logging
        """
    )
    
    parser.add_argument(
        '-c', '--config',
        type=str,
        help='Path to configuration file (default: config/settings.yaml)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='AlgoSpace Trading System v0.1.0'
    )
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        import os
        os.environ['LOG_LEVEL'] = 'DEBUG'
    
    # Print startup banner
    print("\n" + "="*60)
    print("         AlgoSpace MARL Trading System v0.1.0")
    print("="*60 + "\n")
    
    # Run the system
    try:
        asyncio.run(run_system(args.config))
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()