"""
Deployment script for data pipeline components.

This script manages the lifecycle of data pipeline components including
initialization, startup, monitoring, and graceful shutdown.
"""

import asyncio
import signal
import logging
import yaml
import sys
from pathlib import Path
from typing import Dict, Any
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.event_adapter import EventBus
from src.data.data_handler import BacktestDataHandler, LiveDataHandler
from src.data.bar_generator import BarGenerator
from src.data.data_utils import DataQualityMonitor, DataRecorder
from src.utils.monitoring import setup_prometheus_metrics

logger = logging.getLogger(__name__)


class DataPipelineManager:
    """Manages data pipeline components lifecycle."""
    
    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['data_pipeline']
            
        # Initialize event bus
        self.event_bus = EventBus()
        
        # Components
        self.data_handler = None
        self.bar_generator = None
        self.quality_monitor = None
        self.data_recorder = None
        
        # Shutdown flag
        self.shutdown_requested = False
        
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing data pipeline...")
        
        # Create data handler based on mode
        mode = self.config['data_handler']['mode']
        
        if mode == 'backtest':
            handler_config = {
                **self.config['data_handler']['backtest'],
                'symbol': self.config['data_handler']['symbol'],
                **self.config['data_handler']['validation']
            }
            self.data_handler = BacktestDataHandler(handler_config, self.event_bus)
            
        elif mode == 'live':
            handler_config = {
                **self.config['data_handler']['live'],
                'symbol': self.config['data_handler']['symbol'],
                **self.config['data_handler']['validation'],
                'rate_limit': self.config['data_handler'].get('rate_limit')
            }
            self.data_handler = LiveDataHandler(handler_config, self.event_bus)
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        # Create bar generator
        bar_config = {
            'symbol': self.config['bar_generator']['symbol'],
            'timeframes': self.config['bar_generator']['timeframes']
        }
        self.bar_generator = BarGenerator(bar_config, self.event_bus)
        
        # Create quality monitor if enabled
        if self.config['monitoring']['enabled']:
            monitor_config = {
                'symbol': self.config['data_handler']['symbol'],
                **self.config['monitoring']['thresholds']
            }
            self.quality_monitor = DataQualityMonitor(monitor_config)
            
            # Subscribe to events for monitoring
            from src.core.events import EventType
            
            await self.event_bus.subscribe(
                EventType.NEW_TICK,
                lambda e: self.quality_monitor.update_tick(e.data['tick'])
            )
            
            await self.event_bus.subscribe(
                EventType.NEW_5MIN_BAR,
                lambda e: self.quality_monitor.update_bar(e.data['bar'])
            )
            
            await self.event_bus.subscribe(
                EventType.NEW_30MIN_BAR,
                lambda e: self.quality_monitor.update_bar(e.data['bar'])
            )
            
        # Create data recorder if enabled
        if self.config['recording']['enabled']:
            recorder_config = {
                'output_dir': self.config['recording']['output_dir'],
                'buffer_size': self.config['recording']['buffer_size']
            }
            self.data_recorder = DataRecorder(recorder_config)
            
            # Subscribe to events for recording
            await self.event_bus.subscribe(
                EventType.NEW_TICK,
                lambda e: self.data_recorder.record_tick(e.data['tick'])
            )
            
            await self.event_bus.subscribe(
                EventType.NEW_5MIN_BAR,
                lambda e: self.data_recorder.record_bar(e.data['bar'])
            )
            
        # Setup metrics if enabled
        if self.config['metrics']['enabled']:
            setup_prometheus_metrics(
                port=self.config['metrics']['prometheus']['port']
            )
            
        logger.info("Data pipeline initialized successfully")
        
    async def start(self):
        """Start all components."""
        logger.info("Starting data pipeline...")
        
        # Start components in order
        await self.bar_generator.start()
        
        if self.data_recorder:
            await self.data_recorder.start_recording(
                f"session_{self.config['data_handler']['symbol']}"
            )
            
        await self.data_handler.start()
        
        logger.info("Data pipeline started")
        
        # Run until shutdown
        while not self.shutdown_requested:
            await asyncio.sleep(1)
            
            # Periodic health check
            if self.quality_monitor:
                report = self.quality_monitor.get_report()
                if report['health_score'] < 0.5:
                    logger.warning(
                        f"Low data health score: {report['health_score']:.2f}"
                    )
                    
    async def stop(self):
        """Stop all components gracefully."""
        logger.info("Stopping data pipeline...")
        
        # Stop in reverse order
        await self.data_handler.stop()
        
        if self.data_recorder:
            await self.data_recorder.stop_recording()
            
        await self.bar_generator.stop()
        
        # Final quality report
        if self.quality_monitor:
            report = self.quality_monitor.get_report()
            logger.info(f"Final data quality report: {report}")
            
        logger.info("Data pipeline stopped")
        
    def request_shutdown(self):
        """Request graceful shutdown."""
        logger.info("Shutdown requested")
        self.shutdown_requested = True


async def main():
    """Main entry point."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get config path from arguments
    if len(sys.argv) < 2:
        print("Usage: python deploy_data_pipeline.py <config_path>")
        sys.exit(1)
        
    config_path = sys.argv[1]
    
    # Create pipeline manager
    manager = DataPipelineManager(config_path)
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}")
        manager.request_shutdown()
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize
        await manager.initialize()
        
        # Start
        await manager.start()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
        
    finally:
        # Ensure cleanup
        await manager.stop()
        

if __name__ == "__main__":
    asyncio.run(main())