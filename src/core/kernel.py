# src/core/kernel.py
"""
The System Kernel & Orchestration class. This is the master conductor.
"""
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from .config import load_config
from .event_bus import EventBus

# Component imports - these will be replaced with actual imports as they are developed
try:
    from ..components.data_handler import LiveDataHandler, BacktestDataHandler
except ImportError:
    LiveDataHandler = BacktestDataHandler = None

try:
    from ..components.bar_generator import BarGenerator
except ImportError:
    BarGenerator = None

try:
    from ..components.indicator_engine import IndicatorEngine
except ImportError:
    IndicatorEngine = None

try:
    from ..components.matrix_assembler import MatrixAssembler30m, MatrixAssembler5m, MatrixAssemblerRegime
except ImportError:
    MatrixAssembler30m = MatrixAssembler5m = MatrixAssemblerRegime = None

try:
    from ..agents.regime_engine import RegimeDetectionEngine
except ImportError:
    RegimeDetectionEngine = None

try:
    from ..agents.risk_manager import RiskManagementSubsystem
except ImportError:
    RiskManagementSubsystem = None

try:
    from ..agents.marl_core import MainMARLCore, SynergyDetector
except ImportError:
    MainMARLCore = SynergyDetector = None

try:
    from ..components.execution_handler import LiveExecutionHandler, BacktestExecutionHandler
except ImportError:
    LiveExecutionHandler = BacktestExecutionHandler = None

logger = logging.getLogger(__name__)


class AlgoSpaceKernel:
    """
    The main system kernel that orchestrates all components of the AlgoSpace trading system.
    
    This class is responsible for:
    - Loading configuration
    - Instantiating all system components
    - Wiring components together via the event bus
    - Managing the system lifecycle
    """
    
    def __init__(self, config_path: str = 'config/settings.yaml'):
        """
        Initializes the Kernel, but does not start it.
        
        Args:
            config_path: Path to the system configuration file.
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.event_bus = EventBus()
        self.components: Dict[str, Any] = {}
        self.running = False
        
        # Configure logging
        self._setup_logging()
        
        logger.info(f"AlgoSpace Kernel initialized with config path: {config_path}")

    def _setup_logging(self) -> None:
        """Configure logging for the system."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/algospace.log', mode='a')
            ]
        )

    def initialize(self) -> None:
        """
        Initializes and wires all system components in the correct dependency order.
        
        Raises:
            Exception: If initialization fails at any stage.
        """
        try:
            logger.info("=== AlgoSpace System Initialization Starting ===")
            
            # Load configuration
            self.config = load_config(self.config_path)
            logger.info("Configuration loaded successfully")
            
            # Phase 1: Component Instantiation
            logger.info("\n--- Phase 1: Component Instantiation ---")
            self._instantiate_components()
            
            # Phase 2: Event Wiring
            logger.info("\n--- Phase 2: Event Wiring ---")
            self._wire_events()
            
            # Phase 3: Component Initialization
            logger.info("\n--- Phase 3: Component Initialization ---")
            self._initialize_components()
            
            logger.info("\n=== AlgoSpace Initialization Complete. System is READY. ===")
            
        except Exception as e:
            logger.error(f"Kernel initialization failed: {e}", exc_info=True)
            self.shutdown()
            raise

    def _instantiate_components(self) -> None:
        """Instantiates all system components based on configuration."""
        # Data Pipeline
        mode = self.config['data']['mode']
        logger.info(f"Instantiating components for mode: {mode}")
        
        if mode == 'live':
            if LiveDataHandler:
                self.components['data_handler'] = LiveDataHandler(self.config, self.event_bus)
                logger.info("LiveDataHandler instantiated")
            else:
                logger.warning("LiveDataHandler not available")
        else:
            if BacktestDataHandler:
                self.components['data_handler'] = BacktestDataHandler(self.config, self.event_bus)
                logger.info("BacktestDataHandler instantiated")
            else:
                logger.warning("BacktestDataHandler not available")
        
        # Bar Generation
        if BarGenerator:
            self.components['bar_generator'] = BarGenerator(self.config, self.event_bus)
            logger.info("BarGenerator instantiated")
        
        # Indicator Engine
        if IndicatorEngine:
            self.components['indicator_engine'] = IndicatorEngine(self.config, self.event_bus)
            logger.info("IndicatorEngine instantiated")
        
        # Feature Preparation - Matrix Assemblers
        if MatrixAssembler30m:
            self.components['matrix_30m'] = MatrixAssembler30m(self.config)
            logger.info("MatrixAssembler30m instantiated")
            
        if MatrixAssembler5m:
            self.components['matrix_5m'] = MatrixAssembler5m(self.config)
            logger.info("MatrixAssembler5m instantiated")
            
        if MatrixAssemblerRegime:
            self.components['matrix_regime'] = MatrixAssemblerRegime(self.config)
            logger.info("MatrixAssemblerRegime instantiated")
        
        # Intelligence Layer
        if SynergyDetector:
            self.components['synergy_detector'] = SynergyDetector(self.config, self.event_bus)
            logger.info("SynergyDetector instantiated")
        
        # Pre-trained models
        if RegimeDetectionEngine:
            self.components['rde'] = RegimeDetectionEngine(self.config)
            logger.info("RegimeDetectionEngine instantiated")
        
        if RiskManagementSubsystem:
            self.components['m_rms'] = RiskManagementSubsystem(self.config)
            logger.info("RiskManagementSubsystem instantiated")
        
        # Main MARL Core
        if MainMARLCore:
            self.components['main_marl_core'] = MainMARLCore(self.config, self.components)
            logger.info("MainMARLCore instantiated")
        
        # Execution Layer
        exec_mode = self.config.get('execution', {}).get('mode', 'backtest')
        if exec_mode == 'live':
            if LiveExecutionHandler:
                self.components['execution_handler'] = LiveExecutionHandler(self.config, self.event_bus)
                logger.info("LiveExecutionHandler instantiated")
        else:
            if BacktestExecutionHandler:
                self.components['execution_handler'] = BacktestExecutionHandler(self.config, self.event_bus)
                logger.info("BacktestExecutionHandler instantiated")
        
        logger.info(f"Total components instantiated: {len(self.components)}")

    def _wire_events(self) -> None:
        """Connects all components via event subscriptions."""
        # Data Flow Events
        if 'bar_generator' in self.components:
            self.event_bus.subscribe('NEW_TICK', self.components['bar_generator'].on_new_tick)
            logger.info("Wired: NEW_TICK -> BarGenerator")
        
        if 'indicator_engine' in self.components:
            self.event_bus.subscribe('NEW_5MIN_BAR', self.components['indicator_engine'].on_new_bar)
            self.event_bus.subscribe('NEW_30MIN_BAR', self.components['indicator_engine'].on_new_bar)
            logger.info("Wired: NEW_*_BAR -> IndicatorEngine")
        
        # Matrix Assembly Events
        for matrix_name in ['matrix_30m', 'matrix_5m', 'matrix_regime']:
            if matrix_name in self.components:
                self.event_bus.subscribe('INDICATORS_READY', 
                                       self.components[matrix_name].on_indicators_ready)
                logger.info(f"Wired: INDICATORS_READY -> {matrix_name}")
        
        # Decision Flow Events
        if 'synergy_detector' in self.components:
            self.event_bus.subscribe('INDICATORS_READY', 
                                   self.components['synergy_detector'].check_synergy)
            logger.info("Wired: INDICATORS_READY -> SynergyDetector")
        
        if 'main_marl_core' in self.components:
            self.event_bus.subscribe('SYNERGY_DETECTED', 
                                   self.components['main_marl_core'].initiate_qualification)
            logger.info("Wired: SYNERGY_DETECTED -> MainMARLCore")
        
        if 'execution_handler' in self.components:
            self.event_bus.subscribe('EXECUTE_TRADE', 
                                   self.components['execution_handler'].execute_trade)
            logger.info("Wired: EXECUTE_TRADE -> ExecutionHandler")
        
        # Feedback Loop Events
        if 'main_marl_core' in self.components:
            self.event_bus.subscribe('TRADE_CLOSED', 
                                   self.components['main_marl_core'].record_outcome)
            logger.info("Wired: TRADE_CLOSED -> MainMARLCore")
        
        # System Events
        self.event_bus.subscribe('SYSTEM_ERROR', self._handle_system_error)
        self.event_bus.subscribe('SHUTDOWN_REQUEST', lambda _: self.shutdown())
        
        logger.info("Event wiring completed")

    def _initialize_components(self) -> None:
        """Initialize components that require post-instantiation setup."""
        # Load pre-trained models
        if 'rde' in self.components:
            model_path = self.config.get('models', {}).get('rde_path')
            if model_path and hasattr(self.components['rde'], 'load_model'):
                try:
                    self.components['rde'].load_model(model_path)
                    logger.info(f"RDE model loaded from: {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load RDE model: {e}")
        
        if 'm_rms' in self.components:
            model_path = self.config.get('models', {}).get('mrms_path')
            if model_path and hasattr(self.components['m_rms'], 'load_model'):
                try:
                    self.components['m_rms'].load_model(model_path)
                    logger.info(f"M-RMS model loaded from: {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load M-RMS model: {e}")
        
        if 'main_marl_core' in self.components:
            if hasattr(self.components['main_marl_core'], 'load_models'):
                try:
                    self.components['main_marl_core'].load_models()
                    logger.info("MARL models loaded")
                except Exception as e:
                    logger.error(f"Failed to load MARL models: {e}")

    def _handle_system_error(self, error_info: Dict[str, Any]) -> None:
        """
        Handles system-wide errors.
        
        Args:
            error_info: Dictionary containing error details.
        """
        logger.error(f"System error: {error_info}")
        
        # Determine if error is critical
        if error_info.get('critical', False):
            logger.critical("Critical error detected. Initiating shutdown.")
            self.shutdown()

    def run(self) -> None:
        """Starts the main system loop."""
        if not self.components:
            raise RuntimeError("Kernel not initialized. Call initialize() first.")
        
        self.running = True
        logger.info("\n=== AlgoSpace System Running ===")
        
        try:
            # Start data stream
            if 'data_handler' in self.components:
                logger.info("Starting data stream...")
                if hasattr(self.components['data_handler'], 'start_stream'):
                    self.components['data_handler'].start_stream()
            
            # Run the event loop
            logger.info("Starting event dispatcher...")
            self.event_bus.dispatch_forever()
            
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            self.shutdown()
        except Exception as e:
            logger.error(f"Critical error in main loop: {e}", exc_info=True)
            self.shutdown()

    def shutdown(self) -> None:
        """Initiates a graceful shutdown of the system."""
        if not self.running:
            return
            
        logger.info("\n=== Graceful Shutdown Initiated ===")
        self.running = False
        
        try:
            # Stop data streams
            if 'data_handler' in self.components:
                if hasattr(self.components['data_handler'], 'stop_stream'):
                    self.components['data_handler'].stop_stream()
                    logger.info("Data stream stopped")
            
            # Close all positions
            if 'execution_handler' in self.components:
                if hasattr(self.components['execution_handler'], 'close_all_positions'):
                    self.components['execution_handler'].close_all_positions()
                    logger.info("All positions closed")
            
            # Save component states
            for name, component in self.components.items():
                if hasattr(component, 'save_state'):
                    try:
                        component.save_state()
                        logger.info(f"State saved for: {name}")
                    except Exception as e:
                        logger.error(f"Failed to save state for {name}: {e}")
            
            # Stop event bus
            self.event_bus.stop()
            logger.info("Event bus stopped")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)
        
        logger.info("=== System Shutdown Complete ===")

    def get_component(self, name: str) -> Optional[Any]:
        """
        Retrieves a component by name.
        
        Args:
            name: The component name.
            
        Returns:
            The component instance or None if not found.
        """
        return self.components.get(name)

    def get_status(self) -> Dict[str, Any]:
        """
        Returns the current system status.
        
        Returns:
            Dictionary containing system status information.
        """
        return {
            'running': self.running,
            'mode': self.config.get('data', {}).get('mode', 'unknown'),
            'components': list(self.components.keys()),
            'event_queue_size': self.event_bus.event_queue.qsize(),
        }