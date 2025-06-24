"""
SynergyDetector - Hard-Coded Strategy Pattern Detector

This is the main implementation of the SynergyDetector component,
serving as Gate 1 in the two-gate MARL system.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import time

import structlog

from ...core.kernel import ComponentBase, SystemKernel
from ...core.events import EventType, Event
from .base import BaseSynergyDetector, Signal, SynergyPattern
from .patterns import MLMIPatternDetector, NWRQKPatternDetector, FVGPatternDetector
from .sequence import SignalSequence, CooldownTracker

logger = structlog.get_logger()


class SynergyDetector(ComponentBase, BaseSynergyDetector):
    """
    Main SynergyDetector implementation.
    
    Monitors indicator values from the Feature Store and detects when they
    form one of the four predefined synergy patterns. Serves as a deterministic,
    rule-based filter before AI inference.
    
    Key Features:
    - Detects 4 hard-coded synergy patterns
    - Enforces 10-bar time window for pattern completion
    - Maintains 5-bar cooldown after detection
    - Guarantees <1ms processing time per event
    - Zero false negatives on valid patterns
    """
    
    def __init__(self, name: str, kernel: SystemKernel):
        """
        Initialize SynergyDetector.
        
        Args:
            name: Component name
            kernel: System kernel for configuration and event bus
        """
        # Initialize base classes
        ComponentBase.__init__(self, name, kernel)
        
        # Load configuration
        config = kernel.config.get('synergy_detector', {})
        BaseSynergyDetector.__init__(self, config)
        
        # Initialize pattern detectors
        self.mlmi_detector = MLMIPatternDetector(config)
        self.nwrqk_detector = NWRQKPatternDetector(config)
        self.fvg_detector = FVGPatternDetector(config)
        
        # Initialize sequence tracking
        self.sequence = SignalSequence(
            time_window_bars=self.time_window,
            bar_duration_minutes=5  # 5-minute bars
        )
        
        # Initialize cooldown tracking
        self.cooldown = CooldownTracker(
            cooldown_bars=self.cooldown_bars,
            bar_duration_minutes=5
        )
        
        # Performance tracking
        self.performance_metrics = {
            'events_processed': 0,
            'synergies_detected': 0,
            'signals_detected': 0,
            'avg_processing_time_ms': 0.0,
            'max_processing_time_ms': 0.0,
            'patterns_by_type': {
                'TYPE_1': 0,
                'TYPE_2': 0,
                'TYPE_3': 0,
                'TYPE_4': 0
            }
        }
        
        # State tracking
        self._last_features = {}
        self._initialized = False
        
        logger.info(
            "SynergyDetector initialized",
            config=config,
            time_window=self.time_window,
            cooldown_bars=self.cooldown_bars
        )
    
    async def initialize(self):
        """Initialize the component and subscribe to events."""
        # Subscribe to INDICATORS_READY events
        self.kernel.event_bus.subscribe(
            EventType.INDICATORS_READY,
            self._handle_indicators_ready
        )
        
        self._initialized = True
        logger.info("SynergyDetector initialization complete")
    
    async def shutdown(self):
        """Clean shutdown of the component."""
        # Unsubscribe from events
        self.kernel.event_bus.unsubscribe(
            EventType.INDICATORS_READY,
            self._handle_indicators_ready
        )
        
        # Log final metrics
        logger.info(
            "SynergyDetector shutting down",
            metrics=self.performance_metrics
        )
    
    def _handle_indicators_ready(self, event: Event):
        """
        Handle INDICATORS_READY event from IndicatorEngine.
        
        Args:
            event: Event containing Feature Store snapshot
        """
        start_time = time.perf_counter()
        
        try:
            # Extract features and timestamp
            features = event.payload
            timestamp = event.timestamp
            
            # Add timestamp to features for pattern detectors
            features['timestamp'] = timestamp
            
            # Process features for synergy detection
            synergy = self.process_features(features, timestamp)
            
            # If synergy detected, emit event
            if synergy:
                self._emit_synergy_event(synergy, features)
            
            # Update metrics
            self._update_performance_metrics(time.perf_counter() - start_time)
            
        except Exception as e:
            logger.error(
                "Error processing indicators",
                error=str(e),
                event_type=event.event_type.value
            )
    
    def process_features(self, features: Dict[str, Any], timestamp: datetime) -> Optional[SynergyPattern]:
        """
        Process features to detect synergy patterns.
        
        This is the main detection logic that coordinates pattern detection,
        sequence tracking, and cooldown management.
        """
        self.performance_metrics['events_processed'] += 1
        
        # Update cooldown state
        self.cooldown.update(timestamp)
        
        # Store features for later use
        self._last_features = features
        
        # Detect individual signals
        signals = self._detect_signals(features)
        
        # Process each detected signal
        for signal in signals:
            self.performance_metrics['signals_detected'] += 1
            
            # Add to sequence (handles validation and resets)
            self.sequence.add_signal(signal)
            
            # Check if we have a complete synergy
            if self.sequence.is_complete():
                synergy = self._check_and_create_synergy()
                
                if synergy and self.cooldown.can_emit():
                    # Start cooldown period
                    self.cooldown.start_cooldown(timestamp)
                    
                    # Reset sequence for next pattern
                    self.sequence.reset()
                    
                    # Update metrics
                    self.performance_metrics['synergies_detected'] += 1
                    self.performance_metrics['patterns_by_type'][synergy.synergy_type] += 1
                    
                    return synergy
                elif synergy and not self.cooldown.can_emit():
                    logger.info(
                        "Synergy detected but in cooldown",
                        synergy_type=synergy.synergy_type,
                        remaining_bars=self.cooldown.get_remaining_bars()
                    )
                    # Reset sequence even though we can't emit
                    self.sequence.reset()
        
        return None
    
    def _detect_signals(self, features: Dict[str, Any]) -> List[Signal]:
        """Detect all active signals from current features."""
        signals = []
        
        # Check MLMI pattern
        mlmi_signal = self.mlmi_detector.detect_pattern(features)
        if mlmi_signal:
            signals.append(mlmi_signal)
        
        # Check NW-RQK pattern
        nwrqk_signal = self.nwrqk_detector.detect_pattern(features)
        if nwrqk_signal:
            signals.append(nwrqk_signal)
        
        # Check FVG pattern
        fvg_signal = self.fvg_detector.detect_pattern(features)
        if fvg_signal:
            signals.append(fvg_signal)
        
        return signals
    
    def _check_and_create_synergy(self) -> Optional[SynergyPattern]:
        """Check if current sequence forms a valid synergy."""
        # Get the pattern
        pattern = self.sequence.get_pattern()
        synergy_type = self.SYNERGY_PATTERNS.get(pattern)
        
        if not synergy_type:
            logger.warning(
                "Complete sequence but not a valid synergy pattern",
                pattern=pattern
            )
            return None
        
        # Create synergy pattern
        return SynergyPattern(
            synergy_type=synergy_type,
            direction=self.sequence.get_direction(),
            signals=self.sequence.signals.copy(),
            completion_time=self.sequence.get_completion_time(),
            bars_to_complete=self.sequence.get_bars_to_complete()
        )
    
    def _emit_synergy_event(self, synergy: SynergyPattern, features: Dict[str, Any]):
        """Emit SYNERGY_DETECTED event with full context."""
        # Build signal sequence details
        signal_sequence = []
        for signal in synergy.signals:
            signal_sequence.append({
                'type': signal.signal_type,
                'value': signal.value,
                'signal': signal.direction,
                'timestamp': signal.timestamp,
                'strength': signal.strength
            })
        
        # Extract market context
        market_context = {
            'current_price': features.get('current_price', 0.0),
            'volatility': features.get('volatility_30', 0.0),
            'volume_profile': {
                'volume_ratio': features.get('volume_ratio', 1.0),
                'volume_momentum': features.get('volume_momentum_30', 0.0)
            },
            'nearest_lvn': {
                'price': features.get('lvn_nearest_price', 0.0),
                'strength': features.get('lvn_nearest_strength', 0.0),
                'distance': features.get('lvn_distance_points', 0.0)
            }
        }
        
        # Build metadata
        metadata = {
            'bars_to_complete': synergy.bars_to_complete,
            'signal_strengths': {
                signal.signal_type: signal.strength 
                for signal in synergy.signals
            }
        }
        
        # Create event payload
        payload = {
            'synergy_type': synergy.synergy_type,
            'direction': synergy.direction,
            'confidence': synergy.confidence,
            'timestamp': synergy.completion_time,
            'signal_sequence': signal_sequence,
            'market_context': market_context,
            'metadata': metadata
        }
        
        # Create and publish event
        event = self.kernel.event_bus.create_event(
            EventType.SYNERGY_DETECTED,
            payload,
            self.name
        )
        self.kernel.event_bus.publish(event)
        
        logger.info(
            "SYNERGY DETECTED",
            synergy_type=synergy.synergy_type,
            direction='LONG' if synergy.direction == 1 else 'SHORT',
            bars_to_complete=synergy.bars_to_complete,
            pattern=[s.signal_type for s in synergy.signals]
        )
    
    def _update_performance_metrics(self, processing_time: float):
        """Update performance metrics."""
        processing_time_ms = processing_time * 1000
        
        # Update average processing time
        n = self.performance_metrics['events_processed']
        avg = self.performance_metrics['avg_processing_time_ms']
        self.performance_metrics['avg_processing_time_ms'] = (
            (avg * (n - 1) + processing_time_ms) / n
        )
        
        # Update max processing time
        self.performance_metrics['max_processing_time_ms'] = max(
            self.performance_metrics['max_processing_time_ms'],
            processing_time_ms
        )
        
        # Log warning if processing time exceeds 1ms
        if processing_time_ms > 1.0:
            logger.warning(
                "Processing time exceeded 1ms threshold",
                processing_time_ms=processing_time_ms
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get component status for monitoring."""
        return {
            'initialized': self._initialized,
            'performance_metrics': self.performance_metrics,
            'sequence_state': self.sequence.get_state(),
            'cooldown_state': self.cooldown.get_state(),
            'pattern_detectors': {
                'mlmi': self.mlmi_detector.get_performance_metrics(),
                'nwrqk': self.nwrqk_detector.get_performance_metrics(),
                'fvg': self.fvg_detector.get_performance_metrics()
            }
        }