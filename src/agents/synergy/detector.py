"""
SynergyDetector - Hard-Coded Strategy Pattern Detector

This is the main implementation of the SynergyDetector component,
serving as Gate 1 in the two-gate MARL system.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import time

import structlog

from ...core.component_base import ComponentBase
from ...core.kernel import AlgoSpaceKernel
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
    
    # CRITICAL: Define the 4 synergy patterns mapping
    SYNERGY_PATTERNS = {
        ('mlmi', 'nwrqk', 'fvg'): 'TYPE_1',  # Classic momentum continuation
        ('mlmi', 'fvg', 'nwrqk'): 'TYPE_2',  # Early gap with late breakout
        ('nwrqk', 'fvg', 'mlmi'): 'TYPE_3',  # Breakout-gap-momentum sequence
        ('nwrqk', 'mlmi', 'fvg'): 'TYPE_4'   # Range break with momentum
    }
    
    def __init__(self, name: str, kernel: AlgoSpaceKernel):
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
        
        # Extract configuration values
        self.time_window_bars = config.get('time_window_bars', 10)
        self.cooldown_bars = config.get('cooldown_bars', 5)
        self.bar_duration_minutes = config.get('bar_duration_minutes', 5)
        self.required_signals = config.get('required_signals', 3)
        self.processing_time_warning_ms = config.get('processing_time_warning_ms', 1.0)
        
        # Initialize sequence tracking
        self.sequence = SignalSequence(
            time_window_bars=self.time_window_bars,
            bar_duration_minutes=self.bar_duration_minutes,
            required_signals=self.required_signals
        )
        
        # Initialize cooldown tracking
        self.cooldown = CooldownTracker(
            cooldown_bars=self.cooldown_bars,
            bar_duration_minutes=self.bar_duration_minutes
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
        
        logger.info(f"SynergyDetector initialized config={config} time_window_bars={self.time_window_bars} cooldown_bars={self.cooldown_bars}")
    
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
        logger.info(f"SynergyDetector shutting down metrics={self.performance_metrics}")
    
    async def _handle_indicators_ready(self, event: Event):
        """
        Handle INDICATORS_READY event.
        
        Args:
            event: Event containing feature store snapshot
        """
        try:
            features = event.payload
            timestamp = event.timestamp
            
            # Validate features
            if not self._validate_features(features):
                logger.warning(
                    "Invalid features received",
                    event_id=event.id
                )
                return
            
            # Process features for synergy detection
            synergy = self.process_features(features, timestamp)
            
            if synergy:
                logger.info(
                    "Synergy pattern detected via event",
                    synergy_type=synergy.synergy_type,
                    event_id=event.id
                )
        
        except Exception as e:
            logger.error(
                "Error handling INDICATORS_READY event",
                error=str(e),
                event_id=event.id
            )
    
    def _validate_features(self, features: Dict[str, Any]) -> bool:
        """
        Validate that features contain required fields.
        
        Args:
            features: Feature dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = [
            'timestamp', 'current_price',
            'mlmi_signal', 'mlmi_value',
            'nwrqk_signal', 'nwrqk_slope',
            'fvg_mitigation_signal'
        ]
        
        for field in required_fields:
            if field not in features:
                logger.warning(f"Missing required field: {field}")
                return False
        
        return True
    
    def process_features(self, features: Dict[str, Any], timestamp: datetime) -> Optional[SynergyPattern]:
        """
        Process features and detect synergy patterns.
        
        This is the main detection logic that coordinates pattern detection,
        sequence tracking, and cooldown management.
        
        Args:
            features: Feature store snapshot
            timestamp: Current timestamp
            
        Returns:
            SynergyPattern if detected, None otherwise
        """
        start_time = time.perf_counter()
        
        try:
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
                
                logger.debug(
                    "Signal detected",
                    signal_type=signal.signal_type,
                    direction=signal.direction,
                    strength=signal.strength
                )
                
                # Add to sequence (handles validation and resets)
                added = self.sequence.add_signal(signal)
                
                if not added:
                    logger.debug(
                        "Signal rejected by sequence",
                        signal_type=signal.signal_type,
                        reason="duplicate or time window exceeded"
                    )
                    continue
                
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
                        
                        # Log detection
                        logger.info(
                            "SYNERGY DETECTED",
                            synergy_type=synergy.synergy_type,
                            direction=synergy.direction,
                            bars_to_complete=synergy.bars_to_complete,
                            signals=[s.signal_type for s in synergy.signals]
                        )
                        
                        # Emit event
                        self._emit_synergy_event(synergy, features)
                        
                        return synergy
                        
                    elif synergy and not self.cooldown.can_emit():
                        logger.info(
                            "Synergy detected but in cooldown",
                            synergy_type=synergy.synergy_type,
                            remaining_bars=self.cooldown.get_remaining_bars()
                        )
                        # Reset sequence even though we can't emit
                        self.sequence.reset()
                    else:
                        # Complete sequence but not a valid pattern
                        logger.warning(
                            "Invalid pattern detected",
                            pattern=self.sequence.get_pattern()
                        )
                        self.sequence.reset()
        
        finally:
            # Track processing time
            processing_time = (time.perf_counter() - start_time) * 1000
            self._update_performance_metrics(processing_time)
            
            if processing_time > self.processing_time_warning_ms:
                logger.warning(
                    "Slow processing detected",
                    processing_time_ms=processing_time,
                    threshold_ms=self.processing_time_warning_ms
                )
        
        return None
    
    def _detect_signals(self, features: Dict[str, Any]) -> List[Signal]:
        """
        Detect all active signals from current features.
        
        Args:
            features: Feature store snapshot
            
        Returns:
            List of detected signals
        """
        signals = []
        timestamp = features.get('timestamp', datetime.now())
        
        # Check MLMI pattern
        try:
            mlmi_signal = self.mlmi_detector.detect_pattern(features)
            if mlmi_signal and self.mlmi_detector.validate_signal(mlmi_signal, features):
                mlmi_signal.timestamp = timestamp
                signals.append(mlmi_signal)
        except Exception as e:
            logger.error("MLMI detection error", error=str(e))
        
        # Check NW-RQK pattern
        try:
            nwrqk_signal = self.nwrqk_detector.detect_pattern(features)
            if nwrqk_signal and self.nwrqk_detector.validate_signal(nwrqk_signal, features):
                nwrqk_signal.timestamp = timestamp
                signals.append(nwrqk_signal)
        except Exception as e:
            logger.error("NW-RQK detection error", error=str(e))
        
        # Check FVG pattern
        try:
            fvg_signal = self.fvg_detector.detect_pattern(features)
            if fvg_signal and self.fvg_detector.validate_signal(fvg_signal, features):
                fvg_signal.timestamp = timestamp
                signals.append(fvg_signal)
        except Exception as e:
            logger.error("FVG detection error", error=str(e))
        
        return signals
    
    def _check_and_create_synergy(self) -> Optional[SynergyPattern]:
        """
        Check if current sequence forms a valid synergy pattern.
        
        Returns:
            SynergyPattern if valid, None otherwise
        """
        # Verify we have exactly 3 signals
        if len(self.sequence.signals) != self.required_signals:
            return None
        
        # Check direction consistency
        directions = [s.direction for s in self.sequence.signals]
        if not all(d == directions[0] for d in directions):
            logger.debug(
                "Direction mismatch in sequence",
                directions=directions
            )
            return None
        
        # Get the pattern tuple
        pattern = self.sequence.get_pattern()
        synergy_type = self.SYNERGY_PATTERNS.get(pattern)
        
        if not synergy_type:
            logger.warning(
                "Complete sequence but not a valid synergy pattern",
                pattern=pattern,
                valid_patterns=list(self.SYNERGY_PATTERNS.keys())
            )
            return None
        
        # Create synergy pattern
        synergy = SynergyPattern(
            synergy_type=synergy_type,
            direction=self.sequence.get_direction(),
            signals=self.sequence.signals.copy(),
            completion_time=self.sequence.get_completion_time(),
            bars_to_complete=self.sequence.get_bars_to_complete()
        )
        
        logger.debug(
            "Valid synergy pattern created",
            synergy_type=synergy_type,
            pattern=pattern
        )
        
        return synergy
    
    def _emit_synergy_event(self, synergy: SynergyPattern, features: Dict[str, Any]):
        """
        Emit SYNERGY_DETECTED event with full context.
        
        Args:
            synergy: Detected synergy pattern
            features: Current feature store snapshot
        """
        # Build signal sequence with full details
        signal_sequence = []
        for signal in synergy.signals:
            signal_data = {
                'type': signal.signal_type,
                'value': signal.value,
                'signal': signal.direction,
                'timestamp': signal.timestamp,
                'strength': signal.strength,
                'metadata': signal.metadata or {}
            }
            signal_sequence.append(signal_data)
        
        # Extract signal strengths
        signal_strengths = {
            'mlmi': next((s.strength for s in synergy.signals if s.signal_type == 'mlmi'), 0.0),
            'nwrqk': next((s.strength for s in synergy.signals if s.signal_type == 'nwrqk'), 0.0),
            'fvg': next((s.strength for s in synergy.signals if s.signal_type == 'fvg'), 0.0)
        }
        
        # Build market context
        market_context = {
            'current_price': features.get('current_price', 0.0),
            'volatility': features.get('volatility_30', 0.0),
            'volume_ratio': features.get('volume_ratio', 1.0),
            'vix': features.get('vix', 20.0),
            'trend_strength_5m': features.get('trend_strength_5', 0.0),
            'trend_strength_30m': features.get('trend_strength_30', 0.0),
            'nearest_lvn': {
                'price': features.get('lvn_nearest_price', 0.0),
                'strength': features.get('lvn_nearest_strength', 0.0),
                'distance': features.get('lvn_distance_points', 0.0)
            },
            'timestamp': features.get('timestamp', datetime.now())
        }
        
        # Build complete payload
        payload = {
            'synergy_type': synergy.synergy_type,
            'direction': synergy.direction,
            'confidence': synergy.confidence,
            'timestamp': synergy.completion_time,
            'signal_sequence': signal_sequence,
            'signal_strengths': signal_strengths,
            'market_context': market_context,
            'metadata': {
                'bars_to_complete': synergy.bars_to_complete,
                'pattern_quality': self._calculate_pattern_quality(synergy),
                'detection_timestamp': datetime.now()
            }
        }
        
        # Create and publish event
        event = self.kernel.event_bus.create_event(
            EventType.SYNERGY_DETECTED,
            payload,
            source=self.name
        )
        
        self.kernel.event_bus.publish(event)
        
        logger.info(
            "SYNERGY_DETECTED event emitted",
            synergy_type=synergy.synergy_type,
            direction=synergy.direction,
            bars_to_complete=synergy.bars_to_complete
        )
    
    def _calculate_pattern_quality(self, synergy: SynergyPattern) -> float:
        """
        Calculate overall quality score for the pattern.
        
        Args:
            synergy: The synergy pattern
            
        Returns:
            Quality score between 0 and 1
        """
        # Average signal strengths
        avg_strength = sum(s.strength for s in synergy.signals) / len(synergy.signals)
        
        # Time efficiency (faster completion = higher quality)
        time_efficiency = 1.0 - (synergy.bars_to_complete / self.time_window_bars)
        
        # Direction consistency bonus
        direction_bonus = 0.1 if all(s.direction == synergy.direction for s in synergy.signals) else 0.0
        
        # Weighted quality score
        quality = (avg_strength * 0.6 + time_efficiency * 0.3 + direction_bonus)
        
        return min(1.0, max(0.0, quality))
    
    def _update_performance_metrics(self, processing_time_ms: float):
        """Update performance metrics with new measurement."""
        metrics = self.performance_metrics
        
        # Update average processing time
        total_events = metrics['events_processed']
        if total_events > 0:
            current_avg = metrics['avg_processing_time_ms']
            metrics['avg_processing_time_ms'] = (
                (current_avg * (total_events - 1) + processing_time_ms) / total_events
            )
        
        # Update max processing time
        metrics['max_processing_time_ms'] = max(
            metrics['max_processing_time_ms'],
            processing_time_ms
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive component status."""
        return {
            'component': self.name,
            'initialized': self._initialized,
            'performance_metrics': self.performance_metrics.copy(),
            'sequence_status': {
                'signals_count': len(self.sequence.signals),
                'pattern': self.sequence.get_pattern() if self.sequence.signals else None,
                'time_remaining': self.sequence.get_time_remaining() if self.sequence.signals else None
            },
            'cooldown_status': {
                'in_cooldown': self.cooldown.is_in_cooldown(),
                'remaining_bars': self.cooldown.get_remaining_bars()
            },
            'pattern_detectors': {
                'mlmi': self.mlmi_detector._performance_metrics.copy(),
                'nwrqk': self.nwrqk_detector._performance_metrics.copy(),
                'fvg': self.fvg_detector._performance_metrics.copy()
            }
        }
    
    def reset(self):
        """Reset the detector state."""
        self.sequence.reset()
        self.cooldown.reset()
        self._last_features = {}
        logger.info("SynergyDetector reset")