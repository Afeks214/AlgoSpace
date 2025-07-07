"""
Utilities for data handling and monitoring.

This module provides production utilities for data quality monitoring,
recording, and mock data generation for testing.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from collections import deque
import logging
import json
from pathlib import Path

from src.data.data_handler import TickData
from src.data.bar_generator import BarData

logger = logging.getLogger(__name__)


class DataQualityMonitor:
    """
    Monitors data quality in real-time.
    
    Tracks:
    - Tick arrival rates
    - Price anomalies
    - Gap frequency
    - Latency metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.symbol = config['symbol']
        
        # Sliding windows
        self.tick_window = deque(maxlen=1000)
        self.bar_windows = {
            5: deque(maxlen=100),
            30: deque(maxlen=20)
        }
        
        # Anomaly detection
        self.price_mean = None
        self.price_std = None
        self.volume_mean = None
        self.volume_std = None
        
        # Metrics
        self.metrics = {
            'total_ticks': 0,
            'anomalous_ticks': 0,
            'gaps_detected': 0,
            'synthetic_bars': 0,
            'avg_tick_rate': 0.0,
            'max_price_spike': 0.0
        }
        
    def update_tick(self, tick: TickData):
        """Update monitor with new tick."""
        self.tick_window.append({
            'timestamp': tick.timestamp,
            'price': tick.price,
            'volume': tick.volume
        })
        
        self.metrics['total_ticks'] += 1
        
        # Check for anomalies
        if self._is_anomalous_tick(tick):
            self.metrics['anomalous_ticks'] += 1
            logger.warning(f"Anomalous tick detected: {tick}")
            
        # Update statistics
        self._update_statistics()
        
    def update_bar(self, bar: BarData):
        """Update monitor with new bar."""
        self.bar_windows[bar.timeframe].append({
            'timestamp': bar.timestamp,
            'ohlcv': (bar.open, bar.high, bar.low, bar.close, bar.volume),
            'is_synthetic': bar.is_synthetic
        })
        
        if bar.is_synthetic:
            self.metrics['synthetic_bars'] += 1
            
    def _is_anomalous_tick(self, tick: TickData) -> bool:
        """Check if tick is anomalous."""
        if self.price_mean is None:
            return False
            
        # Price spike detection (z-score)
        z_score = abs(tick.price - self.price_mean) / (self.price_std + 1e-8)
        
        if z_score > 3:
            spike_pct = abs(tick.price - self.price_mean) / self.price_mean
            self.metrics['max_price_spike'] = max(
                self.metrics['max_price_spike'],
                spike_pct
            )
            return True
            
        # Volume spike detection
        if self.volume_mean is not None:
            vol_z_score = abs(tick.volume - self.volume_mean) / (self.volume_std + 1e-8)
            if vol_z_score > 4:
                return True
                
        return False
        
    def _update_statistics(self):
        """Update rolling statistics."""
        if len(self.tick_window) < 10:
            return
            
        prices = [t['price'] for t in self.tick_window]
        volumes = [t['volume'] for t in self.tick_window]
        
        self.price_mean = np.mean(prices)
        self.price_std = np.std(prices)
        self.volume_mean = np.mean(volumes)
        self.volume_std = np.std(volumes)
        
        # Calculate tick rate
        if len(self.tick_window) > 1:
            time_span = (
                self.tick_window[-1]['timestamp'] - 
                self.tick_window[0]['timestamp']
            ).total_seconds()
            
            if time_span > 0:
                self.metrics['avg_tick_rate'] = len(self.tick_window) / time_span
                
    def get_report(self) -> Dict[str, Any]:
        """Get quality report."""
        return {
            'symbol': self.symbol,
            'metrics': self.metrics.copy(),
            'health_score': self._calculate_health_score(),
            'recommendations': self._get_recommendations()
        }
        
    def _calculate_health_score(self) -> float:
        """Calculate overall data health score (0-1)."""
        score = 1.0
        
        # Penalize for anomalies
        if self.metrics['total_ticks'] > 0:
            anomaly_rate = self.metrics['anomalous_ticks'] / self.metrics['total_ticks']
            score -= anomaly_rate * 0.5
            
        # Penalize for synthetic bars
        total_bars = sum(len(w) for w in self.bar_windows.values())
        if total_bars > 0:
            synthetic_rate = self.metrics['synthetic_bars'] / total_bars
            score -= synthetic_rate * 0.3
            
        # Penalize for large price spikes
        if self.metrics['max_price_spike'] > 0.05:  # 5%
            score -= 0.2
            
        return max(0.0, score)
        
    def _get_recommendations(self) -> List[str]:
        """Get recommendations for improving data quality."""
        recommendations = []
        
        if self.metrics['anomalous_ticks'] > self.metrics['total_ticks'] * 0.01:
            recommendations.append(
                "High anomaly rate detected. Review tick validation parameters."
            )
            
        if self.metrics['synthetic_bars'] > 10:
            recommendations.append(
                "Multiple gaps detected. Check data feed stability."
            )
            
        if self.metrics['avg_tick_rate'] < 0.5:
            recommendations.append(
                "Low tick rate. May indicate connectivity issues."
            )
            
        return recommendations


class DataRecorder:
    """
    Records market data for analysis and replay.
    
    Features:
    - Efficient binary storage
    - Compression support
    - Metadata tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Recording state
        self.is_recording = False
        self.start_time = None
        self.tick_count = 0
        self.bar_count = 0
        self.session_id = None
        
        # Buffers
        self.tick_buffer = []
        self.bar_buffer = []
        self.buffer_size = config.get('buffer_size', 10000)
        
        # File handles
        self.tick_file = None
        self.bar_file = None
        
    async def start_recording(self, session_id: str):
        """Start recording session."""
        if self.is_recording:
            logger.warning("Recording already in progress")
            return
            
        self.is_recording = True
        self.start_time = datetime.now()
        self.session_id = session_id
        
        # Create file paths
        timestamp = self.start_time.strftime('%Y%m%d_%H%M%S')
        self.tick_file = self.output_dir / f"ticks_{session_id}_{timestamp}.parquet"
        self.bar_file = self.output_dir / f"bars_{session_id}_{timestamp}.parquet"
        
        logger.info(f"Started recording session {session_id}")
        
    async def stop_recording(self):
        """Stop recording and flush buffers."""
        if not self.is_recording:
            return
            
        # Flush remaining buffers
        await self._flush_tick_buffer()
        await self._flush_bar_buffer()
        
        self.is_recording = False
        duration = datetime.now() - self.start_time
        
        # Write metadata
        metadata = {
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat(),
            'duration': duration.total_seconds(),
            'tick_count': self.tick_count,
            'bar_count': self.bar_count,
            'tick_file': str(self.tick_file),
            'bar_file': str(self.bar_file)
        }
        
        metadata_file = self.output_dir / f"metadata_{self.session_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(
            f"Recording stopped. Duration: {duration}, "
            f"Ticks: {self.tick_count}, Bars: {self.bar_count}"
        )
        
    async def record_tick(self, tick: TickData):
        """Record tick data."""
        if not self.is_recording:
            return
            
        self.tick_buffer.append(tick.to_dict())
        self.tick_count += 1
        
        if len(self.tick_buffer) >= self.buffer_size:
            await self._flush_tick_buffer()
            
    async def record_bar(self, bar: BarData):
        """Record bar data."""
        if not self.is_recording:
            return
            
        self.bar_buffer.append(bar.to_dict())
        self.bar_count += 1
        
        if len(self.bar_buffer) >= self.buffer_size:
            await self._flush_bar_buffer()
            
    async def _flush_tick_buffer(self):
        """Flush tick buffer to file."""
        if not self.tick_buffer:
            return
            
        df = pd.DataFrame(self.tick_buffer)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Append to parquet file
        if self.tick_file.exists():
            existing_df = pd.read_parquet(self.tick_file)
            df = pd.concat([existing_df, df], ignore_index=True)
            
        df.to_parquet(self.tick_file, compression='snappy')
        
        self.tick_buffer.clear()
        
    async def _flush_bar_buffer(self):
        """Flush bar buffer to file."""
        if not self.bar_buffer:
            return
            
        df = pd.DataFrame(self.bar_buffer)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Append to parquet file
        if self.bar_file.exists():
            existing_df = pd.read_parquet(self.bar_file)
            df = pd.concat([existing_df, df], ignore_index=True)
            
        df.to_parquet(self.bar_file, compression='snappy')
        
        self.bar_buffer.clear()


class MockDataGenerator:
    """
    Generates realistic mock data for testing.
    
    Features:
    - Configurable volatility
    - Trend generation
    - Gap simulation
    - Volume patterns
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.symbol = config['symbol']
        
        # Price generation parameters
        self.base_price = config.get('base_price', 100.0)
        self.volatility = config.get('volatility', 0.01)
        self.trend = config.get('trend', 0.0)
        self.mean_reversion = config.get('mean_reversion', 0.1)
        
        # Volume generation
        self.base_volume = config.get('base_volume', 1000)
        self.volume_volatility = config.get('volume_volatility', 0.5)
        
        # State
        self.current_price = self.base_price
        self.current_time = datetime.now()
        self.tick_interval = timedelta(seconds=config.get('tick_interval', 1))
        
    async def generate_ticks(self, count: int) -> List[TickData]:
        """Generate mock tick data."""
        ticks = []
        
        for _ in range(count):
            # Generate price using geometric Brownian motion
            drift = self.trend - self.mean_reversion * (self.current_price / self.base_price - 1)
            diffusion = self.volatility * np.random.randn()
            
            price_return = drift * self.tick_interval.total_seconds() / 86400 + diffusion
            self.current_price *= (1 + price_return)
            
            # Generate volume
            volume_factor = np.exp(self.volume_volatility * np.random.randn())
            volume = int(self.base_volume * volume_factor)
            
            # Create tick
            tick = TickData(
                timestamp=self.current_time,
                symbol=self.symbol,
                price=round(self.current_price, 4),
                volume=volume,
                source='mock'
            )
            
            ticks.append(tick)
            
            # Advance time
            self.current_time += self.tick_interval
            
            # Occasionally skip time to simulate gaps
            if np.random.random() < 0.01:  # 1% chance
                gap_minutes = np.random.randint(5, 30)
                self.current_time += timedelta(minutes=gap_minutes)
                
        return ticks
        
    def generate_historical_data(
        self,
        start_date: datetime,
        end_date: datetime,
        tick_rate: float = 1.0
    ) -> pd.DataFrame:
        """Generate historical tick data."""
        # Calculate total ticks needed
        duration = (end_date - start_date).total_seconds()
        tick_count = int(duration * tick_rate)
        
        # Generate ticks
        self.current_time = start_date
        ticks = asyncio.run(self.generate_ticks(tick_count))
        
        # Convert to DataFrame
        df = pd.DataFrame([t.to_dict() for t in ticks])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        return df