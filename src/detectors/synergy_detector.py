"""
Synergy Detector for AlgoSpace Agent Coordination
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

class SynergyDetector:
    """Detects synergy patterns between MARL agents"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        
    def detect_agent_consensus(self, 
                             agent_outputs: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Detect consensus between agents"""
        
        consensus_scores = {}
        
        # RDE-MRMS synergy
        if 'rde' in agent_outputs and 'mrms' in agent_outputs:
            rde_confidence = np.mean(agent_outputs['rde'])
            mrms_confidence = np.mean(agent_outputs['mrms'])
            consensus_scores['rde_mrms'] = min(rde_confidence, mrms_confidence)
        
        # Main Core consensus
        if 'main_core' in agent_outputs:
            consensus_scores['main_core'] = np.mean(agent_outputs['main_core'])
        
        return consensus_scores
        
    def calculate_synergy_boost(self, consensus_scores: Dict[str, float]) -> float:
        """Calculate synergy boost factor"""
        if not consensus_scores:
            return 1.0
        
        avg_consensus = np.mean(list(consensus_scores.values()))
        
        if avg_consensus > self.confidence_threshold:
            return 1.2  # 20% boost for high consensus
        elif avg_consensus > 0.5:
            return 1.1  # 10% boost for moderate consensus
        else:
            return 1.0  # No boost for low consensus
            
    def detect_conflict(self, agent_outputs: Dict[str, np.ndarray]) -> bool:
        """Detect conflicting agent signals"""
        if len(agent_outputs) < 2:
            return False
        
        # Check for opposing signals
        values = list(agent_outputs.values())
        correlations = []
        
        for i in range(len(values)):
            for j in range(i+1, len(values)):
                corr = np.corrcoef(values[i].flatten(), values[j].flatten())[0,1]
                correlations.append(corr)
        
        # Conflict if negative correlation
        return any(corr < -0.3 for corr in correlations if not np.isnan(corr))