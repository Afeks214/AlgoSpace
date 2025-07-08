#!/usr/bin/env python3
"""
AlgoSpace Data Flow Visualization
Creates a comprehensive diagram showing data flow through the AlgoSpace architecture.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Rectangle
import numpy as np
from typing import Dict, List, Tuple


class DataFlowDiagram:
    """Creates visual representation of AlgoSpace data flow."""
    
    def __init__(self):
        self.fig_width = 16
        self.fig_height = 12
        self.colors = {
            'raw_data': '#e8f4f8',
            'processing': '#b8e0d2',
            'embedder': '#95b8d1',
            'agent': '#809bce',
            'decision': '#483d8b',
            'output': '#2e1a47',
            'frozen': '#d3d3d3',
            'communication': '#ffd700'
        }
        
    def create_comprehensive_diagram(self):
        """Create the complete data flow diagram."""
        fig, ax = plt.subplots(1, 1, figsize=(self.fig_width, self.fig_height))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Title
        ax.text(5, 9.5, 'AlgoSpace Data Flow Architecture', 
                fontsize=20, fontweight='bold', ha='center')
        
        # Layer 1: Raw Data Sources
        self._draw_raw_data_layer(ax)
        
        # Layer 2: Data Processing
        self._draw_processing_layer(ax)
        
        # Layer 3: Feature Extraction
        self._draw_feature_layer(ax)
        
        # Layer 4: Embedders
        self._draw_embedder_layer(ax)
        
        # Layer 5: Agents and Decision
        self._draw_agent_layer(ax)
        
        # Layer 6: Two-Gate System
        self._draw_gate_system(ax)
        
        # Draw connections
        self._draw_connections(ax)
        
        # Add legend
        self._add_legend(ax)
        
        # Add annotations
        self._add_annotations(ax)
        
        plt.tight_layout()
        return fig
    
    def _draw_raw_data_layer(self, ax):
        """Draw raw data sources."""
        # CSV Data
        csv_box = FancyBboxPatch((0.5, 8), 1.5, 0.8, 
                                  boxstyle="round,pad=0.1",
                                  facecolor=self.colors['raw_data'],
                                  edgecolor='black', linewidth=2)
        ax.add_patch(csv_box)
        ax.text(1.25, 8.4, 'Raw CSV Data\n(Tick/OHLCV)', 
                ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Market Data Stream
        stream_box = FancyBboxPatch((2.5, 8), 1.5, 0.8,
                                    boxstyle="round,pad=0.1",
                                    facecolor=self.colors['raw_data'],
                                    edgecolor='black', linewidth=2)
        ax.add_patch(stream_box)
        ax.text(3.25, 8.4, 'Market Stream\n(Real-time)', 
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    def _draw_processing_layer(self, ax):
        """Draw data processing components."""
        # Bar Generator
        bar_gen = FancyBboxPatch((0.5, 6.8), 1.5, 0.8,
                                 boxstyle="round,pad=0.1",
                                 facecolor=self.colors['processing'],
                                 edgecolor='black', linewidth=2)
        ax.add_patch(bar_gen)
        ax.text(1.25, 7.2, 'Bar Generator\n5m, 30m', 
                ha='center', va='center', fontsize=10)
        
        # Indicator Engine
        ind_engine = FancyBboxPatch((2.5, 6.8), 2, 0.8,
                                    boxstyle="round,pad=0.1",
                                    facecolor=self.colors['processing'],
                                    edgecolor='black', linewidth=2)
        ax.add_patch(ind_engine)
        ax.text(3.5, 7.2, 'Indicator Engine\nMLMI, NWRQK, FVG, LVN', 
                ha='center', va='center', fontsize=10)
    
    def _draw_feature_layer(self, ax):
        """Draw feature extraction layer."""
        # Matrix Assemblers
        assemblers = [
            ("Matrix 30m\n48×8", 0.5, 5.5),
            ("Matrix 5m\n60×7", 2, 5.5),
            ("MMD Sequence\n96×12", 3.5, 5.5),
            ("LVN Features\n5 features", 5, 5.5)
        ]
        
        for name, x, y in assemblers:
            box = FancyBboxPatch((x, y), 1.3, 0.8,
                                 boxstyle="round,pad=0.1",
                                 facecolor=self.colors['processing'],
                                 edgecolor='black', linewidth=2)
            ax.add_patch(box)
            ax.text(x + 0.65, y + 0.4, name, 
                    ha='center', va='center', fontsize=9)
    
    def _draw_embedder_layer(self, ax):
        """Draw embedder components."""
        # Embedders with dimensions
        embedders = [
            ("Structure\nEmbedder\n48×8→64D", 0.5, 4),
            ("Tactical\nEmbedder\n60×7→48D", 2, 4),
            ("Regime\nEmbedder\n8D→16D", 3.5, 4),
            ("LVN\nEmbedder\n5→8D", 5, 4)
        ]
        
        for name, x, y in embedders:
            box = FancyBboxPatch((x, y), 1.3, 0.8,
                                 boxstyle="round,pad=0.1",
                                 facecolor=self.colors['embedder'],
                                 edgecolor='black', linewidth=2)
            ax.add_patch(box)
            ax.text(x + 0.65, y + 0.4, name, 
                    ha='center', va='center', fontsize=9, color='white')
        
        # Frozen RDE
        rde_box = FancyBboxPatch((6.5, 5.5), 1.5, 0.8,
                                 boxstyle="round,pad=0.1",
                                 facecolor=self.colors['frozen'],
                                 edgecolor='black', linewidth=2,
                                 linestyle='--')
        ax.add_patch(rde_box)
        ax.text(7.25, 5.9, 'Frozen RDE\n8D Latent', 
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    def _draw_agent_layer(self, ax):
        """Draw MARL agents."""
        # Three specialized agents
        agents = [
            ("Structure\nAnalyzer", 1, 2.5),
            ("Short-term\nTactician", 2.5, 2.5),
            ("Mid-freq\nArbitrageur", 4, 2.5)
        ]
        
        for name, x, y in agents:
            box = FancyBboxPatch((x, y), 1.3, 0.8,
                                 boxstyle="round,pad=0.1",
                                 facecolor=self.colors['agent'],
                                 edgecolor='black', linewidth=2)
            ax.add_patch(box)
            ax.text(x + 0.65, y + 0.4, name, 
                    ha='center', va='center', fontsize=9, color='white')
        
        # Communication Network
        comm_box = FancyBboxPatch((5.5, 2.5), 1.8, 0.8,
                                  boxstyle="round,pad=0.1",
                                  facecolor=self.colors['communication'],
                                  edgecolor='black', linewidth=2)
        ax.add_patch(comm_box)
        ax.text(6.4, 2.9, 'Agent Communication\n3 Rounds', 
                ha='center', va='center', fontsize=9, fontweight='bold')
        
        # MC Dropout Consensus
        mc_box = FancyBboxPatch((2, 1.5), 3, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor=self.colors['decision'],
                                edgecolor='black', linewidth=2)
        ax.add_patch(mc_box)
        ax.text(3.5, 1.8, 'MC Dropout Consensus (50 passes)', 
                ha='center', va='center', fontsize=10, 
                color='white', fontweight='bold')
    
    def _draw_gate_system(self, ax):
        """Draw the two-gate decision system."""
        # Gate 1: Synergy Detection
        gate1_box = FancyBboxPatch((0.5, 0.5), 2, 0.6,
                                   boxstyle="round,pad=0.1",
                                   facecolor=self.colors['decision'],
                                   edgecolor='gold', linewidth=3)
        ax.add_patch(gate1_box)
        ax.text(1.5, 0.8, 'Gate 1: Synergy Detection\nMLMI-NWRQK > 0.2', 
                ha='center', va='center', fontsize=9, 
                color='white', fontweight='bold')
        
        # Gate 2: Decision Gate
        gate2_box = FancyBboxPatch((3, 0.5), 2, 0.6,
                                   boxstyle="round,pad=0.1",
                                   facecolor=self.colors['decision'],
                                   edgecolor='gold', linewidth=3)
        ax.add_patch(gate2_box)
        ax.text(4, 0.8, 'Gate 2: Decision Gate\nConfidence > 0.65', 
                ha='center', va='center', fontsize=9, 
                color='white', fontweight='bold')
        
        # Frozen M-RMS
        mrms_box = FancyBboxPatch((5.5, 0.5), 1.8, 0.6,
                                  boxstyle="round,pad=0.1",
                                  facecolor=self.colors['frozen'],
                                  edgecolor='black', linewidth=2,
                                  linestyle='--')
        ax.add_patch(mrms_box)
        ax.text(6.4, 0.8, 'Frozen M-RMS\n4D Risk Proposal', 
                ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Final Output
        output_box = FancyBboxPatch((8, 0.5), 1.5, 0.6,
                                    boxstyle="round,pad=0.1",
                                    facecolor=self.colors['output'],
                                    edgecolor='gold', linewidth=3)
        ax.add_patch(output_box)
        ax.text(8.75, 0.8, 'Trade Decision', 
                ha='center', va='center', fontsize=10, 
                color='white', fontweight='bold')
    
    def _draw_connections(self, ax):
        """Draw connections between components."""
        # Define connection paths
        connections = [
            # Raw data to processing
            ((1.25, 8), (1.25, 7.6)),
            ((3.25, 8), (3.5, 7.6)),
            
            # Processing to features
            ((1.25, 7), (1.15, 6.3)),
            ((3.5, 7), (2.65, 6.3)),
            ((3.5, 7), (4.15, 6.3)),
            
            # Features to embedders
            ((1.15, 5.5), (1.15, 4.8)),
            ((2.65, 5.5), (2.65, 4.8)),
            ((4.15, 5.5), (4.15, 4.8)),
            ((5.65, 5.5), (5.65, 4.8)),
            
            # RDE connection
            ((7.25, 5.5), (4.15, 4.8)),
            
            # Embedders to agents
            ((1.15, 4), (1.65, 3.3)),
            ((2.65, 4), (3.15, 3.3)),
            ((4.15, 4), (4.65, 3.3)),
            
            # Agents to communication
            ((1.65, 2.5), (5.5, 2.9)),
            ((3.15, 2.5), (5.5, 2.9)),
            ((4.65, 2.5), (5.5, 2.9)),
            
            # To MC Dropout
            ((3.5, 2.5), (3.5, 2.1)),
            
            # MC Dropout to gates
            ((3.5, 1.5), (1.5, 1.1)),
            ((3.5, 1.5), (4, 1.1)),
            
            # M-RMS to Gate 2
            ((5.5, 0.8), (5, 0.8)),
            
            # Gate 2 to output
            ((5, 0.8), (8, 0.8))
        ]
        
        for start, end in connections:
            arrow = ConnectionPatch(start, end, "data", "data",
                                   arrowstyle="->", shrinkA=5, shrinkB=5,
                                   mutation_scale=20, fc="black", lw=1.5)
            ax.add_artist(arrow)
    
    def _add_legend(self, ax):
        """Add legend to the diagram."""
        legend_elements = [
            mpatches.Patch(color=self.colors['raw_data'], label='Raw Data'),
            mpatches.Patch(color=self.colors['processing'], label='Processing'),
            mpatches.Patch(color=self.colors['embedder'], label='Embedders'),
            mpatches.Patch(color=self.colors['agent'], label='Agents'),
            mpatches.Patch(color=self.colors['decision'], label='Decision'),
            mpatches.Patch(color=self.colors['frozen'], label='Frozen Models'),
            mpatches.Patch(color=self.colors['communication'], label='Communication')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', 
                  bbox_to_anchor=(0.98, 0.98), fontsize=9)
    
    def _add_annotations(self, ax):
        """Add important annotations."""
        # Critical parameters
        ax.text(9.5, 9, 'Critical Parameters:\n'
                        '• MC Dropout: 50 passes\n'
                        '• Confidence: 0.65\n'
                        '• Synergy: MLMI-NWRQK > 0.2\n'
                        '• Training: 10,000 episodes\n'
                        '• Batch Size: 256',
                fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='wheat', alpha=0.5),
                verticalalignment='top')
        
        # Feature dimensions
        ax.text(0.2, 3.5, 'Feature Dimensions:\n'
                          '• Structure: 48×8 → 64D\n'
                          '• Tactical: 60×7 → 48D\n'
                          '• Regime: 8D → 16D\n'
                          '• LVN: 5 → 8D',
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.5))
    
    def create_feature_dimension_diagram(self):
        """Create detailed feature dimension flow diagram."""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        ax.text(5, 9.5, 'AlgoSpace Feature Dimension Flow', 
                fontsize=18, fontweight='bold', ha='center')
        
        # Define feature flows
        feature_flows = [
            {
                'name': 'Structure (30m)',
                'stages': [
                    ('Raw OHLCV', '30m bars', (1, 8)),
                    ('Indicators', 'MLMI, NWRQK, LVN', (1, 6.5)),
                    ('Matrix', '48×8', (1, 5)),
                    ('Embedding', '64D', (1, 3.5)),
                    ('Unified State', '136D total', (1, 2))
                ]
            },
            {
                'name': 'Tactical (5m)',
                'stages': [
                    ('Raw OHLCV', '5m bars', (3, 8)),
                    ('Indicators', 'FVG, momentum', (3, 6.5)),
                    ('Matrix', '60×7', (3, 5)),
                    ('Embedding', '48D', (3, 3.5)),
                    ('Unified State', 'Part of 136D', (3, 2))
                ]
            },
            {
                'name': 'Regime',
                'stages': [
                    ('MMD Features', '96×12', (5, 8)),
                    ('RDE Processing', 'Transformer-VAE', (5, 6.5)),
                    ('Latent Space', '8D', (5, 5)),
                    ('Embedding', '16D', (5, 3.5)),
                    ('Unified State', 'Part of 136D', (5, 2))
                ]
            },
            {
                'name': 'LVN',
                'stages': [
                    ('Volume Profile', '20 days', (7, 8)),
                    ('LVN Detection', '5 features', (7, 6.5)),
                    ('Features', '5D', (7, 5)),
                    ('Embedding', '8D', (7, 3.5)),
                    ('Unified State', 'Part of 136D', (7, 2))
                ]
            }
        ]
        
        # Draw each flow
        for flow in feature_flows:
            x_base = flow['stages'][0][2][0]
            
            # Draw title
            ax.text(x_base, 9, flow['name'], fontsize=12, fontweight='bold', ha='center')
            
            # Draw stages
            for i, (stage_name, details, pos) in enumerate(flow['stages']):
                # Draw box
                box = FancyBboxPatch((pos[0]-0.4, pos[1]-0.3), 0.8, 0.6,
                                     boxstyle="round,pad=0.05",
                                     facecolor='lightblue' if i < 2 else 'lightgreen' if i < 4 else 'lightyellow',
                                     edgecolor='black', linewidth=1)
                ax.add_patch(box)
                
                # Add text
                ax.text(pos[0], pos[1], f'{stage_name}\n{details}', 
                        ha='center', va='center', fontsize=8)
                
                # Draw arrow to next stage
                if i < len(flow['stages']) - 1:
                    next_pos = flow['stages'][i+1][2]
                    arrow = ConnectionPatch(
                        (pos[0], pos[1]-0.3), (next_pos[0], next_pos[1]+0.3),
                        "data", "data", arrowstyle="->", 
                        shrinkA=5, shrinkB=5, mutation_scale=15, fc="black"
                    )
                    ax.add_artist(arrow)
        
        # Add final unified state
        unified_box = FancyBboxPatch((3.5, 0.5), 3, 0.8,
                                     boxstyle="round,pad=0.1",
                                     facecolor='gold',
                                     edgecolor='black', linewidth=2)
        ax.add_patch(unified_box)
        ax.text(5, 0.9, 'Unified State Vector\n136D = 64D + 48D + 16D + 8D', 
                ha='center', va='center', fontsize=10, fontweight='bold')
        
        return fig


def main():
    """Generate all data flow diagrams."""
    print("Generating AlgoSpace Data Flow Diagrams...")
    
    visualizer = DataFlowDiagram()
    
    # Create comprehensive data flow diagram
    fig1 = visualizer.create_comprehensive_diagram()
    fig1.savefig('notebooks/algospace_data_flow_comprehensive.png', dpi=300, bbox_inches='tight')
    print("✅ Saved comprehensive data flow diagram")
    
    # Create feature dimension diagram
    fig2 = visualizer.create_feature_dimension_diagram()
    fig2.savefig('notebooks/algospace_feature_dimensions.png', dpi=300, bbox_inches='tight')
    print("✅ Saved feature dimension diagram")
    
    # Create summary statistics
    stats = {
        "Total Input Features": {
            "Structure (30m)": "48 timesteps × 8 features = 384",
            "Tactical (5m)": "60 timesteps × 7 features = 420", 
            "Regime (MMD)": "96 timesteps × 12 features = 1152",
            "LVN": "5 scalar features"
        },
        "Embedding Dimensions": {
            "Structure": "384 → 64D",
            "Tactical": "420 → 48D",
            "Regime": "8D → 16D",
            "LVN": "5 → 8D",
            "Total Unified State": "136D"
        },
        "Processing Steps": {
            "1. Raw Data": "CSV/Stream → Tick Data",
            "2. Bar Generation": "Tick → 5m, 30m bars",
            "3. Indicators": "MLMI, NWRQK, FVG, LVN calculation",
            "4. Matrix Assembly": "Feature matrices creation",
            "5. Embedding": "Neural network embedding",
            "6. Agent Processing": "3 specialized agents",
            "7. MC Dropout": "50 forward passes",
            "8. Two-Gate Decision": "Synergy → Confidence check",
            "9. Risk Integration": "M-RMS 4D proposal",
            "10. Trade Execution": "Final decision"
        }
    }
    
    # Save statistics
    import json
    with open('notebooks/data_flow_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print("✅ Saved data flow statistics")
    
    print("\n✅ All diagrams generated successfully!")
    print("Files created:")
    print("  - algospace_data_flow_comprehensive.png")
    print("  - algospace_feature_dimensions.png")
    print("  - data_flow_statistics.json")


if __name__ == "__main__":
    main()