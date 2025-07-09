"""
Visualization Module for AlgoSpace Strategy
Provides comprehensive plotting functions for analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class StrategyVisualizer:
    """Comprehensive visualization for trading strategies"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.save_plots = self.config.get('save_plots', True)
        self.plot_format = self.config.get('plot_format', 'png')
        self.dpi = self.config.get('dpi', 300)
        self.figure_size = self.config.get('figure_size', [12, 8])
        
    def plot_price_with_indicators(self, df: pd.DataFrame, 
                                 indicators: List[str] = ['mlmi', 'nwrqk_yhat1'],
                                 signals: Optional[Dict[str, pd.Series]] = None,
                                 title: str = "Price Chart with Indicators") -> go.Figure:
        """Create interactive price chart with indicators"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=('Price', 'MLMI', 'Volume')
        )
        
        # Price candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add NW-RQK lines if available
        if 'nwrqk_yhat1' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['nwrqk_yhat1'],
                    mode='lines',
                    name='NW-RQK Primary',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
        
        if 'nwrqk_yhat2' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['nwrqk_yhat2'],
                    mode='lines',
                    name='NW-RQK Secondary',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
        
        # Add entry/exit signals if provided
        if signals:
            if 'long_entries' in signals:
                long_mask = signals['long_entries']
                fig.add_trace(
                    go.Scatter(
                        x=df.index[long_mask],
                        y=df['Low'][long_mask] * 0.99,
                        mode='markers',
                        name='Long Entry',
                        marker=dict(
                            symbol='triangle-up',
                            size=10,
                            color='green'
                        )
                    ),
                    row=1, col=1
                )
            
            if 'short_entries' in signals:
                short_mask = signals['short_entries']
                fig.add_trace(
                    go.Scatter(
                        x=df.index[short_mask],
                        y=df['High'][short_mask] * 1.01,
                        mode='markers',
                        name='Short Entry',
                        marker=dict(
                            symbol='triangle-down',
                            size=10,
                            color='red'
                        )
                    ),
                    row=1, col=1
                )
        
        # MLMI indicator
        if 'mlmi' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['mlmi'],
                    mode='lines',
                    name='MLMI',
                    line=dict(color='purple', width=1)
                ),
                row=2, col=1
            )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # Volume
        if 'Volume' in df.columns:
            colors = ['red' if close < open else 'green' 
                     for close, open in zip(df['Close'], df['Open'])]
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color=colors,
                    showlegend=False
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price",
            height=800,
            template="plotly_dark"
        )
        
        fig.update_xaxes(rangeslider_visible=False)
        
        return fig
    
    def plot_synergy_heatmap(self, df: pd.DataFrame) -> plt.Figure:
        """Create heatmap of synergy occurrences"""
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Create hourly aggregation of synergies
        synergy_cols = [col for col in df.columns if col.startswith('syn') and 
                       ('long' in col or 'short' in col) and 'strength' not in col]
        
        if not synergy_cols:
            logger.warning("No synergy columns found for heatmap")
            return fig
        
        # Aggregate by hour
        hourly_synergies = df[synergy_cols].resample('H').sum()
        
        # Create heatmap
        sns.heatmap(
            hourly_synergies.T,
            cmap='YlOrRd',
            cbar_kws={'label': 'Signal Count'},
            yticklabels=synergy_cols,
            ax=ax
        )
        
        ax.set_title('Synergy Signal Heatmap (Hourly)')
        ax.set_xlabel('Time')
        ax.set_ylabel('Synergy Type')
        
        plt.tight_layout()
        
        if self.save_plots:
            self._save_figure(fig, 'synergy_heatmap')
        
        return fig
    
    def plot_performance_comparison(self, portfolios: Dict[int, any]) -> plt.Figure:
        """Compare performance across different synergy types"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, (syn_type, portfolio) in enumerate(portfolios.items()):
            if portfolio and idx < 4:
                ax = axes[idx]
                
                # Plot cumulative returns
                cum_returns = portfolio.cumulative_returns()
                ax.plot(cum_returns.index, cum_returns.values, 
                       label=f'Type {syn_type}', linewidth=2)
                
                # Add drawdown shading
                drawdown = portfolio.drawdown()
                ax.fill_between(drawdown.index, 0, -drawdown.values, 
                               alpha=0.3, color='red', label='Drawdown')
                
                ax.set_title(f'Synergy Type {syn_type} Performance')
                ax.set_xlabel('Date')
                ax.set_ylabel('Cumulative Return')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Performance Comparison Across Synergy Types', fontsize=16)
        plt.tight_layout()
        
        if self.save_plots:
            self._save_figure(fig, 'performance_comparison')
        
        return fig
    
    def plot_risk_metrics(self, portfolios: Dict[int, any]) -> plt.Figure:
        """Visualize risk metrics for all strategies"""
        metrics_data = []
        
        for syn_type, portfolio in portfolios.items():
            if portfolio:
                stats = portfolio.stats()
                metrics_data.append({
                    'Type': f'Type {syn_type}',
                    'Sharpe Ratio': stats.get('Sharpe Ratio', 0),
                    'Sortino Ratio': stats.get('Sortino Ratio', 0),
                    'Max Drawdown': abs(stats.get('Max Drawdown [%]', 0)),
                    'Volatility': stats.get('Volatility [%]', 0)
                })
        
        if not metrics_data:
            logger.warning("No portfolio data for risk metrics")
            return plt.figure()
        
        metrics_df = pd.DataFrame(metrics_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Sharpe Ratio
        axes[0, 0].bar(metrics_df['Type'], metrics_df['Sharpe Ratio'])
        axes[0, 0].set_title('Sharpe Ratio Comparison')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Sortino Ratio
        axes[0, 1].bar(metrics_df['Type'], metrics_df['Sortino Ratio'])
        axes[0, 1].set_title('Sortino Ratio Comparison')
        axes[0, 1].set_ylabel('Sortino Ratio')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Max Drawdown
        axes[1, 0].bar(metrics_df['Type'], metrics_df['Max Drawdown'], color='red')
        axes[1, 0].set_title('Maximum Drawdown Comparison')
        axes[1, 0].set_ylabel('Max Drawdown (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Risk-Return Scatter
        axes[1, 1].scatter(metrics_df['Volatility'], metrics_df['Sharpe Ratio'], s=100)
        for i, txt in enumerate(metrics_df['Type']):
            axes[1, 1].annotate(txt, (metrics_df['Volatility'].iloc[i], 
                                     metrics_df['Sharpe Ratio'].iloc[i]))
        axes[1, 1].set_xlabel('Volatility (%)')
        axes[1, 1].set_ylabel('Sharpe Ratio')
        axes[1, 1].set_title('Risk-Return Profile')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Risk Metrics Analysis', fontsize=16)
        plt.tight_layout()
        
        if self.save_plots:
            self._save_figure(fig, 'risk_metrics')
        
        return fig
    
    def plot_indicator_correlation(self, df: pd.DataFrame) -> plt.Figure:
        """Plot correlation matrix of indicators"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Select numeric columns
        indicator_cols = ['mlmi', 'FVG_Score', 'nwrqk_slope', 'volatility_20',
                         'atr_20', 'returns']
        
        available_cols = [col for col in indicator_cols if col in df.columns]
        
        if len(available_cols) < 2:
            logger.warning("Not enough indicators for correlation matrix")
            return fig
        
        # Calculate correlation
        corr_matrix = df[available_cols].corr()
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix), k=1)
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title('Indicator Correlation Matrix')
        
        plt.tight_layout()
        
        if self.save_plots:
            self._save_figure(fig, 'indicator_correlation')
        
        return fig
    
    def plot_monte_carlo_results(self, mc_results: List[Dict]) -> plt.Figure:
        """Visualize Monte Carlo simulation results"""
        if not mc_results:
            logger.warning("No Monte Carlo results to plot")
            return plt.figure()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        mc_df = pd.DataFrame(mc_results)
        
        # Extract percentile values
        mc_df['Return Percentile Value'] = mc_df['Return Percentile'].str.rstrip('%').astype(float)
        mc_df['Sharpe Percentile Value'] = mc_df['Sharpe Percentile'].str.rstrip('%').astype(float)
        
        # Bar plot of percentiles
        x = np.arange(len(mc_df))
        width = 0.35
        
        axes[0].bar(x - width/2, mc_df['Return Percentile Value'], 
                   width, label='Return Percentile')
        axes[0].bar(x + width/2, mc_df['Sharpe Percentile Value'], 
                   width, label='Sharpe Percentile')
        
        axes[0].set_xlabel('Synergy Type')
        axes[0].set_ylabel('Percentile (%)')
        axes[0].set_title('Monte Carlo Percentiles')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(mc_df['Synergy'])
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Average percentile
        mc_df['Average Percentile'] = (mc_df['Return Percentile Value'] + 
                                       mc_df['Sharpe Percentile Value']) / 2
        
        axes[1].bar(mc_df['Synergy'], mc_df['Average Percentile'])
        axes[1].set_xlabel('Synergy Type')
        axes[1].set_ylabel('Average Percentile (%)')
        axes[1].set_title('Overall Monte Carlo Performance')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        # Add reference line at 50%
        axes[1].axhline(y=50, color='red', linestyle='--', alpha=0.5, 
                       label='50th Percentile')
        axes[1].legend()
        
        plt.suptitle('Monte Carlo Validation Results', fontsize=16)
        plt.tight_layout()
        
        if self.save_plots:
            self._save_figure(fig, 'monte_carlo_results')
        
        return fig
    
    def plot_trade_analysis(self, trades_df: pd.DataFrame) -> plt.Figure:
        """Analyze trade distribution and patterns"""
        if trades_df.empty:
            logger.warning("No trades to analyze")
            return plt.figure()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Trade distribution by hour
        if 'timestamp' in trades_df.columns:
            trades_df['hour'] = pd.to_datetime(trades_df['timestamp']).dt.hour
            hour_counts = trades_df['hour'].value_counts().sort_index()
            
            axes[0, 0].bar(hour_counts.index, hour_counts.values)
            axes[0, 0].set_xlabel('Hour of Day')
            axes[0, 0].set_ylabel('Number of Trades')
            axes[0, 0].set_title('Trade Distribution by Hour')
            axes[0, 0].set_xticks(range(0, 24, 2))
        
        # Trade direction distribution
        if 'direction' in trades_df.columns:
            direction_counts = trades_df['direction'].value_counts()
            axes[0, 1].pie(direction_counts.values, labels=direction_counts.index,
                          autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title('Trade Direction Distribution')
        
        # Position size distribution
        if 'size' in trades_df.columns:
            axes[1, 0].hist(trades_df['size'], bins=30, edgecolor='black')
            axes[1, 0].set_xlabel('Position Size')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Position Size Distribution')
        
        # Confidence vs Size scatter
        if 'confidence' in trades_df.columns and 'size' in trades_df.columns:
            axes[1, 1].scatter(trades_df['confidence'], trades_df['size'], alpha=0.5)
            axes[1, 1].set_xlabel('Signal Confidence')
            axes[1, 1].set_ylabel('Position Size')
            axes[1, 1].set_title('Confidence vs Position Size')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Trade Analysis', fontsize=16)
        plt.tight_layout()
        
        if self.save_plots:
            self._save_figure(fig, 'trade_analysis')
        
        return fig
    
    def create_performance_dashboard(self, portfolios: Dict[int, any], 
                                   df: pd.DataFrame) -> go.Figure:
        """Create comprehensive performance dashboard"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Cumulative Returns', 'Drawdowns',
                'Monthly Returns', 'Rolling Sharpe',
                'Trade Frequency', 'Win Rate Evolution'
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        colors = ['blue', 'green', 'red', 'purple']
        
        for idx, (syn_type, portfolio) in enumerate(portfolios.items()):
            if portfolio:
                color = colors[idx % len(colors)]
                
                # Cumulative returns
                cum_returns = portfolio.cumulative_returns()
                fig.add_trace(
                    go.Scatter(
                        x=cum_returns.index,
                        y=cum_returns.values,
                        mode='lines',
                        name=f'Type {syn_type}',
                        line=dict(color=color),
                        showlegend=True
                    ),
                    row=1, col=1
                )
                
                # Drawdowns
                drawdown = portfolio.drawdown()
                fig.add_trace(
                    go.Scatter(
                        x=drawdown.index,
                        y=-drawdown.values,
                        mode='lines',
                        name=f'Type {syn_type}',
                        line=dict(color=color),
                        showlegend=False
                    ),
                    row=1, col=2
                )
                
                # Monthly returns
                monthly_returns = portfolio.returns().resample('M').apply(
                    lambda x: (1 + x).prod() - 1
                )
                fig.add_trace(
                    go.Bar(
                        x=monthly_returns.index,
                        y=monthly_returns.values,
                        name=f'Type {syn_type}',
                        marker_color=color,
                        showlegend=False
                    ),
                    row=2, col=1
                )
                
                # Rolling Sharpe (252-day)
                rolling_sharpe = portfolio.rolling_sharpe(window=252)
                fig.add_trace(
                    go.Scatter(
                        x=rolling_sharpe.index,
                        y=rolling_sharpe.values,
                        mode='lines',
                        name=f'Type {syn_type}',
                        line=dict(color=color),
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            height=1000,
            title_text="Performance Dashboard",
            showlegend=True,
            template="plotly_dark"
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=2)
        fig.update_yaxes(title_text="Return", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown %", row=1, col=2)
        fig.update_yaxes(title_text="Monthly Return", row=2, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=2)
        
        return fig
    
    def _save_figure(self, fig, name: str):
        """Save figure to file"""
        if self.save_plots:
            filename = f"{name}.{self.plot_format}"
            if hasattr(fig, 'savefig'):
                fig.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            else:
                fig.write_image(filename)
            logger.info(f"Saved plot: {filename}")