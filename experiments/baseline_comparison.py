"""
Baseline Comparison Experiment for EcoChain-ML Framework

This script compares four methods:
1. Standard (baseline): Round-robin scheduling, no optimization
2. Energy-aware only: Energy-aware scheduling without blockchain
3. Blockchain only: Blockchain verification without energy optimization
4. EcoChain-ML: Full framework with all components

Metrics tracked:
- Total energy consumption (kWh)
- Carbon emissions (gCO2)
- Average latency (seconds)
- Renewable energy utilization (%)
- Total cost ($)
- Blockchain overhead (%)
"""

import sys
import os
import logging
import yaml
import json
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulator import NetworkSimulator, EdgeNode
from src.scheduler import EnergyAwareScheduler, BaselineScheduler, RenewablePredictor
from src.blockchain import BlockchainVerifier
from src.inference import ModelExecutor, ModelCompressor
from src.monitoring import EnergyMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaselineComparison:
    """Conducts baseline comparison experiments."""
    
    def __init__(self, config_path: str = "config/experiment_config.yaml"):
        """Initialize comparison experiment."""
        self.config_path = config_path
        self.config = self._load_config()
        self.results = {}
        
        # Create results directories
        self.results_dir = Path("results/baseline_comparison")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        self.metrics_dir = self.results_dir / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized baseline comparison with config: {config_path}")
    
    def _load_config(self) -> dict:
        """Load experiment configuration."""
        config_file = Path(__file__).parent.parent / self.config_path
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def run_all_experiments(self):
        """Run all baseline comparison experiments."""
        logger.info("=" * 80)
        logger.info("STARTING BASELINE COMPARISON EXPERIMENTS")
        logger.info("=" * 80)
        
        methods = [baseline['name'] for baseline in self.config['baselines']]
        
        for method in methods:
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Running experiment: {method}")
            logger.info(f"{'=' * 80}")
            
            self.results[method] = self._run_single_experiment(method)
            
            # Save intermediate results
            self._save_results(method)
        
        # Generate comparison plots and tables
        self._generate_comparison_plots()
        self._generate_comparison_tables()
        
        logger.info("\n" + "=" * 80)
        logger.info("BASELINE COMPARISON COMPLETED")
        logger.info("=" * 80)
    
    def _run_single_experiment(self, method: str) -> dict:
        """Run a single experiment with specified method."""
        logger.info(f"Initializing NetworkSimulator for method: {method}")
        
        # Create simulator with correct parameter names
        simulator = NetworkSimulator(
            system_config_path="config/system_config.yaml",
            experiment_config_path=self.config_path
        )
        
        # Run simulation with the specified method
        logger.info(f"Running simulation with method: {method}")
        metrics = simulator.run_simulation(method=method)
        
        # Log summary
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Results for {method}:")
        logger.info(f"  Total Energy: {metrics['total_energy_kwh']:.4f} kWh")
        logger.info(f"  Carbon Emissions: {metrics['total_carbon_gco2']:.2f} gCO2")
        logger.info(f"  Avg Latency: {metrics['avg_latency_sec']:.4f} sec")
        logger.info(f"  Renewable Usage: {metrics['renewable_percent']:.2f}%")
        logger.info(f"  Operational Cost: ${metrics['operational_cost_usd']:.6f}")
        logger.info(f"  Carbon Credits Earned: ${metrics['carbon_credits_earned_usd']:.6f}")
        logger.info(f"  Net Cost: ${metrics['net_cost_usd']:.6f}")
        logger.info(f"{'=' * 60}\n")
        
        return metrics
    
    def _save_results(self, method: str):
        """Save results for a single method."""
        output_file = self.metrics_dir / f"{method}_metrics.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results[method], f, indent=2)
        
        logger.info(f"Saved metrics to {output_file}")
    
    def _generate_comparison_plots(self):
        """Generate comparison plots for all metrics."""
        logger.info("Generating comparison plots...")
        
        methods = list(self.results.keys())
        
        # Plot 1: Energy Consumption Comparison
        self._plot_energy_comparison(methods)
        
        # Plot 2: Carbon Emissions Comparison
        self._plot_carbon_comparison(methods)
        
        # Plot 3: Latency Comparison
        self._plot_latency_comparison(methods)
        
        # Plot 4: Renewable Energy Utilization
        self._plot_renewable_comparison(methods)
        
        # Plot 5: Cost Comparison
        self._plot_cost_comparison(methods)
        
        # Plot 6: Multi-metric radar chart
        self._plot_radar_chart(methods)
        
        logger.info(f"All plots saved to {self.plots_dir}")
    
    def _plot_energy_comparison(self, methods: list):
        """Plot energy consumption comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        energy_values = [self.results[m]['total_energy_kwh'] for m in methods]
        renewable_energy = [self.results[m]['renewable_energy_kwh'] for m in methods]
        grid_energy = [self.results[m]['grid_energy_kwh'] for m in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        ax.bar(x - width/2, renewable_energy, width, label='Renewable (↓ Lower is Better)', color='#2ecc71')
        ax.bar(x + width/2, grid_energy, width, label='Grid (↓ Lower is Better)', color='#e74c3c')
        
        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Energy Consumption (kWh)', fontsize=12, fontweight='bold')
        ax.set_title('Energy Consumption Comparison\n(↓ Lower Energy Usage is Better)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'energy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_carbon_comparison(self, methods: list):
        """Plot carbon emissions comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        carbon_values = [self.results[m]['total_carbon_gco2'] for m in methods]
        
        colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
        bars = ax.bar(methods, carbon_values, color=colors, alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Carbon Emissions (gCO2)', fontsize=12, fontweight='bold')
        ax.set_title('Carbon Emissions Comparison\n(↓ Lower is Better)', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'carbon_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_latency_comparison(self, methods: list):
        """Plot latency comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        avg_latency = [self.results[m]['avg_latency_sec'] for m in methods]
        max_latency = [self.results[m]['max_latency_sec'] for m in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        ax.bar(x - width/2, avg_latency, width, label='Average', color='#3498db')
        ax.bar(x + width/2, max_latency, width, label='Maximum', color='#e74c3c')
        
        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latency (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Latency Comparison\n(↓ Lower is Better)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'latency_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_renewable_comparison(self, methods: list):
        """Plot renewable energy utilization comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        renewable_pct = [self.results[m]['renewable_percent'] for m in methods]
        
        colors = ['#e74c3c' if pct < 50 else '#f39c12' if pct < 70 else '#2ecc71' 
                  for pct in renewable_pct]
        bars = ax.bar(methods, renewable_pct, color=colors, alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Renewable Energy Usage (%)', fontsize=12, fontweight='bold')
        ax.set_title('Renewable Energy Utilization\n(↑ Higher is Better)', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=15, ha='right')
        ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'renewable_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cost_comparison(self, methods: list):
        """Plot cost comparison including carbon credits."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left plot: Operational cost breakdown
        op_costs = [self.results[m]['operational_cost_usd'] for m in methods]
        carbon_credits = [self.results[m]['carbon_credits_earned_usd'] for m in methods]
        net_costs = [self.results[m]['net_cost_usd'] for m in methods]
        
        x = np.arange(len(methods))
        width = 0.25
        
        bars1 = ax1.bar(x - width, op_costs, width, label='Op. Cost (↓ Lower)', color='#e74c3c', alpha=0.8)
        bars2 = ax1.bar(x, carbon_credits, width, label='Carbon Credits (↑ Higher)', color='#2ecc71', alpha=0.8)
        bars3 = ax1.bar(x + width, net_costs, width, label='Net Cost (↓ Lower)', color='#3498db', alpha=0.8)
        
        ax1.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Cost (USD)', fontsize=12, fontweight='bold')
        ax1.set_title('Cost Breakdown Comparison\n(↓ Lower Net Cost is Better)', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=15, ha='right')
        ax1.legend(fontsize=9)
        ax1.grid(axis='y', alpha=0.3)
        
        # Right plot: Net cost comparison with savings highlighted
        colors = ['#e74c3c' if nc > 0 else '#2ecc71' for nc in net_costs]
        bars = ax2.bar(methods, net_costs, color=colors, alpha=0.8)
        
        # Add value labels with exact values
        for bar, nc in zip(bars, net_costs):
            height = bar.get_height()
            label = f'${nc:.6f}'
            va = 'bottom' if nc >= 0 else 'top'
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                   label, ha='center', va=va, fontweight='bold', fontsize=9)
        
        ax2.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Net Cost (USD)', fontsize=12, fontweight='bold')
        ax2.set_title('Net Cost After Carbon Credits\n(↓ Lower/Negative is Better = Profit!)', 
                     fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=15, ha='right')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'cost_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_radar_chart(self, methods: list):
        """
        Plot multi-metric radar chart showing percentage improvement relative to baseline.
        
        This visualization makes differences between methods highly visible by:
        - Using Standard (baseline) as 0% reference point
        - Showing improvement as distance from center (outward)
        - Showing degradation as distance from center (inward)
        - Making performance gaps clearly distinguishable
        
        Perfect for academic papers and presentations!
        """
        # Create larger figure with more space for labels
        fig, ax = plt.subplots(figsize=(16, 14), subplot_kw=dict(projection='polar'))
        
        # Define metrics and their properties
        metrics_info = [
            {
                'name': 'Energy\nReduction',
                'key': 'total_energy_kwh',
                'lower_is_better': True,
                'unit': '%'
            },
            {
                'name': 'Carbon\nReduction',
                'key': 'total_carbon_gco2',
                'lower_is_better': True,
                'unit': '%'
            },
            {
                'name': 'Latency\nImprovement',
                'key': 'avg_latency_sec',
                'lower_is_better': True,
                'unit': '%'
            },
            {
                'name': 'Renewable\nEnergy Usage',
                'key': 'renewable_percent',
                'lower_is_better': False,
                'unit': '%'
            },
            {
                'name': 'Cost\nReduction',
                'key': 'net_cost_usd',
                'lower_is_better': True,
                'unit': '%'
            }
        ]
        
        num_metrics = len(metrics_info)
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Get baseline (standard) for comparison
        baseline = self.results['standard']
        
        # Define distinct colors and styles for each method
        method_styles = {
            'standard': {
                'color': '#95a5a6',  # Gray for baseline
                'linestyle': '-', 
                'linewidth': 3, 
                'marker': 'o', 
                'markersize': 10,
                'alpha': 0.9,
                'fill_alpha': 0.15,
                'label': 'Standard (Baseline = 0%)',
                'zorder': 1
            },
            'blockchain_only': {
                'color': '#e67e22',  # Orange
                'linestyle': '-.', 
                'linewidth': 3.5, 
                'marker': '^', 
                'markersize': 11,
                'alpha': 0.9,
                'fill_alpha': 0.25,
                'label': 'Blockchain Only',
                'zorder': 2
            },
            'energy_aware_only': {
                'color': '#3498db',  # Blue
                'linestyle': '--', 
                'linewidth': 4, 
                'marker': 's', 
                'markersize': 13,
                'alpha': 0.95,
                'fill_alpha': 0.28,
                'label': 'Energy-Aware Only',
                'zorder': 3
            },
            'ecochain_ml': {
                'color': '#27ae60',  # Green
                'linestyle': '-', 
                'linewidth': 5, 
                'marker': 'D', 
                'markersize': 15,
                'alpha': 1.0,
                'fill_alpha': 0.30,
                'label': '★ EcoChain-ML (Proposed)',
                'zorder': 4
            }
        }
        
        # Sort methods by zorder
        sorted_methods = sorted(methods, 
                               key=lambda m: method_styles.get(m, {'zorder': 0})['zorder'])
        
        # Calculate improvements and track min/max for scaling
        all_improvements = []
        method_values = {}
        
        # First pass: calculate all improvements
        for method in sorted_methods:
            result = self.results[method]
            values = []
            
            for metric in metrics_info:
                key = metric['key']
                value = result[key]
                baseline_value = baseline[key]
                
                if method == 'standard':
                    # Baseline is always at 0%
                    improvement = 0.0
                else:
                    if metric['lower_is_better']:
                        # For "lower is better": positive % = improvement (reduction)
                        # Negative % = degradation (increase)
                        if baseline_value > 0:
                            improvement = ((baseline_value - value) / baseline_value) * 100
                        else:
                            improvement = 0
                    else:
                        # For "higher is better": show absolute percentage point difference
                        # Positive = improvement, negative = degradation
                        improvement = value - baseline_value
                
                values.append(improvement)
            
            method_values[method] = values
            if method != 'standard':
                all_improvements.extend(values)
        
        # Set dynamic radial limits based on data (including negative values)
        max_val = max(all_improvements) if all_improvements else 50
        min_val = min(all_improvements) if all_improvements else 0
        
        # Determine scale to show both positive and negative clearly
        if min_val >= 0:
            # All positive - no degradation
            lower_limit = -5
            upper_limit = max(25, int(max_val * 1.2))
        else:
            # Has negative values - need space for degradation
            lower_limit = min(-15, int(min_val * 1.3))  # Give space for negative
            upper_limit = max(50, int(max_val * 1.2))
        
        # Round to nice numbers
        if upper_limit <= 30:
            upper_limit = 30
            ticks = list(range(lower_limit, upper_limit + 1, 10))
        elif upper_limit <= 60:
            upper_limit = 60
            ticks = list(range(lower_limit, upper_limit + 1, 15))
        else:
            upper_limit = 100
            ticks = list(range(lower_limit, upper_limit + 1, 20))
        
        # Plot each method
        for method in sorted_methods:
            style = method_styles.get(method, {
                'color': '#9b59b6', 
                'linestyle': '-', 
                'linewidth': 3, 
                'marker': 'o', 
                'markersize': 10,
                'alpha': 0.8,
                'fill_alpha': 0.20,
                'label': method.replace('_', ' ').title(),
                'zorder': 0
            })
            
            values = method_values[method]
            
            # Complete the circle
            values_plot = values + values[:1]
            
            # First, fill the area (only if not baseline)
            if method != 'standard':
                ax.fill(angles, values_plot, 
                       alpha=style['fill_alpha'], 
                       color=style['color'],
                       zorder=style['zorder'])
            
            # Then, plot the line on top
            ax.plot(angles, values_plot, 
                   linestyle=style['linestyle'],
                   linewidth=style['linewidth'], 
                   color=style['color'],
                   marker=style['marker'],
                   markersize=style['markersize'],
                   label=style['label'],
                   alpha=style['alpha'],
                   zorder=style['zorder'] + 10,
                   markeredgecolor='white',
                   markeredgewidth=2)
        
        # Set axis labels with clear indicators
        metric_labels = []
        for metric in metrics_info:
            if metric['lower_is_better']:
                metric_labels.append(f"{metric['name']}\n(Higher % = Better)")
            else:
                metric_labels.append(f"{metric['name']}\n(Higher % = Better)")
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=13, fontweight='bold', 
                          verticalalignment='center')
        ax.tick_params(axis='x', pad=25)
        
        # Set radial limits
        ax.set_ylim(lower_limit, upper_limit)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{t}%' for t in ticks], 
                          fontsize=11, color='#2c3e50', fontweight='bold')
        
        # Add a reference circle at 0% for baseline with enhanced visibility
        ax.plot(angles, [0] * len(angles), 'k-', linewidth=2.5, alpha=0.7, 
               label='Baseline Reference (0%)', zorder=15)
        
        # Shade the negative region lightly to show degradation area
        if lower_limit < 0:
            ax.fill_between(angles, lower_limit, 0, alpha=0.08, color='red', 
                           label='Degradation Zone', zorder=0)
        
        # Add title with clear explanation
        title_text = 'Performance Improvement vs. Baseline (Standard Method)\n\n'
        title_text += '✓ Center (0%) = Standard Baseline Performance\n'
        title_text += '✓ Outward (Positive) = Better Performance | Inward (Negative) = Worse Performance\n'
        title_text += '✓ Larger Positive Area = Superior Overall System'
        
        ax.set_title(title_text, fontsize=17, fontweight='bold', pad=50, 
                    color='#2c3e50', linespacing=1.6)
        
        # Add legend with better positioning
        legend = ax.legend(loc='upper left', bbox_to_anchor=(1.35, 1.15), 
                          fontsize=13, frameon=True, shadow=True, 
                          fancybox=True, title='System Configurations', 
                          title_fontsize=15, borderpad=1.3, labelspacing=1.3)
        legend.get_frame().set_alpha(0.98)
        legend.get_frame().set_edgecolor('#2c3e50')
        legend.get_frame().set_linewidth(2)
        
        # Add grid with better visibility
        ax.grid(True, alpha=0.6, linestyle='--', linewidth=1.2, color='#7f8c8d')
        
        # Add explanatory text box at bottom
        explanation = (
            '📊 How to Interpret This Chart:\n\n'
            '  • Each axis shows % improvement over Standard baseline\n'
            '  • 0% (bold black line) = Standard baseline performance\n'
            '  • Positive values (outward) = Better than baseline\n'
            '  • Negative values (inward) = Worse than baseline\n'
            '  • Compare colored areas: larger positive area = better overall'
        )
        
        fig.text(0.5, 0.01, explanation, 
                ha='center', va='bottom', fontsize=11.5, 
                bbox=dict(boxstyle='round,pad=1.3', facecolor='#ecf0f1', 
                         edgecolor='#2c3e50', linewidth=2.5, alpha=0.97),
                style='normal', color='#2c3e50', linespacing=1.9,
                family='monospace')
        
        # Adjust layout to prevent text overlap
        plt.tight_layout(pad=3.5)
        plt.subplots_adjust(bottom=0.15, top=0.88)
        
        plt.savefig(self.plots_dir / 'radar_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Radar chart created with percentage improvement visualization")
    
    def _generate_comparison_tables(self):
        """Generate comparison tables."""
        logger.info("Generating comparison tables...")
        
        # Create comprehensive comparison table
        table_data = []
        
        for method in self.results.keys():
            result = self.results[method]
            table_data.append({
                'Method': method.replace('_', ' ').title(),
                'Energy (kWh)': f"{result['total_energy_kwh']:.4f}",
                'Carbon (gCO2)': f"{result['total_carbon_gco2']:.2f}",
                'Avg Latency (s)': f"{result['avg_latency_sec']:.4f}",
                'Renewable (%)': f"{result['renewable_percent']:.2f}",
                'Op. Cost ($)': f"${result['operational_cost_usd']:.6f}",
                'Carbon Credits ($)': f"${result['carbon_credits_earned_usd']:.6f}",
                'Net Cost ($)': f"${result['net_cost_usd']:.6f}",
                'Tasks': result['tasks_completed']
            })
        
        df = pd.DataFrame(table_data)
        
        # Save as CSV
        csv_file = self.metrics_dir / 'comparison_table.csv'
        df.to_csv(csv_file, index=False)
        logger.info(f"Saved comparison table to {csv_file}")
        
        # Save as LaTeX table
        latex_file = self.metrics_dir / 'comparison_table.tex'
        latex_table = df.to_latex(index=False, escape=False)
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(latex_table)
        logger.info(f"Saved LaTeX table to {latex_file}")
        
        # Calculate improvement percentages relative to baseline
        baseline = self.results['standard']
        improvement_data = []
        
        for method in self.results.keys():
            if method == 'standard':
                continue
            
            result = self.results[method]
            
            # Calculate net cost improvement (considering carbon credits)
            baseline_net = baseline['operational_cost_usd']  # No credits for baseline
            result_net = result['net_cost_usd']
            net_cost_reduction = ((baseline_net - result_net) / baseline_net * 100) if baseline_net > 0 else 0
            
            improvement_data.append({
                'Method': method.replace('_', ' ').title(),
                'Energy Reduction (%)': f"{((baseline['total_energy_kwh'] - result['total_energy_kwh']) / baseline['total_energy_kwh'] * 100):.2f}",
                'Carbon Reduction (%)': f"{((baseline['total_carbon_gco2'] - result['total_carbon_gco2']) / baseline['total_carbon_gco2'] * 100):.2f}",
                'Latency Change (%)': f"{((result['avg_latency_sec'] - baseline['avg_latency_sec']) / baseline['avg_latency_sec'] * 100):+.2f}",
                'Renewable Increase (%)': f"{(result['renewable_percent'] - baseline['renewable_percent']):+.2f}",
                'Carbon Credits ($)': f"${result['carbon_credits_earned_usd']:.4f}",
                'Net Cost Reduction (%)': f"{net_cost_reduction:.2f}"
            })
        
        df_improvement = pd.DataFrame(improvement_data)
        
        # Save improvement table
        improvement_csv = self.metrics_dir / 'improvement_table.csv'
        df_improvement.to_csv(improvement_csv, index=False)
        logger.info(f"Saved improvement table to {improvement_csv}")
        
        # Save as LaTeX
        improvement_latex = self.metrics_dir / 'improvement_table.tex'
        latex_improvement = df_improvement.to_latex(index=False, escape=False)
        with open(improvement_latex, 'w', encoding='utf-8') as f:
            f.write(latex_improvement)
        logger.info(f"Saved improvement LaTeX table to {improvement_latex}")


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("EcoChain-ML Baseline Comparison Experiment")
    print("=" * 80 + "\n")
    
    # Create and run comparison
    comparison = BaselineComparison()
    comparison.run_all_experiments()
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {comparison.results_dir}")
    print(f"Plots saved to: {comparison.plots_dir}")
    print(f"Metrics saved to: {comparison.metrics_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
