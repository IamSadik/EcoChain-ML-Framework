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
        
        ax.bar(x - width/2, renewable_energy, width, label='Renewable (↑ Higher is Better)', color='#2ecc71')
        ax.bar(x + width/2, grid_energy, width, label='Grid (↓ Lower is Better)', color='#e74c3c')
        
        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Energy Consumption (kWh)', fontsize=12, fontweight='bold')
        ax.set_title('Energy Consumption Comparison\n(↑ Higher Renewable & ↓ Lower Grid is Better)', 
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
        """Plot multi-metric radar chart."""
        # Normalize metrics to 0-100 scale
        metrics_to_plot = ['energy', 'carbon', 'latency', 'renewable', 'cost']
        
        # Get baseline values for normalization
        baseline = self.results['standard']
        
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
        angles += angles[:1]
        
        # Define distinct colors and styles for each method
        method_styles = {
            'standard': {'color': '#e74c3c', 'linestyle': '-', 'linewidth': 2.5, 'marker': 'o', 'markersize': 8},
            'energy_aware_only': {'color': '#3498db', 'linestyle': '--', 'linewidth': 3, 'marker': 's', 'markersize': 10},
            'blockchain_only': {'color': '#f39c12', 'linestyle': '-.', 'linewidth': 2.5, 'marker': '^', 'markersize': 8},
            'ecochain_ml': {'color': '#2ecc71', 'linestyle': '-', 'linewidth': 3.5, 'marker': 'D', 'markersize': 10}
        }
        
        # Default style for any other methods
        default_style = {'color': '#9b59b6', 'linestyle': '-', 'linewidth': 2, 'marker': 'o', 'markersize': 6}
        
        for method in methods:
            values = []
            result = self.results[method]
            style = method_styles.get(method, default_style)
            
            # Energy (lower is better, so invert)
            if baseline['total_energy_kwh'] > 0:
                energy_score = (1 - result['total_energy_kwh'] / baseline['total_energy_kwh']) * 100
            else:
                energy_score = 0
            values.append(max(0, min(100, 50 + energy_score)))
            
            # Carbon (lower is better, so invert)
            if baseline['total_carbon_gco2'] > 0:
                carbon_score = (1 - result['total_carbon_gco2'] / baseline['total_carbon_gco2']) * 100
            else:
                carbon_score = 0
            values.append(max(0, min(100, 50 + carbon_score)))
            
            # Latency (lower is better, so invert)
            if baseline['avg_latency_sec'] > 0:
                latency_score = (1 - result['avg_latency_sec'] / baseline['avg_latency_sec']) * 100
            else:
                latency_score = 0
            values.append(max(0, min(100, 50 + latency_score)))
            
            # Renewable (higher is better)
            renewable_score = result['renewable_percent']
            values.append(renewable_score)
            
            # Cost (lower is better, so invert)
            if baseline['operational_cost_usd'] > 0:
                cost_score = (1 - result['operational_cost_usd'] / baseline['operational_cost_usd']) * 100
            else:
                cost_score = 0
            values.append(max(0, min(100, 50 + cost_score)))
            
            values += values[:1]
            
            # Plot with distinct style
            ax.plot(angles, values, 
                   linestyle=style['linestyle'],
                   linewidth=style['linewidth'], 
                   color=style['color'],
                   marker=style['marker'],
                   markersize=style['markersize'],
                   label=method.replace('_', ' ').title(),
                   zorder=3 if method == 'energy_aware_only' else 2)
            ax.fill(angles, values, alpha=0.1, color=style['color'])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['Energy\n(↓ Lower)', 'Carbon\n(↓ Lower)', 'Latency\n(↓ Lower)', 
                           'Renewable\n(↑ Higher)', 'Cost\n(↓ Lower)'], fontsize=11, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_title('Multi-Metric Performance Comparison\n(Larger Area = Better Overall Performance)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'radar_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
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
