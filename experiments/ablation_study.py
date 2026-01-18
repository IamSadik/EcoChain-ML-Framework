"""
Ablation Study for EcoChain-ML Framework

UPDATED: Now includes multiple runs with statistical significance testing
- 10 runs with different random seeds (matching baseline_comparison.py)
- 95% confidence intervals for all metrics
- Mean and standard deviation reporting
- Paired experimental design (same workload per seed across all configs)

This script performs ablation studies by removing one component at a time:
1. Full EcoChain-ML (all components)
2. Without Renewable Prediction
3. Without DVFS
4. Without Model Compression
5. Without Blockchain Verification

Purpose: Understand the contribution of each component to overall performance.
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
from scipy import stats

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulator import NetworkSimulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AblationStudy:
    """Conducts ablation study experiments with statistical significance testing."""
    
    def __init__(self, config_path: str = "config/experiment_config.yaml"):
        """Initialize ablation study."""
        self.config_path = config_path
        self.config = self._load_config()
        self.results = {}
        self.multi_run_results = {}  # Store results from multiple runs
        
        # Get number of runs and seeds from config (matching baseline comparison)
        self.num_runs = self.config.get('statistical_analysis', {}).get('num_runs', 10)
        self.random_seeds = self.config.get('statistical_analysis', {}).get('random_seeds', 
                                                                             list(range(42, 42 + self.num_runs)))
        self.confidence_level = self.config.get('statistical_analysis', {}).get('confidence_level', 0.95)
        
        # Create results directories
        self.results_dir = Path("results/ablation_study")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        self.metrics_dir = self.results_dir / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized ablation study with config: {config_path}")
        logger.info(f"Statistical analysis: {self.num_runs} runs with {self.confidence_level*100}% confidence intervals")
    
    def _load_config(self) -> dict:
        """Load experiment configuration."""
        config_file = Path(__file__).parent.parent / self.config_path
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def run_all_ablations(self):
        """Run all ablation experiments with multiple runs for statistical significance."""
        logger.info("=" * 80)
        logger.info("STARTING ABLATION STUDY EXPERIMENTS")
        logger.info(f"Running {self.num_runs} iterations per configuration for statistical significance")
        logger.info("=" * 80)
        
        ablation_configs = [
            ('full_ecochain_ml', {'renewable_prediction': True, 'dvfs': True, 
                                  'model_compression': True, 'blockchain': True}),
            ('without_renewable_prediction', {'renewable_prediction': False, 'dvfs': True,
                                             'model_compression': True, 'blockchain': True}),
            ('without_dvfs', {'renewable_prediction': True, 'dvfs': False,
                             'model_compression': True, 'blockchain': True}),
            ('without_compression', {'renewable_prediction': True, 'dvfs': True,
                                    'model_compression': False, 'blockchain': True}),
            ('without_blockchain', {'renewable_prediction': True, 'dvfs': True,
                                   'model_compression': True, 'blockchain': False})
        ]
        
        # Initialize storage for all configurations
        for config_name, _ in ablation_configs:
            self.multi_run_results[config_name] = []
        
        # Run paired experiments (same workload for all configs in each iteration)
        for run_idx, seed in enumerate(self.random_seeds, 1):
            logger.info(f"\n{'=' * 80}")
            logger.info(f"RUN {run_idx}/{self.num_runs} - Using seed={seed}")
            logger.info(f"Generating SHARED workload for all configurations (Paired Design)")
            logger.info(f"{'=' * 80}")
            
            # Set seed for reproducibility
            random.seed(seed)
            np.random.seed(seed)
            
            # Generate workload ONCE for this run (shared by all configs)
            logger.info(f"\n--- Generating shared workload for run {run_idx} ---")
            shared_simulator = NetworkSimulator(
                system_config_path="config/system_config.yaml",
                experiment_config_path=self.config_path,
                random_seed=seed
            )
            shared_workload = shared_simulator.generate_workload(workload_pattern='realistic_bursty')
            logger.info(f"Generated {len(shared_workload)} tasks - WILL BE SHARED BY ALL CONFIGURATIONS")
            
            # Run ALL configurations with the SAME workload
            for config_name, config_params in ablation_configs:
                logger.info(f"\n--- Configuration: {config_name} (run {run_idx}, seed={seed}) ---")
                
                # Create NEW simulator for this config (fresh state)
                simulator = NetworkSimulator(
                    system_config_path="config/system_config.yaml",
                    experiment_config_path=self.config_path,
                    random_seed=seed
                )
                
                # CRITICAL: Inject the SHARED workload (don't generate new one)
                simulator.tasks_generated = shared_workload
                
                # Run ablation with shared workload
                run_result = simulator.run_ablation(config_name, config_params)
                
                # Store results
                self.multi_run_results[config_name].append(run_result)
                
                logger.info(f"  Energy: {run_result['total_energy_kwh']:.4f} kWh, "
                           f"Carbon: {run_result['total_carbon_gco2']:.2f} gCO2, "
                           f"Renewable: {run_result['renewable_percent']:.2f}%")
        
        # Calculate aggregate statistics for each configuration
        for config_name, _ in ablation_configs:
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Calculating aggregate statistics for: {config_name}")
            logger.info(f"{'=' * 80}")
            
            self.results[config_name] = self._calculate_aggregate_metrics(self.multi_run_results[config_name])
            self._save_results(config_name)
        
        # Generate comparison plots and tables with confidence intervals
        self._generate_ablation_plots()
        self._generate_ablation_tables()
        
        logger.info("\n" + "=" * 80)
        logger.info("ABLATION STUDY COMPLETED")
        logger.info("=" * 80)
    
    def _calculate_aggregate_metrics(self, runs: list) -> dict:
        """
        Calculate mean, std, and confidence intervals from multiple runs.
        Matches the statistical approach in baseline_comparison.py
        """
        aggregate = {}
        
        # Get all metric keys from first run
        metric_keys = runs[0].keys()
        
        # Define which keys should be excluded from aggregation (non-numeric)
        exclude_keys = ['method', 'node_stats', 'blockchain_overhead', 'scheduler_stats', 
                        'use_compression', 'use_blockchain', 'ablation_config']
        
        for key in metric_keys:
            # Skip non-numeric fields
            if key in exclude_keys:
                # Just use the value from the first run for non-numeric fields
                aggregate[key] = runs[0][key]
                continue
            
            values = [run[key] for run in runs]
            
            # Check if values are numeric
            try:
                # Try to convert to numpy array - will fail if not numeric
                numeric_values = np.array(values, dtype=float)
                
                # Calculate statistics
                mean_val = np.mean(numeric_values)
                std_val = np.std(numeric_values, ddof=1)  # Sample std deviation
                
                # Calculate confidence interval
                n = len(numeric_values)
                
                if n < 2 or std_val == 0:
                    ci = (mean_val, mean_val)
                else:
                    sem = std_val / np.sqrt(n)  # Standard error of mean
                    ci = stats.t.interval(self.confidence_level, n-1, loc=mean_val, scale=sem)
                
                aggregate[key] = mean_val
                aggregate[f"{key}_std"] = std_val
                aggregate[f"{key}_ci_lower"] = ci[0]
                aggregate[f"{key}_ci_upper"] = ci[1]
                aggregate[f"{key}_all_runs"] = values
            except (ValueError, TypeError):
                # Non-numeric field, just use first run's value
                aggregate[key] = runs[0][key]
        
        logger.info(f"\nAggregate metrics calculated:")
        logger.info(f"  Total Energy: {aggregate['total_energy_kwh']:.4f} ± {aggregate['total_energy_kwh_std']:.4f} kWh")
        logger.info(f"  Carbon Emissions: {aggregate['total_carbon_gco2']:.2f} ± {aggregate['total_carbon_gco2_std']:.2f} gCO2")
        logger.info(f"  Avg Latency: {aggregate['avg_latency_sec']:.4f} ± {aggregate['avg_latency_sec_std']:.4f} sec")
        logger.info(f"  Renewable Usage: {aggregate['renewable_percent']:.2f} ± {aggregate['renewable_percent_std']:.2f}%")
        
        return aggregate
    
    def _run_ablation_experiment(self, config_name: str, config_params: dict) -> dict:
        """Run a single ablation experiment with true component control."""
        # This method is now deprecated in favor of the multi-run approach
        # Kept for backward compatibility but not used in run_all_ablations
        pass
    
    def _convert_to_serializable(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(v) for v in obj]
        return obj

    def _save_results(self, config_name: str):
        """Save results for a single ablation configuration including all runs."""
        # Save aggregate results
        output_file = self.metrics_dir / f"{config_name}_metrics.json"
        serializable_results = self._convert_to_serializable(self.results[config_name])
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save all individual run results
        all_runs_file = self.metrics_dir / f"{config_name}_all_runs.json"
        with open(all_runs_file, 'w') as f:
            json.dump(self._convert_to_serializable(self.multi_run_results[config_name]), f, indent=2)
        
        logger.info(f"Saved aggregate metrics to {output_file}")
        logger.info(f"Saved all runs to {all_runs_file}")
    
    def _generate_ablation_plots(self):
        """Generate ablation study plots with confidence intervals."""
        logger.info("Generating ablation study plots with error bars...")
        
        configs = list(self.results.keys())
        
        # Plot 1: Component contribution to energy savings
        self._plot_component_contribution(configs)
        
        # Plot 2: Multi-metric heatmap
        self._plot_heatmap(configs)
        
        # Plot 3: Stacked bar chart showing impact
        self._plot_stacked_impact(configs)
        
        # Plot 4: Line plot showing degradation
        self._plot_degradation_line(configs)
        
        logger.info(f"All ablation plots saved to {self.plots_dir}")
    
    def _plot_component_contribution(self, configs: list):
        """Plot component contribution to performance with confidence intervals."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        full_config = 'full_ecochain_ml'
        full_result = self.results[full_config]
        
        # Energy contribution with error bars
        energy_increase = [(self.results[c]['total_energy_kwh'] - full_result['total_energy_kwh']) / 
                          full_result['total_energy_kwh'] * 100 for c in configs]
        energy_errors = [self.results[c].get('total_energy_kwh_std', 0) / full_result['total_energy_kwh'] * 100 
                        for c in configs]
        
        colors1 = ['#2ecc71' if x <= 0 else '#e74c3c' for x in energy_increase]
        ax1.barh(range(len(configs)), energy_increase, xerr=energy_errors, color=colors1, alpha=0.7, 
                capsize=5, error_kw={'linewidth': 2})
        ax1.set_yticks(range(len(configs)))
        ax1.set_yticklabels([c.replace('_', ' ').title() for c in configs], fontsize=9)
        ax1.set_xlabel('Energy Change (%)', fontweight='bold')
        ax1.set_title(f'Impact on Energy Consumption\n(↓ Negative/Lower is Better, {self.num_runs} runs)', fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
        
        # Carbon contribution with error bars
        carbon_increase = [(self.results[c]['total_carbon_gco2'] - full_result['total_carbon_gco2']) / 
                          full_result['total_carbon_gco2'] * 100 for c in configs]
        carbon_errors = [self.results[c].get('total_carbon_gco2_std', 0) / full_result['total_carbon_gco2'] * 100 
                        for c in configs]
        
        colors2 = ['#2ecc71' if x <= 0 else '#e74c3c' for x in carbon_increase]
        ax2.barh(range(len(configs)), carbon_increase, xerr=carbon_errors, color=colors2, alpha=0.7,
                capsize=5, error_kw={'linewidth': 2})
        ax2.set_yticks(range(len(configs)))
        ax2.set_yticklabels([c.replace('_', ' ').title() for c in configs], fontsize=9)
        ax2.set_xlabel('Carbon Change (%)', fontweight='bold')
        ax2.set_title(f'Impact on Carbon Emissions\n(↓ Negative/Lower is Better, {self.num_runs} runs)', fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
        
        # Latency contribution with error bars
        latency_increase = [(self.results[c]['avg_latency_sec'] - full_result['avg_latency_sec']) / 
                           full_result['avg_latency_sec'] * 100 for c in configs]
        latency_errors = [self.results[c].get('avg_latency_sec_std', 0) / full_result['avg_latency_sec'] * 100 
                         for c in configs]
        
        colors3 = ['#2ecc71' if x <= 0 else '#e74c3c' for x in latency_increase]
        ax3.barh(range(len(configs)), latency_increase, xerr=latency_errors, color=colors3, alpha=0.7,
                capsize=5, error_kw={'linewidth': 2})
        ax3.set_yticks(range(len(configs)))
        ax3.set_yticklabels([c.replace('_', ' ').title() for c in configs], fontsize=9)
        ax3.set_xlabel('Latency Change (%)', fontweight='bold')
        ax3.set_title(f'Impact on Latency\n(↓ Negative/Lower is Better, {self.num_runs} runs)', fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
        
        # Renewable contribution (higher is better, so show increase as positive)
        renewable_change = [self.results[c]['renewable_percent'] - full_result['renewable_percent'] 
                           for c in configs]
        renewable_errors = [self.results[c].get('renewable_percent_std', 0) for c in configs]
        
        colors4 = ['#2ecc71' if x >= 0 else '#e74c3c' for x in renewable_change]
        ax4.barh(range(len(configs)), renewable_change, xerr=renewable_errors, color=colors4, alpha=0.7,
                capsize=5, error_kw={'linewidth': 2})
        ax4.set_yticks(range(len(configs)))
        ax4.set_yticklabels([c.replace('_', ' ').title() for c in configs], fontsize=9)
        ax4.set_xlabel('Renewable Usage Change (%)', fontweight='bold')
        ax4.set_title(f'Impact on Renewable Utilization\n(↑ Positive/Higher is Better, {self.num_runs} runs)', fontweight='bold')
        ax4.grid(axis='x', alpha=0.3)
        ax4.axvline(x=0, color='black', linestyle='--', linewidth=1)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'component_contribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_heatmap(self, configs: list):
        """Plot heatmap of normalized metrics."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Note: For heatmap, lower is better for Energy, Carbon, Latency, Cost
        # Higher is better for Renewable - we'll invert renewable for consistent coloring
        metrics = ['Energy\n(↓ Lower)', 'Carbon\n(↓ Lower)', 'Latency\n(↓ Lower)', 
                   'Renewable\n(↑ Higher)', 'Cost\n(↓ Lower)']
        
        # Create data matrix
        data = []
        for config in configs:
            result = self.results[config]
            row = [
                result['total_energy_kwh'],
                result['total_carbon_gco2'],
                result['avg_latency_sec'],
                100 - result['renewable_percent'],  # Invert so lower = better (for coloring)
                result['operational_cost_usd']
            ]
            data.append(row)
        
        data = np.array(data)
        
        # Use rank-based normalization to handle outliers better
        data_normalized = np.zeros_like(data, dtype=float)
        for col in range(data.shape[1]):
            sorted_indices = np.argsort(data[:, col])
            ranks = np.empty_like(sorted_indices)
            ranks[sorted_indices] = np.arange(len(sorted_indices))
            data_normalized[:, col] = ranks / (len(ranks) - 1) if len(ranks) > 1 else 0.5
        
        # Use a single-color gradient (shades of blue: white -> dark blue)
        im = ax.imshow(data_normalized, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(configs)))
        ax.set_xticklabels(metrics)
        ax.set_yticklabels([c.replace('_', '\n').title() for c in configs])
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Relative Rank\n(Lighter = Better, Darker = Worse)', 
                      rotation=270, labelpad=25)
        
        # Add text annotations with original values
        original_data = []
        for config in configs:
            result = self.results[config]
            row = [
                result['total_energy_kwh'],
                result['total_carbon_gco2'],
                result['avg_latency_sec'],
                result['renewable_percent'],  # Show actual value
                result['operational_cost_usd']
            ]
            original_data.append(row)
        original_data = np.array(original_data)
        
        for i in range(len(configs)):
            for j in range(len(metrics)):
                text_color = 'white' if data_normalized[i, j] > 0.5 else 'black'
                
                if j == 4:  # Cost column
                    text = ax.text(j, i, f'{original_data[i, j]:.6f}',
                                 ha="center", va="center", color=text_color, 
                                 fontsize=8, fontweight='bold')
                else:
                    text = ax.text(j, i, f'{original_data[i, j]:.2f}',
                                 ha="center", va="center", color=text_color, 
                                 fontsize=8, fontweight='bold')
        
        ax.set_title(f'Ablation Study Heatmap\n(Mean of {self.num_runs} runs, Lighter = Better)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'ablation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_stacked_impact(self, configs: list):
        """Plot stacked bar chart showing cumulative impact."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        full_config = 'full_ecochain_ml'
        full_result = self.results[full_config]
        
        # Calculate percentage changes relative to full system
        energy_pct = [(self.results[c]['total_energy_kwh'] / full_result['total_energy_kwh'] - 1) * 100 
                      for c in configs]
        carbon_pct = [(self.results[c]['total_carbon_gco2'] / full_result['total_carbon_gco2'] - 1) * 100 
                      for c in configs]
        
        x = np.arange(len(configs))
        width = 0.35
        
        ax.bar(x - width/2, energy_pct, width, label='Energy Change (↓ Lower is Better)', color='#e74c3c', alpha=0.8)
        ax.bar(x + width/2, carbon_pct, width, label='Carbon Change (↓ Lower is Better)', color='#3498db', alpha=0.8)
        
        ax.set_ylabel('Change Relative to Full System (%)', fontweight='bold')
        ax.set_title(f'Performance Impact When Removing Components\n(Mean of {self.num_runs} runs, ↓ Bars Below Zero = Better)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('_', '\n').title() for c in configs], fontsize=9)
        ax.legend()
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'stacked_impact.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_degradation_line(self, configs: list):
        """Plot line chart showing performance degradation."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        full_config = 'full_ecochain_ml'
        full_result = self.results[full_config]
        
        # Metrics to plot - normalized to 100% = full system
        metrics_data = {
            'Energy (↓ Lower is Better)': [self.results[c]['total_energy_kwh'] / full_result['total_energy_kwh'] * 100 
                      for c in configs],
            'Carbon (↓ Lower is Better)': [self.results[c]['total_carbon_gco2'] / full_result['total_carbon_gco2'] * 100 
                      for c in configs],
            'Renewable (↑ Higher is Better)': [self.results[c]['renewable_percent'] / full_result['renewable_percent'] * 100 
                         for c in configs]
        }
        
        x = np.arange(len(configs))
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        
        for (metric_name, values), color in zip(metrics_data.items(), colors):
            ax.plot(x, values, marker='o', linewidth=2, label=metric_name, markersize=8, color=color)
        
        ax.set_xlabel('Configuration', fontweight='bold')
        ax.set_ylabel('Percentage of Full System Value (%)', fontweight='bold')
        ax.set_title(f'Metric Comparison Across Ablation Configurations\n(Mean of {self.num_runs} runs, 100% = Full EcoChain-ML)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('_', '\n').title() for c in configs], fontsize=9)
        ax.axhline(y=100, color='black', linestyle='--', linewidth=1, label='Full System Baseline (100%)')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'degradation_line.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_ablation_tables(self):
        """Generate ablation study tables with statistical measures."""
        logger.info("Generating ablation study tables with confidence intervals...")
        
        # Create comprehensive ablation table
        table_data = []
        
        full_config = 'full_ecochain_ml'
        full_result = self.results[full_config]
        
        for config in self.results.keys():
            result = self.results[config]
            
            # Calculate relative changes
            energy_change = ((result['total_energy_kwh'] - full_result['total_energy_kwh']) / 
                           full_result['total_energy_kwh'] * 100)
            carbon_change = ((result['total_carbon_gco2'] - full_result['total_carbon_gco2']) / 
                           full_result['total_carbon_gco2'] * 100)
            latency_change = ((result['avg_latency_sec'] - full_result['avg_latency_sec']) / 
                            full_result['avg_latency_sec'] * 100)
            renewable_change = result['renewable_percent'] - full_result['renewable_percent']
            
            table_data.append({
                'Configuration': config.replace('_', ' ').title(),
                'Energy (kWh)': f"{result['total_energy_kwh']:.6f}",
                'Energy Δ (%)': f"{energy_change:+.2f}",
                'Carbon (gCO2)': f"{result['total_carbon_gco2']:.4f}",
                'Carbon Δ (%)': f"{carbon_change:+.2f}",
                'Latency (s)': f"{result['avg_latency_sec']:.4f}",
                'Latency Δ (%)': f"{latency_change:+.2f}",
                'Renewable (%)': f"{result['renewable_percent']:.2f}",
                'Renewable Δ (%)': f"{renewable_change:+.2f}",
                'Cost ($)': f"{result['operational_cost_usd']:.6f}"
            })
        
        df = pd.DataFrame(table_data)
        
        # Save as CSV
        csv_file = self.metrics_dir / 'ablation_table.csv'
        try:
            df.to_csv(csv_file, index=False)
            logger.info(f"Saved ablation table to {csv_file}")
        except PermissionError:
            # Fallback if file is open in Excel
            timestamp = datetime.now().strftime("%H%M%S")
            alt_csv_file = self.metrics_dir / f'ablation_table_{timestamp}.csv'
            logger.warning(f"Could not save to {csv_file} (Permission Denied). "
                           f"The file might be open in another program. "
                           f"Saving to {alt_csv_file} instead.")
            df.to_csv(alt_csv_file, index=False)
        
        # Save as LaTeX
        latex_file = self.metrics_dir / 'ablation_table.tex'
        latex_table = df.to_latex(index=False, escape=False)
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(latex_table)
        logger.info(f"Saved LaTeX table to {latex_file}")
        
        # Save detailed statistics table
        stats_data = []
        for config in self.results.keys():
            result = self.results[config]
            stats_data.append({
                'Configuration': config.replace('_', ' ').title(),
                'Energy Mean±SD': f"{result['total_energy_kwh']:.4f}±{result.get('total_energy_kwh_std', 0):.4f}",
                'Carbon Mean±SD': f"{result['total_carbon_gco2']:.2f}±{result.get('total_carbon_gco2_std', 0):.2f}",
                'Latency Mean±SD': f"{result['avg_latency_sec']:.4f}±{result.get('avg_latency_sec_std', 0):.4f}",
                'Renewable Mean±SD': f"{result['renewable_percent']:.2f}±{result.get('renewable_percent_std', 0):.2f}"
            })
        
        df_stats = pd.DataFrame(stats_data)
        stats_file = self.metrics_dir / 'ablation_statistics.csv'
        df_stats.to_csv(stats_file, index=False)
        logger.info(f"Saved detailed statistics to {stats_file}")


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("EcoChain-ML Ablation Study Experiment")
    print("=" * 80 + "\n")
    
    # Create and run ablation study
    ablation = AblationStudy()
    ablation.run_all_ablations()
    
    print("\n" + "=" * 80)
    print("ABLATION STUDY COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {ablation.results_dir}")
    print(f"Plots saved to: {ablation.plots_dir}")
    print(f"Metrics saved to: {ablation.metrics_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
