"""
SOTA Comparison Experiment for EcoChain-ML Framework

Compares EcoChain-ML against state-of-the-art baselines:
1. GreenLLM (2025): Grid carbon-aware temporal/spatial shifting
2. CASPER (2024): Carbon-aware serverless edge routing
3. Kubernetes: Standard load-balanced scheduling (no carbon awareness)
4. Compression-Only: INT8 quantization without carbon-aware scheduling
5. EcoChain-ML: Full framework (prediction + scheduling + compression + blockchain)

This addresses reviewer concern: "Comparison to at least one SOTA system"
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
from src.scheduler.sota_baselines import GreenLLMBaseline, CASPERBaseline, KubernetesBaseline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class SOTAComparison:
    """
    Compares EcoChain-ML against SOTA carbon-aware scheduling approaches.
    
    Key differentiators to highlight:
    - GreenLLM: Uses grid carbon intensity, no renewable prediction, no DVFS
    - CASPER: Reactive routing (current state), no prediction, no compression
    - Kubernetes: No carbon awareness at all (industry standard baseline)
    - EcoChain-ML: Predictive + DVFS + Compression + Blockchain (full integration)
    """
    
    def __init__(self, config_path: str = "config/experiment_config.yaml"):
        """Initialize SOTA comparison experiment."""
        self.config_path = config_path
        self.config = self._load_config()
        self.results = {}
        self.multi_run_results = {}
        
        # Statistical parameters
        self.num_runs = 10
        self.random_seeds = list(range(42, 42 + self.num_runs))
        self.confidence_level = 0.95
        
        # Create results directories
        self.results_dir = Path("results/sota_comparison")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        self.metrics_dir = self.results_dir / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized SOTA Comparison with {self.num_runs} runs per method")
    
    def _load_config(self) -> dict:
        """Load experiment configuration."""
        config_file = Path(__file__).parent.parent / self.config_path
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def run_all_comparisons(self):
        """Run comparisons against all SOTA baselines."""
        logger.info("=" * 80)
        logger.info("SOTA COMPARISON EXPERIMENT")
        logger.info("Comparing EcoChain-ML vs. GreenLLM, CASPER, Kubernetes")
        logger.info("=" * 80)
        
        methods = [
            'kubernetes',      # No carbon awareness (industry baseline)
            'casper',          # SOTA: Carbon-aware serverless (2024)
            'green_llm',       # SOTA: Grid carbon-aware scheduling (2025)
            'compression_only', # Ablation: compression without carbon-aware scheduling
            'ecochain_ml'      # Our full framework
        ]
        
        # Initialize storage
        for method in methods:
            self.multi_run_results[method] = []
        
        # Run paired experiments
        for run_idx, seed in enumerate(self.random_seeds, 1):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"RUN {run_idx}/{self.num_runs} - Seed={seed}")
            logger.info(f"{'=' * 60}")
            
            # Set seed
            random.seed(seed)
            np.random.seed(seed)
            
            # Generate shared workload
            shared_simulator = NetworkSimulator(
                system_config_path="config/system_config.yaml",
                experiment_config_path=self.config_path,
                random_seed=seed
            )
            shared_tasks = shared_simulator.generate_workload(workload_pattern='realistic_bursty')
            logger.info(f"Generated {len(shared_tasks)} shared tasks")
            
            # Run each method with same workload
            for method in methods:
                logger.info(f"\n--- Running: {method.upper()} ---")
                
                result = self._run_single_method(method, seed, shared_tasks)
                self.multi_run_results[method].append(result)
                
                logger.info(f"  Energy: {result['total_energy_kwh']:.4f} kWh")
                logger.info(f"  Carbon: {result['total_carbon_gco2']:.2f} gCO2")
                logger.info(f"  Latency: {result['avg_latency_sec']:.4f} s")
                logger.info(f"  Renewable: {result['renewable_percent']:.2f}%")
        
        # Calculate aggregate statistics
        for method in methods:
            self.results[method] = self._calculate_aggregate_metrics(
                self.multi_run_results[method]
            )
        
        # Perform statistical tests
        self._perform_statistical_tests()
        
        # Generate outputs
        self._generate_comparison_table()
        self._generate_comparison_plots()
        self._generate_latex_table()
        
        logger.info("\n" + "=" * 80)
        logger.info("SOTA COMPARISON COMPLETED")
        logger.info(f"Results saved to: {self.results_dir}")
        logger.info("=" * 80)
    
    def _run_single_method(self, method: str, seed: int, shared_tasks) -> dict:
        """Run a single method with shared workload."""
        
        # Create fresh simulator
        simulator = NetworkSimulator(
            system_config_path="config/system_config.yaml",
            experiment_config_path=self.config_path,
            random_seed=seed
        )
        simulator.tasks_generated = shared_tasks
        
        if method == 'ecochain_ml':
            # Full EcoChain-ML
            result = simulator.run_simulation(method='ecochain_ml')
            
        elif method == 'compression_only':
            # Compression without carbon-aware scheduling
            result = simulator.run_simulation(method='compression_only')
            
        elif method == 'kubernetes':
            # Standard Kubernetes (round-robin, no carbon awareness)
            result = self._run_with_sota_scheduler(
                simulator, 'kubernetes', seed
            )
            
        elif method == 'casper':
            # CASPER: Carbon-aware serverless
            result = self._run_with_sota_scheduler(
                simulator, 'casper', seed
            )
            
        elif method == 'green_llm':
            # GreenLLM: Grid carbon-aware
            result = self._run_with_sota_scheduler(
                simulator, 'green_llm', seed
            )
        
        result['method'] = method
        return result
    
    def _run_with_sota_scheduler(self, simulator, scheduler_type: str, seed: int) -> dict:
        """
        Run simulation with a SOTA baseline scheduler.
        
        Since SOTA baselines don't use our full simulation pipeline,
        we simulate their behavior by:
        1. Using their scheduling decisions
        2. Disabling EcoChain-ML-specific features (DVFS, prediction, compression)
        """
        # Reset simulator state
        simulator._reset_simulation()
        
        # Initialize the appropriate SOTA scheduler
        if scheduler_type == 'kubernetes':
            sota_scheduler = KubernetesBaseline(simulator.nodes, strategy='round_robin')
            use_compression = False
        elif scheduler_type == 'casper':
            sota_scheduler = CASPERBaseline(simulator.nodes)
            use_compression = False
        elif scheduler_type == 'green_llm':
            sota_scheduler = GreenLLMBaseline(simulator.nodes)
            use_compression = False
        
        # Track metrics
        total_energy = 0.0
        total_renewable = 0.0
        total_grid = 0.0
        latencies = []
        
        # Process each task
        for task in simulator.tasks_generated:
            current_time = task.arrival_time
            
            try:
                # Use SOTA scheduler for node selection
                task_dict = task.to_dict()
                
                # Get scheduling decision from SOTA baseline
                # Note: SOTA baselines run at max frequency, no compression
                result = sota_scheduler.schedule_task(
                    task_dict, 
                    current_time,
                    compressed=use_compression
                )
                
                # Accumulate metrics
                if result:
                    total_energy += result.get('energy_consumed', 0)
                    total_renewable += result.get('renewable_energy', 0)
                    total_grid += result.get('grid_energy', 0)
                    latencies.append(result.get('execution_time', 0))
                    simulator.tasks_completed.append(result)
                    
            except Exception as e:
                logger.warning(f"Task failed: {e}")
        
        # Calculate carbon (grid energy * carbon intensity)
        carbon_intensity = 400.0  # gCO2/kWh
        total_carbon = total_grid * carbon_intensity
        
        # Compile results
        renewable_pct = (total_renewable / total_energy * 100) if total_energy > 0 else 0
        
        return {
            'total_energy_kwh': total_energy,
            'renewable_energy_kwh': total_renewable,
            'grid_energy_kwh': total_grid,
            'renewable_percent': renewable_pct,
            'total_carbon_gco2': total_carbon,
            'avg_latency_sec': np.mean(latencies) if latencies else 0,
            'std_latency_sec': np.std(latencies) if latencies else 0,
            'max_latency_sec': np.max(latencies) if latencies else 0,
            'tasks_completed': len(simulator.tasks_completed),
            'tasks_generated': len(simulator.tasks_generated),
            'operational_cost_usd': total_grid * 0.12,  # $0.12/kWh
            'net_cost_usd': total_grid * 0.12,
            'carbon_credits_earned_usd': 0.0
        }
    
    def _calculate_aggregate_metrics(self, runs: list) -> dict:
        """Calculate mean, std, and CI from multiple runs."""
        aggregate = {}
        
        numeric_keys = [
            'total_energy_kwh', 'renewable_energy_kwh', 'grid_energy_kwh',
            'renewable_percent', 'total_carbon_gco2', 'avg_latency_sec',
            'tasks_completed', 'operational_cost_usd', 'net_cost_usd'
        ]
        
        for key in numeric_keys:
            values = [run.get(key, 0) for run in runs]
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)
            n = len(values)
            
            if n > 1 and std_val > 0:
                sem = std_val / np.sqrt(n)
                ci = stats.t.interval(self.confidence_level, n-1, loc=mean_val, scale=sem)
            else:
                ci = (mean_val, mean_val)
            
            aggregate[key] = mean_val
            aggregate[f"{key}_std"] = std_val
            aggregate[f"{key}_ci_lower"] = ci[0]
            aggregate[f"{key}_ci_upper"] = ci[1]
        
        # Keep method name
        aggregate['method'] = runs[0].get('method', 'unknown')
        
        return aggregate
    
    def _perform_statistical_tests(self):
        """Perform statistical significance tests: EcoChain-ML vs each SOTA."""
        logger.info("\n" + "=" * 60)
        logger.info("STATISTICAL SIGNIFICANCE TESTS")
        logger.info("=" * 60)
        
        ecochain_results = self.multi_run_results['ecochain_ml']
        test_results = []
        
        for method in ['kubernetes', 'casper', 'green_llm', 'compression_only']:
            method_results = self.multi_run_results[method]
            
            logger.info(f"\n--- EcoChain-ML vs. {method.upper()} ---")
            
            metrics = [
                ('total_energy_kwh', 'Energy (kWh)', 'lower'),
                ('total_carbon_gco2', 'Carbon (gCO2)', 'lower'),
                ('avg_latency_sec', 'Latency (s)', 'lower'),
                ('renewable_percent', 'Renewable (%)', 'higher'),
            ]
            
            for metric_key, metric_name, better_direction in metrics:
                eco_values = [r[metric_key] for r in ecochain_results]
                method_values = [r[metric_key] for r in method_results]
                
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(eco_values, method_values)
                
                # Cohen's d
                mean_diff = np.mean(eco_values) - np.mean(method_values)
                pooled_std = np.sqrt((np.var(eco_values, ddof=1) + np.var(method_values, ddof=1)) / 2)
                cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                
                # Improvement direction
                if better_direction == 'lower':
                    improvement = mean_diff < 0
                    improvement_pct = ((np.mean(method_values) - np.mean(eco_values)) / np.mean(method_values) * 100) if np.mean(method_values) != 0 else 0
                else:
                    improvement = mean_diff > 0
                    improvement_pct = ((np.mean(eco_values) - np.mean(method_values)) / np.mean(method_values) * 100) if np.mean(method_values) != 0 else 0
                
                sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
                
                logger.info(f"  {metric_name}: Δ={improvement_pct:+.2f}%, p={p_value:.4f} {sig}, d={cohens_d:.2f}")
                
                test_results.append({
                    'comparison': f"EcoChain-ML vs. {method}",
                    'metric': metric_name,
                    'ecochain_mean': np.mean(eco_values),
                    'baseline_mean': np.mean(method_values),
                    'improvement_pct': improvement_pct,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'significant': p_value < 0.05,
                    'improvement': improvement
                })
        
        # Save results
        test_df = pd.DataFrame(test_results)
        test_df.to_csv(self.metrics_dir / 'statistical_tests.csv', index=False)
        
        return test_results
    
    def _generate_comparison_table(self):
        """Generate comparison table CSV."""
        table_data = []
        
        # Reference: EcoChain-ML results
        eco = self.results['ecochain_ml']
        
        for method in ['kubernetes', 'casper', 'green_llm', 'compression_only', 'ecochain_ml']:
            r = self.results[method]
            
            # Calculate improvements vs Kubernetes (industry baseline)
            k8s = self.results['kubernetes']
            energy_vs_k8s = ((k8s['total_energy_kwh'] - r['total_energy_kwh']) / k8s['total_energy_kwh'] * 100)
            carbon_vs_k8s = ((k8s['total_carbon_gco2'] - r['total_carbon_gco2']) / k8s['total_carbon_gco2'] * 100)
            
            table_data.append({
                'Method': method.replace('_', ' ').title(),
                'Energy (kWh)': f"{r['total_energy_kwh']:.4f}",
                'Carbon (gCO2)': f"{r['total_carbon_gco2']:.2f}",
                'Latency (s)': f"{r['avg_latency_sec']:.4f}",
                'Renewable (%)': f"{r['renewable_percent']:.2f}",
                'Energy vs K8s (%)': f"{energy_vs_k8s:+.2f}",
                'Carbon vs K8s (%)': f"{carbon_vs_k8s:+.2f}",
            })
        
        df = pd.DataFrame(table_data)
        df.to_csv(self.metrics_dir / 'sota_comparison_table.csv', index=False)
        logger.info(f"Saved comparison table to {self.metrics_dir / 'sota_comparison_table.csv'}")
        
        # Print table
        print("\n" + "=" * 80)
        print("SOTA COMPARISON RESULTS")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)
    
    def _generate_latex_table(self):
        """Generate publication-ready LaTeX table."""
        
        latex = r"""
\begin{table}[t]
\centering
\caption{Comparison with State-of-the-Art Carbon-Aware Scheduling Approaches}
\label{tab:sota_comparison}
\small
\begin{tabular}{lcccccc}
\hline
\textbf{Method} & \textbf{Energy} & \textbf{Carbon} & \textbf{Latency} & \textbf{Renewable} & \textbf{Prediction} & \textbf{DVFS} \\
 & (kWh) & (gCO$_2$) & (s) & (\%) & & \\
\hline
"""
        
        method_features = {
            'kubernetes': ('No', 'No'),
            'casper': ('No', 'No'),
            'green_llm': ('Grid CI*', 'No'),
            'compression_only': ('No', 'No'),
            'ecochain_ml': ('Renewable', 'Yes'),
        }
        
        for method in ['kubernetes', 'casper', 'green_llm', 'compression_only', 'ecochain_ml']:
            r = self.results[method]
            pred, dvfs = method_features[method]
            
            name = method.replace('_', ' ').title()
            if method == 'ecochain_ml':
                name = r'\textbf{EcoChain-ML}'
            elif method == 'green_llm':
                name = 'GreenLLM (2025)'
            elif method == 'casper':
                name = 'CASPER (2024)'
            elif method == 'kubernetes':
                name = 'Kubernetes'
            elif method == 'compression_only':
                name = 'Compression Only'
            
            latex += f"{name} & {r['total_energy_kwh']:.4f} & {r['total_carbon_gco2']:.2f} & "
            latex += f"{r['avg_latency_sec']:.4f} & {r['renewable_percent']:.2f} & {pred} & {dvfs} \\\\\n"
        
        latex += r"""
\hline
\end{tabular}
\vspace{2mm}
\footnotesize{*Grid CI = Grid Carbon Intensity forecasting (no renewable prediction)}
\end{table}
"""
        
        with open(self.metrics_dir / 'sota_comparison_table.tex', 'w') as f:
            f.write(latex)
        
        logger.info(f"Saved LaTeX table to {self.metrics_dir / 'sota_comparison_table.tex'}")
    
    def _generate_comparison_plots(self):
        """Generate comparison visualizations."""
        
        methods = ['kubernetes', 'casper', 'green_llm', 'compression_only', 'ecochain_ml']
        labels = ['Kubernetes', 'CASPER\n(2024)', 'GreenLLM\n(2025)', 'Compression\nOnly', 'EcoChain-ML\n(Ours)']
        colors = ['#95a5a6', '#3498db', '#9b59b6', '#f39c12', '#27ae60']
        
        # Figure 1: Bar chart comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Energy comparison
        ax = axes[0, 0]
        energy = [self.results[m]['total_energy_kwh'] for m in methods]
        energy_err = [self.results[m]['total_energy_kwh_std'] for m in methods]
        bars = ax.bar(labels, energy, color=colors, yerr=energy_err, capsize=5, alpha=0.8)
        ax.set_ylabel('Energy (kWh)', fontweight='bold')
        ax.set_title('Energy Consumption (↓ Lower is Better)', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        # Highlight best
        min_idx = np.argmin(energy)
        bars[min_idx].set_edgecolor('black')
        bars[min_idx].set_linewidth(2)
        
        # Carbon comparison
        ax = axes[0, 1]
        carbon = [self.results[m]['total_carbon_gco2'] for m in methods]
        carbon_err = [self.results[m]['total_carbon_gco2_std'] for m in methods]
        bars = ax.bar(labels, carbon, color=colors, yerr=carbon_err, capsize=5, alpha=0.8)
        ax.set_ylabel('Carbon (gCO₂)', fontweight='bold')
        ax.set_title('Carbon Emissions (↓ Lower is Better)', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        min_idx = np.argmin(carbon)
        bars[min_idx].set_edgecolor('black')
        bars[min_idx].set_linewidth(2)
        
        # Latency comparison
        ax = axes[1, 0]
        latency = [self.results[m]['avg_latency_sec'] for m in methods]
        latency_err = [self.results[m]['avg_latency_sec_std'] for m in methods]
        bars = ax.bar(labels, latency, color=colors, yerr=latency_err, capsize=5, alpha=0.8)
        ax.set_ylabel('Latency (seconds)', fontweight='bold')
        ax.set_title('Average Latency (↓ Lower is Better)', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        min_idx = np.argmin(latency)
        bars[min_idx].set_edgecolor('black')
        bars[min_idx].set_linewidth(2)
        
        # Renewable comparison
        ax = axes[1, 1]
        renewable = [self.results[m]['renewable_percent'] for m in methods]
        renewable_err = [self.results[m]['renewable_percent_std'] for m in methods]
        bars = ax.bar(labels, renewable, color=colors, yerr=renewable_err, capsize=5, alpha=0.8)
        ax.set_ylabel('Renewable (%)', fontweight='bold')
        ax.set_title('Renewable Utilization (↑ Higher is Better)', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        max_idx = np.argmax(renewable)
        bars[max_idx].set_edgecolor('black')
        bars[max_idx].set_linewidth(2)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'sota_comparison_bars.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Radar chart
        self._plot_radar_chart(methods, labels, colors)
        
        # Figure 3: Improvement waterfall
        self._plot_improvement_waterfall()
        
        logger.info(f"Saved plots to {self.plots_dir}")
    
    def _plot_radar_chart(self, methods, labels, colors):
        """Generate radar chart for multi-metric comparison."""
        
        # Normalize metrics (0-1 scale, higher = better)
        metrics = ['Energy\nEfficiency', 'Carbon\nReduction', 'Latency', 'Renewable\nUsage']
        
        # Get values and normalize
        k8s = self.results['kubernetes']  # Baseline for normalization
        
        data = []
        for method in methods:
            r = self.results[method]
            # Normalize: higher = better
            energy_norm = k8s['total_energy_kwh'] / r['total_energy_kwh'] if r['total_energy_kwh'] > 0 else 1
            carbon_norm = k8s['total_carbon_gco2'] / r['total_carbon_gco2'] if r['total_carbon_gco2'] > 0 else 1
            latency_norm = k8s['avg_latency_sec'] / r['avg_latency_sec'] if r['avg_latency_sec'] > 0 else 1
            renewable_norm = r['renewable_percent'] / max(self.results[m]['renewable_percent'] for m in methods)
            
            data.append([energy_norm, carbon_norm, latency_norm, renewable_norm])
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for i, (method, label, color) in enumerate(zip(methods, labels, colors)):
            values = data[i] + data[i][:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=label.replace('\n', ' '), color=color)
            ax.fill(angles, values, alpha=0.1, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylim(0, 2)
        ax.set_title('Multi-Metric SOTA Comparison\n(Normalized to Kubernetes Baseline)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'sota_comparison_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_improvement_waterfall(self):
        """Plot improvement waterfall showing EcoChain-ML advantages."""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Compare EcoChain-ML vs each baseline
        baselines = ['kubernetes', 'casper', 'green_llm', 'compression_only']
        baseline_labels = ['vs. Kubernetes', 'vs. CASPER', 'vs. GreenLLM', 'vs. Compression']
        
        eco = self.results['ecochain_ml']
        
        carbon_improvements = []
        for b in baselines:
            baseline = self.results[b]
            improvement = (baseline['total_carbon_gco2'] - eco['total_carbon_gco2']) / baseline['total_carbon_gco2'] * 100
            carbon_improvements.append(improvement)
        
        colors = ['#27ae60' if x > 0 else '#e74c3c' for x in carbon_improvements]
        bars = ax.bar(baseline_labels, carbon_improvements, color=colors, alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, carbon_improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:+.1f}%',
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontweight='bold', fontsize=12)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Carbon Reduction (%)', fontweight='bold', fontsize=12)
        ax.set_title('EcoChain-ML Carbon Reduction vs. SOTA Baselines\n(Positive = EcoChain-ML Better)', 
                    fontweight='bold', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'sota_improvement_waterfall.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("EcoChain-ML SOTA Comparison Experiment")
    print("Comparing against: GreenLLM (2025), CASPER (2024), Kubernetes")
    print("=" * 80 + "\n")
    
    comparison = SOTAComparison()
    comparison.run_all_comparisons()
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED!")
    print(f"Results: {comparison.results_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
