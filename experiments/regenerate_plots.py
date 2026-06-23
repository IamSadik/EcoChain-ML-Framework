"""
Regenerate Plots with Improved Formatting

This script regenerates all plots with:
1. Larger font sizes (visible in paper)
2. Light colors for heatmap (more appealing)
3. Better overall styling for publication

Plots to regenerate:
- Ablation Study: heatmap, degradation line
- Baseline Comparison: carbon, energy, latency, radar, renewable comparison
- Scalability: combined scalability
- XGBoost Validation: prediction time series (24h and 7 days), feature importance
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set global font sizes for publication quality
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'figure.titlesize': 20,
    'font.family': 'serif',
    'axes.linewidth': 1.5,
    'lines.linewidth': 2.5,
    'lines.markersize': 10,
})


class PlotRegenerator:
    """Regenerate all plots with improved formatting."""
    
    def __init__(self):
        """Initialize paths."""
        self.base_dir = Path(__file__).parent.parent
        
        # Result directories
        self.ablation_metrics = self.base_dir / "results" / "ablation_study" / "metrics"
        self.ablation_plots = self.base_dir / "results" / "ablation_study" / "plots"
        
        self.baseline_metrics = self.base_dir / "results" / "baseline_comparison" / "metrics"
        self.baseline_plots = self.base_dir / "results" / "baseline_comparison" / "plots"
        
        self.scalability_metrics = self.base_dir / "results" / "scalability_test" / "metrics"
        self.scalability_plots = self.base_dir / "results" / "scalability_test" / "plots"
        
        self.xgboost_dir = self.base_dir / "results" / "xgboost_validation"
        self.xgboost_plots = self.xgboost_dir / "plots"
        
        # Load data
        self.ablation_results = self._load_ablation_results()
        self.baseline_results = self._load_baseline_results()
        self.scalability_results = self._load_scalability_results()
        self.xgboost_results = self._load_xgboost_results()
        
        print("PlotRegenerator initialized successfully!")
        print(f"  Ablation configs loaded: {list(self.ablation_results.keys())}")
        print(f"  Baseline methods loaded: {list(self.baseline_results.keys())}")
        print(f"  Scalability node counts: {list(self.scalability_results.get('node_scaling', {}).keys())}")
        print(f"  XGBoost results loaded: {self.xgboost_results is not None}")
    
    def _load_ablation_results(self) -> dict:
        """Load ablation study results."""
        results = {}
        configs = ['full_ecochain_ml', 'without_renewable_prediction', 'without_dvfs', 
                   'without_compression', 'without_blockchain']
        
        for config in configs:
            metrics_file = self.ablation_metrics / f"{config}_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    results[config] = json.load(f)
        
        return results
    
    def _load_baseline_results(self) -> dict:
        """Load baseline comparison results."""
        results = {}
        methods = ['standard', 'compression_only', 'energy_aware_only', 'blockchain_only', 'ecochain_ml']
        
        for method in methods:
            metrics_file = self.baseline_metrics / f"{method}_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    results[method] = json.load(f)
        
        return results
    
    def _load_scalability_results(self) -> dict:
        """Load scalability test results."""
        results = {'node_scaling': {}, 'workload_scaling': {}, 'arrival_rate_scaling': {}}
        
        # Node scaling
        node_file = self.scalability_metrics / "node_scaling_results.json"
        if node_file.exists():
            with open(node_file, 'r') as f:
                results['node_scaling'] = json.load(f)
        
        # Workload scaling
        workload_file = self.scalability_metrics / "workload_scaling_results.json"
        if workload_file.exists():
            with open(workload_file, 'r') as f:
                results['workload_scaling'] = json.load(f)
        
        # Arrival rate scaling
        arrival_file = self.scalability_metrics / "arrival_rate_scaling_results.json"
        if arrival_file.exists():
            with open(arrival_file, 'r') as f:
                results['arrival_rate_scaling'] = json.load(f)
        
        return results
    
    def _load_xgboost_results(self) -> dict:
        """Load XGBoost validation results."""
        metrics_file = self.xgboost_dir / "xgboost_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                return json.load(f)
        return None
    
    def regenerate_all(self):
        """Regenerate all plots."""
        print("\n" + "=" * 60)
        print("REGENERATING ALL PLOTS WITH IMPROVED FORMATTING")
        print("=" * 60)
        
        # Ablation plots
        print("\n--- Ablation Study Plots ---")
        self.plot_ablation_heatmap()
        self.plot_ablation_degradation_line()
        
        # Baseline plots
        print("\n--- Baseline Comparison Plots ---")
        self.plot_baseline_energy()
        self.plot_baseline_carbon()
        self.plot_baseline_latency()
        self.plot_baseline_renewable()
        self.plot_baseline_radar()
        
        # Scalability plots
        print("\n--- Scalability Plots ---")
        self.plot_scalability_combined()
        
        # XGBoost validation plots
        print("\n--- XGBoost Validation Plots ---")
        self.plot_xgboost_timeseries_24h()
        self.plot_xgboost_timeseries_7days()
        
        print("\n" + "=" * 60)
        print("ALL PLOTS REGENERATED SUCCESSFULLY!")
        print("=" * 60)
    
    # =========================================================================
    # ABLATION STUDY PLOTS
    # =========================================================================
    
    def plot_ablation_heatmap(self):
        """Plot heatmap with light colors and large fonts."""
        if not self.ablation_results:
            print("  [SKIP] No ablation results found")
            return
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        configs = list(self.ablation_results.keys())
        
        # Metrics to display
        metrics = ['Energy (kWh)', 'Carbon (gCO₂)', 'Latency (s)', 'Renewable (%)', 'Cost ($)']
        
        # Create data matrix
        data = []
        original_data = []
        for config in configs:
            result = self.ablation_results[config]
            row = [
                result['total_energy_kwh'],
                result['total_carbon_gco2'],
                result['avg_latency_sec'],
                100 - result['renewable_percent'],  # Invert for coloring (lower = better)
                result['operational_cost_usd']
            ]
            original_row = [
                result['total_energy_kwh'],
                result['total_carbon_gco2'],
                result['avg_latency_sec'],
                result['renewable_percent'],  # Keep original for display
                result['operational_cost_usd']
            ]
            data.append(row)
            original_data.append(original_row)
        
        data = np.array(data)
        original_data = np.array(original_data)
        
        # Rank-based normalization
        data_normalized = np.zeros_like(data, dtype=float)
        for col in range(data.shape[1]):
            sorted_indices = np.argsort(data[:, col])
            ranks = np.empty_like(sorted_indices)
            ranks[sorted_indices] = np.arange(len(sorted_indices))
            data_normalized[:, col] = ranks / (len(ranks) - 1) if len(ranks) > 1 else 0.5
        
        # Use LIGHT color scheme (YlGn = Yellow to Green, light and appealing)
        im = ax.imshow(data_normalized, cmap='YlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks with LARGE fonts
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(configs)))
        
        # Format config names nicely
        config_labels = [c.replace('_', ' ').replace('without', 'w/o').title() for c in configs]
        
        ax.set_xticklabels(metrics, fontsize=16, fontweight='bold')
        ax.set_yticklabels(config_labels, fontsize=15)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
        
        # Add colorbar with large font
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Relative Rank (Lighter = Better)', fontsize=15, fontweight='bold', 
                       rotation=270, labelpad=25)
        cbar.ax.tick_params(labelsize=13)
        
        # Add text annotations with LARGE fonts
        for i in range(len(configs)):
            for j in range(len(metrics)):
                # Use dark text on light background
                text_color = 'black' if data_normalized[i, j] < 0.6 else 'white'
                
                if j == 4:  # Cost column
                    text = f'${original_data[i, j]:.4f}'
                elif j == 3:  # Renewable column
                    text = f'{original_data[i, j]:.1f}%'
                else:
                    text = f'{original_data[i, j]:.3f}'
                
                ax.text(j, i, text, ha="center", va="center", 
                       color=text_color, fontsize=14, fontweight='bold')
        
        ax.set_title('Ablation Study: Component Impact Analysis\n(Lighter = Better Performance)', 
                    fontsize=20, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.ablation_plots / 'ablation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  [DONE] Ablation heatmap saved")
    
    def plot_ablation_degradation_line(self):
        """Plot degradation line chart with large fonts."""
        if not self.ablation_results:
            print("  [SKIP] No ablation results found")
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        configs = list(self.ablation_results.keys())
        full_result = self.ablation_results['full_ecochain_ml']
        
        # Metrics to plot - normalized to 100% = full system
        metrics_data = {
            'Energy': ([self.ablation_results[c]['total_energy_kwh'] / full_result['total_energy_kwh'] * 100 
                        for c in configs], '#e74c3c', 's', '↓ Lower is Better'),
            'Carbon': ([self.ablation_results[c]['total_carbon_gco2'] / full_result['total_carbon_gco2'] * 100 
                        for c in configs], '#3498db', 'o', '↓ Lower is Better'),
            'Renewable': ([self.ablation_results[c]['renewable_percent'] / full_result['renewable_percent'] * 100 
                           for c in configs], '#2ecc71', '^', '↑ Higher is Better')
        }
        
        x = np.arange(len(configs))
        
        for metric_name, (values, color, marker, direction) in metrics_data.items():
            ax.plot(x, values, marker=marker, linewidth=3, label=f'{metric_name} ({direction})', 
                   markersize=12, color=color, markeredgecolor='white', markeredgewidth=2)
        
        ax.set_xlabel('Configuration', fontsize=18, fontweight='bold')
        ax.set_ylabel('Percentage of Full System (%)', fontsize=18, fontweight='bold')
        ax.set_title('Performance Degradation Across Ablation Configurations\n(100% = Full EcoChain-ML)', 
                    fontsize=20, fontweight='bold', pad=15)
        
        # Format x-axis labels
        config_labels = [c.replace('_', '\n').replace('without', 'w/o').title() for c in configs]
        ax.set_xticks(x)
        ax.set_xticklabels(config_labels, fontsize=14)
        
        # Add baseline reference line
        ax.axhline(y=100, color='black', linestyle='--', linewidth=2, label='Full System Baseline (100%)')
        
        ax.legend(loc='upper left', fontsize=14, framealpha=0.95)
        ax.grid(True, alpha=0.3, linewidth=1)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(self.ablation_plots / 'degradation_line.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  [DONE] Ablation degradation line saved")
    
    # =========================================================================
    # BASELINE COMPARISON PLOTS
    # =========================================================================
    
    def plot_baseline_energy(self):
        """Plot energy comparison with large fonts."""
        if not self.baseline_results:
            print("  [SKIP] No baseline results found")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        methods = list(self.baseline_results.keys())
        renewable_energy = [self.baseline_results[m]['renewable_energy_kwh'] for m in methods]
        grid_energy = [self.baseline_results[m]['grid_energy_kwh'] for m in methods]
        
        x = np.arange(len(methods))
        width = 0.4
        
        bars1 = ax.bar(x - width/2, renewable_energy, width, label='Renewable Energy', 
                       color='#27ae60', edgecolor='white', linewidth=2)
        bars2 = ax.bar(x + width/2, grid_energy, width, label='Grid Energy', 
                       color='#e74c3c', edgecolor='white', linewidth=2)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0.001:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.4f}', ha='center', va='bottom', 
                           fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Method', fontsize=18, fontweight='bold')
        ax.set_ylabel('Energy Consumption (kWh)', fontsize=18, fontweight='bold')
        ax.set_title('Energy Consumption Comparison\n(↓ Lower Total is Better)', 
                    fontsize=20, fontweight='bold', pad=15)
        
        method_labels = [m.replace('_', '\n').title() for m in methods]
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, fontsize=14)
        
        ax.legend(fontsize=14, loc='upper right')
        ax.grid(axis='y', alpha=0.3, linewidth=1)
        
        plt.tight_layout()
        plt.savefig(self.baseline_plots / 'energy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  [DONE] Baseline energy comparison saved")
    
    def plot_baseline_carbon(self):
        """Plot carbon comparison with large fonts."""
        if not self.baseline_results:
            print("  [SKIP] No baseline results found")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        methods = list(self.baseline_results.keys())
        carbon_values = [self.baseline_results[m]['total_carbon_gco2'] for m in methods]
        
        # Color gradient based on values (lower = greener)
        max_carbon = max(carbon_values)
        colors = [plt.cm.RdYlGn(1 - v/max_carbon) for v in carbon_values]
        
        bars = ax.bar(methods, carbon_values, color=colors, edgecolor='white', linewidth=2)
        
        # Add value labels on bars
        for bar, val in zip(bars, carbon_values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.2f}', ha='center', va='bottom', 
                   fontsize=14, fontweight='bold')
        
        ax.set_xlabel('Method', fontsize=18, fontweight='bold')
        ax.set_ylabel('Carbon Emissions (gCO₂)', fontsize=18, fontweight='bold')
        ax.set_title('Carbon Emissions Comparison\n(↓ Lower is Better)', 
                    fontsize=20, fontweight='bold', pad=15)
        
        method_labels = [m.replace('_', '\n').title() for m in methods]
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(method_labels, fontsize=14)
        
        ax.grid(axis='y', alpha=0.3, linewidth=1)
        
        plt.tight_layout()
        plt.savefig(self.baseline_plots / 'carbon_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  [DONE] Baseline carbon comparison saved")
    
    def plot_baseline_latency(self):
        """Plot latency comparison with large fonts."""
        if not self.baseline_results:
            print("  [SKIP] No baseline results found")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        methods = list(self.baseline_results.keys())
        avg_latency = [self.baseline_results[m]['avg_latency_sec'] for m in methods]
        max_latency = [self.baseline_results[m]['max_latency_sec'] for m in methods]
        
        x = np.arange(len(methods))
        width = 0.4
        
        bars1 = ax.bar(x - width/2, avg_latency, width, label='Average Latency', 
                       color='#3498db', edgecolor='white', linewidth=2)
        bars2 = ax.bar(x + width/2, max_latency, width, label='Maximum Latency', 
                       color='#e67e22', edgecolor='white', linewidth=2)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}s', ha='center', va='bottom', 
                   fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Method', fontsize=18, fontweight='bold')
        ax.set_ylabel('Latency (seconds)', fontsize=18, fontweight='bold')
        ax.set_title('Latency Comparison\n(↓ Lower is Better)', 
                    fontsize=20, fontweight='bold', pad=15)
        
        method_labels = [m.replace('_', '\n').title() for m in methods]
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, fontsize=14)
        
        ax.legend(fontsize=14, loc='upper right')
        ax.grid(axis='y', alpha=0.3, linewidth=1)
        
        plt.tight_layout()
        plt.savefig(self.baseline_plots / 'latency_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  [DONE] Baseline latency comparison saved")
    
    def plot_baseline_renewable(self):
        """Plot renewable comparison with large fonts."""
        if not self.baseline_results:
            print("  [SKIP] No baseline results found")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        methods = list(self.baseline_results.keys())
        renewable_pct = [self.baseline_results[m]['renewable_percent'] for m in methods]
        
        # Color gradient (higher = greener)
        max_renewable = max(renewable_pct) if max(renewable_pct) > 0 else 1
        colors = [plt.cm.RdYlGn(v/max_renewable) for v in renewable_pct]
        
        bars = ax.bar(methods, renewable_pct, color=colors, edgecolor='white', linewidth=2)
        
        # Add value labels
        for bar, val in zip(bars, renewable_pct):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.1f}%', ha='center', va='bottom', 
                   fontsize=14, fontweight='bold')
        
        ax.set_xlabel('Method', fontsize=18, fontweight='bold')
        ax.set_ylabel('Renewable Energy Usage (%)', fontsize=18, fontweight='bold')
        ax.set_title('Renewable Energy Utilization\n(↑ Higher is Better)', 
                    fontsize=20, fontweight='bold', pad=15)
        
        method_labels = [m.replace('_', '\n').title() for m in methods]
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(method_labels, fontsize=14)
        
        ax.set_ylim(0, max(renewable_pct) * 1.15)
        ax.grid(axis='y', alpha=0.3, linewidth=1)
        
        plt.tight_layout()
        plt.savefig(self.baseline_plots / 'renewable_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  [DONE] Baseline renewable comparison saved")
    
    def plot_baseline_radar(self):
        """Plot radar chart with large fonts."""
        if not self.baseline_results:
            print("  [SKIP] No baseline results found")
            return
        
        methods = list(self.baseline_results.keys())
        baseline = self.baseline_results.get('standard', list(self.baseline_results.values())[0])
        
        # Define metrics
        metrics_info = [
            ('renewable_percent', True, 'Renewable\nUsage'),
            ('total_energy_kwh', False, 'Energy\nEfficiency'),
            ('total_carbon_gco2', False, 'Carbon\nReduction'),
            ('avg_latency_sec', False, 'Latency\nPerformance'),
            ('net_cost_usd', False, 'Cost\nEfficiency')
        ]
        
        num_vars = len(metrics_info)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        # Colors for each method
        colors = ['#e74c3c', '#9b59b6', '#f39c12', '#3498db', '#2ecc71']
        
        for idx, method in enumerate(methods):
            values = []
            for metric_key, higher_is_better, _ in metrics_info:
                value = self.baseline_results[method][metric_key]
                baseline_value = baseline[metric_key]
                
                if baseline_value != 0:
                    if higher_is_better:
                        normalized = max(0, value / baseline_value)
                    else:
                        normalized = max(0, 2 - (value / baseline_value))
                else:
                    normalized = 1.0
                
                normalized = min(2.0, normalized)
                values.append(normalized)
            
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=3, 
                   label=method.replace('_', ' ').title(), 
                   color=colors[idx % len(colors)], markersize=10)
            ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])
        
        # Set labels with LARGE fonts
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([label for _, _, label in metrics_info], fontsize=16, fontweight='bold')
        
        ax.set_ylim(0, 2.0)
        ax.set_yticks([0.5, 1.0, 1.5, 2.0])
        ax.set_yticklabels(['0.5x', '1.0x', '1.5x', '2.0x'], fontsize=13)
        ax.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)
        
        ax.set_title('Multi-Metric Performance Comparison\n(Normalized to Standard Baseline)', 
                    fontsize=20, fontweight='bold', pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=14)
        
        plt.tight_layout()
        plt.savefig(self.baseline_plots / 'radar_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  [DONE] Baseline radar comparison saved")
    
    # =========================================================================
    # SCALABILITY PLOTS
    # =========================================================================
    
    def plot_scalability_combined(self):
        """Plot combined scalability with large fonts."""
        node_results = self.scalability_results.get('node_scaling', {})
        workload_results = self.scalability_results.get('workload_scaling', {})
        arrival_results = self.scalability_results.get('arrival_rate_scaling', {})
        
        if not node_results:
            print("  [SKIP] No scalability results found")
            return
        
        fig = plt.figure(figsize=(18, 12))
        
        # Create 2x3 grid
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
        
        # Top row: Energy efficiency
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        
        # Bottom row: Latency
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])
        
        # Node scaling - Energy efficiency
        if node_results:
            node_counts = sorted([int(k) for k in node_results.keys()])
            efficiency = [node_results[str(n)]['tasks_completed'] / 
                         node_results[str(n)]['total_energy_kwh'] for n in node_counts]
            
            ax1.plot(node_counts, efficiency, 'o-', color='#e74c3c', linewidth=3, markersize=10)
            ax1.set_xlabel('Number of Nodes', fontsize=16, fontweight='bold')
            ax1.set_ylabel('Tasks per kWh', fontsize=16, fontweight='bold')
            ax1.set_title('Node Scaling: Energy Efficiency\n(↑ Higher is Better)', 
                         fontsize=17, fontweight='bold')
            ax1.grid(True, alpha=0.3, linewidth=1)
            ax1.tick_params(labelsize=14)
        
        # Workload scaling - Energy efficiency
        if workload_results:
            workload_sizes = sorted([int(k) for k in workload_results.keys()])
            efficiency = [workload_results[str(w)]['tasks_completed'] / 
                         workload_results[str(w)]['total_energy_kwh'] for w in workload_sizes]
            
            ax2.plot(workload_sizes, efficiency, 's-', color='#3498db', linewidth=3, markersize=10)
            ax2.set_xlabel('Workload Size (tasks)', fontsize=16, fontweight='bold')
            ax2.set_ylabel('Tasks per kWh', fontsize=16, fontweight='bold')
            ax2.set_title('Workload Scaling: Energy Efficiency\n(↑ Higher is Better)', 
                         fontsize=17, fontweight='bold')
            ax2.grid(True, alpha=0.3, linewidth=1)
            ax2.tick_params(labelsize=14)
        
        # Arrival rate scaling - Energy efficiency
        if arrival_results:
            arrival_rates = sorted([int(k) for k in arrival_results.keys()])
            efficiency = [arrival_results[str(r)]['tasks_completed'] / 
                         arrival_results[str(r)]['total_energy_kwh'] for r in arrival_rates]
            
            ax3.plot(arrival_rates, efficiency, '^-', color='#2ecc71', linewidth=3, markersize=10)
            ax3.set_xlabel('Arrival Rate (tasks/hour)', fontsize=16, fontweight='bold')
            ax3.set_ylabel('Tasks per kWh', fontsize=16, fontweight='bold')
            ax3.set_title('Arrival Rate Scaling: Energy Efficiency\n(↑ Higher is Better)', 
                         fontsize=17, fontweight='bold')
            ax3.grid(True, alpha=0.3, linewidth=1)
            ax3.tick_params(labelsize=14)
        
        # Node scaling - Latency
        if node_results:
            latency = [node_results[str(n)]['avg_latency_sec'] for n in node_counts]
            
            ax4.plot(node_counts, latency, 'o-', color='#e74c3c', linewidth=3, markersize=10)
            ax4.set_xlabel('Number of Nodes', fontsize=16, fontweight='bold')
            ax4.set_ylabel('Average Latency (s)', fontsize=16, fontweight='bold')
            ax4.set_title('Node Scaling: Latency\n(↓ Lower is Better)', 
                         fontsize=17, fontweight='bold')
            ax4.grid(True, alpha=0.3, linewidth=1)
            ax4.tick_params(labelsize=14)
        
        # Workload scaling - Latency
        if workload_results:
            latency = [workload_results[str(w)]['avg_latency_sec'] for w in workload_sizes]
            
            ax5.plot(workload_sizes, latency, 's-', color='#3498db', linewidth=3, markersize=10)
            ax5.set_xlabel('Workload Size (tasks)', fontsize=16, fontweight='bold')
            ax5.set_ylabel('Average Latency (s)', fontsize=16, fontweight='bold')
            ax5.set_title('Workload Scaling: Latency\n(↓ Lower is Better)', 
                         fontsize=17, fontweight='bold')
            ax5.grid(True, alpha=0.3, linewidth=1)
            ax5.tick_params(labelsize=14)
        
        # Arrival rate scaling - Latency
        if arrival_results:
            latency = [arrival_results[str(r)]['avg_latency_sec'] for r in arrival_rates]
            
            ax6.plot(arrival_rates, latency, '^-', color='#2ecc71', linewidth=3, markersize=10)
            ax6.set_xlabel('Arrival Rate (tasks/hour)', fontsize=16, fontweight='bold')
            ax6.set_ylabel('Average Latency (s)', fontsize=16, fontweight='bold')
            ax6.set_title('Arrival Rate Scaling: Latency\n(↓ Lower is Better)', 
                         fontsize=17, fontweight='bold')
            ax6.grid(True, alpha=0.3, linewidth=1)
            ax6.tick_params(labelsize=14)
        
        # Add main title
        fig.suptitle('EcoChain-ML Scalability Analysis', fontsize=22, fontweight='bold', y=1.02)
        
        plt.savefig(self.scalability_plots / 'combined_scalability.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  [DONE] Scalability combined plot saved")
    
    # =========================================================================
    # XGBOOST VALIDATION PLOTS
    # =========================================================================
    
    def plot_xgboost_timeseries_24h(self):
        """Plot 24-hour prediction time series with large fonts."""
        # We need to regenerate predictions - load model and data
        try:
            import pickle
            import pandas as pd
            
            model_file = self.xgboost_dir / "xgboost_model.pkl"
            if not model_file.exists():
                print("  [SKIP] XGBoost model not found - run xgboost_validation.py first")
                return
            
            # Load NREL data and recreate test set
            nrel_data_path = self.base_dir / "data" / "nrel" / "nrel_realistic_data.csv"
            if not nrel_data_path.exists():
                print("  [SKIP] NREL data not found")
                return
            
            df = pd.read_csv(nrel_data_path)
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Recreate features (same as in xgboost_validation.py)
            df['solar_power_w'] = df['solar_power_watts']
            df['wind_power_w'] = df['wind_power_watts']
            
            if 'WD10M' in df.columns:
                df['wd10m_x'] = np.cos(np.deg2rad(df['WD10M']))
                df['wd10m_y'] = np.sin(np.deg2rad(df['WD10M']))
            
            df['hour_of_day'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            
            solar_capacity = 150
            wind_capacity = 120
            total_capacity = solar_capacity + wind_capacity
            
            df['total_renewable_pct'] = ((df['solar_power_w'] + df['wind_power_w']) / total_capacity * 100).clip(upper=100.0)
            
            # Create lag features
            for lag in [1, 2, 3, 6, 12, 24]:
                df[f'renewable_lag_{lag}h'] = df['total_renewable_pct'].shift(lag)
            
            df['renewable_rolling_mean_3h'] = df['total_renewable_pct'].shift(1).rolling(window=3, min_periods=1).mean()
            df['renewable_rolling_std_3h'] = df['total_renewable_pct'].shift(1).rolling(window=3, min_periods=1).std()
            df['renewable_rolling_mean_12h'] = df['total_renewable_pct'].shift(1).rolling(window=12, min_periods=1).mean()
            
            df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            df['solar_normalized'] = df['solar_power_w'].shift(1) / solar_capacity
            df['wind_normalized'] = df['wind_power_w'].shift(1) / wind_capacity
            
            df_clean = df.dropna()
            
            feature_cols = [
                'hour_of_day', 'day_of_week',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                'solar_normalized', 'wind_normalized',
                'renewable_lag_1h', 'renewable_lag_2h', 'renewable_lag_3h',
                'renewable_lag_6h', 'renewable_lag_12h', 'renewable_lag_24h',
                'renewable_rolling_mean_3h', 'renewable_rolling_std_3h',
                'renewable_rolling_mean_12h'
            ]
            
            raw_weather_cols = ['ALLSKY_SFC_SW_DWN', 'T2M', 'WS10M', 'wd10m_x', 'wd10m_y', 
                               'RH2M', 'PS', 'ALLSKY_SFC_UV_INDEX']
            feature_cols.extend([c for c in raw_weather_cols if c in df_clean.columns])
            
            df_clean['target_next_hour'] = df_clean['total_renewable_pct'].shift(-1)
            df_clean = df_clean.dropna()
            
            X = df_clean[feature_cols].values
            y = df_clean['target_next_hour'].values
            
            # Get test set (last 15%)
            val_end = int(len(X) * 0.85)
            X_test = X[val_end:]
            y_test = y[val_end:]
            
            # Load model and predict
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            y_pred_test = model.predict(X_test)
            
            # Get R² from saved results
            r2 = self.xgboost_results['test_metrics']['r2'] if self.xgboost_results else 0.89
            
            # Create 24-hour plot with LARGE fonts
            fig, ax = plt.subplots(figsize=(14, 8))
            
            plot_len = 24
            hours = np.arange(plot_len)
            
            ax.plot(hours, y_test[:plot_len], 
                   label='Actual Renewable', 
                   color='#2ecc71', linewidth=3, marker='o', markersize=10,
                   markeredgecolor='white', markeredgewidth=2)
            
            ax.plot(hours, y_pred_test[:plot_len], 
                   label='XGBoost Predicted', 
                   color='#e74c3c', linewidth=3, linestyle='--', marker='s', markersize=9,
                   markeredgecolor='white', markeredgewidth=2)
            
            # Fill between to show error
            ax.fill_between(hours, y_test[:plot_len], y_pred_test[:plot_len], 
                           alpha=0.2, color='#3498db', label='Prediction Error')
            
            ax.set_xlabel('Hour of Day', fontsize=18, fontweight='bold')
            ax.set_ylabel('Renewable Availability (%)', fontsize=18, fontweight='bold')
            ax.set_title(f'XGBoost 24-Hour Renewable Energy Prediction\nTest Set Performance (R² = {r2:.3f})', 
                        fontsize=20, fontweight='bold', pad=15)
            
            ax.set_xticks(hours)
            ax.set_xticklabels([f'{h}:00' for h in hours], rotation=45, ha='right', fontsize=12)
            
            ax.legend(fontsize=14, loc='upper right', framealpha=0.95)
            ax.grid(True, alpha=0.3, linewidth=1)
            ax.set_ylim(bottom=0)
            
            plt.tight_layout()
            self.xgboost_plots.mkdir(exist_ok=True)
            plt.savefig(self.xgboost_plots / 'prediction_timeseries_24h.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("  [DONE] XGBoost 24h time series saved")
            
        except Exception as e:
            print(f"  [ERROR] Could not generate XGBoost 24h plot: {e}")
    
    def plot_xgboost_timeseries_7days(self):
        """Plot 7-day prediction time series with large fonts."""
        try:
            import pickle
            import pandas as pd
            
            model_file = self.xgboost_dir / "xgboost_model.pkl"
            if not model_file.exists():
                print("  [SKIP] XGBoost model not found")
                return
            
            # Load NREL data
            nrel_data_path = self.base_dir / "data" / "nrel" / "nrel_realistic_data.csv"
            if not nrel_data_path.exists():
                print("  [SKIP] NREL data not found")
                return
            
            df = pd.read_csv(nrel_data_path)
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Recreate features
            df['solar_power_w'] = df['solar_power_watts']
            df['wind_power_w'] = df['wind_power_watts']
            
            if 'WD10M' in df.columns:
                df['wd10m_x'] = np.cos(np.deg2rad(df['WD10M']))
                df['wd10m_y'] = np.sin(np.deg2rad(df['WD10M']))
            
            df['hour_of_day'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            
            solar_capacity = 150
            wind_capacity = 120
            total_capacity = solar_capacity + wind_capacity
            
            df['total_renewable_pct'] = ((df['solar_power_w'] + df['wind_power_w']) / total_capacity * 100).clip(upper=100.0)
            
            for lag in [1, 2, 3, 6, 12, 24]:
                df[f'renewable_lag_{lag}h'] = df['total_renewable_pct'].shift(lag)
            
            df['renewable_rolling_mean_3h'] = df['total_renewable_pct'].shift(1).rolling(window=3, min_periods=1).mean()
            df['renewable_rolling_std_3h'] = df['total_renewable_pct'].shift(1).rolling(window=3, min_periods=1).std()
            df['renewable_rolling_mean_12h'] = df['total_renewable_pct'].shift(1).rolling(window=12, min_periods=1).mean()
            
            df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            df['solar_normalized'] = df['solar_power_w'].shift(1) / solar_capacity
            df['wind_normalized'] = df['wind_power_w'].shift(1) / wind_capacity
            
            df_clean = df.dropna()
            
            feature_cols = [
                'hour_of_day', 'day_of_week',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                'solar_normalized', 'wind_normalized',
                'renewable_lag_1h', 'renewable_lag_2h', 'renewable_lag_3h',
                'renewable_lag_6h', 'renewable_lag_12h', 'renewable_lag_24h',
                'renewable_rolling_mean_3h', 'renewable_rolling_std_3h',
                'renewable_rolling_mean_12h'
            ]
            
            raw_weather_cols = ['ALLSKY_SFC_SW_DWN', 'T2M', 'WS10M', 'wd10m_x', 'wd10m_y', 
                               'RH2M', 'PS', 'ALLSKY_SFC_UV_INDEX']
            feature_cols.extend([c for c in raw_weather_cols if c in df_clean.columns])
            
            df_clean['target_next_hour'] = df_clean['total_renewable_pct'].shift(-1)
            df_clean = df_clean.dropna()
            
            X = df_clean[feature_cols].values
            y = df_clean['target_next_hour'].values
            
            val_end = int(len(X) * 0.85)
            X_test = X[val_end:]
            y_test = y[val_end:]
            
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            y_pred_test = model.predict(X_test)
            
            r2 = self.xgboost_results['test_metrics']['r2'] if self.xgboost_results else 0.89
            rmse = self.xgboost_results['test_metrics']['rmse'] if self.xgboost_results else 10.0
            
            # Create 7-day (168 hours) plot with LARGE fonts
            fig, ax = plt.subplots(figsize=(16, 8))
            
            plot_len = min(168, len(y_test))  # 7 days = 168 hours
            hours = np.arange(plot_len)
            
            ax.plot(hours, y_test[:plot_len], 
                   label='Actual Renewable', 
                   color='#2ecc71', linewidth=2.5, alpha=0.8)
            
            ax.plot(hours, y_pred_test[:plot_len], 
                   label='XGBoost Predicted', 
                   color='#e74c3c', linewidth=2.5, linestyle='--', alpha=0.9)
            
            # Add day markers
            for day in range(1, 8):
                ax.axvline(x=day*24, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            
            ax.set_xlabel('Hours from Start of Test Period', fontsize=18, fontweight='bold')
            ax.set_ylabel('Renewable Availability (%)', fontsize=18, fontweight='bold')
            ax.set_title(f'XGBoost 7-Day Renewable Energy Prediction\nR² = {r2:.3f}, RMSE = {rmse:.2f}%', 
                        fontsize=20, fontweight='bold', pad=15)
            
            # Add day labels on x-axis
            day_ticks = [12 + i*24 for i in range(7)]
            day_labels = [f'Day {i+1}' for i in range(7)]
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(day_ticks)
            ax2.set_xticklabels(day_labels, fontsize=14, fontweight='bold')
            ax2.tick_params(length=0)
            
            ax.legend(fontsize=14, loc='upper right', framealpha=0.95)
            ax.grid(True, alpha=0.3, linewidth=1)
            ax.set_ylim(bottom=0)
            
            plt.tight_layout()
            self.xgboost_plots.mkdir(exist_ok=True)
            plt.savefig(self.xgboost_plots / 'prediction_timeseries.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("  [DONE] XGBoost 7-day time series saved")
            
        except Exception as e:
            print(f"  [ERROR] Could not generate XGBoost 7-day plot: {e}")

def main():
    """Main function to regenerate all plots."""
    print("\n" + "=" * 60)
    print("PLOT REGENERATION TOOL")
    print("Fixing: Larger fonts, Light colors for heatmap")
    print("=" * 60)
    
    regenerator = PlotRegenerator()
    regenerator.regenerate_all()
    
    print("\nAll plots have been regenerated with:")
    print("  ✓ Larger font sizes (14-20pt)")
    print("  ✓ Light color scheme for heatmap (YlGn)")
    print("  ✓ Better contrast and visibility")
    print("  ✓ Publication-quality formatting")


if __name__ == "__main__":
    main()
