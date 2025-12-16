"""
Edge Node Simulator for EcoChain-ML Framework

This module simulates edge computing nodes with renewable energy sources,
DVFS capabilities, and energy monitoring.
"""

import numpy as np
import time
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EdgeNode:
    """
    Simulates an edge computing node with renewable energy capabilities.
    
    This class models:
    - CPU/GPU computational resources
    - Renewable energy sources (solar, wind, hybrid)
    - Dynamic voltage and frequency scaling (DVFS)
    - Energy consumption tracking
    - Battery management
    """
    
    def __init__(self, config: Dict[str, Any], renewable_trace: Optional[Dict[float, float]] = None):
        """
        Initialize an edge node with given configuration.
        
        Args:
            config: Dictionary containing node configuration
                Required keys: id, name, cpu_cores, gpu_available, 
                renewable_source, renewable_capacity_watts, etc.
            renewable_trace: Optional pre-computed renewable power trace for deterministic behavior.
                Dictionary mapping time (hours) to power (Watts).
                If provided, overrides stochastic generation.
        """
        # Node identification
        self.node_id = config['id']
        self.name = config['name']
        
        # Hardware configuration
        self.cpu_cores = config['cpu_cores']
        self.gpu_available = config['gpu_available']
        self.gpu_memory_gb = config.get('gpu_memory_gb', 0)
        
        # Frequency scaling parameters (DVFS)
        self.current_frequency = config['max_frequency_ghz']
        self.max_frequency = config['max_frequency_ghz']
        self.min_frequency = config['min_frequency_ghz']
        
        # Power consumption parameters
        self.base_power = config['base_power_watts']  # Idle power
        self.max_power = config['max_power_watts']    # Full load power
        
        # Renewable energy configuration
        self.renewable_source = config['renewable_source']
        self.renewable_capacity = config['renewable_capacity_watts']
        self.battery_percent = config.get('initial_battery_percent', 50)
        
        # CRITICAL: Grid-only nodes must have zero renewable capacity
        if self.renewable_source == 'grid':
            self.renewable_capacity = 0
        
        # FIX Issue 6: Support deterministic renewable traces
        self.renewable_trace = renewable_trace  # Dictionary: {time_hours: power_watts}
        self.use_deterministic_trace = (renewable_trace is not None and 
                                        self.renewable_source != 'grid' and 
                                        self.renewable_capacity > 0)
        
        # REALISTIC BATTERY STORAGE: Very limited capacity for edge devices
        # Real edge nodes have small batteries (5-15 minutes, not hours)
        # Battery serves to smooth out short-term variability, not store large amounts
        # This is NOT grid-scale storage - these are IoT/edge devices with minimal backup
        if self.renewable_source == 'solar':
            # Solar: ~9 minutes of storage at avg power (0.15 hours)
            # Edge devices need minimal buffer for cloud transitions
            self.battery_capacity_kwh = (self.renewable_capacity * 0.15) / 1000.0
        elif self.renewable_source == 'wind':
            # Wind: ~6 minutes of storage (0.10 hours) - wind changes very quickly
            # Minimal smoothing for rapid wind fluctuations
            self.battery_capacity_kwh = (self.renewable_capacity * 0.10) / 1000.0
        elif self.renewable_source == 'hybrid':
            # Hybrid: ~12 minutes of storage (0.20 hours)
            # Slightly more capacity due to dual sources
            self.battery_capacity_kwh = (self.renewable_capacity * 0.20) / 1000.0
        else:
            self.battery_capacity_kwh = 0.0
        
        # State tracking
        self.is_busy = False
        self.current_task = None
        
        # Metrics tracking
        self.tasks_completed = 0
        self.total_energy_consumed = 0.0  # kWh
        self.renewable_energy_used = 0.0  # kWh
        self.grid_energy_used = 0.0       # kWh
        self.total_execution_time = 0.0   # seconds
        
        # Track renewable energy budget with battery constraint
        # Renewable energy can only accumulate up to battery capacity
        self.renewable_energy_generated_kwh = 0.0  # Total generated during simulation
        self.renewable_energy_available_kwh = 0.0  # Current available (capped by battery)
        self.last_generation_update_time = 0.0     # Last time we updated generation
        
        # History for analysis
        self.energy_history = []
        self.frequency_history = []
        self.task_history = []
        
        logger.info(f"Initialized {self.name} (ID: {self.node_id})")
        logger.info(f"  Renewable: {self.renewable_source}, "
                   f"Capacity: {self.renewable_capacity}W, "
                   f"Deterministic: {self.use_deterministic_trace}")
    
    def get_renewable_power(self, current_time: float) -> float:
        """
        Calculate available renewable power at given time.
        
        Args:
            current_time: Current simulation time in hours (0-24)
            
        Returns:
            Available renewable power in Watts
        """
        # FIX Issue 6: Use deterministic trace if available
        if self.use_deterministic_trace:
            return self._get_trace_power(current_time)
        
        if self.renewable_source == 'solar':
            return self._get_solar_power(current_time)
        elif self.renewable_source == 'wind':
            return self._get_wind_power(current_time)
        elif self.renewable_source == 'hybrid':
            return self._get_hybrid_power(current_time)
        else:  # grid only
            return 0.0
    
    def _get_trace_power(self, current_time: float) -> float:
        """
        Get renewable power from pre-computed deterministic trace.
        
        Uses linear interpolation between trace points.
        """
        if not self.renewable_trace:
            return 0.0
        
        # Get sorted time points
        times = sorted(self.renewable_trace.keys())
        
        if not times:
            return 0.0
        
        # Handle edge cases
        if current_time <= times[0]:
            return self.renewable_trace[times[0]]
        if current_time >= times[-1]:
            return self.renewable_trace[times[-1]]
        
        # Find bracketing time points for interpolation
        for i in range(len(times) - 1):
            if times[i] <= current_time <= times[i + 1]:
                t0, t1 = times[i], times[i + 1]
                p0, p1 = self.renewable_trace[t0], self.renewable_trace[t1]
                
                # Linear interpolation
                alpha = (current_time - t0) / (t1 - t0)
                power = p0 + alpha * (p1 - p0)
                
                return max(0.0, power)
        
        # Fallback
        return 0.0
    
    def _get_solar_power(self, current_time: float) -> float:
        """
        Simulate solar power generation using sinusoidal pattern.
        
        Solar power peaks at noon and is zero at night.
        Realistic capacity factor: ~20-25% average over 24 hours
        """
        hour = current_time % 24
        
        # Solar is available from 6 AM to 6 PM
        if 6 <= hour <= 18:
            # Sinusoidal pattern: peak at noon (hour 12)
            solar_factor = np.sin((hour - 6) * np.pi / 12)
            
            # Add realistic variability (clouds, etc.) - more conservative
            variability = np.random.normal(0.85, 0.15)  # Mean 85% of peak
            variability = np.clip(variability, 0.4, 1.0)
            
            power = self.renewable_capacity * solar_factor * variability
            return max(0, power)
        else:
            return 0.0
    
    def _get_wind_power(self, current_time: float) -> float:
        """
        Simulate wind power generation using stochastic model.
        
        Wind power is more variable and available 24/7, but HIGHLY INTERMITTENT.
        Realistic capacity factor: ~25-35% on average.
        
        FIXED: Now uses Weibull distribution (standard for wind modeling) instead of
        unrealistic binary on/off behavior. This provides continuous variability
        while maintaining realistic average capacity factor of ~30%.
        """
        # Weibull distribution parameters (standard in wind energy modeling)
        # Shape k=2.0 (Rayleigh distribution) is typical for wind speeds
        # Scale adjusted to achieve ~30% average capacity factor
        shape_k = 2.0
        scale_lambda = 0.35
        
        # Generate wind capacity factor using Weibull distribution
        # This gives continuous values, not binary on/off
        capacity_factor = np.random.weibull(shape_k) * scale_lambda
        capacity_factor = np.clip(capacity_factor, 0.0, 0.95)  # Cap at 95% max
        
        # Add diurnal pattern (wind is typically stronger at night and morning)
        hour = current_time % 24
        if 0 <= hour < 6 or 18 <= hour < 24:
            # Night and evening: slightly stronger wind (10% boost)
            diurnal_factor = 1.10
        else:
            # Day: slightly weaker wind (10% reduction)
            diurnal_factor = 0.90
        
        # Calculate power output
        power = self.renewable_capacity * capacity_factor * diurnal_factor
        
        # Ensure within valid range
        return max(0.0, min(power, self.renewable_capacity))
    
    def _get_hybrid_power(self, current_time: float) -> float:
        """
        Simulate hybrid (solar + small wind) power generation.
        
        Combines solar during day with small wind component.
        Capacity factor: ~30-40% due to diversification
        """
        hour = current_time % 24
        
        # 70% solar capacity, 30% wind capacity
        solar_capacity = self.renewable_capacity * 0.7
        wind_capacity = self.renewable_capacity * 0.3
        
        # Solar component (same as solar but with smaller capacity)
        solar_power = 0.0
        if 6 <= hour <= 18:
            solar_factor = np.sin((hour - 6) * np.pi / 12)
            variability = np.random.normal(0.85, 0.12)
            variability = np.clip(variability, 0.5, 1.0)
            solar_power = solar_capacity * solar_factor * variability
        
        # Wind component (smaller, steadier)
        wind_factor = 0.35
        wind_variability = np.random.normal(1.0, 0.25)
        wind_variability = np.clip(wind_variability, 0.3, 1.3)
        wind_power = wind_capacity * wind_factor * wind_variability
        
        return max(0, solar_power + wind_power)
    
    def _update_renewable_generation(self, current_time: float) -> None:
        """
        Update the renewable energy generated since last check.
        This models renewable as a resource that accumulates over time.
        
        CRITICAL: Renewable energy is capped by battery capacity.
        Excess generation beyond battery capacity is wasted (can't be stored).
        """
        if current_time <= self.last_generation_update_time:
            return
        
        # Grid-only nodes never generate renewable
        if self.renewable_source == 'grid' or self.renewable_capacity == 0:
            self.last_generation_update_time = current_time
            return
            
        time_delta_hours = current_time - self.last_generation_update_time
        
        # Calculate average renewable power over this time period
        # Sample at multiple points for accuracy
        num_samples = max(1, int(time_delta_hours * 6))  # 6 samples per hour
        total_power = 0.0
        for i in range(num_samples):
            sample_time = self.last_generation_update_time + (i + 0.5) * time_delta_hours / num_samples
            total_power += self.get_renewable_power(sample_time)
        avg_power = total_power / num_samples
        
        # Energy generated (kWh) = Power (W) * Time (h) / 1000
        energy_generated = (avg_power * time_delta_hours) / 1000.0
        
        # CRITICAL FIX: Calculate current available renewable (considering what's been used)
        current_available = self.renewable_energy_generated_kwh - self.renewable_energy_used
        
        # Add newly generated energy, but cap at battery capacity
        new_available = current_available + energy_generated
        
        # REALISTIC CONSTRAINT: Battery can only store limited energy
        # If battery is full, excess generation is wasted
        if new_available > self.battery_capacity_kwh:
            # Battery is full - cap the available energy
            self.renewable_energy_generated_kwh = self.renewable_energy_used + self.battery_capacity_kwh
            # Excess energy is wasted (logged for analysis)
            energy_wasted = new_available - self.battery_capacity_kwh
        else:
            # Battery has capacity - store all generated energy
            self.renewable_energy_generated_kwh += energy_generated
        
        self.last_generation_update_time = current_time
    
    def calculate_power_consumption(self, utilization: float = 1.0) -> float:
        """
        Calculate current power consumption based on frequency and utilization.
        
        Power consumption scales with frequency and utilization:
        P = P_base + (P_max - P_base) * (freq/freq_max) * utilization
        
        Args:
            utilization: CPU/GPU utilization (0.0 to 1.0)
            
        Returns:
            Current power consumption in Watts
        """
        freq_ratio = self.current_frequency / self.max_frequency
        dynamic_power = (self.max_power - self.base_power) * freq_ratio * utilization
        total_power = self.base_power + dynamic_power
        
        return total_power
    
    def set_frequency(self, target_frequency: float) -> None:
        """
        Set CPU/GPU frequency (DVFS).
        
        Args:
            target_frequency: Target frequency in GHz
        """
        # Clamp to valid range
        self.current_frequency = np.clip(
            target_frequency,
            self.min_frequency,
            self.max_frequency
        )
        
        self.frequency_history.append({
            'time': time.time(),
            'frequency': self.current_frequency
        })
        
        logger.debug(f"{self.name}: Frequency set to {self.current_frequency:.2f} GHz")
    
    def execute_task(
        self, 
        task: Dict[str, Any], 
        current_time: float,
        compressed: bool = False
    ) -> Dict[str, Any]:
        """
        Execute an ML inference task on this node.
        
        Args:
            task: Task dictionary with 'id', 'model', 'complexity', etc.
            current_time: Current simulation time in hours
            compressed: Whether model compression is applied
            
        Returns:
            Result dictionary with execution metrics
        """
        if self.is_busy:
            raise RuntimeError(f"{self.name} is already busy!")
        
        self.is_busy = True
        self.current_task = task
        
        # Calculate execution time based on task complexity and frequency
        # Base execution time (at max frequency)
        base_time = task.get('execution_time', 1.0)  # seconds
        
        # Adjust for current frequency (lower freq = slower execution)
        freq_ratio = self.current_frequency / self.max_frequency
        execution_time = base_time / freq_ratio
        
        # Apply compression speedup (quantization reduces computation)
        # REALISTIC VALUES: INT8 quantization provides ~1.2-1.4x speedup and ~15-25% energy reduction
        if compressed:
            compression_speedup = 1.25  # 25% faster with INT8 quantization (realistic)
            compression_energy_factor = 0.80  # 20% less energy (realistic for INT8)
            execution_time = execution_time / compression_speedup
        else:
            compression_energy_factor = 1.0
        
        # Add small random variation
        execution_time *= np.random.uniform(0.95, 1.05)
        
        # Calculate average utilization during task
        avg_utilization = 0.85  # Typical ML inference utilization
        
        # Calculate power consumption during execution
        # REALISTIC: Power scales with frequency^1.5 (not ^2, accounting for base power)
        freq_power_factor = 0.35 + 0.65 * (freq_ratio ** 1.5)  # 35% base + 65% dynamic
        base_power = self.calculate_power_consumption(avg_utilization)
        effective_power = base_power * freq_power_factor * compression_energy_factor
        
        # Calculate energy consumed (kWh)
        energy_kwh = (effective_power * execution_time) / (1000 * 3600)
        
        # CRITICAL FIX: Grid-only nodes NEVER have renewable energy
        if self.renewable_source == 'grid' or self.renewable_capacity == 0:
            # Grid-only node - all energy comes from grid
            renewable_energy = 0.0
            grid_energy = energy_kwh
        else:
            # Update renewable generation up to current time
            self._update_renewable_generation(current_time)
            
            # Calculate how much renewable energy is AVAILABLE (generated - already used)
            available_renewable_kwh = max(0, self.renewable_energy_generated_kwh - self.renewable_energy_used)
            
            # Determine energy sources based on available renewable budget
            if available_renewable_kwh >= energy_kwh:
                # Enough renewable available - use all from renewable
                renewable_energy = energy_kwh
                grid_energy = 0.0
            elif available_renewable_kwh > 0:
                # Partial renewable - use what's available from budget
                renewable_energy = available_renewable_kwh
                grid_energy = energy_kwh - available_renewable_kwh
            else:
                # No renewable budget available
                renewable_energy = 0.0
                grid_energy = energy_kwh
        
        # Update metrics
        self.total_energy_consumed += energy_kwh
        self.renewable_energy_used += renewable_energy
        self.grid_energy_used += grid_energy
        self.total_execution_time += execution_time
        self.tasks_completed += 1
        
        # Record in history
        result = {
            'task_id': task.get('id', 'unknown'),
            'node_id': self.node_id,
            'execution_time': execution_time,
            'energy_consumed': energy_kwh,
            'renewable_energy': renewable_energy,
            'grid_energy': grid_energy,
            'renewable_percent': (renewable_energy / energy_kwh * 100) if energy_kwh > 0 else 0,
            'frequency': self.current_frequency,
            'compressed': compressed,
            'timestamp': current_time
        }
        
        self.task_history.append(result)
        self.energy_history.append({
            'time': current_time,
            'energy': energy_kwh,
            'renewable': renewable_energy
        })
        
        # Task completed
        self.is_busy = False
        self.current_task = None
        
        logger.debug(f"{self.name}: Completed task {task.get('id', 'unknown')} "
                    f"in {execution_time:.3f}s, "
                    f"Energy: {energy_kwh*1000:.4f} Wh "
                    f"({result['renewable_percent']:.1f}% renewable)")
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current node status and metrics.
        
        Returns:
            Dictionary with current status and accumulated metrics
        """
        renewable_pct = (
            (self.renewable_energy_used / self.total_energy_consumed * 100)
            if self.total_energy_consumed > 0 else 0
        )
        
        return {
            'node_id': self.node_id,
            'name': self.name,
            'is_busy': self.is_busy,
            'current_frequency': self.current_frequency,
            'tasks_completed': self.tasks_completed,
            'total_energy_kwh': self.total_energy_consumed,
            'renewable_energy_kwh': self.renewable_energy_used,
            'grid_energy_kwh': self.grid_energy_used,
            'renewable_percent': renewable_pct,
            'avg_task_time': (
                self.total_execution_time / self.tasks_completed 
                if self.tasks_completed > 0 else 0
            )
        }
    
    def reset_metrics(self) -> None:
        """Reset all accumulated metrics."""
        self.tasks_completed = 0
        self.total_energy_consumed = 0.0
        self.renewable_energy_used = 0.0
        self.grid_energy_used = 0.0
        self.total_execution_time = 0.0
        self.energy_history = []
        self.frequency_history = []
        self.task_history = []
        
        # Reset renewable tracking
        self.renewable_energy_generated_kwh = 0.0
        self.last_generation_update_time = 0.0
        
        logger.info(f"{self.name}: Metrics reset")
    
    def __repr__(self) -> str:
        return (f"EdgeNode(id={self.node_id}, name={self.name}, "
                f"renewable={self.renewable_source}, "
                f"tasks={self.tasks_completed})")
