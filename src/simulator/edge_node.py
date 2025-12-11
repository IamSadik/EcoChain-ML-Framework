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
    - Renewable energy sources (solar, wind)
    - Dynamic voltage and frequency scaling (DVFS)
    - Energy consumption tracking
    - Battery management
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize an edge node with given configuration.
        
        Args:
            config: Dictionary containing node configuration
                Required keys: id, name, cpu_cores, gpu_available, 
                renewable_source, renewable_capacity_watts, etc.
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
        self.battery_percent = config['initial_battery_percent']
        
        # State tracking
        self.is_busy = False
        self.current_task = None
        
        # Metrics tracking
        self.tasks_completed = 0
        self.total_energy_consumed = 0.0  # kWh
        self.renewable_energy_used = 0.0  # kWh
        self.grid_energy_used = 0.0       # kWh
        self.total_execution_time = 0.0   # seconds
        
        # History for analysis
        self.energy_history = []
        self.frequency_history = []
        self.task_history = []
        
        logger.info(f"Initialized {self.name} (ID: {self.node_id})")
        logger.info(f"  Renewable: {self.renewable_source}, "
                   f"Capacity: {self.renewable_capacity}W")
    
    def get_renewable_power(self, current_time: float) -> float:
        """
        Calculate available renewable power at given time.
        
        Args:
            current_time: Current simulation time in hours (0-24)
            
        Returns:
            Available renewable power in Watts
        """
        if self.renewable_source == 'solar':
            return self._get_solar_power(current_time)
        elif self.renewable_source == 'wind':
            return self._get_wind_power(current_time)
        else:  # grid only
            return 0.0
    
    def _get_solar_power(self, current_time: float) -> float:
        """
        Simulate solar power generation using sinusoidal pattern.
        
        Solar power peaks at noon and is zero at night.
        """
        hour = current_time % 24
        
        # Solar is available from 6 AM to 6 PM
        if 6 <= hour <= 18:
            # Sinusoidal pattern: peak at noon (hour 12)
            # Normalize to 0-1, then scale by capacity
            solar_factor = np.sin((hour - 6) * np.pi / 12)
            
            # Add some random variability (clouds, etc.)
            variability = np.random.normal(1.0, 0.1)
            variability = np.clip(variability, 0.7, 1.3)
            
            power = self.renewable_capacity * solar_factor * variability
            return max(0, power)
        else:
            return 0.0
    
    def _get_wind_power(self, current_time: float) -> float:
        """
        Simulate wind power generation using stochastic model.
        
        Wind power is more variable and available 24/7.
        """
        # Wind capacity factor typically 30-40%
        base_factor = 0.35
        
        # Add significant variability
        variability = np.random.normal(1.0, 0.4)
        variability = np.clip(variability, 0.2, 1.5)
        
        power = self.renewable_capacity * base_factor * variability
        return max(0, power)
    
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
        # Using cubic relationship: time scales with 1/freq
        freq_ratio = self.current_frequency / self.max_frequency
        execution_time = base_time / freq_ratio
        
        # Apply compression speedup (quantization reduces computation significantly)
        if compressed:
            # INT8 quantization provides ~2x speedup and ~40% energy reduction
            compression_speedup = 1.8  # 80% faster with INT8 quantization
            compression_energy_factor = 0.6  # 40% less energy
            execution_time = execution_time / compression_speedup
        else:
            compression_energy_factor = 1.0
        
        # Add small random variation
        execution_time *= np.random.uniform(0.95, 1.05)
        
        # Calculate average utilization during task
        avg_utilization = 0.85  # Typical ML inference utilization
        
        # Calculate power consumption during execution
        # Power scales with frequency^2 * voltage (and voltage scales with frequency)
        # So power roughly scales with frequency^3, but we use ^2 for conservative estimate
        freq_power_factor = freq_ratio ** 2
        base_power = self.calculate_power_consumption(avg_utilization)
        effective_power = base_power * freq_power_factor * compression_energy_factor
        
        # Calculate energy consumed (kWh)
        # Energy = Power * Time
        energy_kwh = (effective_power * execution_time) / (1000 * 3600)
        
        # Check renewable energy availability
        renewable_power = self.get_renewable_power(current_time)
        
        # Determine energy sources based on power demand vs renewable availability
        if renewable_power >= effective_power:
            # Fully renewable - all energy from renewable source
            renewable_energy = energy_kwh
            grid_energy = 0.0
        elif renewable_power > 0:
            # Partial renewable - use what's available
            renewable_fraction = renewable_power / effective_power
            renewable_energy = energy_kwh * renewable_fraction
            grid_energy = energy_kwh * (1 - renewable_fraction)
        else:
            # No renewable available
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
        
        logger.info(f"{self.name}: Metrics reset")
    
    def __repr__(self) -> str:
        return (f"EdgeNode(id={self.node_id}, name={self.name}, "
                f"renewable={self.renewable_source}, "
                f"tasks={self.tasks_completed})")
