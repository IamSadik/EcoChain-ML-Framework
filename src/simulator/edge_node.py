"""
src/simulator/edge_node.py
"""

import simpy
import random

class EdgeNode:
    def __init__(self, env, node_id, cpu_cores=4, gpu_available=False):
        self.env = env
        self.node_id = node_id
        self.cpu_cores = cpu_cores
        self.gpu_available = gpu_available
        
        # Energy parameters
        self.base_power = 50  # Watts (idle)
        self.max_power = 200  # Watts (full load)
        self.current_frequency = 2.4  # GHz
        self.min_frequency = 1.2
        self.max_frequency = 3.6
        
        # Renewable energy
        self.renewable_source = 'solar'  # or 'wind', 'grid'
        self.renewable_capacity = 150  # Watts
        
        # State
        self.is_busy = False
        self.tasks_completed = 0
        self.total_energy_consumed = 0  # kWh
        self.renewable_energy_used = 0  # kWh
        
    def execute_inference(self, task, model):
        """
        Simulate ML inference execution
        """
        self.is_busy = True
        
        # Calculate execution time based on model complexity and frequency
        base_time = task.complexity / (self.current_frequency / 2.4)
        execution_time = base_time * random.uniform(0.9, 1.1)  # Add noise
        
        # Simulate execution delay
        yield self.env.timeout(execution_time)
        
        # Calculate energy consumed
        avg_power = self.base_power + (self.max_power - self.base_power) * \
                    (self.current_frequency / self.max_frequency)
        
        energy_kwh = (avg_power * execution_time) / (1000 * 3600)  # Convert to kWh
        
        # Check renewable availability
        renewable_available = self.get_renewable_power(self.env.now)
        if renewable_available >= avg_power:
            self.renewable_energy_used += energy_kwh
        else:
            renewable_fraction = renewable_available / avg_power
            self.renewable_energy_used += energy_kwh * renewable_fraction
        
        self.total_energy_consumed += energy_kwh
        self.tasks_completed += 1
        self.is_busy = False
        
        return {
            'result': model.inference(task.input_data),
            'energy_consumed': energy_kwh,
            'execution_time': execution_time,
            'renewable_used': renewable_available >= avg_power
        }
    
    def get_renewable_power(self, timestamp):
        """
        Simulate renewable energy availability
        For solar: peak at noon, zero at night
        """
        if self.renewable_source == 'solar':
            hour = (timestamp % 24)
            if 6 <= hour <= 18:
                # Sinusoidal pattern for solar
                solar_factor = np.sin((hour - 6) * np.pi / 12)
                return self.renewable_capacity * solar_factor
            else:
                return 0
        elif self.renewable_source == 'wind':
            # Random wind pattern
            return self.renewable_capacity * random.uniform(0.3, 0.9)
        else:
            return 0  # Grid only
    
    def set_frequency(self, frequency):
        """Apply DVFS"""
        self.current_frequency = max(self.min_frequency, 
                                     min(frequency, self.max_frequency))