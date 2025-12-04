import numpy as np

# We'll need the model compressor logic, so let's import it (assuming it's in the same folder structure)
from src.inference.quantization import ModelCompressor # <-- NEW IMPORT

class EcoScheduler:
    def __init__(self, system_config, edge_nodes):
        self.config = system_config
        self.nodes = edge_nodes
        self.alpha_energy = 0.7 
        self.beta_latency = 0.3
        self.compressor = ModelCompressor(system_config) # <-- NEW COMPONENT

    def get_power_at_freq(self, node_power_profile, frequency_mhz, max_freq_mhz):
        """
        Simulates power consumption based on DVFS. 
        Power often scales with the cube of frequency (P=f^3), but we use 
        a simplified linear interpolation for a simulation prototype.
        """
        idle_power = node_power_profile['idle']
        max_active_power = node_power_profile['max']
        
        # Calculate Power based on frequency
        power_ratio = (frequency_mhz - min(self.config['edge_nodes']['cpu_frequencies'])) / \
                      (max_freq_mhz - min(self.config['edge_nodes']['cpu_frequencies']))
        
        # Interpolate between base and max power
        active_power = idle_power + (max_active_power - idle_power) * power_ratio
        
        return active_power

    def select_best_node(self, task, current_renewable):
        """
        Iteratively finds the optimal (Node, Frequency, Quantization) configuration.
        """
        best_config = {'score': float('inf'), 'node': None, 'frequency': None, 'quant_bits': None}
        
        # Get quantization options (including 32-bit for baseline)
        quantization_options = self.config['quantization']['bits'] + [32]
        model_name = task['model_name'] # Task needs to carry the model name

        for node in self.nodes:
            max_freq_mhz = max(node.cpu_freqs)

            # 1. Iterate through all possible CPU Frequencies (DVFS)
            for freq in sorted(node.cpu_freqs, reverse=True): 
                
                # 2. Iterate through all possible Quantization Levels (Compression)
                for bits in quantization_options:
                    
                    # --- A. Calculate Latency & Accuracy ---
                    exec_params = self.compressor.simulate_execution(
                        task_ops=task['ops'], 
                        frequency_mhz=freq, 
                        model_name=model_name, 
                        bits=bits
                    )
                    estimated_latency = exec_params['latency']
                    
                    # Check QoS Constraint
                    if estimated_latency * 1000 > self.config['scheduler']['qos_constraint']['max_latency']:
                        continue # This configuration is too slow, try next one

                    # --- B. Calculate Energy Cost ---
                    
                    # Power used at this frequency (Watts)
                    estimated_power = self.get_power_at_freq(node.power_profile, freq, max_freq_mhz)
                    total_energy_needed = estimated_power * estimated_latency # Joules

                    # Renewable check
                    available_power_renewable = current_renewable.get(node.id, 0.0)
                    max_renewable_covered_joules = available_power_renewable * estimated_latency
                    
                    covered_by_green = min(total_energy_needed, max_renewable_covered_joules)
                    grid_energy_cost = total_energy_needed - covered_by_green
                    
                    # --- C. Calculate Final Score (Optimization Goal) ---
                    norm_energy = grid_energy_cost 
                    norm_latency = estimated_latency

                    # Total Cost Score
                    score = (self.alpha_energy * norm_energy) + (self.beta_latency * norm_latency)
                    
                    # Save the best configuration found so far
                    if score < best_config['score']:
                        best_config.update({
                            'score': score, 
                            'node': node, 
                            'frequency': freq, 
                            'quant_bits': bits,
                            'latency': estimated_latency,
                            'power': estimated_power,
                            'accuracy': exec_params['accuracy']
                        })

        return best_config