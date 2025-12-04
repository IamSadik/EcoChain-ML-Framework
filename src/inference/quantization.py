import numpy as np

class ModelCompressor:
    """
    Simulates the effect of quantization on latency, energy, and accuracy.
    """
    def __init__(self, config):
        self.quant_config = config['quantization']
        self.model_configs = config['ml_models']
        
    def get_quant_params(self, model_name, bits):
        """
        Retrieves the quantization effects for a given model and bit level.
        In this simulation, we use the general config for all models.
        """
        # Find the index of the chosen bit level
        try:
            bit_index = self.quant_config['bits'].index(bits)
        except ValueError:
            # If bits are not defined, assume baseline (no change)
            return {'speedup': 1.0, 'accuracy_loss': 0.0}

        speedup = self.quant_config['speedup'][bit_index]
        accuracy_loss = self.quant_config['accuracy_loss'][bit_index]
        
        # Find the baseline accuracy of the model
        baseline_accuracy = next(m['accuracy'] for m in self.model_configs if m['name'] == model_name)
        
        return {
            'speedup': speedup,
            'accuracy_loss': accuracy_loss,
            'final_accuracy': baseline_accuracy * (1 - accuracy_loss)
        }

    def simulate_execution(self, task_ops, frequency_mhz, model_name, bits=32):
        """
        Calculates execution metrics for a task given a frequency and compression level.
        bits=32 means no quantization (baseline float precision).
        """
        params = self.get_quant_params(model_name, bits)
        
        # Latency calculation: Time = Ops / (Frequency * Efficiency * Speedup)
        ops_per_second = frequency_mhz * 0.5 * 1e6 # Base ops
        
        # Quantization speeds up execution
        final_ops_per_second = ops_per_second * params['speedup']
        
        latency = task_ops / final_ops_per_second
        
        # Energy calculation is handled by the scheduler's DVFS logic, 
        # but the latency is key here.
        
        return {
            'latency': latency,
            'accuracy': params['final_accuracy'] if bits != 32 else next(m['accuracy'] for m in self.model_configs if m['name'] == model_name)
        }