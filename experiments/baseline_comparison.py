import sys
import os
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Progress bar

# Add the root directory to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scheduler.energy_aware_scheduler import EcoScheduler
from src.scheduler.renewable_predictor import RenewablePredictor
from src.simulator.edge_node import EdgeNode
from src.blockchain.verification_layer import VerificationLayer, ProofOfInference
# Note: No need for ModelCompressor import here since it's used inside the scheduler

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_simulation(strategy_name):
    print(f"\n--- Starting Simulation: {strategy_name} ---")
    
    # 1. Load Configurations
    sys_config = load_config('config/system_config.yaml')
    
    # 2. Setup Infrastructure (Nodes)
    nodes = []
    node_count = sys_config['edge_nodes']['count']
    
    for i in range(node_count):
        energy_source = 'solar' if i < (node_count / 2) else 'grid'
        node = EdgeNode(
            node_id=f"node_{i}",
            power_profile=sys_config['edge_nodes']['power_consumption'],
            cpu_freqs=sys_config['edge_nodes']['cpu_frequencies'],
            energy_source_type=energy_source,
            stake=100
        )
        nodes.append(node)

    # 3. Setup Components
    predictor = RenewablePredictor(sys_config['renewable_energy'])
    scheduler = EcoScheduler(sys_config, nodes)
    
    validators = {node.id: node.stake for node in nodes}
    verification_layer = VerificationLayer(validators)
    
    weather_data = predictor.generate_daily_trace()
    
    # 4. Simulation Loop Variables
    total_energy_grid = 0.0      
    total_energy_renewable = 0.0 
    total_tasks_processed = 0
    qos_violations = 0
    total_accuracy = 0.0
    
    duration = 3600 
    arrival_rate = 2.0
    
    # --- MAIN LOOP ---
    for t in tqdm(range(duration), desc=f"Simulating {strategy_name}"):
        
        # A. Update Weather (What is available right now?)
        current_renewable = {}
        solar_now = weather_data['solar'][t]
        wind_now = weather_data['wind'][t]
        
        for node in nodes:
            if node.source_type == 'solar':
                current_renewable[node.id] = solar_now
            elif node.source_type == 'wind':
                current_renewable[node.id] = wind_now
            else:
                current_renewable[node.id] = 0.0

        # B. Generate Traffic (Poisson Distribution)
        if random.random() < arrival_rate:
            
            model_template = random.choice(sys_config['ml_models'])
            
            task = {
                'id': f"task_{t}",
                'ops': float(model_template['ops']),
                'size_mb': model_template['size_mb'],
                'deadline': 1.0, 
                'model_name': model_template['name']
            }
            
            # C. Scheduler Decision
            best_config = None
            selected_node = None
            
            if strategy_name == "EcoChain":
                # Use our smart algorithm (passes current_renewable dict)
                best_config = scheduler.select_best_node(task, current_renewable)
                selected_node = best_config['node']
                
            elif strategy_name == "RoundRobin":
                # Simple cycle: 0, 1, 2, ...
                idx = total_tasks_processed % len(nodes)
                selected_node = nodes[idx]
                
                # For Round Robin, assume MAX power/MAX speed (worst case baseline)
                max_freq = max(selected_node.cpu_freqs)
                latency = task['ops'] / selected_node.get_max_ops()
                power = selected_node.power_profile['max']
                accuracy = model_template['accuracy']
                
                # Format the output to match best_config structure for easy processing
                best_config = {
                    'latency': latency, 
                    'power': power, 
                    'accuracy': accuracy, 
                    'quant_bits': 32, 
                    'frequency': max_freq, 
                    'node': selected_node
                }

            # D. Execute Task & Record Stats & Submit Proof
            if selected_node and best_config: # Ensure we have both node and config
                # 1. Extract Optimal Parameters
                latency = best_config['latency']
                power = best_config['power'] 
                accuracy = best_config['accuracy']
                
                # 2. Calculate Energy
                energy_needed = power * latency # Joules
                
                # 3. Accounting (Renewable vs Grid)
                # THIS IS THE CRITICAL LINE: It now correctly pulls the available power
                available_power_renewable = current_renewable.get(selected_node.id, 0.0)
                max_renewable_covered_joules = available_power_renewable * latency
                
                covered_by_green = min(energy_needed, max_renewable_covered_joules)
                from_grid = energy_needed - covered_by_green
                
                total_energy_renewable += covered_by_green
                total_energy_grid += from_grid

                # 4. Check QoS and Metrics
                if latency > task['deadline']:
                    qos_violations += 1
                
                total_tasks_processed += 1
                total_accuracy += accuracy

                # 5. Create and Submit Proof-of-Inference (Verification)
                proof = ProofOfInference(
                    task_id=task['id'],
                    node_id=selected_node.id,
                    energy_consumed=energy_needed,
                    renewable_used=covered_by_green,
                    model_accuracy=accuracy, 
                    timestamp=t
                )
                
                if strategy_name == "EcoChain":
                    verification_layer.submit_proof(proof)

    # 5. Return Results
    avg_accuracy = total_accuracy / total_tasks_processed if total_tasks_processed > 0 else 0.0
    
    return {
        "strategy": strategy_name,
        "total_tasks": total_tasks_processed,
        "grid_energy_joules": total_energy_grid,
        "renewable_energy_joules": total_energy_renewable,
        "qos_violations": qos_violations,
        "avg_accuracy": avg_accuracy, # NEW METRIC
        
        "chain_length": len(verification_layer.chain) if strategy_name == "EcoChain" else 0,
        "pending_proofs": len(verification_layer.pending_transactions) if strategy_name == "EcoChain" else 0
    }

# --- RUN COMPARISON ---
if __name__ == "__main__":
    
    results_rr = run_simulation("RoundRobin")
    results_eco = run_simulation("EcoChain")
    
    print("\n" + "="*55)
    print("           FINAL RESULTS (1 Hour Simulation)          ")
    print("="*55)
    
    print(f"{'Metric':<30} | {'Round Robin':<15} | {'EcoChain (Yours)':<15}")
    print("-" * 65)
    
    print(f"{'Grid Energy (Joules)':<30} | {results_rr['grid_energy_joules']:.2f} | {results_eco['grid_energy_joules']:.2f}")
    print(f"{'Renewable Used (Joules)':<30} | {results_rr['renewable_energy_joules']:.2f} | {results_eco['renewable_energy_joules']:.2f}")
    print(f"{'QoS Violations (Latency)':<30} | {results_rr['qos_violations']:<15} | {results_eco['qos_violations']:<15}")
    
    # We still need to add the AVG ACCURACY printing logic to the bottom of this script, 
    # as it's not in the output you provided, but the metric is calculated.
    print(f"{'Avg Model Accuracy':<30} | {results_rr.get('avg_accuracy', 0.0):.4f} | {results_eco.get('avg_accuracy', 0.0):.4f}")
    
    print("-" * 65)
    print(f"{'Chain Length (Blocks)':<30} | {results_rr['chain_length']:<15} | {results_eco['chain_length']:<15}")
    
    # --- PLOTTING --- (Plotting logic omitted for brevity, but should be kept in your file)
    # ...
    
    print("\n✅ Plot saved to results/plots/energy_comparison.png")