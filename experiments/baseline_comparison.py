"""
experiments/baseline_comparison.py
"""

import simpy
from src.scheduler import EnergyAwareScheduler
from src.simulator import EdgeNode
from src.blockchain import BlockchainVerifier

def run_baseline_experiment():
    """
    Compare 4 approaches:
    1. Standard (no optimization)
    2. Energy-aware only
    3. Blockchain verification only
    4. EcoChain-ML (integrated)
    """
    
    # Setup
    env = simpy.Environment()
    nodes = [EdgeNode(env, i) for i in range(5)]
    
    results = {
        'standard': run_standard(env, nodes),
        'energy_aware': run_energy_aware(env, nodes),
        'blockchain_only': run_blockchain_only(env, nodes),
        'ecochain_ml': run_ecochain_ml(env, nodes)
    }
    
    # Analyze results
    compare_metrics(results)
    generate_plots(results)
    
    return results

def run_ecochain_ml(env, nodes):
    """Run full EcoChain-ML framework"""
    
    scheduler = EnergyAwareScheduler(nodes, renewable_predictor)
    verifier = BlockchainVerifier()
    
    # Generate workload
    tasks = generate_ml_workload(num_tasks=1000)
    
    metrics = {
        'total_energy': 0,
        'renewable_pct': 0,
        'carbon_emissions': 0,
        'avg_latency': 0,
        'verification_overhead': 0
    }
    
    for task in tasks:
        # Schedule task
        node, result = scheduler.schedule_inference(task)
        
        # Verify on blockchain
        tx_hash = verifier.verify_inference_result(
            task.id, result['output'], result['energy'], node.id
        )
        
        # Collect metrics
        metrics['total_energy'] += result['energy']
        metrics['renewable_pct'] += result['renewable_used']
        metrics['carbon_emissions'] += calculate_carbon(result['energy'])
        metrics['avg_latency'] += result['latency']
        metrics['verification_overhead'] += verifier.calculate_blockchain_overhead()
    
    # Normalize
    metrics['renewable_pct'] /= len(tasks)
    metrics['avg_latency'] /= len(tasks)
    
    return metrics