"""
Network Simulator for EcoChain-ML Framework

Simulates the complete distributed system with edge nodes, scheduler,
blockchain, and workload generation.
"""

import numpy as np
import yaml
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

from .edge_node import EdgeNode
from ..scheduler.energy_aware_scheduler import EnergyAwareScheduler, BaselineScheduler
from ..scheduler.renewable_predictor import RenewablePredictor
from ..blockchain.verification_layer import BlockchainVerifier
from ..monitoring.energy_monitor import EnergyMonitor
from ..inference.model_executor import InferenceTask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkSimulator:
    """
    Main simulator for the EcoChain-ML distributed system.
    
    Orchestrates:
    - Edge nodes with renewable energy
    - Energy-aware scheduler
    - Blockchain verification
    - Workload generation and execution
    - Metrics collection
    """
    
    def __init__(
        self,
        system_config_path: str,
        experiment_config_path: str,
        num_nodes: Optional[int] = None,
        num_tasks: Optional[int] = None,
        arrival_rate: Optional[float] = None
    ):
        """
        Initialize network simulator.
        
        Args:
            system_config_path: Path to system configuration YAML
            experiment_config_path: Path to experiment configuration YAML
            num_nodes: Override number of nodes (for scalability testing)
            num_tasks: Override number of tasks (for scalability testing)
            arrival_rate: Override arrival rate (for scalability testing)
        """
        # Load configurations
        with open(system_config_path, 'r') as f:
            self.system_config = yaml.safe_load(f)
        
        with open(experiment_config_path, 'r') as f:
            self.experiment_config = yaml.safe_load(f)
        
        # Store overrides for scalability testing
        self.num_nodes_override = num_nodes
        self.num_tasks_override = num_tasks
        self.arrival_rate_override = arrival_rate
        
        # Initialize components
        self.nodes: List[EdgeNode] = []
        self.scheduler: Optional[EnergyAwareScheduler] = None
        self.blockchain: Optional[BlockchainVerifier] = None
        self.energy_monitor: Optional[EnergyMonitor] = None
        self.predictor: Optional[RenewablePredictor] = None
        
        # Simulation state
        self.current_time = 0.0  # Simulation time in hours
        self.tasks_generated = []
        self.tasks_completed = []
        
        # Initialize all components
        self._initialize_nodes()
        self._initialize_blockchain()
        self._initialize_energy_monitor()
        self._initialize_predictor()
        self._initialize_scheduler()
        
        logger.info("NetworkSimulator initialized successfully")
    
    def _initialize_nodes(self) -> None:
        """Initialize edge nodes from configuration."""
        logger.info("Initializing edge nodes...")
        
        base_nodes = self.system_config['edge_nodes']
        
        # Determine how many nodes to create
        if self.num_nodes_override is not None:
            target_count = self.num_nodes_override
        else:
            target_count = len(base_nodes)
        
        # Create nodes - cycle through base configs if we need more
        for i in range(target_count):
            base_config = base_nodes[i % len(base_nodes)].copy()
            # Update node ID to be unique
            base_config['id'] = f"node_{i+1}"
            base_config['name'] = f"{base_config.get('name', 'Node').split('#')[0].strip()} #{i+1}"
            
            # Vary renewable capacity for different nodes to create diversity
            if i >= len(base_nodes):
                # For additional nodes, vary the renewable capacity
                capacity_factor = 0.7 + 0.6 * np.random.random()  # 70% to 130%
                base_config['renewable_capacity_watts'] = int(
                    base_config['renewable_capacity_watts'] * capacity_factor
                )
            
            node = EdgeNode(base_config)
            self.nodes.append(node)
        
        logger.info(f"Initialized {len(self.nodes)} edge nodes")
    
    def _initialize_blockchain(self) -> None:
        """Initialize blockchain verification layer."""
        logger.info("Initializing blockchain...")
        
        blockchain_config = self.system_config['blockchain']
        
        # Use node IDs as validators
        validators = [node.node_id for node in self.nodes]
        stake_amounts = {node.node_id: 100.0 for node in self.nodes}
        
        self.blockchain = BlockchainVerifier(
            validators=validators,
            stake_amounts=stake_amounts,
            block_time=blockchain_config['block_time_seconds'],
            carbon_intensity=self.system_config['monitoring']['carbon_intensity_gco2_per_kwh']
        )
        
        logger.info("Blockchain initialized")
    
    def _initialize_energy_monitor(self) -> None:
        """Initialize energy monitoring system."""
        logger.info("Initializing energy monitor...")
        
        monitoring_config = self.system_config['monitoring']
        
        self.energy_monitor = EnergyMonitor(
            carbon_intensity=monitoring_config['carbon_intensity_gco2_per_kwh'],
            sampling_interval=monitoring_config['sampling_interval_seconds']
        )
        
        logger.info("Energy monitor initialized")
    
    def _initialize_predictor(self) -> None:
        """Initialize renewable energy predictor."""
        logger.info("Initializing renewable predictor...")
        
        self.predictor = RenewablePredictor(
            lookback_hours=24,
            prediction_horizon_hours=1,
            device='cpu'
        )
        
        logger.info("Renewable predictor initialized")
    
    def _initialize_scheduler(
        self,
        dvfs_enabled: bool = True,
        renewable_prediction_enabled: bool = True,
        energy_aware_routing: bool = True
    ) -> None:
        """Initialize energy-aware scheduler with configurable components."""
        logger.info("Initializing scheduler...")
        
        scheduler_config = self.system_config['scheduler']
        
        self.scheduler = EnergyAwareScheduler(
            nodes=self.nodes,
            predictor=self.predictor,
            qos_weight=scheduler_config['qos_weight'],
            energy_weight=scheduler_config['energy_weight'],
            renewable_weight=scheduler_config['renewable_weight'],
            dvfs_enabled=dvfs_enabled,
            renewable_prediction_enabled=renewable_prediction_enabled,
            energy_aware_routing=energy_aware_routing
        )
        
        logger.info("Scheduler initialized")
    
    def generate_workload(
        self,
        num_tasks: Optional[int] = None,
        duration_hours: Optional[float] = None
    ) -> List[InferenceTask]:
        """
        Generate synthetic workload for simulation.
        
        Args:
            num_tasks: Number of tasks to generate (uses config if None)
            duration_hours: Duration over which to generate tasks
            
        Returns:
            List of InferenceTask objects
        """
        if num_tasks is None:
            num_tasks = self.num_tasks_override or self.experiment_config['workload']['num_tasks']
        
        if duration_hours is None:
            duration_hours = self.system_config['simulation']['duration_hours']
        
        logger.info(f"Generating {num_tasks} tasks over {duration_hours} hours...")
        
        workload_config = self.experiment_config['workload']
        task_types = workload_config['task_types']
        
        tasks = []
        
        # Generate task arrival times (Poisson process)
        arrival_rate = self.arrival_rate_override or workload_config['arrival_rate_per_hour']
        
        # Calculate how many tasks we need based on arrival rate and duration
        expected_tasks = int(arrival_rate * duration_hours)
        actual_num_tasks = max(num_tasks, expected_tasks)
        
        inter_arrival_times = np.random.exponential(
            scale=1.0/arrival_rate,
            size=actual_num_tasks
        )
        arrival_times = np.cumsum(inter_arrival_times)
        
        # Filter to duration and limit to num_tasks
        arrival_times = arrival_times[arrival_times <= duration_hours]
        if len(arrival_times) > num_tasks:
            arrival_times = arrival_times[:num_tasks]
        
        for i, arrival_time in enumerate(arrival_times):
            # Select task type
            task_type = np.random.choice(
                [t['type'] for t in task_types],
                p=[t['probability'] for t in task_types]
            )
            
            # Get execution time for this task type
            task_config = next(t for t in task_types if t['type'] == task_type)
            exec_time = task_config['avg_execution_time_sec']
            
            # Add variability
            exec_time *= np.random.uniform(0.8, 1.2)
            
            # Select model
            models = self.experiment_config['models']
            if 'image' in task_type:
                model_choices = [m['name'] for m in models if m['type'] == 'image_classification']
            elif 'text' in task_type:
                model_choices = [m['name'] for m in models if m['type'] == 'text_classification']
            else:
                model_choices = [m['name'] for m in models]
            
            model_name = np.random.choice(model_choices) if model_choices else 'default_model'
            
            # Create task
            task = InferenceTask(
                task_id=f"task_{i:04d}",
                model_name=model_name,
                input_data=None,
                priority=np.random.uniform(0.3, 0.9)
            )
            
            # Add custom attributes
            task.arrival_time = arrival_time
            task.execution_time = exec_time
            task.task_type = task_type
            
            tasks.append(task)
        
        self.tasks_generated = tasks
        logger.info(f"Generated {len(tasks)} tasks")
        
        return tasks
    
    def run_simulation(
        self,
        method: str = 'ecochain_ml',
        use_compression: bool = True,
        use_blockchain: bool = True,
        dvfs_enabled: bool = True,
        renewable_prediction_enabled: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete simulation.
        
        Args:
            method: Scheduling method ('standard', 'energy_aware_only', 'blockchain_only', 'ecochain_ml')
            use_compression: Whether to use model compression
            use_blockchain: Whether to use blockchain verification
            dvfs_enabled: Whether to enable DVFS (for ablation)
            renewable_prediction_enabled: Whether to enable renewable prediction (for ablation)
            
        Returns:
            Dictionary with simulation results
        """
        # Determine configuration based on method
        if method == 'standard':
            use_compression = False
            use_blockchain = False
            use_energy_aware = False
            dvfs_enabled = False
            renewable_prediction_enabled = False
        elif method == 'energy_aware_only':
            use_compression = True
            use_blockchain = False
            use_energy_aware = True
        elif method == 'blockchain_only':
            use_compression = False
            use_blockchain = True
            use_energy_aware = False
            dvfs_enabled = False
            renewable_prediction_enabled = False
        else:  # ecochain_ml (full)
            use_compression = True
            use_blockchain = True
            use_energy_aware = True
        
        logger.info(f"Starting simulation with method='{method}'")
        logger.info(f"  Energy-Aware Scheduling: {use_energy_aware}")
        logger.info(f"  Compression: {use_compression}")
        logger.info(f"  Blockchain: {use_blockchain}")
        logger.info(f"  DVFS: {dvfs_enabled}")
        logger.info(f"  Renewable Prediction: {renewable_prediction_enabled}")
        
        # Reset all components
        self._reset_simulation()
        
        # Generate workload if not already generated
        if not self.tasks_generated:
            self.generate_workload()
        
        # Re-initialize scheduler with correct component settings
        if use_energy_aware:
            self._initialize_scheduler(
                dvfs_enabled=dvfs_enabled,
                renewable_prediction_enabled=renewable_prediction_enabled,
                energy_aware_routing=True
            )
            scheduler = self.scheduler
        else:
            scheduler = BaselineScheduler(self.nodes)
        
        # Simulate task execution
        for task in self.tasks_generated:
            self.current_time = task.arrival_time
            
            try:
                # Schedule and execute task
                result = scheduler.schedule_task(
                    task.to_dict(),
                    self.current_time,
                    compressed=use_compression
                )
                
                # Record in blockchain if enabled
                if use_blockchain and self.blockchain:
                    self.blockchain.submit_verification(
                        task_id=result['task_id'],
                        node_id=result['node_id'],
                        result={'output': result},
                        energy_consumed=result['energy_consumed'],
                        renewable_energy=result['renewable_energy'],
                        grid_energy=result['grid_energy']
                    )
                
                self.tasks_completed.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to execute task {task.task_id}: {e}")
        
        # Create blocks for pending transactions
        if use_blockchain and self.blockchain:
            while self.blockchain.pending_transactions:
                self.blockchain.create_block()
        
        # Collect results
        results = self._collect_results(method, use_compression, use_blockchain)
        
        logger.info(f"Simulation complete: {len(self.tasks_completed)} tasks completed")
        
        return results
    
    def run_ablation(
        self,
        config_name: str,
        config_params: Dict[str, bool]
    ) -> Dict[str, Any]:
        """
        Run ablation study with specific component configuration.
        
        Args:
            config_name: Name of the ablation configuration
            config_params: Dictionary with component enable/disable flags:
                - renewable_prediction: bool
                - dvfs: bool
                - model_compression: bool
                - blockchain: bool
                
        Returns:
            Dictionary with simulation results
        """
        logger.info(f"Running ablation: {config_name}")
        logger.info(f"  Config: {config_params}")
        
        # Reset simulation
        self._reset_simulation()
        
        # Generate workload if needed
        if not self.tasks_generated:
            self.generate_workload()
        
        # Extract component settings
        use_renewable_prediction = config_params.get('renewable_prediction', True)
        use_dvfs = config_params.get('dvfs', True)
        use_compression = config_params.get('model_compression', True)
        use_blockchain = config_params.get('blockchain', True)
        
        # Re-initialize scheduler with specific component settings
        self._initialize_scheduler(
            dvfs_enabled=use_dvfs,
            renewable_prediction_enabled=use_renewable_prediction,
            energy_aware_routing=True  # Always use energy-aware routing for ablation
        )
        
        # Run simulation with specific settings
        for task in self.tasks_generated:
            self.current_time = task.arrival_time
            
            try:
                result = self.scheduler.schedule_task(
                    task.to_dict(),
                    self.current_time,
                    compressed=use_compression
                )
                
                if use_blockchain and self.blockchain:
                    self.blockchain.submit_verification(
                        task_id=result['task_id'],
                        node_id=result['node_id'],
                        result={'output': result},
                        energy_consumed=result['energy_consumed'],
                        renewable_energy=result['renewable_energy'],
                        grid_energy=result['grid_energy']
                    )
                
                self.tasks_completed.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to execute task {task.task_id}: {e}")
        
        # Create blocks if blockchain enabled
        if use_blockchain and self.blockchain:
            while self.blockchain.pending_transactions:
                self.blockchain.create_block()
        
        # Collect results
        method_name = config_name
        results = self._collect_results(method_name, use_compression, use_blockchain)
        results['ablation_config'] = config_params
        
        return results
    
    def _collect_results(
        self,
        method: str,
        use_compression: bool,
        use_blockchain: bool
    ) -> Dict[str, Any]:
        """Collect and aggregate simulation results."""
        
        # Aggregate node metrics
        total_energy = 0.0
        total_renewable = 0.0
        total_grid = 0.0
        
        node_stats = []
        for node in self.nodes:
            status = node.get_status()
            node_stats.append(status)
            total_energy += status['total_energy_kwh']
            total_renewable += status['renewable_energy_kwh']
            total_grid += status['grid_energy_kwh']
        
        # Calculate task metrics
        if self.tasks_completed:
            latencies = [t['execution_time'] for t in self.tasks_completed]
            avg_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            max_latency = np.max(latencies)
        else:
            avg_latency = std_latency = max_latency = 0.0
        
        # Renewable percentage
        renewable_pct = (total_renewable / total_energy * 100) if total_energy > 0 else 0
        
        # Carbon emissions
        carbon_intensity = self.system_config['monitoring']['carbon_intensity_gco2_per_kwh']
        total_carbon_gco2 = total_grid * carbon_intensity
        
        # Blockchain overhead and carbon credits
        blockchain_overhead = {}
        carbon_credits_earned = 0.0
        carbon_avoided_gco2 = 0.0
        
        if use_blockchain and self.blockchain:
            blockchain_overhead = self.blockchain.calculate_blockchain_overhead()
            carbon_credits_earned = blockchain_overhead.get('carbon_credits_earned_usd', 0.0)
            carbon_avoided_gco2 = blockchain_overhead.get('carbon_avoided_gco2', 0.0)
        
        # Operational cost (grid electricity only)
        electricity_price = self.system_config['monitoring']['electricity_price_per_kwh']
        operational_cost = total_grid * electricity_price
        
        # Net cost = operational cost - carbon credits earned
        net_cost = operational_cost - carbon_credits_earned
        
        results = {
            'method': method,
            'use_compression': use_compression,
            'use_blockchain': use_blockchain,
            
            # Energy metrics
            'total_energy_kwh': total_energy,
            'renewable_energy_kwh': total_renewable,
            'grid_energy_kwh': total_grid,
            'renewable_percent': renewable_pct,
            
            # Carbon metrics
            'total_carbon_gco2': total_carbon_gco2,
            'total_carbon_kg': total_carbon_gco2 / 1000,
            'carbon_avoided_gco2': carbon_avoided_gco2,
            
            # Performance metrics
            'tasks_completed': len(self.tasks_completed),
            'tasks_generated': len(self.tasks_generated),
            'completion_rate': len(self.tasks_completed) / len(self.tasks_generated) if self.tasks_generated else 0,
            'avg_latency_sec': avg_latency,
            'std_latency_sec': std_latency,
            'max_latency_sec': max_latency,
            
            # Cost metrics
            'operational_cost_usd': operational_cost,
            'carbon_credits_earned_usd': carbon_credits_earned,
            'net_cost_usd': net_cost,
            
            # Blockchain metrics
            'blockchain_overhead': blockchain_overhead,
            
            # Node statistics
            'node_stats': node_stats,
            
            # Scheduler statistics
            'scheduler_stats': self.scheduler.get_statistics() if self.scheduler else {}
        }
        
        return results
    
    def _reset_simulation(self) -> None:
        """Reset simulation state."""
        self.current_time = 0.0
        self.tasks_completed = []
        
        # Reset nodes
        for node in self.nodes:
            node.reset_metrics()
        
        # Reset scheduler
        if self.scheduler:
            self.scheduler.reset()
        
        # Reset blockchain
        if self.blockchain:
            self.blockchain.reset()
        
        logger.info("Simulation state reset")
