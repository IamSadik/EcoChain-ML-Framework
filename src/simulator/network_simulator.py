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
        arrival_rate: Optional[float] = None,
        use_deterministic_renewable: bool = False,
        random_seed: Optional[int] = None
    ):
        """
        Initialize network simulator.
        
        Args:
            system_config_path: Path to system configuration YAML
            experiment_config_path: Path to experiment configuration YAML
            num_nodes: Override number of nodes (for scalability testing)
            num_tasks: Override number of tasks (for scalability testing)
            arrival_rate: Override arrival rate (for scalability testing)
            use_deterministic_renewable: If True, use deterministic renewable traces (Issue 6 fix)
            random_seed: Random seed for reproducibility (for stochastic renewable)
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
        self.use_deterministic_renewable = use_deterministic_renewable
        self.random_seed = random_seed
        
        # Generate deterministic renewable traces if requested
        self.renewable_traces = {}
        if use_deterministic_renewable:
            self.renewable_traces = self._generate_deterministic_traces()
            
        # Set random seed if provided
        # FIX: Move this AFTER trace generation so it doesn't get overridden
        if random_seed is not None:
            np.random.seed(random_seed)
        
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
    
    def _generate_deterministic_traces(self, duration_hours: float = 24.0, 
                                       resolution_minutes: float = 10.0) -> Dict[str, Dict[float, float]]:
        """
        Generate deterministic renewable power traces for all renewable source types.
        
        FIX Issue 6: Pre-computed traces ensure identical renewable conditions across runs.
        
        Args:
            duration_hours: Duration of trace in hours
            resolution_minutes: Time resolution in minutes
            
        Returns:
            Dictionary mapping source type to trace: {source: {time_hours: power_watts}}
        """
        logger.info(f"Generating deterministic renewable traces (duration: {duration_hours}h, "
                   f"resolution: {resolution_minutes}min)...")
        
        # Use fixed seed for deterministic generation
        np.random.seed(42)
        
        # Generate time points
        resolution_hours = resolution_minutes / 60.0
        times = np.arange(0, duration_hours + resolution_hours, resolution_hours)
        
        traces = {}
        
        # Solar trace (sinusoidal with realistic variability)
        solar_trace = {}
        for t in times:
            hour = t % 24
            if 6 <= hour <= 18:
                # Sinusoidal pattern peaking at noon
                solar_factor = np.sin((hour - 6) * np.pi / 12)
                # Fixed variability pattern (not random, but realistic)
                variability = 0.85 + 0.10 * np.sin(t * 2 * np.pi / 3)  # 3-hour cloud cycle
                variability = np.clip(variability, 0.5, 1.0)
                power = solar_factor * variability  # Normalized (0-1)
            else:
                power = 0.0
            solar_trace[t] = power
        traces['solar'] = solar_trace
        
        # Wind trace (Weibull-based with temporal correlation)
        wind_trace = {}
        # Generate correlated wind pattern using AR(1) process
        wind_series = [0.3]  # Start at 30% capacity
        for i in range(1, len(times)):
            # AR(1) with phi=0.9 for strong temporal correlation
            prev = wind_series[-1]
            innovation = np.random.normal(0, 0.05)
            new_val = 0.9 * prev + 0.1 * 0.30 + innovation  # Mean revert to 0.30
            new_val = np.clip(new_val, 0.0, 0.95)
            wind_series.append(new_val)
        
        # Add diurnal pattern (stronger at night)
        for i, t in enumerate(times):
            hour = t % 24
            if 0 <= hour < 6 or 18 <= hour < 24:
                diurnal_factor = 1.10
            else:
                diurnal_factor = 0.90
            wind_trace[t] = wind_series[i] * diurnal_factor
        traces['wind'] = wind_trace
        
        # Hybrid trace (70% solar + 30% wind)
        hybrid_trace = {}
        for t in times:
            solar_component = solar_trace[t] * 0.7
            wind_component = wind_trace[t] * 0.30 / 0.35  # Normalize wind to similar scale
            hybrid_trace[t] = solar_component + wind_component
        traces['hybrid'] = hybrid_trace
        
        # Grid trace (always zero)
        traces['grid'] = {t: 0.0 for t in times}
        
        # FIX: Do not reset random seed to None (which uses clock), as it kills reproducibility.
        # The caller (__init__) will set the correct simulation seed immediately after this returns.
        
        logger.info(f"Generated traces for: {list(traces.keys())}")
        return traces
    
    def _initialize_nodes(self) -> None:
        """Initialize edge nodes from configuration."""
        logger.info("Initializing edge nodes...")
        
        base_nodes = self.system_config['edge_nodes']
        
        # Determine how many nodes to create
        if self.num_nodes_override is not None:
            target_count = self.num_nodes_override
        else:
            target_count = len(base_nodes)
        
        # ============================================================================
        # CRITICAL FIX (Perplexity Issue #1): MAINTAIN 50% RENEWABLE AT ALL SCALES
        # ============================================================================
        # Previous: Random distribution caused renewable % to drop (67% → 49%)
        # New: Deterministic pattern ensures EXACTLY 50% renewable at all scales
        # 
        # Pattern:
        # - Even-numbered nodes (0, 2, 4, ...): Renewable (alternating solar/wind)
        # - Odd-numbered nodes (1, 3, 5, ...): Grid-only
        # 
        # Result:
        # - 4 nodes: 2 renewable (50%)
        # - 8 nodes: 4 renewable (50%)
        # - 16 nodes: 8 renewable (50%)
        # - 32 nodes: 16 renewable (50%)
        # ============================================================================
        
        # For scalability tests, use deterministic renewable assignment
        apply_deterministic = self.num_nodes_override is not None
        
        # Create nodes with FIXED 50% renewable ratio
        for i in range(target_count):
            if i < len(base_nodes) and not apply_deterministic:
                # Use base configs for first few nodes (default behavior for non-scaling tests)
                base_config = base_nodes[i].copy()
                # Ensure name matches renewable source
                source = base_config.get('renewable_source', 'grid')
                if source == 'solar':
                    base_config['name'] = f"Solar Edge Node #{i+1}"
                elif source == 'wind':
                    base_config['name'] = f"Wind Edge Node #{i+1}"
                elif source == 'hybrid':
                    base_config['name'] = f"Hybrid Edge Node #{i+1}"
                else:
                    base_config['name'] = f"Grid-Only Node #{i+1}"
            else:
                # DETERMINISTIC assignment for scalability tests
                # Use a random base config as template for hardware specs
                template_idx = i % len(base_nodes)
                base_config = base_nodes[template_idx].copy()
                
                # CRITICAL: Assign renewable based on node index
                # Even indices (0, 2, 4, ...) = Renewable
                # Odd indices (1, 3, 5, ...) = Grid-only
                if i % 2 == 0:
                    # Renewable node: Alternate between solar and wind
                    if (i // 2) % 2 == 0:
                        # Solar node
                        base_config['renewable_source'] = 'solar'
                        base_config['renewable_capacity_watts'] = 100  # 100W solar
                        base_config['name'] = f"Solar Edge Node #{i+1}"
                        base_config['initial_battery_percent'] = 50.0
                        base_config['battery_capacity_hours'] = 2.5
                    else:
                        # Wind node
                        base_config['renewable_source'] = 'wind'
                        base_config['renewable_capacity_watts'] = 80  # 80W wind
                        base_config['name'] = f"Wind Edge Node #{i+1}"
                        base_config['initial_battery_percent'] = 60.0
                        base_config['battery_capacity_hours'] = 3.0
                else:
                    # Grid-only node
                    base_config['renewable_source'] = 'grid'
                    base_config['renewable_capacity_watts'] = 0
                    base_config['name'] = f"Grid-Only Node #{i+1}"
                    base_config['initial_battery_percent'] = 100.0
                    base_config['battery_capacity_hours'] = 0.0
            
            # Update node ID to be unique
            base_config['id'] = f"node_{i+1}"
            
            # FIX Issue 6: Pass renewable trace to node if deterministic mode is enabled
            renewable_trace = None
            if self.use_deterministic_renewable and base_config['renewable_source'] in self.renewable_traces:
                source_type = base_config['renewable_source']
                # Scale the normalized trace by the node's capacity
                capacity = base_config['renewable_capacity_watts']
                renewable_trace = {
                    time: power * capacity 
                    for time, power in self.renewable_traces[source_type].items()
                }
            
            node = EdgeNode(base_config, renewable_trace=renewable_trace)
            self.nodes.append(node)
        
        # Log node distribution
        renewable_count = sum(1 for n in self.nodes if n.renewable_source != 'grid')
        solar_count = sum(1 for n in self.nodes if n.renewable_source == 'solar')
        wind_count = sum(1 for n in self.nodes if n.renewable_source == 'wind')
        grid_count = len(self.nodes) - renewable_count
        
        logger.info(f"Initialized {len(self.nodes)} edge nodes:")
        logger.info(f"  Renewable: {renewable_count} ({100*renewable_count/len(self.nodes):.1f}%) "
                   f"[{solar_count} solar, {wind_count} wind]")
        logger.info(f"  Grid-only: {grid_count} ({100*grid_count/len(self.nodes):.1f}%)")
    
    def _initialize_blockchain(self) -> None:
        """Initialize blockchain verification layer."""
        logger.info("Initializing blockchain...")
        
        blockchain_config = self.system_config['blockchain']
        monitoring_config = self.system_config['monitoring']
        
        # Use node IDs as validators
        validators = [node.node_id for node in self.nodes]
        stake_amounts = {node.node_id: 100.0 for node in self.nodes}
        
        self.blockchain = BlockchainVerifier(
            validators=validators,
            stake_amounts=stake_amounts,
            block_time=blockchain_config['block_time_seconds'],
            carbon_intensity=monitoring_config['carbon_intensity_gco2_per_kwh'],
            carbon_credit_rate=monitoring_config.get('carbon_credit_rate', 0.00005)  # Use config value
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
        duration_hours: Optional[float] = None,
        workload_pattern: str = 'realistic_bursty'  # NEW: 'poisson', 'realistic_bursty', 'diurnal'
    ) -> List[InferenceTask]:
        """
        Generate synthetic workload for simulation.
        
        Args:
            num_tasks: Number of tasks to generate (uses config if None)
            duration_hours: Duration over which to generate tasks
            workload_pattern: Pattern type ('poisson', 'realistic_bursty', 'diurnal')
            
        Returns:
            List of InferenceTask objects
        """
        if num_tasks is None:
            num_tasks = self.num_tasks_override or self.experiment_config['workload']['num_tasks']
        
        # Get the arrival rate (tasks per hour)
        arrival_rate = self.arrival_rate_override or self.experiment_config['workload']['arrival_rate_per_hour']
        
        # Calculate the expected duration based on arrival rate
        expected_duration = num_tasks / arrival_rate
        
        if duration_hours is None:
            duration_hours = expected_duration * 1.1
        
        logger.info(f"Generating {num_tasks} tasks over {duration_hours:.1f} hours "
                   f"(arrival rate: {arrival_rate} tasks/h, pattern: {workload_pattern})...")
        
        workload_config = self.experiment_config['workload']
        task_types = workload_config['task_types']
        
        tasks = []
        
        # ========================================================================
        # FIX #3: REALISTIC WORKLOAD PATTERNS (Perplexity Fix)
        # ========================================================================
        # Real edge workloads are NOT Poisson! They exhibit:
        # 1. Bursty traffic (video analytics, IoT sensors send data in bursts)
        # 2. Diurnal patterns (more traffic during day, less at night)
        # 3. Idle periods (minutes with zero tasks)
        # 4. Flash crowds (sudden spikes from events)
        #
        # This creates STRUCTURAL variance in task arrival patterns
        # ========================================================================
        
        if workload_pattern == 'realistic_bursty':
            # Generate realistic bursty workload with diurnal patterns
            arrival_times = self._generate_bursty_arrivals(num_tasks, duration_hours, arrival_rate)
        elif workload_pattern == 'diurnal':
            # Pure diurnal pattern (day/night variation)
            arrival_times = self._generate_diurnal_arrivals(num_tasks, duration_hours, arrival_rate)
        else:  # 'poisson' (original)
            # Simple Poisson process (for comparison)
            inter_arrival_times = np.random.exponential(
                scale=1.0/arrival_rate,
                size=num_tasks
            )
            arrival_times = np.cumsum(inter_arrival_times)
        
        # ========================================================================
        # CRITICAL FIX: EXPAND TASK HETEROGENEITY (10-100× RANGE)
        # ========================================================================
        # Previous: 2-4× energy range (too narrow)
        # Real ML workloads: 10-100× energy variance
        # 
        # Real-world ML model energy consumption:
        # - MobileNetV2: ~0.01 Wh (tiny)
        # - ResNet-18: ~0.05 Wh (small)
        # - ResNet-50: ~0.10 Wh (medium - baseline)
        # - ResNet-152: ~0.50 Wh (large)
        # - EfficientNet-B7: ~1.00 Wh (huge)
        # - Range: 100× variance!
        # 
        # This creates realistic scheduling challenge for edge systems
        # ========================================================================
        task_profiles = [
            {
                'weight': 'tiny', 
                'energy_factor': 0.1,      # 10× less energy than baseline
                'time_factor': 0.2,        # 5× faster than baseline
                'probability': 0.15,       # 15% of tasks
                'model_example': 'MobileNetV2'
            },
            {
                'weight': 'small', 
                'energy_factor': 0.5,      # 2× less energy than baseline
                'time_factor': 0.6,        # 40% faster than baseline
                'probability': 0.25,       # 25% of tasks
                'model_example': 'ResNet-18'
            },
            {
                'weight': 'medium', 
                'energy_factor': 1.0,      # Baseline energy
                'time_factor': 1.0,        # Baseline time
                'probability': 0.30,       # 30% of tasks
                'model_example': 'ResNet-50'
            },
            {
                'weight': 'large', 
                'energy_factor': 5.0,      # 5× more energy than baseline
                'time_factor': 3.0,        # 3× slower than baseline
                'probability': 0.20,       # 20% of tasks
                'model_example': 'ResNet-152'
            },
            {
                'weight': 'huge', 
                'energy_factor': 10.0,     # 10× more energy than baseline
                'time_factor': 5.0,        # 5× slower than baseline
                'probability': 0.10,       # 10% of tasks
                'model_example': 'EfficientNet-B7'
            },
        ]
        # Total range: 0.1× to 10.0× = 100× energy variance!
        
        for i, arrival_time in enumerate(arrival_times):
            # FIX: Select HETEROGENEOUS task profile (tiny/small/medium/large/huge)
            profile = np.random.choice(
                task_profiles,
                p=[p['probability'] for p in task_profiles]
            )
            
            # Select task type
            task_type = np.random.choice(
                [t['type'] for t in task_types],
                p=[t['probability'] for t in task_types]
            )
            
            # Get execution time for this task type
            task_config = next(t for t in task_types if t['type'] == task_type)
            exec_time = task_config['avg_execution_time_sec']
            
            # FIX: Apply profile-based time factor + realistic variance (±40%)
            exec_time *= profile['time_factor']
            exec_time *= np.random.uniform(0.6, 1.4)  # INCREASED from ±20% to ±40%
            
            # Select model
            models = self.experiment_config['models']
            if 'image' in task_type:
                model_choices = [m['name'] for m in models if m['type'] == 'image_classification']
            elif 'text' in task_type:
                model_choices = [m['name'] for m in models if m['type'] == 'text_classification']
            else:
                model_choices = [m['name'] for m in models]
            
            model_name = np.random.choice(model_choices) if model_choices else 'default_model'
            
            # FIX: Add VARIABLE model size (50MB-500MB) - major energy impact
            model_size_mb = np.random.uniform(50, 500)
            
            # Create task
            task = InferenceTask(
                task_id=f"task_{i:04d}",
                model_name=model_name,
                input_data=None,
                priority=np.random.uniform(0.3, 0.9)
            )
            
            # FIX: Add custom attributes with HETEROGENEOUS profiles
            task.arrival_time = arrival_time
            task.execution_time = exec_time
            task.task_type = task_type
            task.task_profile = profile['weight']  # tiny/small/medium/large/huge
            task.energy_factor = profile['energy_factor']  # 0.1x to 10.0x energy
            task.model_size_mb = model_size_mb  # Variable model size
            
            tasks.append(task)
        
        self.tasks_generated = tasks
        
        # Calculate actual duration and effective rate
        actual_duration = arrival_times[-1] if len(arrival_times) > 0 else 0
        effective_rate = len(tasks) / actual_duration if actual_duration > 0 else 0
        
        # FIX: Log task profile distribution
        tiny_count = sum(1 for t in tasks if t.task_profile == 'tiny')
        small_count = sum(1 for t in tasks if t.task_profile == 'small')
        medium_count = sum(1 for t in tasks if t.task_profile == 'medium')
        large_count = sum(1 for t in tasks if t.task_profile == 'large')
        huge_count = sum(1 for t in tasks if t.task_profile == 'huge')
        
        logger.info(f"Generated {len(tasks)} tasks (actual duration: {actual_duration:.1f}h, "
                   f"effective rate: {effective_rate:.1f} tasks/h)")
        logger.info(f"  Task profiles: {tiny_count} tiny (15%), {small_count} small (25%), "
                   f"{medium_count} medium (30%), {large_count} large (20%), {huge_count} huge (10%)")
        
        return tasks
    
    def _generate_bursty_arrivals(
        self,
        num_tasks: int,
        duration_hours: float,
        base_rate: float
    ) -> np.ndarray:
        """
        Generate realistic bursty task arrivals with diurnal patterns.
        
        Mimics real edge workloads:
        - Video analytics: Bursts when motion detected
        - IoT sensors: Periodic bursts of sensor data
        - Voice assistants: Bursts during peak hours
        
        Returns:
            Array of arrival times (hours)
        """
        arrival_times = []
        current_time = 0.0
        tasks_generated = 0
        
        # Burst parameters
        burst_probability = 0.15  # 15% chance of burst in each time window
        burst_duration_mean = 0.05  # 3 minutes average burst duration
        burst_rate_multiplier = 5.0  # 5× normal rate during bursts
        
        while tasks_generated < num_tasks and current_time < duration_hours:
            hour_of_day = current_time % 24
            
            # Diurnal pattern: Higher activity during day (8am-10pm)
            if 8 <= hour_of_day < 22:
                diurnal_factor = 1.5  # 50% more traffic during day
            elif 22 <= hour_of_day < 24 or 0 <= hour_of_day < 6:
                diurnal_factor = 0.3  # 70% less traffic at night
            else:
                diurnal_factor = 1.0  # Normal traffic (6-8am)
            
            # Decide if this is a burst period
            if np.random.random() < burst_probability:
                # BURST! Generate tasks at 5× rate for short duration
                burst_duration = np.random.exponential(burst_duration_mean)
                burst_rate = base_rate * burst_rate_multiplier * diurnal_factor
                burst_end = current_time + burst_duration
                
                while current_time < burst_end and tasks_generated < num_tasks:
                    inter_arrival = np.random.exponential(1.0 / burst_rate)
                    current_time += inter_arrival
                    if tasks_generated < num_tasks:
                        arrival_times.append(current_time)
                        tasks_generated += 1
            else:
                # Normal period: Generate tasks at base rate
                current_rate = base_rate * diurnal_factor
                inter_arrival = np.random.exponential(1.0 / current_rate)
                current_time += inter_arrival
                if tasks_generated < num_tasks:
                    arrival_times.append(current_time)
                    tasks_generated += 1
        
        return np.array(arrival_times[:num_tasks])
    
    def _generate_diurnal_arrivals(
        self,
        num_tasks: int,
        duration_hours: float,
        base_rate: float
    ) -> np.ndarray:
        """
        Generate task arrivals with strong diurnal (day/night) pattern.
        
        Peak traffic during business hours (9am-5pm),
        minimal traffic at night (midnight-6am).
        """
        arrival_times = []
        current_time = 0.0
        tasks_generated = 0
        
        while tasks_generated < num_tasks and current_time < duration_hours:
            hour_of_day = current_time % 24
            
            # Strong diurnal pattern using sinusoidal function
            # Peak at 2pm (hour 14), minimum at 2am (hour 2)
            hour_angle = (hour_of_day - 14) * 2 * np.pi / 24
            diurnal_factor = 0.2 + 0.8 * (1 + np.cos(hour_angle)) / 2
            # Range: 0.2× (at 2am) to 1.0× (at 2pm)
            
            current_rate = base_rate * diurnal_factor
            inter_arrival = np.random.exponential(1.0 / current_rate)
            current_time += inter_arrival
            
            if tasks_generated < num_tasks:
                arrival_times.append(current_time)
                tasks_generated += 1
        
        return np.array(arrival_times[:num_tasks])
    
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
            method: Scheduling method ('standard', 'compression_only', 'energy_aware_only', 'blockchain_only', 'ecochain_ml')
            use_compression: Whether to use model compression
            use_blockchain: Whether to use blockchain verification
            dvfs_enabled: Whether to enable DVFS (for ablation)
            renewable_prediction_enabled: Whether to enable renewable prediction (for ablation)
            
        Returns:
            Dictionary with simulation results
        """
        # ========================================================================
        # FIX #2: COMPRESSION-ONLY BASELINE (Addresses Perplexity Novelty Concern)
        # ========================================================================
        # Problem: "Compression dominates (49.9%) - looks like INT8 quantization paper"
        # Solution: Add compression-only baseline to prove EcoChain-ML provides
        #          15-20% ADDITIONAL savings beyond standard INT8 quantization
        # ========================================================================
        
        # Determine configuration based on method
        if method == 'standard':
            use_compression = False
            use_blockchain = False
            use_energy_aware = False
            dvfs_enabled = False
            renewable_prediction_enabled = False
        elif method == 'compression_only':
            # NEW: Compression-only baseline (standard practice)
            use_compression = True
            use_blockchain = False
            use_energy_aware = False
            dvfs_enabled = False
            renewable_prediction_enabled = False
        elif method == 'energy_aware_only':
            use_compression = True
            use_blockchain = False
            use_energy_aware = True
        elif method == 'blockchain_only':
            use_compression = True  # FIXED: Blockchain baseline also uses compression
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
        
        # Track failure statistics
        total_failures = 0
        total_retries = 0
        
        # Simulate task execution
        for task in self.tasks_generated:
            self.current_time = task.arrival_time
            
            try:
                # ============================================================
                # CRITICAL FIX #11: TASK FAILURE RETRY LOGIC
                # ============================================================
                # Real edge systems must handle task failures with retry logic
                # Failures add:
                # - Energy overhead (failed attempt consumes energy)
                # - Latency penalty (retry takes additional time)
                # - Reduced completion rate (if retry also fails)
                # ============================================================
                
                # Schedule and execute task (first attempt)
                result = scheduler.schedule_task(
                    task.to_dict(),
                    self.current_time,
                    compressed=use_compression
                )
                
                # Check if task failed
                if result.get('failed', False):
                    total_failures += 1
                    failure_reason = result.get('failure_reason', 'UNKNOWN')
                    logger.warning(f"Task {result['task_id']} failed on first attempt: {failure_reason}")
                    
                    # RETRY LOGIC: Try once more on a different node
                    # In real systems, retries add latency and energy overhead
                    retry_delay = 0.01  # 36 seconds delay (0.01 hours)
                    self.current_time += retry_delay
                    
                    try:
                        # Retry on a different node
                        result_retry = scheduler.schedule_task(
                            task.to_dict(),
                            self.current_time,
                            compressed=use_compression
                        )
                        
                        if result_retry.get('failed', False):
                            # Retry also failed - task is lost
                            logger.error(f"Task {result_retry['task_id']} FAILED on retry: {result_retry.get('failure_reason')}")
                            total_retries += 1
                            # Record the failed retry
                            result_retry['retry_failed'] = True
                            result_retry['execution_time'] = result['execution_time'] + retry_delay * 3600 + result_retry['execution_time']
                            result_retry['energy_consumed'] = result['energy_consumed'] + result_retry['energy_consumed']
                            self.tasks_completed.append(result_retry)
                        else:
                            # Retry succeeded!
                            logger.info(f"Task {result_retry['task_id']} succeeded on retry")
                            total_retries += 1
                            # Add overhead from failed attempt + retry delay
                            result_retry['retry_succeeded'] = True
                            result_retry['execution_time'] = result['execution_time'] + retry_delay * 3600 + result_retry['execution_time']
                            result_retry['energy_consumed'] = result['energy_consumed'] + result_retry['energy_consumed']
                            result_retry['renewable_energy'] = result['renewable_energy'] + result_retry['renewable_energy']
                            result_retry['grid_energy'] = result['grid_energy'] + result_retry['grid_energy']
                            
                            # Record in blockchain if enabled
                            if use_blockchain and self.blockchain:
                                self.blockchain.submit_verification(
                                    task_id=result_retry['task_id'],
                                    node_id=result_retry['node_id'],
                                    result={'output': result_retry},
                                    energy_consumed=result_retry['energy_consumed'],
                                    renewable_energy=result_retry['renewable_energy'],
                                    grid_energy=result_retry['grid_energy']
                                )
                            
                            self.tasks_completed.append(result_retry)
                    except Exception as e:
                        logger.error(f"Retry failed with exception: {e}")
                        total_retries += 1
                else:
                    # Task succeeded on first attempt
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
        
        # Log failure statistics
        failure_rate = (total_failures / len(self.tasks_generated) * 100) if self.tasks_generated else 0
        retry_success_rate = ((total_failures - sum(1 for t in self.tasks_completed if t.get('retry_failed', False))) / total_failures * 100) if total_failures > 0 else 0
        
        logger.info(f"Task Failure Statistics:")
        logger.info(f"  Total failures: {total_failures} ({failure_rate:.1f}%)")
        logger.info(f"  Retries attempted: {total_retries}")
        logger.info(f"  Retry success rate: {retry_success_rate:.1f}%")
        
        # Create blocks for pending transactions
        if use_blockchain and self.blockchain:
            while self.blockchain.pending_transactions:
                self.blockchain.create_block()
        
        # Collect results
        results = self._collect_results(method, use_compression, use_blockchain)
        
        # Add failure statistics to results
        results['task_failures'] = total_failures
        results['task_failure_rate'] = failure_rate
        results['retries_attempted'] = total_retries
        results['retry_success_rate'] = retry_success_rate
        
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
        
        # Track failure statistics for ablation
        total_failures = 0
        total_retries = 0
        
        # Run simulation with specific settings
        for task in self.tasks_generated:
            self.current_time = task.arrival_time
            
            try:
                result = self.scheduler.schedule_task(
                    task.to_dict(),
                    self.current_time,
                    compressed=use_compression
                )
                
                # ============================================================
                # CRITICAL FIX: TASK FAILURE RETRY LOGIC FOR ABLATION
                # ============================================================
                if result.get('failed', False):
                    total_failures += 1
                    
                    # RETRY LOGIC: Try once more on a different node
                    retry_delay = 0.01  # 36 seconds delay
                    self.current_time += retry_delay
                    
                    try:
                        # Retry on a different node
                        result_retry = self.scheduler.schedule_task(
                            task.to_dict(),
                            self.current_time,
                            compressed=use_compression
                        )
                        
                        if result_retry.get('failed', False):
                            # Retry also failed - log and keep failure result
                            logger.error(f"Task {result_retry['task_id']} FAILED on retry: {result_retry.get('failure_reason')}")
                            total_retries += 1
                            result_retry['retry_failed'] = True
                            # Accumulate costs
                            result_retry['execution_time'] = result['execution_time'] + retry_delay * 3600 + result_retry['execution_time']
                            result_retry['energy_consumed'] = result['energy_consumed'] + result_retry['energy_consumed']
                            
                            self.tasks_completed.append(result_retry)
                        else:
                            # Retry succeeded!
                            logger.info(f"Task {result_retry['task_id']} succeeded on retry after initial failure: {result.get('failure_reason', 'UNKNOWN')}")
                            total_retries += 1
                            result_retry['retry_succeeded'] = True
                            # Accumulate costs from first failed attempt
                            result_retry['execution_time'] = result['execution_time'] + retry_delay * 3600 + result_retry['execution_time']
                            result_retry['energy_consumed'] = result['energy_consumed'] + result_retry['energy_consumed']
                            result_retry['renewable_energy'] = result['renewable_energy'] + result_retry['renewable_energy']
                            result_retry['grid_energy'] = result['grid_energy'] + result_retry['grid_energy']
                            
                            # Blockchain verification for successful retry
                            if use_blockchain and self.blockchain:
                                self.blockchain.submit_verification(
                                    task_id=result_retry['task_id'],
                                    node_id=result_retry['node_id'],
                                    result={'output': result_retry},
                                    energy_consumed=result_retry['energy_consumed'],
                                    renewable_energy=result_retry['renewable_energy'],
                                    grid_energy=result_retry['grid_energy']
                                )
                            
                            self.tasks_completed.append(result_retry)
                    except Exception as e:
                        logger.warning(f"Retry failed for task {task.task_id}: {e}")
                        self.tasks_completed.append(result)
                else:
                    # Success on first try
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
        
        # Add failure stats
        results['task_failures'] = total_failures
        results['retries_attempted'] = total_retries
        
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
            avg_latency = float(np.mean(latencies))
            std_latency = float(np.std(latencies))
            max_latency = float(np.max(latencies))
        else:
            avg_latency = std_latency = max_latency = 0.0
        
        # Renewable percentage
        renewable_pct = float((total_renewable / total_energy * 100) if total_energy > 0 else 0)
        
        # Carbon emissions
        carbon_intensity = self.system_config['monitoring']['carbon_intensity_gco2_per_kwh']
        total_carbon_gco2 = total_grid * carbon_intensity
        
        # Blockchain overhead and carbon credits
        blockchain_overhead = {}
        carbon_credits_earned = 0.0
        carbon_avoided_gco2 = 0.0
        
        if use_blockchain and self.blockchain:
            blockchain_overhead = self.blockchain.calculate_blockchain_overhead()
            carbon_credits_earned = float(blockchain_overhead.get('carbon_credits_earned_usd', 0.0))
            carbon_avoided_gco2 = float(blockchain_overhead.get('carbon_avoided_gco2', 0.0))
        
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
            'total_energy_kwh': float(total_energy),
            'renewable_energy_kwh': float(total_renewable),
            'grid_energy_kwh': float(total_grid),
            'renewable_percent': renewable_pct,
            
            # Carbon metrics
            'total_carbon_gco2': float(total_carbon_gco2),
            'total_carbon_kg': float(total_carbon_gco2 / 1000),
            'carbon_avoided_gco2': float(carbon_avoided_gco2),
            
            # Performance metrics
            'tasks_completed': int(len(self.tasks_completed)),
            'tasks_generated': int(len(self.tasks_generated)),
            'completion_rate': float(len(self.tasks_completed) / len(self.tasks_generated) if self.tasks_generated else 0),
            'avg_latency_sec': avg_latency,
            'std_latency_sec': std_latency,
            'max_latency_sec': max_latency,
            
            # Cost metrics
            'operational_cost_usd': float(operational_cost),
            'carbon_credits_earned_usd': float(carbon_credits_earned),
            'net_cost_usd': float(net_cost),
            
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
