"""
Energy-Aware Scheduler with DVFS for EcoChain-ML

This module implements an intelligent scheduler that:
1. Selects optimal edge nodes based on renewable energy availability
2. Applies Dynamic Voltage and Frequency Scaling (DVFS)
3. Balances QoS (latency) with energy efficiency
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging
from .renewable_predictor import RenewablePredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnergyAwareScheduler:
    """
    Intelligent scheduler for energy-efficient ML inference.
    
    The scheduler makes decisions based on:
    - Renewable energy availability (predicted)
    - Node energy efficiency
    - Task latency requirements
    - Load balancing across nodes
    """
    
    def __init__(
        self,
        nodes: List[Any],
        predictor: Optional[RenewablePredictor] = None,
        qos_weight: float = 0.4,
        energy_weight: float = 0.3,
        renewable_weight: float = 0.3,
        dvfs_enabled: bool = True,
        renewable_prediction_enabled: bool = True,
        energy_aware_routing: bool = True
    ):
        """
        Initialize the scheduler.
        
        Args:
            nodes: List of EdgeNode objects
            predictor: RenewablePredictor for forecasting
            qos_weight: Weight for QoS (latency) in scheduling decision
            energy_weight: Weight for energy efficiency
            renewable_weight: Weight for renewable energy usage
            dvfs_enabled: Whether to apply DVFS
            renewable_prediction_enabled: Whether to use renewable prediction
            energy_aware_routing: Whether to route based on energy/renewable status
        """
        self.nodes = nodes
        self.predictor = predictor
        
        # Component enable flags for ablation study
        self.dvfs_enabled = dvfs_enabled
        self.renewable_prediction_enabled = renewable_prediction_enabled
        self.energy_aware_routing = energy_aware_routing
        
        # Scheduling weights (must sum to 1.0)
        total = qos_weight + energy_weight + renewable_weight
        self.qos_weight = qos_weight / total
        self.energy_weight = energy_weight / total
        self.renewable_weight = renewable_weight / total
        
        # Task queue
        self.task_queue = []
        
        # Scheduling history
        self.scheduling_history = []
        
        logger.info(f"Initialized EnergyAwareScheduler with {len(nodes)} nodes")
        logger.info(f"  Weights: QoS={self.qos_weight:.2f}, "
                   f"Energy={self.energy_weight:.2f}, "
                   f"Renewable={self.renewable_weight:.2f}")
        logger.info(f"  DVFS: {'Enabled' if dvfs_enabled else 'Disabled'}")
        logger.info(f"  Renewable Prediction: {'Enabled' if renewable_prediction_enabled else 'Disabled'}")
        logger.info(f"  Energy-Aware Routing: {'Enabled' if energy_aware_routing else 'Disabled'}")
    
    def schedule_task(
        self,
        task: Dict[str, Any],
        current_time: float,
        compressed: bool = False
    ) -> Dict[str, Any]:
        """
        Schedule a task to the optimal node.
        
        Algorithm:
        1. Predict renewable availability for each node (if enabled)
        2. Calculate multi-objective score for each node (if energy-aware)
        3. Select node with highest score
        4. Apply DVFS based on renewable availability (if enabled)
        5. Execute task
        """
        # Filter available nodes
        available_nodes = [n for n in self.nodes if not n.is_busy]
        
        if not available_nodes:
            # All nodes busy - use least loaded node
            available_nodes = self.nodes
        
        # Step 1: Get renewable energy info for each node
        renewable_forecasts = {}
        for node in available_nodes:
            if self.renewable_prediction_enabled and self.predictor and self.predictor.is_trained:
                # Use LSTM predictor for future renewable availability
                renewable_power = node.get_renewable_power(current_time)
                # Add prediction horizon bonus (anticipate future availability)
                future_power = node.get_renewable_power(current_time + 0.5)  # 30 min ahead
                predicted_power = 0.7 * renewable_power + 0.3 * future_power
                avg_power = node.calculate_power_consumption(0.85)
                renewable_pct = min(100, (predicted_power / avg_power * 100) if avg_power > 0 else 0)
            elif self.renewable_prediction_enabled:
                # Use current renewable as prediction
                renewable_power = node.get_renewable_power(current_time)
                avg_power = node.calculate_power_consumption(0.85)
                renewable_pct = min(100, (renewable_power / avg_power * 100) if avg_power > 0 else 0)
            else:
                # No prediction - assume average availability (50%)
                renewable_pct = 50.0
            
            renewable_forecasts[node.node_id] = renewable_pct
        
        # Step 2: Calculate scores and select node
        if self.energy_aware_routing:
            # Full energy-aware scoring
            scores = {}
            for node in available_nodes:
                score = self._calculate_node_score(
                    node,
                    task,
                    renewable_forecasts[node.node_id],
                    current_time,
                    compressed
                )
                scores[node.node_id] = score
            
            best_node_id = max(scores, key=scores.get)
            best_node = next(n for n in available_nodes if n.node_id == best_node_id)
            best_score = scores[best_node_id]
        else:
            # Simple round-robin (no energy awareness)
            best_node = available_nodes[hash(task.get('id', 0)) % len(available_nodes)]
            best_score = 0.5
        
        # Step 3: Apply DVFS
        if self.dvfs_enabled:
            optimal_freq = self._calculate_dvfs_setting(
                best_node,
                task,
                renewable_forecasts[best_node.node_id]
            )
            best_node.set_frequency(optimal_freq)
        else:
            # No DVFS - run at maximum frequency (wastes energy but faster)
            best_node.set_frequency(best_node.max_frequency)
        
        # Step 4: Execute task
        result = best_node.execute_task(task, current_time, compressed)
        
        # Record scheduling decision
        self.scheduling_history.append({
            'task_id': task.get('id', 'unknown'),
            'node_id': best_node.node_id,
            'score': best_score,
            'renewable_forecast': renewable_forecasts[best_node.node_id],
            'frequency': best_node.current_frequency,
            'compressed': compressed,
            'timestamp': current_time
        })
        
        return result
    
    def _calculate_node_score(
        self,
        node: Any,
        task: Dict[str, Any],
        renewable_pct: float,
        current_time: float,
        compressed: bool
    ) -> float:
        """
        Calculate multi-objective score for a node.
        
        Score combines:
        - QoS: Lower latency is better
        - Energy: Lower energy consumption is better
        - Renewable: Higher renewable percentage is better
        
        Args:
            node: EdgeNode object
            task: Task dictionary
            renewable_pct: Predicted renewable percentage (0-100)
            current_time: Current time
            compressed: Whether model is compressed
            
        Returns:
            Normalized score (0-1, higher is better)
        """
        # 1. QoS Score (based on expected latency)
        # Higher frequency = lower latency = better QoS
        freq_ratio = node.current_frequency / node.max_frequency
        qos_score = freq_ratio
        
        # 2. Energy Score (based on expected energy consumption)
        # Lower power = better energy efficiency
        avg_power = node.calculate_power_consumption(0.85)
        # Normalize by max power (lower is better, so invert)
        energy_score = 1.0 - (avg_power / node.max_power)
        
        # 3. Renewable Score (based on forecast)
        # Higher renewable percentage = better
        renewable_score = renewable_pct / 100.0
        
        # 4. Load balancing bonus
        # Prefer less loaded nodes
        load_factor = node.tasks_completed / (max(n.tasks_completed for n in self.nodes) + 1)
        load_bonus = 1.0 - load_factor
        
        # Combine scores with weights
        total_score = (
            self.qos_weight * qos_score +
            self.energy_weight * energy_score +
            self.renewable_weight * renewable_score +
            0.1 * load_bonus  # Small load balancing bonus
        )
        
        return total_score
    
    def _calculate_dvfs_setting(
        self,
        node: Any,
        task: Dict[str, Any],
        renewable_pct: float
    ) -> float:
        """
        Calculate optimal CPU/GPU frequency using DVFS.
        
        Strategy:
        - High renewable availability → use higher frequency (faster, more power)
        - Low renewable availability → use lower frequency (slower, less power)
        - Balance with task urgency/priority
        
        Args:
            node: EdgeNode object
            task: Task dictionary
            renewable_pct: Predicted renewable percentage (0-100)
            
        Returns:
            Optimal frequency in GHz
        """
        # Base frequency selection on renewable availability
        renewable_factor = renewable_pct / 100.0
        
        # Task priority/urgency (if specified)
        priority = task.get('priority', 0.5)  # 0-1, default medium
        
        # Combine factors
        # More renewable → higher frequency
        # Higher priority → higher frequency
        combined_factor = 0.7 * renewable_factor + 0.3 * priority
        
        # Calculate target frequency
        freq_range = node.max_frequency - node.min_frequency
        target_freq = node.min_frequency + freq_range * combined_factor
        
        # Apply some hysteresis to avoid frequent switching
        current_freq = node.current_frequency
        if abs(target_freq - current_freq) < 0.2:  # 200 MHz threshold
            target_freq = current_freq
        
        logger.debug(f"DVFS: renewable={renewable_pct:.1f}%, "
                    f"priority={priority:.2f}, "
                    f"target_freq={target_freq:.2f} GHz")
        
        return target_freq
    
    def schedule_batch(
        self,
        tasks: List[Dict[str, Any]],
        current_time: float,
        compressed: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Schedule a batch of tasks.
        
        Args:
            tasks: List of task dictionaries
            current_time: Current simulation time
            compressed: Whether to use compressed models
            
        Returns:
            List of execution results
        """
        results = []
        
        for task in tasks:
            try:
                result = self.schedule_task(task, current_time, compressed)
                results.append(result)
            except RuntimeError as e:
                logger.warning(f"Failed to schedule task {task['id']}: {e}")
                # Queue for later
                self.task_queue.append(task)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get scheduling statistics.
        
        Returns:
            Dictionary with scheduling metrics
        """
        if not self.scheduling_history:
            return {}
        
        # Aggregate node utilization
        node_usage = {}
        for record in self.scheduling_history:
            node_id = record['node_id']
            node_usage[node_id] = node_usage.get(node_id, 0) + 1
        
        # Average renewable forecast
        avg_renewable = np.mean([r['renewable_forecast'] for r in self.scheduling_history])
        
        # Average frequency
        avg_frequency = np.mean([r['frequency'] for r in self.scheduling_history])
        
        return {
            'total_tasks_scheduled': len(self.scheduling_history),
            'node_utilization': node_usage,
            'avg_renewable_forecast': avg_renewable,
            'avg_frequency': avg_frequency,
            'compression_rate': sum(1 for r in self.scheduling_history if r['compressed']) / len(self.scheduling_history)
        }
    
    def reset(self) -> None:
        """Reset scheduler state."""
        self.task_queue = []
        self.scheduling_history = []
        logger.info("Scheduler reset")


class BaselineScheduler:
    """
    Baseline scheduler without energy awareness (for comparison).
    
    Simply round-robin schedules tasks to available nodes.
    """
    
    def __init__(self, nodes: List[Any]):
        """Initialize baseline scheduler."""
        self.nodes = nodes
        self.current_node_idx = 0
        self.scheduling_history = []
        
        logger.info(f"Initialized BaselineScheduler with {len(nodes)} nodes")
    
    def schedule_task(
        self,
        task: Dict[str, Any],
        current_time: float,
        compressed: bool = False
    ) -> Dict[str, Any]:
        """
        Schedule task using simple round-robin.
        
        Args:
            task: Task dictionary
            current_time: Current time
            compressed: Whether to use compression
            
        Returns:
            Execution result
        """
        # Find next available node (round-robin)
        attempts = 0
        while attempts < len(self.nodes):
            node = self.nodes[self.current_node_idx]
            self.current_node_idx = (self.current_node_idx + 1) % len(self.nodes)
            
            if not node.is_busy:
                # Set to max frequency (no DVFS)
                node.set_frequency(node.max_frequency)
                
                # Execute
                result = node.execute_task(task, current_time, compressed)
                
                self.scheduling_history.append({
                    'task_id': task['id'],
                    'node_id': node.node_id,
                    'timestamp': current_time
                })
                
                return result
            
            attempts += 1
        
        raise RuntimeError("No available nodes!")
    
    def schedule_batch(
        self,
        tasks: List[Dict[str, Any]],
        current_time: float,
        compressed: bool = False
    ) -> List[Dict[str, Any]]:
        """Schedule batch of tasks."""
        results = []
        for task in tasks:
            try:
                result = self.schedule_task(task, current_time, compressed)
                results.append(result)
            except RuntimeError:
                pass
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduling statistics."""
        if not self.scheduling_history:
            return {}
        
        node_usage = {}
        for record in self.scheduling_history:
            node_id = record['node_id']
            node_usage[node_id] = node_usage.get(node_id, 0) + 1
        
        return {
            'total_tasks_scheduled': len(self.scheduling_history),
            'node_utilization': node_usage
        }
    
    def reset(self) -> None:
        """Reset scheduler."""
        self.current_node_idx = 0
        self.scheduling_history = []
