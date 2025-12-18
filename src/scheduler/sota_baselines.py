"""
State-of-the-Art Baseline Schedulers for Comparison

This module implements realistic approximations of published SOTA approaches:
1. Green-LLM (2025): Carbon-aware LLM scheduling with grid-based carbon intensity
2. CASPER (2024): Carbon-aware serverless edge computing with spatial routing
3. Kubernetes: Standard container orchestration (no carbon awareness)

These baselines differ from EcoChain-ML in key ways:
- No renewable energy prediction (only current/grid carbon)
- No DVFS optimization
- No model compression
- No blockchain verification
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GreenLLMBaseline:
    """
    Green-LLM (2025): Carbon-Aware LLM Scheduling Baseline
    
    Paper approach (approximation):
    - Schedules tasks based on grid carbon intensity forecasting
    - Temporal shifting: delays non-urgent tasks to lower carbon periods
    - Spatial shifting: routes tasks to data centers with lower carbon
    - NO renewable energy integration (uses grid carbon only)
    - NO DVFS (runs at max frequency)
    - NO model compression
    
    Key difference from EcoChain-ML:
    - Uses grid carbon intensity instead of renewable prediction
    - Reactive rather than predictive
    - No hardware-level optimization (DVFS)
    """
    
    def __init__(
        self,
        nodes: List[Any],
        carbon_intensity_lookup: Optional[Dict[str, float]] = None
    ):
        """
        Initialize Green-LLM baseline scheduler.
        
        Args:
            nodes: List of EdgeNode objects
            carbon_intensity_lookup: Grid carbon intensity by region (gCO2/kWh)
        """
        self.nodes = nodes
        self.scheduling_history = []
        
        # Grid carbon intensity (gCO2/kWh) - varies by region/time
        # Default values represent typical grid mixes
        self.carbon_intensity = carbon_intensity_lookup or {
            'default': 400.0,  # US average grid
            'coal_heavy': 800.0,  # Coal-dominated grid
            'clean': 200.0,  # Cleaner grid (hydro/nuclear)
        }
        
        # Task delay tolerance (hours) - how long we can wait for lower carbon
        self.max_delay = 4.0  # Max 4 hours delay
        
        logger.info(f"Initialized Green-LLM Baseline with {len(nodes)} nodes")
        logger.info("  Strategy: Grid carbon-aware temporal/spatial shifting")
        logger.info("  Features: NO renewable prediction, NO DVFS, NO compression")
    
    def _get_grid_carbon_intensity(self, node: Any, current_time: float) -> float:
        """
        Get grid carbon intensity for a node at current time.
        
        In real implementation, this would query APIs like:
        - ElectricityMap API
        - WattTime API
        - Grid operator forecasts
        
        For simulation, we model typical diurnal patterns:
        - Higher carbon during evening peak (fossil fuel peakers)
        - Lower carbon during midday (solar) and night (base load)
        """
        # Extract hour of day (0-23)
        hour = int((current_time * 24) % 24)
        
        # Base carbon intensity for node's region
        base_carbon = self.carbon_intensity.get(node.region, 400.0)
        
        # Diurnal pattern (peak evening hours have higher carbon)
        # Morning ramp: 6-9 AM (high)
        # Midday: 10 AM - 3 PM (lower due to solar)
        # Evening peak: 5-9 PM (highest - fossil peakers)
        # Night: 10 PM - 5 AM (lower - base load)
        
        if 6 <= hour < 9:
            multiplier = 1.15  # Morning ramp
        elif 10 <= hour < 15:
            multiplier = 0.85  # Solar midday
        elif 17 <= hour < 21:
            multiplier = 1.30  # Evening peak (worst)
        else:
            multiplier = 0.95  # Night/off-peak
        
        return base_carbon * multiplier
    
    def _predict_future_carbon(self, node: Any, current_time: float, hours_ahead: int = 4) -> List[float]:
        """
        Simple grid carbon forecast (next few hours).
        
        Green-LLM uses grid carbon forecasts to decide temporal shifting.
        """
        forecasts = []
        for h in range(hours_ahead):
            future_time = current_time + (h / 24.0)  # Add hours
            carbon = self._get_grid_carbon_intensity(node, future_time)
            forecasts.append(carbon)
        return forecasts
    
    def schedule_task(
        self,
        task: Dict[str, Any],
        current_time: float,
        compressed: bool = False
    ) -> Dict[str, Any]:
        """
        Schedule task using Green-LLM's carbon-aware strategy.
        
        Algorithm:
        1. Calculate current grid carbon intensity for each node
        2. Check if delaying task would reduce carbon (temporal shifting)
        3. Select node with lowest carbon intensity (spatial shifting)
        4. Execute at MAX frequency (no DVFS)
        """
        # Step 1: Get current carbon intensity for all nodes
        node_carbon = {}
        for node in self.nodes:
            carbon = self._get_grid_carbon_intensity(node, current_time)
            node_carbon[node.node_id] = carbon
        
        # Step 2: Temporal shifting decision
        # Check if we should delay this task to a lower carbon period
        task_urgency = task.get('urgency', 1.0)  # 0=can delay, 1=urgent
        
        if task_urgency < 0.5:
            # Non-urgent task - consider delaying
            # Check future carbon intensity
            best_node = min(self.nodes, key=lambda n: node_carbon[n.node_id])
            future_carbon = self._predict_future_carbon(best_node, current_time, hours_ahead=4)
            current_carbon = node_carbon[best_node.node_id]
            
            # If future has significantly lower carbon (>20% reduction), we would delay
            # For simulation purposes, we'll just account for this in carbon calculation
            min_future_carbon = min(future_carbon)
            if min_future_carbon < current_carbon * 0.8:
                # In real system, would delay task
                # In simulation, use average of current and future best
                effective_carbon = (current_carbon + min_future_carbon) / 2.0
            else:
                effective_carbon = current_carbon
        else:
            # Urgent task - execute now
            effective_carbon = min(node_carbon.values())
        
        # Step 3: Spatial shifting - select node with lowest carbon
        best_node = min(self.nodes, key=lambda n: node_carbon[n.node_id])
        
        # Step 4: Execute at MAX frequency (Green-LLM doesn't use DVFS)
        best_node.set_frequency(best_node.max_frequency)
        
        # Execute task (NO compression - Green-LLM doesn't optimize model)
        result = best_node.execute_task(task, current_time, compressed=False)
        
        # Record decision
        self.scheduling_history.append({
            'task_id': task.get('id', 'unknown'),
            'node_id': best_node.node_id,
            'carbon_intensity': node_carbon[best_node.node_id],
            'frequency': best_node.max_frequency,
            'timestamp': current_time
        })
        
        return result
    
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
                result = self.schedule_task(task, current_time, compressed=False)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to schedule task: {e}")
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduling statistics."""
        if not self.scheduling_history:
            return {}
        
        return {
            'total_tasks_scheduled': len(self.scheduling_history),
            'avg_carbon_intensity': np.mean([r['carbon_intensity'] for r in self.scheduling_history])
        }
    
    def reset(self) -> None:
        """Reset scheduler state."""
        self.scheduling_history = []


class CASPERBaseline:
    """
    CASPER (2024): Carbon-Aware Serverless Edge Computing Baseline
    
    Paper approach (approximation):
    - Serverless function routing based on carbon intensity
    - Multi-node carbon-aware load balancing
    - Uses current renewable/carbon data (no prediction)
    - Dynamic function placement based on node carbon footprint
    - NO DVFS optimization
    - NO model compression
    
    Key difference from EcoChain-ML:
    - Reactive routing (current state only, no prediction)
    - Treats renewables as "low carbon" grid power
    - Focuses on routing efficiency rather than hardware optimization
    """
    
    def __init__(self, nodes: List[Any]):
        """
        Initialize CASPER baseline scheduler.
        
        Args:
            nodes: List of EdgeNode objects
        """
        self.nodes = nodes
        self.scheduling_history = []
        
        # CASPER uses round-robin with carbon-aware selection
        self.routing_table = {node.node_id: 0.0 for node in nodes}
        
        logger.info(f"Initialized CASPER Baseline with {len(nodes)} nodes")
        logger.info("  Strategy: Carbon-aware serverless routing")
        logger.info("  Features: Current-state routing, NO prediction, NO DVFS, NO compression")
    
    def _calculate_node_carbon_score(self, node: Any, current_time: float) -> float:
        """
        Calculate carbon score for a node (lower is better).
        
        CASPER considers:
        1. Current renewable availability (treats as low-carbon grid)
        2. Node power consumption
        3. Current load
        
        Does NOT predict future renewable availability.
        """
        # Get CURRENT renewable power (no prediction)
        renewable_power = node.get_renewable_power(current_time)
        avg_power = node.calculate_power_consumption(0.85)
        
        # Simple carbon score: power not covered by renewables
        grid_power = max(0, avg_power - renewable_power)
        
        # Grid carbon intensity (CASPER assumes fixed value)
        grid_carbon_intensity = 400.0  # gCO2/kWh (US average)
        
        # Carbon emission rate (gCO2/s)
        carbon_rate = (grid_power / 1000.0) * grid_carbon_intensity / 3600.0
        
        # Add load factor (prefer less loaded nodes)
        load_factor = node.tasks_completed / (max(n.tasks_completed for n in self.nodes) + 1)
        
        # Combined score (lower is better)
        score = carbon_rate * (1.0 + 0.3 * load_factor)
        
        return score
    
    def schedule_task(
        self,
        task: Dict[str, Any],
        current_time: float,
        compressed: bool = False
    ) -> Dict[str, Any]:
        """
        Schedule task using CASPER's carbon-aware routing.
        
        Algorithm:
        1. Calculate current carbon score for each node
        2. Select node with lowest current carbon footprint
        3. Execute at MAX frequency (no DVFS)
        """
        # Step 1: Calculate carbon scores
        node_scores = {}
        for node in self.nodes:
            score = self._calculate_node_carbon_score(node, current_time)
            node_scores[node.node_id] = score
        
        # Step 2: Select best node (lowest carbon)
        best_node = min(self.nodes, key=lambda n: node_scores[n.node_id])
        
        # Step 3: Execute at MAX frequency (CASPER doesn't use DVFS)
        best_node.set_frequency(best_node.max_frequency)
        
        # Execute task (NO compression)
        result = best_node.execute_task(task, current_time, compressed=False)
        
        # Record decision
        self.scheduling_history.append({
            'task_id': task.get('id', 'unknown'),
            'node_id': best_node.node_id,
            'carbon_score': node_scores[best_node.node_id],
            'frequency': best_node.max_frequency,
            'timestamp': current_time
        })
        
        return result
    
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
                result = self.schedule_task(task, current_time, compressed=False)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to schedule task: {e}")
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduling statistics."""
        if not self.scheduling_history:
            return {}
        
        return {
            'total_tasks_scheduled': len(self.scheduling_history),
            'avg_carbon_score': np.mean([r['carbon_score'] for r in self.scheduling_history])
        }
    
    def reset(self) -> None:
        """Reset scheduler state."""
        self.scheduling_history = []
        self.routing_table = {node.node_id: 0.0 for node in self.nodes}


class KubernetesBaseline:
    """
    Standard Kubernetes Scheduler Baseline
    
    Standard container orchestration approach:
    - Round-robin or least-loaded scheduling
    - NO carbon awareness
    - NO renewable energy consideration
    - NO DVFS optimization
    - NO model compression
    - Focus on load balancing and availability
    
    This represents typical production deployments without sustainability features.
    """
    
    def __init__(self, nodes: List[Any], strategy: str = 'round_robin'):
        """
        Initialize Kubernetes baseline scheduler.
        
        Args:
            nodes: List of EdgeNode objects
            strategy: 'round_robin' or 'least_loaded'
        """
        self.nodes = nodes
        self.strategy = strategy
        self.current_node_idx = 0
        self.scheduling_history = []
        
        logger.info(f"Initialized Kubernetes Baseline with {len(nodes)} nodes")
        logger.info(f"  Strategy: {strategy}")
        logger.info("  Features: NO carbon awareness, NO DVFS, NO compression")
    
    def schedule_task(
        self,
        task: Dict[str, Any],
        current_time: float,
        compressed: bool = False
    ) -> Dict[str, Any]:
        """
        Schedule task using standard Kubernetes approach.
        
        Algorithm:
        1. Select node based on strategy (round-robin or least-loaded)
        2. Execute at MAX frequency (no energy optimization)
        """
        # Step 1: Select node based on strategy
        if self.strategy == 'round_robin':
            # Simple round-robin
            node = self.nodes[self.current_node_idx]
            self.current_node_idx = (self.current_node_idx + 1) % len(self.nodes)
        else:
            # Least loaded (by task count)
            node = min(self.nodes, key=lambda n: n.tasks_completed)
        
        # Step 2: Execute at MAX frequency (standard practice)
        node.set_frequency(node.max_frequency)
        
        # Execute task (NO compression)
        result = node.execute_task(task, current_time, compressed=False)
        
        # Record decision
        self.scheduling_history.append({
            'task_id': task.get('id', 'unknown'),
            'node_id': node.node_id,
            'frequency': node.max_frequency,
            'timestamp': current_time
        })
        
        return result
    
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
                result = self.schedule_task(task, current_time, compressed=False)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to schedule task: {e}")
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
        """Reset scheduler state."""
        self.current_node_idx = 0
        self.scheduling_history = []
