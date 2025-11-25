"""
src/scheduler/energy_aware_scheduler.py
"""

class EnergyAwareScheduler:
    def __init__(self, nodes, renewable_predictor):
        self.nodes = nodes  # List of edge computing nodes
        self.predictor = renewable_predictor
        self.task_queue = []
        
    def schedule_inference(self, task):
        """
        Schedule ML inference task to optimal node
        
        Algorithm:
        1. Predict renewable availability for each node
        2. Calculate energy cost for each node
        3. Estimate task completion time
        4. Select node maximizing: renewable_usage / (energy_cost * latency)
        5. Apply DVFS if needed
        """
        
        # Step 1: Get renewable predictions
        renewable_forecast = {}
        for node in self.nodes:
            renewable_forecast[node.id] = self.predictor.predict(
                node_id=node.id,
                horizon=task.estimated_duration
            )
        
        # Step 2: Calculate scores
        scores = {}
        for node in self.nodes:
            if node.is_available():
                renewable_pct = renewable_forecast[node.id]
                energy_cost = self.estimate_energy_cost(node, task)
                latency = self.estimate_latency(node, task)
                
                # Multi-objective score
                scores[node.id] = (
                    0.4 * renewable_pct +           # 40% weight on renewables
                    0.3 * (1 / energy_cost) +       # 30% weight on energy
                    0.3 * (1 / latency)             # 30% weight on latency
                )
        
        # Step 3: Select best node
        best_node_id = max(scores, key=scores.get)
        best_node = self.get_node(best_node_id)
        
        # Step 4: Apply DVFS
        optimal_freq = self.calculate_dvfs_setting(
            best_node, 
            task, 
            renewable_forecast[best_node_id]
        )
        best_node.set_frequency(optimal_freq)
        
        # Step 5: Dispatch task
        return self.dispatch(best_node, task)
    
    def calculate_dvfs_setting(self, node, task, renewable_pct):
        """
        Dynamic Voltage and Frequency Scaling
        Higher renewable % → allow higher frequency
        """
        base_freq = node.min_frequency
        max_freq = node.max_frequency
        
        # Linear scaling based on renewable availability
        target_freq = base_freq + (max_freq - base_freq) * (renewable_pct / 100)
        
        return target_freq