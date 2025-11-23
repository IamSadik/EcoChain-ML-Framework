# simulator/env.py
import simpy

class EcoChainEnv:
    """
    Simulation environment for EcoChain-ML.
    Tracks time, ML tasks, edge nodes, and energy usage.
    """
    def __init__(self, num_nodes=3):
        self.env = simpy.Environment()
        self.num_nodes = num_nodes
        self.nodes = [f"EdgeNode-{i+1}" for i in range(num_nodes)]
        self.task_queue = []

    def add_task(self, task):
        """Add ML inference task to the queue."""
        self.task_queue.append(task)

    def run(self, until=100):
        """Run the simulation for a given time."""
        print(f"Starting simulation for {until} units of time with {self.num_nodes} nodes...")
        self.env.run(until=until)
        print("Simulation finished.")
