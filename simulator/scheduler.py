# simulator/scheduler.py
import random

class Scheduler:
    """
    Energy-aware ML inference scheduler.
    Assigns tasks to edge nodes based on simulated energy availability.
    """
    def __init__(self, env, energy_profiles):
        self.env = env
        self.energy_profiles = energy_profiles  # dict: node -> energy value
        self.task_log = []

    def schedule_task(self, task):
        """
        Assign task to the node with highest available energy.
        """
        node = max(self.energy_profiles, key=self.energy_profiles.get)
        energy = self.energy_profiles[node]
        self.task_log.append({
            "task": task,
            "node": node,
            "energy_available": energy
        })
        print(f"Scheduled task '{task}' on {node} (Energy: {energy:.2f})")
        # Optionally decrease energy to simulate usage
        self.energy_profiles[node] -= random.uniform(0.1, 0.5)

    def get_task_log(self):
        return self.task_log
