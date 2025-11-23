# simulator/energy_profiles.py
import random

def generate_energy_profiles(num_nodes=3, renewable_ratio=0.7):
    """
    Generate initial energy availability for edge nodes.
    renewable_ratio: percentage of energy from renewable sources.
    """
    profiles = {}
    for i in range(num_nodes):
        base_energy = random.uniform(5.0, 10.0)  # kWh
        renewable_energy = base_energy * renewable_ratio
        profiles[f"EdgeNode-{i+1}"] = renewable_energy
    return profiles
