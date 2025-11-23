from simulator.env import EcoChainEnv
from simulator.scheduler import Scheduler
from simulator.energy_profiles import generate_energy_profiles

# Setup
env = EcoChainEnv(num_nodes=3)
energy_profiles = generate_energy_profiles()
scheduler = Scheduler(env, energy_profiles)

# Add and schedule tasks
tasks = ["task1", "task2", "task3"]
for t in tasks:
    scheduler.schedule_task(t)

# Run simulation
env.run(until=10)

# View task log
task_log = scheduler.get_task_log()
print(task_log)
