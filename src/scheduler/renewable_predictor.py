import numpy as np
import math

class RenewablePredictor:
    def __init__(self, config):
        """
        Initializes the predictor with system configuration.
        config: The 'renewable_energy' section from system_config.yaml
        """
        self.solar_config = config['solar']
        self.wind_config = config['wind']
        
    def generate_daily_trace(self, duration_seconds=86400):
        """
        Generates a 24-hour trace (second-by-second) for Solar and Wind.
        Returns: Dictionary of numpy arrays
        """
        time_steps = np.arange(duration_seconds)
        
        # --- 1. Generate Solar Trace (Sinusoidal) ---
        # Peak at noon (roughly 43200th second). 
        # We use a clipped sine wave so night time is 0.
        # Formula: max(0, sin(t))
        
        # Shift sine wave so peak is at noon
        period = 86400
        phase_shift = -np.pi / 2 # Starts at 6am (roughly)
        
        # Calculate angle for the whole day (0 to 2pi)
        x = (time_steps / period) * 2 * np.pi 
        
        # Generate raw sine wave centered at noon
        solar_raw = np.sin(x - (np.pi/2) * 2) 
        
        # Apply physics: Night time is 0 
        solar_power = np.maximum(0, solar_raw) * self.solar_config['peak_power']
        
        # Add slight cloud noise (random reduction)
        # Use a slightly less aggressive cloud factor to ensure power > 0 during the day
        cloud_factor = np.random.uniform(0.9, 1.0, size=duration_seconds) # <-- TWEAKED FROM 0.8
        solar_final = solar_power * cloud_factor 
        # Add a minimum non-zero value during the day to prevent zero division/underflow issues
        solar_final[solar_final > 0] += 0.01

        # --- 2. Generate Wind Trace (Stochastic) ---
        # Wind is unpredictable. We use a random walk.
        wind_trace = np.zeros(duration_seconds)
        current_wind = self.wind_config['peak_power'] * 0.5 # Start at 50%
        
        for t in range(1, duration_seconds):
            # Random fluctuation
            change = np.random.uniform(-2, 2) 
            current_wind += change
            # Clip to physics limits (0 to Max Power)
            current_wind = np.clip(current_wind, 0, self.wind_config['peak_power'])
            wind_trace[t] = current_wind

        return {
            "solar": solar_final,
            "wind": wind_trace
        }

    def predict_future_availability(self, current_time, window_size=3600):
        """
        Simulates an LSTM forecast. 
        In a real system, this would run a neural net.
        In simulation, we return the ground truth with slight added 'prediction error'.
        """
        # Placeholder for phase 2 (Simulation Loop)
        pass