"""
Energy Monitoring System for EcoChain-ML

Tracks energy consumption, renewable usage, and carbon emissions.
"""

import time
import psutil
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EnergyMeasurement:
    """Single energy measurement record."""
    timestamp: float
    node_id: str
    power_watts: float
    energy_kwh: float
    renewable_power_watts: float
    renewable_energy_kwh: float
    grid_energy_kwh: float
    cpu_percent: float
    memory_percent: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EnergyMonitor:
    """
    Monitors and tracks energy consumption for edge nodes.
    
    Provides:
    - Real-time power monitoring
    - Energy accumulation
    - Renewable vs grid energy tracking
    - Carbon emissions calculation
    - System resource monitoring (CPU, memory)
    """
    
    def __init__(
        self,
        carbon_intensity: float = 400.0,
        sampling_interval: float = 1.0
    ):
        """
        Initialize energy monitor.
        
        Args:
            carbon_intensity: Grid carbon intensity in gCO2/kWh
            sampling_interval: Measurement sampling interval in seconds
        """
        self.carbon_intensity = carbon_intensity
        self.sampling_interval = sampling_interval
        
        # Measurement history
        self.measurements: List[EnergyMeasurement] = []
        
        # Cumulative metrics
        self.total_energy_kwh = 0.0
        self.total_renewable_kwh = 0.0
        self.total_grid_kwh = 0.0
        self.total_carbon_gco2 = 0.0
        
        # Monitoring state
        self.is_monitoring = False
        self.last_measurement_time = None
        
        logger.info(f"Initialized EnergyMonitor")
        logger.info(f"  Carbon intensity: {carbon_intensity} gCO2/kWh")
        logger.info(f"  Sampling interval: {sampling_interval}s")
    
    def measure(
        self,
        node_id: str,
        power_watts: float,
        renewable_power_watts: float,
        duration_seconds: Optional[float] = None
    ) -> EnergyMeasurement:
        """
        Record an energy measurement.
        
        Args:
            node_id: ID of the node being measured
            power_watts: Total power consumption in Watts
            renewable_power_watts: Renewable power available in Watts
            duration_seconds: Duration of measurement (uses sampling_interval if None)
            
        Returns:
            EnergyMeasurement object
        """
        if duration_seconds is None:
            duration_seconds = self.sampling_interval
        
        # Calculate energy consumed during this period
        energy_kwh = (power_watts * duration_seconds) / (1000 * 3600)
        
        # Determine renewable vs grid energy
        if renewable_power_watts >= power_watts:
            # Fully powered by renewables
            renewable_energy_kwh = energy_kwh
            grid_energy_kwh = 0.0
        else:
            # Partial renewable
            renewable_fraction = renewable_power_watts / power_watts if power_watts > 0 else 0
            renewable_energy_kwh = energy_kwh * renewable_fraction
            grid_energy_kwh = energy_kwh * (1 - renewable_fraction)
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_percent = psutil.virtual_memory().percent
        
        # Create measurement
        measurement = EnergyMeasurement(
            timestamp=time.time(),
            node_id=node_id,
            power_watts=power_watts,
            energy_kwh=energy_kwh,
            renewable_power_watts=renewable_power_watts,
            renewable_energy_kwh=renewable_energy_kwh,
            grid_energy_kwh=grid_energy_kwh,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent
        )
        
        # Update cumulative metrics
        self.total_energy_kwh += energy_kwh
        self.total_renewable_kwh += renewable_energy_kwh
        self.total_grid_kwh += grid_energy_kwh
        self.total_carbon_gco2 += grid_energy_kwh * self.carbon_intensity
        
        # Store measurement
        self.measurements.append(measurement)
        self.last_measurement_time = measurement.timestamp
        
        return measurement
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current cumulative metrics.
        
        Returns:
            Dictionary with energy and carbon metrics
        """
        renewable_percent = (
            (self.total_renewable_kwh / self.total_energy_kwh * 100)
            if self.total_energy_kwh > 0 else 0
        )
        
        return {
            'total_energy_kwh': self.total_energy_kwh,
            'renewable_energy_kwh': self.total_renewable_kwh,
            'grid_energy_kwh': self.total_grid_kwh,
            'renewable_percent': renewable_percent,
            'total_carbon_gco2': self.total_carbon_gco2,
            'total_carbon_kg': self.total_carbon_gco2 / 1000,
            'num_measurements': len(self.measurements)
        }
    
    def get_time_series(
        self,
        metric: str = 'power',
        node_id: Optional[str] = None
    ) -> Dict[str, List]:
        """
        Get time series data for a specific metric.
        
        Args:
            metric: Metric to retrieve ('power', 'energy', 'renewable', 'carbon')
            node_id: Optional node ID to filter by
            
        Returns:
            Dictionary with 'timestamps' and 'values' lists
        """
        # Filter measurements
        filtered = self.measurements
        if node_id:
            filtered = [m for m in filtered if m.node_id == node_id]
        
        timestamps = [m.timestamp for m in filtered]
        
        # Extract values based on metric
        if metric == 'power':
            values = [m.power_watts for m in filtered]
        elif metric == 'energy':
            values = [m.energy_kwh * 1000 for m in filtered]  # Convert to Wh
        elif metric == 'renewable':
            values = [m.renewable_power_watts for m in filtered]
        elif metric == 'renewable_percent':
            values = [
                (m.renewable_energy_kwh / m.energy_kwh * 100) 
                if m.energy_kwh > 0 else 0
                for m in filtered
            ]
        elif metric == 'carbon':
            values = [m.grid_energy_kwh * self.carbon_intensity for m in filtered]
        elif metric == 'cpu':
            values = [m.cpu_percent for m in filtered]
        elif metric == 'memory':
            values = [m.memory_percent for m in filtered]
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return {
            'timestamps': timestamps,
            'values': values
        }
    
    def calculate_statistics(
        self,
        window_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Calculate statistical summary of measurements.
        
        Args:
            window_size: Number of recent measurements to analyze (None = all)
            
        Returns:
            Dictionary with statistical metrics
        """
        if not self.measurements:
            return {}
        
        # Select measurements
        measurements = self.measurements[-window_size:] if window_size else self.measurements
        
        # Extract values
        powers = [m.power_watts for m in measurements]
        renewable_pcts = [
            (m.renewable_energy_kwh / m.energy_kwh * 100) 
            if m.energy_kwh > 0 else 0
            for m in measurements
        ]
        
        return {
            'avg_power_watts': np.mean(powers),
            'max_power_watts': np.max(powers),
            'min_power_watts': np.min(powers),
            'std_power_watts': np.std(powers),
            'avg_renewable_percent': np.mean(renewable_pcts),
            'max_renewable_percent': np.max(renewable_pcts),
            'min_renewable_percent': np.min(renewable_pcts),
            'num_measurements': len(measurements)
        }
    
    def generate_report(self) -> str:
        """
        Generate a human-readable energy report.
        
        Returns:
            Formatted report string
        """
        metrics = self.get_current_metrics()
        stats = self.calculate_statistics()
        
        report = f"""
╔════════════════════════════════════════════════════════╗
║           EcoChain-ML Energy Report                    ║
╚════════════════════════════════════════════════════════╝

Energy Consumption:
  Total Energy:        {metrics['total_energy_kwh']:.4f} kWh
  Renewable Energy:    {metrics['renewable_energy_kwh']:.4f} kWh ({metrics['renewable_percent']:.1f}%)
  Grid Energy:         {metrics['grid_energy_kwh']:.4f} kWh

Carbon Emissions:
  Total Carbon:        {metrics['total_carbon_kg']:.3f} kg CO2
  Carbon Intensity:    {self.carbon_intensity} gCO2/kWh

Power Statistics:
  Average Power:       {stats.get('avg_power_watts', 0):.2f} W
  Peak Power:          {stats.get('max_power_watts', 0):.2f} W
  Min Power:           {stats.get('min_power_watts', 0):.2f} W

Renewable Usage:
  Average:             {stats.get('avg_renewable_percent', 0):.1f}%
  Peak:                {stats.get('max_renewable_percent', 0):.1f}%

Measurements:          {metrics['num_measurements']}
"""
        return report
    
    def export_to_csv(self, filename: str) -> None:
        """
        Export measurements to CSV file.
        
        Args:
            filename: Output CSV filename
        """
        import pandas as pd
        
        if not self.measurements:
            logger.warning("No measurements to export")
            return
        
        # Convert to DataFrame
        data = [m.to_dict() for m in self.measurements]
        df = pd.DataFrame(data)
        
        # Save to CSV
        df.to_csv(filename, index=False)
        
        logger.info(f"Exported {len(self.measurements)} measurements to {filename}")
    
    def reset(self) -> None:
        """Reset all measurements and metrics."""
        self.measurements = []
        self.total_energy_kwh = 0.0
        self.total_renewable_kwh = 0.0
        self.total_grid_kwh = 0.0
        self.total_carbon_gco2 = 0.0
        self.last_measurement_time = None
        
        logger.info("Energy monitor reset")


class CarbonCalculator:
    """
    Utility for carbon emissions calculations.
    
    Provides regional carbon intensity data and conversion utilities.
    """
    
    # Carbon intensity by region (gCO2/kWh)
    CARBON_INTENSITY = {
        'US_average': 400,
        'US_california': 200,
        'EU_average': 300,
        'EU_france': 50,  # High nuclear
        'EU_germany': 350,
        'China': 600,
        'India': 700,
        'renewable': 50,  # Solar/wind lifecycle emissions
        'coal': 900,
        'natural_gas': 400,
        'nuclear': 12,
        'hydro': 24,
        'wind': 11,
        'solar': 45
    }
    
    @staticmethod
    def calculate_carbon(
        energy_kwh: float,
        region: str = 'US_average'
    ) -> float:
        """
        Calculate carbon emissions for given energy consumption.
        
        Args:
            energy_kwh: Energy consumption in kWh
            region: Region or energy source
            
        Returns:
            Carbon emissions in gCO2
        """
        intensity = CarbonCalculator.CARBON_INTENSITY.get(region, 400)
        return energy_kwh * intensity
    
    @staticmethod
    def calculate_savings(
        baseline_energy_kwh: float,
        optimized_energy_kwh: float,
        region: str = 'US_average'
    ) -> Dict[str, float]:
        """
        Calculate energy and carbon savings.
        
        Args:
            baseline_energy_kwh: Baseline energy consumption
            optimized_energy_kwh: Optimized energy consumption
            region: Region for carbon intensity
            
        Returns:
            Dictionary with savings metrics
        """
        energy_saved = baseline_energy_kwh - optimized_energy_kwh
        energy_reduction_pct = (energy_saved / baseline_energy_kwh * 100) if baseline_energy_kwh > 0 else 0
        
        baseline_carbon = CarbonCalculator.calculate_carbon(baseline_energy_kwh, region)
        optimized_carbon = CarbonCalculator.calculate_carbon(optimized_energy_kwh, region)
        carbon_saved = baseline_carbon - optimized_carbon
        carbon_reduction_pct = (carbon_saved / baseline_carbon * 100) if baseline_carbon > 0 else 0
        
        return {
            'energy_saved_kwh': energy_saved,
            'energy_reduction_percent': energy_reduction_pct,
            'carbon_saved_gco2': carbon_saved,
            'carbon_saved_kg': carbon_saved / 1000,
            'carbon_reduction_percent': carbon_reduction_pct
        }
    
    @staticmethod
    def get_intensity(region: str) -> float:
        """Get carbon intensity for a region."""
        return CarbonCalculator.CARBON_INTENSITY.get(region, 400)
