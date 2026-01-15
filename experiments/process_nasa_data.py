import pandas as pd
import numpy as np
import json
import os

def load_and_clean_data(filepath):
    """
    Loads the NASA POWER CSV data, skipping the variable header.
    """
    # Read the file to find the start of the data
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    header_end_idx = 0
    for i, line in enumerate(lines):
        if "-END HEADER-" in line:
            header_end_idx = i + 1
            break
            
    # Load data skipping the header metadata
    df = pd.read_csv(filepath, skiprows=header_end_idx)
    
    # Rename columns for consistency
    df = df.rename(columns={
        'YEAR': 'year', 
        'MO': 'month', 
        'DY': 'day', 
        'HR': 'hour'
    })

    # Create datetime index
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    
    return df

def calculate_solar_power(df, array_area=10.0, panel_efficiency=0.18):
    """
    Calculates Solar PV output based on irradiance and temperature.
    Model: P_pv = eta * A * G * (1 - beta * (T_cell - T_ref))
    """
    # Constants
    BETA = 0.004  # Temperature coefficient (1/C)
    T_REF = 25.0  # Reference temperature (C)
    NOCT = 45.0   # Nominal Operating Cell Temperature (C)
    
    # Inputs
    G_irr = df['ALLSKY_SFC_SW_DWN']  # Global Horizontal Irradiance (Wh/m^2 which is approx W/m^2 hourly average)
    T_amb = df['T2M']                # Ambient Temperature (C)
    
    # Calculate Cell Temperature
    # T_cell = T_amb + (NOCT - 20)/800 * G
    t_cell = T_amb + ((NOCT - 20.0) / 800.0) * G_irr
    
    # Calculate Power
    # Ensure no negative generation
    temp_factor = 1 - BETA * (t_cell - T_REF)
    power_raw = panel_efficiency * array_area * G_irr * temp_factor
    
    return power_raw.clip(lower=0.0)

def calculate_wind_power(df, rotor_area=10.0, hub_height=20.0):
    """
    Calculates Wind Turbine output based on wind speed and air density.
    Model: P_wind = 0.5 * rho * A * v^3 * Cp
    """
    # Constants
    R_SPECIFIC = 287.058  # J/(kg*K) for dry air
    CP = 0.4              # Power coefficient (efficiency)
    ALPHA = 0.143         # Shear exponent for height correction
    REF_HEIGHT = 10.0     # Reference height for wind speed data (m)
    CUT_IN_SPEED = 3.0    # m/s
    CUT_OUT_SPEED = 25.0  # m/s
    RATED_SPEED = 12.0    # m/s
    
    # Inputs
    v_ref = df['WS10M']  # Wind speed at 10m (m/s)
    P_kpa = df['PS']     # Surface Pressure (kPa)
    T_C = df['T2M']      # Temperature (C)
    
    # 1. Calculate Air Density (rho)
    # P must be in Pa, T in Kelvin
    P_pa = P_kpa * 1000.0
    T_K = T_C + 273.15
    rho = P_pa / (R_SPECIFIC * T_K)
    
    # 2. Correct Wind Speed to Hub Height
    v_hub = v_ref * (hub_height / REF_HEIGHT) ** ALPHA
    
    # 3. Calculate Power
    # P = 0.5 * rho * A * v^3 * Cp
    p_wind_theoretical = 0.5 * rho * rotor_area * (v_hub ** 3) * CP
    
    # Apply cut-in/cut-out logic
    # This vectorizes the condition application
    conditions = [
        (v_hub < CUT_IN_SPEED) | (v_hub > CUT_OUT_SPEED),
        (v_hub >= CUT_IN_SPEED) & (v_hub < RATED_SPEED),
        (v_hub >= RATED_SPEED)
    ]
    
    # Calculate rated power for clamping
    # We approximate rated power using standard conditions (rho=1.225) at rated speed
    rated_power = 0.5 * 1.225 * rotor_area * (RATED_SPEED ** 3) * CP
    
    choices = [
        0.0,
        p_wind_theoretical,
        rated_power
    ]
    
    p_wind_final = np.select(conditions, choices)
    
    return p_wind_final

def main():
    input_file = os.path.join("data", "POWER_Point_Hourly_20241201_20251231_043d80N_079d64W_LST.csv")
    output_dir = os.path.join("data", "nrel")
    output_csv_file = os.path.join(output_dir, "nrel_realistic_data.csv")
    output_stats_file = os.path.join(output_dir, "nrel_data_stats.json")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return

    print(f"Processing {input_file}...")
    df = load_and_clean_data(input_file)
    
    # Calculate Generation
    print("Calculating Solar Power...")
    df['solar_power_watts'] = calculate_solar_power(df)
    
    print("Calculating Wind Power...")
    df['wind_power_watts'] = calculate_wind_power(df)
    
    # Create normalized columns (0-1) for simplistic observation
    df['solar_normalized'] = df['solar_power_watts'] / df['solar_power_watts'].max()
    df['wind_normalized'] = df['wind_power_watts'] / df['wind_power_watts'].max()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed data
    # We keep relevant original columns + new calculation columns
    # USER REQUEST: Keep ALL raw columns to enable rich feature prediction
    # expanded_cols includes all useful raw data + derived names
    final_cols = [
        'datetime', 'year', 'month', 'day', 'hour',
        'solar_power_watts', 'wind_power_watts',
        # Solar related
        'ALLSKY_SFC_SW_DWN', 'ALLSKY_SFC_SW_DIFF', 'ALLSKY_SFC_UV_INDEX', 
        'ALLSKY_SRF_ALB', 'ALLSKY_SFC_PAR_TOT',
        # Wind/Atmosphere related
        'T2M', 'QV2M', 'RH2M', 'PS', 
        'WS10M', 'WD10M', 'WD50M', 'WS50M'
    ]
    
    print(f"Saving processing data to {output_csv_file}...")
    # exist_cols ensures we only save columns that actually exist in the dataframe
    exist_cols = [c for c in final_cols if c in df.columns]
    df[exist_cols].to_csv(output_csv_file, index=False)
    
    # Calculate statistics
    stats = {
        "solar_mean": float(df['solar_power_watts'].mean()),
        "solar_max": float(df['solar_power_watts'].max()),
        "wind_mean": float(df['wind_power_watts'].mean()),
        "wind_max": float(df['wind_power_watts'].max()),
        "data_points": len(df),
        "date_range": {
            "start": str(df['datetime'].min()),
            "end": str(df['datetime'].max())
        }
    }
    
    print(f"Saving statistics to {output_stats_file}...")
    with open(output_stats_file, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print("Done.")

if __name__ == "__main__":
    main()

