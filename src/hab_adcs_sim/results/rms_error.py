import numpy as np
import json
from pathlib import Path

results = Path(r"C:\Users\abouh\OneDrive\Documents\McGill\Thesis\Simulation\src\hab_adcs_sim\results\simulation_results.json")
def compute_rms_error(data: Path) -> float:

    with open(data, 'r') as f:
        data = json.load(f)
    
    errors = np.array(data['yaw']) - np.arctan2(np.array(data['y']), np.array(data['x']))
    rms_error = np.sqrt(np.mean(errors**2))
    return np.rad2deg(rms_error)

print(f"RMS Yaw Error: {compute_rms_error(results)} deg")