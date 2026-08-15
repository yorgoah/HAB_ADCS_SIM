import numpy as np
import pickle
from pathlib import Path

results = Path(__file__).resolve().parents[3] / "results" / "simulation_results_ec60.pkl"

def compute_rms_error(data: Path) -> float:

    with open(data, 'rb') as f:
        cached = pickle.load(f)
    data = cached["data"]

    errors = np.array(data['yaw']) - np.arctan2(np.array(data['y']), np.array(data['x']))
    rms_error = np.sqrt(np.mean(errors**2))
    return np.rad2deg(rms_error)

if __name__ == "__main__":
    print(f"RMS Yaw Error: {compute_rms_error(results)} deg")