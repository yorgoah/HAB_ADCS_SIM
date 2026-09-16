import copy
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = REPO_ROOT / "config" / "sim_params" / "parameters_ec60.json"


@pytest.fixture
def base_params() -> dict:
    """A fresh copy of the real ec60 parameters, safe for a test to mutate."""
    with open(PARAMS_PATH) as f:
        return json.load(f)


@pytest.fixture
def short_params(base_params: dict) -> dict:
    """base_params trimmed to a short duration so integration tests stay fast."""
    params = copy.deepcopy(base_params)
    params["simulation"]["duration"] = 0.2
    return params
