"""Tiny helper to load params.yaml from the project root.

Every analysis stage that references a tunable parameter calls
load_params() at the top of main(). DVC tracks params.yaml as a
dependency, so any change here invalidates the relevant stages.
"""
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_params():
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)