import os

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


DRY_RUN = True

MAX_SAMPLES = 3
SEED = 123
USE_MPS = _env_bool("USE_MPS", True)
USE_CUDA = _env_bool("USE_CUDA", False)
ENGINE = "mps" if USE_MPS else ("gpu" if USE_CUDA else "cpu")

if DRY_RUN:
    N_SAMPLES = 2
    LIMIT_EPOCHS = True
else:
    N_SAMPLES = 5000
    LIMIT_EPOCHS = False

DATASETS = ['monash_m1_monthly',
            'monash_m1_quarterly',
            'monash_m3_monthly',
            'monash_m3_quarterly',
            'monash_tourism_monthly',
            'monash_tourism_quarterly',
            'monash_hospital',
            "ECL",
            "Exchange",
            "TrafficL",
            "Weather", ]
LH_DATASETS = ["ECL",
               "Exchange",
               "TrafficL",
               "Weather"]
