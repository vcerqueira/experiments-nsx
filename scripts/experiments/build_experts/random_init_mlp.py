"""
BUILD A SET OF MLP'S WITH DISTINCT INITIALIZATIONS
"""
import warnings
from pathlib import Path

from neuralforecast import NeuralForecast
from neuralforecast.models import MLP

from src.loaders import ChronosDataset, LongHorizonDatasetR
from src.moe.experts import expert_init_kwargs
from src.config import SEED, ENGINE, LIMIT_EPOCHS, DATASETS, LH_DATASETS

warnings.filterwarnings('ignore')

NUM_EXPERTS = 10
INPUT_SIZE_MULTIPLIER = 2
MAX_STEPS = 1000
ASSETS_PATH = Path(__file__).resolve().parents[2] / 'assets' / 'sota'
# ASSETS_PATH = Path('./assets/experts').resolve()


def load_experts(dataset):
    """Load the fitted NeuralForecast experts saved for this dataset name."""
    path = ASSETS_PATH / dataset
    if not (path / 'configuration.pkl').exists():
        raise FileNotFoundError(f'No saved experts for dataset {dataset!r} in {path}')
    return NeuralForecast.load(path=str(path))

# load_experts('monash_m1_monthly')


def build_mlp_experts(horizon, input_size):
    max_steps = 2 if LIMIT_EPOCHS else MAX_STEPS
    mlp_kwargs = expert_init_kwargs('mlp', None)
    # Distinct aliases. NeuralForecast names checkpoints from the alias, and
    # identical names collapse to one model on load.
    return [
        MLP(
            h=horizon,
            input_size=input_size,
            learning_rate=1e-3,
            scaler_type='standard',
            max_steps=max_steps,
            batch_size=128,
            windows_batch_size=256,
            random_seed=SEED + index + 1,
            alias=f'MLP_{index}',
            accelerator=ENGINE,
            **mlp_kwargs,
        )
        for index in range(NUM_EXPERTS)
    ]


if __name__ == '__main__':
    print(ASSETS_PATH.resolve())

    for target in DATASETS:
        if target in LH_DATASETS:
            _, horizon, n_lags, _, _ = LongHorizonDatasetR.load_everything(target, resample_to='D')
            df, horizon, n_lags, freq, _ = LongHorizonDatasetR.load_everything(
                target,
                min_n_instances=2 * (n_lags + horizon),
                resample_to='D',
            )
        else:
            _, horizon, n_lags, _, _ = ChronosDataset.load_everything(target)
            df, horizon, n_lags, freq, _ = ChronosDataset.load_everything(
                target,
                min_n_instances=2 * (n_lags + horizon),
            )

        train, _ = ChronosDataset.time_wise_split(df, horizon)

        save_path = ASSETS_PATH / target
        if (save_path / 'configuration.pkl').exists():
            print(f'Skipping {target} -- already saved')
            continue

        print(f'Fitting {NUM_EXPERTS} MLP experts on {target}')
        nf = NeuralForecast(
            models=build_mlp_experts(horizon, n_lags * INPUT_SIZE_MULTIPLIER),
            freq=freq,
        )
        nf.fit(df=train)
        save_path.mkdir(parents=True, exist_ok=True)
        nf.save(
            path=str(save_path),
            model_index=None,
            overwrite=True,
            save_dataset=True,
        )
        print(f'Saved {target} to {save_path.resolve()}')
