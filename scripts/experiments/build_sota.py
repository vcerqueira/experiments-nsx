import warnings
from pathlib import Path

from neuralforecast import NeuralForecast

from src.loaders import ChronosDataset, LongHorizonDatasetR
from src.neuralnets import ModelsConfig
from src.config import ENGINE, LIMIT_EPOCHS, DATASETS, LH_DATASETS

warnings.filterwarnings('ignore')

N_SAMPLES = 20
ASSETS_PATH = Path(__file__).resolve().parents[2] / 'assets' / 'sota'


def load_sota(dataset):
    """Load the fitted Auto models saved for this dataset name."""
    path = ASSETS_PATH / dataset
    if not (path / 'configuration.pkl').exists():
        raise FileNotFoundError(f'No saved Auto models for dataset {dataset!r} in {path}')
    return NeuralForecast.load(path=str(path))


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

        print(f'Fitting Auto models on {target}')
        models = ModelsConfig.get_auto_nf_models(
            horizon=horizon,
            n_samples=N_SAMPLES,
            engine=ENGINE,
            limit_epochs=LIMIT_EPOCHS,
        )
        nf = NeuralForecast(models=models, freq=freq)
        # The last horizon of the original series stays out. This horizon of
        # the training split is the Auto search validation window; refit_with_val
        # then fits the chosen configs on all of train.
        nf.fit(df=train, val_size=horizon)
        save_path.mkdir(parents=True, exist_ok=True)
        nf.save(
            path=str(save_path),
            model_index=None,
            overwrite=True,
            save_dataset=True,
        )
        print(f'Saved {target} to {save_path.resolve()}')
