import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SeriesIndexedDataset(Dataset):
    """Dataset wrapper that keeps NeuralForecast's integer series index."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = dict(self.base[idx])
        item["series_idx"] = int(idx)
        return item

    def __getattr__(self, name):
        return getattr(self.base, name)


def install_series_collate():
    """Keep series_idx in batches. The stock collate drops every key it does not know."""
    from neuralforecast.tsdataset import TimeSeriesLoader

    if getattr(TimeSeriesLoader, "_nsx_series_collate", False):
        return
    original = TimeSeriesLoader._collate_fn

    def _collate_fn(self, batch):
        collated = original(self, batch)
        if batch and isinstance(batch[0], dict) and "series_idx" in batch[0]:
            collated["series_idx"] = torch.tensor(
                [int(item["series_idx"]) for item in batch], dtype=torch.long
            )
        return collated

    TimeSeriesLoader._collate_fn = _collate_fn
    TimeSeriesLoader._nsx_series_collate = True


def ensure_loss_table(model, n_series):
    table = torch.zeros(n_series, model.num_experts, device=model.device)
    if "series_cumloss" in model._buffers:
        model.series_cumloss = table
    else:
        model.register_buffer("series_cumloss", table)


def prepare_fit(model):
    install_series_collate()
    dataset = model.trainer.datamodule.dataset
    if not isinstance(dataset, SeriesIndexedDataset):
        dataset = SeriesIndexedDataset(dataset)
        model.trainer.datamodule.dataset = dataset
    ensure_loss_table(model, len(dataset))


def wrap_predict_dataset(dataset):
    install_series_collate()
    if isinstance(dataset, SeriesIndexedDataset):
        return dataset
    return SeriesIndexedDataset(dataset)


def _window_count(length, window_size, step):
    return (length - window_size) // step + 1


def train_windows_per_serie(model, temporal):
    window_size = model.input_size + model.h
    if model.val_size + model.test_size > 0:
        cutoff = -model.val_size - model.test_size
        temporal = temporal[:, :, :cutoff]
    temporal = model.padder_train(temporal)
    return _window_count(temporal.shape[-1], window_size, model.step_size)


def predict_windows_per_serie(model, batch):
    temporal = batch["temporal"]
    window_size = model.input_size + model.h
    initial_input = temporal.shape[-1] - model.test_size
    if initial_input <= model.input_size:
        temporal = F.pad(
            temporal,
            pad=(model.input_size - initial_input, 0),
            mode="constant",
            value=0.0,
        )
    cutoff = -model.input_size - model.test_size
    temporal = temporal[:, :, cutoff:]
    if model.test_size == 0 and len(model.futr_exog_list) == 0:
        temporal = F.pad(temporal, pad=(0, model.h), mode="constant", value=0.0)
    return _window_count(temporal.shape[-1], window_size, model.predict_step_size)


def validation_windows_per_serie(model, batch):
    """Match NeuralForecast's validation branch of ``_create_windows``.

    Validation keeps every unfolded window. It does not apply the training
    availability filter, and it steps by ``step_size``.
    """
    temporal = batch["temporal"]
    window_size = model.input_size + model.h
    cutoff = -model.input_size - model.val_size - model.test_size
    if model.test_size > 0:
        temporal = temporal[:, :, cutoff:-model.test_size]
    else:
        temporal = temporal[:, :, cutoff:]
    if temporal.shape[-1] < window_size:
        initial_input = temporal.shape[-1] - model.val_size
        temporal = F.pad(
            temporal,
            pad=(model.input_size - initial_input, 0),
            mode="constant",
            value=0.0,
        )
    return _window_count(temporal.shape[-1], window_size, model.step_size)


def window_series_index(model, batch, final_condition, w_idxs):
    if "series_idx" not in batch:
        raise RuntimeError("series_state requires series_idx on the batch")
    flat = final_condition[w_idxs]
    windows_per_serie = train_windows_per_serie(model, batch["temporal"])
    local = torch.div(flat, windows_per_serie, rounding_mode="floor")
    series_idx = batch["series_idx"].to(device=local.device)
    return series_idx[local]


def _gathered_losses(model, series_index):
    table = model._buffers.get("series_cumloss")
    if table is None:
        raise RuntimeError("series_state weights require the cumulative table from fit")
    losses = table.to(device=series_index.device, dtype=torch.float32)
    n_series = losses.shape[0]
    valid = (series_index >= 0) & (series_index < n_series)
    safe_index = series_index.clamp(0, max(n_series - 1, 0))
    return losses[safe_index], valid


def series_scores(model, series_index):
    """Detached pre-softmax Hedge scores, ``-eta * cumulative loss``."""
    gathered, valid = _gathered_losses(model, series_index)
    scores = -model.series_state_eta * gathered
    if not torch.all(valid):
        scores = torch.where(valid.unsqueeze(-1), scores, torch.zeros_like(scores))
    return scores.detach()


def _series_major(model, series_idx, windows_per_serie, values_fn):
    n_series = series_idx.shape[0]
    if windows_per_serie < 1 or n_series == 0:
        empty = series_idx.new_empty(0)
        return values_fn(model, empty)
    flat = torch.arange(n_series * windows_per_serie, device=series_idx.device)
    local = torch.div(flat, windows_per_serie, rounding_mode="floor")
    series = series_idx.to(device=flat.device)[local]
    return values_fn(model, series)


def series_major_scores(model, series_idx, windows_per_serie):
    return _series_major(model, series_idx, windows_per_serie, series_scores)


def arm_predict(model, batch):
    if "series_idx" not in batch:
        raise RuntimeError("series_state predict requires series_idx on the batch")
    windows_per_serie = predict_windows_per_serie(model, batch)
    model._predict_hedge_weights = series_major_scores(
        model, batch["series_idx"], windows_per_serie
    )
    model._predict_hedge_cursor = 0


def arm_validation(model, batch):
    if "series_idx" not in batch:
        raise RuntimeError("series_state validation requires series_idx on the batch")
    windows_per_serie = validation_windows_per_serie(model, batch)
    model._val_hedge_weights = series_major_scores(
        model, batch["series_idx"], windows_per_serie
    )
    model._val_hedge_cursor = 0


def consume_cursor(weights, cursor, n_windows, message):
    chunk = weights[cursor:cursor + n_windows]
    if chunk.shape[0] != n_windows:
        raise RuntimeError(message)
    return chunk, cursor + n_windows


def consume_predict(model, n_windows):
    chunk, cursor = consume_cursor(
        model._predict_hedge_weights,
        model._predict_hedge_cursor,
        n_windows,
        "series_state predict window count does not match the fitted table",
    )
    model._predict_hedge_cursor = cursor
    return chunk


def consume_validation(model, n_windows):
    if getattr(model, "_val_hedge_weights", None) is None:
        raise RuntimeError("series_state validation requires the hedge table for this batch")
    chunk, cursor = consume_cursor(
        model._val_hedge_weights,
        model._val_hedge_cursor,
        n_windows,
        "series_state validation window count does not match the fitted table",
    )
    model._val_hedge_cursor = cursor
    return chunk


def accumulate_losses(model, batch, final_condition, w_idxs, per_window_losses):
    global_idx = window_series_index(model, batch, final_condition, w_idxs)
    losses = torch.stack(per_window_losses, dim=1).detach()
    if model.series_cumloss.device != losses.device:
        model.series_cumloss = model.series_cumloss.to(device=losses.device)
    global_idx = global_idx.to(device=model.series_cumloss.device)
    model.series_cumloss.index_add_(0, global_idx, losses)
