from typing import Union
import torch

from neuralforecast.losses.pytorch import _weighted_mean, BasePointLoss


class MAEGrad(BasePointLoss):

    def __init__(self, horizon_weight=None):
        super(MAEGrad, self).__init__(
            horizon_weight=horizon_weight, outputsize_multiplier=1, output_names=[""]
        )

    def __call__(
            self,
            y: torch.Tensor,
            y_hat: torch.Tensor,
            y_hat_c: torch.Tensor = None,
            mask: Union[torch.Tensor, None] = None,
    ):
        """
        **Parameters:**<br>
        `y`: tensor, Actual values.<br>
        `y_hat`: tensor, Predicted values.<br>
        `mask`: tensor, Specifies datapoints to consider in loss.<br>

        **Returns:**<br>
        `mae`: tensor (single value).
        """
        # losses = torch.abs(y - y_hat)
        if y_hat_c is not None:
            losses = torch.sign(y_hat_c - y) * y_hat
        else:
            losses = torch.abs(y - y_hat)

        weights = self._compute_weights(y=y, mask=mask)
        return _weighted_mean(losses=losses, weights=weights)
