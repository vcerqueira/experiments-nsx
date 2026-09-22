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
            mask: Union[torch.Tensor, None] = None,
            y_insample: Union[torch.Tensor, None] = None,
            y_hat_c: torch.Tensor = None,
            gate_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        **Parameters:**<br>
        `y`: tensor, Actual values.<br>
        `y_hat`: tensor, Predicted values.<br>
        `mask`: tensor, Specifies datapoints to consider in loss.<br>
        `y_insample`: tensor, Actual insample values. Accepted for neuralforecast
        compatibility and unused in this loss.<br>
        `y_hat_c`: tensor, Combined prediction used for the signed gradient.<br>
        `gate_weight`: tensor, Per-sample expert weight. Multiplies the signed
        term so each expert is updated in proportion to its gate weight.<br>

        **Returns:**<br>
        `mae`: tensor (single value).
        """
        if y_hat_c is not None:
            losses = torch.sign(y_hat_c - y) * y_hat
            if gate_weight is not None:
                weight = gate_weight
                while weight.ndim < losses.ndim:
                    weight = weight.unsqueeze(-1)
                losses = losses * weight
        else:
            losses = torch.abs(y - y_hat)

        weights = self._compute_weights(y=y, mask=mask)
        return _weighted_mean(losses=losses, weights=weights)
