import torch
import torch.nn as nn


class ExponentiatedGradient(nn.Module):
    """One shared mixture, updated by the absolute-loss subgradient."""

    def __init__(self, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.register_buffer("logits", torch.zeros(num_experts))
        self.register_buffer("weight_sum", torch.zeros(num_experts))
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))
        self.register_buffer("grad_bound", torch.zeros(()))
        self.history = []
        self._round_weights = None

    def weight_vector(self):
        if self.training or int(self.steps) == 0:
            return torch.softmax(self.logits, dim=0)
        averaged = self.weight_sum / self.steps.to(self.weight_sum.dtype)
        return averaged / averaged.sum().clamp_min(1e-12)

    def played_weights(self, batch_size):
        weights = self.weight_vector()
        if self.training:
            self._round_weights = weights.detach()
        return weights.detach().unsqueeze(0).expand(batch_size, -1)

    def mae_weight_grad(self, y_hat, y, expert_forecasts, mask, loss):
        """Subgradient of weighted MAE with respect to the mixture weights."""
        y_hat = y_hat.detach()
        expert_forecasts = expert_forecasts.detach()
        residual_sign = torch.sign(y_hat - y)
        sample_weights = loss._compute_weights(y=y, mask=mask).to(residual_sign.dtype)
        if residual_sign.ndim == 3 and residual_sign.size(-1) == 1:
            residual_sign = residual_sign.squeeze(-1)
            sample_weights = sample_weights.squeeze(-1)
        numer = torch.einsum("bh,beh->e", sample_weights * residual_sign, expert_forecasts)
        denom = sample_weights.sum().clamp_min(1e-8)
        return numer / denom

    def update(self, y_hat, y, expert_forecasts, mask, loss):
        """One exponentiated-gradient step. The played weights stay those of this round."""
        grad = self.mae_weight_grad(y_hat, y, expert_forecasts, mask, loss)
        grad_inf = grad.abs().max()
        self.grad_bound.copy_(torch.maximum(self.grad_bound, grad_inf))
        step = self.steps.to(device=grad.device, dtype=grad.dtype) + 1
        log_experts = torch.log(grad.new_tensor(float(self.num_experts)))
        grad_bound = self.grad_bound.clamp_min(1e-8)
        eta = torch.sqrt(log_experts / (step * grad_bound.square()))
        played = self._round_weights.detach()
        self.logits.sub_(eta * grad)
        self.logits.sub_(self.logits.mean())
        self.weight_sum.add_(played)
        self.steps.add_(1)
        return eta.detach(), grad.detach(), played.detach()

    def note_step(self, step, played_loss, expert_losses, eta, grad, played):
        self.history.append({
            "step": step,
            "played_loss": played_loss,
            "expert_losses": expert_losses,
            "eta": eta,
            "g": grad,
            "w": played,
        })

    def regret(self):
        """Regret of the played gate on the training-step sequence."""
        if not self.history:
            raise RuntimeError("online_eg has not recorded any training steps")
        played_loss = 0.0
        expert_totals = None
        linearized = 0.0
        linearized_expert = None
        for row in self.history:
            played_loss += row["played_loss"]
            if expert_totals is None:
                expert_totals = [0.0] * len(row["expert_losses"])
                linearized_expert = [0.0] * len(row["g"])
            for index, loss_value in enumerate(row["expert_losses"]):
                expert_totals[index] += loss_value
            linearized += sum(
                weight * grad_value for weight, grad_value in zip(row["w"], row["g"])
            )
            for index, grad_value in enumerate(row["g"]):
                linearized_expert[index] += grad_value
        return {
            "mae_regret_vs_best_expert": played_loss - min(expert_totals),
            "linearized_regret": linearized - min(linearized_expert),
            "steps": len(self.history),
        }
