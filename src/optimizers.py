"""Optimizer controllers used by the configurable DeepHedger.

The Muon and Schedule-Free AdamW updates are adapted from the Apache-2.0
reference code released with:

    Gorishniy et al., "Benchmarking Optimizers for MLPs in Tabular Deep
    Learning", arXiv:2604.15297, official code commit c34e3315.

Only the single-device paths needed by this project are retained. Muon is
applied to hidden linear weights; the output head and vector parameters use
the auxiliary AdamW update, matching the authors' practical recipe.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterable, Iterator

import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


def _zeropower_via_newton_schulz5(
    gradient: torch.Tensor,
    steps: int = 5,
) -> torch.Tensor:
    if gradient.ndim < 2:
        raise ValueError("Muon requires a matrix-like gradient.")
    a, b, c = 3.4445, -4.7750, 2.0315
    value = gradient.bfloat16()
    transposed = value.size(-2) > value.size(-1)
    if transposed:
        value = value.mT
    value = value / (value.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = value @ value.mT
        value = a * value + (b * gram + c * gram @ gram) @ value
    if transposed:
        value = value.mT
    return value


def _muon_update(
    gradient: torch.Tensor,
    momentum_buffer: torch.Tensor,
    *,
    momentum: float,
    nesterov: bool = True,
    ns_steps: int = 5,
) -> torch.Tensor:
    momentum_buffer.lerp_(gradient, 1.0 - momentum)
    update = gradient.lerp(momentum_buffer, momentum) if nesterov else momentum_buffer
    update = _zeropower_via_newton_schulz5(update, steps=ns_steps)
    update *= max(1.0, gradient.size(-2) / gradient.size(-1)) ** 0.5
    return update


def _adam_update(
    gradient: torch.Tensor,
    first_moment: torch.Tensor,
    second_moment: torch.Tensor,
    step: int,
    betas: tuple[float, float],
    eps: float,
) -> torch.Tensor:
    first_moment.lerp_(gradient, 1.0 - betas[0])
    second_moment.lerp_(gradient.square(), 1.0 - betas[1])
    corrected_first = first_moment / (1.0 - betas[0] ** step)
    corrected_second = second_moment / (1.0 - betas[1] ** step)
    return corrected_first / (corrected_second.sqrt() + eps)


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """Muon for hidden matrices and AdamW for all remaining parameters."""

    def __init__(self, param_groups: list[dict[str, Any]]) -> None:
        for group in param_groups:
            if "use_muon" not in group:
                raise ValueError("Every Muon parameter group must define use_muon.")
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0.0)
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.999))
                group["eps"] = group.get("eps", 1e-8)
                group["weight_decay"] = group.get("weight_decay", 0.0)
        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], float] | None = None,
    ) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    state = self.state[parameter]
                    if not state:
                        state["momentum_buffer"] = torch.zeros_like(parameter)
                    update = _muon_update(
                        parameter.grad,
                        state["momentum_buffer"],
                        momentum=float(group["momentum"]),
                    )
                    parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                    parameter.add_(update.reshape(parameter.shape), alpha=-group["lr"])
            else:
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    state = self.state[parameter]
                    if not state:
                        state["exp_avg"] = torch.zeros_like(parameter)
                        state["exp_avg_sq"] = torch.zeros_like(parameter)
                        state["step"] = 0
                    state["step"] += 1
                    update = _adam_update(
                        parameter.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                    parameter.add_(update, alpha=-group["lr"])
        return loss


class AdamWScheduleFree(torch.optim.Optimizer):
    """Single-device Schedule-Free AdamW with explicit train/eval iterates."""

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        *,
        lr: float = 0.0025,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
    ) -> None:
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "r": r,
            "k": 0,
            "warmup_steps": warmup_steps,
            "train_mode": False,
            "weight_sum": 0.0,
            "lr_max": -1.0,
            "scheduled_lr": 0.0,
            "weight_lr_power": weight_lr_power,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def eval(self) -> None:
        for group in self.param_groups:
            beta1, _ = group["betas"]
            if group["train_mode"]:
                for parameter in group["params"]:
                    state = self.state[parameter]
                    if "z" in state:
                        parameter.lerp_(state["z"], weight=1.0 - 1.0 / beta1)
                group["train_mode"] = False

    @torch.no_grad()
    def train(self) -> None:
        for group in self.param_groups:
            beta1, _ = group["betas"]
            if not group["train_mode"]:
                for parameter in group["params"]:
                    state = self.state[parameter]
                    if "z" in state:
                        parameter.lerp_(state["z"], weight=1.0 - beta1)
                group["train_mode"] = True

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], float] | None = None,
    ) -> float | None:
        if not self.param_groups[0]["train_mode"]:
            raise RuntimeError("Schedule-Free AdamW step() called outside train mode.")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            step = int(group["k"])
            warmup_steps = int(group["warmup_steps"])
            schedule = (
                (step + 1) / warmup_steps if step < warmup_steps else 1.0
            )
            learning_rate = group["lr"] * schedule
            group["scheduled_lr"] = learning_rate
            group["lr_max"] = max(learning_rate, group["lr_max"])
            weight = ((step + 1) ** group["r"]) * (
                group["lr_max"] ** group["weight_lr_power"]
            )
            group["weight_sum"] += weight
            checkpoint = (
                0.0 if group["weight_sum"] == 0.0 else weight / group["weight_sum"]
            )
            correction = 1.0 - beta2 ** (step + 1)

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if "z" not in state:
                    state["z"] = parameter.detach().clone()
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["exp_avg_sq"].mul_(beta2).addcmul_(
                    parameter.grad,
                    parameter.grad,
                    value=1.0 - beta2,
                )
                denominator = (
                    state["exp_avg_sq"].div(correction).sqrt().add_(group["eps"])
                )
                normalized_gradient = parameter.grad / denominator
                if group["weight_decay"] != 0.0:
                    normalized_gradient = normalized_gradient.add(
                        parameter,
                        alpha=group["weight_decay"],
                    )
                parameter.lerp_(state["z"], weight=checkpoint)
                parameter.add_(
                    normalized_gradient,
                    alpha=learning_rate * (beta1 * (1.0 - checkpoint) - 1.0),
                )
                state["z"].sub_(normalized_gradient, alpha=learning_rate)
            group["k"] = step + 1
        return loss


def _split_weight_decay_parameters(
    policy: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in policy.named_parameters():
        if (
            parameter.ndim < 2
            or name.endswith("bias")
            or "embedding" in name
            or name.endswith("frequencies")
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return decay, no_decay


def _weight_decay_groups(
    policy: nn.Module,
    weight_decay: float,
) -> list[dict[str, Any]]:
    decay, no_decay = _split_weight_decay_parameters(policy)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def _muon_parameter_groups(
    policy: nn.Module,
    *,
    learning_rate: float,
    muon_learning_rate: float,
    weight_decay: float,
    momentum: float,
) -> list[dict[str, Any]]:
    output_weights: set[nn.Parameter] = set()
    for module in policy.modules():
        mlp = getattr(module, "mlp", None)
        if isinstance(mlp, nn.Sequential):
            linear_layers = [child for child in mlp if isinstance(child, nn.Linear)]
            if linear_layers:
                output_weights.add(linear_layers[-1].weight)

    muon_parameters = {
        module.weight
        for module in policy.modules()
        if isinstance(module, nn.Linear)
        and module.weight not in output_weights
        and min(module.weight.shape) > 1
    }
    decay, no_decay = _split_weight_decay_parameters(policy)
    auxiliary_decay = [p for p in decay if p not in muon_parameters]
    auxiliary_no_decay = [p for p in no_decay if p not in muon_parameters]
    groups: list[dict[str, Any]] = []
    if muon_parameters:
        groups.append(
            {
                "params": list(muon_parameters),
                "lr": muon_learning_rate,
                "momentum": momentum,
                "weight_decay": weight_decay,
                "use_muon": True,
            }
        )
    for parameters, group_decay in (
        (auxiliary_decay, weight_decay),
        (auxiliary_no_decay, 0.0),
    ):
        if parameters:
            groups.append(
                {
                    "params": parameters,
                    "lr": learning_rate,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                    "weight_decay": group_decay,
                    "use_muon": False,
                }
            )
    if not muon_parameters:
        raise ValueError("Muon found no eligible hidden matrix weights.")
    return groups


class OptimizerController:
    """Own optimizer-specific training, evaluation, and averaging semantics."""

    def __init__(
        self,
        policy: nn.Module,
        optimizer: torch.optim.Optimizer,
        family: str,
        ema_decay: float | None = None,
    ) -> None:
        self.policy = policy
        self.optimizer = optimizer
        self.family = family
        self.ema_policy = (
            AveragedModel(
                policy,
                multi_avg_fn=get_ema_multi_avg_fn(ema_decay),
                use_buffers=True,
            )
            if ema_decay is not None
            else None
        )

    def begin_training(self) -> None:
        if isinstance(self.optimizer, AdamWScheduleFree):
            self.optimizer.train()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def step(self) -> None:
        self.optimizer.step()
        if self.ema_policy is not None:
            self.ema_policy.update_parameters(self.policy)

    @contextmanager
    def validation_policy(self) -> Iterator[nn.Module]:
        schedule_free = isinstance(self.optimizer, AdamWScheduleFree)
        if schedule_free:
            self.optimizer.eval()
        try:
            yield self.ema_policy.module if self.ema_policy is not None else self.policy
        finally:
            if schedule_free:
                self.optimizer.train()


def make_optimizer_controller(
    policy: nn.Module,
    *,
    family: str,
    learning_rate: float,
    weight_decay: float,
    ema_decay: float,
    muon_learning_rate: float,
    muon_momentum: float,
) -> OptimizerController:
    if family == "adam":
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        return OptimizerController(policy, optimizer, family)
    if family in {"adamw", "adamw_ema"}:
        optimizer = torch.optim.AdamW(
            _weight_decay_groups(policy, weight_decay),
            lr=learning_rate,
        )
        return OptimizerController(
            policy,
            optimizer,
            family,
            ema_decay=ema_decay if family == "adamw_ema" else None,
        )
    if family == "schedule_free_adamw":
        optimizer = AdamWScheduleFree(
            _weight_decay_groups(policy, weight_decay),
            lr=learning_rate,
        )
        return OptimizerController(policy, optimizer, family)
    if family == "muon":
        optimizer = SingleDeviceMuonWithAuxAdam(
            _muon_parameter_groups(
                policy,
                learning_rate=learning_rate,
                muon_learning_rate=muon_learning_rate,
                weight_decay=weight_decay,
                momentum=muon_momentum,
            )
        )
        return OptimizerController(policy, optimizer, family)
    raise ValueError(f"Unknown optimizer family: {family!r}.")
