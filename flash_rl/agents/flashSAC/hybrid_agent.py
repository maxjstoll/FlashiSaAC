import math
import os
from dataclasses import replace
from typing import Any, Optional, cast

import gymnasium as gym
import numpy as np
import torch
from torch.amp.grad_scaler import GradScaler

from flash_rl.agents.flashSAC.agent import (
    FlashSACAgent,
    FlashSACConfig,
    _init_flashsac_networks,
    _init_temperature,
    _sample_continuous_actions,
)
from flash_rl.agents.flashSAC.hybrid_update import update_hybrid_actor, update_hybrid_critic
from flash_rl.agents.flashSAC.layer import BernoulliPolicyHead, DiscretePolicyHead
from flash_rl.agents.flashSAC.network import HybridFlashSACActor
from flash_rl.agents.flashSAC.update import update_target_network, update_temperature
from flash_rl.agents.utils.network import Network
from flash_rl.types import NDArray

# The updates evaluate the critics for all 2^n joint Bernoulli actions.
MAX_BERNOULLI_ACTIONS = 4


def split_hybrid_action_space(
    action_space: gym.spaces.Space[Any],
) -> tuple[gym.spaces.Box, gym.spaces.MultiBinary]:
    """
    Split a hybrid action space into its continuous and discrete part.

    Supported: a Dict or Tuple of an unbatched Box (tanh-Gaussian actions) followed by a MultiBinary (Bernoulli
    actions) in `gym.spaces.flatten` order. Note that Dict sorts its keys. The agent's actions are the flat vectors
    [continuous | discrete] in this order.
    """
    if isinstance(action_space, gym.spaces.Dict):
        subspaces = list(action_space.spaces.values())
    elif isinstance(action_space, gym.spaces.Tuple):
        subspaces = list(action_space.spaces)
    else:
        raise TypeError(f"Expected a Box, Dict or Tuple action space, got {action_space}.")

    for space in subspaces:
        if isinstance(space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)):
            raise NotImplementedError(f"Only MultiBinary (Bernoulli) discrete actions are supported, got {space}.")
        if isinstance(space, gym.spaces.Box) and np.issubdtype(space.dtype, np.integer):
            raise ValueError(
                f"Got an integer Box action space {space}: gym.vector.utils.batch_space turns MultiBinary into an"
                " integer Box, pass the unbatched (single) action space instead."
            )
    if (
        len(subspaces) != 2
        or not isinstance(subspaces[0], gym.spaces.Box)
        or not isinstance(subspaces[1], gym.spaces.MultiBinary)
    ):
        raise ValueError(f"Expected a Box followed by a MultiBinary action space (in flatten order), got {action_space}.")

    continuous_space, discrete_space = subspaces
    if len(continuous_space.shape) != 1 or len(discrete_space.shape) != 1:
        raise ValueError(f"Expected unbatched 1-D action spaces, got {action_space}.")
    return continuous_space, discrete_space


def _bernoulli_entropy(prob: float) -> float:
    return -(prob * math.log(prob) + (1.0 - prob) * math.log(1.0 - prob))


def _sample_hybrid_flashsac_actions(
    actor: Network,
    noise: torch.Tensor,
    observations: torch.Tensor,
    temperature: float,
    cur_count: torch.Tensor,
    cur_n: torch.Tensor,
    zeta_cdf: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample hybrid actions with noise repeat logic fully in torch.

    The noise repeat only applies to the continuous actions: the discrete actions are sampled in every step.
    """
    # forward actor → distribution mean and std, discrete distribution params
    mean, std, discrete_params = actor.apply(
        "get_distribution_params",
        observations=observations,
        training=False,
    )
    noise, actions, cur_count, cur_n = _sample_continuous_actions(
        mean=mean,
        std=std,
        noise=noise,
        temperature=temperature,
        cur_count=cur_count,
        cur_n=cur_n,
        zeta_cdf=zeta_cdf,
    )

    discrete_head = cast(DiscretePolicyHead, actor.module.discrete_predictor)
    if temperature == 0.0:
        discrete_actions = discrete_head.mode(discrete_params)
    else:
        discrete_actions = discrete_head.sample(discrete_params)
    actions = torch.cat([actions, discrete_actions.to(actions.dtype)], dim=-1)

    return noise, actions, cur_count, cur_n


def _update_hybrid_networks(
    batch: dict[str, torch.Tensor],
    actor: Network,
    critic: Network,
    target_critic: Network,
    temperature: Network,
    discrete_temperature: Network,
    discrete_support: torch.Tensor,
    cfg: FlashSACConfig,
    do_actor_update: bool,
    device: torch.device,
    grad_scaler: Optional[GradScaler],
) -> dict[str, torch.Tensor]:
    if do_actor_update:
        # Update actor
        actor_info = update_hybrid_actor(
            actor=actor,
            critic=critic,
            temperature=temperature,
            discrete_temperature=discrete_temperature,
            discrete_support=discrete_support,
            batch=batch,  # type: ignore
            device=device,
            use_amp=cfg.use_amp,
            grad_scaler=grad_scaler,
        )

        # Update temperatures
        temperature_info = update_temperature(
            temperature=temperature,
            entropy=actor_info["actor/entropy"],
            target_entropy=cfg.temp_target_entropy,
        )
        temperature_info.update(
            update_temperature(
                temperature=discrete_temperature,
                entropy=actor_info["actor/discrete_entropy"],
                target_entropy=cfg.discrete_temp_target_entropy,
                prefix="discrete_temperature",
            )
        )
    else:
        actor_info = {}
        temperature_info = {}

    # Update critic
    critic_info = update_hybrid_critic(
        actor=actor,  # updated
        critic=critic,
        target_critic=target_critic,
        temperature=temperature,  # updated
        discrete_temperature=discrete_temperature,  # updated
        discrete_support=discrete_support,
        batch=batch,  # type: ignore
        min_v=cfg.critic_min_v,
        max_v=cfg.critic_max_v,
        num_bins=cfg.critic_num_bins,
        gamma=cfg.gamma,
        n_step=cfg.n_step,
        device=device,
        use_amp=cfg.use_amp,
        grad_scaler=grad_scaler,
    )

    target_critic_info = update_target_network(
        target_network=target_critic,
    )

    # Merge all info dicts
    update_info = {
        **actor_info,
        **critic_info,
        **target_critic_info,
        **temperature_info,
    }

    return update_info


class HybridFlashSACAgent(FlashSACAgent):
    """
    FlashSAC with continuous and Bernoulli actions, see `split_hybrid_action_space`.

    The actions (as sampled, stored in the replay buffer and fed to the critic) are the flat vectors
    [continuous | discrete]. The updates take the exact expectation over the discrete actions (see `hybrid_update.py`),
    whose entropy is regularized with a separate temperature.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space[NDArray],
        action_space: gym.spaces.Space[Any],
        env_info: dict[str, Any],
        cfg: FlashSACConfig,
    ):
        continuous_space, discrete_space = split_hybrid_action_space(action_space)
        # set before the base initialization, which uses `_continuous_action_dim`
        self._discrete_action_dim: int = discrete_space.shape[-1]

        if self._discrete_action_dim > MAX_BERNOULLI_ACTIONS:
            raise ValueError(
                f"At most {MAX_BERNOULLI_ACTIONS} Bernoulli actions are supported, got {self._discrete_action_dim}:"
                " the updates evaluate the critics for all joint discrete actions."
            )
        if cfg.actor_bc_alpha > 0:
            raise ValueError("actor_bc_alpha > 0 is not supported with discrete actions.")
        for name in ("discrete_initial_probability", "discrete_temp_target_probability"):
            if not 0.0 < getattr(cfg, name) < 1.0:
                raise ValueError(f"{name} ({getattr(cfg, name)}) must be in the open interval (0, 1).")

        discrete_temp_target_entropy = self._discrete_action_dim * _bernoulli_entropy(
            cfg.discrete_temp_target_probability
        )
        cfg = replace(cfg, discrete_temp_target_entropy=discrete_temp_target_entropy)

        flat_action_space = gym.spaces.Box(
            low=np.concatenate([continuous_space.low, np.zeros(self._discrete_action_dim)]),
            high=np.concatenate([continuous_space.high, np.ones(self._discrete_action_dim)]),
            dtype=np.float32,
        )
        super().__init__(observation_space, flat_action_space, env_info, cfg)

    @property
    def _continuous_action_dim(self) -> int:
        return self._action_dim - self._discrete_action_dim

    @property
    def _discrete_support(self) -> torch.Tensor:
        return cast(DiscretePolicyHead, self._actor.module.discrete_predictor).support

    def _init_networks(self) -> None:
        discrete_head = BernoulliPolicyHead(
            hidden_dim=self._cfg.actor_hidden_dim,
            num_actions=self._discrete_action_dim,
            initial_prob=self._cfg.discrete_initial_probability,
        )
        actor_net = HybridFlashSACActor(
            num_blocks=self._cfg.actor_num_blocks,
            input_dim=self._actor_observation_dim,
            hidden_dim=self._cfg.actor_hidden_dim,
            action_dim=self._continuous_action_dim,
            discrete_head=discrete_head,
        )
        (
            self._actor,
            self._critic,
            self._target_critic,
            self._temperature,
        ) = _init_flashsac_networks(
            actor_net=actor_net,
            critic_observation_dim=self._critic_observation_dim,
            action_dim=self._action_dim,
            cfg=self._cfg,
            device=self._device,
        )
        # Manually compile `get_distribution_params` function
        if self._cfg.use_compile:
            self._actor.network.get_distribution_params = torch.compile(self._actor.network.get_distribution_params, mode=self._cfg.compile_mode)  # type: ignore

        self._discrete_temperature = _init_temperature(self._cfg.discrete_temp_initial_value, self._cfg, self._device)

    def _sample_policy_actions(self, observations: torch.Tensor, temperature: float) -> torch.Tensor:
        (
            self._cached_noise,
            actions,
            self._cur_noise_repeat_count,
            self._cur_noise_repeat_n,
        ) = _sample_hybrid_flashsac_actions(
            actor=self._actor,
            noise=self._cached_noise,
            observations=observations,
            temperature=temperature,
            cur_count=self._cur_noise_repeat_count,
            cur_n=self._cur_noise_repeat_n,
            zeta_cdf=self._zeta_cdf,
        )
        return actions

    def _apply_update(self, batch: dict[str, torch.Tensor], do_actor_update: bool) -> dict[str, torch.Tensor]:
        return _update_hybrid_networks(
            batch=batch,
            actor=self._actor,
            critic=self._critic,
            target_critic=self._target_critic,
            temperature=self._temperature,
            discrete_temperature=self._discrete_temperature,
            discrete_support=self._discrete_support,
            cfg=self._cfg,
            do_actor_update=do_actor_update,
            device=self._device,
            grad_scaler=self._grad_scaler,
        )

    def save(self, path: str) -> None:
        super().save(path)
        self._discrete_temperature.save(os.path.join(path, "discrete_temperature.pt"))

    def load(self, path: str) -> None:
        self._discrete_temperature.load(
            os.path.join(path, "discrete_temperature.pt"), load_optimizer=self._cfg.load_optimizer
        )
        super().load(path)

    def get_metrics(self) -> dict[str, Any]:
        metrics = super().get_metrics()
        assert self._discrete_temperature.optimizer is not None
        metrics["discrete_temperature/learning_rate"] = self._discrete_temperature.optimizer.param_groups[0]["lr"]
        return metrics

    @property
    def discrete_action_dim(self) -> int:
        """Number of discrete actions, the trailing action dimensions."""
        return self._discrete_action_dim
