"""
Updates of FlashSAC with hybrid (continuous and discrete) actions.

The losses take the exact expectation over the finite support of the discrete actions instead of sampling them,
see hybrid SAC (https://arxiv.org/abs/1912.11077): the critic evaluates every joint discrete action for the same
continuous actions. The discrete actions are regularized by their entropy with a separate temperature.
"""

from typing import Optional

import torch
from torch.amp.grad_scaler import GradScaler

from flash_rl.agents.flashSAC.update import (
    _compute_categorical_td_target,
    _optimizer_step,
    _select_min_q_log_probs,
    add_prefix_to_keys,
)
from flash_rl.agents.utils.network import Network
from flash_rl.buffers import Batch


def _tile(x: torch.Tensor, k: int) -> torch.Tensor:
    """Repeat x k times along the batch dimension: (B, ...) -> (k*B, ...)."""
    return x.repeat(k, *([1] * (x.dim() - 1)))


def _with_discrete_support(actions: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """
    Append every joint discrete action of the support (K, D) to the continuous actions (B, A).
    Returns (K*B, A + D), ordered by support element first.
    """
    batch_size = actions.shape[0]
    discrete_actions = support.to(actions.dtype).repeat_interleave(batch_size, dim=0)
    return torch.cat([_tile(actions, support.shape[0]), discrete_actions], dim=-1)


def update_hybrid_actor(
    actor: Network,
    critic: Network,
    temperature: Network,
    discrete_temperature: Network,
    discrete_support: torch.Tensor,
    batch: Batch,
    device: torch.device,
    use_amp: bool,
    grad_scaler: Optional[GradScaler],
) -> dict[str, torch.Tensor]:
    """Update actor network with hybrid actions.

    The loss is the expectation over the discrete actions k for the reparameterized continuous actions a:
    alpha * log pi(a) + sum_k p_k * (alpha_d * log p_k - Q(a, k)).

    Args:
        actor: Hybrid actor network.
        critic: Critic network.
        temperature: Temperature network of the continuous actions.
        discrete_temperature: Temperature network of the discrete actions.
        discrete_support: Joint discrete actions (K, D) of the discrete head.
        batch: Batch of transitions.
        device: Device to use.
        use_amp: Whether to use automatic mixed precision.
        grad_scaler: GradScaler for FP16 AMP.
    """

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        actor_obs_all = torch.cat([batch["actor_observation"], batch["actor_next_observation"]], dim=0)  # type: ignore
        actions_all, info = actor(
            observations=actor_obs_all,
            training=True,
        )

        actions = torch.chunk(actions_all, 2, dim=0)[0]
        log_probs = torch.chunk(info["log_prob"], 2, dim=0)[0]
        discrete_log_probs = torch.chunk(info["discrete_log_probs"], 2, dim=0)[0]  # (B, K)
        discrete_probs = discrete_log_probs.exp()
        batch_size, support_size = discrete_log_probs.shape

        # Disable critic gradients to prevent CUDA graph overwriting
        critic.network.requires_grad_(False)
        # Q of every joint discrete action for the same continuous actions
        qs, q_infos = critic(
            observations=_tile(batch["observation"], support_size),  # type: ignore
            actions=_with_discrete_support(actions, discrete_support),
            training=False,
        )
        q = torch.minimum(qs[0], qs[1]).reshape(support_size, batch_size).T  # (B, K)
        critic.network.requires_grad_(True)

        # clone: with CUDA-graph compile modes, the output is overwritten by the next compiled call
        temp_value = temperature().detach().clone()
        discrete_temp_value = discrete_temperature().detach().clone()
        discrete_term = (discrete_probs * (discrete_temp_value * discrete_log_probs - q)).sum(-1)
        actor_loss = (log_probs * temp_value + discrete_term).mean()

        entropy = -log_probs.mean()
        mean_action = actions.mean()
        discrete_entropy = -(discrete_probs * discrete_log_probs).sum(-1).mean()
        discrete_mean_action = (discrete_probs @ discrete_support.to(discrete_probs.dtype)).mean()

    # Gradient step
    _optimizer_step(actor, actor_loss, use_amp, grad_scaler)

    # Weight Normalization
    # NOTE: Make sure you finish all computation before this (e.g., computing info values)
    actor.normalize_parameters()

    update_info = {
        "loss": actor_loss,
        "entropy": entropy,
        "mean_action": mean_action,
        "discrete_entropy": discrete_entropy,
        "discrete_mean_action": discrete_mean_action,
    }
    update_info = add_prefix_to_keys(update_info, "actor")

    return update_info


def update_hybrid_critic(
    actor: Network,
    critic: Network,
    target_critic: Network,
    temperature: Network,
    discrete_temperature: Network,
    discrete_support: torch.Tensor,
    batch: Batch,
    min_v: float,
    max_v: float,
    num_bins: int,
    gamma: float,
    n_step: int,
    device: torch.device,
    use_amp: bool,
    grad_scaler: Optional[GradScaler],
) -> dict[str, torch.Tensor]:
    """Update critic network with hybrid actions.

    The target is the mixture of the categorical targets of every joint discrete action k at the next observation,
    weighted by p_k. The critics evaluate the current and all next state-actions in one batch, since they normalize
    with the batch statistics.

    Args:
        actor: Hybrid actor network.
        critic: Critic network.
        target_critic: Target critic network.
        temperature: Temperature network of the continuous actions.
        discrete_temperature: Temperature network of the discrete actions.
        discrete_support: Joint discrete actions (K, D) of the discrete head.
        batch: Batch of transitions.
        min_v: Minimum value for categorical distribution.
        max_v: Maximum value for categorical distribution.
        num_bins: Number of bins for categorical distribution.
        gamma: Discount factor.
        n_step: N-step return.
        device: Device to use.
        use_amp: Whether to use automatic mixed precision.
        grad_scaler: GradScaler for FP16 AMP.
    """

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        # Compute target values
        with torch.no_grad():
            next_actions, info = actor(
                observations=batch["actor_next_observation"],
                training=False,
            )
            # Clone variables to prevent overwriting
            next_actions = next_actions.clone()
            next_actor_log_probs = info["log_prob"].clone()
            next_discrete_log_probs = info["discrete_log_probs"].clone()  # (B, K)
            batch_size, support_size = next_discrete_log_probs.shape

            # clone: with CUDA-graph compile modes, the output is overwritten by the next compiled call
            temp_value = temperature().clone()
            discrete_temp_value = discrete_temperature().clone()

            next_actor_entropy = temp_value * next_actor_log_probs
            # (B, K): entropy bonus of the continuous actions and of every joint discrete action
            next_entropy = next_actor_entropy.unsqueeze(-1) + discrete_temp_value * next_discrete_log_probs

            obs_all = torch.cat(
                [batch["observation"], _tile(batch["next_observation"], support_size)], dim=0  # type: ignore
            )
            act_all = torch.cat(
                [batch["action"], _with_discrete_support(next_actions, discrete_support)], dim=0  # type: ignore
            )

            # qs_all: (2, (1+K)*B)
            # q_infos_all['log_probs']: (2, (1+K)*B, num_bins)
            qs_all, q_infos_all = target_critic(
                observations=obs_all,
                actions=act_all,
                training=True,
            )
            next_qs = qs_all[:, batch_size:]
            next_q_log_probs = q_infos_all["log_prob"][:, batch_size:]
            next_q_log_probs = _select_min_q_log_probs(next_qs, next_q_log_probs)  # (K*B, num_bins)

            # Compute target probs of every joint discrete action, ordered by support element first
            target_probs = _compute_categorical_td_target(
                target_log_probs=next_q_log_probs,
                reward=_tile(batch["reward"], support_size),  # type: ignore
                done=_tile(batch["terminated"], support_size),  # type: ignore
                actor_entropy=next_entropy.T.reshape(-1),
                gamma=gamma**n_step,
                num_bins=num_bins,
                min_v=min_v,
                max_v=max_v,
            )
            # mixture over the discrete actions: (K, B, num_bins) -> (B, num_bins)
            next_discrete_probs = next_discrete_log_probs.exp().T.unsqueeze(-1)
            target_probs = (next_discrete_probs * target_probs.reshape(support_size, batch_size, -1)).sum(0)
            max_entropy_bonus = next_actor_entropy.max()

        # Compute predicted q-value
        pred_qs_all, pred_q_infos = critic(
            observations=obs_all,
            actions=act_all,
            training=True,
        )
        pred_log_probs = pred_q_infos["log_prob"][:, :batch_size]

        ce_loss = -(target_probs.unsqueeze(0) * pred_log_probs).sum(dim=-1)  # (2, B)
        critic_loss = ce_loss.mean()

    # Gradient step
    _optimizer_step(critic, critic_loss, use_amp, grad_scaler)

    # Weight Normalization
    critic.normalize_parameters()

    update_info = {
        "loss": critic_loss,
        "max_entropy_bonus": max_entropy_bonus,
    }
    update_info = add_prefix_to_keys(update_info, "critic")

    return update_info
