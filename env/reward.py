from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class RewardComponents:
    """
    Phase 3: breakdown of reward components for reward shaping.
    """

    delivery_reward: float  # base reward for successful delivery
    cost_penalty: float  # negative reward based on travel distance * cost_per_distance
    delay_penalty: float  # negative reward proportional to delivery delay
    stockout_penalty: float  # large negative reward for stockouts


def compute_reward(
    *,
    delivered: bool,
    travel_distance: float,
    travel_time: float,
    delay: Optional[float],
    cost_per_distance: float = 1.0,
    delay_penalty_scale: float = 0.5,
    stockout_penalty: float = 50.0,
    delivery_reward: float = 10.0,
) -> tuple[float, RewardComponents]:
    """
    Computes reward for a single order assignment.

    Args:
        delivered: whether order was successfully delivered (not stockout)
        travel_distance: total distance traveled (vehicle->warehouse->customer)
        travel_time: total travel time
        delay: time from order creation to delivery (None if not delivered)
        cost_per_distance: cost multiplier for distance
        delay_penalty_scale: penalty per unit delay
        stockout_penalty: large penalty for stockouts
        delivery_reward: base positive reward for delivery

    Returns:
        (total_reward, RewardComponents)
    """
    if not delivered:
        penalty = -stockout_penalty
        return penalty, RewardComponents(
            delivery_reward=0.0,
            cost_penalty=0.0,
            delay_penalty=0.0,
            stockout_penalty=penalty,
        )

    # Successful delivery: reward - cost - delay penalty
    cost = -travel_distance * cost_per_distance
    delay_penalty = -delay * delay_penalty_scale if delay is not None else 0.0

    total = delivery_reward + cost + delay_penalty
    return total, RewardComponents(
        delivery_reward=delivery_reward,
        cost_penalty=cost,
        delay_penalty=delay_penalty,
        stockout_penalty=0.0,
    )


def normalize_reward(reward: float, reward_scale: float = 1.0) -> float:
    """
    Simple reward normalization (can be extended with running stats).
    """
    return float(reward * reward_scale)

