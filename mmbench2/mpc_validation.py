"""Validate full MPC results before resolving a failed evaluation."""
import math

PROTOCOL = dict(horizon=32, replan_stride=16, population=32, iterations=3,
                rollouts_per_candidate=2, denoising_steps=8, bc_warm_start=True)


def valid_episode(row, task, variant, index):
    if (row.get('task'), row.get('variant'), row.get('episode'), row.get('seed')) != (task, variant, index, 731 + index):
        return False
    score = row.get('normalized_score')
    return (row.get('completed_episode') is True and row.get('steps', 0) > 0
            and isinstance(score, (int, float)) and math.isfinite(score) and 0 <= score <= 1)


def valid_result(data, task, variant):
    if not isinstance(data, dict):
        return False
    if any(data.get('protocol', {}).get(k) != v for k, v in PROTOCOL.items()):
        return False
    episodes = data.get('episodes', [])
    if len(episodes) != 3:
        return False
    for index, row in enumerate(episodes):
        if not valid_episode(row, task, variant, index):
            return False
    mean = data.get('mean_normalized_score')
    return (isinstance(mean, (int, float)) and math.isfinite(mean)
            and abs(mean - sum(row['normalized_score'] for row in episodes) / 3) < 1e-9)
