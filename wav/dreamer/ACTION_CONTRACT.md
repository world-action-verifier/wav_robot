# Dreamer Action/State Contract

This document captures the public interface between `Dreamer`, `ResidualPolicy`, and trainer services.

## `Dreamer.get_action(obs_orig, state)` output

- `action_dict["base_action"]`: `torch.Tensor`, shape `(num_envs, action_dim)`, dtype `float32`
- `action_dict["residual_action"]`: `torch.Tensor`, shape `(num_envs, action_dim)`, dtype `float32`
- `action_dict["reward_output"]` (optional): model-dependent scalar/tensor reward estimate
- `state`: tuple `(latent, action_sum)`

## `state` structure

- `latent`: dict of recurrent world model tensors, device = configured training device
- `action_sum`: `torch.Tensor`, shape `(num_envs, action_dim)`, dtype `float32`

`state` should be treated as opaque by callers and only passed back into `Dreamer.get_action`.

## WM training/evaluation interfaces

- `Dreamer.train_step(data, training_step)`:
  - `data` is batch dict (same schema as replay dataset output)
  - returns metric dict with scalar values or arrays
- `Dreamer.evaluate_wm_batch_metrics(batch)`:
  - `batch` follows same temporal schema used by world model training
  - returns numeric metrics used by eval aggregation
