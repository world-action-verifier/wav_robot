import collections
import copy
import time

import numpy as np
import torch
from termcolor import cprint


class RelabelService:
    def relabel_with_mppi_post(
        self, trainer, num_trajs_to_relabel, batch_size=32, select_from_end=True
    ):
        num_replay_trajs = len(trainer.replay_buffer.keys())
        if num_trajs_to_relabel > num_replay_trajs:
            print("num_trajs_to_relabel > num_replay_trajs, relabelling all trajs")
            to_relable_keys = list(trainer.replay_buffer.keys())
        elif select_from_end:
            print(
                f"Num Trajs in Replay Buffer: {num_replay_trajs}. Relabling last {num_trajs_to_relabel} trajectories"
            )
            all_keys = sorted(list(trainer.replay_buffer.keys()))
            to_relable_keys = all_keys[-num_trajs_to_relabel:]
        else:
            print(
                f"Num Trajs in Replay Buffer: {num_replay_trajs}. Relabelling {num_trajs_to_relabel} trajectories"
            )
            to_relable_keys = np.random.choice(
                list(trainer.replay_buffer.keys()), num_trajs_to_relabel, replace=False
            )

        start_time = time.time()
        relabelled_buffer = collections.OrderedDict()

        batched_trajkeys = [
            to_relable_keys[i : i + batch_size]
            for i in range(0, len(to_relable_keys), batch_size)
        ]

        cprint(
            f"Number of Batches: {len(batched_trajkeys)}, Batch Size: {batch_size}",
            "yellow",
        )

        for idx, trajkeys in enumerate(batched_trajkeys):
            data_keys = None
            for trajkey in trajkeys:
                data_traj_i = copy.deepcopy(trainer.replay_buffer[trajkey])
                data_traj_i["base_action"] = np.stack(data_traj_i["base_action"])
                relabelled_buffer[trajkey] = data_traj_i
                if data_keys is None:
                    data_keys = data_traj_i.keys()

            batch_traj_lens = [len(trainer.replay_buffer[key]["state"]) for key in trajkeys]

            all_mppi_actions = {key: [] for key in trajkeys}
            latent = None
            action = None
            for i in range(max(batch_traj_lens)):
                idx_to_select = []
                dones = []
                for key_idx, key in enumerate(trajkeys):
                    traj_len = batch_traj_lens[key_idx]
                    if i < traj_len:
                        idx_to_select.append(i)
                        dones.append(False)
                    else:
                        idx_to_select.append(traj_len - 1)
                        dones.append(True)

                obs_dreamer = {}
                for data_key in data_keys:
                    data = []
                    for key_idx, trajkey in enumerate(trajkeys):
                        data.append(
                            relabelled_buffer[trajkey][data_key][idx_to_select[key_idx]]
                        )
                    obs_dreamer[data_key] = np.expand_dims(np.stack(data), axis=1)

                obs_dreamer = trainer.dreamer_class.preprocess_for_wm(obs_dreamer)
                embed = trainer.dreamer_class.encode_wm_obs(obs_dreamer)
                embed = embed.squeeze(1)
                latent, _ = trainer.dreamer_class.obs_step_wm(
                    latent, action, embed, obs_dreamer["is_first"]
                )

                base_action = []
                for key_idx, key in enumerate(trajkeys):
                    base_action.append(
                        relabelled_buffer[key]["base_action"][idx_to_select[key_idx]][
                            ..., : trainer.config.pred_horizon
                        ]
                    )
                base_action = np.stack(base_action)

                with torch.no_grad():
                    mppi_actions = trainer.dreamer_class.get_mppi_actions(
                        latent=latent,
                        base_action=torch.tensor(
                            base_action,
                            dtype=torch.float32,
                            device=trainer.config.device,
                        ),
                    )
                    mppi_actions = mppi_actions.cpu().numpy()

                for key_idx, key in enumerate(trajkeys):
                    if not dones[key_idx]:
                        all_mppi_actions[key].append(mppi_actions[key_idx])

                latent = {k: v.detach() for k, v in latent.items()}
                action = []
                for key_idx, key in enumerate(trajkeys):
                    action.append(
                        torch.tensor(
                            relabelled_buffer[key]["action"][idx_to_select[key_idx]],
                            dtype=torch.float32,
                            device=trainer.config.device,
                        )
                    )
                action = torch.stack(action)

            for key in trajkeys:
                all_mppi_actions[key] = np.stack(all_mppi_actions[key])
                relabelled_buffer[key]["residual_action"] = all_mppi_actions[key]
                summed_action = np.clip(
                    relabelled_buffer[key]["base_action"][
                        ..., : trainer.config.pred_horizon
                    ]
                    + all_mppi_actions[key],
                    -1,
                    1,
                )
                relabelled_buffer[key]["action"] = summed_action

            print(
                f"[{idx+1}/{len(batched_trajkeys)}] Relabelled Batch of Trajectories, max_traj_len: {max(batch_traj_lens)}"
            )

        print("Finished Relabelling with MPPI in ", time.time() - start_time)
        return relabelled_buffer
