import collections
import time

from termcolor import cprint

from wav.trainer_utils import label_expert_eps, make_retrain_dp_dataset


class RoundLoop:
    def run(self, trainer):
        if getattr(trainer.config, "wm_only_mode", False):
            cprint(
                "\n-------------WM-only mode: training world model only-------------",
                "green",
                attrs=["bold"],
            )
            trainer.warm_start_wm()
            trainer.train_wm_critic(
                itrs=int(getattr(trainer.config, "wm_only_train_itrs", 5000))
            )
            save_dir = trainer.config.logdir / "latest_residual_checkpoint.pt"
            trainer.dreamer_class.save_checkpoint(path=save_dir)
            cprint("WM-only training finished.", "green")
            return

        trainer.warm_start_wm()
        trainer.eval_base_policy(prefix="init", round_id=-1, base_policy=trainer.base_policy)
        relabelled_buffer = collections.OrderedDict()

        for round_id in range(10000000):
            trainer.state.round_id = round_id
            cprint(
                f"\n-------------Starting Round: {round_id} | Num Env Steps: {trainer._env_step}-------------",
                "green",
                attrs=["bold"],
            )
            start_time = time.time()

            if trainer._env_step >= trainer.config.train_dp_mppi_params["n_env_steps"]:
                print(
                    f"Reached max env steps: {trainer.config.train_dp_mppi_params['n_env_steps']}. Stopping training."
                )
                trainer.eval_mppi_policy(prefix=f"round_{round_id}", round_id=round_id)
                trainer.eval_base_policy(
                    prefix=f"round_{round_id}",
                    round_id=round_id,
                    base_policy=trainer.base_policy,
                )
                save_dir = trainer.config.logdir / "latest_residual_checkpoint.pt"
                trainer.dreamer_class.save_checkpoint(path=save_dir)

                ckpt_file = trainer.config.logdir / "latest_base_policy.pt"
                trainer.base_policy.trainer.save_checkpoint(
                    ckpt_file, global_step=trainer._step
                )
                break

            label_expert_eps(
                expert_eps=trainer.expert_eps,
                dreamer_class=trainer.dreamer_class,
            )

            n_steps_collected = trainer.collect_trajs()
            trainer.trim_buffer(trainer.replay_buffer)

            if bool(getattr(trainer.config, "enable_stage_wm_update", True)):
                cprint(f"\nStarting WM + Critic Training at Round: {round_id}", "green")
                trainer.train_wm_critic(
                    itrs=int(
                        trainer.config.train_dp_mppi_params["rounds_train_ratio"]
                        * n_steps_collected
                    )
                )
            else:
                cprint(
                    "Skipping WM update stage (enable_stage_wm_update=False)",
                    "yellow",
                )

            if round_id % trainer.config.train_dp_mppi_params["eval_every_round"] == 0:
                trainer.eval_mppi_policy(prefix=f"round_{round_id}", round_id=round_id)
                save_dir = trainer.config.logdir / "latest_residual_checkpoint.pt"
                trainer.dreamer_class.save_checkpoint(path=save_dir)
                ckpt_file = trainer.config.logdir / "latest_base_policy.pt"
                trainer.base_policy.trainer.save_checkpoint(
                    ckpt_file, global_step=trainer._step
                )

            if (
                round_id % trainer.config.train_dp_mppi_params["update_dp_every"] == 0
                and round_id > 0
                and trainer._env_step
                <= int(0.95 * trainer.config.train_dp_mppi_params["n_env_steps"])
            ):
                cprint(f"\nBegin Relabelling with MPPI at Round: {round_id}", "green")
                relabelled_buffer_curr = trainer.relabel_with_mppi_post(
                    num_trajs_to_relabel=trainer.config.train_dp_mppi_params[
                        "n_traj_to_relabel_per_round"
                    ]
                )
                relabelled_buffer.update(relabelled_buffer_curr)

                sorted_keys = sorted(list(relabelled_buffer.keys()))
                if (
                    len(sorted_keys)
                    > trainer.config.train_dp_mppi_params["n_dp_traj_buffer_size"]
                ):
                    keys_to_keep = sorted_keys[
                        -trainer.config.train_dp_mppi_params["n_dp_traj_buffer_size"] :
                    ]
                    relabelled_buffer = {k: relabelled_buffer[k] for k in keys_to_keep}
                    print(
                        f"Trimmed dp_train_buffer to {len(relabelled_buffer)} trajectories"
                    )

                print(
                    "Num Trajectories in DP Train Buffer: ",
                    len(relabelled_buffer.keys()),
                )
                cprint(f"\nTraining DP with MPPI at Round: {round_id}", "green")
                dp_dataset = make_retrain_dp_dataset(
                    replay_buffer=relabelled_buffer,
                    expert_eps=trainer.expert_eps,
                    config=trainer.config,
                )
                trainer._step = trainer.base_policy.train_base_policy(
                    train_dataset=dp_dataset,
                    expert_val_eps=trainer.expert_val_eps,
                    eval_envs=trainer.eval_envs,
                    init_step=trainer._step,
                    train_steps=trainer.config.train_dp_mppi_params["n_dp_train_itrs"],
                    log_prefix="base_dp",
                    run_eval=False,
                )
                trainer.state.step = trainer._step

            if round_id % trainer.config.train_dp_mppi_params["eval_every_round"] == 0:
                trainer.eval_base_policy(
                    prefix=f"round_{round_id}",
                    round_id=round_id,
                    base_policy=trainer.base_policy,
                )

            print(f"Round {round_id} took {time.time() - start_time} seconds")
            if trainer.logger is not None:
                trainer.logger.scalar("round_time", time.time() - start_time)
