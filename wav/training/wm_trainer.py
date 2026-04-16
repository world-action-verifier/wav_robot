import collections

import numpy as np
from termcolor import cprint

from wav.classes.rollout_utils import mixed_sample
from wav.dreamer import tools


class WorldModelTrainer:
    def train_wm_critic(self, trainer, itrs):
        num_buffer_transitions = trainer.count_n_transitions_fn(trainer.replay_buffer)
        print("Number of expert transitions: ", trainer.num_expert_transitions)
        print("Number of buffer transitions: ", num_buffer_transitions)
        print(f"Mixed training with 50% expert data for {itrs} iterations")

        expert_dataset = tools.make_dataset(
            trainer.expert_eps,
            batch_length=trainer.config.batch_length,
            batch_size=trainer.config.batch_size // 2,
        )
        train_dataset = tools.make_dataset(
            trainer.replay_buffer,
            batch_length=trainer.config.batch_length,
            batch_size=trainer.config.batch_size // 2,
        )
        sample_dataset = None
        if len(trainer.sample_eps.keys()) > 0:
            sample_dataset = tools.make_dataset(
                trainer.sample_eps,
                batch_length=trainer.config.batch_length,
                batch_size=trainer.config.batch_size // 2,
            )

        if (
            getattr(trainer.config, "wm_only_mode", False)
            and float(getattr(trainer.config, "wm_only_sample_mix_ratio", 0.0)) > 0.0
        ):
            cprint(
                f"WM-only sample mixing enabled: start_itr={int(getattr(trainer.config, 'wm_only_sample_start_itr', 0))}, "
                f"mix_ratio={float(getattr(trainer.config, 'wm_only_sample_mix_ratio', 0.0))}",
                "yellow",
            )

        if trainer._dynamic_sample_selection_enabled:
            cprint(
                f"Dynamic sample selection enabled from source pool: {trainer.config.wm_only_sample_source_pool_jsonl}",
                "yellow",
            )

        for n_wm_itr in range(itrs):
            if trainer._dynamic_sample_selection_enabled and trainer.sample_selector.should_refresh_sample_pool(
                trainer, n_wm_itr
            ):
                refreshed = trainer.sample_selector.refresh_sample_pool(trainer, n_wm_itr)
                if refreshed and len(trainer.sample_eps.keys()) > 0:
                    sample_dataset = tools.make_dataset(
                        trainer.sample_eps,
                        batch_length=trainer.config.batch_length,
                        batch_size=trainer.config.batch_size // 2,
                    )

            train_source = "replay"
            train_dataset_curr = train_dataset
            can_use_sample_pool = (
                sample_dataset is not None
                and len(trainer.sample_eps.keys()) > 0
                and float(getattr(trainer.config, "wm_only_sample_mix_ratio", 0.0)) > 0.0
                and n_wm_itr
                >= int(getattr(trainer.config, "wm_only_sample_start_itr", 0))
            )
            if can_use_sample_pool and (
                np.random.rand()
                < float(getattr(trainer.config, "wm_only_sample_mix_ratio", 0.0))
            ):
                train_dataset_curr = sample_dataset
                train_source = "sample"

            batch = mixed_sample(
                batch_size=trainer.config.batch_size,
                expert_dataset=expert_dataset,
                train_dataset=train_dataset_curr,
                device=trainer.config.device,
                remove_obs_stack=False,
                sqil_discriminator=trainer.config.train_dp_mppi_params["use_discrim"],
            )

            metrics = trainer.dreamer_class.train_step(
                data=batch,
                training_step=trainer._step,
            )
            trainer._step += 1
            trainer.state.step = trainer._step

            wm_eval_every = int(getattr(trainer.config, "wm_eval_every", 0))
            if wm_eval_every > 0 and (
                ((n_wm_itr + 1) % wm_eval_every == 0) or (n_wm_itr == itrs - 1)
            ):
                self.eval_world_model(trainer, eval_itr=n_wm_itr)

            if trainer._step % trainer.config.log_every == 0:
                log_loss = (
                    metrics["value_loss"]
                    if "value_loss" in metrics
                    else np.mean(metrics.get("model_loss", 0.0))
                )
                print(
                    f"[WM + Critic Training] Itr: {n_wm_itr}/{itrs}, Loss: {log_loss}"
                )

                if trainer.logger is not None:
                    for key, value in metrics.items():
                        if not isinstance(value, float):
                            value = np.mean(value)
                        trainer.logger.scalar(f"wm_critic_train/{key}", value)
                    trainer.logger.scalar(
                        "wm_critic_train/sample_batch_used",
                        1.0 if train_source == "sample" else 0.0,
                    )
                    trainer.logger.scalar(
                        "wm_critic_train/sample_pool_size",
                        float(len(trainer.sample_eps.keys())),
                    )
                    openloop_video = trainer.dreamer_class.get_last_openloop_video()
                    if openloop_video is not None:
                        trainer.logger.video(
                            "wm_critic_train/openloop_img_pred_vs_gt",
                            openloop_video,
                        )
                    for video_name, video_value in trainer.dreamer_class.get_last_rollout_videos().items():
                        trainer.logger.video(
                            f"wm_critic_train/{video_name}",
                            video_value,
                        )

                    trainer.logger.scalar("wm_critic_train/step", trainer._step)
                    trainer.logger.scalar("wm_critic_train/itr", n_wm_itr)
                    trainer.logger.write(step=trainer._step, fps=True)

    def iter_eval_chunks(self, trainer, episodes):
        batch_length = int(trainer.config.batch_length)
        for ep in episodes.values():
            if "action" not in ep:
                continue
            total = int(len(ep["action"]))
            if total < 2:
                continue
            for start in range(0, total - 1, batch_length):
                end = min(start + batch_length, total)
                if end - start < 2:
                    continue
                chunk = {}
                for key, value in ep.items():
                    if "log_" in key:
                        continue
                    arr = np.asarray(value)
                    if arr.shape[0] < end:
                        continue
                    chunk[key] = arr[start:end].copy()

                if "action" not in chunk:
                    continue
                t = int(len(chunk["action"]))
                if t < 2:
                    continue

                if "is_first" not in chunk:
                    chunk["is_first"] = np.zeros((t,), dtype=bool)
                chunk["is_first"][0] = True
                if "is_terminal" not in chunk:
                    chunk["is_terminal"] = np.zeros((t,), dtype=bool)
                if "discount" not in chunk:
                    chunk["discount"] = np.ones((t,), dtype=np.float32)
                if "reward" not in chunk:
                    chunk["reward"] = np.zeros((t,), dtype=np.float32)

                yield {k: np.expand_dims(v, axis=0) for k, v in chunk.items()}

    def eval_world_model(self, trainer, eval_itr):
        eval_eps = (
            trainer.wm_eval_eps if len(trainer.wm_eval_eps) > 0 else trainer.expert_val_eps
        )
        if eval_eps is None or len(eval_eps) == 0:
            return

        cprint(
            f"[WM Eval] Running world-model eval at itr {eval_itr}",
            "cyan",
        )

        agg = collections.defaultdict(list)
        n_chunks = 0
        for batch in self.iter_eval_chunks(trainer, eval_eps):
            metrics = trainer.dreamer_class.evaluate_wm_batch_metrics(batch)
            for key, value in metrics.items():
                if isinstance(value, (float, int, np.floating, np.integer)):
                    agg[key].append(float(value))
            n_chunks += 1

        if n_chunks == 0:
            return

        if trainer.logger is not None:
            for key, values in agg.items():
                if len(values) == 0:
                    continue
                trainer.logger.scalar(f"wm_eval/{key}", float(np.mean(values)))
            trainer.logger.scalar("wm_eval/n_chunks", float(n_chunks))
            trainer.logger.scalar("wm_eval/step", float(trainer._step))

            for video_name, video_value in trainer.dreamer_class.get_last_rollout_videos().items():
                trainer.logger.video(f"wm_eval/{video_name}", video_value)

            trainer.logger.write(step=trainer._step, fps=False)
