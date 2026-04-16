import os
import random
import time

import einops
import imageio
import numpy as np
import torch

from wav.dreamer.tools import set_seed_everywhere


class ModelEvaluator:
    def __init__(
        self,
        agent,
        envs,
        default_seed,
        visualize=False,
        parent_output_dir="",
        eval_num_runs=10,
        step=0,
        MAX_EPISODE_LENGTH=2000,
    ):
        self.eval_num_runs = eval_num_runs
        self.default_seed = default_seed
        self.agent = agent
        self.envs = envs
        self.visualize = visualize
        self.parent_output_dir = parent_output_dir
        self.MAX_EPISODE_LENGTH = MAX_EPISODE_LENGTH
        self.step = step
        self.NUM_RUNS = int(np.ceil(eval_num_runs / envs.num_envs))

    def enter_seed(self, seed):
        # Store random states for torch and numpy
        self.random_state = random.getstate()
        self.np_random_state = np.random.get_state()
        self.torch_random_state = torch.random.get_rng_state()

        # Set the seeds
        set_seed_everywhere(seed)

    def exit_seed(self):
        # Reset the seeds
        set_seed_everywhere(self.default_seed)

        # Restore random states for torch and numpy
        # (To continue same sequence outside the scope of enter_seed and exit_seed)
        random.setstate(self.random_state)
        np.random.set_state(self.np_random_state)
        torch.random.set_rng_state(self.torch_random_state)

    # Write meta ENV that operates on list of envs and can do things like step and all
    def evaluate_agent_seed(self, seed):
        self.enter_seed(seed)

        # Make Environment and Agent
        envs = self.envs
        num_envs = envs.num_envs

        # Run the Evaluation
        successes = np.zeros(num_envs, dtype=bool)
        total_rewards = np.zeros(num_envs)
        total_orig_rewards = np.zeros(num_envs)
        length_episodes = np.zeros(num_envs)

        dones = np.zeros(num_envs, dtype=bool)
        image_frames = []
        handcam_frames = []
        success_frames = []
        step_rewards_seed = {
            "gt_reward": [],
            "reward_output": [],
        }

        step_obs = envs.reset()
        self.agent.reset()

        print(f"EVALUATING {num_envs} ENVIRONMENTS on SEED {seed}")
        for step_id in range(self.MAX_EPISODE_LENGTH):
            if np.all(dones):
                break

            action = self.agent.get_action(step_obs)
            step_obs, step_rewards, step_dones, step_infos = envs.step(action)

            # Add reward if env is not done
            total_rewards += (1 - dones) * step_rewards

            # Add orig_reward from step_infos if available
            if "orig_reward" in step_infos:
                total_orig_rewards += (1 - dones) * step_infos["orig_reward"]

            # Succeeded currently or previously
            successes = np.logical_or(successes, step_infos["success"])

            # Add if not succeeded
            length_episodes += np.logical_not(successes) * 1

            # Done currently or Succeeded
            dones = np.logical_or(step_dones, successes)

            if self.visualize:
                image_key = (
                    "agentview_image_highres"
                    if "agentview_image_highres" in step_obs
                    else "agentview_image"
                )
                # Save convert to np array if its a tensor
                if isinstance(step_obs[image_key], torch.Tensor):
                    step_obs[image_key] = step_obs[image_key].cpu().numpy()

                eye_in_hand_key = (
                    "robot0_eye_in_hand_image_highres"
                    if "robot0_eye_in_hand_image_highres" in step_obs
                    else "robot0_eye_in_hand_image"
                )
                if isinstance(step_obs[eye_in_hand_key], torch.Tensor):
                    step_obs[eye_in_hand_key] = step_obs[eye_in_hand_key].cpu().numpy()

                image_frames.append(step_obs[image_key])
                handcam_frames.append(step_obs[eye_in_hand_key])
                success_frames.append(successes)

                step_rewards_seed["gt_reward"].append(step_rewards)

                if hasattr(self.agent, "reward_output"):
                    step_rewards_seed["reward_output"].append(self.agent.reward_output)

        average_success_rate = np.mean(successes)
        average_total_reward = np.mean(total_rewards)
        average_length_episode = np.mean(length_episodes)
        average_total_orig_reward = np.mean(total_orig_rewards)

        # Exit the seed
        self.exit_seed()

        return (
            average_success_rate,
            average_total_reward,
            average_length_episode,
            average_total_orig_reward,
            [image_frames, handcam_frames],
            success_frames,
            step_rewards_seed,
        )

    def evaluate_agent(self):
        start_time = time.time()
        success_rates = []
        total_avg_rewards = []
        total_avg_orig_rewards = []
        episode_lengths = []
        print(
            f"Evaluating Agent num_eval_runs: {self.eval_num_runs}, num_envs: {self.envs.num_envs}, NUM_RUNS: {self.NUM_RUNS}"
        )
        for seed in range(self.NUM_RUNS):
            print("Seed:", seed)
            (
                success_rate,
                total_avg_reward,
                avg_length_episode,
                total_avg_orig_reward,
                seed_images,
                seed_success_frames,
                step_rewards_seed,
            ) = self.evaluate_agent_seed(seed)
            success_rates.append(success_rate)
            total_avg_rewards.append(total_avg_reward)
            episode_lengths.append(avg_length_episode)
            total_avg_orig_rewards.append(total_avg_orig_reward)
            print(f"\tAverage Success Rate: {success_rate}")
            print(f"\tAverage Total Reward: {total_avg_reward}")
            print(f"\tAverage Episode Length: {avg_length_episode}")
            print(f"\tAverage Total Orig Reward: {total_avg_orig_reward}")
            if self.visualize:
                os.makedirs(self.parent_output_dir / f"step_{self.step}", exist_ok=True)

                # Save npz files of rollout if we are conducting a highres rollout to generate evaluation videos
                if "highres" in str(self.step):
                    for i in range(self.envs.num_envs):
                        unit_eval_key = f"seed_{seed}_env_{i}"

                        # Find end index
                        where_arr = np.where(
                            np.array(seed_success_frames)[:, i] == True
                        )[0]
                        if len(where_arr) > 0:
                            end_index = where_arr[0] + 1
                        else:
                            end_index = len(seed_success_frames)

                        unit_eval_data = {
                            "cam_0": np.array(seed_images[0])[:end_index, i, ...],
                            "gt_reward": np.array(step_rewards_seed["gt_reward"])[
                                :end_index, i
                            ],
                            "reward_output": np.array(
                                step_rewards_seed["reward_output"]
                            )[:end_index, i],
                            "success": np.array(seed_success_frames)[:end_index, i],
                        }

                        # Save all_data_dict as a .npz file
                        np.savez(
                            f"{self.parent_output_dir}/step_{self.step}/{unit_eval_key}.npz",
                            data=unit_eval_data,
                            allow_pickle=True,
                        )
                        print(
                            "Saved all_data_dict to",
                            f"{self.parent_output_dir}/step_{self.step}/{unit_eval_key}.npz",
                        )
                else:
                    # Save videos of the rollouts
                    for i, image_frame in enumerate(seed_images):
                        video_path = f"{self.parent_output_dir}/step_{self.step}/seed_{seed}_cam_{i}_succ_{success_rate:.2f}_rew_{total_avg_reward:.2f}.mp4"
                        self.save_video(image_frame, seed_success_frames, video_path)

        avg_success_rate = np.mean(success_rates)
        avg_total_avg_reward = np.mean(total_avg_rewards)
        episode_length = np.mean(episode_lengths)
        avg_total_avg_orig_reward = np.mean(total_avg_orig_rewards)
        print(f"Average Success Rate: {avg_success_rate}")
        print(f"Average Total Average Reward: {avg_total_avg_reward}")
        print(f"Average Episode Length: {episode_length}")
        print(f"Average Total Average Orig Reward: {avg_total_avg_orig_reward}")
        print("Time taken:", time.time() - start_time)
        self.envs.reset()
        return (
            avg_success_rate,
            avg_total_avg_reward,
            episode_length,
            avg_total_avg_orig_reward,
        )

    def save_video(self, frames, frame_successes, video_path):
        frames = np.array(frames)  # Shape (time, n_envs, frame_height, frame_width, 3)
        n_envs = frames.shape[1]
        frame_successes = np.array(frame_successes)  # Shape (time, n_envs)
        frame_successes = np.tile(
            frame_successes[..., None, None], (1, 1, frames.shape[2], frames.shape[3])
        )  # Shape (time, n_envs, frame_height, frame_width)

        # If success, bump up greenness in image
        frames[..., 1] = np.clip(
            np.array(frames[..., 1], dtype=np.float32)
            + np.array(frame_successes, dtype=np.float32) * 50,
            0,
            255,
        )

        # Stack in columns of 5
        n_cols = 5
        n_rows = int(np.ceil(n_envs / n_cols))
        n_pad = (n_rows * n_cols) - n_envs
        if n_pad > 0:
            zero_frames = np.zeros(
                (
                    frames.shape[0],
                    n_pad,
                    frames.shape[2],
                    frames.shape[3],
                    frames.shape[4],
                )
            )
            frames = np.concatenate([frames, zero_frames], axis=1)

        # Rearrange to shape n_imgs x (n_rows * frame_height) x (n_cols * frame_width) x c
        frames = einops.rearrange(
            frames,
            "n_imgs (n_rows n_cols) frame_height frame_width c -> n_imgs (n_rows frame_height) (n_cols frame_width) c",
            n_rows=n_rows,
            n_cols=n_cols,
        )
        frames = np.uint8(frames)

        # Use imageio to save video
        imageio.mimwrite(
            video_path, frames, fps=30, codec="libx264", format="mp4", quality=10
        )

        print("Saved video to", video_path)
