import environments.wrappers as wrappers
from environments.concurrent_envs import ConcurrentEnvs


def make_env(config):
    suite, task = config.task.split("__", 1)
    task = task.lower()
    if suite == "robomimic":
        from environments.robomimic.constants import IMAGE_OBS_KEYS
        from environments.robomimic.env_make import make_env_robomimic
        from environments.robomimic.utils import (
            create_shape_meta,
            get_robomimic_dataset_path_and_env_meta,
        )

        _dataset_path, env_meta = get_robomimic_dataset_path_and_env_meta(
            env_id=task,
            shaped=config.shape_rewards,
            image_size=config.image_size,
            done_mode=config.done_mode,
            datadir=config.datadir,
        )
        shape_meta = create_shape_meta(img_size=config.image_size, include_state=True)

        env = make_env_robomimic(
            env_meta,
            IMAGE_OBS_KEYS,
            shape_meta,
            add_state=True,
            reward_shaping=config.shape_rewards,
            config=config,
            offscreen_render=False,
        )
        env = wrappers.TimeLimit(env, duration=config.time_limit)
        env = wrappers.SelectAction(env, key="action")
        env = wrappers.UUID(env)
    elif suite == "robocasa":
        from environments.robocasa.utils import make_env_robocasa

        env = make_env_robocasa(
            config=config,
            task=task,
            suite=suite,
        )
        env = wrappers.TimeLimit(env, duration=config.time_limit)
        env = wrappers.SelectAction(env, key="action")
        env = wrappers.UUID(env)
    elif suite == "maniskill":
        from environments.maniskill.utils import make_maniskill_env

        env = make_maniskill_env(config, suite=suite, task=task)
        env = wrappers.UUID(env)
    else:
        raise ValueError(f"Unknown env suite {suite}")

    return env


def build_train_envs(config):
    suite, _task = config.task.split("__", 1)
    if suite in ["robomimic", "robocasa"]:
        return ConcurrentEnvs(config=config, env_make=make_env, num_envs=config.num_envs)
    if suite == "maniskill":
        if config.use_cpu_env:
            return ConcurrentEnvs(
                config=config, env_make=make_env, num_envs=config.num_envs
            )
        return make_env(config)
    raise ValueError(f"Unknown env suite {suite}")
