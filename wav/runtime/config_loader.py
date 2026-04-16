import argparse
import os
import pathlib
import sys
from typing import Any, Dict, Iterable, Tuple

import ruamel.yaml as yaml

import wav.dreamer.tools as tools


def recursive_update(base: Dict[str, Any], update: Dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            recursive_update(base[key], value)
        else:
            base[key] = value


def _convert_override_value(raw_value: str, current_value: Any) -> Any:
    if current_value is None:
        lowered = raw_value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        try:
            return int(raw_value)
        except ValueError:
            try:
                return float(raw_value)
            except ValueError:
                return raw_value
    if isinstance(current_value, bool):
        lowered = raw_value.lower()
        if lowered not in {"true", "false"}:
            raise ValueError(f"Expected bool override but got '{raw_value}'")
        return lowered == "true"
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int(raw_value)
    if isinstance(current_value, float):
        return float(raw_value)
    return raw_value


def update_nested_obj(obj: Any, key_str: str, value_str: str) -> None:
    keys = key_str.split(".")
    if not keys:
        raise ValueError("Empty --set key is not allowed")

    parent = obj
    for key in keys[:-1]:
        if isinstance(parent, dict):
            if key not in parent:
                raise KeyError(f"Key path '{key_str}' missing segment '{key}'")
            parent = parent[key]
        else:
            if not hasattr(parent, key):
                raise AttributeError(f"Attribute path '{key_str}' missing segment '{key}'")
            parent = getattr(parent, key)

    leaf_key = keys[-1]
    if isinstance(parent, dict):
        if leaf_key not in parent:
            raise KeyError(f"Key '{leaf_key}' not found while applying --set '{key_str}'")
        parent[leaf_key] = _convert_override_value(value_str, parent[leaf_key])
    else:
        if not hasattr(parent, leaf_key):
            raise AttributeError(
                f"Attribute '{leaf_key}' not found while applying --set '{key_str}'"
            )
        current = getattr(parent, leaf_key)
        setattr(parent, leaf_key, _convert_override_value(value_str, current))


def _load_defaults_from_configs(config_names: Iterable[str]) -> Dict[str, Any]:
    yaml_loader = yaml.YAML(typ="safe", pure=True)
    configs = yaml_loader.load(
        (pathlib.Path(sys.argv[0]).parent / "wav/configs.yaml").read_text()
    )

    defaults: Dict[str, Any] = {}
    for name in config_names:
        if name not in configs:
            raise KeyError(f"Unknown config profile '{name}'")
        recursive_update(defaults, configs[name])
    return defaults


def build_config_from_cli(argv: Iterable[str] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--expt_name", type=str, default=None)
    parser.add_argument("--resume_run", type=bool, default=False)
    args, remaining = parser.parse_known_args(argv)

    config_names = ["defaults", *(args.configs or [])]
    defaults = _load_defaults_from_configs(config_names)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--set",
        nargs="+",
        action="append",
        help="Set a configuration key, e.g. --set dp.pretrained_ckpt path/to/model.pt",
    )

    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    final_config = parser.parse_known_args(remaining)[0]

    if final_config.set:
        for set_arg in final_config.set:
            if len(set_arg) < 2:
                raise ValueError(
                    f"Invalid --set override: {set_arg}. Expected '--set key value'."
                )
            key_str, value_str = set_arg[0], set_arg[1]
            update_nested_obj(final_config, key_str, value_str)

    return final_config


def finalize_runtime_config(config) -> Tuple[Any, str, str]:
    config.mppi["horizon"] = config.pred_horizon
    config.dp["ac_chunk"] = config.pred_horizon

    suite, task = config.task.split("__", 1)
    task = task.lower()

    exp_name = f"{str(config.task).lower()}/{config.wandb_exp_name}_demos{config.num_exp_trajs}"
    config.wandb_exp_name = exp_name
    config.time_limit = config.env_time_limits[task]
    config.logdir = f"{config.scratch_dir}/logs/{exp_name}/seed{config.seed}"
    config.datadir = os.path.join("datasets", f"{suite}_datasets")

    if config.generate_highres_eval:
        config.high_res_render = True

    if not config.debug:
        config.train_dp_mppi_params["n_env_steps"] = config.env_max_steps[task]

    return config, suite, task
