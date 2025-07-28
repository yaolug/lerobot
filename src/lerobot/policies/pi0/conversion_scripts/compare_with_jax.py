# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import pickle
from pathlib import Path
import argparse

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy


def display(tensor: torch.Tensor):
    if tensor.dtype == torch.bool:
        tensor = tensor.float()
    print(f"Shape: {tensor.shape}")
    print(f"Mean: {tensor.mean().item()}")
    print(f"Std: {tensor.std().item()}")
    print(f"Min: {tensor.min().item()}")
    print(f"Max: {tensor.max().item()}")


def main():
    parser = argparse.ArgumentParser(description="Compare PyTorch and JAX models")
    parser.add_argument("--ckpt-torch-dir", type=str, help="Path to PyTorch checkpoint directory")
    parser.add_argument("--ckpt-jax-dir", type=str, help="Path to JAX checkpoint directory")
    parser.add_argument("--save-dir", type=str, help="Path to save directory")
    args = parser.parse_args()

    num_motors = 14
    device = "cuda"
    # model_name = "pi0_aloha_towel"
    model_name = "pi0_aloha_sim"

    if model_name == "pi0_aloha_towel":
        dataset_repo_id = "lerobot/aloha_static_towel"
    else:
        dataset_repo_id = "lerobot/aloha_sim_transfer_cube_human"

    # Use command line arguments or default values
    ckpt_torch_dir = Path(args.ckpt_torch_dir) if args.ckpt_torch_dir else Path.home() / f".cache/openpi/openpi-assets/checkpoints/{model_name}_pytorch"
    ckpt_jax_dir = Path(args.ckpt_jax_dir) if args.ckpt_jax_dir else Path.home() / f".cache/openpi/openpi-assets/checkpoints/{model_name}"
    save_dir = Path(args.save_dir) if args.save_dir else Path(f"/home/jasonlu/workspace/openpi/data/{model_name}/save")

    with open(save_dir / "example.pkl", "rb") as f:
        example = pickle.load(f)
    with open(save_dir / "outputs.pkl", "rb") as f:
        outputs = pickle.load(f)
    with open(save_dir / "noise.pkl", "rb") as f:
        noise = pickle.load(f)
    with open(save_dir / "processed_inputs.pkl", "rb") as f:
        processed_inputs = pickle.load(f)

    with open(ckpt_jax_dir / "assets/lerobot/aloha_sim_transfer_cube_human/norm_stats.json") as f:
        norm_stats = json.load(f)

    # Override stats
    dataset_meta = LeRobotDatasetMetadata(dataset_repo_id)
    print("=== Stats Override ===")
    print(f"Original dataset stats for observation.state:")
    print(f"  mean: {dataset_meta.stats['observation.state']['mean']}")
    print(f"  std: {dataset_meta.stats['observation.state']['std']}")

    dataset_meta.stats["observation.state"]["mean"] = torch.tensor(
        norm_stats["norm_stats"]["state"]["mean"][:num_motors], dtype=torch.float32
    )
    dataset_meta.stats["observation.state"]["std"] = torch.tensor(
        norm_stats["norm_stats"]["state"]["std"][:num_motors], dtype=torch.float32
    )
    dataset_meta.stats["action"]["mean"] = torch.tensor(
        norm_stats["norm_stats"]["actions"]["mean"][:num_motors], dtype=torch.float32
    )
    dataset_meta.stats["action"]["std"] = torch.tensor(
        norm_stats["norm_stats"]["actions"]["std"][:num_motors], dtype=torch.float32
    )


    print(f"Overridden with JAX stats (first {num_motors} motors):")
    print(f"  mean: {dataset_meta.stats['observation.state']['mean']}")
    print(f"  std: {dataset_meta.stats['observation.state']['std']}")
    print()

    # Create LeRobot batch from Jax
    batch = {}
    for cam_key, uint_chw_array in example["images"].items():
        batch[f"observation.images.{cam_key}"] = torch.from_numpy(uint_chw_array) / 255.0
    batch["observation.state"] = torch.from_numpy(example["state"])
    batch["action"] = torch.from_numpy(outputs["actions"])
    batch["task"] = example["prompt"]

    if model_name == "pi0_aloha_towel":
        del batch["observation.images.cam_low"]
    elif model_name == "pi0_aloha_sim":
        batch["observation.images.top"] = batch["observation.images.cam_high"]
        del batch["observation.images.cam_high"]

    # Batchify
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].unsqueeze(0)
        elif isinstance(batch[key], str):
            batch[key] = [batch[key]]
        else:
            raise ValueError(f"{key}, {batch[key]}")

    # To device
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device=device, dtype=torch.float32)

    noise = torch.from_numpy(noise).to(device=device, dtype=torch.float32).unsqueeze(0)

    from lerobot import policies  # noqa

    cfg = PreTrainedConfig.from_pretrained(ckpt_torch_dir)
    cfg.pretrained_path = ckpt_torch_dir
    policy = make_policy(cfg, dataset_meta)

    actions = []
    for i in range(50):
        action = policy.select_action(batch, noise=noise)
        actions.append(action)

    actions = torch.stack(actions, dim=1)
    pi_actions = batch["action"]
    print("actions")
    display(actions)
    print()
    print("pi_actions")
    display(pi_actions)
    print("atol=3e-2", torch.allclose(actions, pi_actions, atol=3e-2))
    print("atol=2e-2", torch.allclose(actions, pi_actions, atol=2e-2))
    print("atol=1e-2", torch.allclose(actions, pi_actions, atol=1e-2))

    # Calculate max absolute error
    abs_diff = torch.abs(actions - pi_actions)
    max_abs_error = torch.max(abs_diff).item()
    print(f"\nMax absolute error: {max_abs_error:.6f}")

    # Calculate max relative error
    # Add small epsilon to avoid division by zero
    eps = 1e-8
    rel_diff = abs_diff / (torch.abs(pi_actions) + eps)
    max_rel_error = torch.max(rel_diff).item()
    print(f"Max relative error: {max_rel_error:.6f}")


if __name__ == "__main__":
    main()
