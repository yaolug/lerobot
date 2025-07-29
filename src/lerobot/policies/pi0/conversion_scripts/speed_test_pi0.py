#!/usr/bin/env python


import os
import time
from pathlib import Path

import torch

# Set environment variable to help with memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy
from lerobot.policies.pi0.configuration_pi0 import PI0Config

@torch.no_grad
def run_speed_test(policy_to_test, model_to_test, test_name, batch):
    """Run speed test on a policy model.
    
    Args:
        policy_to_test: The policy instance
        model_to_test: The model to test (could be compiled or non-compiled)
        test_name: Name for the test (for display)
        batch: Input batch for testing
        
    Returns:
        tuple: (actions, stats_dict)
    """
    print(f"\n=== Speed Testing {test_name} ===")

    policy_to_test.eval()
    
    # Create a copy of batch to avoid modifying original
    batch_copy = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    if policy_to_test.config.adapt_to_pi_aloha:
        batch_copy["observation.state"] = policy_to_test._pi_aloha_decode_state(batch_copy["observation.state"])

    batch_normalized = policy_to_test.normalize_inputs(batch_copy)
    images, img_masks = policy_to_test.prepare_images(batch_normalized)
    print(f"Number of images: {len(images)}")
    state = policy_to_test.prepare_state(batch_normalized)
    lang_tokens, lang_masks = policy_to_test.prepare_language(batch_normalized)

    # Clear GPU memory before testing
    torch.cuda.empty_cache()
    
    # Warmup
    print("Warming up...")
    for i in range(5):
        _ = model_to_test(
            images, img_masks, lang_tokens, lang_masks, state
        )
        # Clear memory after each warmup to prevent accumulation
        torch.cuda.empty_cache()
    
    # Reset action queue for clean timing
    policy_to_test.reset()
    
    individual_times = []
    
    # Time total execution
    torch.cuda.synchronize()
    total_start_time = time.time()
    
    # Reduce number of iterations to prevent OOM
    num_iterations = 50  # Reduced from 50 to manage memory
    
    for i in range(num_iterations):
        # Time individual calls
        torch.cuda.synchronize()
        start_time = time.time()
        
        actions = model_to_test(
            images, img_masks, lang_tokens, lang_masks, state
        )
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        individual_times.append(end_time - start_time)
        
        # Clear memory every few iterations to prevent accumulation
        torch.cuda.empty_cache()
        
        if i == 0:
            print(f"First call (queue fill): {individual_times[0]*1000:.2f} ms")
        elif i < 5:
            print(f"Call {i+1}: {individual_times[i]*1000:.2f} ms")
    
    torch.cuda.synchronize()
    total_end_time = time.time()
    
    total_time = total_end_time - total_start_time
    avg_time_per_call = total_time / num_iterations
    inference_time = individual_times[0]  # First call does inference
    avg_subsequent_time = sum(individual_times[1:]) / (num_iterations - 1) if num_iterations > 1 else 0
    
    print(f"\n=== {test_name} Timing Results ===")
    print(f"Total time for {num_iterations} calls: {total_time*1000:.2f} ms")
    print(f"Average time per call: {avg_time_per_call*1000:.2f} ms")
    print(f"Inference time (1st call): {inference_time*1000:.2f} ms")
    print(f"Subsequent calls (avg): {avg_subsequent_time*1000:.2f} ms")
    print(f"Throughput: {num_iterations/total_time:.1f} actions/second")
    if avg_subsequent_time > 0:
        print(f"Model inference overhead: {(inference_time - avg_subsequent_time)*1000:.2f} ms")
    
    # Clear memory after test
    torch.cuda.empty_cache()
    
    return actions, {
        'total_time': total_time,
        'inference_time': inference_time,
        'avg_subsequent_time': avg_subsequent_time,
        'throughput': num_iterations/total_time
    }


def compare_compiled_vs_noncompiled(policy, batch):
    """Compare performance between compiled and non-compiled policy models.
    
    Args:
        policy: The policy instance to test
        batch: Input batch for testing
        
    Returns:
        tuple: (compiled_policy, actions_original, actions_compiled, stats_original, stats_compiled)
    """
    print("=== Testing Non-Compiled Policy Speed ===")
    
    # Test non-compiled policy
    actions_original, stats_original = run_speed_test(policy, policy.model.sample_actions, "Non-Compiled Policy", batch)
    
    print("\n=== Compiling Policy with torch.compile ===")
    print("Compiling... (this may take a while)")
    
    # Compile the model
    compiled_model = torch.compile(policy.model.sample_actions, mode="reduce-overhead")
    
    print("Compilation finished!")
    
    # Test compiled policy  
    actions_compiled, stats_compiled = run_speed_test(policy, compiled_model, "Compiled Policy", batch)
    
    # Compare results
    print(f"\n=== Compilation Performance Comparison ===")
    speedup_total = stats_original['total_time'] / stats_compiled['total_time']
    speedup_inference = stats_original['inference_time'] / stats_compiled['inference_time']
    speedup_throughput = stats_compiled['throughput'] / stats_original['throughput']
    
    print(f"Total time speedup: {speedup_total:.2f}x")
    print(f"Inference speedup: {speedup_inference:.2f}x") 
    print(f"Throughput improvement: {speedup_throughput:.2f}x")
    if stats_original['avg_subsequent_time'] > 0 and stats_compiled['avg_subsequent_time'] > 0:
        print(f"Subsequent calls speedup: {stats_original['avg_subsequent_time']/stats_compiled['avg_subsequent_time']:.2f}x")
    
    # Verify outputs are the same
    actions_match = torch.allclose(actions_original, actions_compiled, atol=1e-6)
    print(f"Actions match between compiled/non-compiled: {actions_match}")
    
    return compiled_model, actions_original, actions_compiled, stats_original, stats_compiled


def main():
    """Standalone speed test for pi0 policy."""
    device = "cuda"
    
    dataset_repo_id = "lerobot/aloha_sim_transfer_cube_human"
    
    # Create a simple test batch
    batch = {
        "observation.images.top": torch.randn(1, 3, 224, 224, device=device),
        "observation.images.empty_camera_0": torch.randn(1, 3, 224, 224, device=device),
        "observation.images.empty_camera_1": torch.randn(1, 3, 224, 224, device=device),
        "observation.state": torch.randn(1, 14, device=device),
        "task": ["pick up the cube"]
    }
    
    # Load dataset metadata and policy
    dataset_meta = LeRobotDatasetMetadata(dataset_repo_id)
    
    cfg = PI0Config(
        n_action_steps=1,
        empty_cameras=2,
        #num_steps=5,
        resize_imgs_with_padding=(448, 448)
    )
    print(cfg)
    
    policy = make_policy(cfg, dataset_meta)
    
    # Run speed comparison
    compiled_model, actions_orig, actions_comp, stats_orig, stats_comp = compare_compiled_vs_noncompiled(
        policy, batch
    )
    
    print("\n=== Speed Test Complete ===")
    print(f"Best throughput: {max(stats_orig['throughput'], stats_comp['throughput']):.1f} actions/second")


if __name__ == "__main__":
    main() 