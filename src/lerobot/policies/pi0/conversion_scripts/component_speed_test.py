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
from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks


class ComponentTimer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str, device: str = "cuda"):
        self.name = name
        self.device = device
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.end_time = time.time()
        
    @property
    def elapsed_ms(self):
        if self.start_time is None or self.end_time is None:
            return 0
        return (self.end_time - self.start_time) * 1000


class ComponentBenchmark:
    """Class to benchmark individual components with and without compilation."""
    
    def __init__(self, policy):
        self.policy = policy
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Compiled component functions
        self.compiled_vision = None
        self.compiled_llm = None
        self.compiled_diffusion = None
        
    def compile_components(self):
        """Compile individual components for testing."""
        print("Compiling individual components...")
        
        # Create wrapper functions for compilation
        def vision_forward(images, lang_tokens):
            # Process images
            concatenated_images = torch.cat(images, dim=0)
            concatenated_img_embs = self.policy.model.paligemma_with_expert.embed_image(concatenated_images)
            concatenated_img_embs = concatenated_img_embs.to(dtype=torch.bfloat16)
            
            # Normalize image embeddings
            img_emb_dim = concatenated_img_embs.shape[-1]
            concatenated_img_embs = concatenated_img_embs * torch.tensor(
                img_emb_dim**0.5, dtype=concatenated_img_embs.dtype, device=concatenated_img_embs.device
            )
            
            # Process language tokens
            lang_emb = self.policy.model.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            lang_emb = lang_emb * (lang_emb_dim**0.5)
            
            return concatenated_img_embs, lang_emb
        
        def llm_forward(prefix_embs, prefix_att_2d_masks, prefix_position_ids):
            # Add head dimension to attention mask: [B, seq_len, seq_len] -> [B, 1, seq_len, seq_len]
            prefix_att_2d_masks_4d = prefix_att_2d_masks[:, None, :, :]
            _, past_key_values = self.policy.model.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=self.policy.config.use_cache,
                fill_kv_cache=True,
            )
            return past_key_values
        
        def diffusion_step(state, prefix_pad_masks, past_key_values, x_t, timestep):
            return self.policy.model.denoise_step(
                state, prefix_pad_masks, past_key_values, x_t, timestep
            )
        
        # Compile each component
        print("  Compiling vision tower...")
        self.compiled_vision = torch.compile(vision_forward, mode="reduce-overhead")
        
        print("  Compiling LLM processing...")
        self.compiled_llm = torch.compile(llm_forward, mode="reduce-overhead")
        
        print("  Compiling diffusion step...")
        self.compiled_diffusion = torch.compile(diffusion_step, mode="reduce-overhead")
        
        print("Compilation complete!")
    
    def benchmark_vision(self, images, lang_tokens, num_iterations=20, use_compiled=False):
        """Benchmark vision tower component."""
        times = []
        
        vision_func = self.compiled_vision if use_compiled else self._vision_forward
        
        # Warmup
        for _ in range(3):
            _ = vision_func(images, lang_tokens)
            torch.cuda.empty_cache()
        
        for _ in range(num_iterations):
            with ComponentTimer("Vision", self.device) as timer:
                _ = vision_func(images, lang_tokens)
            times.append(timer.elapsed_ms)
            
        return times
    
    def benchmark_llm(self, prefix_embs, prefix_att_2d_masks, prefix_position_ids, 
                      num_iterations=20, use_compiled=False):
        """Benchmark LLM processing component."""
        times = []
        
        llm_func = self.compiled_llm if use_compiled else self._llm_forward
        
        # Warmup
        for _ in range(3):
            _ = llm_func(prefix_embs, prefix_att_2d_masks, prefix_position_ids)
            torch.cuda.empty_cache()
        
        for _ in range(num_iterations):
            with ComponentTimer("LLM", self.device) as timer:
                _ = llm_func(prefix_embs, prefix_att_2d_masks, prefix_position_ids)
            times.append(timer.elapsed_ms)
            
        return times
    
    def benchmark_diffusion(self, state, prefix_pad_masks, past_key_values, actions_shape,
                           num_iterations=10, use_compiled=False):
        """Benchmark diffusion component."""
        times = []
        
        diffusion_func = self.compiled_diffusion if use_compiled else self.policy.model.denoise_step
        
        # Warmup
        for _ in range(2):
            noise = self.policy.model.sample_noise(actions_shape, self.device)
            expanded_time = torch.tensor(0.5, dtype=torch.float32, device=self.device).expand(actions_shape[0])
            _ = diffusion_func(state, prefix_pad_masks, past_key_values, noise, expanded_time)
            torch.cuda.empty_cache()
        
        for _ in range(num_iterations):
            noise = self.policy.model.sample_noise(actions_shape, self.device)
            expanded_time = torch.tensor(0.5, dtype=torch.float32, device=self.device).expand(actions_shape[0])
            
            with ComponentTimer("Diffusion", self.device) as timer:
                _ = diffusion_func(state, prefix_pad_masks, past_key_values, noise, expanded_time)
            times.append(timer.elapsed_ms)
            
        return times
    
    def _vision_forward(self, images, lang_tokens):
        """Non-compiled vision forward."""
        # Process images
        concatenated_images = torch.cat(images, dim=0)
        concatenated_img_embs = self.policy.model.paligemma_with_expert.embed_image(concatenated_images)
        concatenated_img_embs = concatenated_img_embs.to(dtype=torch.bfloat16)
        
        # Normalize image embeddings
        img_emb_dim = concatenated_img_embs.shape[-1]
        concatenated_img_embs = concatenated_img_embs * torch.tensor(
            img_emb_dim**0.5, dtype=concatenated_img_embs.dtype, device=concatenated_img_embs.device
        )
        
        # Process language tokens
        lang_emb = self.policy.model.paligemma_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * (lang_emb_dim**0.5)
        
        return concatenated_img_embs, lang_emb
    
    def _llm_forward(self, prefix_embs, prefix_att_2d_masks, prefix_position_ids):
        """Non-compiled LLM forward."""
        # Add head dimension to attention mask: [B, seq_len, seq_len] -> [B, 1, seq_len, seq_len]
        prefix_att_2d_masks_4d = prefix_att_2d_masks[:, None, :, :]
        _, past_key_values = self.policy.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.policy.config.use_cache,
            fill_kv_cache=True,
        )
        return past_key_values


@torch.no_grad
def test_component_compilation(policy, batch, num_iterations=15):
    """Test compilation benefits for each component."""
    print("\n" + "="*60)
    print("TORCH.COMPILE COMPONENT TESTING")
    print("="*60)
    
    policy.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Prepare inputs
    batch_copy = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    if policy.config.adapt_to_pi_aloha:
        batch_copy["observation.state"] = policy._pi_aloha_decode_state(batch_copy["observation.state"])

    batch_normalized = policy.normalize_inputs(batch_copy)
    images, img_masks = policy.prepare_images(batch_normalized)
    state = policy.prepare_state(batch_normalized)
    lang_tokens, lang_masks = policy.prepare_language(batch_normalized)
    
    bsize = state.shape[0]
    actions_shape = (bsize, policy.config.n_action_steps, policy.config.max_action_dim)
    
    # Set up benchmark
    benchmark = ComponentBenchmark(policy)
    
    # Prepare common inputs for LLM testing
    prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    
    # Get past_key_values for diffusion testing
    # Add head dimension to attention mask: [B, seq_len, seq_len] -> [B, 1, seq_len, seq_len]
    prefix_att_2d_masks_4d = prefix_att_2d_masks[:, None, :, :]
    _, past_key_values = policy.model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=policy.config.use_cache,
        fill_kv_cache=True,
    )
    
    results = {}
    
    # Test each component before compilation
    print("\nTesting non-compiled components...")
    
    print("  Vision tower...")
    vision_times_orig = benchmark.benchmark_vision(images, lang_tokens, num_iterations, use_compiled=False)
    
    print("  LLM processing...")
    llm_times_orig = benchmark.benchmark_llm(prefix_embs, prefix_att_2d_masks, prefix_position_ids, 
                                            num_iterations, use_compiled=False)
    
    print("  Diffusion step...")
    diffusion_times_orig = benchmark.benchmark_diffusion(state, prefix_pad_masks, past_key_values, 
                                                         actions_shape, num_iterations, use_compiled=False)
    
    # Compile components
    benchmark.compile_components()
    
    # Test compiled components
    print("\nTesting compiled components...")
    
    print("  Vision tower (compiled)...")
    vision_times_comp = benchmark.benchmark_vision(images, lang_tokens, num_iterations, use_compiled=True)
    
    print("  LLM processing (compiled)...")
    llm_times_comp = benchmark.benchmark_llm(prefix_embs, prefix_att_2d_masks, prefix_position_ids, 
                                            num_iterations, use_compiled=True)
    
    # print("  Diffusion step (compiled)...")
    # diffusion_times_comp = benchmark.benchmark_diffusion(state, prefix_pad_masks, past_key_values, 
    #                                                      actions_shape, num_iterations, use_compiled=True)
    
    # Calculate results
    def calc_stats(times):
        return {
            'mean': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
            'std': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5
        }
    
    results = {
        'vision': {
            'original': calc_stats(vision_times_orig),
            'compiled': calc_stats(vision_times_comp),
            'speedup': calc_stats(vision_times_orig)['mean'] / calc_stats(vision_times_comp)['mean']
        },
        'llm': {
            'original': calc_stats(llm_times_orig),
            'compiled': calc_stats(llm_times_comp),
            'speedup': calc_stats(llm_times_orig)['mean'] / calc_stats(llm_times_comp)['mean']
        },
        # 'diffusion': {
        #     'original': calc_stats(diffusion_times_orig),
        #     'compiled': calc_stats(diffusion_times_comp),
        #     'speedup': calc_stats(diffusion_times_orig)['mean'] / calc_stats(diffusion_times_comp)['mean']
        # }
    }
    
    return results


def print_compilation_results(results):
    """Print compilation speedup results for each component."""
    print("\n" + "="*60)
    print("TORCH.COMPILE SPEEDUP RESULTS")
    print("="*60)
    
    components = ['vision', 'llm', 'diffusion']
    component_names = ['Vision Tower', 'LLM Processing', 'Diffusion Step']
    
    print(f"\n{'Component':<15} {'Original (ms)':<15} {'Compiled (ms)':<15} {'Speedup':<10}")
    print("-" * 60)
    
    for comp, name in zip(components, component_names):
        orig_time = results[comp]['original']['mean']
        comp_time = results[comp]['compiled']['mean']
        speedup = results[comp]['speedup']
        
        print(f"{name:<15} {orig_time:<15.2f} {comp_time:<15.2f} {speedup:<10.2f}x")
    
    # Find best component for compilation
    best_component = max(components, key=lambda x: results[x]['speedup'])
    best_speedup = results[best_component]['speedup']
    
    print(f"\nBest compilation candidate: {component_names[components.index(best_component)]} ({best_speedup:.2f}x speedup)")
    
    # Calculate overall potential speedup if all components were compiled
    total_orig = sum(results[comp]['original']['mean'] for comp in components)
    total_comp = sum(results[comp]['compiled']['mean'] for comp in components)
    overall_speedup = total_orig / total_comp
    
    print(f"Overall potential speedup: {overall_speedup:.2f}x")


@torch.no_grad
def test_component_speeds(policy, batch, num_iterations=20):
    """Test the speed of individual components in the PI0 model.
    
    Args:
        policy: The PI0 policy instance
        batch: Input batch for testing
        num_iterations: Number of iterations to run for timing
        
    Returns:
        dict: Timing results for each component
    """
    print("=== Component Speed Testing ===")
    
    policy.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Prepare inputs
    batch_copy = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    if policy.config.adapt_to_pi_aloha:
        batch_copy["observation.state"] = policy._pi_aloha_decode_state(batch_copy["observation.state"])

    batch_normalized = policy.normalize_inputs(batch_copy)
    images, img_masks = policy.prepare_images(batch_normalized)
    state = policy.prepare_state(batch_normalized)
    lang_tokens, lang_masks = policy.prepare_language(batch_normalized)
    
    print(f"Number of images: {len(images)}")
    print(f"Batch size: {state.shape[0]}")
    print(f"Testing {num_iterations} iterations...")
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    # Warmup
    print("Warming up...")
    for _ in range(3):
        _ = policy.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state)
        torch.cuda.empty_cache()
    
    # Initialize timing storage
    vision_times = []
    llm_times = []
    diffusion_times = []
    total_times = []
    
    # Run timing tests
    for i in range(num_iterations):
        with ComponentTimer("Total", device) as total_timer:
            
            # 1. VISION TOWER TIMING
            with ComponentTimer("Vision Tower", device) as vision_timer:
                # Process all images through vision encoder
                if images:
                    bsize = images[0].shape[0]
                    concatenated_images = torch.cat(images, dim=0)
                    concatenated_img_embs = policy.model.paligemma_with_expert.embed_image(concatenated_images)
                    concatenated_img_embs = concatenated_img_embs.to(dtype=torch.bfloat16)
                    
                    # Normalize image embeddings
                    img_emb_dim = concatenated_img_embs.shape[-1]
                    concatenated_img_embs = concatenated_img_embs * torch.tensor(
                        img_emb_dim**0.5, dtype=concatenated_img_embs.dtype, device=concatenated_img_embs.device
                    )
                    
                    # Split back into per-image embeddings
                    img_embs = torch.split(concatenated_img_embs, bsize, dim=0)
                
                # Process language tokens
                lang_emb = policy.model.paligemma_with_expert.embed_language_tokens(lang_tokens)
                lang_emb_dim = lang_emb.shape[-1]
                lang_emb = lang_emb * (lang_emb_dim**0.5)
            
            vision_times.append(vision_timer.elapsed_ms)
            
            # 2. LLM TIMING (Prefix processing + KV cache)
            with ComponentTimer("LLM Processing", device) as llm_timer:
                prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
                    images, img_masks, lang_tokens, lang_masks
                )
                prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
                prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
                
                # Compute image and language key value cache
                # Add head dimension to attention mask: [B, seq_len, seq_len] -> [B, 1, seq_len, seq_len]
                prefix_att_2d_masks_4d = prefix_att_2d_masks[:, None, :, :]
                _, past_key_values = policy.model.paligemma_with_expert.forward(
                    attention_mask=prefix_att_2d_masks_4d,
                    position_ids=prefix_position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, None],
                    use_cache=policy.config.use_cache,
                    fill_kv_cache=True,
                )
            
            llm_times.append(llm_timer.elapsed_ms)
            
            # 3. DIFFUSION TIMING (Flow matching denoising)
            with ComponentTimer("Diffusion/Flow Matching", device) as diffusion_timer:
                # Sample initial noise
                actions_shape = (bsize, policy.config.n_action_steps, policy.config.max_action_dim)
                noise = policy.model.sample_noise(actions_shape, device)
                
                dt = -1.0 / policy.config.num_steps
                dt = torch.tensor(dt, dtype=torch.float32, device=device)
                
                x_t = noise
                time = torch.tensor(1.0, dtype=torch.float32, device=device)
                
                # Run denoising steps
                step_count = 0
                while time >= -dt / 2:
                    expanded_time = time.expand(bsize)
                    v_t = policy.model.denoise_step(
                        state,
                        prefix_pad_masks,
                        past_key_values,
                        x_t,
                        expanded_time,
                    )
                    
                    # Euler step
                    x_t += dt * v_t
                    time += dt
                    step_count += 1
            
            diffusion_times.append(diffusion_timer.elapsed_ms)
            
        total_times.append(total_timer.elapsed_ms)
        
        # Clear memory every few iterations
        if i % 5 == 0:
            torch.cuda.empty_cache()
        
        if i == 0:
            print(f"Iteration 1: Total={total_timer.elapsed_ms:.2f}ms, "
                  f"Vision={vision_timer.elapsed_ms:.2f}ms, "
                  f"LLM={llm_timer.elapsed_ms:.2f}ms, "
                  f"Diffusion={diffusion_timer.elapsed_ms:.2f}ms")
    
    # Calculate statistics
    def calculate_stats(times):
        return {
            'mean': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
            'std': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5
        }
    
    results = {
        'vision_tower': calculate_stats(vision_times),
        'llm_processing': calculate_stats(llm_times),
        'diffusion_flow': calculate_stats(diffusion_times),
        'total': calculate_stats(total_times),
        'num_diffusion_steps': step_count,
        'raw_times': {
            'vision': vision_times,
            'llm': llm_times,
            'diffusion': diffusion_times,
            'total': total_times
        }
    }
    
    return results


def print_component_results(results):
    """Print detailed timing results for each component."""
    print("\n" + "="*60)
    print("COMPONENT TIMING RESULTS")
    print("="*60)
    
    components = ['vision_tower', 'llm_processing', 'diffusion_flow', 'total']
    component_names = ['Vision Tower', 'LLM Processing', 'Diffusion/Flow', 'Total Pipeline']
    
    for comp, name in zip(components, component_names):
        stats = results[comp]
        print(f"\n{name}:")
        print(f"  Mean:  {stats['mean']:.2f} ms")
        print(f"  Min:   {stats['min']:.2f} ms")
        print(f"  Max:   {stats['max']:.2f} ms")
        print(f"  Std:   {stats['std']:.2f} ms")
    
    # Calculate percentages
    total_mean = results['total']['mean']
    print(f"\n" + "-"*40)
    print("COMPONENT BREAKDOWN (% of total time):")
    print("-"*40)
    
    vision_pct = (results['vision_tower']['mean'] / total_mean) * 100
    llm_pct = (results['llm_processing']['mean'] / total_mean) * 100
    diffusion_pct = (results['diffusion_flow']['mean'] / total_mean) * 100
    
    print(f"Vision Tower:     {vision_pct:5.1f}% ({results['vision_tower']['mean']:.1f} ms)")
    print(f"LLM Processing:   {llm_pct:5.1f}% ({results['llm_processing']['mean']:.1f} ms)")
    print(f"Diffusion/Flow:   {diffusion_pct:5.1f}% ({results['diffusion_flow']['mean']:.1f} ms)")
    print(f"Total:           100.0% ({total_mean:.1f} ms)")
    
    print(f"\nDiffusion steps per inference: {results['num_diffusion_steps']}")
    print(f"Throughput: {1000/total_mean:.1f} inferences/second")


def main():
    """Main function to run component speed tests."""
    device = "cuda"
    
    # Create test batch
    batch = {
        "observation.images.top": torch.randn(1, 3, 224, 224, device=device),
        "observation.images.empty_camera_0": torch.randn(1, 3, 224, 224, device=device),
        "observation.images.empty_camera_1": torch.randn(1, 3, 224, 224, device=device),
        "observation.state": torch.randn(1, 14, device=device),
        "task": ["pick up the cube"]
    }
    
    # Load policy
    dataset_repo_id = "lerobot/aloha_sim_transfer_cube_human"
    dataset_meta = LeRobotDatasetMetadata(dataset_repo_id)
    
    cfg = PI0Config(
        n_action_steps=1,
        empty_cameras=2,
        resize_imgs_with_padding=(448, 448)
    )
    
    policy = make_policy(cfg, dataset_meta)
    
    print(f"Testing PI0 component speeds with {cfg.num_steps} diffusion steps...")
    
    # Test individual components
    results = test_component_speeds(policy, batch, num_iterations=30)
    print_component_results(results)
    
    # Test compilation benefits
    compilation_results = test_component_compilation(policy, batch, num_iterations=15)
    print_compilation_results(compilation_results)

    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main() 