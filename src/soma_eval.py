#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
SOMA Evaluation Script
----------------------
Runs the SOMA Agent evaluation loop using components from `soma_core.py`.
"""

import concurrent.futures as cf
import json
import logging
import threading
import time
import sys
import os
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import einops
import gymnasium as gym
import numpy as np
import torch
from tqdm import trange

# LeRobot Imports
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env
from lerobot.envs.utils import (
    add_envs_task,
    close_envs,
    preprocess_observation,
)
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.utils.constants import ACTION
from lerobot.utils.io_utils import write_video
from lerobot.envs.libero import get_libero_dummy_action
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    inside_slurm,
)

# Optional NVTX
try:
    import nvtx
except ImportError:
    class NVTXModule:
        def annotate(self, *args, **kwargs): return nullcontext()
    nvtx = NVTXModule()

# === IMPORT SOMA CORE ===
# 假设 soma_core.py 在同一目录下
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from soma_core import (
    PerceptionModule, 
    MemoryBank, 
    EmbeddingEncoder, 
    ExperienceLogger
)

# ==============================================================================
# Configuration
# ==============================================================================
FILTER_ID = 5 
TASK_DESCRIPTION_1 = "pick up the ketchup and place it in the basket"
TASK_DESCRIPTION_2 = "pick up the tomato sauce and place it in the basket"

# ==============================================================================
# SOMA Rollout Logic
# ==============================================================================

def rollout(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    perception_module: PerceptionModule,  # From soma_core
    seeds: list[int] | None = None,
    return_observations: bool = False,
    render_callback = None,
    *,
    memory_bank: MemoryBank | None = None,
    embed_encoder: EmbeddingEncoder | None = None,
) -> dict:
    logging.info("="*60)
    logging.info(" STARTING SOMA EPISODE (Task Switch + Recovery)")
    logging.info("="*60)

    policy.reset()
    observation, info = env.reset(seed=seeds)
    if render_callback: render_callback(env)

    # RAG Retrieval (Initial)
    rag_results = None
    if memory_bank:
        # TODO: Implement actual embedding retrieval
        pass

    # Metrics
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []
    inference_times = []

    # SOMA State Variables
    done = np.array([False] * env.num_envs)
    max_steps = 500
    accumulated_actions = np.zeros((env.num_envs, 7), dtype=np.float32)
    reset_action_sequence = None
    reset_action_index = 0
    
    step = 0
    progbar = trange(max_steps, desc="SOMA Step", disable=inside_slurm(), leave=False)

    while not np.all(done) and step < max_steps:
        step_start_time = time.time()
        
        # --- Phase Logic ---
        is_reset_phase = (200 <= step < 325)
        current_task_desc = TASK_DESCRIPTION_1 if step < 325 else TASK_DESCRIPTION_2
            
        action_numpy = None
        
        if is_reset_phase:
            # === Phase 2: Open-Loop Recovery (Encore) ===
            preprocess_time = inference_time = postprocess_time = 0.0
            
            if step == 200:
                logging.info(f"[SOMA Encore] Triggering Physical Rollback at step {step}")
                reset_action_sequence = -accumulated_actions / 100.0
                reset_action_index = 0
                action_numpy = reset_action_sequence
            elif step < 300:
                action_numpy = reset_action_sequence
            elif step < 325:
                # Buffer period
                dummy = get_libero_dummy_action()
                action_numpy = np.array([dummy] * env.num_envs, dtype=np.float32)

        else:
            # === Phase 1 & 3: Closed-Loop Policy with SOMA Perception ===
            
            # 1. Preprocess
            t0 = time.time()
            with nvtx.annotate("preprocessing"):
                observation = preprocess_observation(observation)
            preprocess_time = time.time() - t0
            
            # 2. SOMA Perception / Brain Intervention
            # Only run every 10 steps to simulate "Thinking" latency and save compute
            use_soma = (step % 10 == 0)
            visual_key = next((k for k in observation.keys() if "image" in k or "rgb" in k), None)
            
            if use_soma and visual_key and perception_module:
                try:
                    # Tensor -> Numpy Image
                    raw_tensor = observation[visual_key][0]
                    img_np = raw_tensor.cpu().numpy().transpose(1, 2, 0)
                    img_np = (img_np * 255).astype(np.uint8)
                    
                    # CALL SOMA CORE
                    processed_img, soma_refined_task = perception_module.process_frame(
                        img_np, current_task_desc, step, rag_results
                    )
                    
                    # Apply Visual Prompting (Paint-to-Action)
                    proc_tensor = torch.from_numpy(processed_img.transpose(2, 0, 1)).float() / 255.0
                    proc_tensor = proc_tensor.to(observation[visual_key].device)
                    observation[visual_key] = einops.repeat(proc_tensor, "c h w -> b c h w", b=env.num_envs)
                    
                    # Apply Instruction Refinement
                    current_task_desc = soma_refined_task
                    
                except Exception as e:
                    logging.error(f"[SOMA Loop] Perception Error: {e}")

            # Inject Task
            observation = add_envs_task(env, observation, task_desc=current_task_desc)
            observation = preprocessor(observation)

            # 3. Policy Inference
            t0 = time.time()
            with torch.inference_mode(), nvtx.annotate("policy_inference"):
                action = policy.select_action(observation)
            inference_time = time.time() - t0
            
            # 4. Postprocess
            t0 = time.time()
            action = postprocessor(action)
            postprocess_time = time.time() - t0
            
            action_numpy = action.to("cpu").numpy()[:, :7]
            
            # Accumulate for rollback
            if step < 200:
                accumulated_actions += action_numpy

        # Step Env
        step_total_time = time.time() - step_start_time
        inference_times.append({
            "step": step,
            "total_ms": step_total_time * 1000
        })

        observation, reward, terminated, truncated, info = env.step(action_numpy)
        if render_callback: render_callback(env)

        # Record Data
        if "final_info" in info:
            successes = info["final_info"]["is_success"].tolist()
        else:
            successes = [False] * env.num_envs
            
        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)

        all_actions.append(torch.from_numpy(action_numpy))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))
        
        step += 1
        progbar.update()

    return {
        ACTION: torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
        "inference_timing": inference_times 
    }

# ==============================================================================
# Standard Eval Boilerplate
# ==============================================================================

def eval_policy(env, policy, preprocessor, postprocessor, perception_module, n_episodes, videos_dir, memory_bank=None, experience_logger=None, embed_encoder=None):
    # Standard evaluation loop (Simplified for brevity)
    policy.eval()
    video_paths = []
    
    # Simple Render Callback
    ep_frames = []
    def render_frame(env):
        if len(video_paths) < 10: # Limit render count
            if isinstance(env, gym.vector.SyncVectorEnv):
                ep_frames.append(np.stack([env.envs[i].render() for i in range(env.num_envs)]))
            else:
                ep_frames.append(np.stack(env.call("render")))

    # Run Batches
    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)
    for batch_ix in range(n_batches):
        ep_frames = []
        rollout_data = rollout(
            env, policy, preprocessor, postprocessor, perception_module,
            render_callback=render_frame,
            memory_bank=memory_bank,
            embed_encoder=embed_encoder
        )
        
        # Save Video logic...
        if ep_frames:
            videos_dir.mkdir(parents=True, exist_ok=True)
            v_path = videos_dir / f"soma_eval_{batch_ix}.mp4"
            video_paths.append(str(v_path))
            stacked = np.stack(ep_frames, axis=1)
            threading.Thread(target=write_video, args=(str(v_path), stacked[0], 30)).start()

    return {"video_paths": video_paths}

@parser.wrap()
def main(cfg: EvalPipelineConfig):
    init_logging()
    device = get_safe_torch_device(cfg.policy.device)
    
    # 1. Setup Env & Policy
    envs = make_env(cfg.env, n_envs=cfg.eval.batch_size)
    policy = make_policy(cfg.policy, cfg.env, cfg.rename_map)
    policy.eval()
    
    preprocessor, postprocessor = make_pre_post_processors(
        cfg.policy, cfg.policy.pretrained_path, 
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}, "rename_observations_processor": {"rename_map": cfg.rename_map}}
    )
    
    # 2. Initialize SOMA System
    logging.info("Initializing SOMA Agent from Core...")
    
    # 配置模型路径 (确保路径存在)
    vlm_path = "/mnt/disk1/shared_data/lzy/models/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
    sam3_path = "/mnt/disk1/shared_data/lzy/models/sam/sam3.pt"
    
    perception_module = PerceptionModule(
        device=str(device), 
        sam3_weight_path=sam3_path,
        vlm_model_path=vlm_path
    )
    
    memory_bank = MemoryBank(encoder=EmbeddingEncoder())
    
    # 3. Run Eval
    tasks = [(tg, tid, vec) for tg, group in envs.items() for tid, vec in group.items()]
    for task_group, task_id, env in tasks:
        if task_id != FILTER_ID: continue
        
        logging.info(f"Running SOMA Eval on Task {task_id}")
        videos_dir = Path(cfg.output_dir) / "videos" / f"{task_group}_{task_id}"
        
        eval_policy(
            env, policy, preprocessor, postprocessor, perception_module,
            n_episodes=cfg.eval.n_episodes,
            videos_dir=videos_dir,
            memory_bank=memory_bank
        )
    
    close_envs(envs)
    logging.info("SOMA Eval Complete.")

if __name__ == "__main__":
    main()