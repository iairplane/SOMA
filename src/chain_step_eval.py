#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
SOMA: Self-Organizing Memory Agent
==================================

Description:
------------
Evaluate a policy on an environment using a Sequential Task Chain.
(SOMA-Enhanced Version: Task Chaining + Robust Key Detection + Encore)

This script is specifically designed to evaluate policies on long-horizon, 
multi-stage tasks. Instead of relying on the VLM to dynamically decompose 
tasks, it accepts a predefined sequence of subtasks (a `task_chain`). 

Key Features:
- Sequential Execution: Automatically transitions to the next subtask when 
  the current subtask reaches its maximum step limit.
- Reset/Encore Injection: Optionally triggers a physical arm reset or an 
  Encore rollback maneuver between subtask transitions to re-center the 
  agent and clear accumulated errors.
"""

import concurrent.futures as cf
import json
import logging
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from functools import partial
from pathlib import Path
from pprint import pformat
from typing import Any, TypedDict, Optional, Tuple, List

import einops
import gymnasium as gym
import numpy as np
import torch
from termcolor import colored
from torch import Tensor, nn
from tqdm import trange
import cv2
from PIL import Image
from gymnasium.vector import VectorEnv
import base64
import io
import sys
import os
import hashlib
from datetime import datetime

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env
from lerobot.envs.utils import (
    add_envs_task,
    check_env_attributes_and_types,
    close_envs,
    preprocess_observation,
)
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, DONE, OBS_STR, REWARD
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    inside_slurm,
)

# ==============================================================================
# [SOMA] Integration
# ==============================================================================
try:
    from soma_agent import SOMAAgent
except ImportError:
    logging.warning("Warning: 'soma_agent' module not found. SOMA features disabled.")
    SOMAAgent = None

try:
    from soma_control_flow import RollbackState, TaskDecomposeState, KeyStepRetryState
except ImportError:
    # If the file is missing, log an error and set variables to None to prevent crashing
    logging.error("Missing soma_control_flow.py! Advanced control disabled.")
    RollbackState = None
    TaskDecomposeState = None
    KeyStepRetryState = None

# ==============================================================================
# Helper Functions
# ==============================================================================
def apply_hard_arm_reset(env, target_qpos=None):
    """
    Rigorous Reset: Only resets the robotic arm's joint positions (qpos), 
    velocities (qvel), and control torques (ctrl).
    Object positions in the environment remain unchanged.
    """
    if target_qpos is None:
        # Reference standard init_qpos from MountedPanda.py
        target_qpos = np.array([0, -1.61037389e-01, 0.00, -2.44459747e00, 0.00, 2.22675220e00, np.pi / 4])

    # Loop through each sub-environment for VectorEnv
    num_envs = getattr(env, "num_envs", 1)
    
    for i in range(num_envs):
        # 1. Penetrate the Wrapper to get the actual underlying robosuite/LIBERO environment object
        try:
            # Depending on your environment nesting depth, you might need multiple .env calls
            raw_env = env.envs[i].env.env 
        except AttributeError:
            raw_env = env # If it is not a VectorEnv
            
        # 2. Force modification of joint positions (qpos)
        # _ref_joint_pos_indexes is the joint index automatically mapped by robosuite
        raw_env.sim.data.qpos[raw_env.robots[0]._ref_joint_pos_indexes] = target_qpos
        
        # 3. Force clear joint velocities (qvel) to prevent explosions caused by inertia
        raw_env.sim.data.qvel[raw_env.robots[0]._ref_joint_vel_indexes] = 0.0
        
        # 4. Clear the controller's current output torque (ctrl)
        raw_env.sim.data.ctrl[raw_env.robots[0]._ref_joint_actuator_indexes] = 0.0
        
        # 5. Reset gripper state (force open)
        raw_env.robots[0].gripper.set_gripper_pos([0.04, -0.04]) 
        
        # 6. Critical: Synchronize the physics engine state
        raw_env.sim.forward()
        
    logging.warning("[PHYSICS] Arm has been hard-reset to initial pose. Objects stayed put.")

def find_visual_key(observation: dict) -> Optional[str]:
    """
    Helper to robustly find the primary image key in observation.
    Must ensure the value is actually a Tensor/Array, not a dictionary.
    """
    # Candidate key name list
    priority_keys = [
        "observation.images.image", 
        "observation.images.agentview_image", 
        "observation.image", 
        "pixels",
        "agent_image",
        "image"
    ]
    
    # 1. Prioritize checking standard keys, and they must contain valid data
    for k in priority_keys:
        if k in observation:
            val = observation[k]
            # Exclude dictionaries, only accept Tensor or Numpy
            if not isinstance(val, dict) and hasattr(val, "shape"):
                return k
    
    # 2. Fallback search: look for any 4D/3D tensor that looks like an image
    for k, v in observation.items():
        # Exclude dictionaries
        if isinstance(v, dict): 
            continue
            
        # Check if it is a Tensor/Numpy array
        is_tensor = isinstance(v, torch.Tensor) or isinstance(v, np.ndarray)
        if not is_tensor:
            continue
            
        # Check dimensions (B,C,H,W) or (C,H,W)
        if v.ndim in [3, 4]:
            # Simple name filtering to prevent selecting state
            if "image" in k or "rgb" in k or "pixel" in k:
                return k
                
    return None

# ==============================================================================
# Helper Functions (Critical Fixes)
# ==============================================================================
def extract_task_from_env(env, env_idx=0) -> str:
    """
    Robust task description extraction.
    Fixes the bug where string was being converted to list of chars (e.g. 'p', 'i', 'c'...).
    """
    try:
        # env.call returns a list of results from all envs
        task_result = env.call("task_description")
        
        # Get the result for the specific env index
        if isinstance(task_result, (list, tuple)):
            if len(task_result) > env_idx:
                raw_task = task_result[env_idx]
            else:
                return ""
        else:
            raw_task = task_result

        # Handle nested tuple case sometimes returned by Libero wrappers
        if isinstance(raw_task, (tuple, list)) and len(raw_task) > 0:
            # Recursively unwrap if it's not a string
            if not isinstance(raw_task, str):
                raw_task = raw_task[0]

        return str(raw_task).strip()
    except Exception as e:
        logging.warning(f"Task extraction failed: {e}")
        return ""

def find_visual_obs(obs):
    """
    Recursively find the primary image in observation dict.
    READ-ONLY: Does not modify the input dictionary to avoid side-effects.
    Returns: (key_name, value_tensor_or_numpy)
    """
    if obs is None or not isinstance(obs, dict):
        return None, None
    
    # Priority keys
    visual_keys = ["image", "agentview", "eye_in_hand", "rgb", "pixels"]
    
    # 1. Direct search
    for k, v in obs.items():
        if k is None: continue
        
        # Check constraints: Must be array/tensor and have image-like shape
        is_array = isinstance(v, (np.ndarray, torch.Tensor))
        if not is_array: continue
        
        # Shape Check: Must have at least 3 dims and spatial dims > 16
        # Eliminates (1, 1, 256) vectors
        shape = v.shape
        if len(shape) < 3: continue
        if min(shape[-2:]) < 16 and min(shape[0], shape[1]) < 16: continue 

        lower_k = k.lower()
        if any(vk in lower_k for vk in visual_keys):
            return k, v

    # 2. Recursive search
    for k, v in obs.items():
        if isinstance(v, dict):
            sub_k, sub_val = find_visual_obs(v)
            if sub_k is not None:
                return f"{k}.{sub_k}", sub_val
                
    return None, None

# ==============================================================================
# Rollout Logic (Enhanced for Sequential Task Chain)
# ==============================================================================
def rollout(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    soma_agent: Optional[SOMAAgent],
    seeds: list[int] | None = None,
    return_observations: bool = False,
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None,
    *,
    task_prompt: str | None = None,
    task_chain: list[dict] | None = None, # <--- [New Parameter] Task Chain
    # Compatibility Args
    memory_bank: Any = None,
    keyframes_dir_init: Path | None = None,
    embed_encoder: Any = None,
    soma_debug_dir: Path | None = None,
    keyframes_dir_periodic: Path | None = None, 
    n_episodes_rendered: int = 0 
) -> dict:
    
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    # 1. Reset Env & Policy
    policy.reset()
    observation, info = env.reset(seed=seeds)
    if render_callback is not None:
        render_callback(env)

    # === [CTRL 1] Initialize Advanced Controllers ===
    use_advanced_ctrl = (RollbackState is not None)
    rollback_ctrl = RollbackState() if use_advanced_ctrl else None
    
    # Disable VLM's automatic task decomposition because we are manually hardcoding the task chain
    task_ctrl = None 
    retry_ctrl = KeyStepRetryState() if use_advanced_ctrl else None

    # === [NEW] Task Chain State Machine ===
    if not task_chain:
        # If no chain is provided, treat it as a single-task episode
        base_desc = task_prompt or extract_task_from_env(env, 0)
        task_chain = [{"desc": base_desc, "max_steps": env.call("_max_episode_steps")[0]}]

    current_task_idx = 0
    current_task_desc = task_chain[current_task_idx]["desc"]
    subtask_max_steps = task_chain[current_task_idx]["max_steps"]
    subtask_step_counter = 0 # Steps executed in the current subtask
    
    logging.info(f"Starting Task Chain [1/{len(task_chain)}]: '{current_task_desc}' (Max Steps: {subtask_max_steps})")
    
    # Initial keyframe saving logic (Init Keyframe)
    if keyframes_dir_init:
        try:
            vis_key, vis_val = find_visual_obs(observation) # CPU numpy state
            if vis_key is not None:
                if isinstance(vis_val, torch.Tensor): frame_data = vis_val[0].cpu().numpy()
                else: frame_data = vis_val[0]
                if frame_data.shape[0] in [1, 3]: frame_data = frame_data.transpose(1, 2, 0)
                if frame_data.max() <= 1.05: frame_data = (frame_data * 255).astype(np.uint8)
                else: frame_data = frame_data.astype(np.uint8)
                
                save_path = keyframes_dir_init / f"episode_init_seed{seeds[0] if seeds else 'x'}.png"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(frame_data).save(save_path)
        except Exception as e:
            logging.warning(f"Failed to save init keyframe: {e}")
            
            
    rag_context = {}
    vis_key, vis_val = find_visual_obs(observation)
    if vis_key and soma_agent:
        try:
            init_frame = vis_val[0]
            if isinstance(init_frame, torch.Tensor): init_frame = init_frame.cpu().numpy()
            if init_frame.shape[0] in [1, 3]: init_frame = init_frame.transpose(1, 2, 0)
            if init_frame.max() <= 1.05: init_frame = (init_frame * 255).astype(np.uint8)
            else: init_frame = init_frame.astype(np.uint8)
            rag_context = soma_agent.init_episode(init_frame, current_task_desc)
        except Exception as e:
            logging.warning(f"[SOMA] Init Failed: {e}")

    # Containers
    all_observations, all_actions, all_rewards, all_successes, all_dones = [], [], [], [], []
    step = 0
    done = np.array([False] * env.num_envs)
    global_max_steps = env.call("_max_episode_steps")[0] # Absolute maximum steps allowed by the environment
    progbar = trange(global_max_steps, desc=f"Rollout", disable=inside_slurm(), leave=False)
    
    # Action Accumulator (for Rollback calculation)
    action_dim = 7 
    accumulated_actions = np.zeros((env.num_envs, action_dim), dtype=np.float32)

    while not np.all(done) and step < global_max_steps:
        observation = preprocess_observation(observation)
        if return_observations: all_observations.append(deepcopy(observation))

        is_rolling_back = (use_advanced_ctrl and rollback_ctrl.active)
        
        if keyframes_dir_periodic and (step % 5 == 0):
            try:
                vis_key, vis_val = find_visual_obs(observation)
                if vis_key is not None:
                    if isinstance(vis_val, torch.Tensor): curr_frame = vis_val[0].cpu().numpy()
                    else: curr_frame = vis_val[0]
                    if curr_frame.shape[0] in [1, 3]: curr_frame = curr_frame.transpose(1, 2, 0)
                    if curr_frame.max() <= 1.05: curr_frame = (curr_frame * 255).astype(np.uint8)
                    else: curr_frame = curr_frame.astype(np.uint8)
                    
                    current_ep_keyframe_dir = keyframes_dir_periodic / f"episode_{n_episodes_rendered}"
                    current_ep_keyframe_dir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(curr_frame).save(current_ep_keyframe_dir / f"step_{step:04d}.jpg")
            except Exception as e:
                logging.warning(f"Failed to save periodic keyframe: {e}")
                
        # === [NEW] Task Switching Logic ===
        # Only check if a task switch is needed when not actively rolling back
        if not is_rolling_back:
            # Check if the current subtask has timed out
            if subtask_step_counter >= subtask_max_steps:
                logging.warning(f"Subtask '{current_task_desc}' reached max steps ({subtask_max_steps}).")
                
                current_task_idx += 1
                if current_task_idx < len(task_chain):
                    # --- Prepare to enter the next subtask ---
                    current_task_desc = task_chain[current_task_idx]["desc"]
                    subtask_max_steps = task_chain[current_task_idx]["max_steps"]
                    # subtask_step_counter = 0 # Counter reset happens below
                    
                    logging.info(f"Switching to Task [{current_task_idx+1}/{len(task_chain)}]: '{current_task_desc}'")
                    
                    # Trigger Encore (Rollback to initial position)
                    if use_advanced_ctrl:
                        logging.info("Triggering Full Rollback to starting position...")
                        # Set reverse_steps equal to all steps taken so far, effectively returning to origin
                        rollback_ctrl.start_from_accumulated(
                            accumulated_actions, 
                            reverse_steps=subtask_step_counter, # Reverse all steps
                            buffer_steps=15, 
                            reason="Task_Switch_Encore"
                        )
                    subtask_step_counter = 0
                    accumulated_actions.fill(0.0) # Reset accumulator
                else:
                    # All subtasks have timed out; end the entire Episode
                    logging.warning("All tasks in chain completed or timed out.")
                    done = np.ones_like(done, dtype=bool)
                    break
        
        # [SOMA] Perception (Maintains original logic, uses current_task_desc)
 
        # Inject Task & Preprocess
        # Note: Must inject the current_task_desc maintained by the state machine
        observation = add_envs_task(env, observation, task_desc=current_task_desc)
        observation = preprocessor(observation)

        # =========================================================
        # Action Generation
        # =========================================================
        action_numpy = None
        
        if use_advanced_ctrl and rollback_ctrl.active:
            # Currently reversing
            action_numpy = rollback_ctrl.step_action(env.num_envs)
            accumulated_actions.fill(0.0) # Do not accumulate during reverse phase
            # if rollback_ctrl.need_hard_reset:
            #     # Execute physical surgery: Reset arm state
            #     apply_hard_arm_reset(env)
            #     # Reset flag to prevent repeating reset in the next frame
            #     rollback_ctrl.need_hard_reset = False
                
            #     # [Important] Tell the policy model: The environment has changed, please re-perceive
            #     policy.reset() 
                
            #     # Skip the current frame's step, or proceed directly to the next loop iteration
            #     obs, _, _, _, _ = env.step(np.zeros_like(action_numpy)) 
            #     continue
        else:
            # Normal Execution
            with torch.inference_mode():
                action = policy.select_action(observation)
            action = postprocessor(action)
            action_numpy = action.to("cpu").numpy()
            
            # Accumulate actions
            if action_numpy.shape[1] >= action_dim:
                accumulated_actions += action_numpy[:, :action_dim]
            
            # Only increment the subtask counter during normal execution
            subtask_step_counter += 1

        # Step Environment
        observation, reward, terminated, truncated, info = env.step(action_numpy)
        if render_callback is not None: render_callback(env)

        # =========================================================
        # Post-Step Logic (Success Check)
        # =========================================================
        if "final_info" in info:
            final_info = info["final_info"]
            successes = final_info["is_success"].tolist() if isinstance(final_info, dict) else [False]*env.num_envs
        else:
            successes = [False] * env.num_envs
        
        is_success_any = any(successes)

        # What if the current subtask succeeds?
        # If you want to immediately advance to the next task upon success, add logic here.
        # By default, we continue attempting until max_steps is reached.

        done = terminated | truncated | done
        if step + 1 == global_max_steps: done = np.ones_like(done, dtype=bool)

        all_actions.append(torch.from_numpy(action_numpy))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))
        step += 1
        progbar.update()

    if return_observations:
        observation = preprocess_observation(observation)
        all_observations.append(deepcopy(observation))

    ret = {
        ACTION: torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }

    if return_observations:
        stacked_observations = {}
        for key in all_observations[0]:
            stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
        ret[OBS_STR] = stacked_observations
        
    if hasattr(policy, "use_original_modules"): policy.use_original_modules()
    return ret


# ==============================================================================
# Eval Wrappers
# ==============================================================================

class TaskMetrics(TypedDict):
    sum_rewards: list[float]
    max_rewards: list[float]
    successes: list[bool]
    video_paths: list[str]
    episode_media: list[dict]

ACC_KEYS = ("sum_rewards", "max_rewards", "successes", "video_paths")

def eval_policy(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    soma_agent: Optional[SOMAAgent],
    n_episodes: int,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    task_prompt: str | None = None,
    task_chain: list[dict] | None = None,
    **kwargs
) -> dict:
    
    if max_episodes_rendered > 0 and not videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")

    policy.eval()
    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    sum_rewards, max_rewards, all_successes, video_paths, episode_media_records = [], [], [], [], []
    n_episodes_rendered = 0
    ep_frames = []

    def render_frame(env: gym.vector.VectorEnv):
        if n_episodes_rendered >= max_episodes_rendered: return
        n_to_render = min(max_episodes_rendered - n_episodes_rendered, env.num_envs)
        if isinstance(env, gym.vector.SyncVectorEnv):
            ep_frames.append(np.stack([env.envs[i].render() for i in range(n_to_render)]))
        elif isinstance(env, gym.vector.AsyncVectorEnv):
            ep_frames.append(np.stack(env.call("render")[:n_to_render]))

    # Ensure root directory exists
    if max_episodes_rendered > 0 and videos_dir:
        (videos_dir / "frame_finish").mkdir(parents=True, exist_ok=True)
        (videos_dir / "keyframes").mkdir(parents=True, exist_ok=True)
        (videos_dir / "soma_debug").mkdir(parents=True, exist_ok=True)

    progbar = trange(n_batches, desc="Eval Batch", disable=inside_slurm())
    for batch_ix in progbar:
        if max_episodes_rendered > 0: ep_frames = []
        
        seeds = range(start_seed + batch_ix*env.num_envs, start_seed + (batch_ix+1)*env.num_envs) if start_seed else None
        
        # Construct debug path for the current Episode
        current_debug_dir = None
        current_periodic_dir = None
        
        if videos_dir and n_episodes_rendered < max_episodes_rendered:
            # SOMA debug image path
            current_debug_dir = videos_dir / "soma_debug"
            # Keyframe path every 5 frames
            current_periodic_dir = videos_dir / "keyframes"
            current_periodic_dir.mkdir(parents=True, exist_ok=True)

        rollout_data = rollout(
            env=env,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            soma_agent=soma_agent,
            seeds=list(seeds) if seeds else None,
            return_observations=return_episode_data,
            render_callback=render_frame if max_episodes_rendered > 0 else None,
            task_prompt=task_prompt,
            keyframes_dir_init=(videos_dir / "keyframes_init") if videos_dir else None,
            # Pass parameters
            task_chain=task_chain,
            soma_debug_dir=current_debug_dir,
            keyframes_dir_periodic=current_periodic_dir,
            n_episodes_rendered=n_episodes_rendered
        )

        n_steps = rollout_data["done"].shape[1]
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()
        
        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
        sum_rewards.extend(batch_sum_rewards.tolist())
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
        max_rewards.extend(batch_max_rewards.tolist())
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
        all_successes.extend(batch_successes.tolist())

        # Video Saving (Keeps the enhanced logic from the previous iteration)
        if max_episodes_rendered > 0 and len(ep_frames) > 0:
            stacked = np.stack(ep_frames, axis=1)
            batch_succ_list = [bool(x) for x in batch_successes.flatten().tolist()]
            for i, (frames, done_idx, succ_flag) in enumerate(zip(stacked, done_indices.tolist(), batch_succ_list)):
                if n_episodes_rendered >= max_episodes_rendered: break
                v_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                k_path = videos_dir / "frame_finish" / f"eval_episode_{n_episodes_rendered}_key.jpg"
                video_paths.append(str(v_path))
                threading.Thread(target=write_video, args=(str(v_path), frames[:done_idx+1], 30)).start()
                
                try:
                    Image.fromarray(frames[done_idx]).save(str(k_path))
                except Exception:
                    pass
                    
                episode_media_records.append({"episode_ix": n_episodes_rendered, "success": succ_flag, "video_path": str(v_path), "frame_path": str(k_path) if k_path else None})
                n_episodes_rendered += 1

        progbar.set_postfix({"success_rate": f"{np.mean(all_successes[:n_episodes]).item() * 100:.1f}%"})

    info = {
        "per_episode": [{"success": s} for s in all_successes],
        "aggregated": {"avg_sum_reward": float(np.nanmean(sum_rewards)), "pc_success": float(np.nanmean(all_successes) * 100)},
        "video_paths": video_paths,
        "episode_media": episode_media_records
    }
    return info
def eval_one(env, policy, preprocessor, postprocessor, soma_agent, n_episodes, max_episodes_rendered, videos_dir, return_episode_data, start_seed, task_prompt, task_chain=None, **kwargs):
    res = eval_policy(
        env, policy, preprocessor, postprocessor, soma_agent,
        n_episodes, max_episodes_rendered, videos_dir, return_episode_data, start_seed, task_prompt, task_chain=task_chain
    )
    return TaskMetrics(
        sum_rewards=[], max_rewards=[], 
        successes=[ep["success"] for ep in res["per_episode"]],
        video_paths=res.get("video_paths", []),
        episode_media=res.get("episode_media", [])
    )

def run_one(task_group, task_id, env, policy, preprocessor, postprocessor, soma_agent, n_episodes, max_episodes_rendered, videos_dir, return_episode_data, start_seed, task_chain=None, **kwargs):
    task_videos_dir = videos_dir / f"{task_group}_{task_id}" if videos_dir else None
    
    task_prompt = f"{task_group}_{task_id}"
    try: task_prompt = env.call("task_description")[0]
    except: pass

    metrics = eval_one(
        env, policy, preprocessor, postprocessor, soma_agent,
        n_episodes, max_episodes_rendered, task_videos_dir, return_episode_data, start_seed, task_prompt, task_chain=task_chain
    )
    
    if soma_agent:
        for item in metrics.get("episode_media", []):
            soma_agent.finish_episode(
                video_path=item["video_path"],
                keyframe_path=item.get("frame_path"),
                task_desc=task_prompt,
                success=item["success"]
            )
    return task_group, task_id, metrics

def eval_policy_all(envs, policy, preprocessor, postprocessor, soma_agent, n_episodes, max_episodes_rendered=0, videos_dir=None, return_episode_data=False, start_seed=None, max_parallel_tasks=1, task_chain=None, **kwargs):
    tasks = [(tg, tid, vec) for tg, group in envs.items() for tid, vec in group.items()]
    
    group_acc = defaultdict(lambda: defaultdict(list))
    overall = defaultdict(list)
    per_task = []

    def _accumulate(group, m):
        for k in ACC_KEYS:
            if m.get(k): 
                group_acc[group][k].extend(m[k])
                overall[k].extend(m[k])

    runner = partial(
        run_one, policy=policy, preprocessor=preprocessor, postprocessor=postprocessor,
        soma_agent=soma_agent, n_episodes=n_episodes, max_episodes_rendered=max_episodes_rendered,
        videos_dir=videos_dir, return_episode_data=return_episode_data, start_seed=start_seed, task_chain=task_chain, 
    )

    if max_parallel_tasks <= 1:
        for args in tasks:
            tg, tid, m = runner(*args)
            _accumulate(tg, m)
            per_task.append({"group": tg, "id": tid, "metrics": m})
    else:
        with cf.ThreadPoolExecutor(max_workers=max_parallel_tasks) as ex:
            futs = [ex.submit(runner, *args) for args in tasks]
            for f in cf.as_completed(futs):
                tg, tid, m = f.result()
                _accumulate(tg, m)
                per_task.append({"group": tg, "id": tid, "metrics": m})

    def _stats(d):
        return {
            "avg_reward": float(np.nanmean(d["sum_rewards"])) if d["sum_rewards"] else 0.0,
            "success_rate": float(np.nanmean(d["successes"])) * 100 if d["successes"] else 0.0
        }

    return {
        "per_task": per_task,
        "per_group": {g: _stats(d) for g, d in group_acc.items()},
        "overall": _stats(overall)
    }

# ==============================================================================
# Main
# ==============================================================================
@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    init_logging()
    device = get_safe_torch_device(cfg.policy.device)
    set_seed(cfg.seed)

    TASK_CHAIN = [
        {"desc": "Pick up the cream cheese and place it in the basket.", "max_steps": 280},
        {"desc": "Pick up the milk and place it in the basket.", "max_steps": 280},
        {"desc": "Pick up the chocolate pudding and place it on the plate.", "max_steps": 300},
        # {"desc": "Pick up the cream cheese and place it in the basket.", "max_steps": 300}
    ]
    logging.info(f"Loaded Task Chain with {len(TASK_CHAIN)} subtasks.")
    
    # 1. [SOMA] Init
    logging.info("Initializing SOMA Agent...")
    soma_config = {
        "sam3_base_url": "http://127.0.0.1:5001",  # Point to your sam3_service address
        "memory_dir": str(Path(cfg.experience_dir) / "experience_db"),
        "vlm_api_key": "sk-xxxxx",
        "vlm_base_url": "https://xxx.com/api/v1",
        "model_id": "xxxxx"
    }
    soma_agent = None
    if SOMAAgent:
        try: soma_agent = SOMAAgent(soma_config)
        except Exception as e: logging.error(f"SOMA Init Failed: {e}")
    soma_agent = None    
    
    # 2. Init LeRobot (Original Args)
    logging.info("Making environment.")
    envs = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)
    
    logging.info("Making policy.")
    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        rename_map=cfg.rename_map,
    )
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    # 3. Run Eval
    info = eval_policy_all(
        envs, policy, preprocessor, postprocessor, soma_agent,
        n_episodes=cfg.eval.n_episodes,
        max_episodes_rendered=10,
        videos_dir=Path(cfg.output_dir) / "videos",
        start_seed=cfg.seed,
        max_parallel_tasks=cfg.env.max_parallel_tasks,
        task_chain=TASK_CHAIN 
    )
    
    if soma_agent is not None:
        soma_agent.wait_until_done()
        
    close_envs(envs)
    with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

if __name__ == "__main__":
    eval_main()