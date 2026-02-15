#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Evaluate a policy on an environment by running rollouts and computing metrics.
(SOMA-Enhanced Version: Robust Key Detection + Encore)
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
    # 如果没有这个文件，为了防止报错，可以定义简单的 Mock 类或者报错
    logging.error("Missing soma_control_flow.py! Advanced control disabled.")
    RollbackState = None
    TaskDecomposeState = None
    KeyStepRetryState = None
# ==============================================================================
# Helper Functions
# ==============================================================================
def _extract_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            return None
    return None

def find_visual_key(observation: dict) -> Optional[str]:
    """
    Helper to robustly find the primary image key in observation.
    Must ensure the value is actually a Tensor/Array, not a dictionary.
    """
    # 候选键名列表
    priority_keys = [
        "observation.images.image", 
        "observation.images.agentview_image", 
        "observation.image", 
        "pixels",
        "agent_image",
        "image"
    ]
    
    # 1. 优先检查标准键名，且必须包含有效数据
    for k in priority_keys:
        if k in observation:
            val = observation[k]
            # 排除字典，只接受 Tensor 或 Numpy
            if not isinstance(val, dict) and hasattr(val, "shape"):
                return k
    
    # 2. 兜底搜索：寻找任何看起来像图像的 4D/3D 张量
    for k, v in observation.items():
        # 排除字典
        if isinstance(v, dict): 
            continue
            
        # 检查是否是 Tensor/Numpy
        is_tensor = isinstance(v, torch.Tensor) or isinstance(v, np.ndarray)
        if not is_tensor:
            continue
            
        # 检查维度 (B,C,H,W) 或 (C,H,W)
        if v.ndim in [3, 4]:
            # 简单的名称过滤，防止选中 state
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
# Rollout Logic (Fixed SOMA Flow)
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

    # === [CTRL 1] 初始化高级控制器 ===
    # 只要 RollbackState 导入成功，就启用高级控制
    use_advanced_ctrl = (RollbackState is not None)
    rollback_ctrl = None
    task_ctrl = None
    retry_ctrl = None

    if use_advanced_ctrl:
        rollback_ctrl = RollbackState()
        task_ctrl = TaskDecomposeState()
        retry_ctrl = KeyStepRetryState()

    # 获取初始任务描述
    base_task_desc = task_prompt or extract_task_from_env(env, 0)
    current_task_desc = base_task_desc
    logging.info(f"Current Task: {current_task_desc}")

    # 初始关键帧保存逻辑 (Init Keyframe)
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

    # 2. [SOMA] Init
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
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(max_steps, desc=f"Rollout", disable=inside_slurm(), leave=False)
    
    # Action Accumulator (for Rollback calculation)
    action_dim = 7 
    accumulated_actions = np.zeros((env.num_envs, action_dim), dtype=np.float32)

    while not np.all(done) and step < max_steps:
        # Preprocess
        observation = preprocess_observation(observation)
        if return_observations: all_observations.append(deepcopy(observation))

        # === [CTRL 2] 动态更新任务描述 (Task Decomposition) ===
        if use_advanced_ctrl:
            # 如果有子任务队列，获取当前子任务；否则使用默认任务
            current_task_desc = task_ctrl.current_task(default_task=base_task_desc)

        # [功能] 每5帧保存一次关键帧 (Periodic Keyframe)
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

        # =========================================================
        # [SOMA] Perception & Control Flag Parsing
        # =========================================================
        # 注意：回退期间不触发 SOMA 感知
        is_rolling_back = (use_advanced_ctrl and rollback_ctrl.active)
        should_perceive = (step % 10 == 0) and (not is_rolling_back) and (soma_agent is not None)
        
        if should_perceive:
            vis_key, vis_val = find_visual_obs(observation)
            
            if vis_key is not None:
                try:
                    # Prepare Image
                    raw_tensor = vis_val[0] 
                    if raw_tensor.shape[0] in [1, 3]: raw_np = raw_tensor.permute(1, 2, 0).cpu().numpy()
                    else: raw_np = raw_tensor.cpu().numpy()
                    if raw_np.max() <= 1.05: raw_np = (raw_np * 255).astype(np.uint8)
                    else: raw_np = raw_np.astype(np.uint8)

                    # [SOMA Debug] Save Original
                    if soma_debug_dir:
                        soma_debug_dir.mkdir(parents=True, exist_ok=True)
                        current_ep_debug_dir = soma_debug_dir / f"episode_{n_episodes_rendered}"
                        current_ep_debug_dir.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(raw_np).save(current_ep_debug_dir / f"step_{step:03d}_original.jpg")

                    # Execute SOMA
                    processed_frame, refined_task_from_vlm, flags = soma_agent.step(
                        raw_np, current_task_desc, step, rag_context
                    )
                    
                    # [SOMA Debug] Save Modified
                    if soma_debug_dir and flags.get("image_modified", False):
                        Image.fromarray(processed_frame).save(current_ep_debug_dir / f"step_{step:03d}_modified.jpg")
                    
                    # Write back visual overlay
                    processed_tensor = torch.from_numpy(processed_frame).float() / 255.0
                    if processed_tensor.shape[-1] in [1, 3]: processed_tensor = processed_tensor.permute(2, 0, 1)
                    updated_batch = einops.repeat(processed_tensor, "c h w -> b c h w", b=env.num_envs).to(vis_val.device)
                    observation[vis_key] = updated_batch
                    
                    # 更新 Prompt (如果 VLM 给了 refined_task)
                    if refined_task_from_vlm:
                        # 仅当没有启用多阶段任务拆分时，才覆盖当前任务
                        # 如果启用了任务拆分，优先使用 task_ctrl 的逻辑
                        if not (use_advanced_ctrl and task_ctrl.subtasks):
                            current_task_desc = refined_task_from_vlm

                    # === [CTRL 3] 处理 SOMA Flags ===
                    if use_advanced_ctrl:
                        # A. 触发回退 (Encore)
                        if flags.get("encore", False):
                            rollback_ctrl.start_from_accumulated(
                                accumulated_actions, 
                                reverse_steps=40,   # CALVIN/LIBERO 推荐值
                                buffer_steps=20,    # 缓冲步数
                                reason="SOMA_Encore"
                            )
                        
                        # B. 更新任务链 (Subtasks)
                        if "subtasks" in flags:
                            task_ctrl.update_from_control_flags(flags)
                        
                        # C. 关键步重试参数
                        if "key_steps" in flags:
                            retry_ctrl.update_from_control_flags(flags)

                except Exception as e:
                    logging.error(f"[SOMA] Step Error: {e}", exc_info=True)

        # Inject Task & Preprocess
        observation = add_envs_task(env, observation, task_desc=current_task_desc)
        observation = preprocessor(observation)

        # =========================================================
        # [CTRL 4] Action Generation (Priority: Rollback > Policy)
        # =========================================================
        action_numpy = None
        
        if use_advanced_ctrl and rollback_ctrl.active:
            # Case A: 控制器接管 (倒车或缓冲)
            # step_action 会自动处理 Reverse -> Buffer -> Done 状态流转
            action_numpy = rollback_ctrl.step_action(env.num_envs)
            
            # 重要：回退期间清空累积动作，防止逻辑错误
            accumulated_actions.fill(0.0)
            
        else:
            # Case B: 正常 Policy 执行
            with torch.inference_mode():
                action = policy.select_action(observation)
            action = postprocessor(action)
            action_numpy = action.to("cpu").numpy()
            
            # 累积动作 (用于未来的回退计算)
            if action_numpy.shape[1] >= action_dim:
                accumulated_actions += action_numpy[:, :action_dim]

        # Step Environment
        observation, reward, terminated, truncated, info = env.step(action_numpy)
        if render_callback is not None: render_callback(env)

        # =========================================================
        # [CTRL 5] Post-Step Logic (Task Advance & Retry)
        # =========================================================
        if "final_info" in info:
            final_info = info["final_info"]
            successes = final_info["is_success"].tolist() if isinstance(final_info, dict) else [False]*env.num_envs
        else:
            successes = [False] * env.num_envs
        
        # 简化判定：任一环境成功即视为成功 (Eval 模式)
        is_success_any = any(successes)

        if use_advanced_ctrl:
            # A. 尝试推进子任务
            task_ctrl.maybe_advance(success_any=is_success_any)

            # B. 检查是否超时需要强制重试
            if retry_ctrl.should_trigger(step=step, success_any=is_success_any):
                retry_ctrl.mark_triggered()
                rollback_ctrl.start_from_accumulated(
                    accumulated_actions, 
                    reverse_steps=50, 
                    reason="Timeout_Retry"
                )

        done = terminated | truncated | done
        if step + 1 == max_steps: done = np.ones_like(done, dtype=bool)

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

    # [修改点 3] 确保根目录存在
    if max_episodes_rendered > 0 and videos_dir:
        (videos_dir / "frame_finish").mkdir(parents=True, exist_ok=True)
        (videos_dir / "keyframes").mkdir(parents=True, exist_ok=True)
        (videos_dir / "soma_debug").mkdir(parents=True, exist_ok=True)
        # (videos_dir / "keyframes_periodic").mkdir(parents=True, exist_ok=True) # 新增

    progbar = trange(n_batches, desc="Eval Batch", disable=inside_slurm())
    for batch_ix in progbar:
        if max_episodes_rendered > 0: ep_frames = []
        
        seeds = range(start_seed + batch_ix*env.num_envs, start_seed + (batch_ix+1)*env.num_envs) if start_seed else None
        
        # [修改点 4] 构建当前 Episode 的调试路径
        current_debug_dir = None
        current_periodic_dir = None
        
        if videos_dir and n_episodes_rendered < max_episodes_rendered:
            # SOMA 调试图路径
            current_debug_dir = videos_dir / "soma_debug"
            # 每5帧关键帧路径
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
            # 传递参数
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

        # Video Saving (保持上一轮的增强版逻辑)
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
def eval_one(env, policy, preprocessor, postprocessor, soma_agent, n_episodes, max_episodes_rendered, videos_dir, return_episode_data, start_seed, task_prompt, **kwargs):
    res = eval_policy(
        env, policy, preprocessor, postprocessor, soma_agent,
        n_episodes, max_episodes_rendered, videos_dir, return_episode_data, start_seed, task_prompt
    )
    return TaskMetrics(
        sum_rewards=[], max_rewards=[], 
        successes=[ep["success"] for ep in res["per_episode"]],
        video_paths=res.get("video_paths", []),
        episode_media=res.get("episode_media", [])
    )

def run_one(task_group, task_id, env, policy, preprocessor, postprocessor, soma_agent, n_episodes, max_episodes_rendered, videos_dir, return_episode_data, start_seed, **kwargs):
    task_videos_dir = videos_dir / f"{task_group}_{task_id}" if videos_dir else None
    
    task_prompt = f"{task_group}_{task_id}"
    try: task_prompt = env.call("task_description")[0]
    except: pass

    metrics = eval_one(
        env, policy, preprocessor, postprocessor, soma_agent,
        n_episodes, max_episodes_rendered, task_videos_dir, return_episode_data, start_seed, task_prompt
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

def eval_policy_all(envs, policy, preprocessor, postprocessor, soma_agent, n_episodes, max_episodes_rendered=0, videos_dir=None, return_episode_data=False, start_seed=None, max_parallel_tasks=1, **kwargs):
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
        videos_dir=videos_dir, return_episode_data=return_episode_data, start_seed=start_seed
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

    # 1. [SOMA] Init
    logging.info("Initializing SOMA Agent...")
    soma_config = {
        # "device": str(device),
        "sam3_base_url": "http://127.0.0.1:5001",  # 指向你的 sam3_service 地址
        "memory_dir": str(Path(cfg.experience_dir) / "experience_db"),
        # "vlm_api_key": "sk-dJ9PDHKGeP7xfsO4Zv7jNw", 
        # "vlm_base_url": "https://models.sjtu.edu.cn/api/v1"
        "vlm_api_key": "sk-c2649c021fd945c88ec8b11cdefebcb6",
        "vlm_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    }
    soma_agent = None
    if SOMAAgent:
        try: soma_agent = SOMAAgent(soma_config)
        except Exception as e: logging.error(f"SOMA Init Failed: {e}")
        
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
        max_parallel_tasks=cfg.env.max_parallel_tasks
    )
    
    if soma_agent is not None:
        soma_agent.wait_until_done()
        
    close_envs(envs)
    with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

if __name__ == "__main__":
    eval_main()