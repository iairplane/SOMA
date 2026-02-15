#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Evaluate a policy with Attention Map Visualization to demonstrate SOMA MCP Tool effects.
(SOMA-Enhanced Version: Attention Analysis + Robust Key Detection + Encore)
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
import types

import einops
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
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
    logging.error("Missing soma_control_flow.py! Advanced control disabled.")
    RollbackState = None
    TaskDecomposeState = None
    KeyStepRetryState = None

# ==============================================================================
# [SOMA] Attention Analysis Helper
# ==============================================================================
class AttentionRecorder:
    """
    增强版记录器：捕获 Vision Tower 所有层的 Attention 权重。
    """
    def __init__(self, model):
        self.layer_attentions = {} # {layer_idx: [tensor, ...]}
        self.hooks = []
        self.original_forwards = {}
        self.patched_modules = []
        self._register_monkey_patch(model)

    def _register_monkey_patch(self, model):
        """
        寻找 Vision Tower 的所有 Attention 层并按顺序 Hook。
        """
        # 1. 找到所有的 Attention 模块，并尝试按层级排序
        vision_attn_modules = []
        
        for name, module in model.named_modules():
            # 筛选条件：属于 Vision Tower 且是 Attention 层
            # 针对 Pi05/PaliGemma/SigLIP 的命名规则
            if ("vision" in name or "visual" in name) and ("attn" in name or "attention" in name):
                # 排除投影层，只找 Attention 主体
                if hasattr(module, "num_heads") or hasattr(module, "num_attention_heads"):
                    vision_attn_modules.append((name, module))
        
        # 简单排序，确保层序正确 (通常 named_modules 已经是拓扑序，但为了保险)
        # 假设名字里包含类似 layers.0, layers.1 ...
        
        logging.info(f"[AttnRecorder] Found {len(vision_attn_modules)} vision attention layers.")
        
        for idx, (name, module) in enumerate(vision_attn_modules):
            self._patch_forward(module, idx)
            self.patched_modules.append(module)

    def _patch_forward(self, module, layer_idx):
        original_forward = module.forward
        self.original_forwards[module] = original_forward

        def new_forward(*args, **kwargs):
            kwargs['output_attentions'] = True
            outputs = original_forward(*args, **kwargs)
            
            attn_weights = None
            if isinstance(outputs, tuple):
                for item in outputs:
                    if isinstance(item, torch.Tensor) and item.ndim == 4:
                        attn_weights = item
                        break
            
            if attn_weights is not None:
                # 存储时带上 layer_idx
                if layer_idx not in self.layer_attentions:
                    self.layer_attentions[layer_idx] = []
                self.layer_attentions[layer_idx].append(attn_weights.detach().cpu())
            
            return outputs

        module.forward = new_forward

    def clear(self):
        self.layer_attentions = {}

    def get_attention(self, layer_offset=-1):
        """
        获取指定层的 Attention。
        layer_offset: -1 表示最后一层，-2 表示倒数第二层，以此类推。
        """
        if not self.layer_attentions:
            return None
        
        total_layers = len(self.layer_attentions)
        # 计算实际索引
        if layer_offset < 0:
            target_idx = total_layers + layer_offset
        else:
            target_idx = layer_offset
            
        if target_idx in self.layer_attentions and self.layer_attentions[target_idx]:
            # 返回该层最新的一次推理结果
            return self.layer_attentions[target_idx][-1]
        return None

    def remove_hooks(self):
        for module in self.patched_modules:
            if module in self.original_forwards:
                module.forward = self.original_forwards[module]

def generate_heatmap(image_np, attn_tensor):
    """
    将 Attention Tensor (B, Heads, N, N) 转换为叠加在 image_np 上的热力图。
    """
    try:
        # === [关键修复] 强制转为 float32，解决 BFloat16 报错 ===
        if isinstance(attn_tensor, torch.Tensor):
            attn_tensor = attn_tensor.to(dtype=torch.float32).detach().cpu()
        
        # attn_tensor: [Batch, Heads, Seq_Len, Seq_Len]
        if attn_tensor.ndim == 4:
            attn_tensor = attn_tensor[0] # 取 Batch 0
        
        # Average across heads: [Seq_Len, Seq_Len]
        attn_avg = torch.mean(attn_tensor, dim=0)
        
        # Seq_Len = 1 (CLS) + H*W (Patches)
        seq_len = attn_avg.shape[-1]
        
        # 自动计算 grid size
        grid_size = int((seq_len - 1) ** 0.5) # 假设有 CLS
        if grid_size * grid_size != (seq_len - 1):
             # 这种情况下可能是没有 CLS token 的架构，或者是 register tokens
             grid_size = int(seq_len ** 0.5) 
             if grid_size * grid_size == seq_len:
                 cls_attn = torch.mean(attn_avg, dim=0) # 取行平均作为显著性
             else:
                 # 尺寸对不上，无法 reshape，放弃生成
                 return image_np
        else:
             # 标准 ViT: 取 CLS token 对其他 patch 的注意力
             cls_attn = attn_avg[0, 1:] 

        # Reshape [H, W]
        cls_attn = cls_attn.reshape(grid_size, grid_size)
        
        # Interpolate to image size [1, 1, H_img, W_img]
        cls_attn = F.interpolate(
            cls_attn.unsqueeze(0).unsqueeze(0), 
            size=(image_np.shape[0], image_np.shape[1]), 
            mode='bilinear',
            align_corners=False
        ).squeeze() # -> [H_img, W_img]
        
        # Normalize 0-1
        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
        
        
        # 转为 Numpy
        cls_attn_np = cls_attn.numpy()
        
        # Apply Colormap (JET) -> 变成红蓝热力图
        heatmap = cv2.applyColorMap(np.uint8(255 * cls_attn_np), cv2.COLORMAP_JET)
        heatmap = heatmap.astype(np.float32) / 255.0
        
        # 转换图像格式 (确保是 float32 0-1)
        image_float = image_np.astype(np.float32) / 255.0
        
        # Overlay: 0.5 heatmap + 0.5 original image
        cam = heatmap * 0.5 + image_float * 0.5
        cam = cam / np.max(cam) # 重新归一化防止溢出
        
        return np.uint8(255 * cam)

    except Exception as e:
        logging.warning(f"Heatmap generation failed: {e}")
        # 如果还是失败，返回原图以保证程序不崩
        return image_np

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
    priority_keys = ["observation.images.image", "observation.images.agentview_image", "observation.image", "pixels", "image"]
    for k in priority_keys:
        if k in observation:
            val = observation[k]
            if not isinstance(val, dict) and hasattr(val, "shape"): return k
    for k, v in observation.items():
        if isinstance(v, dict): continue
        if (isinstance(v, torch.Tensor) or isinstance(v, np.ndarray)) and v.ndim in [3, 4]:
            if "image" in k or "rgb" in k or "pixel" in k: return k
    return None

def find_visual_obs(obs):
    if obs is None or not isinstance(obs, dict): return None, None
    visual_keys = ["image", "agentview", "eye_in_hand", "rgb", "pixels"]
    for k, v in obs.items():
        if k is None: continue
        is_array = isinstance(v, (np.ndarray, torch.Tensor))
        if not is_array: continue
        shape = v.shape
        if len(shape) < 3: continue
        if min(shape[-2:]) < 16 and min(shape[0], shape[1]) < 16: continue 
        lower_k = k.lower()
        if any(vk in lower_k for vk in visual_keys): return k, v
    for k, v in obs.items():
        if isinstance(v, dict):
            sub_k, sub_val = find_visual_obs(v)
            if sub_k is not None: return f"{k}.{sub_k}", sub_val
    return None, None

def extract_task_from_env(env, env_idx=0) -> str:
    try:
        task_result = env.call("task_description")
        if isinstance(task_result, (list, tuple)):
            raw_task = task_result[env_idx] if len(task_result) > env_idx else ""
        else: raw_task = task_result
        if isinstance(raw_task, (tuple, list)) and len(raw_task) > 0:
            if not isinstance(raw_task, str): raw_task = raw_task[0]
        return str(raw_task).strip()
    except Exception: return ""

# ==============================================================================
# Rollout Logic (Enhanced with Attention Viz)
# ==============================================================================
def rollout(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    soma_agent: Optional[SOMAAgent],
    seeds: list[int] | None = None,
    return_observations: bool = False,
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None,
    *,
    task_prompt: str | None = None,
    keyframes_dir_init: Path | None = None,
    soma_debug_dir: Path | None = None,
    keyframes_dir_periodic: Path | None = None, 
    n_episodes_rendered: int = 0 
) -> dict:
    
    # 1. Reset
    policy.reset()
    observation, info = env.reset(seed=seeds)
    if render_callback is not None: render_callback(env)

    # === [CTRL] Init Advanced Controllers ===
    use_advanced_ctrl = (RollbackState is not None)
    rollback_ctrl = RollbackState() if use_advanced_ctrl else None
    task_ctrl = TaskDecomposeState() if use_advanced_ctrl else None
    retry_ctrl = KeyStepRetryState() if use_advanced_ctrl else None

    # === [ATTN] Init Attention Recorder ===
    # 只在第一个 episode 或调试目录下初始化，避免性能开销
    attn_recorder = None
    if soma_debug_dir:
        try:
            attn_recorder = AttentionRecorder(policy)
        except Exception as e:
            logging.warning(f"Failed to init attention recorder: {e}")

    base_task_desc = task_prompt or extract_task_from_env(env, 0)
    current_task_desc = base_task_desc
    logging.info(f"Current Task: {current_task_desc}")

    # Init Keyframe Save
    if keyframes_dir_init:
        # ... (Same as before) ...
        pass # 省略以节省篇幅

    # SOMA Init
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
        except Exception: pass

    # Loop Containers
    all_observations, all_actions, all_rewards, all_successes, all_dones = [], [], [], [], []
    step = 0
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(max_steps, desc=f"Rollout", disable=inside_slurm(), leave=False)
    
    action_dim = 7 
    accumulated_actions = np.zeros((env.num_envs, action_dim), dtype=np.float32)
    device = get_safe_torch_device(policy.config.device)

    while not np.all(done) and step < max_steps:
        observation = preprocess_observation(observation)
        if return_observations: all_observations.append(deepcopy(observation))

        if use_advanced_ctrl:
            current_task_desc = task_ctrl.current_task(default_task=base_task_desc)

        # ... (Periodic Keyframe Saving Omitted) ...

        # =========================================================
        # [SOMA] Perception & Attention Analysis
        # =========================================================
        is_rolling_back = (use_advanced_ctrl and rollback_ctrl.active)
        should_perceive = (step % 10 == 0) and (not is_rolling_back) and (soma_agent is not None)
        
        if should_perceive:
            vis_key, vis_val = find_visual_obs(observation)
            if vis_key is not None:
                try:
                    # Prepare Data
                    raw_tensor = vis_val[0] 
                    if raw_tensor.shape[0] in [1, 3]: raw_np = raw_tensor.permute(1, 2, 0).cpu().numpy()
                    else: raw_np = raw_tensor.cpu().numpy()
                    if raw_np.max() <= 1.05: raw_np = (raw_np * 255).astype(np.uint8)
                    else: raw_np = raw_np.astype(np.uint8)

                    # Execute SOMA
                    processed_frame, refined_task_from_vlm, flags = soma_agent.step(
                        raw_np, current_task_desc, step, rag_context
                    )
                    
                    # === [ATTN] Attention Visualization Logic ===
                    # 如果 SOMA 修改了图像 (flags['image_modified'])，我们进行 A/B 测试
                    if attn_recorder and flags.get("image_modified", False) and soma_debug_dir:
                        print(f"[SOMA] Step {step}: Image modified by SOMA. Capturing attention maps for comparison.")
                        try:
                            # 1. Construct Raw Obs Batch
                            obs_raw = deepcopy(observation)
                            # Ensure raw image is in tensor
                            raw_t = torch.from_numpy(raw_np).float() / 255.0
                            if raw_t.shape[-1] == 3: raw_t = raw_t.permute(2, 0, 1) # HWC->CHW
                            obs_raw[vis_key] = einops.repeat(raw_t, "c h w -> b c h w", b=env.num_envs).to(device)
                            obs_raw = add_envs_task(env, obs_raw, task_desc=current_task_desc)
                            obs_raw = preprocessor(obs_raw)

                            # 2. Run Policy on Raw (Capture Attn)
                            attn_recorder.clear()
                            with torch.inference_mode():
                                policy.select_action(obs_raw)
                            target_layer = -1
                            attn_raw = attn_recorder.get_attention(layer_offset=target_layer)
                            print(f"[SOMA] Captured attention from raw image at layer offset {target_layer}.")
                            # 3. Construct SOMA Obs Batch
                            obs_soma = deepcopy(observation)
                            soma_t = torch.from_numpy(processed_frame).float() / 255.0
                            if soma_t.shape[-1] == 3: soma_t = soma_t.permute(2, 0, 1)
                            obs_soma[vis_key] = einops.repeat(soma_t, "c h w -> b c h w", b=env.num_envs).to(device)
                            obs_soma = add_envs_task(env, obs_soma, task_desc=refined_task_from_vlm)
                            obs_soma = preprocessor(obs_soma)

                            # 4. Run Policy on SOMA (Capture Attn)
                            attn_recorder.clear()
                            with torch.inference_mode():
                                policy.select_action(obs_soma)
                            target_layer = -1
                            attn_soma = attn_recorder.get_attention(layer_offset=target_layer)
                            print(f"[SOMA] Captured attention from SOMA image at layer offset {target_layer}.")
                            
                            # 5. Generate Comparison
                            if attn_raw is not None and attn_soma is not None:
                                viz_raw = generate_heatmap(raw_np, attn_raw)
                                viz_soma = generate_heatmap(processed_frame, attn_soma)
                                
                                # Concat side-by-side
                                comparison = np.hstack([viz_raw, viz_soma])
                                
                                # Save
                                current_ep_debug_dir = soma_debug_dir / f"episode_{n_episodes_rendered}"
                                current_ep_debug_dir.mkdir(parents=True, exist_ok=True)
                                save_p = current_ep_debug_dir / f"attn_step_{step:03d}_compare.jpg"
                                Image.fromarray(comparison).save(save_p)
                                logging.info(f"Saved attention comparison to {save_p}")
                                
                        except Exception as e:
                            logging.error(f"Attention Viz Error: {e}")

                    # Write back modified image to observation for actual control
                    processed_tensor = torch.from_numpy(processed_frame).float() / 255.0
                    if processed_tensor.shape[-1] in [1, 3]: processed_tensor = processed_tensor.permute(2, 0, 1)
                    updated_batch = einops.repeat(processed_tensor, "c h w -> b c h w", b=env.num_envs).to(vis_val.device)
                    observation[vis_key] = updated_batch
                    
                    if refined_task_from_vlm and not (use_advanced_ctrl and task_ctrl.subtasks):
                        current_task_desc = refined_task_from_vlm

                    # [CTRL] Handle Flags
                    if use_advanced_ctrl:
                        if flags.get("encore", False):
                            rollback_ctrl.start_from_accumulated(accumulated_actions, reason="SOMA_Encore")
                        if "subtasks" in flags:
                            task_ctrl.update_from_control_flags(flags)
                        if "key_steps" in flags:
                            retry_ctrl.update_from_control_flags(flags)

                except Exception as e:
                    logging.error(f"[SOMA] Step Error: {e}")

        # Standard Policy Execution Step
        observation = add_envs_task(env, observation, task_desc=current_task_desc)
        observation = preprocessor(observation)

        action_numpy = None
        if use_advanced_ctrl and rollback_ctrl.active:
            action_numpy = rollback_ctrl.step_action(env.num_envs)
            accumulated_actions.fill(0.0)
        else:
            with torch.inference_mode():
                action = policy.select_action(observation)
            action = postprocessor(action)
            action_numpy = action.to("cpu").numpy()
            if action_numpy.shape[1] >= action_dim:
                accumulated_actions += action_numpy[:, :action_dim]

        observation, reward, terminated, truncated, info = env.step(action_numpy)
        if render_callback is not None: render_callback(env)

        # [CTRL] Post Step
        if "final_info" in info:
            final_info = info["final_info"]
            successes = final_info["is_success"].tolist() if isinstance(final_info, dict) else [False]*env.num_envs
        else:
            successes = [False] * env.num_envs
        
        if use_advanced_ctrl:
            is_success_any = any(successes)
            task_ctrl.maybe_advance(success_any=is_success_any)
            if retry_ctrl.should_trigger(step=step, success_any=is_success_any):
                retry_ctrl.mark_triggered()
                rollback_ctrl.start_from_accumulated(accumulated_actions, reason="Timeout_Retry")

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

    # Clean up hooks
    if attn_recorder:
        attn_recorder.remove_hooks()

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
# Eval Wrappers (Standard)
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

    if max_episodes_rendered > 0 and videos_dir:
        (videos_dir / "frame_finish").mkdir(parents=True, exist_ok=True)
        (videos_dir / "keyframes").mkdir(parents=True, exist_ok=True)
        (videos_dir / "soma_debug").mkdir(parents=True, exist_ok=True)

    progbar = trange(n_batches, desc="Eval Batch", disable=inside_slurm())
    for batch_ix in progbar:
        if max_episodes_rendered > 0: ep_frames = []
        
        seeds = range(start_seed + batch_ix*env.num_envs, start_seed + (batch_ix+1)*env.num_envs) if start_seed else None
        
        current_debug_dir = None
        current_periodic_dir = None
        if videos_dir and n_episodes_rendered < max_episodes_rendered:
            current_debug_dir = videos_dir / "soma_debug"
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
    metrics = eval_one(env, policy, preprocessor, postprocessor, soma_agent, n_episodes, max_episodes_rendered, task_videos_dir, return_episode_data, start_seed, task_prompt)
    if soma_agent:
        for item in metrics.get("episode_media", []):
            soma_agent.finish_episode(video_path=item["video_path"], keyframe_path=item.get("frame_path"), task_desc=task_prompt, success=item["success"])
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

    logging.info("Initializing SOMA Agent...")
    soma_config = {
        "sam3_base_url": "http://127.0.0.1:5001",
        "memory_dir": str(Path(cfg.experience_dir) / "experience_db"),
        # "vlm_api_key": "sk-dJ9PDHKGeP7xfsO4Zv7jNw", 
        # "vlm_base_url": "https://models.sjtu.edu.cn/api/v1",
        "vlm_api_key": "sk-c2649c021fd945c88ec8b11cdefebcb6",
        "vlm_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_id": "qwen3-vl-32b-instruct"
    }
    soma_agent = None
    if SOMAAgent:
        try: soma_agent = SOMAAgent(soma_config)
        except Exception as e: logging.error(f"SOMA Init Failed: {e}")
        
    logging.info("Making environment.")
    envs = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)
    
    logging.info("Making policy.")
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

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