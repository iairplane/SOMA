#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
SOMA-Enhanced LeRobot Evaluation Script
---------------------------------------
Integrates:
1. SOMA Agent (External Import)
2. Encore Mechanism (Physical Rollback)
3. Original LeRobot Evaluation Logic
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
# SOMA Integration (Import from external file)
# ==============================================================================
try:
    from soma_agent import SOMAAgent
except ImportError:
    raise ImportError("找不到 soma_agent.py。请确保 SOMA 组件已正确提取并放在同一目录下。")


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

# ==============================================================================
# Rollout Logic (Modified with Encore)
# ==============================================================================

def rollout(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    soma_agent: SOMAAgent,  # [SOMA] 传入 Agent
    seeds: list[int] | None = None,
    return_observations: bool = False,
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None,
    *,
    task_prompt: str | None = None,
    keyframes_dir_init: Path | None = None, # 保留参数兼容性
) -> dict:
    """
    SOMA-Enhanced Rollout:
    - Init: RAG Retrieval
    - Step: Perception (VLM+Tools) -> Policy -> Action -> Accumulate
    - Encore: If triggered, execute reverse trajectory.
    """
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    # 1. Reset Environment
    policy.reset()
    observation, info = env.reset(seed=seeds)
    if render_callback is not None:
        render_callback(env)

    # 2. [SOMA] Init - RAG Retrieval
    rag_context = {}
    try:
        # 查找视觉输入键名
        vis_key = next((k for k in observation.keys() if k in ["image", "pixels", "agent_image", "observation.images.image"]), None)
        # 兼容一些环境嵌套较深的情况
        if not vis_key:
             # 简单递归查找第一层
             for k, v in observation.items():
                 if isinstance(v, torch.Tensor) and v.ndim == 4: # (B, C, H, W)
                     vis_key = k
                     break

        if vis_key and soma_agent:
            # 取第一个环境的第一帧
            init_tensor = observation[vis_key][0] # (C, H, W)
            # Tensor -> Numpy (H, W, C) [0, 255]
            init_np = (init_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            # 调用 Agent
            rag_context = soma_agent.init_episode(init_np, task_prompt)
    except Exception as e:
        logging.warning(f"[SOMA] Init Failed: {e}")

    # 数据容器
    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []

    step = 0
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    
    progbar = trange(max_steps, desc=f"Rollout", disable=inside_slurm(), leave=False)
    
    # 提取任务描述 (Helper)
    def extract_task_from_env(env_idx=0):
        try:
            res = env.call("task_description")
            if isinstance(res, tuple): res = list(res)
            if len(res) > 0 and isinstance(res[0], (tuple, list)): 
                return res[0][env_idx]
            return res[env_idx] if len(res) > env_idx else ""
        except: return ""

    base_task_desc = task_prompt or extract_task_from_env()
    current_task_desc = base_task_desc

    # =========================================================
    # [SOMA] Encore 状态变量
    # =========================================================
    action_dim = 7 # 假设 (x, y, z, rx, ry, rz, gripper)
    accumulated_actions = np.zeros((env.num_envs, action_dim), dtype=np.float32)
    
    encore_active = False
    encore_counter = 0
    ENCORE_DURATION = 20 # 回滚持续步数
    reset_action_step = np.zeros((env.num_envs, action_dim), dtype=np.float32)

    while not np.all(done) and step < max_steps:
        # 1. 预处理
        observation = preprocess_observation(observation)
        if return_observations:
            all_observations.append(deepcopy(observation))

        # 2. [SOMA] 感知介入 (只在正常模式下每10步触发)
        should_perceive = (step % 10 == 0) and (not encore_active) and (soma_agent is not None)
        
        if should_perceive and vis_key:
            try:
                # 准备图像 (CPU Numpy)
                raw_tensor = observation[vis_key][0]
                raw_img = (raw_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                
                # === SOMA Step ===
                # 调用 Agent 的感知模块 (VLM + Tools)
                proc_img, refined_task, flags = soma_agent.step(
                    raw_img, current_task_desc, step, rag_context
                )
                
                # A. 写入修改后的图像 (Eraser/Paint)
                proc_tensor = torch.from_numpy(proc_img).permute(2, 0, 1).float() / 255.0
                proc_tensor = proc_tensor.to(raw_tensor.device)
                observation[vis_key] = einops.repeat(proc_tensor, "c h w -> b c h w", b=env.num_envs)
                
                # B. 更新任务描述 (Chaining)
                current_task_desc = refined_task
                
                # C. 处理 Encore 信号
                if flags.get("encore", False):
                    logging.warning(f"🔄 [Step {step}] SOMA Encore Triggered! Initiating physical rollback...")
                    encore_active = True
                    encore_counter = ENCORE_DURATION
                    # 计算反向步进: 总累积量 / 步数 的反方向
                    # 假设这是一个线性回归原点的过程
                    reset_action_step = -accumulated_actions / float(ENCORE_DURATION)
                    
                    # (可选) 可以在这里对 reset_action_step 做一些安全限制，防止动作过大
                    reset_action_step = np.clip(reset_action_step, -1.0, 1.0)

            except Exception as e:
                logging.error(f"[SOMA] Perception Error: {e}")

        # 3. 注入任务描述
        observation = add_envs_task(env, observation, task_desc=current_task_desc)
        observation = preprocessor(observation)

        # 4. 动作生成 (Policy vs Encore)
        action_numpy = None
        
        if encore_active and encore_counter > 0:
            # === Encore 模式: 执行物理回滚 ===
            action_numpy = reset_action_step
            # 确保 gripper 保持张开 (通常是 -1 或 1，视环境而定，这里假设 -1 是 open)
            if action_dim >= 7:
                action_numpy[:, 6] = -1.0 
            
            encore_counter -= 1
            if encore_counter <= 0:
                logging.info(f"✅ [Step {step}] Encore Recovery Complete. Resuming policy.")
                encore_active = False
                # 回滚完成后，理论上我们回到了某个之前的状态，累积量是否清零？
                # 简单起见，我们假设位置已重置，accumulated_actions 已经被加减抵消到接近0
        else:
            # === 正常模式: Policy 推理 ===
            with torch.inference_mode():
                action = policy.select_action(observation)
            action = postprocessor(action)
            action_numpy = action.to("cpu").numpy()
            
            # === [SOMA] 累积动作 (仅记录 Policy 的意图) ===
            # 注意: 只累积前7维 (Pose + Gripper)
            if action_numpy.shape[1] >= action_dim:
                accumulated_actions += action_numpy[:, :action_dim]

        # 5. 环境步进
        observation, reward, terminated, truncated, info = env.step(action_numpy)
        if render_callback is not None:
            render_callback(env)

        # 6. 数据记录
        if "final_info" in info:
            final_info = info["final_info"]
            successes = final_info["is_success"].tolist()
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
        running_success_rate = (
            einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").float().mean()
        )
        progbar.set_postfix({"success_rate": f"{running_success_rate.item() * 100:.1f}%"})
        progbar.update()

    # 结果打包
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

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()

    return ret

# ==============================================================================
# Evaluation Hierarchy (Pass SOMA Agent Down)
# ==============================================================================

# Typed dictionary for return
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
    soma_agent: SOMAAgent, # [SOMA] 参数
    n_episodes: int,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    task_prompt: str | None = None,
) -> dict:
    """Wrapper for batch rollouts."""
    if max_episodes_rendered > 0 and not videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")

    policy.eval()
    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    sum_rewards = []
    max_rewards = []
    all_successes = []
    episode_media_records = []
    video_paths = []
    
    n_episodes_rendered = 0

    # Rendering Callback
    ep_frames = []
    def render_frame(env: gym.vector.VectorEnv):
        if n_episodes_rendered >= max_episodes_rendered: return
        n_to_render = min(max_episodes_rendered - n_episodes_rendered, env.num_envs)
        if isinstance(env, gym.vector.SyncVectorEnv):
            ep_frames.append(np.stack([env.envs[i].render() for i in range(n_to_render)]))
        else:
            ep_frames.append(np.stack(env.call("render")[:n_to_render]))

    for batch_ix in trange(n_batches, desc="Eval Batch", disable=inside_slurm()):
        if max_episodes_rendered > 0: ep_frames = []
        
        seeds = range(start_seed + batch_ix*env.num_envs, start_seed + (batch_ix+1)*env.num_envs) if start_seed else None
        
        # Call Rollout
        rollout_data = rollout(
            env=env,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            soma_agent=soma_agent, # Pass Down
            seeds=list(seeds) if seeds else None,
            return_observations=return_episode_data,
            render_callback=render_frame if max_episodes_rendered > 0 else None,
            task_prompt=task_prompt
        )

        # Process Metrics (Original Logic)
        n_steps = rollout_data["done"].shape[1]
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()
        
        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
        sum_rewards.extend(batch_sum_rewards.tolist())
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
        max_rewards.extend(batch_max_rewards.tolist())
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
        all_successes.extend(batch_successes.tolist())

        # Save Videos & Log SOMA
        if max_episodes_rendered > 0 and len(ep_frames) > 0:
            stacked = np.stack(ep_frames, axis=1) # (B, T, H, W, C)
            batch_succ_list = [bool(x) for x in batch_successes.flatten().tolist()]
            
            for i, (frames, done_idx, succ) in enumerate(zip(stacked, done_indices.tolist(), batch_succ_list)):
                if n_episodes_rendered >= max_episodes_rendered: break
                
                videos_dir.mkdir(parents=True, exist_ok=True)
                v_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                k_path = videos_dir / f"eval_episode_{n_episodes_rendered}_key.jpg"
                video_paths.append(str(v_path))
                
                # Save Video
                threading.Thread(target=write_video, args=(str(v_path), frames[:done_idx+1], 30)).start()
                
                # Save Keyframe (Last frame)
                try:
                    Image.fromarray(frames[done_idx]).save(str(k_path))
                except: pass

                episode_media_records.append({
                    "episode_ix": n_episodes_rendered,
                    "success": succ,
                    "video_path": str(v_path),
                    "frame_path": str(k_path)
                })
                n_episodes_rendered += 1

    return {
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards)),
            "pc_success": float(np.nanmean(all_successes) * 100),
        },
        "per_episode": [{"success": s} for s in all_successes], # Simplified
        "video_paths": video_paths,
        "episode_media": episode_media_records
    }

def eval_one(env, policy, preprocessor, postprocessor, soma_agent, n_episodes, max_episodes_rendered, videos_dir, return_episode_data, start_seed, task_prompt):
    res = eval_policy(
        env, policy, preprocessor, postprocessor, soma_agent,
        n_episodes, max_episodes_rendered, videos_dir, return_episode_data, start_seed, task_prompt
    )
    # Convert to TaskMetrics format
    succs = [ep["success"] for ep in res["per_episode"]]
    return TaskMetrics(
        sum_rewards=[], # Simplified
        max_rewards=[], 
        successes=succs,
        video_paths=res.get("video_paths", []),
        episode_media=res.get("episode_media", [])
    )

def run_one(task_group, task_id, env, policy, preprocessor, postprocessor, soma_agent, n_episodes, max_episodes_rendered, videos_dir, return_episode_data, start_seed):
    task_videos_dir = videos_dir / f"{task_group}_{task_id}" if videos_dir else None
    
    # Infer task prompt
    task_prompt = f"{task_group}_{task_id}"
    try: task_prompt = env.call("task_description")[0][0]
    except: pass

    metrics = eval_one(
        env, policy, preprocessor, postprocessor, soma_agent,
        n_episodes, max_episodes_rendered, task_videos_dir, return_episode_data, start_seed, task_prompt
    )
    
    # [SOMA] Log to Experience DB
    if soma_agent:
        for item in metrics.get("episode_media", []):
            soma_agent.finish_episode(
                video_path=item["video_path"],
                keyframe_path=item.get("frame_path"),
                task_desc=task_prompt,
                success=item["success"]
            )
            
    return task_group, task_id, metrics

def eval_policy_all(envs, policy, preprocessor, postprocessor, soma_agent, n_episodes, max_episodes_rendered=0, videos_dir=None, return_episode_data=False, start_seed=None, max_parallel_tasks=1):
    tasks = [(tg, tid, vec) for tg, group in envs.items() for tid, vec in group.items()]
    
    runner = partial(
        run_one, policy=policy, preprocessor=preprocessor, postprocessor=postprocessor,
        soma_agent=soma_agent, n_episodes=n_episodes, max_episodes_rendered=max_episodes_rendered,
        videos_dir=videos_dir, return_episode_data=return_episode_data, start_seed=start_seed
    )

    results = []
    if max_parallel_tasks <= 1:
        for args in tasks:
            results.append(runner(*args))
    else:
        with cf.ThreadPoolExecutor(max_workers=max_parallel_tasks) as ex:
            futs = [ex.submit(runner, *args) for args in tasks]
            for f in cf.as_completed(futs):
                results.append(f.result())
                
    # Aggregate logic (simplified for brevity, original logic applies)
    return {"per_task": results}

# ==============================================================================
# Main
# ==============================================================================

@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    init_logging()
    device = get_safe_torch_device(cfg.policy.device)
    set_seed(cfg.seed)

    # 1. Init SOMA Agent
    logging.info("Initializing SOMA Agent...")
    # 请根据实际情况修改配置
    soma_config = {
        "device": str(device),
        "sam3_path": "/mnt/disk1/shared_data/lzy/models/sam/sam3.pt",
        "memory_dir": str(Path(cfg.output_dir) / "experience_db"),
        # "vlm_api_key": "YOUR_KEY", # 如果环境变量里没有
    }
    soma_agent = SOMAAgent(soma_config)

    # 2. Init Env & Policy
    envs = make_env(cfg.env, n_envs=cfg.eval.batch_size)
    policy = make_policy(cfg.policy, cfg.env, cfg.rename_map)
    policy.eval()
    
    preprocessor, postprocessor = make_pre_post_processors(
        cfg.policy, cfg.policy.pretrained_path, 
        {"device_processor": {"device": str(policy.config.device)}, "rename_observations_processor": {"rename_map": cfg.rename_map}}
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
    
    close_envs(envs)
    with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

if __name__ == "__main__":
    eval_main()