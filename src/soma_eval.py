#!/usr/bin/env python

"""SOMA Evaluation Script (Control-Flow MCP enabled)

- 视觉 MCP tools：通过 HTTP 调用独立 sam3_service（soma_tools.py / sam3_service.py）
- 控制流 MCP tools：encore / key_step_retry / task_decompose
  - 维护每步 action_hist
  - 需要回退时对指定区间动作求和并 reverse 执行（参考 lerobot_eval_single.py 的累计反向动作思想）

Memory/RAG：可选。此版本支持从 memory 读取历史统计（success_max_step、key_frame_range 等）。
"""

import json
import logging
import os
import sys
import threading
from contextlib import nullcontext
from pathlib import Path

import einops
import gymnasium as gym
import numpy as np
import torch
from tqdm import trange

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env
from lerobot.envs.utils import add_envs_task, close_envs, preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION
from lerobot.utils.io_utils import write_video
from lerobot.utils.utils import get_safe_torch_device, init_logging, inside_slurm

from lerobot.envs.libero import get_libero_dummy_action

# Optional NVTX
try:
    import nvtx
except ImportError:

    class NVTXModule:
        def annotate(self, *args, **kwargs):
            return nullcontext()

    nvtx = NVTXModule()

# === IMPORT SOMA MODULES ===
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from soma_control_flow import KeyStepRetryState, RollbackState, TaskDecomposeState
from soma_encoder import AdvancedEmbeddingEncoder
from soma_memory import MemoryBank, get_task_plan_defaults
from soma_perception import PerceptionModule

# ==============================================================================
# Configuration (demo defaults)
# ==============================================================================
FILTER_ID = 5
TASK_DESCRIPTION_1 = "pick up the ketchup and place it in the basket"
TASK_DESCRIPTION_2 = "pick up the tomato sauce and place it in the basket"


def _first_visual_key(obs: dict) -> str | None:
    return next((k for k in obs.keys() if "image" in k or "rgb" in k), None)


def _tensor_to_uint8_hwc(t: torch.Tensor) -> np.ndarray:
    img = t.detach().cpu().numpy().transpose(1, 2, 0)
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def _uint8_hwc_to_tensor_chw(img: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
    return t.to(device)


def _sum_actions(action_hist: list[np.ndarray], start: int, end_inclusive: int) -> np.ndarray:
    if not action_hist:
        raise ValueError("action_hist empty")
    start = max(0, start)
    end_inclusive = min(end_inclusive, len(action_hist) - 1)
    if end_inclusive < start:
        raise ValueError(f"invalid window {start}..{end_inclusive}")
    return np.sum(np.stack(action_hist[start : end_inclusive + 1], axis=0), axis=0)


def rollout(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    perception_module: PerceptionModule,
    seeds: list[int] | None = None,
    render_callback=None,
    *,
    memory_bank: MemoryBank | None = None,
    embed_encoder: AdvancedEmbeddingEncoder | None = None,
    max_steps: int = 500,
) -> dict:
    logging.info("=" * 60)
    logging.info(" STARTING SOMA EPISODE (Control-Flow MCP enabled)")
    logging.info("=" * 60)

    policy.reset()
    observation, info = env.reset(seed=seeds)
    if render_callback:
        render_callback(env)

    # ---- optional: read historical task_plan from memory (prototype) ----
    task_plan = get_task_plan_defaults()
    if memory_bank is not None:
        try:
            # prefer success partition
            task_plan = memory_bank.get_best_task_plan(partition="success")
        except Exception:
            task_plan = get_task_plan_defaults()

    key_frame_start = int(task_plan.get("key_frame_range", {}).get("start", 0))
    success_max_step = int(task_plan.get("success_max_step", 0))
    rollback_cfg = task_plan.get("rollback", {}) if isinstance(task_plan.get("rollback"), dict) else {}
    reverse_steps = int(rollback_cfg.get("reverse_steps", 100))
    buffer_steps = int(rollback_cfg.get("buffer_steps", 25))

    # ---- control-flow states ----
    rollback = RollbackState()
    key_retry = KeyStepRetryState()
    task_decompose = TaskDecomposeState()

    # ---- action history ----
    action_hist: list[np.ndarray] = []

    # per-subtask state: record where current subtask started (for "复位")
    subtask_start_step = 0
    pending_subtask_advance = False

    # ---- metrics ----
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []

    done = np.array([False] * env.num_envs)

    base_task = TASK_DESCRIPTION_1

    progbar = trange(max_steps, desc="SOMA Step", disable=inside_slurm(), leave=False)
    for step in progbar:
        if np.all(done):
            break

        current_task_desc = task_decompose.current_task(base_task)

        # ---- if rollback active, emit rollback action (skip policy) ----
        if rollback.active:
            action_numpy = rollback.step_action(env.num_envs)

            # after rollback finishes, if we were doing a subtask transition, commit it
            if (not rollback.active) and pending_subtask_advance:
                task_decompose.maybe_advance(success_any=True)
                pending_subtask_advance = False
                subtask_start_step = len(action_hist)  # next subtask starts from next policy action

        else:
            # ---- preprocess ----
            with nvtx.annotate("preprocessing"):
                observation = preprocess_observation(observation)

            # ---- SOMA perception every N steps ----
            use_soma = step % 10 == 0
            visual_key = _first_visual_key(observation)
            if use_soma and visual_key and perception_module:
                try:
                    raw_tensor = observation[visual_key][0]
                    img_np = _tensor_to_uint8_hwc(raw_tensor)

                    processed_img, refined_task, control_flags = perception_module.process_frame(
                        img_np, current_task_desc, step, rag_context=None
                    )

                    proc_tensor = _uint8_hwc_to_tensor_chw(processed_img, observation[visual_key].device)
                    observation[visual_key] = einops.repeat(proc_tensor, "c h w -> b c h w", b=env.num_envs)

                    if isinstance(refined_task, str) and refined_task:
                        current_task_desc = refined_task

                    if isinstance(control_flags, dict):
                        key_retry.update_from_control_flags(control_flags)
                        task_decompose.update_from_control_flags(control_flags)

                        # encore: immediate rollback (reverse all actions so far)
                        if control_flags.get("encore") is True and action_hist:
                            window_sum = _sum_actions(action_hist, 0, len(action_hist) - 1)
                            rollback.start_from_accumulated(
                                window_sum,
                                reverse_steps=reverse_steps,
                                buffer_steps=buffer_steps,
                                reason="encore",
                            )

                        # key_steps overrides from control_flags
                        ks = control_flags.get("key_steps")
                        if isinstance(ks, dict) and "start" in ks:
                            key_frame_start = int(ks.get("start", key_frame_start))

                except Exception as e:
                    logging.error(f"[SOMA Loop] Perception Error: {e}")

            # ---- inject task & preprocessors ----
            observation = add_envs_task(env, observation, task_desc=current_task_desc)
            observation = preprocessor(observation)

            # ---- policy inference ----
            with torch.inference_mode(), nvtx.annotate("policy_inference"):
                action = policy.select_action(observation)

            action = postprocessor(action)
            action_numpy = action.to("cpu").numpy()[:, :7].astype(np.float32)

            action_hist.append(action_numpy.copy())

        # ---- env step ----
        observation, reward, terminated, truncated, info = env.step(action_numpy)
        if render_callback:
            render_callback(env)

        # success extraction
        if "final_info" in info:
            successes = info["final_info"]["is_success"].tolist()
        else:
            successes = [False] * env.num_envs

        success_any = bool(np.any(successes))

        # ---- task_decompose: success -> 先复位，再推进 ----
        if success_any and task_decompose.subtasks and task_decompose.subtask_idx < len(task_decompose.subtasks) - 1:
            if not rollback.active and len(action_hist) > 0:
                # 复位区间：从 subtask_start_step 到当前 step 对应的最后一个 action（len(action_hist)-1）
                try:
                    window_sum = _sum_actions(action_hist, subtask_start_step, len(action_hist) - 1)
                    rollback.start_from_accumulated(
                        window_sum,
                        reverse_steps=reverse_steps,
                        buffer_steps=buffer_steps,
                        reason=f"task_decompose_reset[subtask_idx={task_decompose.subtask_idx}]",
                    )
                    pending_subtask_advance = True
                except Exception as e:
                    logging.error(f"[CTRL] subtask reset failed: {e}")
                    # fallback: still advance
                    task_decompose.maybe_advance(success_any=True)
                    subtask_start_step = len(action_hist)

        # ---- key_step_retry: 用历史 success_max_step 做阈值；回退到 key_frame_start ----
        if (
            (not rollback.active)
            and (not success_any)
            and success_max_step > 0
            and step >= success_max_step
            and len(action_hist) > 0
        ):
            try:
                window_sum = _sum_actions(action_hist, key_frame_start, len(action_hist) - 1)
                rollback.start_from_accumulated(
                    window_sum,
                    reverse_steps=reverse_steps,
                    buffer_steps=buffer_steps,
                    reason=f"key_step_retry[start={key_frame_start},now={len(action_hist)-1},max={success_max_step}]",
                )
            except Exception as e:
                logging.error(f"[CTRL] key_step_retry window sum failed: {e}")

        # ---- done ----
        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)

        all_actions.append(torch.from_numpy(action_numpy))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))

        progbar.set_postfix({"success_any": success_any, "rollback": rollback.active})

    return {
        ACTION: torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }


def eval_policy(env, policy, preprocessor, postprocessor, perception_module, n_episodes, videos_dir, memory_bank=None, embed_encoder=None):
    policy.eval()
    video_paths = []

    ep_frames = []

    def render_frame(env):
        if len(video_paths) < 10:
            if isinstance(env, gym.vector.SyncVectorEnv):
                ep_frames.append(np.stack([env.envs[i].render() for i in range(env.num_envs)]))
            else:
                ep_frames.append(np.stack(env.call("render")))

    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)
    for batch_ix in range(n_batches):
        ep_frames = []
        rollout_data = rollout(
            env,
            policy,
            preprocessor,
            postprocessor,
            perception_module,
            render_callback=render_frame,
            memory_bank=memory_bank,
            embed_encoder=embed_encoder,
        )

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

    envs = make_env(cfg.env, n_envs=cfg.eval.batch_size)
    policy = make_policy(cfg.policy, cfg.env, cfg.rename_map)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        cfg.policy,
        cfg.policy.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    sam3_url = os.environ.get("SOMA_SAM3_URL", "http://127.0.0.1:5001")
    perception_module = PerceptionModule(sam3_base_url=sam3_url)

    memory_bank = None
    embed_encoder = None
    if os.environ.get("SOMA_ENABLE_MEMORY", "0") == "1":
        memory_dir = Path(cfg.output_dir) / "soma_memory"
        memory_bank = MemoryBank(storage_dir=memory_dir)
        embed_encoder = AdvancedEmbeddingEncoder(device=str(device))

    tasks = [(tg, tid, vec) for tg, group in envs.items() for tid, vec in group.items()]
    for task_group, task_id, env in tasks:
        if task_id != FILTER_ID:
            continue

        logging.info(f"Running SOMA Eval on Task {task_id}")
        videos_dir = Path(cfg.output_dir) / "videos" / f"{task_group}_{task_id}"

        eval_policy(
            env,
            policy,
            preprocessor,
            postprocessor,
            perception_module,
            n_episodes=cfg.eval.n_episodes,
            videos_dir=videos_dir,
            memory_bank=memory_bank,
            embed_encoder=embed_encoder,
        )

    close_envs(envs)
    logging.info("SOMA Eval Complete.")


if __name__ == "__main__":
    main()
