#!/usr/bin/env python

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
"""Evaluate a policy on an environment by running rollouts and computing metrics.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
lerobot-eval \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.
```
lerobot-eval \
    --policy.path=outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

Note that in both examples, the repo/folder should contain at least `config.json` and `model.safetensors` files.

You can learn about the CLI options for this script in the `EvalPipelineConfig` in lerobot/configs/eval.py
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
import supervision as sv
# import gym  # 注意：避免与 gymnasium 冲突，使用 gymnasium 统一接口
from gymnasium.vector import VectorEnv
import base64
import io
import sys
import os
import hashlib
from datetime import datetime
try:
    # imageio.v2 is stable for imwrite
    import imageio.v2 as imageio
except Exception:
    imageio = None
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results

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
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

# =========================
# 辅助函数：提取 JSON
# =========================
def _extract_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except Exception:
        pass
    # 尝试提取 Markdown 代码块中的 JSON
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        snippet = text[start:end+1]
        try:
            return json.loads(snippet)
        except Exception:
            return None
    return None

# =========================
# RAG 相关类：VLM 客户端(API + 本地）、嵌入编码器、经验库
# =========================
class VLMClient:
    """VLM 客户端接口（策略模式）。"""

    def __init__(self) -> None:
        self.provider = "dummy"

    @classmethod
    def from_env(cls) -> "VLMClient":
        """
        工厂方法：优先尝试加载本地 Qwen3-VL 模型。
        如果加载失败（路径不存在、显存不足、依赖缺失），则回退到 API 客户端。
        """
        # 本地模型绝对路径 (请确保路径正确)
        local_model_path = "/mnt/disk1/shared_data/lzy/models/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
        
        # 1. 尝试加载本地模型
        try:
            if not os.path.exists(local_model_path):
                raise FileNotFoundError(f"本地模型路径不存在: {local_model_path}")
            
            logging.info(f"正在尝试加载本地模型: {local_model_path} ...")
            client = Qwen3VLLocalClient(
                model_name=local_model_path,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            logging.info("✅ 本地 Qwen3-VL 模型加载成功！")
            return client

        except Exception as e:
            logging.warning(f"⚠️ 本地 Qwen3VLLocalClient 加载失败，原因: {e}")
            logging.warning("🔄 正在切换到远端 API 模式 (Qwen3VLAPIClient)...")
            
            # 2. 回退到 API 客户端
            try:
                return Qwen3VLAPIClient()
            except Exception as e2:
                logging.error(f"❌ Qwen3VLAPIClient 初始化也失败了: {e2}")
                return cls() # 返回 Dummy

    def _generate(self, messages: list[dict], *, max_new_tokens: int = 1024) -> str:
        return "dummy response"

    # =================================================================
    #  核心业务逻辑：生成成功描述
    # =================================================================
    def generate_success_description(self, image_paths: list[str] | None, task_prompt: str) -> dict:
        if not image_paths:
            return {"visual_context_snapshot": {}, "execution_summary": "未提供有效图像，无法分析。"}
        
        # 过滤有效路径
        valid_paths = [p for p in image_paths if p and os.path.exists(p)]
        if not valid_paths:
            return {"visual_context_snapshot": {}, "execution_summary": "未提供有效图像，无法分析。"}

        # === 这里保留了你原本完整的 System Prompt ===
        system_prompt = (
            "Role: Robotics Perception Analyst. "
            "You will read multiple keyframes of a successful episode and produce a concise, structured JSON. "
            "Return STRICT JSON with keys: 'visual_context_snapshot' and 'execution_summary'. "
            "visual_context_snapshot should capture comfort-zone attributes such as target_object_color, "
            "background_color, lighting_condition, object_texture, spatial_layout. "
            "execution_summary is a short textual summary of why it succeeded."
        )
        user_prompt = (
            f"Task: {task_prompt}\n"
            "Given the chronological keyframes, extract the visual context snapshot describing the object's color/texture, "
            "background, lighting, and spatial layout, plus a one-sentence execution_summary. "
            "Output JSON only with the required keys."
        )
        
        # 构建通用消息格式
        content = [{"type": "text", "text": user_prompt}]
        for p in valid_paths:
            content.append({"type": "image_url", "image_url": {"url": p}})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        
        try:
            text = self._generate(messages, max_new_tokens=1024)
            data = _extract_json(text)
            print(f"[{self.provider}][success_desc][raw]: {text}", flush=True)
            
            if isinstance(data, dict) and ("visual_context_snapshot" in data or "execution_summary" in data):
                return {
                    "visual_context_snapshot": data.get("visual_context_snapshot", {}),
                    "execution_summary": data.get("execution_summary", ""),
                }
            return {"visual_context_snapshot": {}, "execution_summary": text.strip()}
        except Exception as e:
            logging.exception(f"{self.provider} 成功描述生成异常: {e}")
            return {"visual_context_snapshot": {}, "execution_summary": ""}

    # =================================================================
    #  核心业务逻辑：生成失败报告
    # =================================================================
    def generate_failure_report(self, image_paths: list[str] | None, task_prompt: str, anchor_example: dict | None = None) -> dict:
        if not image_paths:
            return {"intrinsic_analysis": {"direct_cause": "未提供有效图像，无法分析。", "observation": ""}}
        
        valid_paths = [p for p in image_paths if p and os.path.exists(p)]
        if not valid_paths:
            return {"intrinsic_analysis": {"direct_cause": "未提供有效图像，无法分析。", "observation": ""}}

        anchor_text = ""
        if anchor_example:
            anchor_text = (
                f"\n参考成功案例 (anchor): "
                f"task_signature={anchor_example.get('task_signature')}, "
                f"visual_context_snapshot={json.dumps(anchor_example.get('visual_context_snapshot', {}), ensure_ascii=False)}, "
                f"execution_summary={anchor_example.get('execution_summary', '')}"
            )

        # === 这里保留了你原本完整的 System Prompt ===
        system_prompt = (
            "Role: Robot Failure Analyst with Counterfactual Reasoning. "
            "You MUST return STRICT JSON with keys: "
            "'intrinsic_analysis', 'comparative_analysis', 'mcp_remediation'. "
            "intrinsic_analysis: direct_cause + observation describing what went wrong in these frames. "
            "comparative_analysis: if anchor provided, identify variable_gap {attribute, failed_value, success_value} "
            "and reasoning. If no anchor, leave variable_gap empty. "
            "mcp_remediation: recommend an MCP tool to fix the issue, choose from "
            "[semantic_recolor, enhance_contrast, remove_distractor], provide tool_name and parameters."
        )
        user_prompt = (
            f"Task: {task_prompt}\n"
            "You are given chronological keyframes of a failed episode."
            f"{anchor_text}\n"
            "请输出 JSON:\n"
            "{\n"
            '  "intrinsic_analysis": {"direct_cause": "...", "observation": "..."},\n'
            '  "comparative_analysis": {"reference_entry_id": "<id or empty>", "variable_gap": {"attribute": "...", "failed_value": "...", "success_value": "..."}, "reasoning": "..."},\n'
            '  "mcp_remediation": {"tool_name": "<semantic_recolor|enhance_contrast|remove_distractor>", "parameters": {...}}\n'
            "}"
        )

        content = [{"type": "text", "text": user_prompt}]
        for p in valid_paths:
            content.append({"type": "image_url", "image_url": {"url": p}})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        try:
            text = self._generate(messages, max_new_tokens=1024)
            data = _extract_json(text)
            print(f"[{self.provider}][failure_report][raw]: {text}", flush=True)

            if isinstance(data, dict) and (
                "intrinsic_analysis" in data
                or "comparative_analysis" in data
                or "mcp_remediation" in data
            ):
                return {
                    "intrinsic_analysis": data.get("intrinsic_analysis", {}),
                    "comparative_analysis": data.get("comparative_analysis", {}),
                    "mcp_remediation": data.get("mcp_remediation", {}),
                }
            return {
                "intrinsic_analysis": {"direct_cause": "free-text", "observation": text.strip()},
                "comparative_analysis": {},
                "mcp_remediation": {},
            }
        except Exception as e:
            logging.exception(f"{self.provider} 失败报告生成异常: {e}")
            return {"intrinsic_analysis": {}, "comparative_analysis": {}, "mcp_remediation": {}}


class Qwen3VLLocalClient(VLMClient):
    """
    本地运行 Qwen2-VL / Qwen2.5-VL (Qwen3) 模型的客户端。
    能够解析来自 API 格式的 Base64 图片消息。
    """
    
    def __init__(self, model_name: str, device: str = "cuda") -> None:
        super().__init__()
        self.provider = "qwen3-vl-local"
        self.device = device
        self.model_name = model_name
        self.model = None
        self.processor = None
        # 延迟加载，防止在初始化时就占用显存，只有在真正调用 generate 时才 check
        self._load_model() 
    
    def _load_model(self):
        """加载 HuggingFace 模型与处理器"""
        if self.model is not None:
            return
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            # 加载模型 (使用 bfloat16 或 float16)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
                device_map="auto", # 自动分配显存
                trust_remote_code=True
            )
            
            # 加载处理器
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, 
                min_pixels=256*28*28, 
                max_pixels=1280*28*28,
                trust_remote_code=True
            )
            
        except Exception as e:
            raise RuntimeError(f"Transformers 加载本地模型失败: {e}")

    def _base64_to_pil(self, base64_str: str) -> Image.Image:
        """解码 Base64 为 PIL Image"""
        if "base64," in base64_str:
            base64_str = base64_str.split("base64,")[1]
        image_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_data)).convert("RGB")

    def _process_messages_for_local(self, messages: list[dict]):
        """
        将 API 格式的消息 (包含 base64 image_url) 转换为 
        Qwen2-VL processor 能够接受的格式 (text + PIL image list)。
        """
        text_content = ""
        images = []
        
        # Qwen2-VL 的 Chat Template 格式
        # 实际上 Qwen2-VL Processor 支持直接处理包含 {"type": "image", "image": pil_img} 的 messages
        # 这里我们手动构建符合 processor.apply_chat_template 预期的结构
        
        qwen_messages = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            new_content = []
            
            if isinstance(content, str):
                new_content.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for item in content:
                    if item["type"] == "text":
                        new_content.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "image_url":
                        # 核心转换：Base64 URL -> PIL Image
                        url = item["image_url"]["url"]
                        if url.startswith("data:image"):
                            pil_img = self._base64_to_pil(url)
                            new_content.append({"type": "image", "image": pil_img})
                        else:
                            # 假设是本地路径
                            new_content.append({"type": "image", "image": url})
            
            qwen_messages.append({"role": role, "content": new_content})

        return qwen_messages

    def _generate(self, messages: list[dict], *, max_new_tokens: int = 1024) -> str:
        """本地推理入口"""
        self._load_model()
        
        # 1. 转换消息格式 (适配 qwen_vl_utils.process_vision_info)
        qwen_messages = self._process_messages_for_local(messages)
        
        # 2. 准备推理输入
        # Qwen2-VL 推荐的处理方式
        text = self.processor.apply_chat_template(
            qwen_messages, tokenize=False, add_generation_prompt=True
        )
        
        # 提取图片对象列表
        image_inputs = []
        for msg in qwen_messages:
            for item in msg["content"]:
                if item["type"] == "image":
                    image_inputs.append(item["image"])
        
        # 如果有 qwen_vl_utils，可以用 process_vision_info (可选，这里用手动提取更稳健兼容旧代码)
        # from qwen_vl_utils import process_vision_info
        # image_inputs, video_inputs = process_vision_info(qwen_messages)

        # 3. 处理输入
        inputs = self.processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # 4. 生成
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,    # 低温，保证指令遵循
                top_p=0.9,
                do_sample=False     # 确定性输出对 json 生成更友好
            )
        
        # 5. 解码 (去掉输入部分的 tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        print(f"[Qwen-VL-Local] Output: {output_text[:100]}...", flush=True)
        return output_text


class Qwen3VLAPIClient(VLMClient):
    """
    使用兼容 OpenAI 格式的 API 进行远程调用的客户端（备用）。
    适用于: Qwen2.5-VL, Qwen-VL-Max 等 API 服务。
    """
    
    # 请根据实际情况修改 MODEL_ID 和 KEY
    MODEL_ID = "qwen3vl" # 或 "qwen-vl-plus", "qwen2.5-vl-72b-instruct"
    API_KEY = "sk-dJ9PDHKGeP7xfsO4Zv7jNw" # 您的 API Key
    BASE_URL = "https://models.sjtu.edu.cn/api/v1" # 您的 Base URL
    
    def __init__(self) -> None:
        super().__init__()
        self.provider = "qwen3-vl-api"
        self.client = None
        self._ensure_loaded()
    
    def _ensure_loaded(self):
        if self.client is not None:
            return
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.API_KEY,
                base_url=self.BASE_URL,
            )
            logging.info(f"Qwen3-VL API 客户端初始化成功: {self.BASE_URL}")
        except ImportError as e:
            raise RuntimeError("OpenAI SDK 未安装。请安装: pip install openai") from e
        except Exception as e:
            # 这里抛出错误，以便 VLMClient.from_env 能够捕获并处理
            raise RuntimeError(f"Qwen3-VL API 客户端初始化失败: {e}") from e
    
    @staticmethod
    def _image_to_base64(image_path: str) -> str:
        """将图像文件转换为 Base64 编码的 Data URL。"""
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
            base64_str = base64.b64encode(image_data).decode("utf-8")
            ext = Path(image_path).suffix.lower()
            mime_type = "image/png" if ext == ".png" else "image/jpeg"
            return f"data:{mime_type};base64,{base64_str}"
        except Exception as e:
            logging.exception(f"图像转 Base64 失败: {image_path}: {e}")
            raise
    
    def _generate(self, messages: list[dict], *, max_new_tokens: int = 1024) -> str:
        """调用 API 生成响应。"""
        self._ensure_loaded()
        try:
            completion = self.client.chat.completions.create(
                model=self.MODEL_ID,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0.01, # 降低随机性
            )
            text = completion.choices[0].message.content
            # print(f"[Qwen-API][raw] {text}", flush=True)
            return text
        except Exception as e:
            logging.exception(f"Qwen3-VL API 调用失败: {e}")
            raise



class EmbeddingEncoder:
    """轻量级嵌入编码器：用首关键帧 + 任务文本生成可用于匹配的向量。"""

    def __init__(self, hash_len: int = 8):
        self.hash_len = hash_len

    def embed(self, image_path: str, text: str) -> list[float]:
        img = Image.open(image_path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        rgb_mean = arr.mean(axis=(0, 1))
        h = hashlib.sha256(text.encode("utf-8")).digest()
        h_vals = np.frombuffer(h, dtype=np.uint8)[: self.hash_len].astype(np.float32) / 255.0
        vec = np.concatenate([rgb_mean, h_vals])
        norm = float(np.linalg.norm(vec) + 1e-8)
        vec = vec / norm
        return vec.astype(float).tolist()


class MemoryBank:
    """简单的基于 JSONL 的检索器，使用 EmbeddingEncoder 的向量做余弦相似度。"""

    def __init__(self, global_outputs_dir: Path | None = None, encoder: EmbeddingEncoder | None = None) -> None:
        global_outputs_dir = "/mnt/disk1/shared_data/lzy/lerobot/eval_logs"
        if global_outputs_dir is None:
            current_file = Path(__file__).resolve()
            lerobot_root = current_file.parent.parent.parent.parent
            global_outputs_dir = lerobot_root / "outputs"
        self.global_outputs_dir = Path(global_outputs_dir)
        self.global_outputs_dir.mkdir(parents=True, exist_ok=True)
        self.encoder = encoder or EmbeddingEncoder()
        self.success_records = self._load_jsonl(self.global_outputs_dir / "success.jsonl")
        self.failure_records = self._load_jsonl(self.global_outputs_dir / "failure.jsonl")

    def _load_jsonl(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        records: list[dict] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        continue
        except Exception as e:
            logging.exception(f"加载经验库失败: {path}: {e}")
        return records

    @staticmethod
    def merge_local_to_global(local_experience_dir: Path, global_outputs_dir: Path | None = None) -> None:
        if global_outputs_dir is None:
            current_file = Path(__file__).resolve()
            lerobot_root = current_file.parent.parent.parent.parent
            global_outputs_dir = lerobot_root / "outputs"
        global_outputs_dir = Path(global_outputs_dir)
        global_outputs_dir.mkdir(parents=True, exist_ok=True)
        
        local_experience_dir = Path(local_experience_dir)
        local_success = local_experience_dir / "success.jsonl"
        local_failure = local_experience_dir / "failure.jsonl"
        global_success = global_outputs_dir / "success.jsonl"
        global_failure = global_outputs_dir / "failure.jsonl"
        
        if local_success.exists():
            try:
                with local_success.open("r", encoding="utf-8") as f_in:
                    with global_success.open("a", encoding="utf-8") as f_out:
                        for line in f_in:
                            line = line.strip()
                            if line:
                                f_out.write(line + "\n")
                # logging.info(f"已将本地成功经验追加到全局文件: {global_success}")
            except Exception as e:
                logging.exception(f"追加成功经验到全局文件失败: {e}")
        
        if local_failure.exists():
            try:
                with local_failure.open("r", encoding="utf-8") as f_in:
                    with global_failure.open("a", encoding="utf-8") as f_out:
                        for line in f_in:
                            line = line.strip()
                            if line:
                                f_out.write(line + "\n")
                # logging.info(f"已将本地失败经验追加到全局文件: {global_failure}")
            except Exception as e:
                logging.exception(f"追加失败经验到全局文件失败: {e}")

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a) + 1e-8
        nb = np.linalg.norm(b) + 1e-8
        return float(np.dot(a, b) / (na * nb))

    def retrieve(self, emb: list[float], top_k: int = 3) -> dict:
        if emb is None:
            return {"success": [], "failure": []}
        q = np.asarray(emb, dtype=np.float32)

        def _rank(records: list[dict], key: str) -> list[dict]:
            scored = []
            for rec in records:
                vec = rec.get(key)
                if not isinstance(vec, list):
                    continue
                try:
                    arr = np.asarray(vec, dtype=np.float32)
                    score = self._cosine(q, arr)
                    scored.append((score, rec))
                except Exception:
                    continue
            scored.sort(key=lambda x: x[0], reverse=True)
            return [
                {"score": s, "record": r}
                for s, r in scored[:top_k]
            ]

        return {
            "success": _rank(self.success_records, "E_I"),
            "failure": _rank(self.failure_records, "E_IT"),
        }


class ExperienceLogger:
    """将每个 episode 的媒体与结果喂给大模型，并落地到经验库。"""

    def __init__(self, base_dir: Path, vlm_client: VLMClient | None = None, memory_bank: MemoryBank | None = None) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.vlm = vlm_client or VLMClient.from_env()
        self.embedding_encoder = EmbeddingEncoder()
        self.memory_bank = memory_bank
        self.failure_path = self.base_dir / "failure.jsonl"
        self.success_path = self.base_dir / "success.jsonl"

    def _now(self) -> str:
        return datetime.utcnow().isoformat() + "Z"

    def _append_jsonl(self, file_path: Path, record: dict) -> None:
        try:
            with file_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logging.exception(f"写入经验库失败: {file_path}: {e}")

    def log_episode(self,
                    *,
                    task_group: str | None,
                    task_id: int | None,
                    task_prompt: str,
                    success: bool,
                    video_path: str | None,
                    frame_path: str | None,
                    keyframes: list[str] | None,
                    episode_ix: int | None) -> None:
        ts = self._now()
        entry_id_prefix = "success" if success else "fail"
        meta = {
            "timestamp": ts,
            "task_group": task_group,
            "task_id": task_id,
            "task_prompt": task_prompt,
            "episode_ix": episode_ix,
            "video_path": video_path,
            "frame_path": frame_path,
            "keyframes": keyframes or [],
        }

        embedding = None
        first_kf = None
        if keyframes:
            first_kf = keyframes[0]
        elif frame_path:
            first_kf = frame_path
        if first_kf and os.path.exists(first_kf):
            try:
                embedding = self.embedding_encoder.embed(first_kf, task_prompt)
            except Exception as e:
                logging.exception(f"生成嵌入失败: {e}")

        if success:
            vlm_out = {}
            try:
                vlm_out = self.vlm.generate_success_description(keyframes or [frame_path] if frame_path else None, task_prompt)
            except Exception as e:
                logging.exception(f"VLM 成功描述失败: {e}")
            record = {
                "entry_id": f"{entry_id_prefix}_{ts}",
                "task_signature": task_prompt,
                "outcome": "success",
                "E_I": embedding,
                "T_raw": task_prompt,
                **vlm_out,
                "media": meta,
            }
            self._append_jsonl(self.success_path, record)
        else:
            vlm_out = {}
            anchor_example = None
            try:
                vlm_out = self.vlm.generate_failure_report(
                    keyframes or [frame_path] if frame_path else None,
                    task_prompt,
                    anchor_example=anchor_example,
                )
            except Exception as e:
                logging.exception(f"VLM 失败分析失败: {e}")
            record = {
                "entry_id": f"{entry_id_prefix}_{ts}",
                "task_signature": task_prompt,
                "outcome": "failure",
                "E_IT": embedding,
                "T_raw": task_prompt,
                **vlm_out,
                "anchor_reference": anchor_example,
                "media": meta,
            }
            self._append_jsonl(self.failure_path, record)
            

class PerceptionModule:
    """
    智能感知模块 (MCP Agent 模式) - 完整版
    集成：Qwen3-VL (决策大脑) + RAG (记忆) + SAM3/Inpainting (执行工具) + 完整日志保存
    """
    def __init__(self, device: str = "cuda", sam3_weight_path: str = "./sam3.pt"):
        self.device = device
        self.sam3_weight_path = sam3_weight_path
        
        # 1. 初始化大脑：Qwen3-VL 客户端
        self.qwen_client = VLMClient.from_env()
        
        # 2. 初始化执行手：SAM3 模型
        self.sam3_processor, self.sam3_model, self.sam3_predictor = self._init_sam3()
        
        # 可视化参数
        self.target_color = (0, 255, 0)    # 绿色 (Target)
        self.destination_color = (255, 0, 0) # 蓝色 (Dest)
        self.negative_color = (0, 0, 255)  # 红色 (Forbidden)
        self.alpha = 0.3

    def _init_sam3(self):
        """初始化 SAM3，如果本地无权重则自动下载"""
        ckpt_path = "/mnt/disk1/shared_data/lzy/models/sam/sam3.pt"
        try:
            logging.info("正在加载 SAM3 模型...")
            model = build_sam3_image_model(
                checkpoint_path=ckpt_path,
                load_from_HF=(ckpt_path is None),
                device=self.device,
                eval_mode=True,
                enable_segmentation=True,
                enable_inst_interactivity=False,
            )
            processor = Sam3Processor(model)
            predictor = Sam3Processor(model, device=self.device)
            model = model.to(self.device).eval()
            return processor, model, predictor
        except Exception as e:
            logging.error(f"SAM3 模型加载失败：{e}", exc_info=True)
            raise

    # ========================== 核心工具函数 (包含保存逻辑) ==========================

    def _save_image(
        self,
        result_rgb: np.ndarray,
        save_path: str | Path,
        save_original: bool,
        save_mask_only: bool,
        mask: np.ndarray,
        color: tuple,
        original_image: np.ndarray = None
    ):
        """辅助函数：处理图片保存的细节（创建目录、格式转换、多文件保存）"""
        try:
            if save_path is None: return
            
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 1. 保存处理后的图像 (RGB -> BGR)
            result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(save_path), result_bgr)
            
            # 2. 保存原始图像
            if save_original and original_image is not None:
                original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
                original_path = save_path.with_name(f"{save_path.stem}_original{save_path.suffix}")
                cv2.imwrite(str(original_path), original_bgr)
            
            # 3. 保存纯掩码可视化
            if save_mask_only and mask is not None:
                mask_vis = np.zeros_like(result_bgr, dtype=np.uint8)
                mask_indices = mask == 1
                mask_vis[mask_indices] = color 
                mask_path = save_path.with_name(f"{save_path.stem}_mask{save_path.suffix}")
                cv2.imwrite(str(mask_path), mask_vis)
        
        except Exception as e:
            logging.error(f"保存图片失败：{e}", exc_info=True)

    def draw_mask(self, 
                image: np.ndarray, 
                mask: np.ndarray, 
                color: tuple, 
                alpha: float = 0.3,
                save_path: str | Path = None,  
                save_original: bool = False,   
                save_mask_only: bool = False) -> np.ndarray:
        """在图像上绘制半透明掩码"""
        try:
            if image.ndim != 3: return image
            
            # 预处理掩码
            mask = np.squeeze(mask)
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), cv2.INTER_NEAREST)
            mask = (mask > 0.5).astype(np.uint8)
            mask_indices = mask == 1
            
            if not np.any(mask_indices): 
                return image.copy()
            
            # 绘制叠加
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            color_mask = np.zeros_like(image_bgr, dtype=np.uint8)
            color_mask[mask_indices] = color
            
            overlay = cv2.addWeighted(color_mask, alpha, image_bgr, 1 - alpha, 0)
            result_bgr = image_bgr.copy()
            result_bgr[mask_indices] = overlay[mask_indices]
            
            result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            
            # 调用保存逻辑
            if save_path is not None:
                self._save_image(
                    result_rgb, save_path, save_original, save_mask_only, mask, color, original_image=image
                )
            return result_rgb

        except Exception as e:
            logging.error(f"绘制掩码失败：{e}", exc_info=True)
            return image.copy()

    # ========================== MCP 工具实现 ==========================

    def _tool_visual_overlay(self, image: np.ndarray, params: dict, save_base_path: Path = None) -> np.ndarray:
        """
        [工具 1] 视觉高亮 (Visual Overlay)
        """
        t_p = params.get("target_object", "target")
        d_p = params.get("destination", "destination")
        n_p = params.get("negative_object", "")
        
        # 1. SAM3 分割
        target_mask, dest_mask, neg_mask = self.sam3_segment(image, t_p, d_p, n_p)
        
        # 2. 绘制半透明掩码 (包含保存)
        res_img = image.copy()
        res_img = self.draw_mask(res_img, target_mask, self.target_color, self.alpha, save_path=save_base_path, save_original=True)
        res_img = self.draw_mask(res_img, dest_mask, self.destination_color, self.alpha, save_path=save_base_path)
        res_img = self.draw_mask(res_img, neg_mask, self.negative_color, self.alpha, save_path=save_base_path, save_mask_only=True)
        
        return res_img

    def _tool_remove_distractor(self, image: np.ndarray, params: dict, save_base_path: Path = None) -> np.ndarray:
        """
        [工具 2] 去除干扰物 (Remove Distractor)
        """
        object_name = params.get("object_to_remove", "")
        if not object_name:
            logging.warning("[MCP] 调用了 remove_distractor 但未指定物体名")
            return image

        logging.info(f"[MCP] 正在执行 remove_distractor，目标: {object_name}")
        try:
            image_pil = Image.fromarray(image).convert("RGB")
            inference_state = self.sam3_predictor.set_image(image_pil)
            target_state = self.sam3_predictor.set_text_prompt(state=inference_state, prompt=object_name)
            
            iou_scores = target_state["scores"].cpu().numpy()
            if len(iou_scores) == 0:
                logging.warning(f"[MCP] 未在图中检测到干扰物: {object_name}")
                return image

            max_idx = np.argmax(iou_scores)
            target_mask = target_state["masks"][max_idx].cpu().numpy().squeeze()
            
            # 掩码膨胀
            mask_uint8 = (target_mask * 255).astype(np.uint8)
            kernel = np.ones((5, 5), np.uint8) 
            mask_dilated = cv2.dilate(mask_uint8, kernel, iterations=2)
            
            # Inpainting
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            inpainted_bgr = cv2.inpaint(image_bgr, mask_dilated, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
            inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
            
            # 保存结果用于调试
            if save_base_path:
                inpaint_path = save_base_path.with_name(f"{save_base_path.stem}_inpainted.jpg")
                cv2.imwrite(str(inpaint_path), inpainted_bgr)
                logging.info(f"[MCP] 修复结果已保存: {inpaint_path}")
            
            return inpainted_rgb

        except Exception as e:
            logging.error(f"[MCP] remove_distractor 执行出错: {e}", exc_info=True)
            return image

    # ========================== 核心大脑逻辑 ==========================

    def plan_with_rag_and_vlm(self, task_desc: str, obs_frame: np.ndarray, rag_results: dict = None) -> dict:
        """
        MCP Host: 根据 图像 + 任务 + RAG记忆，决策调用哪个工具。
        """
        if obs_frame is None or self.qwen_client is None:
            return {"tool_call": "visual_overlay", "params": {"target_object": "target"}, "refined_task": task_desc}
        
        try:
            pil_img = Image.fromarray(obs_frame)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            return {"tool_call": "visual_overlay", "params": {}, "refined_task": task_desc}

        # 构建 RAG 上下文
        rag_context_str = ""
        if rag_results:
            failures = rag_results.get("failure", [])[:2]
            if failures:
                rag_context_str += "\n\n[WARNING: HISTORICAL FAILURES DETECTED]\n"
                for idx, item in enumerate(failures):
                    rec = item.get("record", {})
                    cause = rec.get('intrinsic_analysis', {}).get('direct_cause', 'unknown')
                    rag_context_str += f"- Failure Case {idx+1}: {cause}\n"
                rag_context_str += "Check if these distractors exist in the current image.\n"

        system_prompt = (
            "You are an expert robot perception coordinator using the MCP (Model Context Protocol).\n"
            "Your job: Analyze the image and task, consult historical failure memory (RAG), and select the best preprocessing tool.\n"
            "\n"
            "=== AVAILABLE TOOLS ===\n"
            "1. `visual_overlay`: (Default) Highlights target/destination with colors. Use this if the scene is clean.\n"
            "   Params: target_object, destination, negative_object\n"
            "\n"
            "2. `remove_distractor`: (Advanced) Physically removes an object from the image using inpainting.\n"
            "   CONDITION: Use this ONLY if RAG history warns about a specific distractor, OR if there is severe visual clutter blocking the path.\n"
            "   Params: object_to_remove (e.g., 'red cup', 'shadow')\n"
            "\n"
            "=== OUTPUT FORMAT (JSON ONLY) ===\n"
            "{\n"
            "  \"reasoning\": \"RAG says we failed due to a red cup distractor, and I see a red cup nearby.\",\n"
            "  \"tool_call\": \"remove_distractor\",\n"
            "  \"params\": { \"object_to_remove\": \"red cup\" },\n"
            "  \"refined_task\": \"Pick up the bowl. (Note: Distractor 'red cup' has been removed from your view).\"\n"
            "}"
        )
        
        user_prompt = (
            f"Current Task: {task_desc}\n"
            f"{rag_context_str}\n"
            "Look at the image. Decide which tool to use to maximize success rate."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}, {"type": "image_url", "image_url": {"url": data_url}}]},
        ]

        try:
            output_text = self.qwen_client._generate(messages, max_new_tokens=512)
            print(f"[Qwen3][MCP Decision]: {output_text}", flush=True)
            
            plan = _extract_json(output_text)
            if not isinstance(plan, dict) or "tool_call" not in plan:
                return {"tool_call": "visual_overlay", "params": {"target_object": "target", "destination": "destination"}, "refined_task": task_desc}
            return plan

        except Exception as e:
            logging.warning(f"Qwen3 决策失败: {e}，回退到 visual_overlay")
            return {"tool_call": "visual_overlay", "params": {"target_object": "target"}, "refined_task": task_desc}

    # ========================== 流程入口 ==========================

    def process_frame(self, image: np.ndarray, task_desc: str, step: int = 0, rag_results: dict = None) -> tuple[np.ndarray, str]:
        """
        主入口函数：被 rollout 调用
        """
        logging.info(f"===== [Perception] 处理帧 step={step} =====")
        
        # 1. 准备保存路径 (用于调试)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_base_path = Path("/mnt/disk1/shared_data/lzy/lerobot/eval_logs/mask_output") / f"frame_step_{step}_{timestamp}.jpg"
        
        # 2. Qwen + RAG 制定计划
        plan = self.plan_with_rag_and_vlm(task_desc, image, rag_results)
        
        tool_name = plan.get("tool_call", "visual_overlay")
        params = plan.get("params", {})
        refined_task = plan.get("refined_task", task_desc)
        
        processed_img = image.copy()

        # 3. 路由分发 (Tool Dispatch)
        if tool_name == "remove_distractor":
            print(f"🚀 [MCP Triggered] Removing Distractor: {params.get('object_to_remove')}", flush=True)
            processed_img = self._tool_remove_distractor(image, params, save_base_path)
            
        elif tool_name == "visual_overlay":
            # print(f"✨ [MCP Triggered] Visual Overlay", flush=True)
            processed_img = self._tool_visual_overlay(image, params, save_base_path)
            
        else:
            logging.warning(f"未知工具: {tool_name}，不做处理")

        return processed_img, refined_task

    # ========================== 基础分割函数 ==========================

    def sam3_segment(self, raw_frame: np.ndarray, target_prompt: str, destination: str, negative_prompt: str):
        """基础 SAM3 分割逻辑 (供 visual_overlay 使用)"""
        def get_mask(prompt_text):
            if not prompt_text or (":" in prompt_text and not prompt_text.split(":")[1].strip()):
                return np.zeros((raw_frame.shape[0], raw_frame.shape[1]), dtype=np.uint8)
            
            clean_prompt = prompt_text.split(":", 1)[1].strip() if ":" in prompt_text else prompt_text
            
            try:
                pil_img = Image.fromarray(raw_frame).convert('RGB')
                inference_state = self.sam3_predictor.set_image(pil_img)
                res = self.sam3_predictor.set_text_prompt(state=inference_state, prompt=clean_prompt)
                
                scores = res["scores"].cpu().numpy()
                masks = res["masks"].cpu().numpy()
                
                if len(scores) > 0:
                    best_idx = np.argmax(scores)
                    if scores[best_idx] > 0.4: 
                        return masks[best_idx].squeeze().astype(np.uint8)
            except Exception:
                pass
            return np.zeros((raw_frame.shape[0], raw_frame.shape[1]), dtype=np.uint8)

        return get_mask(target_prompt), get_mask(destination), get_mask(negative_prompt)

# rollout function with perception module integration
def rollout(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    perception_module: PerceptionModule,  # 新增感知模块
    seeds: list[int] | None = None,
    return_observations: bool = False,
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None,
    *,
    memory_bank: MemoryBank | None = None,
    task_prompt: str | None = None,
    keyframes_dir_init: Path | None = None,
    embed_encoder: EmbeddingEncoder | None = None,
) -> dict:
    """Run a batched policy rollout once through a batch of environments.

    Note that all environments in the batch are run until the last environment is done. This means some
    data will probably need to be discarded (for environments that aren't the first one to be done).

    The return dictionary contains:
        (optional) "observation": A dictionary of (batch, sequence + 1, *) tensors mapped to observation
            keys. NOTE that this has an extra sequence element relative to the other keys in the
            dictionary. This is because an extra observation is included for after the environment is
            terminated or truncated.
        "action": A (batch, sequence, action_dim) tensor of actions applied based on the observations (not
            including the last observations).
        "reward": A (batch, sequence) tensor of rewards received for applying the actions.
        "success": A (batch, sequence) tensor of success conditions (the only time this can be True is upon
            environment termination/truncation).
        "done": A (batch, sequence) tensor of **cumulative** done conditions. For any given batch element,
            the first True is followed by True's all the way till the end. This can be used for masking
            extraneous elements from the sequences above.

    Args:
        env: The batch of environments.
        policy: The policy. Must be a PyTorch nn module.
        seeds: The environments are seeded once at the start of the rollout. If provided, this argument
            specifies the seeds for each of the environments.
        return_observations: Whether to include all observations in the returned rollout data. Observations
            are returned optionally because they typically take more memory to cache. Defaults to False.
        render_callback: Optional rendering callback to be used after the environments are reset, and after
            every step.
    Returns:
        The dictionary described above.
    """
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    # Reset the policy and environments.
    policy.reset()
    observation, info = env.reset(seed=seeds)
    if render_callback is not None:
        render_callback(env)

    # 预取初始关键帧（动作前）并做 RAG 检索
    rag_results = None
    if memory_bank is not None and keyframes_dir_init is not None and embed_encoder is not None:
        try:
            init_frames = None
            try:
                init_frames = env.call("render")
            except Exception:
                init_frames = None
            if init_frames is not None:
                if isinstance(init_frames, (list, tuple)):
                    frames_iter = init_frames
                else:
                    frames_iter = [init_frames]
                for idx, frm in enumerate(frames_iter):
                    try:
                        img = Image.fromarray(frm)
                    except Exception:
                        continue
                    kf_path = keyframes_dir_init / f"episode_init_env{idx}.png"
                    kf_path.parent.mkdir(parents=True, exist_ok=True)
                    img.save(kf_path)
                    # 嵌入
                    emb = embed_encoder.embed(str(kf_path), task_prompt or "")
                    rag_results = memory_bank.retrieve(emb, top_k=3)
                    # logging.info(f"[RAG][env {idx}] init retrieval: {json.dumps(rag_results, ensure_ascii=False)[:2000]}")
                    # print(f"[Pipeline][RAG] env={idx} init_emb={emb} retrieval={json.dumps(rag_results, ensure_ascii=False)[:300]}", flush=True)
                    # print(f"[RAG][env {idx}] init retrieval:", json.dumps(rag_results, ensure_ascii=False), flush=True)
                    break  # 只处理第一个环境的检索结果
            else:
                logging.info("初始渲染失败，跳过 RAG 检索")
        except Exception as e:
            logging.exception(f"初始 RAG 检索失败: {e}")

    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []

    step = 0
    # Keep track of which environments are done.
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(
        max_steps,
        desc=f"Running rollout with at most {max_steps} steps",
        disable=inside_slurm(),  # we dont want progress bar when we use slurm, since it clutters the logs
        leave=False,
    )
    check_env_attributes_and_types(env)
    # Infer task description from environment for perception module
    def extract_task_from_env(env: VectorEnv, env_idx: int = 0) -> str:
        """
        从 env 中提取指定索引的 task 描述（适配你的嵌套元组格式）
        Args:
            env: VectorEnv 环境实例
            env_idx: 要提取的子环境索引（默认第0个）
        Returns:
            纯净的 task 字符串（如 "pick up the black bowl between the plate and the ramekin and place it on the plate"）
        """
        try:
            # 1. 调用 task_description（你的环境返回嵌套元组）
            task_result = env.call("task_description")
            # logging.info(f"原始 task 返回值：{task_result}")

            # 2. 解析嵌套结构：(('task1', 'task2'),) → ['task1', 'task2']
            # 第一层：元组转列表
            if isinstance(task_result, tuple):
                task_result = list(task_result)
            # 第二层：取出第一个元素（仍是元组）并转列表
            if len(task_result) > 0 and isinstance(task_result[0], (tuple, list)):
                task_list = list(task_result[0])
            else:
                task_list = task_result

            # 3. 校验列表长度和索引合法性
            if len(task_list) <= env_idx:
                logging.warning(f"env_idx {env_idx} 超出 task 列表长度 {len(task_list)}，返回空字符串")
                return ""
            # 4. 提取指定索引的 task 并清理空格
            task_desc = task_list[env_idx].strip()
            logging.info(f"提取到 task_desc（索引 {env_idx}）：{task_desc}")
            return task_desc

        except Exception as e:
            logging.error(f"提取 task 失败：{e}", exc_info=True)
            return ""

    # 替换原 task_desc 赋值
    task_desc = extract_task_from_env(env, env_idx=0)
        
    while not np.all(done) and step < max_steps:
        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
        observation = preprocess_observation(observation)
        if return_observations:
            all_observations.append(deepcopy(observation))

        # 每 50 个 step 运行一次感知 + Qwen + SAM3，其余 step 使用原始 obs 和 task。
        use_refined_perception = (step % 10 == 0)
        # use_refined_perception = False

        def find_visual_obs(obs):
            if obs is None or not isinstance(obs, dict):
                return None, None
            
            visual_keys = ["image", "cam", "rgb", "view", "vision", "obs_img", "camera_rgb"]
            for k in obs.keys():
                if k is None:
                    continue
                lower_k = k.lower()
                if any(vk in lower_k for vk in visual_keys):
                    val = obs.get(k)
                    if val is None:
                        continue
                    # 直接转 CUDA Tensor（避免 numpy 中转）
                    if isinstance(val, np.ndarray):
                        val = torch.from_numpy(val).to("cuda:0")
                        obs[k] = val
                    elif isinstance(val, torch.Tensor):
                        val = val.to("cuda:0")
                        obs[k] = val
                    # 校验有效图像 Tensor
                    if isinstance(val, torch.Tensor) and len(val.shape) >= 3:
                        return k, val
            # 递归查找嵌套字典
            for k, v in obs.items():
                if k is None or v is None:
                    continue
                if isinstance(v, dict):
                    sub_k, sub_val = find_visual_obs(v)
                    if sub_k is not None and sub_val is not None:
                        return f"{k}.{sub_k}", sub_val
            return None, None

        if use_refined_perception:
            # 1. 查找视觉观测
            visual_obs_key, visual_obs_val = find_visual_obs(observation)
            # logging.info(f"Step {step} - 视觉观测查找结果：key={visual_obs_key}, val_type={type(visual_obs_val)}, shape={visual_obs_val.shape if hasattr(visual_obs_val, 'shape') else 'None'}")

            # 2. 处理视觉观测（核心逻辑）
            if visual_obs_key is not None:
                try:
                    # ===== 新增：统一转换为 numpy 数组（兼容 Tensor）=====
                    if isinstance(visual_obs_val, torch.Tensor):
                        # Tensor → numpy（先转 CPU，避免 CUDA 数组报错）
                        raw_obs_np = visual_obs_val.cpu().numpy()
                        # logging.info(f"Step {step} - 将 Tensor 转换为 numpy 数组，形状：{raw_obs_np.shape}")
                    elif isinstance(visual_obs_val, np.ndarray):
                        raw_obs_np = visual_obs_val
                    else:
                        logging.warning(f"Step {step} - 不支持的观测类型：{type(visual_obs_val)}")
                        observation = add_envs_task(env, observation)
                        continue

                    # ===== 原有逻辑（基于 numpy 处理）=====
                    # 取第一个环境的帧 (B, C, H, W) -> (C, H, W)
                    raw_frame = raw_obs_np[0]
                    # 维度调整：确保是 (C, H, W) 后转 (H, W, C)
                    if raw_frame.shape[0] in [1, 3]:  # 通道在前（C, H, W）
                        raw_frame = raw_frame.transpose(1, 2, 0)
                    # 归一化到0-255
                    if raw_frame.max() <= 1.0:
                        raw_frame = (raw_frame * 255).astype(np.uint8)
                    else:
                        raw_frame = raw_frame.astype(np.uint8)
                    
                    # 感知模块处理：生成掩码+细化任务（传入 RAG 检索结果）
                    processed_frame, refined_task = perception_module.process_frame(raw_frame, task_desc, step=step, rag_results=rag_results)
                    # logging.info(f"Step {step} - frame已处理，细化任务描述：{refined_task}")
                    # print(f"[Pipeline][Step {step}] refined_task={refined_task}", flush=True)
                    
                    # ===== 修复：更新观测时兼容原格式（Tensor/numpy）=====
                    # 转回 (C, H, W) 并归一化到0-1
                    processed_frame_tensor = torch.from_numpy(processed_frame.transpose(2, 0, 1)).float() / 255.0
                    # logging.info(f"Step {step} - 处理后帧转换为 Tensor，形状：{processed_frame_tensor.shape}, 数值范围：[{processed_frame_tensor.min().item()}, {processed_frame_tensor.max().item()}]")
                    # 扩展到batch维度 (C, H, W) -> (B, C, H, W)
                    updated_obs = einops.repeat(
                        processed_frame_tensor, "c h w -> b c h w", b=env.num_envs
                    )
                    # logging.info(f"Step {step} - 更新观测形状：{updated_obs.shape}, 设备：{'cuda' if torch.cuda.is_available() else 'cpu'}")
                    # 还原为原观测格式（Tensor/numpy）
                    if isinstance(visual_obs_val, torch.Tensor):
                        updated_obs = updated_obs.to(visual_obs_val.device)  # 同步设备
                        # logging.info(f"Step {step} - 还原观测格式为 Tensor，设备：{updated_obs.device}")
                    else:
                        updated_obs = updated_obs.numpy()
                    if visual_obs_key not in observation:
                        logging.warning(f"Step {step} - 观测键 {visual_obs_key} 不存在，跳过更新")
                    else:
                        observation[visual_obs_key] = updated_obs
                    #     logging.info(
                    #         f"Step {step} - 视觉观测已更新到 observation 中，键：{visual_obs_key}, 数值范围："
                    #         f"[{observation[visual_obs_key].min() if isinstance(observation[visual_obs_key], torch.Tensor) else observation[visual_obs_key].min()}, "
                    #         f"{observation[visual_obs_key].max() if isinstance(observation[visual_obs_key], torch.Tensor) else observation[visual_obs_key].max()}]"
                    #     )
                    logging.info(f"Step {step} - 视觉观测已更新到 observation 中")
                    # 更新任务描述（使用精简后的 refined_task）
                    observation = add_envs_task(env, observation, task_desc=refined_task)
                
                except Exception as e:
                    logging.error(f"Step {step} - 处理视觉观测失败：{e}", exc_info=True)
                    # 出错时退回到原始任务描述
                    observation = add_envs_task(env, observation)
            else:
                # 找不到视觉观测时，直接使用原始任务描述
                observation = add_envs_task(env, observation)
        else:
            # 本 step 不做视觉增强，直接使用原始 observation + 原始任务描述
            observation = add_envs_task(env, observation, task_desc)

        # Infer "task" from attributes of environments.
        # TODO: works with SyncVectorEnv but not AsyncVectorEnv
        # observation = add_envs_task(env, observation)
        observation = preprocessor(observation)
        # print(f"observation keys after preprocessor: {list(observation.keys())}", flush=True)
        # print(f"observation.language.tokens:{observation.get('language', {}).get('tokens', None)}", flush=True)
        # print(f"observation.language.attention_mask:{observation.get('language', {}).get('attention_mask', None)}", flush=True)
        # print(f"observation.task:{observation.get('task', None)}", flush=True)
        with torch.inference_mode():
            action = policy.select_action(observation)
        action = postprocessor(action)

        # Convert to CPU / numpy.
        action_numpy: np.ndarray = action.to("cpu").numpy()
        assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"

        # Apply the next action.
        observation, reward, terminated, truncated, info = env.step(action_numpy)
        if render_callback is not None:
            render_callback(env)

        # VectorEnv stores is_success in `info["final_info"][env_index]["is_success"]`. "final_info" isn't
        # available if none of the envs finished.
        if "final_info" in info:
            final_info = info["final_info"]
            if not isinstance(final_info, dict):
                raise RuntimeError(
                    "Unsupported `final_info` format: expected dict (Gymnasium >= 1.0). "
                    "You're likely using an older version of gymnasium (< 1.0). Please upgrade."
                )
            successes = final_info["is_success"].tolist()
        else:
            successes = [False] * env.num_envs

        # Keep track of which environments are done so far.
        # Mark the episode as done if we reach the maximum step limit.
        # This ensures that the rollout always terminates cleanly at `max_steps`,
        # and allows logging/saving (e.g., videos) to be triggered consistently.
        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)

        all_actions.append(torch.from_numpy(action_numpy))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))

        step += 1
        running_success_rate = (
            einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy().mean()
        )
        progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
        progbar.update()

    # Track the final observation.
    if return_observations:
        observation = preprocess_observation(observation)
        all_observations.append(deepcopy(observation))

    # Stack the sequence along the first dimension so that we have (batch, sequence, *) tensors.
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

# rollout function with perception module integration
def eval_policy(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    perception_module: PerceptionModule,  # 新增感知模块
    n_episodes: int,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    *,
    memory_bank: MemoryBank | None = None,
    embed_encoder: EmbeddingEncoder | None = None,
    task_prompt: str | None = None,
    experience_logger: ExperienceLogger | None = None,
) -> dict:
    """
    Args:
        env: The batch of environments.
        policy: The policy.
        n_episodes: The number of episodes to evaluate.
        max_episodes_rendered: Maximum number of episodes to render into videos.
        videos_dir: Where to save rendered videos.
        return_episode_data: Whether to return episode data for online training. Incorporates the data into
            the "episodes" key of the returned dictionary.
        start_seed: The first seed to use for the first individual rollout. For all subsequent rollouts the
            seed is incremented by 1. If not provided, the environments are not manually seeded.
    Returns:
        Dictionary with metrics and data regarding the rollouts.
    """
    if max_episodes_rendered > 0 and not videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")

    if not isinstance(policy, PreTrainedPolicy):
        raise ValueError(
            f"Policy of type 'PreTrainedPolicy' is expected, but type '{type(policy)}' was provided."
        )

    start = time.time()
    policy.eval()

    # Determine how many batched rollouts we need to get n_episodes. Note that if n_episodes is not evenly
    # divisible by env.num_envs we end up discarding some data in the last batch.
    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    # Keep track of some metrics.
    sum_rewards = []
    max_rewards = []
    all_successes = []
    all_seeds = []
    threads = []  # for video saving threads
    n_episodes_rendered = 0  # for saving the correct number of videos

    # Callback for visualization.
    def render_frame(env: gym.vector.VectorEnv):
        # noqa: B023
        if n_episodes_rendered >= max_episodes_rendered:
            return
        n_to_render_now = min(max_episodes_rendered - n_episodes_rendered, env.num_envs)
        if isinstance(env, gym.vector.SyncVectorEnv):
            ep_frames.append(np.stack([env.envs[i].render() for i in range(n_to_render_now)]))  # noqa: B023
        elif isinstance(env, gym.vector.AsyncVectorEnv):
            # Here we must render all frames and discard any we don't need.
            ep_frames.append(np.stack(env.call("render")[:n_to_render_now]))

    if max_episodes_rendered > 0:
        video_paths: list[str] = []
        frames_dir: Path | None = videos_dir / "frames" if videos_dir is not None else None
        if frames_dir is not None:
            frames_dir.mkdir(parents=True, exist_ok=True)
        keyframes_dir: Path | None = videos_dir / "keyframes" if videos_dir is not None else None
        if keyframes_dir is not None:
            keyframes_dir.mkdir(parents=True, exist_ok=True)
        keyframes_dir_init: Path | None = videos_dir / "keyframes_init" if videos_dir is not None else None
        if keyframes_dir_init is not None:
            keyframes_dir_init.mkdir(parents=True, exist_ok=True)
    else:
        frames_dir = None
        keyframes_dir = None
        keyframes_dir_init = None

    # 收集所有batch的媒体记录，供经验库使用
    episode_media_records: list[dict] = []

    if return_episode_data:
        episode_data: dict | None = None

    # we dont want progress bar when we use slurm, since it clutters the logs
    progbar = trange(n_batches, desc="Stepping through eval batches", disable=inside_slurm())
    for batch_ix in progbar:
        # Cache frames for rendering videos. Each item will be (b, h, w, c), and the list indexes the rollout
        # step.
        if max_episodes_rendered > 0:
            ep_frames: list[np.ndarray] = []

        if start_seed is None:
            seeds = None
        else:
            seeds = range(
                start_seed + (batch_ix * env.num_envs), start_seed + ((batch_ix + 1) * env.num_envs)
            )
        
        rollout_data = rollout(
            env=env,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            perception_module = perception_module, # 传入感知模块
            seeds=list(seeds) if seeds else None,
            return_observations=return_episode_data,
            render_callback=render_frame if max_episodes_rendered > 0 else None,
            memory_bank=memory_bank,
            task_prompt=task_prompt,
            keyframes_dir_init=keyframes_dir_init,
            embed_encoder=embed_encoder,
        )
        # logging.info(f"Rollout 完成，第 {batch_ix+1} 批次数据获取成功。")
        # Figure out where in each rollout sequence the first done condition was encountered (results after
        # this won't be included).
        n_steps = rollout_data["done"].shape[1]
        # Note: this relies on a property of argmax: that it returns the first occurrence as a tiebreaker.
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)

        # Make a mask with shape (batch, n_steps) to mask out rollout data after the first done
        # (batch-element-wise). Note the `done_indices + 1` to make sure to keep the data from the done step.
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()
        # Extend metrics.
        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
        sum_rewards.extend(batch_sum_rewards.tolist())
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
        max_rewards.extend(batch_max_rewards.tolist())
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
        all_successes.extend(batch_successes.tolist())
        if seeds:
            all_seeds.extend(seeds)
        else:
            all_seeds.append(None)

        # FIXME: episode_data is either None or it doesn't exist
        if return_episode_data:
            this_episode_data = _compile_episode_data(
                rollout_data,
                done_indices,
                start_episode_index=batch_ix * env.num_envs,
                start_data_index=(0 if episode_data is None else (episode_data["index"][-1].item() + 1)),
                fps=env.unwrapped.metadata["render_fps"],
            )
            if episode_data is None:
                episode_data = this_episode_data
            else:
                # Some sanity checks to make sure we are correctly compiling the data.
                assert episode_data["episode_index"][-1] + 1 == this_episode_data["episode_index"][0]
                assert episode_data["index"][-1] + 1 == this_episode_data["index"][0]
                # Concatenate the episode data.
                episode_data = {k: torch.cat([episode_data[k], this_episode_data[k]]) for k in episode_data}

        # Maybe render video for visualization.
        if max_episodes_rendered > 0 and len(ep_frames) > 0:
            batch_stacked_frames = np.stack(ep_frames, axis=1)  # (b, t, *)
            batch_successes_list = [bool(x) for x in batch_successes.flatten().tolist()]
            for local_ix, (stacked_frames, done_index, succ_flag) in enumerate(
                zip(batch_stacked_frames, done_indices.flatten().tolist(), batch_successes_list, strict=False)
            ):
                if n_episodes_rendered >= max_episodes_rendered:
                    break

                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                video_paths.append(str(video_path))
                
                # 同步保存最终帧作为缩略图（便于 VLM 分析）
                frame_path = None
                keyframe_paths: list[str] = []
                try:
                    if imageio is not None:
                        if frames_dir is not None:
                            frame_path = frames_dir / f"eval_episode_{n_episodes_rendered}.png"
                            imageio.imwrite(str(frame_path), stacked_frames[: done_index + 1][-1])
                        # 关键帧：每5帧取1帧，并确保包含终帧
                        if keyframes_dir is not None:
                            for idx, frm in enumerate(stacked_frames[: done_index + 1]):
                                if idx % 5 == 0 or idx == done_index:
                                    kf_path = keyframes_dir / f"eval_episode_{n_episodes_rendered}_k{idx}.png"
                                    imageio.imwrite(str(kf_path), frm)
                                    keyframe_paths.append(str(kf_path))
                except Exception as e:
                    logging.exception(f"保存终帧/关键帧失败: {e}")
                    frame_path = None

                thread = threading.Thread(
                    target=write_video,
                    args=(
                        str(video_path),
                        stacked_frames[: done_index + 1],  # + 1 to capture the last observation
                        env.unwrapped.metadata["render_fps"],
                    ),
                )
                thread.start()
                threads.append(thread)
                
                # 记录该 episode 的媒体与标注（用于经验库）
                episode_media_records.append({
                    "episode_ix": batch_ix * env.num_envs + local_ix,
                    "success": succ_flag,
                    "video_path": str(video_path),
                    "frame_path": str(frame_path) if frame_path is not None else None,
                    "keyframes": keyframe_paths,
                })
                
                n_episodes_rendered += 1

        progbar.set_postfix(
            {"running_success_rate": f"{np.mean(all_successes[:n_episodes]).item() * 100:.1f}%"}
        )

    # Wait till all video rendering threads are done.
    for thread in threads:
        thread.join()

    # Compile eval info.
    info = {
        "per_episode": [
            {
                "episode_ix": i,
                "sum_reward": sum_reward,
                "max_reward": max_reward,
                "success": success,
                "seed": seed,
            }
            for i, (sum_reward, max_reward, success, seed) in enumerate(
                zip(
                    sum_rewards[:n_episodes],
                    max_rewards[:n_episodes],
                    all_successes[:n_episodes],
                    all_seeds[:n_episodes],
                    strict=True,
                )
            )
        ],
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
            "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / n_episodes,
        },
    }

    if return_episode_data:
        info["episodes"] = episode_data

    if max_episodes_rendered > 0:
        info["video_paths"] = video_paths
        info["episode_media"] = episode_media_records

    return info


def _compile_episode_data(
    rollout_data: dict, done_indices: Tensor, start_episode_index: int, start_data_index: int, fps: float
) -> dict:
    """Convenience function for `eval_policy(return_episode_data=True)`

    Compiles all the rollout data into a Hugging Face dataset.

    Similar logic is implemented when datasets are pushed to hub (see: `push_to_hub`).
    """
    ep_dicts = []
    total_frames = 0
    for ep_ix in range(rollout_data[ACTION].shape[0]):
        # + 2 to include the first done frame and the last observation frame.
        num_frames = done_indices[ep_ix].item() + 2
        total_frames += num_frames

        # Here we do `num_frames - 1` as we don't want to include the last observation frame just yet.
        ep_dict = {
            ACTION: rollout_data[ACTION][ep_ix, : num_frames - 1],
            "episode_index": torch.tensor([start_episode_index + ep_ix] * (num_frames - 1)),
            "frame_index": torch.arange(0, num_frames - 1, 1),
            "timestamp": torch.arange(0, num_frames - 1, 1) / fps,
            DONE: rollout_data["done"][ep_ix, : num_frames - 1],
            "next.success": rollout_data["success"][ep_ix, : num_frames - 1],
            REWARD: rollout_data["reward"][ep_ix, : num_frames - 1].type(torch.float32),
        }

        # For the last observation frame, all other keys will just be copy padded.
        for k in ep_dict:
            ep_dict[k] = torch.cat([ep_dict[k], ep_dict[k][-1:]])

        for key in rollout_data[OBS_STR]:
            ep_dict[key] = rollout_data[OBS_STR][key][ep_ix, :num_frames]

        ep_dicts.append(ep_dict)

    data_dict = {}
    for key in ep_dicts[0]:
        data_dict[key] = torch.cat([x[key] for x in ep_dicts])

    data_dict["index"] = torch.arange(start_data_index, start_data_index + total_frames, 1)

    return data_dict


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    # logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    # logging.info("Making environment.")
    envs = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)
    # logging.info("Making policy.")

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        rename_map=cfg.rename_map,
    )

    policy.eval()

    # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    
    # logging.info("Initializing Perception Module (Qwen3-VL + SAM3)...")
    perception_module = PerceptionModule(device=str(device), sam3_weight_path=str("/mnt/disk1/shared_data/lzy/models/sam/sam3.pt"))
    
    # 初始化经验库记录器与 RAG 检索器
    experience_db_dir = Path(cfg.output_dir) / "experience_db"
    embed_encoder = EmbeddingEncoder()
    # MemoryBank 从全局文件读取（lerobot/outputs/success.jsonl 和 failure.jsonl）
    memory_bank = MemoryBank(global_outputs_dir=None, encoder=embed_encoder)
    experience_logger = ExperienceLogger(
        base_dir=experience_db_dir,
        vlm_client=VLMClient.from_env(),
        memory_bank=memory_bank,
    )
    
    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            perception_module=perception_module,
            n_episodes=cfg.eval.n_episodes,
            max_episodes_rendered=10,
            videos_dir=Path(cfg.output_dir) / "videos",
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.env.max_parallel_tasks,
            experience_logger=experience_logger,
            memory_bank=memory_bank,
            embed_encoder=embed_encoder,
        )
        print("[Pipeline] eval_policy_all finished, aggregated metrics:", info.get("overall"), flush=True)
        print("Overall Aggregated Metrics:")
        print(info["overall"])

        # Print per-suite stats
        for task_group, task_group_info in info.items():
            print(f"\nAggregated Metrics for {task_group}:")
            print(task_group_info)
    # Close all vec envs
    close_envs(envs)

    # 将本次测试产生的经验追加到全局经验库文件
    experience_db_dir = Path(cfg.output_dir) / "experience_db"
    if experience_db_dir.exists():
        MemoryBank.merge_local_to_global(experience_db_dir, global_outputs_dir=None)
        # logging.info(f"已将本地经验库 ({experience_db_dir}) 追加到全局经验库")

    # Save info
    with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

    # logging.info("End of eval with Perception Module (Qwen3-VL + SAM3 + RAG)")


# ---- typed payload returned by one task eval ----
class TaskMetrics(TypedDict):
    sum_rewards: list[float]
    max_rewards: list[float]
    successes: list[bool]
    video_paths: list[str]
    episode_media: list[dict]


ACC_KEYS = ("sum_rewards", "max_rewards", "successes", "video_paths")


def eval_one(
    env: gym.vector.VectorEnv,
    *,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    perception_module: PerceptionModule,
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
    memory_bank: MemoryBank | None = None,
    embed_encoder: EmbeddingEncoder | None = None,
    task_prompt: str | None = None,
) -> TaskMetrics:
    """Evaluates one task_id of one suite using the provided vec env."""

    task_videos_dir = videos_dir
    # logging.info(f"Evaluating one task for {n_episodes} episodes.")
    task_result = eval_policy(
        env=env,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        perception_module=perception_module,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
        memory_bank=memory_bank,
        embed_encoder=embed_encoder,
        task_prompt=task_prompt,
    )

    per_episode = task_result["per_episode"]
    return TaskMetrics(
        sum_rewards=[ep["sum_reward"] for ep in per_episode],
        max_rewards=[ep["max_reward"] for ep in per_episode],
        successes=[ep["success"] for ep in per_episode],
        video_paths=task_result.get("video_paths", []),
        episode_media=task_result.get("episode_media", []),
    )


def run_one(
    task_group: str,
    task_id: int,
    env,
    *,
    policy,
    preprocessor,
    postprocessor,
    perception_module: PerceptionModule,
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
    experience_logger: ExperienceLogger | None = None,
    memory_bank: MemoryBank | None = None,
    embed_encoder: EmbeddingEncoder | None = None,
):
    """
    Run eval_one for a single (task_group, task_id, env).
    Returns (task_group, task_id, task_metrics_dict).
    This function is intentionally module-level to make it easy to test.
    """
    task_videos_dir = None
    if videos_dir is not None:
        task_videos_dir = videos_dir / f"{task_group}_{task_id}"
        task_videos_dir.mkdir(parents=True, exist_ok=True)
    try:
        from libero.libero import get_libero_path
        import os

        # 1. 获取 BDDL 基础目录 (.../site-packages/libero/libero/bddl_files)
        bddl_base = get_libero_path("bddl_files")
        
        # 2. 定位 tasks_info.txt
        info_path = os.path.join(bddl_base, task_group, "tasks_info.txt")
        
        real_task_name = "Unknown"
        bddl_path = "Unknown"

        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            
            if task_id < len(lines):
                # 【关键修改点】
                # 文件里的内容是: "libero/bddl_files/libero_spatial/task_name.bddl"
                # 我们只需要最后那部分: "task_name.bddl"
                raw_line_path = lines[task_id]
                bddl_filename = os.path.basename(raw_line_path) 
                
                # 记录任务名（去掉 .bddl 后缀，用于日志显示）
                real_task_name = os.path.splitext(bddl_filename)[0]
                
                # 重新拼合正确的绝对路径
                # 结构: [bddl_base] / [task_group] / [filename]
                bddl_path = os.path.join(bddl_base, task_group, bddl_filename)
            else:
                print(f"[WARNING] Task ID {task_id} 越界")
        
        # 打印结果
        print(f"[DEBUG] Parsed BDDL for ({task_group}, {task_id}):")
        print(f"       -> Raw Line  : {lines[task_id] if 'lines' in locals() and task_id < len(lines) else 'N/A'}")
        print(f"       -> Filename  : {os.path.basename(bddl_path)}")
        print(f"       -> Final Path: {bddl_path}")

        if os.path.exists(bddl_path):
            print(f"       -> ✅ 文件存在 (OK)")
        else:
            print(f"       -> ❌ 文件不存在 (NOT FOUND)")

    except Exception as e:
        print(f"[ERROR] Path inference failed: {e}")
    
    # 推断任务文本提示（用于 RAG 检索与打印）
    task_prompt = _infer_task_prompt(env, task_group=task_group, task_id=task_id)

    # Call the existing eval_one (assumed to return TaskMetrics-like dict)
    metrics = eval_one(
        env,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        perception_module=perception_module,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
        memory_bank=memory_bank,
        embed_encoder=embed_encoder,
        task_prompt=task_prompt,
    )
    # ensure we always provide video_paths key to simplify accumulation
    if max_episodes_rendered > 0:
        metrics.setdefault("video_paths", [])

    # 新增：将本任务保存的 episode 媒体喂给经验库
    if experience_logger is not None:
        metrics["task_prompt"] = task_prompt
        for item in metrics.get("episode_media", []) or []:
            try:
                experience_logger.log_episode(
                    task_group=task_group,
                    task_id=task_id,
                    task_prompt=task_prompt,
                    success=bool(item.get("success", False)),
                    video_path=item.get("video_path"),
                    frame_path=item.get("frame_path"),
                    keyframes=item.get("keyframes"),
                    episode_ix=item.get("episode_ix"),
                )
            except Exception as e:
                logging.exception(f"记录经验库失败 (group={task_group}, id={task_id}): {e}")

    metrics.setdefault("task_prompt", task_prompt)
    return task_group, task_id, metrics


def _infer_task_prompt(env: gym.vector.VectorEnv, *, task_group: str, task_id: int) -> str:
    """尽量从环境中提取文本化任务描述，失败则回退到 group/id。"""
    try:
        base = None
        if isinstance(env, gym.vector.SyncVectorEnv) and len(env.envs) > 0:
            base = getattr(env.envs[0], "unwrapped", env.envs[0])
        if base is not None:
            for attr in ["task_prompt", "task_description", "instruction", "goal", "task"]:
                val = getattr(base, attr, None)
                if isinstance(val, str) and len(val) > 0:
                    return val
        for name in ["get_task_prompt", "get_instruction", "get_goal_text"]:
            try:
                vals = env.call(name)
                if isinstance(vals, (list, tuple)) and len(vals) > 0 and isinstance(vals[0], str):
                    if vals[0]:
                        return vals[0]
            except Exception:
                pass
    except Exception:
        pass
    return f"{task_group}:{task_id}"


def eval_policy_all(
    envs: dict[str, dict[int, gym.vector.VectorEnv]],
    policy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    perception_module: PerceptionModule,
    n_episodes: int,
    *,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    max_parallel_tasks: int = 1,
    experience_logger: ExperienceLogger | None = None,
    memory_bank: MemoryBank | None = None,
    embed_encoder: EmbeddingEncoder | None = None,
) -> dict:
    """
    Evaluate a nested `envs` dict: {task_group: {task_id: vec_env}}.
    This implementation flattens tasks, runs them sequentially or via ThreadPoolExecutor,
    accumulates per-group and overall statistics, and returns the same aggregate metrics
    schema as the single-env evaluator (avg_sum_reward / avg_max_reward / pc_success / timings)
    plus per-task infos.
    """
    start_t = time.time()

    # Flatten envs into list of (task_group, task_id, env)
    tasks = [(tg, tid, vec) for tg, group in envs.items() for tid, vec in group.items()]

    # accumulators: track metrics at both per-group level and across all groups
    group_acc: dict[str, dict[str, list]] = defaultdict(lambda: {k: [] for k in ACC_KEYS})
    overall: dict[str, list] = {k: [] for k in ACC_KEYS}
    per_task_infos: list[dict] = []

    # small inline helper to accumulate one task's metrics into accumulators
    def _accumulate_to(group: str, metrics: dict):
        # metrics expected to contain 'sum_rewards', 'max_rewards', 'successes', optionally 'video_paths'
        # but eval_one may store per-episode lists; we assume metrics uses scalars averaged per task as before.
        # To be robust, accept scalars or lists.
        def _append(key, value):
            if value is None:
                return
            if isinstance(value, list):
                group_acc[group][key].extend(value)
                overall[key].extend(value)
            else:
                group_acc[group][key].append(value)
                overall[key].append(value)

        _append("sum_rewards", metrics.get("sum_rewards"))
        _append("max_rewards", metrics.get("max_rewards"))
        _append("successes", metrics.get("successes"))
        # video_paths is list-like
        paths = metrics.get("video_paths", [])
        if paths:
            group_acc[group]["video_paths"].extend(paths)
            overall["video_paths"].extend(paths)

    # Choose runner (sequential vs threaded)
    task_runner = partial(
        run_one,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        perception_module = perception_module,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
        experience_logger=experience_logger,
        memory_bank=memory_bank,
        embed_encoder=embed_encoder,
    )

    if max_parallel_tasks <= 1:
        # sequential path (single accumulator path on the main thread)
        # NOTE: keeping a single-threaded accumulator avoids concurrent list appends or locks
        for task_group, task_id, env in tasks:
            tg, tid, metrics = task_runner(task_group, task_id, env)
            _accumulate_to(tg, metrics)
            per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})
    else:
        # threaded path: submit all tasks, consume completions on main thread and accumulate there
        with cf.ThreadPoolExecutor(max_workers=max_parallel_tasks) as executor:
            fut2meta = {}
            for task_group, task_id, env in tasks:
                fut = executor.submit(task_runner, task_group, task_id, env)
                fut2meta[fut] = (task_group, task_id)
            for fut in cf.as_completed(fut2meta):
                tg, tid, metrics = fut.result()
                _accumulate_to(tg, metrics)
                per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})

    # compute aggregated metrics helper (robust to lists/scalars)
    def _agg_from_list(xs):
        if not xs:
            return float("nan")
        arr = np.array(xs, dtype=float)
        return float(np.nanmean(arr))

    # compute per-group aggregates
    groups_aggregated = {}
    for group, acc in group_acc.items():
        groups_aggregated[group] = {
            "avg_sum_reward": _agg_from_list(acc["sum_rewards"]),
            "avg_max_reward": _agg_from_list(acc["max_rewards"]),
            "pc_success": _agg_from_list(acc["successes"]) * 100 if acc["successes"] else float("nan"),
            "n_episodes": len(acc["sum_rewards"]),
            "video_paths": list(acc["video_paths"]),
        }

    # overall aggregates
    overall_agg = {
        "avg_sum_reward": _agg_from_list(overall["sum_rewards"]),
        "avg_max_reward": _agg_from_list(overall["max_rewards"]),
        "pc_success": _agg_from_list(overall["successes"]) * 100 if overall["successes"] else float("nan"),
        "n_episodes": len(overall["sum_rewards"]),
        "eval_s": time.time() - start_t,
        "eval_ep_s": (time.time() - start_t) / max(1, len(overall["sum_rewards"])),
        "video_paths": list(overall["video_paths"]),
    }

    return {
        "per_task": per_task_infos,
        "per_group": groups_aggregated,
        "overall": overall_agg,
    }


def main():
    init_logging("/mnt/disk1/shared_data/lzy/lerobot/eval_logs/all.log")
    # init_logging()
    eval_main()


if __name__ == "__main__":
    main()
