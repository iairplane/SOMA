import base64
import io
import logging
import time
import cv2
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

from soma_tools import MCPTools
from soma_vlm import Qwen3VLAPIClient


def _maybe_load_image_any(src: Any) -> np.ndarray | None:
    """Load image from possible forms:
    - np.ndarray
    - path str/Path
    - base64 png string
    """
    if src is None:
        return None

    if isinstance(src, np.ndarray):
        return src

    if isinstance(src, (str, Path)):
        s = str(src)
        p = Path(s)
        if p.exists():
            return np.array(Image.open(p).convert("RGB"))

        # maybe base64
        try:
            if "," in s:
                s = s.split(",", 1)[1]
            raw = base64.b64decode(s)
            return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
        except Exception:
            return None

    return None


class PerceptionModule:
    """SOMA 战略编排器 (Strategic Orchestrator)

    负责：Observe -> (RAG hints) -> Plan(VLM) -> Act(MCP tools)
    """

    def __init__(
        self,
        vlm_client: Qwen3VLAPIClient | None = None,
        *,
        sam3_base_url: str = "http://127.0.0.1:5001",
        tool_timeout_s: float = 10.0,
    ):
        self.vlm = vlm_client or Qwen3VLAPIClient()
        self.tools = MCPTools(sam3_base_url=sam3_base_url, timeout_s=tool_timeout_s)
        self.retry_count = 0
        
        # 调试输出目录 (硬编码以保持与 origin 逻辑一致，也可改为传参)
        self.debug_output_dir = Path("/mnt/disk1/shared_data/lzy/lerobot/eval_logs/mask_output")

    def _save_image(
        self,
        result_rgb: np.ndarray,
        save_path: Path,
        save_original: bool = False,
        original_image: np.ndarray = None
    ):
        """
        [恢复的方法] 保存图像用于调试
        由于现在是 CS 架构，Client 端拿不到 Service 端的 mask，因此这里只恢复保存 Result 和 Original 的逻辑。
        """
        try:
            if save_path is None: return
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 1. 保存处理后的图像 (RGB -> BGR for OpenCV)
            result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(save_path), result_bgr)
            
            # 2. 保存原始图像
            if save_original and original_image is not None:
                original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
                original_path = save_path.with_name(f"{save_path.stem}_original{save_path.suffix}")
                cv2.imwrite(str(original_path), original_bgr)
                
            logging.info(f"[SOMA Debug] Saved frame to {save_path}")
        
        except Exception as e:
            logging.error(f"保存图片失败：{e}", exc_info=True)

    def process_frame(
        self,
        image: np.ndarray,
        current_task: str,
        step: int,
        rag_context: Dict | None = None,
    ) -> Tuple[np.ndarray, str, Dict]:
        rag_context = rag_context or {}

        # 1. 准备保存路径 (类似 origin 逻辑)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_base_path = self.debug_output_dir / f"frame_step_{step}_{timestamp}.jpg"

        # 2. RAG Context 构建
        rag_str = ""
        rag_hints: dict[str, Any] = {}

        failures = rag_context.get("failure") or []
        successes = rag_context.get("success") or []

        if failures:
            diags = [str(f.get("diagnosis", "")) for f in failures[:2] if f.get("diagnosis")]
            if diags:
                rag_str = f"Warning! Past failures causes: {'; '.join(diags)}."

        def _has_tex(item: dict, key: str) -> bool:
            info = item.get("info") if isinstance(item, dict) else None
            assets = info.get("assets") if isinstance(info, dict) else None
            if not isinstance(assets, dict):
                return False
            val = assets.get(key)
            return bool(val)

        rag_hints["success_has_object_texture"] = bool(successes) and _has_tex(successes[0], "object_texture_png_b64")
        rag_hints["success_has_floor_texture"] = bool(successes) and _has_tex(successes[0], "floor_texture_png_b64")
        rag_hints["failure_has_object_texture"] = bool(failures) and _has_tex(failures[0], "object_texture_png_b64")
        rag_hints["failure_has_floor_texture"] = bool(failures) and _has_tex(failures[0], "floor_texture_png_b64")

        # 3. VLM Planning
        try:
            plan = self.vlm.orchestrate_perception(
                image=image,
                task_desc=current_task,
                rag_context=rag_str,
                rag_hints=rag_hints,
            )
        except Exception as e:
            logging.error(f"[SOMA] Planning failed: {e}")
            return image, current_task, {}

        tool_chain = plan.get("tool_chain", []) or []
        params = plan.get("params", {}) if isinstance(plan.get("params"), dict) else {}
        refined_task_text = plan.get("refined_task", current_task)
        task_plan = plan.get("task_plan", {}) if isinstance(plan.get("task_plan"), dict) else {}

        processed_img = image.copy()
        control_flags: dict[str, Any] = {"encore": False}

        # Control-flow plans
        if "key_steps" in task_plan:
            control_flags["key_steps"] = task_plan.get("key_steps")
        if "subtasks" in task_plan:
            control_flags["subtasks"] = task_plan.get("subtasks")

        if tool_chain:
            logging.info(f"[SOMA Step {step}] tool_chain={tool_chain} refined_task={refined_task_text}")

        # 标记是否修改过图像
        image_modified = False

        for tool_name in tool_chain:
            print(f"Executing tool: {tool_name} with params: {params}")
            try:
                tool_specific_params = params.get(tool_name, params)
                if tool_name == "remove_distractor":
                    target = tool_specific_params.get("object_to_remove")
                    print(f"remove_distractor target: {target}")
                    if isinstance(target, str) and target:
                        print(f"Applying remove_distractor on target: {target}")
                        processed_img = self.tools.remove_distractor(processed_img, target)
                        image_modified = True

                elif tool_name == "visual_overlay":
                    target = tool_specific_params.get("target_object")
                    print(f"visual_overlay target: {target}")
                    if isinstance(target, str) and target:
                        print(f"Applying visual overlay on target: {target}")
                        processed_img = self.tools.apply_visual_overlay(processed_img, target, color=(0, 255, 0))
                        image_modified = True 

                elif tool_name in ("instruction_refine", "chaining_step"):
                    pass

                elif tool_name == "replace_texture":
                    target = tool_specific_params.get("target_object")
                    source = tool_specific_params.get("source", "rag_success")
                    texture_key = tool_specific_params.get("texture_key", "object_texture_png_b64")

                    pick_list = successes if source == "rag_success" else failures
                    tex_img = None
                    if pick_list:
                        info = pick_list[0].get("info", {})
                        assets = info.get("assets", {}) if isinstance(info, dict) else {}
                        tex_img = _maybe_load_image_any(assets.get(texture_key))

                    if isinstance(target, str) and target and tex_img is not None:
                        processed_img = self.tools.replace_texture(processed_img, target_object=target, texture_image=tex_img)
                        image_modified = True

                elif tool_name == "replace_background":
                    region_prompt = tool_specific_params.get("region_prompt", "floor")
                    source = tool_specific_params.get("source", "rag_success")
                    texture_key = tool_specific_params.get("texture_key", "floor_texture_png_b64")
                    alpha = float(tool_specific_params.get("alpha", 1.0))

                    pick_list = successes if source == "rag_success" else failures
                    tex_img = None
                    if pick_list:
                        info = pick_list[0].get("info", {})
                        assets = info.get("assets", {}) if isinstance(info, dict) else {}
                        tex_img = _maybe_load_image_any(assets.get(texture_key))

                    if isinstance(region_prompt, str) and region_prompt and tex_img is not None:
                        processed_img = self.tools.replace_background(
                            processed_img,
                            region_prompt=region_prompt,
                            texture_image=tex_img,
                            alpha=alpha,
                        )
                        image_modified = True

                elif tool_name == "encore":
                    control_flags["encore"] = True
                    self.retry_count += 1

                elif tool_name == "key_step_retry":
                    control_flags["key_step_retry"] = True
                    if "key_steps" in params:
                        control_flags["key_steps"] = params.get("key_steps")

                elif tool_name == "task_decompose":
                    control_flags["task_decompose"] = True
                    subtasks = params.get("subtasks")
                    if isinstance(subtasks, list) and subtasks:
                        control_flags["subtasks"] = subtasks

            except Exception as e:
                logging.error(f"[SOMA] Tool {tool_name} failed: {e}")

        # 4. 如果执行了视觉修改工具，则保存对比图
        if image_modified:
            self._save_image(
                result_rgb=processed_img,
                save_path=save_base_path,
                save_original=True,
                original_image=image
            )

        return processed_img, refined_task_text, control_flags