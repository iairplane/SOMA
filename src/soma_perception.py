import base64
import io
import logging
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

    本模块只负责：
    - 产出 processed_image + refined_task
    - 产出 control_flags 给 eval 控制流 (encore/key_step_retry/task_decompose)
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

    def process_frame(
        self,
        image: np.ndarray,
        current_task: str,
        step: int,
        rag_context: Dict | None = None,
    ) -> Tuple[np.ndarray, str, Dict]:
        rag_context = rag_context or {}

        # 1) rag text + rag hints
        rag_str = ""
        rag_hints: dict[str, Any] = {}

        failures = rag_context.get("failure") or []
        successes = rag_context.get("success") or []

        if failures:
            diags = [str(f.get("diagnosis", "")) for f in failures[:2] if f.get("diagnosis")]
            if diags:
                rag_str = f"Warning! Past failures causes: {'; '.join(diags)}."

        # Provide a compact hint for available textures to the VLM.
        # VLM 只需要知道“有/无”与推荐 key，不需要塞大 base64。
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

        # 2) VLM plan
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

        # Control-flow plans from VLM (optional)
        if "key_steps" in task_plan:
            control_flags["key_steps"] = task_plan.get("key_steps")
        if "subtasks" in task_plan:
            control_flags["subtasks"] = task_plan.get("subtasks")

        if tool_chain:
            logging.info(f"[SOMA Step {step}] tool_chain={tool_chain} refined_task={refined_task_text}")

        for tool_name in tool_chain:
            try:
                if tool_name == "remove_distractor":
                    target = params.get("object_to_remove")
                    if isinstance(target, str) and target:
                        processed_img = self.tools.remove_distractor(processed_img, target)

                elif tool_name == "visual_overlay":
                    target = params.get("target_object")
                    if isinstance(target, str) and target:
                        processed_img = self.tools.apply_visual_overlay(processed_img, target, color=(0, 255, 0))

                elif tool_name in ("instruction_refine", "chaining_step"):
                    # refined_task_text already handled
                    pass

                elif tool_name == "replace_texture":
                    target = params.get("target_object")
                    source = params.get("source", "rag_success")
                    texture_key = params.get("texture_key", "object_texture_png_b64")

                    pick_list = successes if source == "rag_success" else failures
                    tex_img = None
                    if pick_list:
                        info = pick_list[0].get("info", {})
                        assets = info.get("assets", {}) if isinstance(info, dict) else {}
                        tex_img = _maybe_load_image_any(assets.get(texture_key))

                    if isinstance(target, str) and target and tex_img is not None:
                        processed_img = self.tools.replace_texture(processed_img, target_object=target, texture_image=tex_img)

                elif tool_name == "replace_background":
                    region_prompt = params.get("region_prompt", "floor")
                    source = params.get("source", "rag_success")
                    texture_key = params.get("texture_key", "floor_texture_png_b64")
                    alpha = float(params.get("alpha", 1.0))

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

        return processed_img, refined_task_text, control_flags
