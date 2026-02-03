import os
import io
import json
import base64
import logging
from typing import List, Union, Optional, Dict, Any
from pathlib import Path
from PIL import Image

# 尝试导入 OpenAI SDK
try:
    from openai import OpenAI
except ImportError:
    raise ImportError("请安装 openai 库: pip install openai")


def _extract_json(text: str) -> dict | None:
    """鲁棒的 JSON 提取器，能处理 Markdown 代码块"""
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    return None


class Qwen3VLAPIClient:
    """SOMA VLM Client (API Only)

    负责：
    - 在线编排：orchestrate_perception（输出 tool_chain + params + refined_task + 可选 task_plan）
    - 离线复盘：generate_failure_report / generate_success_description
    """

    MODEL_ID = os.environ.get("SOMA_VLM_MODEL", "qwen3vl")
    API_KEY = os.environ.get("SOMA_VLM_API_KEY", "")
    BASE_URL = os.environ.get("SOMA_VLM_BASE_URL", "https://models.sjtu.edu.cn/api/v1")

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or self.API_KEY
        self.base_url = base_url or self.BASE_URL
        self.client: OpenAI | None = None
        self._init_client()

    def _init_client(self):
        try:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            logging.info(f"[SOMA VLM] API Client initialized connecting to: {self.base_url}")
        except Exception as e:
            logging.error(f"[SOMA VLM] API Client init failed: {e}")
            self.client = None

    def _encode_image(self, image: Union[str, Path, Image.Image]) -> str:
        try:
            img_byte_arr = io.BytesIO()

            if isinstance(image, (str, Path)):
                with open(image, "rb") as f:
                    img_bytes = f.read()
                Image.open(io.BytesIO(img_bytes)).verify()
                b64_str = base64.b64encode(img_bytes).decode("utf-8")
                mime_type = "image/jpeg"

            elif isinstance(image, Image.Image):
                image.save(img_byte_arr, format="PNG")
                b64_str = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
                mime_type = "image/png"
            else:
                raise ValueError(f"不支持的图像类型: {type(image)}")

            return f"data:{mime_type};base64,{b64_str}"

        except Exception as e:
            logging.error(f"Image encoding failed: {e}")
            return ""

    def _generate(self, messages: list[dict], max_tokens: int = 1024, temperature: float = 0.01) -> str:
        if not self.client:
            self._init_client()
            if not self.client:
                return "{}"

        try:
            completion = self.client.chat.completions.create(
                model=self.MODEL_ID,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"[SOMA VLM] API Generation Error: {e}")
            return "{}"

    def generate_success_description(self, images: Union[List[str], List[Image.Image]], task_prompt: str) -> Dict:
        if not images:
            return {}

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

        content = [{"type": "text", "text": user_prompt}]
        for img in images:
            b64_url = self._encode_image(img)
            if b64_url:
                content.append({"type": "image_url", "image_url": {"url": b64_url}})

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]

        raw_text = self._generate(messages)
        data = _extract_json(raw_text)

        if isinstance(data, dict):
            return {
                "visual_context_snapshot": data.get("visual_context_snapshot", {}),
                "execution_summary": data.get("execution_summary", "Success verified."),
            }
        return {"visual_context_snapshot": {}, "execution_summary": raw_text[:100]}

    def generate_failure_report(
        self,
        images: Union[List[str], List[Image.Image]],
        task_prompt: str,
        anchor_example: dict | None = None,
    ) -> Dict:
        if not images:
            return {}

        anchor_text = ""
        if anchor_example:
            anchor_text = (
                f"\nReference Success Case (Anchor): "
                f"visual_context={json.dumps(anchor_example.get('visual_context_snapshot', {}))}, "
                f"summary={anchor_example.get('execution_summary', '')}"
            )

        system_prompt = (
            "Role: Robot Failure Analyst with Counterfactual Reasoning. "
            "You MUST return STRICT JSON with keys: "
            "'intrinsic_analysis', 'comparative_analysis', 'mcp_remediation'. "
            "intrinsic_analysis: direct_cause + observation describing what went wrong.\n"
            "comparative_analysis: if anchor provided, identify variable_gap {attribute, failed_value, success_value}. "
            "mcp_remediation: recommend an MCP tool to fix the issue."
        )

        user_prompt = (
            f"Task: {task_prompt}\n"
            "You are given chronological keyframes of a failed episode."
            f"{anchor_text}\n"
            "Analyze the failure. Return JSON only."
        )

        content = [{"type": "text", "text": user_prompt}]
        for img in images:
            b64_url = self._encode_image(img)
            if b64_url:
                content.append({"type": "image_url", "image_url": {"url": b64_url}})

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]

        raw_text = self._generate(messages)
        data = _extract_json(raw_text)

        if isinstance(data, dict):
            return {
                "intrinsic_analysis": data.get("intrinsic_analysis", {"direct_cause": "Unknown"}),
                "comparative_analysis": data.get("comparative_analysis", {}),
                "mcp_remediation": data.get("mcp_remediation", {}),
            }

        return {
            "intrinsic_analysis": {"direct_cause": "Parsing Error", "raw": raw_text[:100]},
            "comparative_analysis": {},
            "mcp_remediation": {},
        }

    def orchestrate_perception(
        self,
        image: Union[str, Image.Image],
        task_desc: str,
        rag_context: str = "",
        rag_hints: dict | None = None,
    ) -> Dict[str, Any]:
        """在线编排：输出 MCP tool_chain + params + refined_task + 可选 task_plan。

        新增工具：
        - replace_texture: 相似物体替代
          params: {"target_object": str, "source": "rag_success"|"rag_failure", "texture_key": "object_texture_png_b64"}
        - replace_background: 背景替换
          params: {"region_prompt": "floor"|"table"|str, "source": "rag_success", "texture_key": "floor_texture_png_b64", "alpha": float}
        - key_step_retry: 关键步骤重做（控制流信号）
          params: {"key_steps": {"start": int, "end": int, "timeout_grace": int}}
        - task_decompose: 长程拆分（可选输出 subtasks）
          params: {"subtasks": [str, ...]}

        注意：VLM 输出只做“建议”，最终执行/回退逻辑在 soma_eval.py。
        """

        system_prompt = (
            "You are SOMA, an adaptive robot agent.\n"
            "You see a single current RGB image. You may propose a perception/tool intervention.\n\n"
            "Available Tools:\n"
            "1. 'visual_overlay': highlight target. Params: {target_object}\n"
            "2. 'remove_distractor': remove clutter. Params: {object_to_remove}\n"
            "3. 'replace_texture': when encountering novel target appearance, replace its surface using a retrieved texture from memory. Params: {target_object, source, texture_key}\n"
            "4. 'replace_background': when floor/table texture differs from memory, replace background region using memory texture. Params: {region_prompt, source, texture_key, alpha}\n"
            "5. 'chaining_step': refine instruction to immediate sub-step. Use refined_task.\n"
            "6. 'task_decompose': output a list of subtasks in params.subtasks and also set refined_task to the current subtask.\n"
            "7. 'key_step_retry': request key-step rollback/retry (control-flow). Params: {key_steps:{start,end,timeout_grace}}\n"
            "8. 'encore': if execution failure is detected (slip, miss), request retry (control-flow).\n\n"
            "Output STRICT JSON only:\n"
            "{\n"
            "  'tool_chain': ['tool1', ...],\n"
            "  'params': { ... },\n"
            "  'refined_task': '...',\n"
            "  'task_plan': { 'subtasks': [...], 'key_steps': {...} }\n"
            "}\n"
            "If unsure, return tool_chain=[] and refined_task=original task."
        )

        hints_text = ""
        if rag_hints:
            try:
                hints_text = f"\nRAG Hints (json): {json.dumps(rag_hints, ensure_ascii=False)[:800]}"
            except Exception:
                hints_text = ""

        user_prompt = f"Task: {task_desc}\nContext: {rag_context}{hints_text}\nDecide a strategy."

        b64_url = self._encode_image(image)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": b64_url}},
                ],
            },
        ]

        raw_text = self._generate(messages, max_tokens=700)
        data = _extract_json(raw_text)
        if not isinstance(data, dict):
            return {"tool_chain": [], "params": {}, "refined_task": task_desc}

        # 基本兜底
        tool_chain = data.get("tool_chain")
        if not isinstance(tool_chain, list):
            tool_chain = []
        params = data.get("params") if isinstance(data.get("params"), dict) else {}
        refined_task = data.get("refined_task") if isinstance(data.get("refined_task"), str) else task_desc
        task_plan = data.get("task_plan") if isinstance(data.get("task_plan"), dict) else {}

        return {
            "tool_chain": tool_chain,
            "params": params,
            "refined_task": refined_task,
            "task_plan": task_plan,
        }
