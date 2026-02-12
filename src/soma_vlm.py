import base64
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

# 尝试导入 OpenAI 库
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

def _extract_json(text: str) -> dict | None:
    """从大模型输出中提取 JSON 块"""
    try:
        # 尝试直接解析
        return json.loads(text)
    except Exception:
        pass

    # 尝试提取 Markdown 代码块
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text.strip())
    except Exception:
        pass

    # 尝试查找最外层大括号
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            pass
    return None


class Qwen3VLAPIClient:
    """
    SOMA 大脑：负责与 VLM (Qwen3-VL/GPT-4o) 进行交互。
    """
    # 默认配置 (会被 init 参数覆盖)
    DEFAULT_API_KEY = os.environ.get("VLM_API_KEY", "sk-xxxxxxxxxxxx")
    DEFAULT_BASE_URL = os.environ.get("VLM_BASE_URL", "https://models.sjtu.edu.cn/api/v1")
    # MODEL_ID = "qwen3vl"  # 根据你的 API 提供商调整
    MODEL_ID = "qwen3-vl-32b-instruct"

    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or self.DEFAULT_API_KEY
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.client = None
        self._init_client()

    def _init_client(self):
        if OpenAI is None:
            logging.error("OpenAI library not found. Please run `pip install openai`.")
            return
        try:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except Exception as e:
            logging.error(f"[SOMA VLM] Init failed: {e}")

    def _encode_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> str:
        """支持 Numpy/PIL/Path 转 Base64 Data URL"""
        try:
            pil_img = None
            # 1. 处理 Numpy 数组 (H, W, C)
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                pil_img = Image.fromarray(image).convert("RGB")
            # 2. 处理 PIL Image
            elif isinstance(image, Image.Image):
                pil_img = image.convert("RGB")
            # 3. 处理路径字符串
            elif isinstance(image, (str, Path)):
                path_str = str(image)
                if path_str.startswith("data:image"): return path_str
                if os.path.exists(path_str):
                    pil_img = Image.open(path_str).convert("RGB")
            
            if pil_img is not None:
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")
                return f"data:image/png;base64,{b64_data}"

            raise ValueError(f"不支持的图像类型: {type(image)}")
        except Exception as e:
            logging.error(f"Image encoding failed: {e}")
            return ""

    def _generate(self, messages: list[dict], max_tokens: int = 1024, temperature: float = 0.01) -> str:
        if not self.client: self._init_client()
        if not self.client: return "{}"
        try:
            completion = self.client.chat.completions.create(
                model=self.MODEL_ID, messages=messages, max_tokens=max_tokens, temperature=temperature
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"[SOMA VLM] API Generation Error: {e}")
            return "{}"

    # === 感知编排 (Perception) ===
    def orchestrate_perception(self, image, task_desc, rag_context="", rag_hints=None) -> dict:
        b64_img = self._encode_image(image)
        if not b64_img: return {}

        hints_str = ""
        if rag_hints:
            if rag_hints.get("success_has_object_texture"): hints_str += "- Success memory has object texture available.\n"
            if rag_hints.get("failure_has_object_texture"): hints_str += "- Failure memory suggests object texture issue.\n"

        system_prompt = (
            "You are the visual perception cortex of a robot manipulation system (SOMA).\n"
            "Goal: Preprocess input image and task prompt to help robotic policy.\n"
            "TOOLS:\n"
            "1. 'visual_overlay': Highlight objects (e.g. 'red cup'). Params: target_object\n"
            "2. 'remove_distractor': Remove confusing objects. Params: object_to_remove\n"
            "3. 'replace_texture': Replace object texture from success memory. Params: target_object, source='rag_success'\n"
            "4. 'replace_background': Replace background. Params: region_prompt, source='rag_success'\n"
            "5. 'key_step_retry': Flag to trigger key step retry in policy. Params: key_steps (optional list of steps)\n"
            "6. 'task_decompose': Flag to trigger task decomposition in policy. Params: subtasks (optional list of subtask descriptions)\n"
            "7. 'encore': No image modification, but refine task description. Params: refined_task\n"
            "OUTPUT JSON: { \"reasoning\": \"...\", \"refined_task\": \"...\", \"tool_chain\": [\"tool_name_1\"], \"params\": {...} }"
            "Tips:" 
            "1. Use 'visual_overlay' if you identify key objects. Use 'remove_distractor' if there are misleading elements. Use 'replace_texture' if success memory has better visual features. Use 'replace_background' if the background is noisy. Use 'key_step_retry' if key steps need to be retried. Use 'task_decompose' if the task should be broken down into subtasks. Use 'encore' if you want to refine the task description without modifying the image. Always provide reasoning in the output.\n"
            "2. RAG Context and Hints can guide your decision. For example, if success memory has clear object texture but failure doesn't, consider using 'replace_texture' to enhance the current image.\n"
            "3. If the image is already clear and well-aligned with the task, you can choose to do no modification but still provide a refined task description.\n"
            "4. If use visual_overlay, replace the origin object description in refined_task with the highlighted version (e.g. 'cup' -> 'highlighted green cup') to help downstream policy attend to it. We use Green highlight by default.\n"
            "5. You CAN and SHOULD combine tools. For example, use 'remove_distractor' to clear vision AND 'visual_overlay' to guide the policy. The policy will focus on the 'refined_task' after all tools are applied.\n"
            "6. You SHOULD give more detailed object descriptions of visual_overlay and remove_distractor in params to help the vision module execute them accurately. Like 'visual_overlay': {'target_object': 'the green cup between the book and the plate'}, 'remove_distractor': {'object_to_remove': 'the red cup which is the furthest from the plate'}.\n"
            "7. If visual_overlay and remove_distractor are both triggered, execute visual_overlay first to ensure the overlay is not removed by remove_distractor.\n"
            "8. Always keep the refined_task concise and focused on the main manipulation goal. If the original task is 'Pick up the red cup and place it on the plate', a good refined_task could be 'Pick up the highlighted green cup and place it on the plate', NOT 'Pick up the red cup and place it on the plate while ignoring the blue book and the yellow bowl'."
            "9. If the background is too noisy and hinders object recognition, consider using 'replace_background' to simplify the scene."
            "10. If the original task description is noisy or ambiguous, refine it to be clearer and more specific."
        )
        user_prompt = f"Current Task: {task_desc}\nContext: {rag_context}\nHints: {hints_str}\nAnalyze image. Do we need intervention?"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}, {"type": "image_url", "image_url": {"url": b64_img}}]}
        ]
        return _extract_json(self._generate(messages)) or {}

    # === 失败报告生成 (Logger 调用) ===
    def generate_failure_report(self, images: list, task_prompt: str, anchor_example: dict = None) -> dict:
        """生成失败原因分析报告"""
        valid_imgs = [self._encode_image(img) for img in images if img]
        valid_imgs = [img for img in valid_imgs if img] # filter empty
        
        if not valid_imgs:
            return {"intrinsic_analysis": {"direct_cause": "No images provided", "observation": ""}}

        anchor_text = ""
        if anchor_example:
            anchor_text = f"\nReference Success: {anchor_example.get('execution_summary', '')}"

        system_prompt = (
            "Role: Robot Failure Analyst.\n"
            "Return JSON keys: 'intrinsic_analysis' (direct_cause, observation), 'mcp_remediation' (tool_name, parameters).\n"
            "direct_cause examples: 'Visual Ambiguity', 'Distractor Interference', 'Control Error'.\n"
            "mcp_remediation tools: 'visual_overlay', 'remove_distractor', 'enhance_contrast'."
        )
        user_prompt = f"Task: {task_prompt}\nAnalyze these keyframes of a failed episode.{anchor_text}"
        
        content = [{"type": "text", "text": user_prompt}]
        for img in valid_imgs:
            content.append({"type": "image_url", "image_url": {"url": img}})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
        return _extract_json(self._generate(messages)) or {}

    # === 成功描述生成 (Logger 调用) ===
    def generate_success_description(self, images: list, task_prompt: str) -> dict:
        """生成成功执行摘要"""
        valid_imgs = [self._encode_image(img) for img in images if img]
        valid_imgs = [img for img in valid_imgs if img]
        
        if not valid_imgs:
            return {"execution_summary": "No images"}

        system_prompt = (
            "Role: Robotics Perception Analyst.\n"
            "Return JSON keys: 'visual_context_snapshot' (object_color, lighting), 'execution_summary' (one sentence why it worked)."
        )
        user_prompt = f"Task: {task_prompt}\nAnalyze these success keyframes."
        
        content = [{"type": "text", "text": user_prompt}]
        for img in valid_imgs:
            content.append({"type": "image_url", "image_url": {"url": img}})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
        return _extract_json(self._generate(messages)) or {}
    
    # === 物体检测 (SAM3 Service 调用) ===
    def detect_object(self, image: Union[str, Path, Image.Image, np.ndarray], prompt: str) -> Optional[List[int]]:
        """
        请求 VLM 返回目标物体的 Bounding Box [x1, y1, x2, y2]
        """
        b64_img = self._encode_image(image)
        if not b64_img: return None

        # 针对 Qwen-VL 等模型的检测提示词
        system_prompt = "You are a robotic vision assistant. Output the bounding box of the target object."
        user_prompt = (
            f"Detect the bounding box for: '{prompt}'. "
            "Return ONLY the box coordinates in JSON format: [ymin, xmin, ymax, xmax] (normalized 0-1000). "
            "Example: [150, 200, 450, 600]"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": user_prompt}, 
                    {"type": "image_url", "image_url": {"url": b64_img}}
                ]
            }
        ]

        # 复用你已有的 _generate 方法
        content = self._generate(messages, max_tokens=128, temperature=0.0)
        
        # 解析返回的坐标字符串
        import re
        numbers = re.findall(r"\d+", content)
        if len(numbers) >= 4:
            # 假设输入图像是 Numpy array，我们需要获取它的尺寸来反归一化
            # 注意：这里我们传入的是 image 对象，需要获取宽高
            h, w = 0, 0
            if isinstance(image, np.ndarray):
                h, w = image.shape[:2]
            elif isinstance(image, Image.Image):
                w, h = image.size
            else:
                # 如果是路径，暂时无法获取尺寸，只能返回归一化坐标或失败
                # 建议调用此方法前先转成 Numpy/PIL
                return None
            
            if h > 0 and w > 0:
                ymin = float(numbers[0]) / 1000 * h
                xmin = float(numbers[1]) / 1000 * w
                ymax = float(numbers[2]) / 1000 * h
                xmax = float(numbers[3]) / 1000 * w
                return [int(xmin), int(ymin), int(xmax), int(ymax)]
        
        return None