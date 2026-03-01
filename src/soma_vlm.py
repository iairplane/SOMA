"""
SOMA: Self-Organizing Memory Agent
==================================

Description:
------------
This module implements the Vision-Language Model (VLM) API Client, which acts 
as the 'Brain' of the SOMA framework. It is responsible for high-level visual 
reasoning, task decomposition, semantic failure diagnosis, and zero-shot object 
bounding box detection.



Key Features:
- Perception Orchestration: Evaluates the current scene against Retrieval-Augmented 
  Generation (RAG) hints and outputs a structured tool-chain plan (e.g., identifying 
  distractors or suggesting visual overlays).
- Experience Summarization: Generates structured root-cause analysis reports for 
  failed episodes and execution summaries for successful ones.
- Modality Handling: Robustly encodes inputs from various formats (NumPy arrays, 
  PIL Images, or file paths) into base64 Data URLs for API transmission.
"""

import base64
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

# Attempt to import the OpenAI library
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

def _extract_json(text: str) -> dict | None:
    """Extract a JSON block from the Large Language Model's output string."""
    try:
        # Attempt direct parsing
        return json.loads(text)
    except Exception:
        pass

    # Attempt to extract from Markdown code blocks
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text.strip())
    except Exception:
        pass

    # Attempt to find the outermost curly braces
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
    SOMA Brain: Responsible for interacting with the Vision-Language Model (e.g., Qwen3-VL / GPT-4o).
    """
    # Default configuration (will be overridden by init parameters)
    DEFAULT_API_KEY = os.environ.get("VLM_API_KEY", "sk-xxxxx")
    DEFAULT_BASE_URL = os.environ.get("VLM_BASE_URL", "https://xxxx.com/compatible-mode/v1")
    # MODEL_ID = "qwen3vl"  # Adjust according to your API provider
    MODEL_ID = "qwen3-vl-32b-instruct"

    def __init__(self, api_key: str = None, base_url: str = None, model_id: str = None):
        self.api_key = api_key or self.DEFAULT_API_KEY
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.model_id = model_id or self.MODEL_ID
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
        """Supports converting Numpy/PIL/Path to Base64 Data URL"""
        try:
            pil_img = None
            # 1. Process Numpy array (H, W, C)
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                pil_img = Image.fromarray(image).convert("RGB")
            # 2. Process PIL Image
            elif isinstance(image, Image.Image):
                pil_img = image.convert("RGB")
            # 3. Process file path string
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

            raise ValueError(f"Unsupported image type: {type(image)}")
        except Exception as e:
            logging.error(f"Image encoding failed: {e}")
            return ""

    def _generate(self, messages: list[dict], max_tokens: int = 1024, temperature: float = 0.01) -> str:
        if not self.client: self._init_client()
        if not self.client: return "{}"
        try:
            completion = self.client.chat.completions.create(
                model=self.model_id, messages=messages, max_tokens=max_tokens, temperature=temperature
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"[SOMA VLM] API Generation Error: {e}")
            return "{}"

    # === Perception Orchestration ===
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
            "1. 'remove_distractor': Remove ALL confusing objects/distractors. Params: objects_to_remove (List[str])\n"
            "2. 'visual_overlay': Highlight objects (e.g. 'red cup'). Params: target_object\n"
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
            "4. If use visual_overlay, replace the origin object description in refined_task with the highlighted version (e.g. 'cup' -> 'green highlighted cup') to help downstream policy attend to it. We use Green highlight by default. If use else tools, do NOT output 'highlighted' word in the task description.\n"
            "5. You CAN and SHOULD combine tools. For example, use 'remove_distractor' to clear vision AND 'visual_overlay' to guide the policy. The policy will focus on the 'refined_task' after all tools are applied.\n"
            "6. You SHOULD give more detailed object descriptions of visual_overlay and remove_distractor in params to help the vision module execute them accurately. E.g., 'visual_overlay': {'target_object': 'the green cup'}, 'remove_distractor': {'objects_to_remove': ['the red cup on the left', 'the red cup on the right']}.\n"
            "7. If visual_overlay and remove_distractor are both triggered, execute visual_overlay first to ensure the overlay is not removed by remove_distractor.\n"
            "8. Always keep the refined_task concise and focused on the main manipulation goal. If the original task is 'Pick up the red cup and place it on the plate', a good refined_task could be 'Pick up the highlighted green cup and place it on the plate', NOT 'Pick up the red cup and place it on the plate while ignoring the blue book and the yellow bowl'."
            "9. If the background is too noisy and hinders object recognition, consider using 'replace_background' to simplify the scene."
            "10. If the original task description is noisy or ambiguous, refine it to be clearer and more specific."
            "11. CRITICAL: If there are multiple distractors of the same type (e.g. multiple wrong bowls), you MUST list EACH of them individually in the 'objects_to_remove' list. Do NOT use a single generic phrase like 'all red bowls'. Instead, use specific descriptions for each one so the vision system can target them precisely (e.g. ['red bowl on top left', 'red bowl on bottom right'])."
        )
        user_prompt = f"Current Task: {task_desc}\nContext: {rag_context}\nHints: {hints_str}\nAnalyze image. Do we need intervention?"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}, {"type": "image_url", "image_url": {"url": b64_img}}]}
        ]
        return _extract_json(self._generate(messages)) or {}

    # === Failure Report Generation (Called by Logger) ===
    def generate_failure_report(self, images: list, task_prompt: str, anchor_example: dict = None) -> dict:
        """Generate a root-cause analysis report for a failed episode"""
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

    # === Success Description Generation (Called by Logger) ===
    def generate_success_description(self, images: list, task_prompt: str) -> dict:
        """Generate an execution summary for a successful episode"""
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
    
    # === Object Detection (Supports SAM3 Service) ===
    def detect_object(self, image: Union[str, Path, Image.Image, np.ndarray], prompt: str) -> Optional[List[int]]:
        """
        Request the VLM to return the Bounding Box [x1, y1, x2, y2] of the target object.
        """
        b64_img = self._encode_image(image)
        if not b64_img: return None

        # Detection prompt tailored for models like Qwen-VL
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

        # Reuse the existing _generate method
        content = self._generate(messages, max_tokens=128, temperature=0.0)
        
        # Parse the returned coordinate string
        import re
        numbers = re.findall(r"\d+", content)
        if len(numbers) >= 4:
            # Assuming the input image is a Numpy array, we need its dimensions to denormalize
            # Note: We are passing the image object here, so we need to extract width and height
            h, w = 0, 0
            if isinstance(image, np.ndarray):
                h, w = image.shape[:2]
            elif isinstance(image, Image.Image):
                w, h = image.size
            else:
                # If it's a path, we cannot easily get dimensions here, so return None
                # It is recommended to convert to Numpy/PIL before calling this method
                return None
            
            if h > 0 and w > 0:
                ymin = float(numbers[0]) / 1000 * h
                xmin = float(numbers[1]) / 1000 * w
                ymax = float(numbers[2]) / 1000 * h
                xmax = float(numbers[3]) / 1000 * w
                return [int(xmin), int(ymin), int(xmax), int(ymax)]
        
        return None
