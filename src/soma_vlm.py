import os
import io
import json
import base64
import logging
from typing import List, Union, Optional, Dict
from pathlib import Path
from PIL import Image

# 尝试导入 OpenAI SDK
try:
    from openai import OpenAI
except ImportError:
    raise ImportError("请安装 openai 库: pip install openai")

# =========================
# 辅助函数
# =========================
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
    
    # 尝试寻找最外层大括号
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
    SOMA VLM Client (API Only)
    负责所有与大模型的交互，包括决策(Brain)和复盘(Logger)。
    """
    
    # ================= 配置区域 =================
    # 请替换为你的真实 API Key 和 Base URL
    # 这里默认适配兼容 OpenAI 格式的 Qwen 服务 (如 vLLM, FastChat 或 官方API)
    MODEL_ID = "qwen3vl" 
    API_KEY = "sk-dJ9PDHKGeP7xfsO4Zv7jNw" 
    BASE_URL = "https://models.sjtu.edu.cn/api/v1" 
    # ===========================================

    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or self.API_KEY
        self.base_url = base_url or self.BASE_URL
        self.client = None
        
        # 延迟初始化，防止导入时报错
        self._init_client()

    def _init_client(self):
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            logging.info(f"[SOMA VLM] API Client initialized connecting to: {self.base_url}")
        except Exception as e:
            logging.error(f"[SOMA VLM] API Client init failed: {e}")

    def _encode_image(self, image: Union[str, Path, Image.Image]) -> str:
        """将图片(路径或PIL对象)转换为 Base64 Data URL"""
        try:
            img_byte_arr = io.BytesIO()
            
            if isinstance(image, (str, Path)):
                # 如果是路径，读取文件
                with open(image, "rb") as f:
                    img_bytes = f.read()
                # 重新打开以确认格式 (可选，为了稳健)
                Image.open(io.BytesIO(img_bytes)).verify()
                b64_str = base64.b64encode(img_bytes).decode('utf-8')
                mime_type = "image/jpeg" # 简化处理，默认 jpeg 或 png 兼容性好
            
            elif isinstance(image, Image.Image):
                # 如果是 PIL Image
                image.save(img_byte_arr, format='PNG')
                b64_str = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                mime_type = "image/png"
            else:
                raise ValueError(f"不支持的图像类型: {type(image)}")

            return f"data:{mime_type};base64,{b64_str}"
        
        except Exception as e:
            logging.error(f"Image encoding failed: {e}")
            return ""

    def _generate(self, messages: list[dict], max_tokens: int = 1024, temperature: float = 0.01) -> str:
        """通用的 API 调用函数"""
        if not self.client:
            self._init_client()
            if not self.client: return "{}"

        try:
            completion = self.client.chat.completions.create(
                model=self.MODEL_ID,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature, # 低温以保证 JSON 格式稳定
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"[SOMA VLM] API Generation Error: {e}")
            return "{}"

    # =================================================================
    #  功能 1: 成功经验生成 (供 ExperienceLogger 使用)
    #  Reference: lerobot_eval.py Line 140+
    # =================================================================
    def generate_success_description(self, 
                                   images: Union[List[str], List[Image.Image]], 
                                   task_prompt: str) -> Dict:
        """
        当 Episode 成功时，分析关键帧，提取视觉上下文快照。
        """
        if not images: return {}

        # 1. 构建 System Prompt (严格参考 eval 文件)
        system_prompt = (
            "Role: Robotics Perception Analyst. "
            "You will read multiple keyframes of a successful episode and produce a concise, structured JSON. "
            "Return STRICT JSON with keys: 'visual_context_snapshot' and 'execution_summary'. "
            "visual_context_snapshot should capture comfort-zone attributes such as target_object_color, "
            "background_color, lighting_condition, object_texture, spatial_layout. "
            "execution_summary is a short textual summary of why it succeeded."
        )

        # 2. 构建 User Prompt
        user_prompt = (
            f"Task: {task_prompt}\n"
            "Given the chronological keyframes, extract the visual context snapshot describing the object's color/texture, "
            "background, lighting, and spatial layout, plus a one-sentence execution_summary. "
            "Output JSON only with the required keys."
        )

        # 3. 组装消息 (多图模式)
        content = [{"type": "text", "text": user_prompt}]
        for img in images:
            b64_url = self._encode_image(img)
            if b64_url:
                content.append({"type": "image_url", "image_url": {"url": b64_url}})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]

        # 4. 调用与解析
        raw_text = self._generate(messages)
        data = _extract_json(raw_text)
        
        # 兜底返回格式
        if isinstance(data, dict):
            return {
                "visual_context_snapshot": data.get("visual_context_snapshot", {}),
                "execution_summary": data.get("execution_summary", "Success verified.")
            }
        return {"visual_context_snapshot": {}, "execution_summary": raw_text[:100]}

    # =================================================================
    #  功能 2: 失败报告生成 (供 ExperienceLogger 使用)
    #  Reference: lerobot_eval.py Line 200+
    # =================================================================
    def generate_failure_report(self, 
                              images: Union[List[str], List[Image.Image]], 
                              task_prompt: str, 
                              anchor_example: dict = None) -> Dict:
        """
        当 Episode 失败时，进行归因分析，并推荐 MCP 工具。
        """
        if not images: return {}

        # 1. 准备 Anchor 文本 (如果有成功案例对比)
        anchor_text = ""
        if anchor_example:
            anchor_text = (
                f"\nReference Success Case (Anchor): "
                f"visual_context={json.dumps(anchor_example.get('visual_context_snapshot', {}))}, "
                f"summary={anchor_example.get('execution_summary', '')}"
            )

        # 2. 构建 System Prompt (核心：要求输出 mcp_remediation)
        system_prompt = (
            "Role: Robot Failure Analyst with Counterfactual Reasoning. "
            "You MUST return STRICT JSON with keys: "
            "'intrinsic_analysis', 'comparative_analysis', 'mcp_remediation'. "
            "intrinsic_analysis: direct_cause + observation describing what went wrong.\n"
            "comparative_analysis: if anchor provided, identify variable_gap {attribute, failed_value, success_value}. "
            "mcp_remediation: recommend an MCP tool to fix the issue, choose from "
            "[visual_overlay, remove_distractor, instruction_refine], provide tool_name and parameters."
        )

        # 3. 构建 User Prompt
        user_prompt = (
            f"Task: {task_prompt}\n"
            "You are given chronological keyframes of a failed episode."
            f"{anchor_text}\n"
            "Analyze the failure. Return JSON only."
        )

        # 4. 组装消息
        content = [{"type": "text", "text": user_prompt}]
        for img in images:
            b64_url = self._encode_image(img)
            if b64_url:
                content.append({"type": "image_url", "image_url": {"url": b64_url}})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]

        # 5. 调用与解析
        raw_text = self._generate(messages)
        data = _extract_json(raw_text)

        # 兜底返回格式
        if isinstance(data, dict):
            return {
                "intrinsic_analysis": data.get("intrinsic_analysis", {"direct_cause": "Unknown"}),
                "comparative_analysis": data.get("comparative_analysis", {}),
                "mcp_remediation": data.get("mcp_remediation", {}) # 这是 SOMA 闭环的关键
            }
        
        return {
            "intrinsic_analysis": {"direct_cause": "Parsing Error", "raw": raw_text[:100]},
            "comparative_analysis": {},
            "mcp_remediation": {}
        }

    # =================================================================
    #  功能 3: 战略编排 (供 PerceptionModule 使用)
    #  这是 Agent 在运行时的“大脑”
    # =================================================================
    def orchestrate_perception(self, 
                             image: Union[str, Image.Image], 
                             task_desc: str, 
                             rag_context: str = "") -> Dict:
        """
        SOMA Brain 核心: 根据当前图像和 RAG 记忆，决定是否调用 MCP 工具。
        """
        system_prompt = (
            "You are SOMA, an adaptive robot agent.\n"
            "Available Tools:\n"
            "1. 'visual_overlay': Use if target is hard to see. Params: 'target_object'.\n"
            "2. 'remove_distractor': Use if clutter/distractors exist. Params: 'object_to_remove'.\n"
            "3. 'chaining_step': Use if task is long-horizon/ambiguous. Refine 'refined_task' to the immediate sub-step.\n"
            "4. 'encore': Use if you detect execution failure (slip, miss). This triggers a retry.\n"
            "\n"
            "Output JSON: { 'tool_chain': ['tool1', ...], 'params': {...}, 'refined_task': '...' }"
        )
        
        user_prompt = f"Task: {task_desc}\nContext: {rag_context}\nDecide on the perception strategy."
        
        b64_url = self._encode_image(image)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": b64_url}}
            ]}
        ]
        
        raw_text = self._generate(messages, max_tokens=512)
        return _extract_json(raw_text) or {"tool_chain": [], "params": {}, "refined_task": task_desc}

# 简单测试
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # 创建一个空的黑图用于测试
    dummy_img = Image.new('RGB', (100, 100))
    
    client = Qwen3VLAPIClient()
    
    # 测试成功生成
    print("Testing Success Gen...")
    res = client.generate_success_description([dummy_img], "Pick up the cup")
    print(json.dumps(res, indent=2))
    
    # 测试失败生成
    print("\nTesting Failure Gen...")
    res = client.generate_failure_report([dummy_img], "Pick up the cup")
    print(json.dumps(res, indent=2))