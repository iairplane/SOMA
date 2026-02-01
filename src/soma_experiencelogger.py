import logging
import json
import threading
import cv2  # 需要 opencv-python
import numpy as np
from pathlib import Path
from typing import Optional, List, Union, Dict
from datetime import datetime
from PIL import Image

# 引入我们之前定义的模块
try:
    from soma_encoder import AdvancedEmbeddingEncoder
    from soma_memory import MemoryBank
    from soma_vlm import Qwen3VLAPIClient
except ImportError:
    pass

class ExperienceLogger:
    """
    SOMA 经验记录器 (The Scribe) - 完整版
    
    职责:
    1. 数据接收: 接收 Episode 视频路径、任务、结果。
    2. 自动采样: 如果未提供内存图片，自动从视频采样关键帧。
    3. 智能诊断: 失败时调用 VLM 生成结构化归因报告。
    4. 向量化: 调用 Encoder 生成检索向量。
    5. 入库: 存入 MemoryBank。
    """
    
    def __init__(self, 
                 memory_bank: MemoryBank, 
                 encoder: AdvancedEmbeddingEncoder,
                 vlm_client: Qwen3VLAPIClient):
        self.memory = memory_bank
        self.encoder = encoder
        self.vlm = vlm_client
        
    def _extract_frames_from_video(self, video_path: str, num_frames: int = 5) -> List[Image.Image]:
        """
        辅助函数: 从视频文件中均匀提取 N 帧 PIL Image
        """
        frames = []
        if not video_path or not Path(video_path).exists():
            return []
            
        try:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0: return []
            
            # 均匀采样索引
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    # BGR -> RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb))
            cap.release()
        except Exception as e:
            logging.error(f"[Logger] 视频帧提取失败: {e}")
            
        return frames

    def _run_diagnosis(self, task: str, frames: List[Image.Image]) -> str:
        """
        [核心修复] 调用 VLM 生成诊断报告，并提取核心原因字符串。
        """
        if not self.vlm: 
            return "Analysis unavailable (No VLM)."
        
        if not frames:
            return "Analysis unavailable (No frames)."

        # 1. 调用 VLM 的标准接口 (复用 soma_vlm.py 中的 generate_failure_report)
        # 这会返回一个包含 intrinsic_analysis, mcp_remediation 的字典
        try:
            report = self.vlm.generate_failure_report(
                images=frames,
                task_prompt=task,
                anchor_example=None # 未来可以扩展传入成功案例
            )
            
            # 2. 解析结构化输出
            # 目标格式: "Direct Cause: <cause>. Observation: <obs>"
            analysis = report.get("intrinsic_analysis", {})
            direct_cause = analysis.get("direct_cause", "Unknown cause")
            observation = analysis.get("observation", "")
            
            # 3. 提取 MCP 建议 (可选，存入 info 以便后续分析)
            # mcp_suggestion = report.get("mcp_remediation", {})
            
            # 生成存入 MemoryBank 的简洁诊断字符串
            diagnosis_str = f"{direct_cause}: {observation}"
            
            # 限制长度，防止污染检索展示
            if len(diagnosis_str) > 200:
                diagnosis_str = diagnosis_str[:197] + "..."
                
            return diagnosis_str

        except Exception as e:
            logging.error(f"[Logger] VLM Diagnosis Error: {e}")
            return "Diagnosis failed due to API error."

    def log_episode(self,
                    task_desc: str,
                    success: bool,
                    video_path: Union[str, Path],
                    keyframe_path: Union[str, Path],
                    frames: List[Image.Image] = None, 
                    additional_info: dict = None):
        """
        记录单个 Episode。
        Args:
            frames: 如果为 None，则自动从 video_path 读取。
        """
        try:
            # 1. 准备图像资源
            # 如果内存中没有帧，则从视频读取 (保证 Robustness)
            if not frames or len(frames) == 0:
                # 采样 5 帧用于诊断 (首、中、尾)
                diag_frames = self._extract_frames_from_video(video_path, num_frames=5)
                # 取第 0 帧用于 RAG 索引
                index_img = diag_frames[0] if diag_frames else None
            else:
                index_img = frames[0]
                # 如果传入帧太多，降采样到 5 帧以节省 Token
                if len(frames) > 5:
                    indices = np.linspace(0, len(frames)-1, 5, dtype=int)
                    diag_frames = [frames[i] for i in indices]
                else:
                    diag_frames = frames

            # 如果还是没图 (视频读取失败)，尝试读 keyframe 文件
            if index_img is None:
                try:
                    if Path(keyframe_path).exists():
                        index_img = Image.open(keyframe_path).convert("RGB")
                        diag_frames = [index_img]
                    else:
                        logging.warning(f"[Logger] 无图像数据，跳过记录: {task_desc}")
                        return
                except:
                    return

            # 2. 生成 Embedding (Mapping)
            # 使用 index_img (首帧) 和 task_desc 生成索引向量
            embedding = self.encoder.embed(index_img, task_desc)
            
            # 3. 生成诊断 (Diagnosis)
            diagnosis = ""
            if not success:
                logging.info(f"[Logger] 正在诊断失败任务: {task_desc} ...")
                # === 核心调用 ===
                diagnosis = self._run_diagnosis(task_desc, diag_frames)
                logging.info(f"[Logger] VLM 诊断结果: {diagnosis}")
            
            # 4. 存入 MemoryBank (Storage)
            self.memory.add_experience(
                embedding=embedding,
                task_desc=task_desc,
                success=success,
                video_path=video_path,
                keyframe_path=keyframe_path,
                diagnosis=diagnosis,
                additional_info=additional_info
            )
            
        except Exception as e:
            logging.error(f"[Logger] 记录过程发生未捕获异常: {e}", exc_info=True)

# ================= 单元测试 =================
if __name__ == "__main__":
    # 模拟测试
    logging.basicConfig(level=logging.INFO)
    
    # 1. Mock 组件
    class MockMem:
        def add_experience(self, **kwargs):
            print(f"✅ Memory Save: Success={kwargs['success']}, Diag={kwargs['diagnosis']}")

    class MockEnc:
        def embed(self, img, txt): return [0.1] * 1168

    class MockVLM:
        def generate_failure_report(self, images, task_prompt, anchor_example=None):
            return {
                "intrinsic_analysis": {
                    "direct_cause": "Visual Mismatch",
                    "observation": "The robot grasped the shadow instead of the cup."
                }
            }

    # 2. 运行
    logger = ExperienceLogger(MockMem(), MockEnc(), MockVLM())
    
    # 创建假图
    fake_img = Image.new('RGB', (100, 100), color='red')
    
    print("--- Testing Failure Log ---")
    logger.log_episode(
        task_desc="Pick up red cup",
        success=False,
        video_path="dummy.mp4",
        keyframe_path="dummy.jpg",
        frames=[fake_img] # 传入内存图
    )
