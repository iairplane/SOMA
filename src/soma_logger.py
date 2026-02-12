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

    def _run_failure_diagnosis(self, task: str, frames: List[Image.Image]) -> str:
        """调用 VLM 生成失败诊断报告"""
        if not self.vlm or not frames:
            return "Analysis unavailable."
        try:
            report = self.vlm.generate_failure_report(images=frames, task_prompt=task)
            analysis = report.get("intrinsic_analysis", {})
            return f"{analysis.get('direct_cause', 'Unknown')}: {analysis.get('observation', '')}"
        except Exception as e:
            logging.error(f"[Logger] Failure Diagnosis Error: {e}")
            return "Diagnosis failed."
        
    def _run_success_summary(self, task: str, frames: List[Image.Image]) -> str:
        """调用 VLM 生成成功执行摘要"""
        if not self.vlm or not frames:
            return "Success confirmed."
        try:
            report = self.vlm.generate_success_description(images=frames, task_prompt=task)
            return report.get("execution_summary", "Task completed successfully.")
        except Exception as e:
            logging.error(f"[Logger] Success Summary Error: {e}")
            return "Success confirmed (Summary failed)."
        
    def log_episode(self,
                    task_desc: str,
                    success: bool,
                    video_path: Union[str, Path],
                    keyframe_path: Union[str, Path],
                    frames: List[Image.Image] = None, 
                    additional_info: dict = None):
        """记录 Episode，支持同步诊断并入库"""
        try:
            # 1. 资源准备 (自动从视频采样或使用传入帧)
            if not frames:
                diag_frames = self._extract_frames_from_video(video_path, num_frames=5)
            else:
                diag_frames = frames[:5] # 限制帧数节省 Token
            
            index_img = diag_frames[0] if diag_frames else None
            
            # 兜底：如果还是没图，尝试读取磁盘上的 keyframe_path
            if index_img is None and Path(keyframe_path).exists():
                index_img = Image.open(keyframe_path).convert("RGB")
                diag_frames = [index_img]

            if not index_img:
                logging.warning(f"[Logger] 放弃记录：任务 {task_desc} 缺失图像数据")
                return

            # 2. 生成 RAG 索引向量
            embedding = self.encoder.embed(index_img, task_desc)
            
            # 3. 分支诊断逻辑 (核心修复点：确保方法名匹配)
            if not success:
                logging.info(f"[Logger] 正在诊断失败任务: {task_desc}")
                diagnosis = self._run_failure_diagnosis(task_desc, diag_frames)
            else:
                # 成功时也可以存入摘要，帮助 RAG 检索正向经验
                diagnosis = self._run_success_summary(task_desc, diag_frames)
            print(f"📌 诊断结果: {diagnosis}")
            # 4. 落地存储：调用 MemoryBank 写入磁盘
            self.memory.add_experience(
                embedding=embedding,
                task_desc=task_desc,
                success=success,
                video_path=video_path,
                keyframe_path=keyframe_path,
                diagnosis=diagnosis,
                info=additional_info
            )
            logging.info(f"✅ [Logger] 经验保存: {task_desc} (Success={success})")
            
        except Exception as e:
            logging.error(f"[Logger] 关键错误：记录 Episode 失败 - {e}", exc_info=True)

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
