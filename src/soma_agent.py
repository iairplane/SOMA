import logging
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, Dict, Any, Optional

# 引入之前定义的所有模块
# 假设这些文件都在同一目录下
try:
    from soma_vlm import Qwen3VLAPIClient
    from soma_encoder import AdvancedEmbeddingEncoder
    from soma_memory import MemoryBank
    from soma_tools import MCPTools
    from soma_perception import PerceptionModule
    from soma_logger import ExperienceLogger
except ImportError as e:
    logging.error(f"[SOMA Agent] 模块导入失败: {e}. 请确保 soma_*.py 文件都在同一目录。")
    raise

class SOMAAgent:
    """
    SOMA System Facade (统一对外接口)
    将 Brain, Eyes, Limbs, Scribe 封装为一个整体。
    """
    def __init__(self, config: Dict[str, Any]):
        """
        config 示例:
        {
            "device": "cuda",
            "sam3_path": "./sam3.pt",
            "memory_dir": "./soma_memory",
            "vlm_api_key": "sk-...",
            "vlm_base_url": "..."
        }
        """
        self.cfg = config
        self.device = config.get("device", "cuda")
        
        logging.info("========== Initializing SOMA Agent ==========")

        # 1. 初始化大脑 (Brain)
        self.vlm = Qwen3VLAPIClient(
            api_key=config.get("vlm_api_key"),
            base_url=config.get("vlm_base_url")
        )

        # 2. 初始化地图 (Encoder)
        # 权重偏向视觉(0.6)，兼顾语义(0.3)和哈希(0.1)
        self.encoder = AdvancedEmbeddingEncoder(
            device=self.device,
            weights=(0.6, 0.3, 0.1) 
        )

        # 3. 初始化记忆 (Eyes)
        self.memory = MemoryBank(
            storage_dir=config.get("memory_dir", "./soma_memory"),
            dimension=1168 # 768+384+16
        )

        # 4. 初始化感知与工具 (Limbs & Prefrontal)
        # PerceptionModule 内部会初始化 MCPTools
        self.perception = PerceptionModule(
            vlm_client=self.vlm,
            device=self.device,
            sam3_path=config.get("sam3_path", "./sam3.pt")
        )

        # 5. 初始化记录员 (Scribe)
        self.logger = ExperienceLogger(
            memory_bank=self.memory,
            encoder=self.encoder,
            vlm_client=self.vlm
        )
        
        logging.info("========== SOMA Agent Ready ==========")

    def init_episode(self, init_frame: np.ndarray, task_desc: str) -> Dict:
        """
        Episode 开始时调用：执行 RAG 检索
        """
        try:
            img_pil = Image.fromarray(init_frame).convert("RGB")
            
            # 生成 Query Vector (只在开始算一次，开销小)
            query_emb = self.encoder.embed(img_pil, task_desc)
            
            # 检索失败案例 (用于避坑)
            results = self.memory.retrieve(query_emb, partition="failure", top_k=2)
            
            logging.info(f"[SOMA] RAG 检索完成，找到 {len(results)} 条相关失败经验")
            return {"failure": results}
        except Exception as e:
            logging.error(f"[SOMA] Init Episode Error: {e}")
            return {}

    def step(self, 
             frame: np.ndarray, 
             task_desc: str, 
             step_idx: int, 
             rag_context: Dict) -> Tuple[np.ndarray, str, Dict]:
        """
        每 N 步调用一次：执行感知编排
        Returns: (processed_image, refined_task, control_flags)
        """
        # 直接透传给 PerceptionModule
        # 注意：PerceptionModule 内部包含了 VLM 决策和 Tools 执行
        return self.perception.process_frame(frame, task_desc, step_idx, rag_context)

    def finish_episode(self, 
                       video_path: str, 
                       keyframe_path: str, 
                       task_desc: str, 
                       success: bool):
        """
        Episode 结束时调用：异步记录日志
        """
        import threading
        
        def _log_worker():
            self.logger.log_episode(
                task_desc=task_desc,
                success=success,
                video_path=video_path,
                keyframe_path=keyframe_path,
                frames=None # 让 logger 自己去读视频/图片文件
            )
            
        # 启动后台线程写入，不阻塞 Evaluation 主进程
        t = threading.Thread(target=_log_worker, daemon=True)
        t.start()