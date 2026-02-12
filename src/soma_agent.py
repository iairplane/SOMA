import logging
import threading
from typing import Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image

# 引入子模块
try:
    from soma_vlm import Qwen3VLAPIClient
    from soma_encoder import AdvancedEmbeddingEncoder
    from soma_memory import MemoryBank
    from soma_perception import PerceptionModule
    from soma_logger import ExperienceLogger
except ImportError as e:
    logging.error(f"[SOMA Agent] 模块导入失败: {e}. 请确保 soma_*.py 文件都在同一目录。")
    raise

class SOMAAgent:
    """
    SOMA System Facade (统一对外接口) - Client Side
    适配：SAM3 独立服务模式 (HTTP)
    """
    def __init__(self, config: Dict[str, Any]):
        """
        config 示例:
        {
            "device": "cuda",
            "sam3_base_url": "http://127.0.0.1:5001",  # <--- 变更点：不再是 path，而是 URL
            "tool_timeout_s": 20.0,
            "memory_dir": "./soma_memory",
            "vlm_api_key": "sk-...",
            "vlm_base_url": "..."
        }
        """
        self.cfg = config
        self.device = config.get("device", "cuda")
        self.log_threads = []
        
        logging.info("========== Initializing SOMA Agent (Service Mode) ==========")

        # 1. 初始化大脑 (Brain)
        self.vlm = Qwen3VLAPIClient(
            api_key=config.get("vlm_api_key"),
            base_url=config.get("vlm_base_url")
        )

        # 2. 初始化地图 (Encoder)
        self.encoder = AdvancedEmbeddingEncoder(
            device=self.device,
            weights=(0.6, 0.3, 0.1) 
        )

        # 3. 初始化记忆 (Storage)
        self.memory = MemoryBank(
            storage_dir=config.get("memory_dir", "./soma_memory"),
            dimension=1168 
        )

        # 4. 初始化感知 (Perception) -> 连接 SAM3 服务
        # [修改点] 传入 URL 而不是 权重路径
        sam3_url = config.get("sam3_base_url", "http://127.0.0.1:5001")
        tool_timeout = config.get("tool_timeout_s", 20.0)
        
        logging.info(f"[SOMA] Connecting to SAM3 Service at: {sam3_url}")
        self.perception = PerceptionModule(
            vlm_client=self.vlm,
            sam3_base_url=sam3_url,
            tool_timeout_s=tool_timeout
        )

        # 5. 初始化记录员 (Logger)
        self.logger = ExperienceLogger(
            memory_bank=self.memory,
            encoder=self.encoder,
            vlm_client=self.vlm,
            # base_dir=config.get("memory_dir", "./soma_memory")
        )
        
        logging.info("========== SOMA Agent Ready ==========")

    def init_episode(self, init_frame: np.ndarray, task_desc: str) -> Dict:
        """
        Episode 开始时调用：执行 RAG 检索
        """
        try:
            # 确保是 RGB PIL
            if isinstance(init_frame, np.ndarray):
                img_pil = Image.fromarray(init_frame).convert("RGB")
            else:
                img_pil = init_frame

            # 生成 Query Vector
            query_emb = self.encoder.embed(img_pil, task_desc)
            
            # 检索 (分别检索成功和失败案例)
            success_results = self.memory.retrieve(query_emb, partition="success", top_k=1)
            failure_results = self.memory.retrieve(query_emb, partition="failure", top_k=2)
            
            logging.info(f"[SOMA] RAG: Found {len(success_results)} success, {len(failure_results)} failure")
            
            return {
                "success": success_results,
                "failure": failure_results
            }
        except Exception as e:
            logging.error(f"[SOMA] Init Episode Error: {e}")
            return {}

    def step(self, 
             frame: np.ndarray, 
             task_desc: str, 
             step_idx: int, 
             rag_context: Dict) -> Tuple[np.ndarray, str, Dict]:
        """
        Step 调用：感知编排
        Returns: (processed_image, refined_task, control_flags)
        """
        # 直接透传给 PerceptionModule 处理
        return self.perception.process_frame(frame, task_desc, step_idx, rag_context)

    def finish_episode(self, 
                       video_path: str, 
                       keyframe_path: str, 
                       task_desc: str, 
                       success: bool,
                       **kwargs):
        """
        Episode 结束时调用：异步记录日志
        """
        def _log_worker():
            try:
                self.logger.log_episode(
                    task_desc=task_desc,
                    success=success,
                    video_path=video_path,
                    keyframe_path=keyframe_path,
                    frames=None, # 让 logger 自己去读视频/图片文件
                    additional_info=kwargs
                )
            except Exception as e:
                logging.error(f"[SOMA] Async Log Failed: {e}")
            
        # 启动后台线程写入，不阻塞主进程
        t = threading.Thread(target=_log_worker, daemon=True)
        t.start()
        self.log_threads.append(t)
    
    def wait_until_done(self):
        """[新增] 在主程序退出前调用，确保所有日志写完"""
        if not self.log_threads:
            return
        
        logging.info(f"⏳ 正在等待 {len(self.log_threads)} 个 SOMA 记录任务完成...")
        for t in self.log_threads:
            if t.is_alive():
                t.join() # 阻塞直到该线程执行完毕
        self.log_threads = []
        logging.info("✅ 所有经验已安全入库。")