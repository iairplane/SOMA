"""
SOMA: Self-Organizing Memory Agent
==================================

Description:
------------
This module implements the main facade for the SOMA Agent, serving as the client-side 
interface that connects perception, memory, and high-level reasoning modules. 
It operates in a decoupled architecture where heavy vision models (like SAM3) 
are hosted as independent HTTP services to optimize inference latency and resource allocation.

Key Components:
- Brain (VLM): High-level reasoning and visual-language understanding (e.g., Qwen3-VL).
- Perception: Connects to an external SAM3 instance for zero-shot grounding and segmentation.
- Memory & Encoder: Maintains a vector-based Memory Bank for Retrieval-Augmented Generation (RAG),
  allowing the agent to learn from past successful and failed episodes.
- Logger: Asynchronously records episode experiences to prevent blocking the main control loop.

Usage:
------
Initialize the agent with a configuration dictionary containing model endpoints and API keys, 
then call `init_episode()`, `step()`, and `finish_episode()` within your environment loop.
Always call `wait_until_done()` before script termination to ensure async logs are saved.
"""

import logging
import threading
from typing import Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image

# Import submodules
try:
    from soma_vlm import Qwen3VLAPIClient
    from soma_encoder import AdvancedEmbeddingEncoder
    from soma_memory import MemoryBank
    from soma_perception import PerceptionModule
    from soma_logger import ExperienceLogger
except ImportError as e:
    logging.error(f"[SOMA Agent] Module import failed: {e}. Ensure all soma_*.py files are in the same directory.")
    raise

class SOMAAgent:
    """
    SOMA System Facade (Unified External Interface) - Client Side
    Adapter for: SAM3 Independent Service Mode (HTTP)
    """
    def __init__(self, config: Dict[str, Any]):
        """
        config example:
        {
            "device": "cuda",
            "sam3_base_url": "http://127.0.0.1:5001",
            "tool_timeout_s": 20.0,
            "memory_dir": "./soma_memory",
            "vlm_api_key": "sk-...",
            "vlm_base_url": "..."
            "model_id": "xxx"  # Optional: Override default model (qwen3vl)
        }
        """
        self.cfg = config
        self.device = config.get("device", "cuda")
        self.log_threads = []
        
        logging.info("========== Initializing SOMA Agent (Service Mode) ==========")

        # 1. Initialize Brain (VLM)
        self.vlm = Qwen3VLAPIClient(
            api_key=config.get("vlm_api_key"),
            base_url=config.get("vlm_base_url"),
            model_id=config.get("model_id")
        )

        # 2. Initialize Map (Encoder)
        self.encoder = AdvancedEmbeddingEncoder(
            device=self.device,
            weights=(0.6, 0.3, 0.1) 
        )

        # 3. Initialize Memory (Storage)
        self.memory = MemoryBank(
            storage_dir=config.get("memory_dir", "./soma_memory"),
            dimension=1168 
        )

        # 4. Initialize Perception -> Connect to SAM3 Service
        # [Modification] Pass URL instead of weight path
        sam3_url = config.get("sam3_base_url", "http://127.0.0.1:5001")
        tool_timeout = config.get("tool_timeout_s", 20.0)
        
        logging.info(f"[SOMA] Connecting to SAM3 Service at: {sam3_url}")
        self.perception = PerceptionModule(
            vlm_client=self.vlm,
            sam3_base_url=sam3_url,
            tool_timeout_s=tool_timeout
        )

        # 5. Initialize Logger (Experience Logger)
        self.logger = ExperienceLogger(
            memory_bank=self.memory,
            encoder=self.encoder,
            vlm_client=self.vlm,
            # base_dir=config.get("memory_dir", "./soma_memory")
        )
        
        logging.info("========== SOMA Agent Ready ==========")

    def init_episode(self, init_frame: np.ndarray, task_desc: str) -> Dict:
        """
        Called at the beginning of an episode: Execute RAG retrieval.
        """
        try:
            # Ensure it is an RGB PIL Image
            if isinstance(init_frame, np.ndarray):
                img_pil = Image.fromarray(init_frame).convert("RGB")
            else:
                img_pil = init_frame

            # Generate Query Vector
            query_emb = self.encoder.embed(img_pil, task_desc)
            
            # Retrieve (separately for successful and failed cases)
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
        Step call: Perception orchestration.
        Returns: (processed_image, refined_task, control_flags)
        """
        try:
            return self.perception.process_frame(frame, task_desc, step_idx, rag_context)
        except Exception as e:
            logging.error(f"[SOMAAgent] Critical Step Error: {e}", exc_info=True)
            # Fallback return to prevent eval script from crashing
            return frame, task_desc, {}

    def finish_episode(self, 
                       video_path: str, 
                       keyframe_path: str, 
                       task_desc: str, 
                       success: bool,
                       **kwargs):
        """
        Called at the end of an episode: Asynchronously record logs.
        """
        def _log_worker():
            try:
                self.logger.log_episode(
                    task_desc=task_desc,
                    success=success,
                    video_path=video_path,
                    keyframe_path=keyframe_path,
                    frames=None, # Let the logger read video/image files on its own
                    additional_info=kwargs
                )
            except Exception as e:
                logging.error(f"[SOMA] Async Log Failed: {e}")
            
        # Start background thread for writing, non-blocking to the main process
        t = threading.Thread(target=_log_worker, daemon=True)
        t.start()
        self.log_threads.append(t)
    
    def wait_until_done(self):
        """
        [New] Call before the main program exits to ensure all async logs are safely written to disk.
        """
        if not self.log_threads:
            return
        
        logging.info(f"⏳ Waiting for {len(self.log_threads)} SOMA logging tasks to complete...")
        for t in self.log_threads:
            if t.is_alive():
                t.join() # Block until the thread finishes execution
        self.log_threads = []
        logging.info("✅ All experiences have been safely stored.")