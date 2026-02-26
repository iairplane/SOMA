"""
SOMA: Self-Organizing Memory Agent
==================================

Description:
------------
This module implements the Experience Logger (often referred to as 'The Scribe') 
for the SOMA framework. It acts as the pipeline for asynchronously processing 
and storing episodic experiences into the vector database.

Responsibilities:
1. Data Reception: Ingests episode video paths, natural language task descriptions, and success/failure labels.
2. Automatic Sampling: Uniformly samples visual keyframes directly from video files if in-memory frames are discarded.
3. Intelligent Diagnosis: Triggers the Vision-Language Model (VLM) to generate structured causal attribution reports for failed episodes, and execution summaries for successful ones.
4. Vectorization: Calls the multimodal Encoder to generate dense retrieval vectors for the episode.
5. Storage: Commits the fully processed experience payload into the MemoryBank for future Retrieval-Augmented Generation (RAG).
"""

import logging
import cv2  # Requires opencv-python
import numpy as np
from pathlib import Path
from typing import Optional, List, Union, Dict
from datetime import datetime
from PIL import Image

# Import previously defined modules
try:
    from soma_encoder import AdvancedEmbeddingEncoder
    from soma_memory import MemoryBank
    from soma_vlm import Qwen3VLAPIClient
except ImportError:
    pass

class ExperienceLogger:
    """
    SOMA Experience Logger (The Scribe) - Complete Version
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
        Helper function: Uniformly extract N PIL Image frames from a video file.
        """
        frames = []
        if not video_path or not Path(video_path).exists():
            return []
            
        try:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0: return []
            
            # Uniform sampling indices
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb))
            cap.release()
        except Exception as e:
            logging.error(f"[Logger] Video frame extraction failed: {e}")
            
        return frames

    def _run_failure_diagnosis(self, task: str, frames: List[Image.Image]) -> str:
        """Invoke VLM to generate a failure diagnosis report."""
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
        """Invoke VLM to generate a successful execution summary."""
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
        """Log the Episode, supporting synchronous diagnosis and database storage."""
        try:
            # 1. Resource Preparation (Auto-sample from video or use provided frames)
            if not frames:
                diag_frames = self._extract_frames_from_video(video_path, num_frames=5)
            else:
                diag_frames = frames[:5] # Limit frame count to save VLM tokens
            
            index_img = diag_frames[0] if diag_frames else None
            
            # Fallback: If still no image, attempt to read keyframe_path from disk
            if index_img is None and Path(keyframe_path).exists():
                index_img = Image.open(keyframe_path).convert("RGB")
                diag_frames = [index_img]

            if not index_img:
                logging.warning(f"[Logger] Abandoning log: Task {task_desc} is missing image data")
                return

            # 2. Generate RAG index vector
            embedding = self.encoder.embed(index_img, task_desc)
            
            # 3. Branching Diagnosis Logic (Ensure method names match the VLM client)
            if not success:
                logging.info(f"[Logger] Diagnosing failed task: {task_desc}")
                diagnosis = self._run_failure_diagnosis(task_desc, diag_frames)
            else:
                # On success, a summary can also be stored to help RAG retrieve positive experiences
                diagnosis = self._run_success_summary(task_desc, diag_frames)
            print(f"📌 Diagnosis Result: {diagnosis}")
            
            # 4. Persistent Storage: Call MemoryBank to write to disk
            self.memory.add_experience(
                embedding=embedding,
                task_desc=task_desc,
                success=success,
                video_path=video_path,
                keyframe_path=keyframe_path,
                diagnosis=diagnosis,
                info=additional_info
            )
            logging.info(f"✅ [Logger] Experience saved: {task_desc} (Success={success})")
            
        except Exception as e:
            logging.error(f"[Logger] Critical Error: Failed to log Episode - {e}", exc_info=True)

# ================= Unit Tests =================
# if __name__ == "__main__":
#     # Mock testing
#     logging.basicConfig(level=logging.INFO)
    
#     # 1. Mock Components
#     class MockMem:
#         def add_experience(self, **kwargs):
#             print(f"✅ Memory Save: Success={kwargs['success']}, Diag={kwargs['diagnosis']}")

#     class MockEnc:
#         def embed(self, img, txt): return [0.1] * 1168

#     class MockVLM:
#         def generate_failure_report(self, images, task_prompt, anchor_example=None):
#             return {
#                 "intrinsic_analysis": {
#                     "direct_cause": "Visual Mismatch",
#                     "observation": "The robot grasped the shadow instead of the cup."
#                 }
#             }

#     # 2. Execution
#     logger = ExperienceLogger(MockMem(), MockEnc(), MockVLM())
    
#     # Create a fake image
#     fake_img = Image.new('RGB', (100, 100), color='red')
    
#     print("--- Testing Failure Log ---")
#     logger.log_episode(
#         task_desc="Pick up red cup",
#         success=False,
#         video_path="dummy.mp4",
#         keyframe_path="dummy.jpg",
#         frames=[fake_img] # Pass in-memory image
#     )