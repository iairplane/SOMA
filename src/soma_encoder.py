"""
SOMA: Self-Organizing Memory Agent
==================================

Description:
------------
This module implements the Advanced Embedding Encoder for the SOMA framework. 
It generates multi-modal, weighted hybrid embeddings that fuse visual features, 
semantic text features, and task-specific hashes. These composite embeddings 
serve as robust query vectors for the Retrieval-Augmented Generation (RAG) Memory Bank.

To support high-frequency control loops in robotic environments, the encoder is 
designed with FP16 precision, efficient VRAM allocation, and Least Recently Used (LRU) 
caching to minimize latency during real-time inference.
"""

import torch
import numpy as np
import logging
import hashlib
from PIL import Image
from typing import List, Union, Tuple
from functools import lru_cache 

try:
    from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
except ImportError:
    raise ImportError("Required library 'transformers' is missing. Please install it via: pip install transformers")

class AdvancedEmbeddingEncoder:
    """
    Generates fused multi-modal embeddings from visual and textual inputs.
    
    This encoder combines representations from a vision model (e.g., CLIP) and 
    a language model (e.g., MiniLM) alongside a deterministic task hash. 
    Features are L2-normalized and concatenated using predefined weights to 
    form a unified state representation.
    """

    def __init__(self, 
                 vision_model_name: str = "openai/clip-vit-large-patch14",
                 text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 device: str = "cuda",
                 hash_len: int = 16,
                 weights: Tuple[float, float, float] = (0.6, 0.3, 0.1)):
        
        self.device = device if torch.cuda.is_available() else "cpu"
        self.hash_len = hash_len
        self.weights = weights
        
        logging.info("[SOMA Encoder] Initializing models in FP16 precision...")
        
        # Initialize Vision Model
        try:
            self.clip_model = CLIPModel.from_pretrained(
                vision_model_name, 
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32
            ).to(self.device).eval()
            self.clip_processor = CLIPProcessor.from_pretrained(vision_model_name)
        except Exception as e:
            logging.error(f"Failed to load vision model: {e}")
            raise

        # Initialize Text Model
        try:
            self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
            self.text_model = AutoModel.from_pretrained(
                text_model_name,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32
            ).to(self.device).eval()
        except Exception as e:
            logging.warning(f"Failed to load text model: {e}. Falling back to CLIP text encoder.")
            self.text_model = None

    def _l2_normalize(self, vec: np.ndarray) -> np.ndarray:
        """Applies L2 normalization to a given feature vector."""
        norm = np.linalg.norm(vec) + 1e-8
        return vec / norm

    def _get_vision_embedding(self, images: List[Image.Image]) -> np.ndarray:
        """Extracts and normalizes visual features from the input images."""
        with torch.no_grad():
            inputs = self.clip_processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()} 
            
            vision_features = self.clip_model.get_image_features(**inputs)
            vec = vision_features.float().cpu().numpy()[0]
            
        return self._l2_normalize(vec)

    @lru_cache(maxsize=128)
    def _get_text_embedding_cached(self, text: str) -> np.ndarray:
        """
        Extracts and normalizes semantic text features from the task description.
        Utilizes LRU caching to avoid redundant forward passes for recurring tasks.
        """
        with torch.no_grad():
            if self.text_model:
                inputs = self.text_tokenizer([text], padding=True, truncation=True, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.text_model(**inputs)
                token_embeddings = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']
                
                # Perform mean pooling
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                text_features = sum_embeddings / sum_mask
            else:
                # Fallback to CLIP text encoder
                inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                text_features = self.clip_model.get_text_features(**inputs)
            
        vec = text_features.float().cpu().numpy()[0]
        return self._l2_normalize(vec)

    def _get_hash_embedding(self, text: str) -> np.ndarray:
        """Generates a deterministic, normalized hash vector from the task string."""
        h = hashlib.sha256(text.encode("utf-8")).digest()
        h_vals = np.frombuffer(h, dtype=np.uint8)[:self.hash_len].astype(np.float32) / 255.0
        return self._l2_normalize(h_vals)

    def embed(self, image_path: Union[str, Image.Image], task_desc: str) -> list[float]:
        """
        Computes the final weighted hybrid embedding for a given image and task.
        
        Args:
            image_path: Path to the image file, or a PIL Image object.
            task_desc: Natural language string describing the current task.
            
        Returns:
            A list of floats representing the L2-normalized, concatenated embedding.
        """
        # 1. Process Input Image
        if isinstance(image_path, str):
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                logging.warning(f"Failed to open image at {image_path}: {e}. Returning zero vector.")
                return [0.0] * 100 
        else:
            image = image_path

        # 2. Extract Multi-modal Features
        v_vec = self._get_vision_embedding([image])
        t_vec = self._get_text_embedding_cached(task_desc)
        h_vec = self._get_hash_embedding(task_desc)

        # 3. Apply Weights and Concatenate
        w_v, w_t, w_h = self.weights
        combined_vec = np.concatenate([v_vec * w_v, t_vec * w_t, h_vec * w_h])
        
        return self._l2_normalize(combined_vec).tolist()

# ==========================================
# Usage Example
# ==========================================
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     
#     # Initialize encoder with custom modality weights
#     encoder = AdvancedEmbeddingEncoder(weights=(0.7, 0.25, 0.05))
#     
#     dummy_image = Image.new('RGB', (224, 224), color='red')
#     task_instruction = "pick up the cup"
#     
#     embedding = encoder.embed(dummy_image, task_instruction)
#     
#     print(f"Embedding generated successfully. Dimension: {len(embedding)}")
#     print(f"L2 Norm (Expected: 1.0): {np.linalg.norm(embedding):.4f}")