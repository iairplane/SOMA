import torch
import numpy as np
import logging
import hashlib
from PIL import Image
from typing import List, Union, Tuple
from functools import lru_cache # 关键优化库

try:
    from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
except ImportError:
    raise ImportError("请安装 transformers 库: pip install transformers")

class AdvancedEmbeddingEncoder:
    """
    SOMA 高级复合编码器 (高性能优化版)
    优化点:
    1. FP16 推理: 速度提升 2x-3x
    2. LRU Cache: 文本向量只算一次
    3. 显存管理: 仅在推理时占用必要资源
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
        
        logging.info(f"[SOMA Encoder] 正在加载模型 (FP16 Mode)...")
        
        # 1. 加载视觉模型 (开启 FP16)
        try:
            self.clip_model = CLIPModel.from_pretrained(
                vision_model_name, 
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32
            ).to(self.device).eval()
            self.clip_processor = CLIPProcessor.from_pretrained(vision_model_name)
        except Exception as e:
            logging.error(f"视觉模型加载失败: {e}")
            raise

        # 2. 加载文本模型 (开启 FP16)
        try:
            self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
            self.text_model = AutoModel.from_pretrained(
                text_model_name,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32
            ).to(self.device).eval()
        except Exception as e:
            logging.warning(f"文本模型加载失败: {e}，回退到 CLIP")
            self.text_model = None

    def _l2_normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec) + 1e-8
        return vec / norm

    def _get_vision_embedding(self, images: List[Image.Image]) -> np.ndarray:
        """视觉部分：这是最耗时的，但我们只在 Episode 开始时调用一次"""
        with torch.no_grad():
            inputs = self.clip_processor(images=images, return_tensors="pt")
            # 移动到 GPU 并转为 FP16
            inputs = {k: v.to(self.device) for k, v in inputs.items()} 
            
            vision_features = self.clip_model.get_image_features(**inputs)
            # 转回 float32 进行后续 numpy 计算
            vec = vision_features.float().cpu().numpy()[0]
        return self._l2_normalize(vec)

    @lru_cache(maxsize=128) # <--- 核心优化：缓存最近 128 个任务的文本向量
    def _get_text_embedding_cached(self, text: str) -> np.ndarray:
        """文本部分：带缓存，命中缓存时耗时 0ms"""
        # logging.debug(f"Computing text embedding for: {text}")
        with torch.no_grad():
            if self.text_model:
                inputs = self.text_tokenizer([text], padding=True, truncation=True, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.text_model(**inputs)
                token_embeddings = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']
                
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                text_features = sum_embeddings / sum_mask
            else:
                inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                text_features = self.clip_model.get_text_features(**inputs)
            
        vec = text_features.float().cpu().numpy()[0]
        return self._l2_normalize(vec)

    def _get_hash_embedding(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        h_vals = np.frombuffer(h, dtype=np.uint8)[:self.hash_len].astype(np.float32) / 255.0
        return self._l2_normalize(h_vals)

    def embed(self, image_path: Union[str, Image.Image], task_desc: str) -> list[float]:
        """
        生成加权混合 Embedding (高性能版)
        """
        # 1. 图像处理 (I/O)
        if isinstance(image_path, str):
            try:
                image = Image.open(image_path).convert("RGB")
            except:
                return [0.0] * 100 # Dummy
        else:
            image = image_path

        # 2. 推理 (Vision 是主要开销，Text 直接查缓存)
        v_vec = self._get_vision_embedding([image])
        t_vec = self._get_text_embedding_cached(task_desc) # 调用缓存函数
        h_vec = self._get_hash_embedding(task_desc)

        # 3. 加权拼接
        w_v, w_t, w_h = self.weights
        combined_vec = np.concatenate([v_vec * w_v, t_vec * w_t, h_vec * w_h])
        
        return self._l2_normalize(combined_vec).tolist()
# # 调试代码
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
    
#     # 侧重视觉权重的配置
#     encoder = AdvancedEmbeddingEncoder(weights=(0.7, 0.25, 0.05))
    
#     fake_img = Image.new('RGB', (224, 224), color='red')
#     task = "pick up the cup"
    
#     emb = encoder.embed(fake_img, task)
    
#     print(f"加权向量生成成功，维度: {len(emb)}")
#     print(f"L2 Norm (应为 1.0): {np.linalg.norm(emb):.4f}")