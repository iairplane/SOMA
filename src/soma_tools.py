import os
import cv2
import torch
import numpy as np
import logging
from PIL import Image
from typing import Tuple, List, Optional

# 尝试导入 SAM3
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    SAM3_AVAILABLE = True
except ImportError:
    SAM3_AVAILABLE = False
    logging.warning("[SOMA Tools] 未检测到 SAM3 库，视觉处理工具将不可用。")

class MCPTools:
    """
    SOMA 原子工具箱 (The Limbs)
    包含: SAM3 分割, Inpainting 擦除, Visual Prompting 高亮
    """
    def __init__(self, device: str = "cuda", sam3_checkpoint: str = "./sam3.pt"):
        self.device = device
        self.predictor = None
        
        if SAM3_AVAILABLE:
            self._init_sam3(sam3_checkpoint)

    def _init_sam3(self, checkpoint_path):
        """初始化 SAM3 模型"""
        try:
            if not os.path.exists(checkpoint_path):
                logging.warning(f"[SOMA Tools] SAM3 权重未找到: {checkpoint_path}")
                return
            
            logging.info(f"[SOMA Tools] Loading SAM3 from {checkpoint_path}...")
            model = build_sam3_image_model(
                checkpoint_path=checkpoint_path,
                load_from_HF=False,
                device=self.device,
                eval_mode=True,
                enable_segmentation=True
            )
            self.predictor = Sam3Processor(model, device=self.device)
            logging.info("[SOMA Tools] SAM3 Loaded.")
        except Exception as e:
            logging.error(f"[SOMA Tools] SAM3 Init Failed: {e}")

    def _get_mask(self, image_pil: Image.Image, prompt: str) -> np.ndarray:
        """核心原子操作: Text -> Binary Mask"""
        if not self.predictor or not prompt:
            return np.zeros((image_pil.height, image_pil.width), dtype=np.uint8)
        
        try:
            inference_state = self.predictor.set_image(image_pil)
            result = self.predictor.set_text_prompt(state=inference_state, prompt=prompt)
            
            scores = result["scores"].cpu().numpy()
            masks = result["masks"].cpu().numpy()
            
            # 过滤低置信度
            if len(scores) > 0:
                best_idx = np.argmax(scores)
                if scores[best_idx] > 0.35: # 阈值可调
                    mask = masks[best_idx].squeeze()
                    return (mask > 0).astype(np.uint8)
        except Exception as e:
            logging.error(f"SAM3 Segmentation Error ({prompt}): {e}")
            
        return np.zeros((image_pil.height, image_pil.width), dtype=np.uint8)

    # ================= 工具 1: Paint-to-Action (物体换色/高亮) =================
    def apply_visual_overlay(self, image: np.ndarray, target: str, color: Tuple[int,int,int] = (0, 255, 0)) -> np.ndarray:
        """
        在目标物体表面覆盖一层半透明颜色。
        用于解决: Visual Bias (VLA 看不懂某种颜色的物体，强行改成它熟悉的颜色)
        """
        if not self.predictor: return image
        
        img_pil = Image.fromarray(image).convert("RGB")
        mask = self._get_mask(img_pil, target)
        
        if not np.any(mask):
            logging.warning(f"[Tool] 未找到目标: {target}，跳过高亮。")
            return image

        # 图像处理 (Alpha Blending)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        overlay = np.zeros_like(image_bgr)
        overlay[mask == 1] = color # 填充颜色
        
        # 混合: original * 0.6 + overlay * 0.4
        alpha = 0.4
        output = cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0, dtype=cv2.CV_8U)
        
        # 只替换 mask 区域，保持背景清晰
        result = image_bgr.copy()
        result[mask == 1] = output[mask == 1]
        
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    # ================= 工具 2: Eraser (无关物体擦除) =================
    def remove_distractor(self, image: np.ndarray, distractor: str) -> np.ndarray:
        """
        利用 Inpainting 物理擦除干扰物。
        用于解决: Causal Confusion (因果混淆，比如桌上的红块干扰了机器人)
        """
        if not self.predictor: return image
        
        img_pil = Image.fromarray(image).convert("RGB")
        mask = self._get_mask(img_pil, distractor)
        
        if not np.any(mask):
            return image
            
        # 1. 膨胀 Mask (Dilate): 这一步至关重要！
        # 如果不膨胀，物体边缘会残留一圈“光晕”伪影，VLA 依然会把它当成物体。
        kernel = np.ones((7, 7), np.uint8) # 7x7 核
        mask_dilated = cv2.dilate(mask, kernel, iterations=2)
        
        # 2. Inpainting (Telea 算法)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # radius=5, 使用 Navier-Stokes (NS) 或 Telea
        inpainted = cv2.inpaint(image_bgr, mask_dilated, 5, cv2.INPAINT_TELEA)
        
        logging.info(f"[Tool] 已擦除干扰物: {distractor}")
        return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)

    # ================= 工具 3: Debug (保存中间结果) =================
    def save_debug(self, image: np.ndarray, step: int, suffix: str = ""):
        path = f"debug_step_{step}_{suffix}.jpg"
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))