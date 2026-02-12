#!/usr/bin/env python
"""SAM3 独立服务程序 (SOMA)

提供 HTTP API 给主评测进程调用，执行所有依赖 SAM3 的视觉 MCP 工具。

Endpoints:
- GET  /health
- POST /visual_overlay
- POST /remove_distractor
- POST /replace_texture
- POST /replace_background

请求统一格式:
{
  "image": "data:image/png;base64,..." 或 "base64...",
  "prompt": "text prompt"  # 根据接口不同也可能是 target_object / region_prompt
  ...
}

返回统一格式:
{
  "success": true/false,
  "image": "base64..."  # 处理后 PNG
  "message": "..."
}
"""

import argparse
import base64
import io
import logging
import os
import sys
import cv2
import torch
from typing import Any, Dict, Tuple, Optional

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

# SAM3 imports
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError:
    print("Error: sam3 module not found.")
    sys.exit(1)

# SOMA VLM Client import
try:
    from soma_vlm import Qwen3VLAPIClient
except ImportError:
    print("Error: soma_vlm.py not found in current directory.")
    sys.exit(1)
    
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("soma_sam3_service")

app = Flask(__name__)
CORS(app)

sam3_predictor: Sam3Processor | None = None
vlm_client: Qwen3VLAPIClient | None = None

def _decode_image(data_url_or_b64: str) -> np.ndarray:
    if not data_url_or_b64:
        raise ValueError("image is empty")
    if "," in data_url_or_b64:
        data_url_or_b64 = data_url_or_b64.split(",", 1)[1]
    raw = base64.b64decode(data_url_or_b64)
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(pil)


def _encode_png_b64(img: np.ndarray) -> str:
    pil = Image.fromarray(img.astype(np.uint8)).convert("RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def init_sam3_model(device: str, sam3_weight_path: str) -> bool:
    global sam3_predictor
    try:
        logger.info(f"Loading SAM3 model on {device} from {sam3_weight_path}")
        if sam3_weight_path and not os.path.exists(sam3_weight_path):
            logger.error(f"SAM3 weight not found: {sam3_weight_path}")
            return False

        model = build_sam3_image_model(
            checkpoint_path=sam3_weight_path,
            load_from_HF=(sam3_weight_path is None),
            device=device,
            eval_mode=True,
            enable_segmentation=True,
            enable_inst_interactivity=False,
        )
        model = model.to(device).eval()
        sam3_predictor = Sam3Processor(model, device=device)
        logger.info("SAM3 loaded")
        return True
    except Exception as e:
        logger.error(f"SAM3 init failed: {e}", exc_info=True)
        return False


def _get_mask(image: np.ndarray, prompt: str, score_th: float = 0.25) -> Tuple[np.ndarray, float]:
    """
    结合 VLM (引用自 soma_vlm) 和 SAM3 获取 Mask
    """
    global sam3_predictor, vlm_client
    if sam3_predictor is None:
        raise RuntimeError("SAM3 not initialized")
    if not prompt:
        return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8), 0.0

    pil = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    state = sam3_predictor.set_image(pil)
    
    # --- 策略 A: 优先使用 VLM 检测 Box ---
    if vlm_client:
        try:
            # 直接调用 soma_vlm 中的方法
            bbox = vlm_client.detect_object(image, prompt) # [x1, y1, x2, y2]
            print(f"VLM detected bbox: {bbox}")
            if bbox:
                logger.info(f"VLM detected box: {bbox}")
                # 使用 VLM 框辅助筛选 SAM3 的结果 (这是最稳健的集成方式，不需要改 SAM3 源码)
                result = sam3_predictor.set_text_prompt(state=state, prompt=prompt)
                masks = result.get("masks", torch.tensor([])).detach().cpu().numpy()
                scores = result.get("scores", torch.tensor([])).detach().cpu().numpy()
                
                if len(masks) > 0:
                    best_iou = -1.0
                    best_idx = -1
                    box_area = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
                    
                    for i, m in enumerate(masks):
                        m_bool = m.squeeze() > 0
                        if not np.any(m_bool): continue
                        
                        # 计算 Mask 的包围盒
                        ys, xs = np.where(m_bool)
                        mx1, mx2, my1, my2 = xs.min(), xs.max(), ys.min(), ys.max()
                        
                        # 计算交集
                        ix1 = max(bbox[0], mx1); iy1 = max(bbox[1], my1)
                        ix2 = min(bbox[2], mx2); iy2 = min(bbox[3], my2)
                        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                        
                        # 简单的匹配度指标
                        metric = inter / (box_area + 1e-6)
                        if metric > best_iou:
                            best_iou = metric
                            best_idx = i
                    
                    if best_idx != -1 and best_iou > 0.1: # 至少有一定重叠
                        return (masks[best_idx].squeeze() > 0).astype(np.uint8), float(scores[best_idx])
        except Exception as e:
            logger.warning(f"VLM detect failed, falling back to pure SAM3: {e}")

    # --- 策略 B: 纯 SAM3 文本模式 (Fallback) ---
    logger.info(f"Using SAM3 text prompt: {prompt}")
    result = sam3_predictor.set_text_prompt(state=state, prompt=prompt)
    scores = result.get("scores", np.array([])).detach().cpu().numpy()
    masks = result.get("masks", np.array([])).detach().cpu().numpy()

    if len(scores) == 0: return np.zeros(image.shape[:2], dtype=np.uint8), 0.0
    
    best = int(np.argmax(scores))
    if scores[best] < score_th: return np.zeros(image.shape[:2], dtype=np.uint8), float(scores[best])
    
    return (masks[best].squeeze() > 0).astype(np.uint8), float(scores[best])



def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, x2, y2


@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok", "sam3_loaded": sam3_predictor is not None})


@app.post("/visual_overlay")
def visual_overlay() -> Any:
    try:
        data: Dict[str, Any] = request.get_json(force=True)
        img = _decode_image(data.get("image", ""))
        target = data.get("target_object") or data.get("prompt")
        color = data.get("color", [0, 255, 0])
        alpha = float(data.get("alpha", 0.4))
        
        mask, score = _get_mask(img, target)
        if not np.any(mask):
            return jsonify({"success": False, "image": _encode_png_b64(img), "message": f"mask not found (score={score:.3f})"})

        out = img.copy()
        overlay = np.zeros_like(out)
        overlay[mask == 1] = np.array(color, dtype=np.uint8)
        out = (out * (1 - alpha) + overlay * alpha).astype(np.uint8)
        out[mask == 0] = img[mask == 0]

        return jsonify({"success": True, "image": _encode_png_b64(out), "message": "ok"})
    except Exception as e:
        logger.error(f"/visual_overlay failed: {e}", exc_info=True)
        return jsonify({"success": False, "message": str(e)}), 500


@app.post("/remove_distractor")
def remove_distractor() -> Any:
    try:
        import cv2

        data: Dict[str, Any] = request.get_json(force=True)
        img = _decode_image(data.get("image", ""))
        distractor = data.get("object_to_remove") or data.get("prompt")

        mask, score = _get_mask(img, distractor)
        if not np.any(mask):
            return jsonify({"success": False, "image": _encode_png_b64(img), "message": f"mask not found (score={score:.3f})"})

        kernel = np.ones((7, 7), np.uint8)
        mask_d = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        inpainted = cv2.inpaint(img_bgr, mask_d, 5, cv2.INPAINT_TELEA)
        out = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)

        return jsonify({"success": True, "image": _encode_png_b64(out), "message": "ok"})
    except Exception as e:
        logger.error(f"/remove_distractor failed: {e}", exc_info=True)
        return jsonify({"success": False, "message": str(e)}), 500


@app.post("/replace_texture")
def replace_texture() -> Any:
    try:
        data: Dict[str, Any] = request.get_json(force=True)
        img = _decode_image(data.get("image", ""))
        target = data.get("target_object")
        texture_b64 = data.get("texture_image")

        if not target:
            return jsonify({"success": False, "image": _encode_png_b64(img), "message": "target_object is required"}), 400
        if not texture_b64:
            return jsonify({"success": False, "image": _encode_png_b64(img), "message": "texture_image is required"}), 400

        texture = _decode_image(texture_b64)

        mask, score = _get_mask(img, target)
        if not np.any(mask):
            return jsonify({"success": False, "image": _encode_png_b64(img), "message": f"mask not found (score={score:.3f})"})

        bbox = _bbox_from_mask(mask)
        if bbox is None:
            return jsonify({"success": False, "image": _encode_png_b64(img), "message": "bbox not found"})

        x1, y1, x2, y2 = bbox
        h = max(1, y2 - y1 + 1)
        w = max(1, x2 - x1 + 1)

        # resize texture to bbox
        from PIL import Image as PILImage

        tex_pil = PILImage.fromarray(texture).resize((w, h))
        tex = np.array(tex_pil).astype(np.uint8)

        out = img.copy()
        region_mask = mask[y1 : y2 + 1, x1 : x2 + 1]
        region = out[y1 : y2 + 1, x1 : x2 + 1]

        # simple paste using mask
        region[region_mask == 1] = tex[region_mask == 1]
        out[y1 : y2 + 1, x1 : x2 + 1] = region

        return jsonify({"success": True, "image": _encode_png_b64(out), "message": "ok"})
    except Exception as e:
        logger.error(f"/replace_texture failed: {e}", exc_info=True)
        return jsonify({"success": False, "message": str(e)}), 500


@app.post("/replace_background")
def replace_background() -> Any:
    try:
        data: Dict[str, Any] = request.get_json(force=True)
        img = _decode_image(data.get("image", ""))
        region_prompt = data.get("region_prompt") or data.get("prompt")
        texture_b64 = data.get("texture_image")

        if not region_prompt:
            return jsonify({"success": False, "image": _encode_png_b64(img), "message": "region_prompt is required"}), 400
        if not texture_b64:
            return jsonify({"success": False, "image": _encode_png_b64(img), "message": "texture_image is required"}), 400

        texture = _decode_image(texture_b64)

        mask, score = _get_mask(img, region_prompt)
        if not np.any(mask):
            return jsonify({"success": False, "image": _encode_png_b64(img), "message": f"mask not found (score={score:.3f})"})

        # tile texture to full image size
        H, W = img.shape[0], img.shape[1]
        th, tw = texture.shape[0], texture.shape[1]
        if th <= 0 or tw <= 0:
            return jsonify({"success": False, "image": _encode_png_b64(img), "message": "invalid texture"}), 400

        reps_y = int(np.ceil(H / th))
        reps_x = int(np.ceil(W / tw))
        tiled = np.tile(texture, (reps_y, reps_x, 1))[:H, :W, :]

        alpha = float(data.get("alpha", 1.0))
        out = img.copy()
        blended = (img.astype(np.float32) * (1 - alpha) + tiled.astype(np.float32) * alpha).astype(np.uint8)
        out[mask == 1] = blended[mask == 1]

        return jsonify({"success": True, "image": _encode_png_b64(out), "message": "ok"})
    except Exception as e:
        logger.error(f"/replace_background failed: {e}", exc_info=True)
        return jsonify({"success": False, "message": str(e)}), 500


def main() -> None:
    parser = argparse.ArgumentParser(description="SOMA SAM3 Service")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--sam3_weight_path",
        type=str,
        default="/mnt/disk1/shared_data/lzy/models/sam/sam3.pt",
    )
    parser.add_argument("--vlm_base_url", default="https://models.sjtu.edu.cn/api/v1")
    parser.add_argument("--vlm_api_key", required=False, help="API Key for soma_vlm",default="sk-dJ9PDHKGeP7xfsO4Zv7jNw")
    
    args = parser.parse_args()

    if not init_sam3_model(device=args.device, sam3_weight_path=args.sam3_weight_path):
        logger.error("SAM3 init failed; exiting")
        sys.exit(1)

    global vlm_client
    if args.vlm_api_key:
        logger.info("Initializing SOMA VLM Client...")
        vlm_client = Qwen3VLAPIClient(api_key=args.vlm_api_key, base_url=args.vlm_base_url)
    else:
        logger.warning("No VLM API Key. Advanced reasoning disabled.")
        
    logger.info(f"Starting SOMA SAM3 service on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()




