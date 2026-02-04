#!/usr/bin/env python
"""最小可行测试：验证 SAM3 MCP tools (HTTP) 是否能与推理侧协作

目标：
1) 确认 sam3_service 可用（/health）
2) 依次调用 MCPTools:
   - visual_overlay
   - remove_distractor
   - replace_texture
   - replace_background
3) 把处理后的图转成 torch tensor，模拟喂给 policy 的 observation[visual_key]

不依赖 Memory / VLM / 环境。

用法：
python /home/lizhuoran/SOMA/src/test_sam3_mcp_tools.py \
  --sam3_url http://127.0.0.1:5001 \
  --image /path/to/frame.png \
  --target "ketchup" \
  --distractor "red block" \
  --floor_prompt "floor" \
  --texture /path/to/texture.png \
  --out_dir /tmp/soma_mcp_test
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

# local imports
from soma_tools import MCPTools


def _to_np_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _save(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img.astype(np.uint8)).save(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam3_url", type=str, default="http://127.0.0.1:5001")
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--texture", type=str, required=False)
    ap.add_argument("--target", type=str, default="object")
    ap.add_argument("--distractor", type=str, default="distractor")
    ap.add_argument("--floor_prompt", type=str, default="floor")
    ap.add_argument("--out_dir", type=str, default="/tmp/soma_mcp_test")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    img = _to_np_rgb(Path(args.image))
    _save(img, out_dir / "0_input.png")

    # 1) health check
    try:
        r = requests.get(f"{args.sam3_url.rstrip('/')}/health", timeout=3)
        r.raise_for_status()
        print("[OK] /health:", r.json())
    except Exception as e:
        print("[FAIL] sam3_service not reachable:", e)
        print("你需要先启动: python /home/lizhuoran/SOMA/src/sam3_service.py --port 5001 --device cuda --sam3_weight_path ...")
        return 2

    tools = MCPTools(sam3_base_url=args.sam3_url)

    # 2) overlay
    img_overlay = tools.apply_visual_overlay(img, args.target, color=(0, 255, 0))
    _save(img_overlay, out_dir / "1_overlay.png")
    print("[OK] visual_overlay saved")

    # 3) remove distractor
    img_rm = tools.remove_distractor(img, args.distractor)
    _save(img_rm, out_dir / "2_remove_distractor.png")
    print("[OK] remove_distractor saved")

    # 4) replace texture (requires texture)
    if args.texture:
        tex = _to_np_rgb(Path(args.texture))
        img_tex = tools.replace_texture(img, target_object=args.target, texture_image=tex)
        _save(img_tex, out_dir / "3_replace_texture.png")
        print("[OK] replace_texture saved")
    else:
        print("[SKIP] replace_texture (no --texture)")

    # 5) replace background (requires texture)
    if args.texture:
        tex = _to_np_rgb(Path(args.texture))
        img_bg = tools.replace_background(img, region_prompt=args.floor_prompt, texture_image=tex, alpha=1.0)
        _save(img_bg, out_dir / "4_replace_background.png")
        print("[OK] replace_background saved")
    else:
        print("[SKIP] replace_background (no --texture)")

    # 6) simulate inference input tensor
    # Convert HWC uint8 -> BCHW float32 in [0,1]
    proc = img_overlay
    t = torch.from_numpy(proc).permute(2, 0, 1).float() / 255.0
    obs = {"dummy_rgb": t.unsqueeze(0)}  # (B=1,C,H,W)

    print("[OK] dummy observation tensor shape:", tuple(obs["dummy_rgb"].shape), "dtype:", obs["dummy_rgb"].dtype)

    # If you want to plug into your policy, you can replace obs key with your real visual_key and call policy.select_action.
    print("Done. Check outputs in:", str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


