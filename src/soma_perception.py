"""
SOMA: Self-Organizing Memory Agent
==================================

Description:
------------
This module implements the Perception Module, which acts as the Strategic Orchestrator 
for the SOMA framework. It governs the core cognitive loop of the agent: 
Observe -> Retrieve (RAG hints) -> Plan (VLM) -> Act (MCP tools) -> Verify (Closed-Loop).



Key Features:
- Closed-Loop Verification: Leverages the Vision-Language Model (VLM) to perform a 
  self-reflective verification step, ensuring visual modifications are logically sound 
  before committing to them. Supports blocking retries upon verification failure.
- Execution Quotas: Implements a `modification_active_limit` to prevent infinite or 
  excessive visual augmentations. Once the quota is reached, only temporal flow tools 
  (like Encore/Rollback) are permitted.
- State Persistence: Automatically tracks tool usage counts and resets them dynamically 
  when an Encore (rollback) maneuver is triggered, ensuring the agent doesn't get 
  permanently locked out of visual tools after a recovery.
"""

import base64
import io
import logging
import time
import cv2
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

from soma_tools import MCPTools
from soma_vlm import Qwen3VLAPIClient


def _maybe_load_image_any(src: Any) -> np.ndarray | None:
    """Load image from possible forms:
    - np.ndarray
    - path str/Path
    - base64 png string
    """
    if src is None:
        return None

    if isinstance(src, np.ndarray):
        return src

    if isinstance(src, (str, Path)):
        s = str(src)
        p = Path(s)
        if p.exists():
            return np.array(Image.open(p).convert("RGB"))

        # Attempt base64 decoding
        try:
            if "," in s:
                s = s.split(",", 1)[1]
            raw = base64.b64decode(s)
            return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
        except Exception:
            return None

    return None


class PerceptionModule:
    """SOMA Strategic Orchestrator
    
    Responsibilities: Observe -> (RAG hints) -> Plan(VLM) -> Act(MCP tools) -> Verify(VLM Loop)
    Features: Supports blocking retries on Verification failure, supports Encore to reset the visual step counter.
    """

    def __init__(
        self,
        vlm_client: Qwen3VLAPIClient | None = None,
        *,
        sam3_base_url: str = "http://127.0.0.1:5001",
        tool_timeout_s: float = 10.0,
    ):
        self.vlm = vlm_client or Qwen3VLAPIClient()
        self.tools = MCPTools(sam3_base_url=sam3_base_url, timeout_s=tool_timeout_s)
        self.retry_count = 0
        
        # Debug output directory
        self.debug_output_dir = Path("/mnt/disk1/shared_data/lzy/SOMA-debug/perception_outputs")
        # Allow visual modifications for the first n steps; afterward, only allow Encore (rollback)
        self.visual_tool_steps = 0
        self.modification_active_limit = 40

    def _save_image(
        self,
        result_rgb: np.ndarray,
        save_path: Path,
        save_original: bool = False,
        original_image: np.ndarray = None
    ):
        """
        Save image for debugging purposes.
        Because this is a Client-Server (CS) architecture, the Client side cannot directly access 
        the Service side's mask. Therefore, only the logic for saving the Result and Original is retained here.
        """
        try:
            if save_path is None: return
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 1. Save the processed image (RGB -> BGR for OpenCV)
            result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(save_path), result_bgr)
            
            # 2. Save the original image
            if save_original and original_image is not None:
                original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
                original_path = save_path.with_name(f"{save_path.stem}_original{save_path.suffix}")
                cv2.imwrite(str(original_path), original_bgr)
                
            logging.info(f"[SOMA Debug] Saved frame to {save_path}")
        
        except Exception as e:
            logging.error(f"Failed to save image: {e}", exc_info=True)

    def _verify_modification(
        self, 
        original_img: np.ndarray, 
        modified_img: np.ndarray, 
        original_task: str, 
        refined_task: str
    ) -> bool:
        """
        [Closed-Loop Verification] Feed the original/modified images and task instructions back 
        into the VLM to evaluate whether the visual modification is logically sound and helpful.
        """
        logging.info("[SOMA] Verifying modification...")
        
        # Construct Verification Prompt
        # Note: This assumes the VLM supports multi-image input. If not, passing only modified_img is an alternative.
        # A detailed verification prompt is constructed here for completeness.
        prompt = (
            f"I have performed a visual modification to assist a robot policy.\n"
            f"Original Task: '{original_task}'\n"
            f"Refined Task: '{refined_task}'\n\n"
            f"Image 1 is the Original View.\n"
            f"Image 2 is the Modified View (with overlay/removal).\n\n"
            f"Please verify:\n"
            f"1. Does the visual modification correctly highlight the target or remove the distractor described in the task suggestions?\n"
            f"2. Is the refined task instruction clear and accurate?\n"
            f"If the modification is wrong, misleading, or unnecessary, return 'valid': false.\n"
            f"Respond strictly in JSON format: {{'valid': boolean, 'reason': 'string'}}."
        )

        try:
            # Call VLM (assuming self.vlm exposes the underlying client or a generic chat interface)
            # Reusing the interface logic from orchestrate_perception
            original_b64_img = self.vlm._encode_image(original_img)
            modified_b64_img = self.vlm._encode_image(modified_img)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url", 
                            "image_url": {"url": original_b64_img}
                        }, 
                        {
                            "type": "image_url", 
                            "image_url": {"url": modified_b64_img}
                        },
                        {
                            "type": "text", 
                            "text": prompt
                        }
                    ],
                }
            ]
            
            if hasattr(self.vlm, "client"):
                response = self.vlm.client.chat.completions.create(
                    model=self.vlm.model_id,
                    messages=messages,
                    max_tokens=256,
                    temperature=0.1
                )
                content = response.choices[0].message.content
                
                # Parse JSON
                import json
                # Simple JSON extraction logic
                json_str = content[content.find("{"):content.rfind("}")+1]
                result = json.loads(json_str)
                
                is_valid = result.get("valid", False)
                reason = result.get("reason", "No reason provided")
                
                if is_valid:
                    logging.info(f"[SOMA] Verification PASSED: {reason}")
                else:
                    logging.warning(f"[SOMA] Verification FAILED: {reason}")
                
                return is_valid, reason
            else:
                logging.warning("[SOMA] VLM client not accessible for verification, skipping.")
                return True, "VLM client not accessible" # Default to pass if verification is unavailable
                
        except Exception as e:
            logging.error(f"[SOMA] Verification error: {e}")
            return True, f"Verification error: {e}" # Do not block the pipeline on error
    
    def process_frame(
        self,
        image: np.ndarray,
        current_task: str,
        step: int,
        rag_context: Dict | None = None,
    ) -> Tuple[np.ndarray, str, Dict]:
        rag_context = rag_context or {}

        # 1. Prepare save paths (similar to original logic)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_base_path = self.debug_output_dir / f"frame_step_{step}_{timestamp}.jpg"

        # 2. Construct RAG Context
        rag_str_base = ""
        rag_hints: dict[str, Any] = {}

        failures = rag_context.get("failure") or []
        successes = rag_context.get("success") or []

        if failures:
            diags = [str(f.get("diagnosis", "")) for f in failures[:2] if f.get("diagnosis")]
            if diags:
                rag_str_base = f"Warning! Past failures causes: {'; '.join(diags)}."

        def _has_tex(item: dict, key: str) -> bool:
            info = item.get("info") if isinstance(item, dict) else None
            assets = info.get("assets") if isinstance(info, dict) else None
            if not isinstance(assets, dict):
                return False
            val = assets.get(key)
            return bool(val)

        rag_hints["success_has_object_texture"] = bool(successes) and _has_tex(successes[0], "object_texture_png_b64")
        rag_hints["success_has_floor_texture"] = bool(successes) and _has_tex(successes[0], "floor_texture_png_b64")
        rag_hints["failure_has_object_texture"] = bool(failures) and _has_tex(failures[0], "object_texture_png_b64")
        rag_hints["failure_has_floor_texture"] = bool(failures) and _has_tex(failures[0], "floor_texture_png_b64")

        # Define retry parameters
        MAX_RETRIES = 0
        current_attempt = 0
        final_processed_img = image.copy()
        final_refined_task = current_task
        final_control_flags = {"encore": False}
        
        # Define tool sets
        visual_tools = ["visual_overlay", "remove_distractor", "replace_texture", "replace_background"]
        flow_tools = ["encore", "key_step_retry", "task_decompose"]

        # Additional prompt feedback (self-reflection)
        feedback_prompt = ""
        
        while current_attempt < MAX_RETRIES:
            current_attempt += 1
            
            # 1. Concatenate RAG + Feedback
            current_rag_str = rag_str_base
            if feedback_prompt:
                current_rag_str += f"\n[IMMEDIATE FEEDBACK] Your previous plan was rejected. Reason: {feedback_prompt}. Please fix the visual modification."
            
            # 2. VLM Planning
            try:
                plan = self.vlm.orchestrate_perception(
                    image=image,
                    task_desc=current_task,
                    rag_context=current_rag_str,
                    rag_hints=rag_hints,
                )
            except Exception as e:
                logging.error(f"[SOMA] Planning failed: {e}")
                break # Exit directly if planning fails

            tool_chain = plan.get("tool_chain", []) or []
            params = plan.get("params", {}) if isinstance(plan.get("params"), dict) else {}
            refined_task_text = plan.get("refined_task", current_task)
            task_plan = plan.get("task_plan", {}) if isinstance(plan.get("task_plan"), dict) else {}

            # 3. Filter tools based on the active limit
            # After the active limit is reached, filter out all visual modification tools, keeping only encore and control flow tools
            filtered_tool_chain = []
            
            # Use the internal counter to determine if visual modification is permitted
            allow_visual = self.visual_tool_steps <= self.modification_active_limit
            
            for tool in tool_chain:
                if tool in visual_tools:
                    if allow_visual:
                        filtered_tool_chain.append(tool)
                    else:
                        logging.debug(f"[SOMA] Visual tool '{tool}' blocked (Quota: {self.visual_tool_steps}/{self.modification_active_limit})")
                else:
                    # Non-visual tools (e.g., encore) are always executed
                    filtered_tool_chain.append(tool)
        
            tool_chain = filtered_tool_chain

            # 4. Tools Execution
            temp_img = image.copy()
            temp_flags = {"encore": False}
            if "key_steps" in task_plan: temp_flags["key_steps"] = task_plan.get("key_steps")
            if "subtasks" in task_plan: temp_flags["subtasks"] = task_plan.get("subtasks")
            
            if tool_chain:
                logging.info(f"[SOMA Step {step}] tool_chain={tool_chain} refined_task={refined_task_text}")
                
            image_modified_in_this_loop = False

            for tool_name in tool_chain:
                # print(f"Executing tool: {tool_name} with params: {params}")
                try:
                    tool_specific_params = params.get(tool_name, params)
                    if tool_name in visual_tools:
                        if tool_name == "remove_distractor":
                            targets = tool_specific_params.get("objects_to_remove")
                            if targets is None:
                                single_target = tool_specific_params.get("object_to_remove")
                                if single_target:
                                    targets = [single_target]
                            # print(f"remove_distractor target: {target}")
                            if isinstance(targets, list):
                                for tgt in targets:
                                    # print(f"Applying remove_distractor on target: {tgt}")
                                    if isinstance(tgt, str) and tgt:
                                        print(f"Applying remove_distractor on: {tgt}")
                                        # Assuming tools.remove_distractor supports multiple calls on the same image
                                        temp_img = self.tools.remove_distractor(temp_img, tgt)
                                        image_modified_in_this_loop = True

                        elif tool_name == "visual_overlay":
                            target = tool_specific_params.get("target_object")
                            # print(f"visual_overlay target: {target}")
                            if isinstance(target, str) and target:
                                # print(f"Applying visual overlay on target: {target}")
                                temp_img = self.tools.apply_visual_overlay(temp_img, target, color=(0, 255, 0))
                                image_modified_in_this_loop = True 
                        
                        elif tool_name == "replace_texture":
                            target = tool_specific_params.get("target_object")
                            source = tool_specific_params.get("source", "rag_success")
                            texture_key = tool_specific_params.get("texture_key", "object_texture_png_b64")

                            pick_list = successes if source == "rag_success" else failures
                            tex_img = None
                            if pick_list:
                                info = pick_list[0].get("info", {})
                                assets = info.get("assets", {}) if isinstance(info, dict) else {}
                                tex_img = _maybe_load_image_any(assets.get(texture_key))

                            if isinstance(target, str) and target and tex_img is not None:
                                temp_img = self.tools.replace_texture(temp_img, target_object=target, texture_image=tex_img)
                                image_modified_in_this_loop = True

                        elif tool_name == "replace_background":
                            region_prompt = tool_specific_params.get("region_prompt", "floor")
                            source = tool_specific_params.get("source", "rag_success")
                            texture_key = tool_specific_params.get("texture_key", "floor_texture_png_b64")
                            alpha = float(tool_specific_params.get("alpha", 1.0))

                            pick_list = successes if source == "rag_success" else failures
                            tex_img = None
                            if pick_list:
                                info = pick_list[0].get("info", {})
                                assets = info.get("assets", {}) if isinstance(info, dict) else {}
                                tex_img = _maybe_load_image_any(assets.get(texture_key))

                            if isinstance(region_prompt, str) and region_prompt and tex_img is not None:
                                temp_img = self.tools.replace_background(
                                    temp_img,
                                    region_prompt=region_prompt,
                                    texture_image=tex_img,
                                    alpha=alpha,
                                )
                                image_modified_in_this_loop = True
                            
                    elif tool_name in ("instruction_refine", "chaining_step"):
                        pass
                    
                    elif tool_name == "encore":
                        temp_flags["encore"] = True
                        self.retry_count += 1

                    elif tool_name == "key_step_retry":
                        temp_flags["key_step_retry"] = True
                        if "key_steps" in tool_specific_params:
                            temp_flags["key_steps"] = tool_specific_params.get("key_steps")

                    elif tool_name == "task_decompose":
                        temp_flags["task_decompose"] = True
                        subtasks = tool_specific_params.get("subtasks")
                        if isinstance(subtasks, list) and subtasks:
                            temp_flags["subtasks"] = subtasks

                except Exception as e:
                    logging.error(f"[SOMA] Tool {tool_name} failed: {e}")
            
            # 5. Verify Modification if there was a visual change
            verification_passed = True
            reject_reason = ""

            if image_modified_in_this_loop and allow_visual:
                is_valid, reason = self._verify_modification(
                    original_img=image,
                    modified_img=temp_img,
                    original_task=current_task,
                    refined_task=refined_task_text
                )
                
                if not is_valid:
                    verification_passed = False
                    reject_reason = reason
                    logging.warning(f"[SOMA] Attempt {current_attempt}/{MAX_RETRIES} Rejected: {reason}")
            
            # 6. Decision Time: Decide whether to retry based on the verification result
            if verification_passed:
                # Success -> Break the loop
                final_processed_img = temp_img
                final_refined_task = refined_task_text
                final_control_flags = temp_flags
                if image_modified_in_this_loop:
                    final_control_flags["image_modified"] = True
                break
            else:
                # Failure -> Proceed to the next retry
                feedback_prompt = reject_reason

        # 1. If the modified image is ultimately adopted, increase the counter by 10
        if final_control_flags.get("image_modified", False):
            self.visual_tool_steps += 10
            logging.info(f"[SOMA] Visual Step Counter: {self.visual_tool_steps}")
            # The increment steps can be adjusted based on actual conditions, or a more complex scoring system and image saving logic can be introduced.
            # if image_modified_in_this_loop:
            #     self._save_image(
            #         result_rgb=final_processed_img,
            #         save_path=save_base_path,
            #         save_original=True,
            #         original_image=image
            #     )
        else:
            self.visual_tool_steps += 10
            logging.info(f"[SOMA] No visual modification applied in final plan loop. Counter continue at {self.visual_tool_steps}.")    
        # 2. If Encore is triggered, reset the counter    
        if final_control_flags.get("encore", False):
            logging.warning("[SOMA] Encore triggered! Resetting visual tool steps to 0.")
            self.visual_tool_steps = 0

        return final_processed_img, final_refined_task, final_control_flags