import logging
import numpy as np
from typing import Tuple, Dict, Any

# 引入我们的组件
from soma_vlm import Qwen3VLAPIClient
from soma_tools import MCPTools

class PerceptionModule:
    """
    SOMA 战略编排器 (Strategic Orchestrator)
    负责: Observe -> Retrieve -> Plan -> Act
    """
    def __init__(self, 
                 vlm_client: Qwen3VLAPIClient = None,
                 device: str = "cuda", 
                 sam3_path: str = "./sam3.pt"):
        
        # 1. 初始化大脑
        self.vlm = vlm_client or Qwen3VLAPIClient()
        
        # 2. 初始化肢体 (工具箱)
        self.tools = MCPTools(device=device, sam3_checkpoint=sam3_path)
        
        # 状态追踪 (用于 Encore 逻辑)
        self.retry_count = 0 

    def process_frame(self, 
                      image: np.ndarray, 
                      current_task: str, 
                      step: int, 
                      rag_context: Dict = None) -> Tuple[np.ndarray, str, Dict]:
        """
        SOMA 主循环接口。
        
        Returns:
            processed_image: 修改后的图像 (用于 VLA 输入)
            refined_task: 修改后的任务指令 (用于 VLA 输入)
            action_metadata: 包含 encore_flag 等控制信息
        """
        
        # 1. 准备 RAG 上下文字符串
        rag_str = ""
        if rag_context and rag_context.get("failure"):
            # 提取历史失败教训
            diags = [f['diagnosis'] for f in rag_context['failure'][:2]]
            rag_str = f"Warning! Past failures causes: {'; '.join(diags)}."

        # 2. 调用 VLM 进行战略编排 (Orchestration)
        # 我们询问 VLM：面对当前图像和历史教训，我该用什么工具？
        try:
            plan = self.vlm.orchestrate_perception(
                image=image, 
                task_desc=current_task, 
                rag_context=rag_str
            )
        except Exception as e:
            logging.error(f"[SOMA] Planning failed: {e}")
            # 兜底：不使用工具
            return image, current_task, {}

        # 解析 VLM 返回的 JSON
        # 预期格式: {'tool_chain': ['eraser', 'chaining'], 'params': {...}, 'refined_task': '...'}
        tool_chain = plan.get("tool_chain", [])
        params = plan.get("params", {})
        refined_task_text = plan.get("refined_task", current_task)
        
        # 3. 执行工具链 (Execution Loop)
        processed_img = image.copy()
        control_flags = {"encore": False} # 控制信号

        if tool_chain:
            logging.info(f"⚡ [SOMA Step {step}] Strategy: {tool_chain} | Task: {refined_task_text}")
            
            for tool_name in tool_chain:
                
                # --- 工具 1: Eraser (无关物体擦除) ---
                if tool_name == "remove_distractor":
                    target = params.get("object_to_remove")
                    if target:
                        processed_img = self.tools.remove_distractor(processed_img, target)

                # --- 工具 2: Paint-to-Action (表面换色) ---
                elif tool_name == "visual_overlay":
                    target = params.get("target_object")
                    if target:
                        # 默认用绿色高亮，模拟 VLA 喜欢的颜色
                        processed_img = self.tools.apply_visual_overlay(processed_img, target, color=(0, 255, 0))

                # --- 工具 3: Chaining-Step (长程任务拆分) ---
                elif tool_name == "instruction_refine" or tool_name == "chaining_step":
                    # 这是一个"逻辑工具"。VLM 认为原指令太模糊 (如 "Clean table")，
                    # 于是将其替换为当前步骤的具体指令 (如 "Pick up the sponge")。
                    # 这个逻辑已经在 VLM 的输出 `refined_task` 里体现了，这里只需确认应用即可。
                    logging.info(f"   -> Refined Instruction: {current_task} => {refined_task_text}")
                    # 注意：我们直接返回 refined_task_text

                # --- 工具 4: Encore (失败任务重试/状态回滚) ---
                elif tool_name == "encore" or tool_name == "spatial_recovery":
                    # 这是一个"控制流工具"。
                    # 场景：VLM 发现机器人抓空了，或者物体滑落了。
                    # 动作：不修改图片，而是发出信号，让 eval 脚本回滚状态或重试。
                    logging.warning("   -> Encore Triggered! (Retry/Rollback requested)")
                    control_flags["encore"] = True
                    self.retry_count += 1

        return processed_img, refined_task_text, control_flags