import logging
from dataclasses import dataclass, field
from typing import Any, List

import numpy as np

@dataclass
class RollbackState:
    active: bool = False
    phase: str = ""  # "reverse" | "buffer" | "done"
    reverse_steps_total: int = 100
    buffer_steps_total: int = 25
    reverse_step_idx: int = 0
    buffer_step_idx: int = 0
    reverse_action: np.ndarray | None = None  # (B, D)
    reason: str = ""
    
    # [优化] 记录动作维度，防止硬编码
    action_dim: int = 7 

    def start_from_accumulated(self, accumulated_actions: np.ndarray, *, reverse_steps: int = 100, buffer_steps: int = 25, reason: str = ""):
        """
        accumulated_actions: shape (B, D) where D is action dim
        """
        self.active = True
        self.phase = "reverse"
        self.reverse_steps_total = int(reverse_steps)
        self.buffer_steps_total = int(buffer_steps)
        self.reverse_step_idx = 0
        self.buffer_step_idx = 0
        
        # [优化] 自动获取维度
        if accumulated_actions.ndim > 1:
            self.action_dim = accumulated_actions.shape[-1]
        
        # 计算每一步的“倒车”向量
        # 加上 max(1, ...) 防止除以 0
        self.reverse_action = -accumulated_actions / max(1, self.reverse_steps_total)
        self.reason = reason
        
        logging.warning(
            f"[CTRL] Rollback START | Reason: {reason} | Steps: {self.reverse_steps_total} | Dim: {self.action_dim}"
        )

    def step_action(self, num_envs: int) -> np.ndarray:
        if not self.active or self.reverse_action is None:
            raise RuntimeError("RollbackState.step_action called but rollback not active")

        # === 阶段 1: 倒车 (Reverse) ===
        if self.phase == "reverse":
            self.reverse_step_idx += 1
            if self.reverse_step_idx >= self.reverse_steps_total:
                # 倒车结束，进入缓冲或结束
                self.phase = "buffer" if self.buffer_steps_total > 0 else "done"
            
            # 返回计算好的反向向量
            return self.reverse_action.astype(np.float32)

        # === 阶段 2: 缓冲 (Buffer) ===
        if self.phase == "buffer":
            self.buffer_step_idx += 1
            
            # [优化] 动态生成零动作，而不是写死 7 维
            # 默认最后一位是夹爪，设为 -1 (张开)
            dummy = np.zeros(self.action_dim, dtype=np.float32)
            if self.action_dim >= 1:
                dummy[-1] = -1.0 
            
            if self.buffer_step_idx >= self.buffer_steps_total:
                self.phase = "done"
            
            return np.array([dummy] * num_envs, dtype=np.float32)

        # === 阶段 3: 结束 (Done) ===
        self.reset() # 自动重置
        # 递归调用一次以返回正常的 Policy 动作（或让外部处理）
        # 这里返回全0，表示把控制权交还给 Policy
        return np.zeros((num_envs, self.action_dim), dtype=np.float32)

    def reset(self):
        """强制重置状态"""
        self.active = False
        self.phase = "done"
        self.reverse_action = None
        self.reason = ""


@dataclass
class KeyStepRetryState:
    key_steps: dict[str, int] = field(default_factory=lambda: {"start": 0, "end": 0, "timeout_grace": 0})
    armed: bool = False
    triggered: bool = False

    def update_from_control_flags(self, control_flags: dict[str, Any]):
        ks = control_flags.get("key_steps")
        if isinstance(ks, dict) and "start" in ks and "end" in ks:
            self.key_steps = {
                "start": int(ks.get("start", 0)),
                "end": int(ks.get("end", 0)),
                "timeout_grace": int(ks.get("timeout_grace", 0)),
            }
            self.armed = True
            self.triggered = False # 重置触发状态

        # 支持强制重试指令
        if control_flags.get("key_step_retry") is True:
            self.armed = True
            # 注意：不立即设为 triggered，而是等待下一次 check

    def should_trigger(self, *, step: int, success_any: bool) -> bool:
        if not self.armed or self.triggered:
            return False
        
        # 如果已经成功了，就取消 armed，防止误触发
        if success_any:
            self.armed = False 
            return False
            
        end = self.key_steps.get("end", 0)
        grace = self.key_steps.get("timeout_grace", 0)
        
        # 只有当设定了有效的 end (>0) 且步数超时时才触发
        return step >= (end + grace) and end > 0

    def mark_triggered(self):
        self.triggered = True
        logging.warning(f"[CTRL] KeyStepRetry TRIGGERED | Config: {self.key_steps}")
    
    def reset(self):
        self.armed = False
        self.triggered = False
        self.key_steps = {"start": 0, "end": 0, "timeout_grace": 0}


@dataclass
class TaskDecomposeState:
    subtasks: List[str] = field(default_factory=list)
    subtask_idx: int = 0
    
    # [优化] 增加完成计数，防止抖动
    success_hold_steps: int = 0 

    def update_from_control_flags(self, control_flags: dict[str, Any]):
        sts = control_flags.get("subtasks")
        if isinstance(sts, list) and all(isinstance(x, str) for x in sts) and len(sts) > 0:
            # 只有当新任务列表与当前不同时才更新
            if sts != self.subtasks:
                self.subtasks = sts
                self.subtask_idx = 0
                self.success_hold_steps = 0
                logging.info(f"[CTRL] TaskDecompose UPDATED | Steps: {len(sts)} | First: {sts[0]}")

    def current_task(self, default_task: str) -> str:
        if self.subtasks and 0 <= self.subtask_idx < len(self.subtasks):
            # [Diagram] Logic: Input Default -> Check Index -> Output Subtask[i]
            return self.subtasks[self.subtask_idx]
        return default_task

    def maybe_advance(self, *, success_any: bool):
        """
        注意：对于并行环境 (VectorEnv)，success_any=True 意味着只要有一个环境成功就推进。
        这在 Evaluation 时是可接受的（我们通常取最好的那个），但在 Training 时可能有问题。
        """
        if not self.subtasks:
            return

        if success_any:
            # 增加一个小的保持计数，防止因为传感器噪声瞬间抖动导致跳过任务
            self.success_hold_steps += 1
            if self.success_hold_steps >= 2: # 连续 2 帧检测到成功才推进
                if self.subtask_idx < len(self.subtasks) - 1:
                    self.subtask_idx += 1
                    self.success_hold_steps = 0 # 重置计数
                    logging.info(f"[CTRL] Task ADVANCED >>> Step {self.subtask_idx}: {self.subtasks[self.subtask_idx]}")
                else:
                    # 已经是最后一步了
                    pass
        else:
            self.success_hold_steps = 0
            
    def reset(self):
        self.subtasks = []
        self.subtask_idx = 0
        self.success_hold_steps = 0