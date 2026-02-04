import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RollbackState:
    active: bool = False
    phase: str = ""  # "reverse" | "buffer" | "done"
    reverse_steps_total: int = 100
    buffer_steps_total: int = 25
    reverse_step_idx: int = 0
    buffer_step_idx: int = 0
    reverse_action: np.ndarray | None = None  # (B,7)
    reason: str = ""

    def start_from_accumulated(self, accumulated_actions: np.ndarray, *, reverse_steps: int = 100, buffer_steps: int = 25, reason: str = ""):
        self.active = True
        self.phase = "reverse"
        self.reverse_steps_total = int(reverse_steps)
        self.buffer_steps_total = int(buffer_steps)
        self.reverse_step_idx = 0
        self.buffer_step_idx = 0
        self.reverse_action = -accumulated_actions / max(1, self.reverse_steps_total)
        self.reason = reason
        logging.warning(
            f"[CTRL] Rollback start reason={reason} reverse_steps={self.reverse_steps_total} buffer_steps={self.buffer_steps_total}"
        )

    def step_action(self, num_envs: int) -> np.ndarray:
        if not self.active or self.reverse_action is None:
            raise RuntimeError("RollbackState.step_action called but rollback not active")

        if self.phase == "reverse":
            self.reverse_step_idx += 1
            if self.reverse_step_idx >= self.reverse_steps_total:
                self.phase = "buffer" if self.buffer_steps_total > 0 else "done"
            return self.reverse_action.astype(np.float32)

        if self.phase == "buffer":
            self.buffer_step_idx += 1
            # dummy action: keep gripper open (-1)
            dummy = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
            if self.buffer_step_idx >= self.buffer_steps_total:
                self.phase = "done"
            return np.array([dummy] * num_envs, dtype=np.float32)

        # done
        self.active = False
        self.reverse_action = None
        self.reason = ""
        return self.step_action(num_envs)


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

        if control_flags.get("key_step_retry") is True:
            self.armed = True

    def should_trigger(self, *, step: int, success_any: bool) -> bool:
        if not self.armed or self.triggered:
            return False
        if success_any:
            return False
        end = self.key_steps.get("end", 0)
        grace = self.key_steps.get("timeout_grace", 0)
        return step >= (end + grace) and end > 0

    def mark_triggered(self):
        self.triggered = True
        logging.warning(f"[CTRL] key_step_retry triggered key_steps={self.key_steps}")


@dataclass
class TaskDecomposeState:
    subtasks: list[str] = field(default_factory=list)
    subtask_idx: int = 0

    def update_from_control_flags(self, control_flags: dict[str, Any]):
        sts = control_flags.get("subtasks")
        if isinstance(sts, list) and all(isinstance(x, str) for x in sts) and len(sts) > 0:
            self.subtasks = sts
            self.subtask_idx = 0
            logging.info(f"[CTRL] task_decompose set subtasks n={len(sts)}")

    def current_task(self, default_task: str) -> str:
        if self.subtasks and 0 <= self.subtask_idx < len(self.subtasks):
            return self.subtasks[self.subtask_idx]
        return default_task

    def maybe_advance(self, *, success_any: bool):
        if not self.subtasks:
            return
        if success_any and self.subtask_idx < len(self.subtasks) - 1:
            self.subtask_idx += 1
            logging.info(f"[CTRL] advance to subtask_idx={self.subtask_idx}")


