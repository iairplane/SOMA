import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import os
import tempfile

def make_info_schema(
    *,
    assets: Optional[dict] = None,
    task_plan: Optional[dict] = None,
    mcp_trace: Optional[list] = None,
) -> dict:
    """统一的 SOMA Memory schema

    info.task_plan 建议字段：
    - key_frame_range: {start:int, end:int}
    - success_max_step: int
    - subtask_success_max_steps: list[int]
    - rollback: {reverse_steps:int, buffer_steps:int}
    """

    return {
        "assets": assets or {},
        "task_plan": task_plan or {},
        "mcp_trace": mcp_trace or [],
    }


def get_task_plan_defaults() -> dict:
    return {
        "key_frame_range": {"start": 0, "end": 0},
        "success_max_step": 0,
        "subtask_success_max_steps": [],
        "rollback": {"reverse_steps": 100, "buffer_steps": 25},
    }


class MemoryBank:
    """SOMA 核心记忆库 (Vector DB + Asset Manager)"""

    def __init__(self, storage_dir: Union[str, Path], dimension: int = 1168):
        self.storage_dir = Path(storage_dir)
        self.dimension = dimension

        self.dirs = {
            "success": self.storage_dir / "success",
            "failure": self.storage_dir / "failure",
        }

        self.index = {
            "success": {"vecs": np.empty((0, self.dimension), dtype=np.float32), "meta": []},
            "failure": {"vecs": np.empty((0, self.dimension), dtype=np.float32), "meta": []},
        }

        self._init_storage()

    def _init_storage(self):
        if not self.storage_dir.exists():
            logging.info(f"[MemoryBank] 创建新记忆库: {self.storage_dir}")
            self.storage_dir.mkdir(parents=True)

        for key in ["success", "failure"]:
            self.dirs[key].mkdir(exist_ok=True)
            self._load_partition(key)

    def _load_partition(self, partition: str):
        meta_path = self.dirs[partition] / "metadata.jsonl"
        vec_path = self.dirs[partition] / "vectors.npy"

        meta_list = []
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            meta_list.append(json.loads(line))
            except Exception as e:
                logging.error(f"[MemoryBank] 加载 {partition} 元数据失败: {e}")

        if vec_path.exists():
            try:
                vecs = np.load(vec_path)
                if vecs.ndim != 2 or vecs.shape[1] != self.dimension:
                    logging.warning(
                        f"[MemoryBank] 向量维度不匹配 ({getattr(vecs, 'shape', None)} vs {self.dimension})，将重置向量库！"
                    )
                    vecs = np.empty((0, self.dimension), dtype=np.float32)
                    meta_list = []
                else:
                    if len(vecs) != len(meta_list):
                        logging.warning(
                            f"[MemoryBank] 数据损坏: 向量数({len(vecs)})与元数据数({len(meta_list)})不一致。截断以对齐。"
                        )
                        min_len = min(len(vecs), len(meta_list))
                        vecs = vecs[:min_len]
                        meta_list = meta_list[:min_len]
            except Exception as e:
                logging.error(f"[MemoryBank] 加载 {partition} 向量失败: {e}")
                vecs = np.empty((0, self.dimension), dtype=np.float32)
        else:
            vecs = np.empty((0, self.dimension), dtype=np.float32)

        self.index[partition]["vecs"] = vecs
        self.index[partition]["meta"] = meta_list
        logging.info(f"[MemoryBank] 加载 {partition}: {len(meta_list)} 条经验")

    def _save_vectors(self, partition: str):
        vec_path = self.dirs[partition] / "vectors.npy"
        
        # [修复] 确保父目录存在
        vec_path.parent.mkdir(parents=True, exist_ok=True)
        
        # [修复] 使用 PID 命名临时文件，防止并发冲突
        # 使用 tempfile 模块在同一目录下创建临时文件
        fd, temp_name = tempfile.mkstemp(
            dir=str(vec_path.parent), 
            prefix=f"vectors_{os.getpid()}_", 
            suffix=".tmp.npy"
        )
        
        try:
            # 写入数据到临时文件
            with os.fdopen(fd, 'wb') as f:
                np.save(f, self.index[partition]["vecs"])
            
            # [修复] 原子性替换：在 Unix 上 replace 是原子的
            # 这样即使程序在这一秒崩溃，旧的 vectors.npy 依然是完整的
            os.replace(temp_name, str(vec_path))
            
        except Exception as e:
            logging.error(f"[SOMA] 关键错误：无法保存向量库 {partition} - {e}")
            if os.path.exists(temp_name):
                os.remove(temp_name)

    def _append_metadata(self, partition: str, record: dict):
        meta_path = self.dirs[partition] / "metadata.jsonl"
        with open(meta_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def add_experience(
        self,
        embedding: Union[List[float], np.ndarray],
        task_desc: str,
        success: bool,
        video_path: Union[str, Path],
        keyframe_path: Union[str, Path],
        diagnosis: str = "",
        info: Optional[dict] = None,
    ):
        partition = "success" if success else "failure"

        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)

        if embedding.shape[0] != self.dimension:
            logging.error(f"[MemoryBank] 写入失败: 向量维度 {embedding.shape[0]} != {self.dimension}")
            return

        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding = embedding / norm

        vid_p = str(Path(video_path).absolute())
        kf_p = str(Path(keyframe_path).absolute())

        record = {
            "id": f"{partition}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            "task": task_desc,
            "timestamp": datetime.now().isoformat(),
            "video_path": vid_p,
            "keyframe_path": kf_p,
            "diagnosis": diagnosis,
            "info": info if isinstance(info, dict) else make_info_schema(task_plan=get_task_plan_defaults()),
        }

        current_vecs = self.index[partition]["vecs"]
        self.index[partition]["vecs"] = np.vstack([current_vecs, embedding.reshape(1, -1)])
        self.index[partition]["meta"].append(record)

        self._append_metadata(partition, record)
        self._save_vectors(partition)
    
        logging.info(f"[MemoryBank] 已存入 {partition} 经验: {task_desc[:30]}...")

    def get_best_task_plan(self, *, partition: str = "success") -> dict:
        """取一个可用的 task_plan（用于控制流默认值/原型验证）。

        当前策略：从指定 partition 的 meta 中倒序找第一个包含 info.task_plan 的记录。
        """

        if partition not in self.index:
            return get_task_plan_defaults()

        for rec in reversed(self.index[partition]["meta"]):
            info = rec.get("info") if isinstance(rec, dict) else None
            task_plan = info.get("task_plan") if isinstance(info, dict) else None
            if isinstance(task_plan, dict):
                merged = get_task_plan_defaults()
                merged.update(task_plan)
                # deep merge for nested
                if isinstance(task_plan.get("key_frame_range"), dict):
                    merged["key_frame_range"].update(task_plan["key_frame_range"])
                if isinstance(task_plan.get("rollback"), dict):
                    merged["rollback"].update(task_plan["rollback"])
                return merged

        return get_task_plan_defaults()

    def retrieve(
        self,
        query_vector: Union[List[float], np.ndarray],
        top_k: int = 3,
        partition: str = "failure",
        threshold: float = 0.4,
    ) -> List[Dict]:
        if isinstance(query_vector, list):
            query_vector = np.array(query_vector, dtype=np.float32)

        norm = np.linalg.norm(query_vector)
        if norm > 1e-6:
            query_vector = query_vector / norm

        target_partitions = [partition] if partition != "all" else ["success", "failure"]

        results = []
        for part in target_partitions:
            vecs = self.index[part]["vecs"]
            meta = self.index[part]["meta"]

            if len(vecs) == 0:
                continue

            scores = np.dot(vecs, query_vector)
            k_idx = min(top_k, len(scores))
            top_indices = np.argsort(scores)[-k_idx:][::-1]

            for idx in top_indices:
                score = float(scores[idx])
                if score < threshold:
                    continue
                item = meta[idx].copy()
                item["score"] = score
                item["type"] = part
                results.append(item)

        if partition == "all":
            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:top_k]

        return results

    def get_stats(self) -> Dict:
        return {
            "success_count": len(self.index["success"]["meta"]),
            "failure_count": len(self.index["failure"]["meta"]),
            "vector_dim": self.dimension,
            "storage_path": str(self.storage_dir),
        }
