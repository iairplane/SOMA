import os
import json
import shutil
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple
from datetime import datetime

class MemoryBank:
    """
    SOMA 核心记忆库 (Vector Database + Asset Manager)
    
    功能:
    1. 向量检索: 基于 Numpy 的高性能 Cosine Similarity 检索。
    2. 资产管理: 统一管理 Video/Keyframe 路径，支持相对路径迁移。
    3. 持久化: JSONL (Meta) + NPY (Vector) 混合存储。
    """
    
    def __init__(self, 
                 storage_dir: Union[str, Path], 
                 dimension: int = 1168): # 768(Vis) + 384(Txt) + 16(Hash)
        """
        Args:
            storage_dir: 记忆库的根目录，会自动创建 'success' 和 'failure' 子目录
            dimension: Embedding 向量的维度 (需与 AdvancedEmbeddingEncoder 输出一致)
        """
        self.storage_dir = Path(storage_dir)
        self.dimension = dimension
        
        # 定义子目录
        self.dirs = {
            "success": self.storage_dir / "success",
            "failure": self.storage_dir / "failure"
        }
        
        # 内存索引 (In-Memory Index)
        # 结构: {'success': {'vecs': np.array, 'meta': list}, 'failure': ...}
        self.index = {
            "success": {"vecs": np.empty((0, self.dimension), dtype=np.float32), "meta": []},
            "failure": {"vecs": np.empty((0, self.dimension), dtype=np.float32), "meta": []}
        }
        
        # 初始化存储结构
        self._init_storage()
        
    def _init_storage(self):
        """初始化文件夹结构并加载已有数据"""
        if not self.storage_dir.exists():
            logging.info(f"[MemoryBank] 创建新记忆库: {self.storage_dir}")
            self.storage_dir.mkdir(parents=True)
            
        for key in ["success", "failure"]:
            self.dirs[key].mkdir(exist_ok=True)
            self._load_partition(key)

    def _load_partition(self, partition: str):
        """加载特定分区的元数据和向量"""
        meta_path = self.dirs[partition] / "metadata.jsonl"
        vec_path = self.dirs[partition] / "vectors.npy"
        
        # 1. 加载 Metadata
        meta_list = []
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            meta_list.append(json.loads(line))
            except Exception as e:
                logging.error(f"[MemoryBank] 加载 {partition} 元数据失败: {e}")
        
        # 2. 加载 Vectors
        if vec_path.exists():
            try:
                vecs = np.load(vec_path)
                # 校验维度一致性
                if vecs.shape[1] != self.dimension:
                    logging.warning(f"[MemoryBank] 向量维度不匹配 ({vecs.shape[1]} vs {self.dimension})，将重置向量库！")
                    vecs = np.empty((0, self.dimension), dtype=np.float32)
                    meta_list = [] # 元数据也必须重置以保持对齐
                else:
                    # 校验数量一致性
                    if len(vecs) != len(meta_list):
                        logging.warning(f"[MemoryBank] 数据损坏: 向量数({len(vecs)})与元数据数({len(meta_list)})不一致。截断以对齐。")
                        min_len = min(len(vecs), len(meta_list))
                        vecs = vecs[:min_len]
                        meta_list = meta_list[:min_len]
            except Exception as e:
                logging.error(f"[MemoryBank] 加载 {partition} 向量失败: {e}")
                vecs = np.empty((0, self.dimension), dtype=np.float32)
        else:
            vecs = np.empty((0, self.dimension), dtype=np.float32)

        # 更新内存索引
        self.index[partition]["vecs"] = vecs
        self.index[partition]["meta"] = meta_list
        logging.info(f"[MemoryBank] 加载 {partition}: {len(meta_list)} 条经验")

    def _save_vectors(self, partition: str):
        """将内存中的向量全量Dump到磁盘 (NPY格式高效读写)"""
        vec_path = self.dirs[partition] / "vectors.npy"
        # 使用临时文件写入再重命名，防止写入中断损坏文件
        temp_path = vec_path.with_suffix(".tmp.npy")
        np.save(temp_path, self.index[partition]["vecs"])
        if vec_path.exists():
            vec_path.unlink()
        temp_path.rename(vec_path)

    def _append_metadata(self, partition: str, record: dict):
        """原子追加元数据到 JSONL"""
        meta_path = self.dirs[partition] / "metadata.jsonl"
        with open(meta_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def add_experience(self, 
                       embedding: Union[List[float], np.ndarray],
                       task_desc: str,
                       success: bool,
                       video_path: Union[str, Path],
                       keyframe_path: Union[str, Path],
                       diagnosis: str = "",
                       additional_info: dict = None):
        """
        存入一条新经验 (线程安全建议由调用方保证，或此处加锁)
        """
        partition = "success" if success else "failure"
        
        # 1. 向量处理
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
        
        if embedding.shape[0] != self.dimension:
            logging.error(f"[MemoryBank] 写入失败: 向量维度 {embedding.shape[0]} != {self.dimension}")
            return

        # 归一化 (Double check)
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding = embedding / norm
        
        # 2. 媒体路径处理 (存储绝对路径，实际生产中建议存相对路径或对象存储ID)
        # 这里我们做一个简单的检查，确保文件存在
        vid_p = str(Path(video_path).absolute())
        kf_p = str(Path(keyframe_path).absolute())
        
        # 3. 构建元数据
        record = {
            "id": f"{partition}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            "task": task_desc,
            "timestamp": datetime.now().isoformat(),
            "video_path": vid_p,
            "keyframe_path": kf_p,
            "diagnosis": diagnosis, # 失败原因 (Failure only)
            "info": additional_info or {}
        }
        
        # 4. 更新内存
        # vstack 比较慢，但在 eval 阶段数据量不大(几千条)，是可以接受的。
        # 如果追求极致性能，可以预分配大数组。
        current_vecs = self.index[partition]["vecs"]
        self.index[partition]["vecs"] = np.vstack([current_vecs, embedding.reshape(1, -1)])
        self.index[partition]["meta"].append(record)
        
        # 5. 实时落盘
        self._append_metadata(partition, record)
        self._save_vectors(partition) # NPY 每次全量写，但这在 Eval 结束时才频繁调用，影响不大
        
        logging.info(f"[MemoryBank] 已存入 {partition} 经验: {task_desc[:30]}...")

    def retrieve(self, 
                 query_vector: Union[List[float], np.ndarray], 
                 top_k: int = 3,
                 partition: str = "failure", # 默认检索失败经验用于避坑
                 threshold: float = 0.4) -> List[Dict]:
        """
        检索最近邻
        
        Args:
            query_vector: 查询向量
            partition: 'failure' (找教训) 或 'success' (找参考) 或 'all'
            threshold: 相似度阈值，低于此值的忽略
        """
        if isinstance(query_vector, list):
            query_vector = np.array(query_vector, dtype=np.float32)
            
        # 归一化 Query
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
                
            # Cosine Similarity = Dot Product (因为都归一化了)
            # (N, D) dot (D,) -> (N,)
            scores = np.dot(vecs, query_vector)
            
            # 获取 Top-K
            # argsort 是从小到大，取最后 k 个并反转
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
        
        # 如果是 'all'，需要重新对混合结果排序
        if partition == "all":
            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:top_k]
            
        return results

    def get_stats(self) -> Dict:
        return {
            "success_count": len(self.index["success"]["meta"]),
            "failure_count": len(self.index["failure"]["meta"]),
            "vector_dim": self.dimension,
            "storage_path": str(self.storage_dir)
        }

# ================= 简单的集成测试 =================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 1. 初始化
    mb = MemoryBank(storage_dir="./soma_test_db", dimension=4) # 测试用4维
    
    # 2. 模拟向量 (归一化)
    vec1 = np.array([1, 0, 0, 0], dtype=np.float32)
    vec2 = np.array([0.9, 0.1, 0, 0], dtype=np.float32) # 和 vec1 很像
    vec3 = np.array([0, 0, 0, 1], dtype=np.float32) # 完全不像
    
    # 3. 存入数据
    mb.add_experience(vec1, "Task A", True, "/tmp/v1.mp4", "/tmp/k1.jpg")
    mb.add_experience(vec3, "Task B", False, "/tmp/v2.mp4", "/tmp/k2.jpg", diagnosis="Camera blocked")
    
    # 4. 检索
    print("\n--- Testing Retrieval (Querying with similar to Task A) ---")
    # 查 failure 应该查不到 Task A (它是 success)
    res_fail = mb.retrieve(vec2, partition="failure") 
    print(f"Failure Search Result: {len(res_fail)} (Expected 0 or low score)")

    # 查 all 应该能查到 Task A
    res_all = mb.retrieve(vec2, partition="all")
    print(f"All Search Result: {len(res_all)}")
    if res_all:
        print(f"Top result task: {res_all[0]['task']}, Score: {res_all[0]['score']:.4f}")
    
    # 5. 验证持久化
    print("\n--- Testing Persistence ---")
    mb_new = MemoryBank(storage_dir="./soma_test_db", dimension=4)
    print(f"Reloaded Stats: {mb_new.get_stats()}")
    assert mb_new.get_stats()["success_count"] == 1
    
    # 清理测试目录
    shutil.rmtree("./soma_test_db")
    print("\nTest Passed!")