"""人脸特征库：用「*.npy + index.csv」替代数据库，实现轻量持久化。

每个学生一条 512 维归一化特征向量；检索使用内积（等价于余弦相似度），
优先使用 FAISS ``IndexFlatIP``，未安装时自动退化为 NumPy 矩阵乘法。
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config


class FaceGallery:
    """人脸特征库（注册库）。

    Parameters
    ----------
    directory : 特征库目录，默认 ``config.GALLERY_DIR``
    dim : 特征维度，ArcFace w600k_r50 为 512
    """

    CSV_FIELDS = ("student_id", "name", "class_name", "registered_at")

    def __init__(self, directory: Optional[str | Path] = None, dim: int = config.EMBED_DIM) -> None:
        self.dir = Path(directory or config.GALLERY_DIR)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.csv"
        self.dim = dim

        self._ids: List[str] = []
        self._meta: Dict[str, Dict[str, str]] = {}
        self._matrix = np.zeros((0, dim), dtype=np.float32)
        self._index = None
        self.backend = "numpy"
        self.load()

    # ------------------------------------------------------------ 持久化
    def load(self) -> None:
        """从磁盘加载全部特征与元数据。"""
        self._ids, self._meta, feats = [], {}, []

        for npy in sorted(self.dir.glob("*.npy")):
            vector = np.load(str(npy)).astype(np.float32).reshape(-1)
            if vector.size != self.dim:
                continue
            feats.append(vector / (np.linalg.norm(vector) + 1e-8))
            self._ids.append(npy.stem)

        if self.index_path.exists():
            with self.index_path.open("r", encoding="utf-8-sig", newline="") as fh:
                for row in csv.DictReader(fh):
                    sid = (row.get("student_id") or "").strip()
                    if sid:
                        self._meta[sid] = {k: (row.get(k) or "").strip() for k in self.CSV_FIELDS}

        self._matrix = np.stack(feats) if feats else np.zeros((0, self.dim), dtype=np.float32)
        self._rebuild_index()

    def save_metadata(self) -> None:
        """把元数据整体写回 index.csv（只保留仍存在特征文件的记录）。"""
        with self.index_path.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(self.CSV_FIELDS))
            writer.writeheader()
            for sid in self._ids:
                row = {"student_id": sid}
                row.update({k: self._meta.get(sid, {}).get(k, "") for k in self.CSV_FIELDS[1:]})
                writer.writerow(row)

    # ------------------------------------------------------------ 索引
    def _rebuild_index(self) -> None:
        self._index = None
        self.backend = "numpy"
        if len(self._ids) == 0:
            return
        try:  # FAISS 可用时使用精确内积索引
            import faiss

            index = faiss.IndexFlatIP(self.dim)
            index.add(self._matrix)
            self._index, self.backend = index, "faiss"
        except Exception:
            self._index = None

    @property
    def ids(self) -> List[str]:
        return list(self._ids)

    @property
    def size(self) -> int:
        return len(self._ids)

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix.copy()

    def metadata(self, student_id: str) -> Dict[str, str]:
        return dict(self._meta.get(student_id, {"student_id": student_id, "name": "", "class_name": ""}))

    def name_of(self, student_id: str) -> str:
        return self._meta.get(student_id, {}).get("name", "") or student_id

    # ------------------------------------------------------------ 增删改查
    def add(
        self,
        student_id: str,
        feature: np.ndarray,
        name: str = "",
        class_name: str = "",
        save: bool = True,
    ) -> None:
        """注册/更新一条人脸特征（同名文件覆盖）。"""
        vector = np.asarray(feature, dtype=np.float32).reshape(-1)
        if vector.size != self.dim:
            raise ValueError(f"特征维度应为 {self.dim}，实际 {vector.size}")
        vector = vector / (np.linalg.norm(vector) + 1e-8)

        if save:
            np.save(str(self.dir / f"{student_id}.npy"), vector)

        self._meta[student_id] = {
            "student_id": student_id,
            "name": name,
            "class_name": class_name,
            "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if student_id in self._ids:
            self._matrix[self._ids.index(student_id)] = vector
        else:
            self._ids.append(student_id)
            self._matrix = np.vstack([self._matrix, vector[None, :]])
        self._rebuild_index()
        if save:
            self.save_metadata()

    def remove(self, student_id: str) -> bool:
        """删除某人的人脸特征。"""
        if student_id not in self._ids:
            return False
        pos = self._ids.index(student_id)
        self._ids.pop(pos)
        self._matrix = np.delete(self._matrix, pos, axis=0)
        self._meta.pop(student_id, None)
        npy = self.dir / f"{student_id}.npy"
        if npy.exists():
            npy.unlink()
        self._rebuild_index()
        self.save_metadata()
        return True

    def search(self, feature: np.ndarray, top_k: int = 1) -> List[Tuple[str, float]]:
        """在库中检索最相似的 top_k 个人，返回 [(student_id, cosine)]。"""
        if self.size == 0:
            return []
        query = np.asarray(feature, dtype=np.float32).reshape(1, -1)
        query /= np.linalg.norm(query) + 1e-8
        k = min(top_k, self.size)

        if self._index is not None:
            scores, indices = self._index.search(query, k)
            pairs = zip(scores[0], indices[0])
        else:
            scores = (self._matrix @ query.ravel())
            indices = np.argsort(-scores)[:k]
            pairs = zip(scores[indices], indices)

        return [(self._ids[i], float(s)) for s, i in pairs if 0 <= int(i) < self.size]

    def as_dataframe(self):
        """元数据转 DataFrame（便于在 Notebook 中展示）。"""
        import pandas as pd

        rows = [self.metadata(sid) for sid in self._ids]
        return pd.DataFrame(rows, columns=list(self.CSV_FIELDS))

    def __len__(self) -> int:
        return self.size

    def __contains__(self, student_id: str) -> bool:
        return student_id in self._ids
