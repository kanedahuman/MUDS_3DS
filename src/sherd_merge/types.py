from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Fragment:
    """切り出された 1 破片の点群。side は 'front' か 'back'。"""
    frag_id: str
    side: str
    points: np.ndarray            # (N, 3) float
    normals: Optional[np.ndarray] = None  # (N, 3) float or None

    @property
    def centroid_xy(self) -> np.ndarray:
        return self.points[:, :2].mean(axis=0)


@dataclass
class Descriptor:
    """1 破片の形状記述子。"""
    frag_id: str
    contour_xy: np.ndarray        # (M, 2) 主平面投影した外周輪郭
    area: float                   # 投影面積
    thickness_hist: np.ndarray    # (K,) 厚み分布ヒストグラム（正規化済み）
    max_dim: float
    aspect: float
    pca_axes: np.ndarray          # (3, 3) 主軸（行ベクトル）


@dataclass
class MatchPair:
    """表裏の対応 1 組。"""
    front_id: str
    back_id: str
    cost: float
    needs_review: bool = False
