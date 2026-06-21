import numpy as np
from scipy.optimize import linear_sum_assignment
from .types import Fragment, Descriptor, MatchPair
from .descriptors import compute_descriptor


def mirror_lr(points: np.ndarray) -> np.ndarray:
    """左右反転（x 符号反転）。裏トレイを表と比較可能にする。"""
    out = points.copy()
    out[:, 0] *= -1
    return out


def _hist_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).sum() / 2.0)  # L1/2（正規化ヒスト同士で 0..1）


def pair_cost(df: Descriptor, db: Descriptor) -> float:
    """形状コスト：面積差・厚み分布差・アスペクト差の重み付き和（小さいほど類似）。"""
    area_term = abs(df.area - db.area) / max(df.area, db.area, 1e-9)
    thick_term = _hist_distance(df.thickness_hist, db.thickness_hist)
    aspect_term = abs(df.aspect - db.aspect) / max(df.aspect, db.aspect, 1e-9)
    return 0.5 * area_term + 0.3 * thick_term + 0.2 * aspect_term


def match_front_back(fronts: list[Fragment], backs: list[Fragment],
                     position_prune_radius: float = 20.0,
                     cost_threshold: float = 0.25) -> list[MatchPair]:
    """位置事前情報で候補を枝刈りし、形状コストでハンガリアン割当を行う。

    裏破片は生スキャン空間のまま渡す（左右反転は不要）。取得時に破片を
    同じ位置で左右反転するため、裏は表と同じ XY に共在し（形状のみミラー像）。
    - 位置枝刈り：生の重心同士を比較（共在しているので正しく近接する）。
    - 形状コスト：面積・厚み分布・アスペクトはいずれ左右反転で不変なので、
      反転せずにそのまま比較してよい。
    3D 統合のための実際のミラーは M5（registration）が担う。
    """
    fd = [compute_descriptor(f) for f in fronts]
    bd = [compute_descriptor(b) for b in backs]
    f_xy = np.array([f.centroid_xy for f in fronts])
    b_xy = np.array([b.centroid_xy for b in backs])

    n, m = len(fronts), len(backs)
    BIG = 1e6
    cost = np.full((n, m), BIG)
    for i in range(n):
        for j in range(m):
            dist = np.linalg.norm(f_xy[i] - b_xy[j])
            if dist > position_prune_radius:
                continue
            shape = pair_cost(fd[i], bd[j])
            pos_term = 0.1 * (dist / position_prune_radius)
            cost[i, j] = shape + pos_term

    row, col = linear_sum_assignment(cost)
    pairs = []
    for r, c in zip(row, col):
        if cost[r, c] >= BIG:
            continue
        needs_review = cost[r, c] > cost_threshold
        pairs.append(MatchPair(front_id=fronts[r].frag_id,
                              back_id=backs[c].frag_id,
                              cost=float(cost[r, c]),
                              needs_review=needs_review))
    return pairs
