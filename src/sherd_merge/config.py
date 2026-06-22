from dataclasses import dataclass
import yaml


@dataclass
class Config:
    input_front: str
    input_back: str
    output_dir: str = "output"
    work_dir: str = "work"
    # M0
    target_unit_mm: float = 1.0       # 入力 1 単位 = この mm 数（1.0 なら既に mm）
    # M1
    plane_dist_threshold: float = 1.0  # mm
    denoise_neighbors: int = 20
    denoise_std_ratio: float = 2.0
    voxel_size: float = 0.0             # mm（0 でダウンサンプルなし）
    # M2
    cluster_eps: float = 3.0           # mm
    cluster_min_points: int = 50
    # M3
    thickness_bins: int = 16
    # M4
    position_prune_radius: float = 20.0  # mm（左右反転後の重心移動許容半径）
    match_cost_threshold: float = 0.25   # これ超で要確認
    # M5（縁ベース登録）
    icp_max_corr_dist: float = 2.0       # mm
    icp_max_iter: int = 50
    edge_k: int = 30                     # 縁検出の近傍数
    edge_gap_deg: float = 90.0           # 縁判定の角度ギャップ閾値（度）
    shell_min_sep: float = 0.8           # この未満の厚み分離は潰れ＝要確認
    # M6
    mesh_for_measurement: bool = True    # 計測用にポアソンメッシュも出すか


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    inp = raw.get("input", {})
    seg = raw.get("segment", {})
    pre = raw.get("preprocess", {})
    mat = raw.get("match", {})
    reg = raw.get("registration", {})
    out = raw.get("output", {})
    return Config(
        input_front=inp["front"],
        input_back=inp["back"],
        output_dir=raw.get("output_dir", "output"),
        work_dir=raw.get("work_dir", "work"),
        target_unit_mm=raw.get("target_unit_mm", 1.0),
        plane_dist_threshold=pre.get("plane_dist_threshold", 1.0),
        denoise_neighbors=pre.get("denoise_neighbors", 20),
        denoise_std_ratio=pre.get("denoise_std_ratio", 2.0),
        voxel_size=pre.get("voxel_size", 0.0),
        cluster_eps=seg.get("cluster_eps", 3.0),
        cluster_min_points=seg.get("cluster_min_points", 50),
        thickness_bins=raw.get("descriptors", {}).get("thickness_bins", 16),
        position_prune_radius=mat.get("position_prune_radius", 20.0),
        match_cost_threshold=mat.get("cost_threshold", 0.25),
        icp_max_corr_dist=reg.get("icp_max_corr_dist", 2.0),
        icp_max_iter=reg.get("icp_max_iter", 50),
        edge_k=reg.get("edge_k", 30),
        edge_gap_deg=reg.get("edge_gap_deg", 90.0),
        shell_min_sep=reg.get("shell_min_sep", 0.8),
        mesh_for_measurement=out.get("mesh_for_measurement", True),
    )
