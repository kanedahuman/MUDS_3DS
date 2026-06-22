import numpy as np
from sherd_merge.types import Fragment
from sherd_merge.registration import (
    extract_edge_loop, shell_separation, register_pair,
)
from tests.conftest import make_edged_shell


def _ry180(pts):
    out = pts.copy(); out[:, 0] *= -1; out[:, 2] *= -1; return out


def test_extract_edge_loop_finds_outer_ring():
    front, _, _ = make_edged_shell(radius=18.0, thickness=6.0, n=700, seed=1)
    edge = extract_edge_loop(front, k=30, gap_deg=90.0)
    assert len(edge) > 20
    # 縁点は中心から遠い（外周）
    r_all = np.linalg.norm(front[:, :2], axis=1)
    r_edge = np.linalg.norm(edge[:, :2], axis=1)
    assert r_edge.mean() > r_all.mean()


def test_shell_separation_detects_offset_vs_collapsed():
    # 厚み 6 で離した2面
    ang = np.linspace(0, 2*np.pi, 400)
    rad = 10*np.sqrt(np.linspace(0, 1, 400))
    x = rad*np.cos(ang); y = rad*np.sin(ang)
    top = np.column_stack([x, y, np.full_like(x, 3.0)])
    bot = np.column_stack([x, y, np.full_like(x, -3.0)])
    s_off = shell_separation(top, bot)
    assert s_off["sep"] > 4.0          # ~6
    # 重ねた2面
    s_col = shell_separation(top, top.copy())
    assert s_col["sep"] < 1.0


def test_register_pair_preserves_thickness_not_collapsed():
    # ギザギザ縁を持つ殻。裏は反転＋擾乱した生スキャン。
    front, back_phys, _ = make_edged_shell(radius=18.0, thickness=6.0,
                                           n=800, seed=7)
    theta = np.deg2rad(20.0)
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta),  np.cos(theta), 0],
                   [0, 0, 1]])
    back_scan = _ry180(back_phys) @ Rz.T + np.array([60.0, -25.0, 3.0])
    f = Fragment("f", "front", front)
    b = Fragment("b", "back", back_scan)
    merged, info = register_pair(f, b, max_corr_dist=3.0, max_iter=80)
    # 潰れず厚みを保つ（真の厚み 6 の半分以上）
    assert info["shell_sep"] > 3.0
    assert info["collapsed"] is False
    assert len(merged) >= len(front)
