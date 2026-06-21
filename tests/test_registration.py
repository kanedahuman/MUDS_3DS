import numpy as np
from sherd_merge.types import Fragment
from sherd_merge.registration import (
    extract_boundary, initial_pose_from_mirror, register_pair,
)
from tests.conftest import make_shell


def test_extract_boundary_returns_outer_ring():
    front, _ = make_shell(radius=20.0, thickness=6.0, n=800, seed=1)
    bnd = extract_boundary(front, quantile=0.15)
    r_all = np.linalg.norm(front[:, :2], axis=1)
    r_bnd = np.linalg.norm(bnd[:, :2], axis=1)
    assert r_bnd.mean() > r_all.mean()


def test_register_pair_recovers_known_overlap():
    front, back = make_shell(radius=18.0, thickness=6.0, n=900, seed=7)
    theta = np.deg2rad(8.0)
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    moved_back = back @ R.T + np.array([3.0, -2.0, 1.0])
    f = Fragment("f", "front", front)
    b = Fragment("b", "back", moved_back)
    merged, info = register_pair(f, b, max_corr_dist=3.0, max_iter=60,
                                 boundary_quantile=0.15)
    assert info["rmse"] < 1.5
    assert len(merged) >= len(front)
