import numpy as np
from sherd_merge.types import Fragment
from sherd_merge.descriptors import compute_descriptor
from sherd_merge.matching import (
    mirror_lr, pair_cost, match_front_back,
)
from tests.conftest import make_shell


def _frag(points, fid, side):
    return Fragment(frag_id=fid, side=side, points=points)


def test_mirror_lr_flips_x_sign():
    pts = np.array([[1.0, 2.0, 3.0]])
    out = mirror_lr(pts)
    np.testing.assert_allclose(out, [[-1.0, 2.0, 3.0]])


def test_pair_cost_low_for_same_shape():
    front, back = make_shell(radius=18.0, thickness=6.0, n=500, seed=5)
    df = compute_descriptor(_frag(front, "f", "front"))
    db = compute_descriptor(_frag(mirror_lr(back), "b", "back"))
    same = pair_cost(df, db)
    other_front, _ = make_shell(radius=30.0, thickness=6.0, n=500, seed=6)
    do = compute_descriptor(_frag(other_front, "o", "front"))
    diff = pair_cost(df, do)
    assert same < diff


def test_match_front_back_recovers_known_pairs():
    fronts, backs = [], []
    radii = [12.0, 20.0, 28.0]
    positions = [(-40, 0), (0, 0), (40, 0)]
    for i, (r, (px, py)) in enumerate(zip(radii, positions)):
        f, b = make_shell(radius=r, thickness=6.0, n=400, seed=10 + i)
        f[:, 0] += px; f[:, 1] += py
        b = mirror_lr(b)
        b[:, 0] += px + 2.0; b[:, 1] += py - 1.0
        fronts.append(_frag(f, f"front_{i}", "front"))
        backs.append(_frag(b, f"back_{i}", "back"))
    pairs = match_front_back(fronts, backs, position_prune_radius=20.0,
                             cost_threshold=0.5)
    mapping = {p.front_id: p.back_id for p in pairs}
    assert mapping["front_0"] == "back_0"
    assert mapping["front_1"] == "back_1"
    assert mapping["front_2"] == "back_2"


def test_match_front_back_fallback_without_position():
    f, b = make_shell(radius=15.0, thickness=6.0, n=400, seed=20)
    bb = mirror_lr(b)
    bb[:, 0] += 200.0
    fr = [_frag(f, "front_0", "front")]
    bk = [_frag(bb, "back_0", "back")]
    pairs = match_front_back(fr, bk, position_prune_radius=np.inf,
                             cost_threshold=0.5)
    assert len(pairs) == 1
    assert pairs[0].front_id == "front_0"
    assert pairs[0].back_id == "back_0"
