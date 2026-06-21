import numpy as np
from tests.conftest import make_shell, make_tray


def test_make_shell_returns_front_and_back_points():
    front, back = make_shell(radius=20.0, thickness=6.0, n=400, seed=1)
    assert front.shape[1] == 3 and back.shape[1] == 3
    # 表は裏より平均 z が高い（厚み分だけ離れている）
    assert front[:, 2].mean() > back[:, 2].mean()


def test_make_tray_places_separated_blobs():
    pts, labels = make_tray(n_frags=3, seed=2)
    assert set(np.unique(labels)) == {0, 1, 2}
    # 各破片の重心が十分離れている
    cents = np.array([pts[labels == i].mean(axis=0) for i in range(3)])
    dists = np.linalg.norm(cents[:, None, :2] - cents[None, :, :2], axis=-1)
    np.fill_diagonal(dists, np.inf)
    assert dists.min() > 30.0
