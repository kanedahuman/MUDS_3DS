import numpy as np
from sherd_merge.descriptors import compute_descriptor, pca_axes
from sherd_merge.types import Fragment
from tests.conftest import make_shell


def test_pca_axes_orthonormal():
    pts = np.random.rand(200, 3) * 10
    axes = pca_axes(pts)
    np.testing.assert_allclose(axes @ axes.T, np.eye(3), atol=1e-6)


def test_compute_descriptor_area_and_thickness():
    front, _ = make_shell(radius=20.0, thickness=6.0, n=600, seed=3)
    frag = Fragment(frag_id="f", side="front", points=front)
    desc = compute_descriptor(frag, thickness_bins=16)
    assert 800 < desc.area < 2000
    assert desc.thickness_hist.shape == (16,)
    np.testing.assert_allclose(desc.thickness_hist.sum(), 1.0, atol=1e-6)
    assert desc.contour_xy.shape[1] == 2
    assert desc.max_dim > 0
