"""Core alpha and batch invariants for RestoreCropBox V4."""

import importlib
import pathlib
import sys
import types

import numpy as np
import pytest
import torch


# Load this node as a small namespace package so tests do not import every
# LayerStyle node (or require a running ComfyUI installation).
ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE_NAME = "_layerstyle_restore_test"
if PACKAGE_NAME not in sys.modules:
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(ROOT / "py")]
    sys.modules[PACKAGE_NAME] = package
RestoreCropBoxV4 = importlib.import_module(
    f"{PACKAGE_NAME}.restore_crop_box_v4"
).RestoreCropBoxV4


@pytest.fixture
def node():
    return RestoreCropBoxV4()


def test_crop_box_normal_has_zero_edges_and_full_core(node):
    alpha = node._overlap_rect_alpha(9, 9, 2)

    assert np.all(alpha[0, :] == 0)
    assert np.all(alpha[-1, :] == 0)
    assert np.all(alpha[:, 0] == 0)
    assert np.all(alpha[:, -1] == 0)
    assert alpha[4, 4] == 1
    assert np.all((alpha >= 0) & (alpha <= 1))
    assert np.allclose(alpha, alpha[::-1, :])
    assert np.allclose(alpha, alpha[:, ::-1])
    # A corner combines both ramps and must be softer than a point on one edge.
    assert alpha[1, 1] < alpha[1, 4]


def test_crop_box_uses_mask_bbox_as_protected_region(node):
    mask = np.zeros((20, 24), dtype=np.float32)
    mask[5:13, 8:18] = 1
    alpha = node._adaptive_region_alpha(node._bbox_region(mask), 8)

    assert np.all(alpha[0, :] == 0)
    assert np.all(alpha[-1, :] == 0)
    assert np.all(alpha[:, 0] == 0)
    assert np.all(alpha[:, -1] == 0)
    # The complete bounding rectangle, including holes between mask components, stays opaque.
    assert np.all(alpha[5:13, 8:18] == 1)
    assert alpha[2, 12] < alpha[4, 12] < alpha[5, 12]


def test_crop_box_clamps_to_reachable_full_strength_core(node):
    mask = np.zeros((10, 10), dtype=np.float32)
    mask[3:7, 3:7] = 1
    alpha = node._adaptive_region_alpha(node._bbox_region(mask), 100)

    assert alpha.max() == 1
    assert np.any(alpha == 1)
    assert np.all(alpha[0, :] == 0)
    assert np.all(alpha[-1, :] == 0)
    assert np.all(alpha[:, 0] == 0)
    assert np.all(alpha[:, -1] == 0)


def test_crop_box_overlap_allows_ramps_to_overlap(node):
    normal = node._overlap_rect_alpha(10, 10, 2)
    overlap = node._overlap_rect_alpha(10, 10, 100)

    assert np.all(overlap[0, :] == 0)
    assert np.all(overlap[-1, :] == 0)
    assert np.all(overlap[:, 0] == 0)
    assert np.all(overlap[:, -1] == 0)
    assert 0 < overlap.max() < 1
    assert normal.max() == 1
    assert not np.allclose(normal, overlap)


def test_protect_mask_preserves_subject_and_fades_to_crop_edges(node):
    mask = np.zeros((30, 30), dtype=np.float32)
    mask[10:20, 10:20] = 1
    alpha = node._adaptive_region_alpha(mask >= 0.5, 20)
    outside = mask < 0.5

    assert np.all(alpha[mask >= 0.5] == 1)
    assert np.all(alpha[0, :] == 0)
    assert np.all(alpha[-1, :] == 0)
    assert np.all(alpha[:, 0] == 0)
    assert np.all(alpha[:, -1] == 0)
    assert np.any((alpha[outside] > 0) & (alpha[outside] < 1))


def test_protect_mask_touching_crop_edge_keeps_protected_pixels_opaque(node):
    mask = np.zeros((10, 10), dtype=np.float32)
    mask[0:3, 3:7] = 1
    alpha = node._adaptive_region_alpha(mask >= 0.5, 4)

    assert np.all(alpha[mask >= 0.5] == 1)
    # Protecting source pixels takes precedence when the mask touches the crop edge.
    assert np.all(alpha[0, 3:7] == 1)


def test_protect_mask_touching_one_edge_still_feathers_other_edges(node):
    mask = np.zeros((21, 21), dtype=np.float32)
    mask[0:8, 5:16] = 1
    alpha = node._adaptive_region_alpha(mask >= 0.5, 5)

    assert np.all(alpha[mask >= 0.5] == 1)
    # The touching top edge cannot transition without weakening protected pixels.
    assert np.all(alpha[0, 5:16] == 1)
    # Other sides still have available crop space and keep their feather.
    assert np.any((alpha[8:13, 8:13] > 0) & (alpha[8:13, 8:13] < 1))
    assert np.any(alpha[8:13, 8:13] > 0)


def test_protect_mask_touching_edge_does_not_leave_alpha_on_unrelated_edge_pixels(node):
    mask = np.zeros((40, 50), dtype=np.float32)
    mask[0:12, 20:30] = 1
    alpha = node._adaptive_region_alpha(mask, 12)

    assert np.all(alpha[0, :20] == 0)
    assert np.all(alpha[0, 20:30] == 1)
    assert np.all(alpha[0, 30:] == 0)
    assert np.any((alpha[10:24, 8:20] > 0) & (alpha[10:24, 8:20] < 1))


def test_crop_box_bbox_touching_edge_has_local_edge_constraints(node):
    mask = np.zeros((30, 40), dtype=np.float32)
    mask[0:10, 15:25] = 1
    protected = node._bbox_region(mask)
    alpha = node._adaptive_region_alpha(protected, 10)

    assert np.all(alpha[0, 15:25] == 1)
    assert np.all(alpha[0, :15] == 0)
    assert np.all(alpha[0, 25:] == 0)


def test_adaptive_feather_is_monotonic_with_width_outside_region(node):
    region = np.zeros((80, 80), dtype=np.uint8)
    region[25:55, 30:50] = 1
    alpha_small = node._adaptive_region_alpha(region, 8)
    alpha_large = node._adaptive_region_alpha(region, 20)

    outside = region == 0
    assert np.all(alpha_large[outside] >= alpha_small[outside])


def test_subject_envelope_fills_mask_gaps_and_uses_2d_feather(node):
    mask = np.zeros((80, 80), dtype=np.float32)
    # Two separated lobes create a gap that the convex subject envelope protects.
    mask[5:30, 30:50] = 1
    mask[45:70, 35:45] = 1
    _, envelope = node._convex_hull_region(mask)
    alpha = node._adaptive_region_alpha(envelope, 16)

    assert np.all(alpha[mask >= 0.5] == 1)
    assert alpha[38, 40] == 1  # Between the lobes, still inside the hull.
    assert np.all(alpha[0, :] == 0)
    assert np.all(alpha[-1, :] == 0)
    assert np.all(alpha[:, 0] == 0)
    assert np.all(alpha[:, -1] == 0)
    # The envelope's curved sides receive a non-binary transition.
    assert np.any((alpha[30:45, 15:30] > 0) & (alpha[30:45, 15:30] < 1))


def test_subject_envelope_keeps_touching_subject_and_feathers_free_sides(node):
    mask = np.zeros((40, 60), dtype=np.float32)
    mask[0:18, 24:36] = 1
    _, envelope = node._convex_hull_region(mask)
    alpha = node._adaptive_region_alpha(envelope, 12)

    assert np.all(alpha[mask >= 0.5] == 1)
    # Subject protection wins where the envelope itself reaches the crop edge.
    assert np.all(alpha[0, 24:36] == 1)
    # Space on the left remains available for a two-dimensional transition.
    assert np.any((alpha[5:16, 8:22] > 0) & (alpha[5:16, 8:22] < 1))


def test_composite_batch_broadcast_and_invalid_batch_contract(node):
    background = torch.zeros((1, 6, 6, 3), dtype=torch.float32)
    edits = torch.ones((3, 2, 2, 3), dtype=torch.float32)
    output, blend_mask, original_mask = node.restore_crop_box_v4(
        background, edits, [1, 1, 5, 5], "Crop Box Overlap", 0
    )

    assert output.shape == (3, 6, 6, 3)
    assert blend_mask.shape == (3, 6, 6)
    assert original_mask.shape == (3, 6, 6)
    assert torch.allclose(output, blend_mask.unsqueeze(-1).expand_as(output))

    invalid_background = torch.zeros((4, 6, 6, 3), dtype=torch.float32)
    invalid_edits = torch.ones((2, 2, 2, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match="Incompatible batch sizes"):
        node.restore_crop_box_v4(
            invalid_background, invalid_edits, [1, 1, 5, 5], "Crop Box Overlap", 0
        )
