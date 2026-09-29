"""Restore a processed crop with selectable feathering strategies."""

import cv2
import numpy as np
import torch
from .mask_utils import MASK_THRESHOLD as DEFAULT_MASK_THRESHOLD


class RestoreCropBoxV4:
    MASK_THRESHOLD = DEFAULT_MASK_THRESHOLD
    def __init__(self):
        self.NODE_NAME = "RestoreCropBox V4"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_image": ("IMAGE",), "croped_image": ("IMAGE",), "crop_box": ("BOX",),
                "feather_mode": (["Crop Box", "Crop Box Overlap", "Protect Mask", "Subject Envelope"], {
                    "default": "Crop Box",
                    "tooltip": "Crop Box protects the mask bounding box; Overlap feathers the crop rectangle; Protect Mask protects the exact mask; Subject Envelope protects a convex hull and feathers it with crop-aware 2D distances.",
                }),
                "feather_width": ("INT", {"default": 8, "min": 0, "max": 4096, "step": 1,
                    "tooltip": "Feather width in final original-image pixels. Connect an INT to control it dynamically."}),
            },
            "optional": {"croped_mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("image", "blend_mask", "original_mask")
    FUNCTION = "restore_crop_box_v4"
    CATEGORY = "😺dzNodes/LayerUtility"

    @staticmethod
    def _smoothstep(value):
        value = np.clip(value, 0.0, 1.0)
        return value * value * (3.0 - 2.0 * value)

    @staticmethod
    def _resize_batch(batch, height, width):
        downsampling = height < batch.shape[1] or width < batch.shape[2]
        values = torch.nn.functional.interpolate(
            batch.movedim(-1, 1), size=(height, width), mode="bilinear",
            align_corners=False, antialias=downsampling,
        )
        return values.movedim(1, -1)

    @classmethod
    def _overlap_rect_alpha(cls, height, width, feather_width):
        # Overlapping edge ramps intentionally allow the whole patch to stay translucent.
        requested = float(max(0, feather_width))
        if requested <= 0:
            return np.ones((height, width), dtype=np.float32)
        vertical_width = requested
        horizontal_width = requested
        # Pixel-edge distances start at zero, so the patch's outer row/column is transparent.
        y = np.arange(height, dtype=np.float32)
        x = np.arange(width, dtype=np.float32)
        if vertical_width <= 0:
            vertical = np.ones(height, dtype=np.float32)
        else:
            vertical = np.minimum(cls._smoothstep(y / vertical_width), cls._smoothstep((height - 1 - y) / vertical_width))
        if horizontal_width <= 0:
            horizontal = np.ones(width, dtype=np.float32)
        else:
            horizontal = np.minimum(cls._smoothstep(x / horizontal_width), cls._smoothstep((width - 1 - x) / horizontal_width))
        # Multiply both axis ramps so corner feathering accumulates into a rounded transition.
        alpha = vertical[:, None] * horizontal[None, :]
        return alpha

    @classmethod
    def _bbox_region(cls, mask):
        """Return a filled binary mask bounding box as the protected geometry."""
        binary = mask >= cls.MASK_THRESHOLD
        if not np.any(binary):
            raise ValueError("Crop Box mode requires a non-empty croped_mask.")
        ys, xs = np.where(binary)
        h, w = binary.shape
        region = np.zeros((h, w), dtype=np.uint8)
        region[int(ys.min()):int(ys.max()) + 1, int(xs.min()):int(xs.max()) + 1] = 1
        return region

    @classmethod
    def _convex_hull_region(cls, mask):
        """Build a coarse subject envelope that fills gaps between mask parts."""
        binary = (mask >= cls.MASK_THRESHOLD).astype(np.uint8)
        if not np.any(binary):
            raise ValueError("Subject Envelope mode requires a non-empty croped_mask.")

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("Subject Envelope mode requires a non-empty croped_mask.")
        points = np.vstack(contours)
        hull = cv2.convexHull(points)
        envelope = np.zeros_like(binary)
        if len(hull) >= 3:
            cv2.fillConvexPoly(envelope, hull, 1)
        elif len(hull) == 2:
            cv2.line(envelope, tuple(hull[0, 0]), tuple(hull[1, 0]), 1, thickness=1)
        else:
            x, y = hull[0, 0]
            envelope[y, x] = 1

        # Preserve every thresholded mask pixel despite contour rasterization.
        envelope = np.maximum(envelope, binary)
        return binary, envelope

    @classmethod
    def _adaptive_region_alpha(cls, protected_region, feather_width):
        """Feather any protected geometry with subject and crop-edge distances."""
        region = (protected_region != 0).astype(np.uint8)
        if not np.any(region):
            raise ValueError("Protected region must be non-empty.")
        height, width = region.shape
        alpha = region.astype(np.float32)
        requested = float(max(0, feather_width))
        if requested <= 0:
            return alpha

        # Subject distance uses EDT; distance to the rectangular crop edge is analytic.
        distance_subject = cv2.distanceTransform(
            1 - region, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        y = np.arange(height, dtype=np.float32)
        x = np.arange(width, dtype=np.float32)
        distance_edge = np.minimum(
            np.minimum(y, height - 1 - y)[:, None],
            np.minimum(x, width - 1 - x)[None, :],
        )

        outside = region == 0
        available = distance_subject + distance_edge
        effective_width = np.minimum(requested, available)
        progress = np.ones_like(distance_subject, dtype=np.float32)
        np.divide(distance_subject, effective_width, out=progress, where=effective_width > 0)
        alpha[outside] = 1.0 - cls._smoothstep(progress[outside])

        # Where subject protection meets the crop edge, preserving the subject wins.
        crop_edge = np.zeros((height, width), dtype=bool)
        crop_edge[0, :] = crop_edge[-1, :] = True
        crop_edge[:, 0] = crop_edge[:, -1] = True
        alpha[crop_edge & outside] = 0.0
        alpha[region != 0] = 1.0
        return alpha

    def restore_crop_box_v4(self, background_image, croped_image, crop_box, feather_mode,
                            feather_width, croped_mask=None):
        box = tuple(int(value) for value in crop_box)
        if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"Invalid crop_box: {crop_box}")
        x1, y1, x2, y2 = box
        patch_width, patch_height = x2 - x1, y2 - y1
        if background_image.ndim != 4 or croped_image.ndim != 4:
            raise ValueError("Expected IMAGE tensors with shape [B,H,W,C].")
        if background_image.shape[-1] != croped_image.shape[-1]:
            raise ValueError("background_image and croped_image must have matching channels.")
        bg_height, bg_width = background_image.shape[1:3]
        if x1 < 0 or y1 < 0 or x2 > bg_width or y2 > bg_height:
            raise ValueError("crop_box must fit inside background_image.")
        edits = self._resize_batch(croped_image, patch_height, patch_width)
        masks = None
        if croped_mask is not None:
            if croped_mask.ndim == 2:
                croped_mask = croped_mask.unsqueeze(0)
            if croped_mask.ndim != 3:
                raise ValueError("Expected MASK tensor with shape [B,H,W].")
            masks = torch.nn.functional.interpolate(croped_mask.unsqueeze(1).float(),
                size=(patch_height, patch_width), mode="bilinear", align_corners=False,
                antialias=(patch_height < croped_mask.shape[-2] or patch_width < croped_mask.shape[-1]))[:, 0].cpu().numpy()
        batch_sizes = [len(background_image), len(edits)]
        if masks is not None:
            batch_sizes.append(len(masks))
        count = max(batch_sizes)
        if any(size not in (1, count) for size in batch_sizes):
            raise ValueError(f"Incompatible batch sizes for restore: {batch_sizes}. Each must be 1 or {count}.")
        outputs, blends, originals = [], [], []
        for index in range(count):
            bg = background_image[index if index < len(background_image) else -1].clone()
            edit = edits[index if len(edits) > 1 else 0].to(device=bg.device, dtype=bg.dtype)
            original = torch.zeros((bg_height, bg_width), dtype=torch.float32, device=bg.device)
            blend = torch.zeros_like(original)
            source = masks[index if len(masks) > 1 else 0] if masks is not None else None
            if source is not None:
                original[y1:y2, x1:x2] = torch.as_tensor(source, device=bg.device)
            if feather_mode == "Protect Mask":
                if source is None:
                    raise ValueError("Protect Mask mode requires croped_mask.")
                protected = (source >= self.MASK_THRESHOLD).astype(np.uint8)
                alpha_np = self._adaptive_region_alpha(protected, feather_width)
            elif feather_mode == "Crop Box":
                if source is None:
                    raise ValueError("Crop Box mode requires croped_mask to protect its bounding box.")
                protected = self._bbox_region(source)
                # Preserve the established zero-width Crop Box behavior: paste the full patch.
                alpha_np = (np.ones_like(protected, dtype=np.float32) if feather_width <= 0
                            else self._adaptive_region_alpha(protected, feather_width))
            elif feather_mode == "Crop Box Overlap":
                alpha_np = self._overlap_rect_alpha(patch_height, patch_width, feather_width)
            elif feather_mode == "Subject Envelope":
                if source is None:
                    raise ValueError("Subject Envelope mode requires croped_mask.")
                _, protected = self._convex_hull_region(source)
                alpha_np = self._adaptive_region_alpha(protected, feather_width)
            else:
                raise ValueError(f"Unsupported feather_mode: {feather_mode}")
            alpha = torch.as_tensor(alpha_np, dtype=bg.dtype, device=bg.device)
            blend[y1:y2, x1:x2] = alpha
            a = alpha.unsqueeze(-1)
            bg[y1:y2, x1:x2] = edit * a + bg[y1:y2, x1:x2] * (1.0 - a)
            outputs.append(bg.unsqueeze(0)); blends.append(blend.unsqueeze(0)); originals.append(original.unsqueeze(0))
        return torch.cat(outputs, 0), torch.cat(blends, 0), torch.cat(originals, 0)


NODE_CLASS_MAPPINGS = {"LayerUtility: RestoreCropBox V4": RestoreCropBoxV4}
NODE_DISPLAY_NAME_MAPPINGS = {"LayerUtility: RestoreCropBox V4": "RestoreCropBox V4"}
