import math
from math import gcd

import torch

from .imagefunc import (
    draw_rect,
    gaussian_blur,
    image2mask,
    log,
    mask2image,
    mask_area,
    max_inscribed_rect,
    min_bounding_rect,
    pil2tensor,
    tensor2pil,
)
from .mask_utils import MASK_THRESHOLD as DEFAULT_MASK_THRESHOLD


class CropNotFeasibleError(ValueError):
    """Raised when a crop cannot satisfy the requested geometric constraints."""


class CropByMaskV4:
    """Crop a mask region while preserving a strict target aspect ratio."""

    ASPECT_RATIOS = [
        "auto", "1:1", "4:5", "3:4", "2:3", "9:16",
        "5:4", "4:3", "3:2", "16:9",
    ]
    MASK_THRESHOLD = DEFAULT_MASK_THRESHOLD

    def __init__(self):
        self.NODE_NAME = "CropByMask V4"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "invert_mask": ("BOOLEAN", {"default": False, "tooltip": "Invert the mask before detecting the crop region."}),
                "detect": (["mask_area", "min_bounding_rect", "max_inscribed_rect"], {"tooltip": "Method used to detect the region that must be contained."}),
                "aspect_ratio": (cls.ASPECT_RATIOS, {"tooltip": "Strict output ratio. Auto selects the feasible standard ratio with the smallest final crop."}),
                # The UI calls this step; the implementation keeps the old concept/name.
                "step": (["1", "8", "16", "32", "64", "128"], {"default": "8", "tooltip": "Preferred width/height step. If impossible, V4 may lower it to the next smaller step."}),
                "padding": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1, "tooltip": "Maximum requested border padding in pixels."}),
                "auto_padding": ("BOOLEAN", {"default": False, "tooltip": "If enabled, reduce padding from the requested value when needed to keep the preferred step."}),
                "position_mode": (["auto", "center"], {"default": "auto", "tooltip": "Auto keeps the crop inside the image; center errors if the centered crop does not fit."}),
            },
            "optional": {
                "crop_box": ("BOX",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOX", "IMAGE")
    RETURN_NAMES = ("croped_image", "croped_mask", "crop_box", "box_preview")
    FUNCTION = "crop_by_mask_v4"
    CATEGORY = "😺dzNodes/LayerUtility"

    @staticmethod
    def _parse_ratio(aspect_ratio):
        try:
            width, height = (int(value) for value in aspect_ratio.split(":", 1))
        except (AttributeError, ValueError):
            raise ValueError(f"Invalid aspect ratio: {aspect_ratio}")
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid aspect ratio: {aspect_ratio}")
        divisor = gcd(width, height)
        return width // divisor, height // divisor

    @staticmethod
    def _geometric_expansion(base_box, ratio):
        """Continuous area expansion before step quantization."""
        base_width = base_box[2] - base_box[0]
        base_height = base_box[3] - base_box[1]
        ratio_width, ratio_height = ratio
        scale = max(base_width / ratio_width, base_height / ratio_height)
        raw_width = ratio_width * scale
        raw_height = ratio_height * scale
        return (raw_width * raw_height) / (base_width * base_height)

    @staticmethod
    def _detect_box(mask_image, detect):
        """Return a validated half-open (x1, y1, x2, y2) detected box."""
        try:
            if detect == "min_bounding_rect":
                x, y, width, height = min_bounding_rect(mask_image)
            elif detect == "max_inscribed_rect":
                x, y, width, height = max_inscribed_rect(mask_image)
                # The helper returns the last contour coordinate as x2/y2.
                width += 1
                height += 1
            else:
                x, y, width, height = mask_area(mask_image)
                # mask_area returns inclusive extrema, while PIL crops are half-open.
                width += 1
                height += 1
        except (IndexError, ValueError) as exc:
            raise ValueError("Unable to detect a non-empty region from the mask.") from exc
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("Unable to detect a non-empty region from the mask.")
        return int(x), int(y), int(x + width), int(y + height)

    @staticmethod
    def _step_multiplier(ratio_width, ratio_height, step):
        """Smallest multiplier making both ratio dimensions step-aligned."""
        return math.lcm(step // gcd(step, ratio_width), step // gcd(step, ratio_height))

    @classmethod
    def _make_ratio_box(cls, base_box, canvas_size, ratio, step, position_mode):
        bx1, by1, bx2, by2 = base_box
        canvas_width, canvas_height = canvas_size
        base_width = bx2 - bx1
        base_height = by2 - by1
        if base_width <= 0 or base_height <= 0:
            raise CropNotFeasibleError("The detected crop area is empty.")

        ratio_width, ratio_height = ratio
        multiplier = math.ceil(max(base_width / ratio_width, base_height / ratio_height))
        if step is not None:
            step_multiplier = cls._step_multiplier(ratio_width, ratio_height, step)
            multiplier = math.ceil(multiplier / step_multiplier) * step_multiplier
        crop_width = ratio_width * multiplier
        crop_height = ratio_height * multiplier

        center_x = (bx1 + bx2) / 2.0
        center_y = (by1 + by2) / 2.0
        desired_x1 = int(round(center_x - crop_width / 2.0))
        desired_y1 = int(round(center_y - crop_height / 2.0))

        if position_mode == "center":
            x1, y1 = desired_x1, desired_y1
            if x1 < 0 or y1 < 0 or x1 + crop_width > canvas_width or y1 + crop_height > canvas_height:
                raise CropNotFeasibleError(
                    f"Crop area {crop_width}x{crop_height} does not fit within the {canvas_width}x{canvas_height} image in center mode."
                )
        else:
            # Find positions that both fit the canvas and still contain the padded base box.
            x_low = max(0, bx2 - crop_width)
            x_high = min(bx1, canvas_width - crop_width)
            y_low = max(0, by2 - crop_height)
            y_high = min(by1, canvas_height - crop_height)
            if x_low > x_high or y_low > y_high:
                raise CropNotFeasibleError(
                    f"Crop area {crop_width}x{crop_height} cannot contain the padded mask area within the {canvas_width}x{canvas_height} image."
                )
            x1 = min(max(desired_x1, x_low), x_high)
            y1 = min(max(desired_y1, y_low), y_high)

        return int(x1), int(y1), int(x1 + crop_width), int(y1 + crop_height)

    @classmethod
    def _solve_aspect_ratio(cls, base_box, canvas_size, aspect_ratio, step, position_mode):
        if aspect_ratio != "auto":
            ratio = cls._parse_ratio(aspect_ratio)
            return ratio, cls._make_ratio_box(base_box, canvas_size, ratio, step, position_mode)

        candidates = []
        for preference, candidate_name in enumerate(cls.ASPECT_RATIOS[1:]):
            ratio = cls._parse_ratio(candidate_name)
            try:
                candidate_box = cls._make_ratio_box(base_box, canvas_size, ratio, step, position_mode)
            except CropNotFeasibleError:
                continue
            final_width = candidate_box[2] - candidate_box[0]
            final_height = candidate_box[3] - candidate_box[1]
            geometric_score = cls._geometric_expansion(base_box, ratio)
            candidates.append((
                final_width * final_height,
                geometric_score,
                preference,
                ratio,
                candidate_box,
            ))

        if not candidates:
            raise CropNotFeasibleError("No standard aspect ratio can contain the padded region within the image.")
        _, _, _, ratio, candidate_box = min(candidates, key=lambda candidate: candidate[:3])
        return ratio, candidate_box

    @classmethod
    def _solve_padding(cls, detected_box, canvas_size, requested_padding,
                       auto_padding, aspect_ratio, step, position_mode):
        """Solve padding and the crop box, optionally reducing padding to fit."""
        canvas_width, canvas_height = canvas_size

        def solve_for(padding):
            x1, y1, x2, y2 = detected_box
            base_box = (x1 - padding, y1 - padding, x2 + padding, y2 + padding)
            if base_box[0] < 0 or base_box[1] < 0 or base_box[2] > canvas_width or base_box[3] > canvas_height:
                raise CropNotFeasibleError("Padded detected region exceeds image bounds.")
            ratio, crop_box = cls._solve_aspect_ratio(
                base_box, canvas_size, aspect_ratio, step, position_mode
            )
            return padding, ratio, crop_box

        if not auto_padding:
            try:
                return solve_for(requested_padding)
            except CropNotFeasibleError as exc:
                raise CropNotFeasibleError(
                    f"Padding {requested_padding} cannot satisfy the aspect ratio, step, and image bounds."
                ) from exc

        # Feasibility is monotonic as padding grows: if a value fits, every
        # smaller value can fit. Search for the largest usable value <= request.
        try:
            return solve_for(requested_padding)
        except CropNotFeasibleError:
            pass
        low, high = 0, int(requested_padding)
        best = None
        while low <= high:
            candidate = (low + high) // 2
            try:
                result = solve_for(candidate)
            except CropNotFeasibleError:
                high = candidate - 1
            else:
                best = result
                low = candidate + 1
        if best is None:
            raise CropNotFeasibleError(
                "No valid crop exists even with auto padding reduced to 0. "
                "Reduce step, choose a different aspect ratio, or use a larger image."
            )
        return best

    @classmethod
    def _solve_step_and_padding(cls, detected_box, canvas_size, requested_padding,
                                auto_padding, aspect_ratio, requested_step,
                                position_mode):
        """Keep the requested step when possible, then reduce padding, then step."""
        requested_step = int(requested_step)
        supported_steps = [128, 64, 32, 16, 8, 1]
        step_candidates = [step for step in supported_steps if step <= requested_step]
        if not step_candidates:
            step_candidates = [1]

        last_error = None
        for step in step_candidates:
            try:
                result = cls._solve_padding(
                    detected_box, canvas_size, requested_padding, auto_padding,
                    aspect_ratio, step, position_mode
                )
            except CropNotFeasibleError as exc:
                last_error = exc
            else:
                return result[0], step, result[1], result[2]

        raise CropNotFeasibleError(
            "No valid crop exists after trying the requested step and automatic "
            "step fallbacks. Reduce padding, choose a smaller step, or use a larger image."
        ) from last_error

    def crop_by_mask_v4(
        self,
        image,
        mask,
        invert_mask,
        detect,
        aspect_ratio,
        padding,
        auto_padding,
        position_mode,
        step,
        crop_box=None,
    ):
        if image.ndim != 4 or image.shape[-1] != 3:
            raise ValueError(
                f"Expected ComfyUI RGB IMAGE tensor with shape [B,H,W,3], got {tuple(image.shape)}."
            )
        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)
        if mask.dim() != 3:
            raise ValueError(f"Expected MASK tensor with shape [B,H,W], got {tuple(mask.shape)}.")
        image_height, image_width = image.shape[1:3]
        if tuple(mask.shape[-2:]) != (image_height, image_width):
            raise ValueError(
                f"Mask size {tuple(mask.shape[-2:])} does not match image size "
                f"{(image_height, image_width)}."
            )
        if mask.shape[0] > 1:
            log("Warning: Multiple mask inputs, using the first.", message_type="warning")
            mask = torch.unsqueeze(mask[0], 0)
        if invert_mask:
            mask = 1 - mask

        mask_image = tensor2pil(torch.unsqueeze(mask, 0)).convert("L")
        mask_rgba = mask2image(mask)
        preview_image = tensor2pil(mask).convert("RGBA")
        canvas_width, canvas_height = int(image_width), int(image_height)
        if crop_box is None:
            blurred_mask = gaussian_blur(mask_rgba, 20).convert("L")
            # mask_area has always meant the extent of the original mask. Only
            # contour-based detectors use the legacy blurred input.
            if detect == "mask_area":
                x1, y1, x2, y2 = self._detect_box(mask_rgba, detect)
            else:
                # Small or thin masks can fall below imagefunc's fixed threshold
                # after blur, so fall back to the unblurred mask if necessary.
                try:
                    x1, y1, x2, y2 = self._detect_box(blurred_mask, detect)
                except (IndexError, ValueError):
                    x1, y1, x2, y2 = self._detect_box(mask_rgba, detect)
            width, height = x2 - x1, y2 - y1
            step_value = int(step)
            actual_padding, actual_step, resolved_ratio, crop_box = self._solve_step_and_padding(
                (x1, y1, x2, y2),
                (canvas_width, canvas_height),
                int(padding),
                auto_padding,
                aspect_ratio,
                step_value,
                position_mode,
            )
            if actual_step < step_value:
                log(
                    f"{self.NODE_NAME}: Step reduced from {step_value} to {actual_step}.",
                    message_type="warning",
                )
            if auto_padding and actual_padding < padding:
                log(
                    f"{self.NODE_NAME}: Auto padding reduced from {padding} to {actual_padding}.",
                    message_type="info",
                )
            ratio_text = f"{resolved_ratio[0]}:{resolved_ratio[1]}"
            log(
                f"{self.NODE_NAME}: detected={(x1, y1, x2, y2)}, "
                f"requested_padding={padding}, actual_padding={actual_padding}, "
                f"aspect_ratio={ratio_text}, step={actual_step}, "
                f"crop_size={(crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])}, "
                f"position={position_mode}",
                message_type="info",
            )
            preview_image = draw_rect(
                preview_image,
                x1,
                y1,
                width,
                height,
                line_color="#F00000",
                line_width=max(1, (width + height) // 100),
            )
        else:
            crop_box = tuple(int(value) for value in crop_box)
            if (
                len(crop_box) != 4
                or crop_box[2] <= crop_box[0]
                or crop_box[3] <= crop_box[1]
                or crop_box[0] < 0
                or crop_box[1] < 0
                or crop_box[2] > canvas_width
                or crop_box[3] > canvas_height
            ):
                raise ValueError(f"Invalid crop_box: {crop_box}")

        preview_image = draw_rect(
            preview_image,
            crop_box[0],
            crop_box[1],
            crop_box[2] - crop_box[0],
            crop_box[3] - crop_box[1],
            line_color="#00F000",
            line_width=max(1, (crop_box[2] - crop_box[0] + crop_box[3] - crop_box[1]) // 200),
        )

        ret_images = []
        ret_masks = []
        for current_image in image:
            canvas = tensor2pil(torch.unsqueeze(current_image, 0)).convert("RGB")
            ret_images.append(pil2tensor(canvas.crop(crop_box)))
            ret_masks.append(image2mask(mask_image.crop(crop_box)))

        return (
            torch.cat(ret_images, dim=0),
            torch.cat(ret_masks, dim=0),
            list(crop_box),
            pil2tensor(preview_image),
        )


NODE_CLASS_MAPPINGS = {
    "LayerUtility: CropByMask V4": CropByMaskV4,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: CropByMask V4": "LayerUtility: CropByMask V4",
}
