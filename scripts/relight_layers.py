#!/usr/bin/env python3
"""Composite an inferred RGB lighting field onto the native photograph.

The model supplies a DATA image, not a replacement photograph: RGB code 128 is
zero EV, and (code - 128) / 127 * ev_range is the per-channel log2 gain. The
default add-only mode discards negative EV so a new lamp does not erase existing
ambient light. The source pixels are never warped or replaced with model pixels.

Dependencies: numpy, Pillow, scipy. Examples:
  python relight_layers.py prepare --source original.jpg --out reference
  python relight_layers.py composite --source original.jpg --field gain.png --out result
  python relight_layers.py composite --source original.jpg --field key.png --field rim.png --field-strength 1 --field-strength 0.6 --out two_lamps
  python relight_layers.py composite --source original.jpg --field result/relight_arrays.npz --field-strength 0.7 --out weaker
  python relight_layers.py composite --source original.jpg --field diffuse.png --additive-field highlights.png --additive-scale 0.2 --out flash

Use --save-arrays for the native-size float32 EV field and linear-sRGB delta.
--strength multiplies EV (legacy exposure control); --field-strength weights
each lamp's linear added contribution. Multiple lamps add gain-1, never EV.
Saved NPZ EV is before highlight rolloff and already includes saved coverage and
strength. It is reused without another filtering pass; do not reapply its mask.
Black-zero RGB additive maps supply absolute linear-sRGB light independently of
the source albedo: code 255 adds --additive-scale radiance before rolloff. Saved
NPZ additive_linear is the separately retained pre-rolloff addition, so even an
originally black pixel can be relit and reused without inventing an EV ratio.
The RGBA overlay uses normal alpha in ENCODED sRGB; linear-light blend settings
in an editor will differ. This is an approximate, single-view lighting effect,
not measured radiometry, an intrinsic decomposition, or a RAW processing engine.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageCms, ImageOps
from scipy.ndimage import gaussian_filter, uniform_filter

VERSION = "2.2.0"
SRGB_PROFILE = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
SRGB_ICC = SRGB_PROFILE.tobytes()
RATIO_TOLERANCE = 0.01


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode, orient, and color-manage an opaque photographic input once."""
    path = Path(path)
    with Image.open(path) as opened:
        opened.load()
        stored_size = opened.size
        original_mode = opened.mode
        orientation = int(opened.getexif().get(274, 1))
        profile_bytes = opened.info.get("icc_profile")
        image = ImageOps.exif_transpose(opened)
        if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
            if image.convert("RGBA").getchannel("A").getextrema() != (255, 255):
                raise ValueError("The source must be an opaque photograph; transparent source pixels are unsupported.")
            image = image.convert("RGB")
        elif image.mode == "P":
            image = image.convert("RGB")
        if profile_bytes:
            try:
                profile = ImageCms.ImageCmsProfile(io.BytesIO(profile_bytes))
                profile_name = ImageCms.getProfileDescription(profile).strip()
                image = ImageCms.profileToProfile(image, profile, SRGB_PROFILE, outputMode="RGB", renderingIntent=0)
            except (OSError, ValueError, ImageCms.PyCMSError) as exc:
                raise ValueError("Embedded ICC profile could not be converted to sRGB; refusing an unlabelled color change.") from exc
            color_action = "embedded ICC converted to sRGB using LittleCMS, perceptual rendering intent"
        else:
            if image.mode not in ("RGB", "RGBA", "L", "LA", "P"):
                raise ValueError(f"Unprofiled source mode {image.mode} is not supported; provide a color-managed sRGB image.")
            image = image.convert("RGB")
            profile_name = None
            color_action = "no embedded ICC: decoded RGB assumed sRGB"
        pixels = np.asarray(image, dtype=np.uint8).copy()
    return pixels, {
        "sha256": file_hash(path),
        "file_name": path.name,
        "stored_size": list(stored_size),
        "display_size": [pixels.shape[1], pixels.shape[0]],
        "exif_orientation": orientation,
        "original_mode": original_mode,
        "embedded_icc": bool(profile_bytes),
        "source_profile_name": profile_name,
        "color_action": color_action,
        "working_color": "linear sRGB (float32)",
        "output_encoding": "8-bit sRGB PNG with embedded sRGB profile; no copied EXIF/GPS",
    }


def load_data_image(path: str | Path, mode: str) -> np.ndarray:
    """Load a code-value data map; do not apply its embedded color profile."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if (image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info) and image.convert("RGBA").getchannel("A").getextrema() != (255, 255):
            raise ValueError("Data maps must be opaque; use a separate coverage mask, not implicit alpha.")
        return np.asarray(image.convert(mode), dtype=np.uint8).copy()


def ratio_check(size: tuple[int, int], reference: tuple[int, int], label: str) -> float:
    error = abs((size[0] / size[1]) / (reference[0] / reference[1]) - 1)
    if error > RATIO_TOLERANCE:
        raise ValueError(f"{label} aspect ratio differs by {error:.2%}; expected the same complete FOV (no automatic crop/warp).")
    return error


def resize_float(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear resample a scalar float data plane without color conversion."""
    if (array.shape[1], array.shape[0]) == size:
        return array.astype(np.float32, copy=True)
    return np.asarray(Image.fromarray(array.astype(np.float32), mode="F").resize(size, Image.Resampling.BILINEAR), dtype=np.float32).copy()


def srgb_to_linear(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32) / np.float32(255)
    return np.where(array <= 0.04045, array / 12.92, ((array + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(array: np.ndarray) -> np.ndarray:
    array = np.clip(array, 0, 1)
    encoded = np.where(array <= 0.0031308, array * 12.92, 1.055 * np.power(array, 1 / 2.4) - 0.055)
    return np.clip(np.rint(encoded * 255), 0, 255).astype(np.uint8)


def luminance_srgb(pixels: np.ndarray) -> np.ndarray:
    # Encoded luminance is intentional: a stable low-frequency edge guide, not
    # a radiometric brightness estimate. Fine photo texture never enters EV.
    return np.einsum("ijk,k->ij", pixels.astype(np.float32) / 255, np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))


def guided_coefficients(guide: np.ndarray, field: np.ndarray, radius: int, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    window = 2 * radius + 1
    mean_i = uniform_filter(guide, size=window, mode="reflect")
    variance = uniform_filter(guide * guide, size=window, mode="reflect") - mean_i * mean_i
    a = np.empty_like(field, dtype=np.float32)
    b = np.empty_like(field, dtype=np.float32)
    for channel in range(3):
        p = field[:, :, channel]
        mean_p = uniform_filter(p, size=window, mode="reflect")
        covariance = uniform_filter(guide * p, size=window, mode="reflect") - mean_i * mean_p
        coefficient = covariance / (np.maximum(variance, 0) + epsilon)
        offset = mean_p - coefficient * mean_i
        a[:, :, channel] = uniform_filter(coefficient, size=window, mode="reflect")
        b[:, :, channel] = uniform_filter(offset, size=window, mode="reflect")
    return a, b


def soft_rolloff(original: np.ndarray, target: np.ndarray, knee: float = 0.8) -> np.ndarray:
    """Compress only positive additions, using one factor for all RGB channels.

    Ambient pixels and negative deltas are not globally tone-mapped. Above the
    knee the positive excursion approaches remaining white headroom smoothly.
    This reduces clipping without changing the chromaticity of the added light.
    """
    delta = target - original
    negative = np.minimum(delta, 0)
    positive = np.maximum(delta, 0)
    base = np.maximum(original + negative, 0)
    base_peak = np.max(base, axis=2)
    target_peak = np.max(base + positive, axis=2)
    increase = target_peak - base_peak
    start = np.maximum(knee - base_peak, 0)
    room = np.maximum(1 - np.maximum(knee, base_peak), 0)
    tail = np.maximum(increase - start, 0)
    # A rational shoulder keeps useful headroom even at a strong gain; an
    # exponential shoulder rounds to exactly 1.0 too early in float32.
    compressed_tail = np.zeros_like(tail)
    np.divide(room * tail, room + tail, out=compressed_tail, where=room + tail > 0)
    allowed_increase = np.minimum(increase, start + compressed_tail)
    factor = np.ones_like(increase)
    np.divide(allowed_increase, increase, out=factor, where=increase > 1e-8)
    return np.clip(base + positive * factor[:, :, None], 0, 1)


def make_overlay(original: np.ndarray, result: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Find the minimum feasible normal alpha and its corresponding sRGB color."""
    base = original.astype(np.float32) / 255
    target = result.astype(np.float32) / 255
    difference = target - base
    requirement = np.zeros_like(base)
    np.divide(difference, 1 - base, out=requirement, where=difference > 0)
    negative_requirement = np.zeros_like(base)
    np.divide(-difference, base, out=negative_requirement, where=difference < 0)
    minimum_alpha = np.maximum(requirement, negative_requirement).max(axis=2)
    alpha_code = np.clip(np.ceil(minimum_alpha * 255), 0, 255).astype(np.uint8)
    alpha = alpha_code.astype(np.float32) / 255
    color = np.zeros_like(base)
    numerator = target - base * (1 - alpha[:, :, None])
    np.divide(numerator, alpha[:, :, None], out=color, where=alpha[:, :, None] > 0)
    color_code = np.clip(np.rint(color * 255), 0, 255).astype(np.uint8)
    overlay = np.dstack([color_code, alpha_code])
    reconstructed = np.clip(np.rint(color_code.astype(np.float32) * alpha[:, :, None] + original * (1 - alpha[:, :, None])), 0, 255).astype(np.uint8)
    error = np.abs(reconstructed.astype(np.int16) - result.astype(np.int16))
    return overlay, {
        "blend": "normal alpha in encoded sRGB; opacity 100%",
        "native_size": [original.shape[1], original.shape[0]],
        "reconstruction_max_code_error": int(error.max()),
        "reconstruction_mean_code_error": float(error.mean()),
        "transparent_fraction": float(np.mean(alpha_code == 0)),
        "alpha_max": int(alpha_code.max()),
    }


def neutral_specular_shoulder(adjusted: np.ndarray, target: np.ndarray, addition: np.ndarray) -> np.ndarray:
    """Let strong near-white additive peaks approach neutral display white.

    This is an artistic display shoulder, not a BRDF estimate. The earlier
    ambient-preserving shoulder alone lets a red skin channel limit every
    added channel, leaving even white specular peaks orange. Only near-neutral
    absolute additions entering the bright shoulder receive this treatment;
    colored rims, gain-only light, and untouched photo pixels are excluded.
    """
    high = np.max(addition, axis=2)
    low = np.min(addition, axis=2)
    ratio = np.zeros_like(low)
    np.divide(low, high, out=ratio, where=high > 1e-8)
    neutrality = np.clip((ratio - .8) / .2, 0, 1)
    shoulder = np.clip((np.max(target, axis=2) - .8) / .6, 0, 1)
    amount = neutrality * np.clip(low / .4, 0, 1) * shoulder
    peak = np.max(adjusted, axis=2)
    return adjusted + amount[:, :, None] * (peak[:, :, None] - adjusted)


def save_rgb(pixels: np.ndarray, path: Path) -> None:
    Image.fromarray(pixels, mode="RGB").save(path, icc_profile=SRGB_ICC)


def preview_size(size: tuple[int, int], maximum: int = 1600) -> tuple[int, int]:
    scale = min(1, maximum / max(size))
    return max(1, round(size[0] * scale)), max(1, round(size[1] * scale))


def prepare(source: str | Path, out: str | Path, max_edge: int = 1600) -> dict[str, Any]:
    if max_edge < 16:
        raise ValueError("max-edge must be at least 16 pixels")
    pixels, source_info = load_source(source)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if Path(source).resolve() == (out / "prepared.png").resolve():
        raise ValueError("The reference output must not overwrite the source file")
    image = Image.fromarray(pixels, mode="RGB")
    size = preview_size(image.size, max_edge)
    image.resize(size, Image.Resampling.LANCZOS).save(out / "prepared.png", icc_profile=SRGB_ICC)
    report = {"version": VERSION, "operation": "prepare", "source": source_info, "reference_size": list(size), "crop": "none", "source_unchanged": file_hash(Path(source)) == source_info["sha256"]}
    (out / "prepare.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def load_ev_field(path: str | Path, original: np.ndarray, source_hash: str, *, ev_range: float, smooth: float, edge_radius: int, epsilon: float, mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Prepare one EV field. Saved NPZ fields are already filtered native data."""
    path = Path(path)
    height, width = original.shape[:2]
    native_size = (width, height)
    info = {"file_name": path.name, "sha256": file_hash(path)}
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as saved:
            if "ev" not in saved:
                raise ValueError("Reusable NPZ field must contain the float32 'ev' array")
            ev_full = np.asarray(saved["ev"], dtype=np.float32).copy()
            if ev_full.shape != (height, width, 3):
                raise ValueError("Saved NPZ field must match this photograph's exact native display dimensions")
            if not np.all(np.isfinite(ev_full)) or np.max(np.abs(ev_full)) > 32:
                raise ValueError("Saved NPZ EV values must be finite and in the supported -32..32 range")
            stored_hash = str(saved["source_sha256"].item()) if "source_sha256" in saved else None
            if stored_hash and stored_hash != source_hash:
                raise ValueError("Saved NPZ belongs to a different original source; refusing to reuse its lighting field")
        if mode == "add-only":
            np.maximum(ev_full, 0, out=ev_full)
        info.update({"size": list(native_size), "relative_aspect_error": 0, "encoding": "native float32 EV from NPZ; already includes its saved coverage/strength, before highlight rolloff", "processing": "no repeated smoothing or guided filtering", "source_hash_verified": bool(stored_hash)})
        return ev_full, info
    field_codes = load_data_image(path, "RGB")
    map_height, map_width = field_codes.shape[:2]
    map_size = (map_width, map_height)
    field_ratio_error = ratio_check(map_size, native_size, "Lighting field")
    ev_low = (field_codes.astype(np.float32) - 128) / 127 * ev_range
    # Filter EV before enlargement; never blur or denoise the photograph itself.
    if smooth:
        ev_low = gaussian_filter(ev_low, sigma=(smooth, smooth, 0), mode="reflect")
    guide_native = luminance_srgb(original)
    guide_native = gaussian_filter(guide_native, sigma=1.25 * max(width / map_width, height / map_height), mode="reflect")
    guide_low = resize_float(guide_native, map_size)
    if edge_radius:
        coefficients, offsets = guided_coefficients(guide_low, ev_low, edge_radius, epsilon)
    else:
        coefficients = np.zeros_like(ev_low)
        offsets = ev_low
        guide_native.fill(0)
    ev_full = np.empty((height, width, 3), dtype=np.float32)
    for channel in range(3):
        coefficient = resize_float(coefficients[:, :, channel], native_size)
        offset = resize_float(offsets[:, :, channel], native_size)
        ev_full[:, :, channel] = coefficient * guide_native + offset
    minimum_ev = 0 if mode == "add-only" else -128 / 127 * ev_range
    np.clip(ev_full, minimum_ev, ev_range, out=ev_full)
    info.update({"size": list(map_size), "relative_aspect_error": field_ratio_error, "encoding": "EV[channel] = (uint8_code[channel] - 128) / 127 * ev_range; ICC ignored for data maps", "processing": "smooth EV and original-aware guided enlargement"})
    return ev_full, info


def load_saved_additive(path: str | Path, native_size: tuple[int, int]) -> np.ndarray | None:
    """Load the separate absolute-light component from a reusable NPZ.

    load_ev_field performs the shared source hash and geometry checks first.
    Old NPZ files simply have no additive component.
    """
    if Path(path).suffix.lower() != ".npz":
        return None
    with np.load(path, allow_pickle=False) as saved:
        if "additive_linear" not in saved:
            return None
        addition = np.asarray(saved["additive_linear"], dtype=np.float32).copy()
    if addition.shape != (native_size[1], native_size[0], 3):
        raise ValueError("Saved NPZ additive_linear must match the native display dimensions")
    if not np.all(np.isfinite(addition)) or np.any(addition < 0):
        raise ValueError("Saved NPZ additive_linear must be finite and nonnegative")
    return addition


def load_additive_field(path: str | Path, native_size: tuple[int, int], *, smooth: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Read black-zero linear data, smoothing only the new-light map itself."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        raise ValueError("Use --field to reuse a combined NPZ; --additive-field expects a black-zero RGB image")
    codes = load_data_image(path, "RGB")
    size = (codes.shape[1], codes.shape[0])
    ratio_error = ratio_check(size, native_size, "Additive lighting field")
    light_low = codes.astype(np.float32) / 255
    if smooth:
        light_low = gaussian_filter(light_low, sigma=(smooth, smooth, 0), mode="reflect")
    # Compact highlights should not pass through the broad diffuse guided
    # filter. Only the light data is bilinearly enlarged; coverage protects
    # the photo boundary independently.
    addition = np.stack([resize_float(light_low[:, :, channel], native_size) for channel in range(3)], axis=2)
    info = {"file_name": path.name, "sha256": file_hash(path), "size": list(size), "relative_aspect_error": ratio_error,
            "encoding": "black-zero linear RGB data: code / 255 times additive_scale; ICC ignored",
            "processing": "independent light-map smoothing then bilinear enlargement; no diffuse guided filtering"}
    return addition, info


def composite(source: str | Path, field_path: str | Path | list[str | Path] | tuple[str | Path, ...] | None, out: str | Path, *, mask_path: str | Path | None = None, strength: float = 1, ev_range: float = 2, smooth: float = 1.5, edge_radius: int = 8, epsilon: float = 0.01, mode: str = "add-only", save_arrays: bool = False, field_strengths: list[float] | tuple[float, ...] | None = None, additive_paths: list[str | Path] | tuple[str | Path, ...] | None = None, additive_strengths: list[float] | tuple[float, ...] | None = None, additive_scale: float = 0.2, additive_smooth: float = 0.8, additive_texture: float = 0) -> dict[str, Any]:
    values = [strength, ev_range, smooth, epsilon, additive_scale, additive_smooth, additive_texture]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("All numerical parameters must be finite")
    if not 0 <= strength <= 2 or not 0 < ev_range <= 8 or not 0 <= smooth <= 64 or not 0 <= edge_radius <= 128 or epsilon <= 0:
        raise ValueError("Expected strength 0..2, EV range (0,8], smooth 0..64, edge radius 0..128, epsilon > 0")
    if not 0 <= additive_scale <= 4 or not 0 <= additive_smooth <= 64:
        raise ValueError("Expected additive-scale 0..4 and additive-smooth 0..64")
    if not 0 <= additive_texture <= 1:
        raise ValueError("Expected additive-texture in 0..1")
    if mode not in ("add-only", "signed"):
        raise ValueError("mode must be add-only or signed")
    field_paths = list(field_path) if isinstance(field_path, (list, tuple)) else [field_path] if field_path is not None else []
    additive_paths = list(additive_paths) if additive_paths is not None else []
    if not field_paths and not additive_paths:
        raise ValueError("At least one EV field or additive lighting field is required")
    field_weights = list(field_strengths) if field_strengths is not None else [1.0] * len(field_paths)
    if len(field_weights) != len(field_paths) or any(not math.isfinite(weight) or not 0 <= weight <= 4 for weight in field_weights):
        raise ValueError("Provide one field-strength in 0..4 for each lighting field (or omit all for 1.0)")
    additive_weights = list(additive_strengths) if additive_strengths is not None else [1.0] * len(additive_paths)
    if len(additive_weights) != len(additive_paths) or any(not math.isfinite(weight) or not 0 <= weight <= 4 for weight in additive_weights):
        raise ValueError("Provide one additive-strength in 0..4 for each additive field (or omit all for 1.0)")
    original, source_info = load_source(source)
    height, width = original.shape[:2]
    native_size = (width, height)
    mask_info = None
    if mask_path:
        coverage_codes = load_data_image(mask_path, "L")
        mask_size = (coverage_codes.shape[1], coverage_codes.shape[0])
        mask_ratio_error = ratio_check(mask_size, native_size, "Coverage mask")
        coverage = resize_float(coverage_codes.astype(np.float32) / 255, native_size)
        np.clip(coverage, 0, 1, out=coverage)
        mask_info = {"file_name": Path(mask_path).name, "sha256": file_hash(Path(mask_path)), "size": list(mask_size), "relative_aspect_error": mask_ratio_error}
    else:
        coverage = np.ones((height, width), dtype=np.float32)
    field_infos = []
    additive_infos = []
    additive_full = None
    single_unweighted = len(field_paths) == 1 and field_weights[0] == 1
    combined_increment = None if single_unweighted else np.zeros((height, width, 3), dtype=np.float32)
    for current_path, weight in zip(field_paths, field_weights):
        current_ev, info = load_ev_field(current_path, original, source_info["sha256"], ev_range=ev_range, smooth=smooth, edge_radius=edge_radius, epsilon=epsilon, mode=mode)
        current_ev *= coverage[:, :, None] * strength
        info["linear_contribution_weight"] = weight
        saved_additive = load_saved_additive(current_path, native_size)
        info["includes_saved_additive_linear"] = saved_additive is not None
        if saved_additive is not None:
            saved_additive *= coverage[:, :, None] * strength * weight
            if additive_full is None:
                additive_full = saved_additive
            else:
                additive_full += saved_additive
        field_infos.append(info)
        if single_unweighted:
            ev_full = current_ev
        else:
            # Lamps add radiance. Adding EV would multiply their gains and
            # count the same base illumination more than once.
            for start in range(0, height, 192):
                stop = min(start + 192, height)
                combined_increment[start:stop] += weight * np.expm1(current_ev[start:stop] * np.float32(math.log(2)))
            del current_ev
    if not single_unweighted:
        ev_full = np.log2(np.maximum(1 + combined_increment, np.float32(1e-8))).astype(np.float32)
        del combined_increment
    for current_path, weight in zip(additive_paths, additive_weights):
        addition, info = load_additive_field(current_path, native_size, smooth=additive_smooth)
        if additive_texture:
            # A bounded heuristic keeps original fine bright/dark structure
            # visible in a strong artistic light pass. It does not estimate
            # albedo/normals, invent detail, or modify the source photograph.
            guide = luminance_srgb(original)
            low = gaussian_filter(guide, sigma=max(width, height) / 250, mode="reflect")
            detail = np.clip(((guide + .01) / (low + .01)) ** 2, .1, 2.5)
            addition *= (1 - additive_texture + additive_texture * detail)[:, :, None]
            del guide, low, detail
        info["source_texture_modulation"] = additive_texture
        addition *= coverage[:, :, None] * strength * weight * additive_scale
        info["linear_contribution_weight"] = weight
        info["scale_linear"] = additive_scale
        additive_infos.append(info)
        if additive_full is None:
            additive_full = addition
        else:
            additive_full += addition
    final = np.empty_like(original)
    if additive_full is not None and not np.all(np.isfinite(additive_full)):
        raise ValueError("Combined additive light exceeds finite float32 storage")
    delta_full = np.empty_like(ev_full) if save_arrays else None
    # Bounded row batches avoid several simultaneous full-image float RGB copies.
    for start in range(0, height, 192):
        stop = min(start + 192, height)
        native = original[start:stop]
        ev = ev_full[start:stop]
        linear = srgb_to_linear(native)
        target = linear * np.exp2(ev)
        if additive_full is not None:
            target += additive_full[start:stop]
        adjusted = soft_rolloff(linear, target)
        if additive_full is not None:
            adjusted = neutral_specular_shoulder(adjusted, target, additive_full[start:stop])
        encoded = linear_to_srgb(adjusted)
        unchanged = np.all(ev == 0, axis=2)
        if additive_full is not None:
            unchanged &= np.all(additive_full[start:stop] == 0, axis=2)
        encoded[unchanged] = native[unchanged]
        final[start:stop] = encoded
        if delta_full is not None:
            delta_full[start:stop] = adjusted - linear
            delta_full[start:stop][unchanged] = 0
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    output_names = ["final.png", "effect_overlay.png", "field_preview.png", "coverage_preview.png", "report.json", "relight_arrays.npz"]
    if additive_full is not None:
        output_names.append("additive_preview.png")
    inputs = [Path(path).resolve() for path in [source, *field_paths, *additive_paths, mask_path] if path is not None]
    if any((out / name).resolve() in inputs for name in output_names):
        raise ValueError("Output files must not overwrite source, lighting field, or mask inputs")
    save_rgb(final, out / "final.png")
    # Construct the normal overlay in batches too: a 12 MP source should not
    # require several extra full-size floating point RGB arrays for this step.
    overlay_full = np.empty((height, width, 4), dtype=np.uint8)
    overlay_max_error = 0
    overlay_error_sum = 0.0
    for start in range(0, height, 192):
        stop = min(start + 192, height)
        block, block_report = make_overlay(original[start:stop], final[start:stop])
        overlay_full[start:stop] = block
        overlay_max_error = max(overlay_max_error, block_report["reconstruction_max_code_error"])
        overlay_error_sum += block_report["reconstruction_mean_code_error"] * (stop - start)
    Image.fromarray(overlay_full, mode="RGBA").save(out / "effect_overlay.png", icc_profile=SRGB_ICC)
    small_size = preview_size(native_size)
    small_ev = np.stack([resize_float(ev_full[:, :, channel], small_size) for channel in range(3)], axis=2)
    # Preview encoding describes the actual applied EV, including strength and
    # coverage. Its display colors are NOT a normal photographic overlay.
    small_codes = np.clip(np.rint(128 + 127 * small_ev / ev_range), 0, 255).astype(np.uint8)
    Image.fromarray(small_codes, mode="RGB").save(out / "field_preview.png")
    small_mask = np.clip(np.rint(resize_float(coverage, small_size) * 255), 0, 255).astype(np.uint8)
    Image.fromarray(small_mask, mode="L").save(out / "coverage_preview.png")
    if additive_full is not None:
        small_additive = np.stack([resize_float(additive_full[:, :, channel], small_size) for channel in range(3)], axis=2)
        # Preview uses linear data, normalized to a reference 0.2 radiance;
        # values above that display white. Float arrays retain the full data.
        preview_codes = np.clip(np.rint(small_additive / 0.2 * 255), 0, 255).astype(np.uint8)
        Image.fromarray(preview_codes, mode="RGB").save(out / "additive_preview.png")
    if save_arrays:
        arrays = {"ev": ev_full, "delta_linear": delta_full, "source_sha256": np.asarray(source_info["sha256"]),
                  "field_semantics": np.asarray("ev: diffuse log2 gain; optional additive_linear: absolute positive linear RGB addition; both pre-rolloff, already include saved coverage, strengths and weights")}
        if additive_full is not None:
            arrays["additive_linear"] = additive_full
        np.savez_compressed(out / "relight_arrays.npz", **arrays)
    unchanged_pixels = np.all(ev_full == 0, axis=2)
    if additive_full is not None:
        unchanged_pixels &= np.all(additive_full == 0, axis=2)
    outside_mask = coverage == 0
    added_only = bool(np.all(final >= original)) if mode == "add-only" else None
    report = {
        "version": VERSION,
        "operation": "composite",
        "source": source_info,
        "field": field_infos[0] if len(field_infos) == 1 else None,
        "fields": field_infos,
        "additive_fields": additive_infos,
        "mask": mask_info,
        "parameters": {"strength": strength, "strength_semantics": "EV multiplier for diffuse; linear multiplier for additive radiance", "field_strengths": field_weights, "field_strength_semantics": "linear weight of each added gain-1 contribution and any saved absolute addition from that NPZ", "ev_range": ev_range, "smooth_sigma_map_pixels": smooth, "edge_radius_map_pixels": edge_radius, "guide_preblur_sigma_map_pixels": 1.25, "guided_epsilon": epsilon, "mode": mode, "highlight_knee_linear": 0.8, "additive_strengths": additive_weights, "additive_scale_linear": additive_scale, "additive_smooth_sigma_map_pixels": additive_smooth, "additive_texture": additive_texture},
        "composition": "diffuse_gain = 1 + sum(field_strength_i * (exp2(EV_i * strength * coverage) - 1)); target = original_linear * max(diffuse_gain, 1e-8) + sum(absolute_additive_light); apply one highlight rolloff after all sources; no geometric warp, crop, denoise, or generated RGB texture substitution",
        "native_size": list(native_size),
        "applied_ev_min": float(ev_full.min()),
        "applied_ev_max": float(ev_full.max()),
        "additive_linear_max": float(additive_full.max()) if additive_full is not None else 0.0,
        "unchanged_pixel_fraction": float(np.mean(unchanged_pixels)),
        "unchanged_pixel_count": int(np.count_nonzero(unchanged_pixels)),
        "zero_coverage_pixel_count": int(np.count_nonzero(outside_mask)),
        "zero_EV_pixels_exact": bool(np.array_equal(original[unchanged_pixels], final[unchanged_pixels])),
        "zero_EV_pixels_exact_scope": "pixels with zero diffuse EV AND zero absolute additive contribution",
        "mask_zero_pixels_exact": bool(np.array_equal(original[outside_mask], final[outside_mask])),
        "add_only_preserves_each_original_channel": added_only,
        "source_unchanged": file_hash(Path(source)) == source_info["sha256"],
        "overlay": {"blend": "normal alpha in encoded sRGB; opacity 100%", "reconstruction_max_code_error": overlay_max_error, "reconstruction_mean_code_error": overlay_error_sum / height, "transparent_fraction": float(np.mean(overlay_full[:, :, 3] == 0)), "alpha_max": int(overlay_full[:, :, 3].max())},
        "files": [name for name in output_names if name != "relight_arrays.npz" or save_arrays],
        "limitations": [
            "Inference maps may contain misplaced light or false edge structure; inspect before compositing.",
            "Native pixel dimensions and photo geometry are preserved; lighting and quantization change affected pixel values.",
            "No metric depth, measured light transport, physical calibration, or recovery of clipped/missing source detail.",
            "Final and overlay are 8-bit sRGB; color-managed source decoding may differ from raw stored RGB codes.",
            "The normal overlay is specific to this exact source and sRGB blending, not a source-independent light asset.",
            "Combined NPZ adjusts the saved combined effect; retain separate source fields for independent lamp control. Saved EV and additive_linear already include coverage/strength; do not accidentally apply the old mask twice.",
            "Absolute additive maps are artist-directed positive light data, not measured specular radiance; placement and reflection geometry need visual review.",
            "Optional additive texture modulation uses bounded original luminance contrast, not measured reflectance; saved NPZ already includes it and is not modulated again.",
            "Near-neutral absolute specular peaks enter an artistic display-white shoulder; this is not measured sensor clipping or a new surface texture.",
        ],
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    reference = subparsers.add_parser("prepare", help="Make an oriented, color-managed reference without cropping")
    reference.add_argument("--source", required=True, type=Path)
    reference.add_argument("--out", required=True, type=Path)
    reference.add_argument("--max-edge", type=int, default=1600)
    render = subparsers.add_parser("composite", help="Apply an RGB EV field to the native source")
    render.add_argument("--source", required=True, type=Path)
    render.add_argument("--field", type=Path, action="append", help="Repeat for independent lamps; PNG RGB code field or reusable native EV/additive NPZ")
    render.add_argument("--field-strength", type=float, action="append", help="Repeat once per field: linear added-light weight in 0..4; omitted means all 1")
    render.add_argument("--additive-field", type=Path, action="append", help="Black-zero opaque RGB linear-light data image; repeat for separate highlights or rims")
    render.add_argument("--additive-strength", type=float, action="append", help="Repeat once per additive field: linear added-radiance weight in 0..4; omitted means all 1")
    render.add_argument("--additive-scale", type=float, default=0.2, help="Absolute linear-sRGB radiance represented by code 255, in 0..4")
    render.add_argument("--additive-smooth", type=float, default=0.8, help="Independent additive-map smoothing sigma, in map pixels")
    render.add_argument("--additive-texture", type=float, default=0, help="Optional original-luminance detail modulation in 0..1; applies only to new additive images, not saved NPZ")
    render.add_argument("--out", required=True, type=Path)
    render.add_argument("--mask", type=Path)
    render.add_argument("--strength", type=float, default=1)
    render.add_argument("--ev-range", type=float, default=2)
    render.add_argument("--smooth", type=float, default=1.5)
    render.add_argument("--edge-radius", type=int, default=8)
    render.add_argument("--epsilon", type=float, default=0.01)
    render.add_argument("--mode", choices=("add-only", "signed"), default="add-only")
    render.add_argument("--save-arrays", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            report = prepare(args.source, args.out, args.max_edge)
        else:
            report = composite(args.source, args.field, args.out, mask_path=args.mask, strength=args.strength, ev_range=args.ev_range, smooth=args.smooth, edge_radius=args.edge_radius, epsilon=args.epsilon, mode=args.mode, save_arrays=args.save_arrays, field_strengths=args.field_strength, additive_paths=args.additive_field, additive_strengths=args.additive_strength, additive_scale=args.additive_scale, additive_smooth=args.additive_smooth, additive_texture=args.additive_texture)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps({"operation": args.command, "output_directory": str(args.out.resolve()), "native_size": report.get("native_size", report.get("source", {}).get("display_size")), "source_unchanged": report["source_unchanged"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
