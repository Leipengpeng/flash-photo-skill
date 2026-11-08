"""Precision-aware RGB I/O for the native fusion path."""
from pathlib import Path
import struct
import tempfile
import zlib

import cv2
import numpy as np
from PIL import Image, ImageCms


_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_SRGB_CHROMATICITY = (.3127, .3290, .6400, .3300, .3000, .6000, .1500, .0600)
_SRGB_CICP = b"\x01\x0d\x00\x01"
_SRGB_ICC = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _icc_tags(profile):
    if len(profile) < 132 or profile[16:24] != b"RGB XYZ " or profile[36:40] != b"acsp":
        raise ValueError("Expected an RGB-to-XYZ ICC profile")
    count, = struct.unpack_from(">I", profile, 128)
    table_end = 132 + count * 12
    if table_end > len(profile):
        raise ValueError("Invalid ICC tag table")
    tags = {}
    for name, start, length in struct.iter_unpack(">4sII", profile[132:table_end]):
        if start < table_end or start + length > len(profile):
            raise ValueError("Invalid ICC tag bounds")
        tags[name] = profile[start:start + length]
    return tags


def _canonical_srgb(profile):
    """Identify the built-in sRGB transform despite differing profile metadata."""
    try:
        tags = _icc_tags(profile)
        standard = _icc_tags(_SRGB_ICC)
        conversion = (b"rXYZ", b"gXYZ", b"bXYZ", b"rTRC", b"gTRC", b"bTRC")
        return all(tags[key] == standard[key] for key in conversion) and not any(
            key[:3] in (b"A2B", b"B2A", b"D2B", b"B2D") for key in tags
        )
    except (KeyError, ValueError, struct.error):
        return False


def _convert_icc(pixels, profile):
    if _canonical_srgb(profile):
        return pixels, "16-bit sRGB ICC; native codes retained"
    colorspace = profile[16:20]
    if colorspace == b"RGB ":
        data, mode = pixels, "rgb"
    elif colorspace == b"GRAY" and np.array_equal(pixels[:, :, 0], pixels[:, :, 1]) and np.array_equal(
        pixels[:, :, 0], pixels[:, :, 2]
    ):
        data, mode = np.ascontiguousarray(pixels[:, :, 0]), "gray"
    else:
        raise ValueError("Export the 16-bit image as RGB with an sRGB ICC profile")
    try:
        from imagecodecs import CMS, cms_transform
    except ImportError as exc:
        raise ValueError("16-bit ICC conversion needs imagecodecs; install it or export 16-bit sRGB") from exc
    try:
        converted = cms_transform(data, profile, _SRGB_ICC, colorspace=mode, outcolorspace="rgb",
                                  outdtype="uint16", intent=CMS.INTENT.RELATIVE_COLORIMETRIC)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("Unable to convert the embedded ICC profile at 16 bits; export 16-bit sRGB") from exc
    if converted.dtype != np.uint16 or converted.shape != pixels.shape:
        raise ValueError("ICC conversion did not preserve 16-bit RGB")
    return converted, "ICC converted to sRGB at 16 bits"


def _png_header(path):
    with path.open("rb") as file:
        header = file.read(29)
    if len(header) != 29 or header[:8] != _PNG_MAGIC or header[12:16] != b"IHDR":
        raise ValueError("Invalid PNG header")
    return header[24], header[25]


def _png_cicp(path):
    # Pillow versions without cICP support can still open and downlabel such files.
    with path.open("rb") as file:
        file.seek(8)
        while (header := file.read(8)):
            if len(header) != 8:
                break
            length, name = struct.unpack(">I4s", header)
            if name in (b"IDAT", b"IEND"):
                break
            if name == b"cICP":
                if length != 4:
                    raise ValueError("Invalid PNG cICP chunk")
                return file.read(4)
            file.seek(length + 4, 1)
    return None


def _orient(pixels, orientation):
    if orientation == 2:
        pixels = pixels[:, ::-1]
    elif orientation == 3:
        pixels = pixels[::-1, ::-1]
    elif orientation == 4:
        pixels = pixels[::-1]
    elif orientation == 5:
        pixels = pixels.swapaxes(0, 1)
    elif orientation == 6:
        pixels = np.rot90(pixels, k=3, axes=(0, 1))
    elif orientation == 7:
        pixels = pixels.swapaxes(0, 1)[::-1, ::-1]
    elif orientation == 8:
        pixels = np.rot90(pixels, axes=(0, 1))
    return np.ascontiguousarray(pixels, dtype=np.uint16)


def _png_pixels(path, image, color_type):
    if getattr(image, "n_frames", 1) != 1:
        raise ValueError("Use a single-frame 16-bit PNG for native fusion")
    if "transparency" in image.info:
        raise ValueError("16-bit PNG uses a transparency key; flatten it before native fusion")
    # OpenCV's unchanged decoder preserves uint16. Decode bytes so Unicode paths work on Windows.
    pixels = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if pixels is None or pixels.dtype != np.uint16:
        raise ValueError("Failed to read the 16-bit PNG at native precision")
    if color_type in (4, 6):
        if pixels.ndim != 3 or pixels.shape[2] != 4 or np.any(pixels[:, :, 3] != 65535):
            raise ValueError("Native fusion requires an opaque 16-bit PNG")
        pixels = pixels[:, :, :3]
    if pixels.ndim == 3 and pixels.shape[2] == 3:
        return pixels[:, :, ::-1]
    if pixels.ndim == 2 and color_type == 0:
        return np.repeat(pixels[:, :, None], 3, axis=2)
    raise ValueError("Unsupported 16-bit PNG channels")


def _tiff_pixels(path):
    try:
        import tifffile
    except ImportError as exc:
        raise ValueError("16-bit TIFF decoding requires tifffile; install requirements-native.txt") from exc
    with tifffile.TiffFile(path) as source:
        if len(source.pages) != 1:
            raise ValueError("Use a single-page 16-bit TIFF for native fusion")
        page = source.pages[0]
        if page.dtype != np.uint16 or page.bitspersample != 16:
            raise ValueError("Native fusion supports unsigned 16-bit TIFF samples")
        if page.photometric.name not in ("RGB", "MINISBLACK", "MINISWHITE"):
            raise ValueError("Export the 16-bit TIFF as sRGB RGB or grayscale")
        try:
            pixels = page.asarray()
        except (ValueError, NotImplementedError) as exc:
            raise ValueError("This 16-bit TIFF encoding could not be decoded; install imagecodecs or save as 16-bit sRGB PNG or uncompressed TIFF") from exc
        if page.axes == "SYX":
            pixels = np.moveaxis(pixels, 0, -1)
        elif page.axes not in ("YX", "YXS"):
            raise ValueError("Unsupported 16-bit TIFF channel layout")
        channels = 3 if page.photometric.name == "RGB" else 1
        if pixels.ndim == 2:
            pixels = pixels[:, :, None]
        if pixels.shape[2] == channels + 1:
            if tuple(page.extrasamples) not in ((1,), (2,)) or np.any(pixels[:, :, -1] != 65535):
                raise ValueError("Native fusion requires an opaque 16-bit TIFF")
            pixels = pixels[:, :, :channels]
        if pixels.shape[2] != channels:
            raise ValueError("Unsupported 16-bit TIFF channels")
        if channels == 1:
            if page.photometric.name == "MINISWHITE":
                pixels = 65535 - pixels
            pixels = np.repeat(pixels, 3, axis=2)
        return pixels


def _normalize_color(path, image, pixels):
    profile = image.info.get("icc_profile")
    if image.format == "PNG":
        cicp = _png_cicp(path)
        if cicp is not None and cicp != _SRGB_CICP:
            raise ValueError("Convert this 16-bit PNG to sRGB at 16 bits before native fusion")
        declared_srgb = "srgb" in image.info or cicp == _SRGB_CICP
        if not profile and not declared_srgb:
            gamma = image.info.get("gamma")
            chromaticity = image.info.get("chromaticity")
            if ((gamma is not None and abs(gamma - .45455) > .00005)
                    or (chromaticity is not None and not np.allclose(
                        chromaticity, _SRGB_CHROMATICITY, atol=.00005, rtol=0))):
                raise ValueError("Convert this 16-bit PNG to sRGB at 16 bits before native fusion")
    else:
        declared_srgb = False
        if not profile and any(tag in image.tag_v2 for tag in (301, 318, 319)):
            raise ValueError("Export this TIFF with an embedded sRGB profile at 16 bits")
    if profile:
        return _convert_icc(pixels, profile)
    if declared_srgb:
        return pixels, "16-bit PNG declared sRGB"
    return pixels, "16-bit unprofiled RGB assumed sRGB"


def load_high_bit_rgb(path, image, orientation):
    """Return oriented 16-bit sRGB, or None for the existing 8-bit loader."""
    path = Path(path)
    if image.format == "PNG":
        bits, color_type = _png_header(path)
        if bits < 16:
            return None
        pixels = _png_pixels(path, image, color_type)
    elif image.format == "TIFF":
        bits = image.tag_v2.get(258, (8,))
        if max((bits,) if isinstance(bits, int) else bits) <= 8:
            return None
        pixels = _tiff_pixels(path)
    elif image.mode in ("I", "F") or image.mode.startswith("I;16"):
        raise ValueError("Export high-bit inputs as unsigned 16-bit sRGB PNG or TIFF")
    else:
        return None
    stored_size = (image.height, image.width)
    if image.format == "TIFF":
        # Pillow may already expose display dimensions for rotated TIFF pages.
        stored_size = (image.tag_v2.get(257, image.height), image.tag_v2.get(256, image.width))
    if pixels.shape[:2] != stored_size:
        raise ValueError("Decoded 16-bit dimensions differ from the image header")
    return _normalize_color(path, image, _orient(pixels, orientation))


def _png_chunk(kind, payload):
    crc = zlib.crc32(payload, zlib.crc32(kind))
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def save_rgb_png(path, codes, srgb_icc):
    """Save RGB8 with Pillow or RGB16 with PNG lossless encoding and an sRGB profile."""
    if codes.dtype == np.uint8:
        Image.fromarray(codes).save(path, format="PNG", icc_profile=srgb_icc, compress_level=3)
        return
    if codes.dtype != np.uint16 or codes.ndim != 3 or codes.shape[2] != 3:
        raise ValueError("Expected an 8- or 16-bit RGB image")
    success, encoded = cv2.imencode(".png", codes[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not success or tuple(encoded[24:26]) != (16, 2):
        raise ValueError("Failed to encode a 16-bit RGB PNG")
    profile = _png_chunk(b"iCCP", b"sRGB\x00\x00" + zlib.compress(srgb_icc))
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".png", delete=False) as file:
        temporary = Path(file.name)
    try:
        with temporary.open("wb") as file:
            file.write(memoryview(encoded)[:33])  # signature and IHDR
            file.write(profile)
            file.write(memoryview(encoded)[33:])
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


