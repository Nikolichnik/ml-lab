"""
Watermarking utilities.
"""

import numpy as np
import cv2


def dct2(array: np.ndarray) -> np.ndarray:
    """
    2D Discrete Cosine Transform.
    """
    return cv2.dct(array.astype(np.float32))

def idct2(array: np.ndarray) -> np.ndarray:
    """
    2D Inverse Discrete Cosine Transform.
    """
    return cv2.idct(array.astype(np.float32))

def embed_watermark(
    img_gray: np.ndarray,
    strength: float = 5.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Embed a watermark into a grayscale image using DCT domain modification.

    Args:
        img_gray (numpy.ndarray): Input grayscale image (HxW) with pixel values in [0,1].
        strength (float): Watermark strength.
        seed (int): Random seed for watermark generation.

    Returns:
        tuple: Watermarked image and the watermark pattern.
    """
    # img_gray: HxW in [0,1]
    H, W = img_gray.shape
    rng = np.random.default_rng(seed)
    wm = rng.choice([-1,1], size=(H,W)).astype(np.float32)
    A = dct2(img_gray*255.0)
    A_w = A + strength * wm
    img_w = idct2(A_w)/255.0
    img_w = np.clip(img_w, 0, 1)

    return img_w, wm

def detect_watermark(
    img_gray: np.ndarray,
    wm: np.ndarray,
) -> float:
    """
    Detect a watermark in a grayscale image using DCT domain correlation.

    Args:
        img_gray (numpy.ndarray): Input grayscale image (HxW) with pixel values in [0,1].
        wm (numpy.ndarray): Watermark pattern (HxW).

    Returns:
        float: Correlation score indicating the presence of the watermark.
    """
    # simple correlation detector in DCT domain
    A = dct2(img_gray*255.0)
    score = (A * wm).mean()

    return score

def jpeg_compress(
    img_gray: np.ndarray,
    quality: int = 50,
) -> np.ndarray:
    """
    Simulate JPEG compression on a grayscale image.

    Args:
        img_gray (numpy.ndarray): Input grayscale image (HxW) with pixel values in [0,1].
        quality (int): JPEG quality factor (1-100).

    Returns:
        numpy.ndarray: Compressed and decompressed grayscale image.
    """
    enc = cv2.imencode(".jpg", (img_gray*255).astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1]
    dec = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)/255.0

    return dec
