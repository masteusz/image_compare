import imagehash
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from skimage.metrics import structural_similarity


def compute_ssim(img_a: NDArray, img_b: NDArray) -> float:
    """Compute structural similarity between two images (as numpy arrays).

    Images should be the same shape. Returns a value in [0, 1].
    """
    min_dim = min(img_a.shape[0], img_a.shape[1])
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    if win_size < 3:
        win_size = 3

    is_multichannel = img_a.ndim == 3
    return structural_similarity(
        img_a,
        img_b,
        win_size=win_size,
        channel_axis=2 if is_multichannel else None,
        data_range=255,
    )


def compute_histogram_correlation(img_a: NDArray, img_b: NDArray) -> float:
    """Compute color histogram correlation between two images.

    Returns a value in [-1, 1], where 1 means identical histograms.
    """
    if img_a.ndim == 2:
        channels_a = [img_a]
        channels_b = [img_b]
    else:
        channels_a = [img_a[:, :, c] for c in range(img_a.shape[2])]
        channels_b = [img_b[:, :, c] for c in range(img_b.shape[2])]

    correlations = []
    for ch_a, ch_b in zip(channels_a, channels_b):
        hist_a = np.histogram(ch_a, bins=256, range=(0, 256))[0].astype(float)
        hist_b = np.histogram(ch_b, bins=256, range=(0, 256))[0].astype(float)

        hist_a -= hist_a.mean()
        hist_b -= hist_b.mean()

        denom = np.sqrt(np.sum(hist_a**2) * np.sum(hist_b**2))
        if denom == 0:
            correlations.append(1.0)
        else:
            correlations.append(np.sum(hist_a * hist_b) / denom)

    return float(np.mean(correlations))


def compute_phash_similarity(img_a: Image.Image, img_b: Image.Image) -> float:
    """Compute perceptual hash similarity between two PIL images.

    Returns a value in [0, 1], where 1 means identical hashes.
    """
    hash_a = imagehash.phash(img_a)
    hash_b = imagehash.phash(img_b)
    max_bits = len(hash_a.hash.flatten())
    distance = hash_a - hash_b
    return 1.0 - (distance / max_bits)


def compute_pairwise_similarities(
    images: list[NDArray],
    pil_images: list[Image.Image],
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> NDArray:
    """Compute NxN combined similarity matrix.

    Weights are (ssim_weight, histogram_weight, phash_weight).
    """
    n = len(images)
    matrix = np.ones((n, n), dtype=float)

    w_ssim, w_hist, w_phash = weights

    for i in range(n):
        for j in range(i + 1, n):
            ssim = compute_ssim(images[i], images[j])
            hist = compute_histogram_correlation(images[i], images[j])
            phash = compute_phash_similarity(pil_images[i], pil_images[j])

            # Normalize histogram correlation from [-1, 1] to [0, 1]
            hist_norm = (hist + 1.0) / 2.0

            combined = w_ssim * ssim + w_hist * hist_norm + w_phash * phash
            matrix[i, j] = combined
            matrix[j, i] = combined

    return matrix
