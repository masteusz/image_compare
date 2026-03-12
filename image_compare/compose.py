import math

import numpy as np
import structlog
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
from scipy.cluster.hierarchy import fcluster, linkage

log = structlog.get_logger()


def cluster_images(
    similarity_matrix: NDArray, threshold: float = 0.5
) -> list[list[int]]:
    """Cluster images based on similarity matrix using hierarchical clustering.

    Returns a list of clusters, each cluster is a list of image indices.
    Images within each cluster are ordered by their index.
    """
    n = similarity_matrix.shape[0]
    if n <= 1:
        log.info("clustering skipped", reason="single image or empty")
        return [list(range(n))]

    log.info("clustering images", n_images=n, threshold=threshold, method="average")

    distance_matrix = 1.0 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = np.clip(distance_matrix, 0, None)

    # Convert to condensed form for scipy
    condensed = []
    for i in range(n):
        for j in range(i + 1, n):
            condensed.append(distance_matrix[i, j])
    condensed = np.array(condensed)

    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=threshold, criterion="distance")

    clusters: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(idx)

    result = [clusters[k] for k in sorted(clusters)]
    log.info(
        "clustering complete",
        n_clusters=len(result),
        cluster_sizes=[len(c) for c in result],
    )
    return result


def create_grid(
    images: list[Image.Image],
    labels: list[str],
    cols: int | None = None,
) -> Image.Image:
    """Create a grid image from a list of PIL images with labels.

    All images are resized to a common size (the size of the first image).
    Labels are drawn overlaid on the bottom of each image.
    """
    if not images:
        raise ValueError("No images provided")

    n = len(images)
    if cols is None:
        cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    cell_w, cell_h = images[0].size

    log.info(
        "creating grid",
        n_images=n,
        grid=f"{rows}x{cols}",
        cell_size=f"{cell_w}x{cell_h}",
        total_size=f"{cols * cell_w}x{rows * cell_h}",
    )

    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 24)
        log.debug("font loaded", path="/usr/share/fonts/TTF/DejaVuSans.ttf")
    except OSError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
            log.debug("font loaded", path="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        except OSError:
            font = ImageFont.load_default()
            log.debug("font loaded", path="default (built-in)")

    padding = 6

    for idx, (img, label) in enumerate(zip(images, labels)):
        row, col = divmod(idx, cols)
        x = col * cell_w
        y = row * cell_h

        resized = img.resize((cell_w, cell_h), Image.LANCZOS)
        grid.paste(resized, (x, y))

        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Semi-transparent background bar behind text
        bar_y = y + cell_h - text_h - padding * 2
        overlay = Image.new("RGBA", (cell_w, text_h + padding * 2), (0, 0, 0, 160))
        grid.paste(
            Image.composite(overlay, grid.crop((x, bar_y, x + cell_w, bar_y + overlay.size[1])).convert("RGBA"), overlay),
            (x, bar_y),
        )

        text_x = x + (cell_w - text_w) // 2
        text_y = bar_y + padding
        draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)

    return grid
