import argparse
from pathlib import Path

import numpy as np
import structlog
from PIL import Image

from image_compare import configure_logging
from image_compare.compose import cluster_images, create_grid
from image_compare.loader import find_images
from image_compare.metrics import compute_pairwise_similarities

log = structlog.get_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare images across model folders by sequence number."
    )
    parser.add_argument("root", type=Path, help="Root directory containing model folders")
    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="Output directory for grid images"
    )
    parser.add_argument(
        "-s",
        "--sequence",
        type=int,
        nargs="+",
        help="Process only specific sequence number(s)",
    )
    parser.add_argument("--cols", type=int, help="Force number of grid columns")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose (DEBUG) logging"
    )
    return parser.parse_args()


def process_sequence(
    seq: int, paths: list[Path], output_dir: Path, cols: int | None
) -> None:
    seq_log = log.bind(sequence=seq)
    seq_log.info("processing sequence", n_images=len(paths))

    pil_images: list[Image.Image] = []
    arrays: list[np.ndarray] = []
    labels: list[str] = []

    # Use the first image's size as reference
    ref_size: tuple[int, int] | None = None

    for path in paths:
        img = Image.open(path).convert("RGB")
        if ref_size is None:
            ref_size = img.size
            seq_log.debug("reference size set", size=ref_size, source=path.name)
        else:
            original_size = img.size
            img = img.resize(ref_size, Image.LANCZOS)
            seq_log.debug(
                "image resized",
                file=path.name,
                original=original_size,
                target=ref_size,
            )
        pil_images.append(img)
        arrays.append(np.array(img))
        labels.append(path.parent.name)
        seq_log.debug("image loaded", file=path.name, folder=path.parent.name)

    if len(pil_images) < 2:
        seq_log.warning("skipping comparison", reason="too few images", count=len(pil_images))
        ordered_indices = list(range(len(pil_images)))
    else:
        similarity_matrix = compute_pairwise_similarities(arrays, pil_images)
        clusters = cluster_images(similarity_matrix)
        ordered_indices = [idx for cluster in clusters for idx in cluster]

        upper_tri = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        seq_log.info(
            "similarity stats",
            n_images=len(pil_images),
            min=f"{upper_tri.min():.3f}",
            mean=f"{upper_tri.mean():.3f}",
            max=f"{upper_tri.max():.3f}",
        )

    ordered_images = [pil_images[i] for i in ordered_indices]
    ordered_labels = [labels[i] for i in ordered_indices]

    grid = create_grid(ordered_images, ordered_labels, cols=cols)
    output_path = output_dir / f"sequence_{seq}.webp"
    grid.save(output_path, quality=90)
    seq_log.info("grid saved", path=str(output_path))


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)

    root: Path = args.root
    output_dir: Path = args.output

    log.debug(
        "parsed arguments",
        root=str(root),
        output=str(output_dir),
        sequences=args.sequence,
        cols=args.cols,
        verbose=args.verbose,
    )

    if not root.is_dir():
        log.error("root is not a directory", root=str(root))
        raise SystemExit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    log.debug("output directory ready", path=str(output_dir))

    groups = find_images(root)
    if not groups:
        log.warning("no image groups found")
        raise SystemExit(0)

    sequences = sorted(groups.keys())
    if args.sequence:
        sequences = [s for s in sequences if s in args.sequence]

    log.info(
        "starting processing",
        total_sequences=len(groups),
        processing=len(sequences),
        sequence_numbers=sequences,
    )

    for seq in sequences:
        process_sequence(seq, groups[seq], output_dir, args.cols)

    log.info("done")


if __name__ == "__main__":
    main()
