import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from image_compare.compose import cluster_images, create_grid
from image_compare.loader import find_images
from image_compare.metrics import compute_pairwise_similarities


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
    return parser.parse_args()


def process_sequence(
    seq: int, paths: list[Path], output_dir: Path, cols: int | None
) -> None:
    pil_images: list[Image.Image] = []
    arrays: list[np.ndarray] = []
    labels: list[str] = []

    # Use the first image's size as reference
    ref_size: tuple[int, int] | None = None

    for path in paths:
        img = Image.open(path).convert("RGB")
        if ref_size is None:
            ref_size = img.size
        else:
            img = img.resize(ref_size, Image.LANCZOS)
        pil_images.append(img)
        arrays.append(np.array(img))
        labels.append(path.parent.name)

    if len(pil_images) < 2:
        print(f"  Sequence {seq}: only {len(pil_images)} image(s), skipping comparison")
        ordered_indices = list(range(len(pil_images)))
    else:
        similarity_matrix = compute_pairwise_similarities(arrays, pil_images)
        clusters = cluster_images(similarity_matrix)
        ordered_indices = [idx for cluster in clusters for idx in cluster]

        # Print stats
        upper_tri = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        print(
            f"  Sequence {seq}: {len(pil_images)} images, "
            f"similarity min={upper_tri.min():.3f} "
            f"mean={upper_tri.mean():.3f} "
            f"max={upper_tri.max():.3f}"
        )

    ordered_images = [pil_images[i] for i in ordered_indices]
    ordered_labels = [labels[i] for i in ordered_indices]

    grid = create_grid(ordered_images, ordered_labels, cols=cols)
    output_path = output_dir / f"sequence_{seq}.webp"
    grid.save(output_path, quality=90)
    print(f"  Saved: {output_path}")


def main() -> None:
    args = parse_args()
    root: Path = args.root
    output_dir: Path = args.output

    if not root.is_dir():
        print(f"Error: {root} is not a directory")
        raise SystemExit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    groups = find_images(root)
    if not groups:
        print("No image groups found.")
        raise SystemExit(0)

    sequences = sorted(groups.keys())
    if args.sequence:
        sequences = [s for s in sequences if s in args.sequence]

    print(f"Found {len(groups)} sequence(s), processing {len(sequences)}:")

    for seq in sequences:
        process_sequence(seq, groups[seq], output_dir, args.cols)

    print("Done.")


if __name__ == "__main__":
    main()
