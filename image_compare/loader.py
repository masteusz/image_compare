import re
from pathlib import Path

import structlog

log = structlog.get_logger()

SEQUENCE_RE = re.compile(r"^(\d+)_")


def parse_sequence_number(filename: str) -> int | None:
    """Extract the leading sequence number from a filename like '0_385457_modelname_0001.webp'."""
    match = SEQUENCE_RE.match(filename)
    if match:
        return int(match.group(1))
    log.debug("no sequence number found", filename=filename)
    return None


def find_images(root: Path, glob_pattern: str = "*.webp") -> dict[int, list[Path]]:
    """Scan all subfolders of root for image files and group by sequence number.

    Returns a dict mapping sequence number to a sorted list of image paths.
    """
    log.info("scanning for images", root=str(root), pattern=glob_pattern)
    groups: dict[int, list[Path]] = {}
    total_scanned = 0
    skipped = 0

    for path in root.rglob(glob_pattern):
        if not path.is_file():
            continue
        total_scanned += 1
        seq = parse_sequence_number(path.name)
        if seq is None:
            skipped += 1
            continue
        groups.setdefault(seq, []).append(path)

    for paths in groups.values():
        paths.sort(key=lambda p: p.parent.name)

    log.info(
        "image scan complete",
        files_scanned=total_scanned,
        files_skipped=skipped,
        sequences_found=len(groups),
        images_per_sequence={seq: len(paths) for seq, paths in sorted(groups.items())},
    )
    return groups
