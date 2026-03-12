import re
from pathlib import Path

SEQUENCE_RE = re.compile(r"^(\d+)_")


def parse_sequence_number(filename: str) -> int | None:
    """Extract the leading sequence number from a filename like '0_385457_modelname_0001.webp'."""
    match = SEQUENCE_RE.match(filename)
    if match:
        return int(match.group(1))
    return None


def find_images(root: Path, glob_pattern: str = "*.webp") -> dict[int, list[Path]]:
    """Scan all subfolders of root for image files and group by sequence number.

    Returns a dict mapping sequence number to a sorted list of image paths.
    """
    groups: dict[int, list[Path]] = {}

    for path in root.rglob(glob_pattern):
        if not path.is_file():
            continue
        seq = parse_sequence_number(path.name)
        if seq is None:
            continue
        groups.setdefault(seq, []).append(path)

    for paths in groups.values():
        paths.sort(key=lambda p: p.parent.name)

    return groups
