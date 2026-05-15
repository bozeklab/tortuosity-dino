import os
from pathlib import Path


def extract_label_from_path(path: Path) -> int:
    """Extracts label from the file path. Assumes label is part of the directory name."""
    path_name = path.parent.name
    # Example logic: extract digits from directory name as label
    label_str = [char for char in path_name if char.isdigit()]
    assert len(label_str) == 1, f"Label extraction error for {path}"

    return int(label_str[0]) - 1  # Labels should start from 0 instead of 1-4

def prepare_corn_data(path: Path, file_extension: str = ".tif", sort: bool=False, int_splitter: str=None):
    if int_splitter is None:
        int_splitter = "." if file_extension == ".tif" else "_"
    label_dirs = sorted(os.listdir(path))
    collected_files = []

    for label_dir in label_dirs:
        label_dir = path / label_dir
        tif_files = [label_dir / f for f in os.listdir(label_dir) if f.endswith(file_extension)]
        if sort:
            tif_files = sorted(tif_files, key=lambda x: int(str(x.name).split(int_splitter)[0]))
        collected_files.extend(tif_files)
    labels = [extract_label_from_path(p) for p in collected_files]
    return collected_files, labels
