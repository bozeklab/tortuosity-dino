from pathlib import Path
from typing import Any
import json
import yaml


def read_json(file_path: Path):
    """Reads a JSON file and returns the data."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def write_json(data: dict[Any, Any], file_path: Path):
    """Writes data to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_yaml(file_path: Path):
    """Reads a YAML file and returns the data."""
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    return data
