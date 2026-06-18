import os
from pathlib import Path


def expand_path(value: str | Path) -> Path:
    text = str(value)
    if "${AINA_RUN_ROOT}" in text and "AINA_RUN_ROOT" not in os.environ:
        os.environ["AINA_RUN_ROOT"] = "/home/data/aina-runs"
    return Path(os.path.expandvars(text)).expanduser()
