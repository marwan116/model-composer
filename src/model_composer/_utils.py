"""Generic utilities for model composer."""
import shutil
from contextlib import suppress
from pathlib import Path
from typing import cast, Optional


def make_path(path_str: str) -> Path:
    """Make a path object from a string."""
    path: Optional[Path] = None

    with suppress(ImportError):
        from cloudpathlib import CloudPath
        from cloudpathlib.exceptions import InvalidPrefixError

        try:
            path = cast(Path, CloudPath(path_str))  # type: ignore
        except InvalidPrefixError:
            path = Path(path_str)

    if path is None:
        path = Path(path_str)

    return path


def copy_path(path: Path, destination: str) -> None:
    """Copy a path object."""
    path_types = [Path]

    with suppress(ImportError):
        from cloudpathlib import CloudPath

        path_types.append(CloudPath)  # type: ignore

    if len(path_types) == 1:
        if path.is_dir():
            shutil.copytree(str(path), destination)
        elif path.is_file():
            shutil.copy(path, destination)
        else:
            raise ValueError(f"Path {path} is not a file or directory.")
    else:
        if isinstance(path, path_types[0]):
            if path.is_dir():
                shutil.copytree(path, destination)
            elif path.is_file():
                shutil.copy(path, destination)
            else:
                raise ValueError(f"Path {path} is not a file or directory.")
        elif isinstance(path, path_types[1]):  # type: ignore
            if path.is_dir():
                path.copytree(destination)
            elif path.is_file():
                path.copy(destination)
            else:
                raise ValueError(f"Path {path} is not a file or directory.")
    return None
