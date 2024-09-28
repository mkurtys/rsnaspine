from pathlib import Path
from typing import Union, Optional

def to_path_or_none(path: Union[str, Path, None]) -> Optional[Path]:
    return Path(path) if path is not None else None
