from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    source_dir: Path
    local_data_file : Path
    unzip_dir: Path