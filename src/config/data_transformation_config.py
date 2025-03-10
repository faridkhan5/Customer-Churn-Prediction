import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataTransformationConfig:
    preprocessor_file_path: Path = os.path.join("artifacts/data_transformation", 'preprocessor.pkl')