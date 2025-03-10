import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionConfig:
    raw_data_path: Path = os.path.join("artifacts/data_ingestion", 'data.csv')
    train_data_path: Path = os.path.join("artifacts/data_ingestion", 'train_data.csv')
    test_data_path: Path = os.path.join("artifacts/data_ingestion", 'test_data.csv')