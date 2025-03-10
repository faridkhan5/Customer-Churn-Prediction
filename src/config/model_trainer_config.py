import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ModelTrainerConfig:
    trained_model_path: Path = os.path.join("artifacts/model_trainer", 'best_tuned_model.pkl')