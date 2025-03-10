import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PredictPipelineConfig:
    preprocessor_path: Path = os.path.join("artifacts/data_transformation", 'preprocessor.pkl')
    model_path: Path = os.path.join("artifacts/model_trainer", 'customer_churn_model.pkl')