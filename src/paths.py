import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent  # Автоматически определяет корень проекта

PATHS = {
    "train_images": PROJECT_ROOT / "data" / "train",
    "train_annotations": PROJECT_ROOT / "data" / "annotations" / "_annotations.createml_1.json",
    "test_images": PROJECT_ROOT / "data" / "test"
}