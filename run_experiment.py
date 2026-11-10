"""
HEASNet experiment entry point.

The supplied notebook was a sequential, single-cell experiment with shared globals.
The ordered components are executed in one shared namespace to preserve that behavior.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
COMPONENTS = [
    "00_runtime_config.py", "01_data.py", "02_model.py", "03_training.py",
    "04_metrics.py", "05_visualization.py", "06_explainability.py",
    "07_data_audit.py", "08_complexity.py", "09_evaluation.py",
    "10_dataset_runner.py", "11_extra_experiments.py", "12_cross_dataset.py",
    "13_paper_tables.py", "14_export.py", "15_main.py",
]
for component in COMPONENTS:
    component_path = PROJECT_ROOT / "heasnet" / component
    print(f"\\n[HEASNet] Loading {component}")
    source = component_path.read_text(encoding="utf-8")
    exec(compile(source, str(component_path), "exec"), globals(), globals())
