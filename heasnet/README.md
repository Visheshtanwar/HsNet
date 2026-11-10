# HEASNet implementation modules

These files are ordered components of the original sequential notebook. They share one execution namespace when launched through the repository-root `run_experiment.py` entry point.

- `00_runtime_config.py`: imports, hyperparameters, paths, seed, device and logging.
- `01_data.py`: data discovery, image validation, split generation and loaders.
- `02_model.py`: HEASNet and its attention/refinement blocks.
- `03_training.py`: training, optimizer and checkpoint logic.
- `04_metrics.py`: prediction and metric computation.
- `05_visualization.py`: training and evaluation figures.
- `06_explainability.py`: Grad-CAM, attention maps and faithfulness.
- `07_data_audit.py`: near-duplicate audit.
- `08_complexity.py`: parameter/FLOP and speed profiling.
- `09_evaluation.py`: split-level evaluation and reporting.
- `10_dataset_runner.py`: per-dataset orchestration.
- `11_extra_experiments.py`: ablations, baselines and optional sweeps.
- `12_cross_dataset.py`: cross-dataset comparison.
- `13_paper_tables.py`: manuscript-ready tables.
- `14_export.py`: result ZIP export.
- `15_main.py`: main execution function.
