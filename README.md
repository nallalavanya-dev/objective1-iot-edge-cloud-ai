# Objective 1 – AI-Powered IoT–Edge–Cloud Task Decision Lab

This Streamlit application predicts end-to-end IoT task completion time at six candidate execution locations (Local Device, Edge Servers 1–4, and Cloud Server), compares nine regression algorithms, and recommends the lowest predicted-time node that meets the task deadline.

## What's included

- `app.py` – animated Streamlit dashboard; loads pre-trained models and does **not** train models at application startup.
- `trained_models/regression_models_bundle.joblib` – nine pre-trained regression pipelines, held-out benchmark scores, and validation predictions.
- `objective1_iot_multinode_training_dataset.csv` – 72,000 task–node records (12,000 task episodes × six candidate nodes).
- `objective1_task_decision_dataset.csv` – task-level decision dataset.
- `train_models_offline.py` – reproducible offline training script used to create the model bundle.
- `requirements.txt` and `runtime.txt` – pinned runtime dependencies and Python 3.12.

## Deploy on Streamlit Community Cloud

1. Upload the **contents** of this project folder to the root of your GitHub repository (not just the ZIP file).
2. Ensure `app.py`, `requirements.txt`, `runtime.txt`, both CSV files, and the `trained_models` folder are in the repository root/tree as shown below.
3. In Streamlit Community Cloud choose repository, branch `main`, and main file path `app.py`.
4. Deploy. The app loads the pre-trained bundle; it does not fit nine models on every cloud startup.

```text
repository-root/
├── app.py
├── requirements.txt
├── runtime.txt
├── README.md
├── objective1_iot_multinode_training_dataset.csv
├── objective1_task_decision_dataset.csv
├── train_models_offline.py
└── trained_models/
    └── regression_models_bundle.joblib
```

## Retrain models locally

Use Python 3.12 and install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
python train_models_offline.py
```

This recreates `trained_models/regression_models_bundle.joblib` using the train/test partition already provided in the CSV. Upload the updated bundle to GitHub after retraining.

## Models compared

Linear Regression, Ridge Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, Histogram Gradient Boosting, Voting Ensemble, and Stacking Ensemble.

The best model is selected by lowest RMSE on the provided held-out test partition. The app then predicts completion time for all six candidate locations and selects the lowest predicted-time node that meets the user-entered deadline. If none meets the deadline, it reports the least-delay option without claiming it is feasible.

## Research integrity and limitations

- The supplied dataset is a **synthetic benchmark**, not measured ESP32/router/edge/cloud telemetry.
- The task animation is a visual simulation; this prototype does not send packets or execute jobs on remote nodes.
- Delay component charts are analytical approximations; the regression model predicts total completion time.
- Use independent NS-3/SimPy traces and hardware measurements before presenting results as real-world validation.
- Joblib/scikit-learn model files are version-sensitive. Keep the pinned versions in `requirements.txt` when loading the provided model bundle.
