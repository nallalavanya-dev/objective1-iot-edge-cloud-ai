# Objective 1 – IoT Edge–Cloud AI Dashboard

## Project files
- `app.py` – Streamlit dashboard
- `requirements.txt` – Python dependencies
- `objective1_iot_multinode_training_dataset.csv` – multi-node task-completion regression dataset
- `objective1_task_decision_dataset.csv` – task-level decision dataset
- `train_objective1_multinode_models.py` – model training script

## Run locally
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## GitHub upload
Extract this ZIP. Open the `objective1-iot-edge-cloud-ai` folder and upload its files (not the ZIP) to the root of a GitHub repository.

## Streamlit Community Cloud
Choose the repository, branch `main`, and main file path `app.py`.

## Research note
The included dataset is synthetic benchmark data. Validate with independently measured NS-3 and hardware data before making experimental claims.
