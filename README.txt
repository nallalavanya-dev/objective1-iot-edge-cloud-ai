Objective 1 – AI-Powered IoT–Edge–Cloud Task Decision Lab

IMPORTANT: This deployment-ready version loads pre-trained regression models. It does not train all nine models every time Streamlit starts.

Deployment files:
- app.py
- requirements.txt
- runtime.txt
- objective1_iot_multinode_training_dataset.csv
- objective1_task_decision_dataset.csv
- train_models_offline.py
- trained_models/regression_models_bundle.joblib

Upload the contents of this folder to the root of your GitHub repository. In Streamlit Community Cloud set Main file path to app.py.

The application retains the task sidebar, task name, application scenarios, animated network, six execution locations, nine-algorithm comparison, completion-time predictions, deadline decision, and graphs.

The dataset is synthetic benchmark data, not measured hardware data. The animation is illustrative and does not execute real remote offloading.
